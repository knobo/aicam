"""Drawing in the air: ink that stays at the depth it was drawn at.

Every other camera filter that lets you scribble on the picture puts the ink on
the glass. Here the pipeline already knows how far away everything is, so a
stroke can be given the depth of the fingertip that drew it and then be occluded
by whatever is nearer - including you. Draw a circle, step in front of it, and
your shoulder passes over it the way it would over anything else in the room.

That is the same rule the particles follow, so this reuses their idea exactly:
points carry a z, a point whose z is smaller than the scene depth at its pixel
is behind, and the two layers go to `particles.composite`.

Coordinates arrive already mirrored. The frame is flipped before the hand
tracker ever sees it, so pointer space and output space are the same space and
nothing here has to know whether the mirror is on.
"""

import time

import numpy as np
import torch

# What a pinch does. One pinch, one meaning at a time - the tool decides which.
TOOLS = ("pen", "laser", "line", "rect", "eraser")
DEFAULT_TOOL = "pen"

# A rubber you cannot see the edge of is a rubber you cannot aim. It is wider
# than the pen so it is worth switching to, and never so narrow that a hand that
# wobbles a pixel leaves islands of ink behind.
ERASER_SCALE, ERASER_MIN = 2.5, 14.0

# Laser: full strength while you are still pointing at the thing, then gone.
# Long enough to finish the sentence, short enough that nobody asks you to
# erase it.
LASER_HOLD, LASER_FADE = 0.7, 0.9

# How many steps back Undo goes. A snapshot is only as big as the ink that
# existed when it was taken, so a dozen of them costs nothing worth counting.
UNDO_DEPTH = 12

# Ink is laid a hair in front of the fingertip that drew it. Exactly at it, the
# hand covers its own stroke as it is drawn and the line flickers under the
# fingers; a little nearer and it comes out clean, while still going behind you
# when you later step in front of the place it hangs.
Z_BIAS = 0.02

PALETTE = {
    "white":  (1.00, 1.00, 1.00),
    "yellow": (1.00, 0.84, 0.25),
    "pink":   (1.00, 0.38, 0.62),
    "cyan":   (0.35, 0.85, 1.00),
    "green":  (0.45, 0.95, 0.55),
    "orange": (1.00, 0.55, 0.20),
}
DEFAULT_INK = "yellow"


def disc(radius, device, dtype=torch.float32):
    """A soft round brush tip, antialiased over the outermost pixel."""
    size = int(2 * round(radius) + 3)
    grid = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    r = torch.hypot(grid.view(-1, 1), grid.view(1, -1))
    return (radius - r + 0.5).clamp(0, 1)


class Canvas:
    """Points of ink, each with a depth, rasterised into a front and back layer.

    Points rather than line segments: a stroke is filled in densely enough that
    consecutive brush stamps overlap, which makes the renderer a splat loop
    identical to the particles' and avoids a line rasteriser that would have to
    be depth-aware per pixel.
    """

    def __init__(self, device, capacity=24000, clock=time.monotonic):
        self.device = device
        self.capacity = capacity
        self.pos = torch.zeros(capacity, 3, device=device)     # x px, y px, z
        self.color = torch.zeros(capacity, 3, device=device)
        self.radius = torch.zeros(capacity, device=device)
        self.stroke = torch.zeros(capacity, dtype=torch.long, device=device)
        self.born = torch.zeros(capacity, device=device)
        self.fades = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.opacity = torch.ones(capacity, device=device)
        self.count = 0
        self.strokes = 0
        self.tool = DEFAULT_TOOL
        self._clock = clock
        self._open = False
        self._last = None          # (x, y) in pixels, where the live stroke got to
        self._smooth = None        # low-passed pointer, in normalised coordinates
        self._anchor = None        # where a line or a rectangle was started
        self._start = 0            # index the live stroke begins at, for rebuilds
        self._undo = []            # snapshots, one per stroke, newest last

    # ------------------------------------------------------------------ input

    def begin(self, tool=DEFAULT_TOOL):
        """Start a stroke with a tool. Idempotent: a held pinch calls this once.

        The snapshot is what makes Undo uniform. Stepping back used to mean
        dropping the last stroke, which cannot express a rubbing-out: the ink the
        eraser removed is gone. Remembering the canvas as it was before each
        action costs a copy of however much ink existed at the time, and makes
        one Undo mean the same thing whatever the tool was.
        """
        if self._open:
            return
        self._undo.append(self._capture())
        del self._undo[:-UNDO_DEPTH]
        self.strokes += 1
        self.tool = tool if tool in TOOLS else DEFAULT_TOOL
        self._open = True
        self._last = None
        self._smooth = None
        self._anchor = None
        self._start = self.count

    def end(self):
        self._open = False
        self._last = None
        self._smooth = None
        self._anchor = None
        self._start = self.count

    def extend(self, x, y, scene_depth, colour=DEFAULT_INK, width=6.0,
               smoothing=0.45):
        """Take the pointer where it is now and let the current tool act on it.

        `x`, `y` are normalised; `scene_depth` is the depth map, sampled here on
        the GPU for every new point at once so no frame ever waits for a
        readback.
        """
        if not self._open:
            return 0
        _, _, height, width_px = scene_depth.shape
        radius = max(float(width) / 2.0, 0.5)
        px, py = self._follow(x, y, width_px, height, smoothing)

        if self.tool == "eraser":
            return self._rub_out(px, py, max(radius * ERASER_SCALE, ERASER_MIN))
        if self.tool in ("line", "rect"):
            return self._shape_to(px, py, scene_depth, colour, radius)
        return self._draw_to(px, py, scene_depth, colour, radius)

    def _follow(self, x, y, width_px, height, smoothing):
        """Low-passed pointer, in pixels.

        The fingertip jitters by a pixel or two even when a hand is holding
        still, and a brush follows every bit of it. Same low pass as the held
        window - and because this runs every frame against a tracker that
        reports at 10 Hz, the smoothing doubles as the interpolation between
        tracker samples.
        """
        if self._smooth is None:
            self._smooth = (x, y)
        else:
            sx, sy = self._smooth
            self._smooth = (sx + (x - sx) * (1 - smoothing),
                            sy + (y - sy) * (1 - smoothing))
        return self._smooth[0] * width_px, self._smooth[1] * height

    def _draw_to(self, px, py, scene_depth, colour, radius):
        """Freehand: fill in the gap since the last position and stamp it."""
        if self._last is None:
            xs, ys = np.array([px]), np.array([py])
        else:
            lx, ly = self._last
            span = float(np.hypot(px - lx, py - ly))
            if span < radius * 0.5:
                return 0          # hand has not moved far enough to be worth a stamp
            xs, ys = _between(lx, ly, px, py, radius)
        self._last = (px, py)
        return self._append(xs, ys, scene_depth, colour, radius)

    def _shape_to(self, px, py, scene_depth, colour, radius):
        """A line or a box from where the pinch closed to where it is now.

        Redrawn from the anchor every frame, so you see the shape you are about
        to get and can pull it into place before letting go. The rebuild is free
        because a live stroke is always the tail of the buffer: dropping it is
        moving the write head back, not compacting anything.
        """
        if self._anchor is None:
            self._anchor = (px, py)
            return self._append(np.array([px]), np.array([py]),
                                scene_depth, colour, radius)

        ax, ay = self._anchor
        self.count = self._start
        if self.tool == "line":
            xs, ys = _between(ax, ay, px, py, radius, include_first=True)
        else:
            xs, ys = _box(ax, ay, px, py, radius)
        return self._append(xs, ys, scene_depth, colour, radius)

    def _rub_out(self, px, py, radius):
        """Delete the ink under the fingertip. Returns how much went.

        Depth is deliberately ignored. An eraser that only took ink at the
        distance your hand happens to be at would leave ghosts of the same
        stroke hanging at slightly different depths, and no amount of waving
        would clear them.
        """
        if not self.count:
            return 0
        dx = self.pos[:self.count, 0] - px
        dy = self.pos[:self.count, 1] - py
        keep = ((dx * dx + dy * dy) > radius * radius).nonzero(as_tuple=True)[0]
        gone = self.count - keep.numel()
        if gone:
            self._keep(keep)
        return -gone

    def _append(self, xs, ys, scene_depth, colour, radius):
        n = len(xs)
        if n == 0:
            return 0
        if self.count + n > self.capacity:
            self._forget_oldest_stroke(n)
            n = min(n, self.capacity - self.count)
            xs, ys = xs[-n:], ys[-n:]
            if n == 0:
                return 0

        _, _, height, width_px = scene_depth.shape
        x = torch.as_tensor(xs, device=self.device, dtype=torch.float32)
        y = torch.as_tensor(ys, device=self.device, dtype=torch.float32)
        ix = x.round().long().clamp(0, width_px - 1)
        iy = y.round().long().clamp(0, height - 1)
        z = (scene_depth[0, 0, iy, ix].float() + Z_BIAS).clamp(0, 1)

        end = self.count + n
        self.pos[self.count:end, 0] = x
        self.pos[self.count:end, 1] = y
        self.pos[self.count:end, 2] = z
        self.color[self.count:end] = torch.tensor(
            PALETTE.get(colour, PALETTE[DEFAULT_INK]), device=self.device)
        self.radius[self.count:end] = radius
        self.stroke[self.count:end] = self.strokes
        self.born[self.count:end] = self._clock()
        self.fades[self.count:end] = self.tool == "laser"
        self.opacity[self.count:end] = 1.0
        self.count = end
        return n

    # ----------------------------------------------------------------- edits

    def _forget_oldest_stroke(self, needed):
        """Full canvas: drop whole strokes from the front, never half of one.

        Losing the tail of the oldest scribble looks like a mistake; losing the
        whole scribble looks like the wall being wiped, which is what it is.
        """
        while self.count and self.capacity - self.count < needed:
            first = int(self.stroke[0])
            keep = (self.stroke[:self.count] != first).nonzero(as_tuple=True)[0]
            if keep.numel() == 0:
                self.count = 0
                break
            self._keep(keep)

    def _keep(self, index):
        n = index.numel()
        self.pos[:n] = self.pos[index]
        self.color[:n] = self.color[index]
        self.radius[:n] = self.radius[index]
        self.stroke[:n] = self.stroke[index]
        self.born[:n] = self.born[index]
        self.fades[:n] = self.fades[index]
        self.opacity[:n] = self.opacity[index]
        # Kept points keep their order, so wherever the live stroke started, it
        # starts after however many of its predecessors survived. Without this a
        # rebuilt line would either leave a ghost of itself or eat the stroke
        # before it the first time the canvas filled up mid-shape.
        self._start = int((index < self._start).sum())
        self.count = int(n)

    def _capture(self):
        """The canvas as it is now, small enough to keep a dozen of."""
        n = self.count
        return (self.pos[:n].clone(), self.color[:n].clone(),
                self.radius[:n].clone(), self.stroke[:n].clone(),
                self.born[:n].clone(), self.fades[:n].clone(),
                self.opacity[:n].clone(), self.strokes)

    def _restore(self, snap):
        pos, color, radius, stroke, born, fades, opacity, strokes = snap
        n = pos.shape[0]
        self.pos[:n], self.color[:n] = pos, color
        self.radius[:n], self.stroke[:n] = radius, stroke
        self.born[:n], self.fades[:n], self.opacity[:n] = born, fades, opacity
        self.count, self.strokes = n, strokes
        self.end()

    def undo(self):
        """One step back, whatever the last thing was - ink, a rub-out or a wipe."""
        if not self._undo:
            return False
        self._restore(self._undo.pop())
        return True

    def clear(self):
        """Wipe it. Undoable, because a wipe is the easiest thing to do by mistake."""
        if self.count:
            self._undo.append(self._capture())
            del self._undo[:-UNDO_DEPTH]
        self.count = 0
        self.end()

    def update(self, now=None):
        """Age the laser and drop what has finished fading.

        Called once a frame. Everything else keeps opacity 1 and is untouched.
        """
        if not self.count or not bool(self.fades[:self.count].any()):
            return
        age = (now if now is not None else self._clock()) - self.born[:self.count]
        fading = self.fades[:self.count]
        left = ((LASER_HOLD + LASER_FADE - age) / LASER_FADE).clamp(0, 1)
        self.opacity[:self.count] = torch.where(fading, left, torch.ones_like(left))
        if bool((self.opacity[:self.count] <= 0).any()):
            keep = (self.opacity[:self.count] > 0).nonzero(as_tuple=True)[0]
            self._keep(keep)

    @property
    def empty(self):
        return self.count == 0

    # ---------------------------------------------------------------- render

    def render(self, height, width, scene_depth=None):
        """Rasterise into (front, behind), each a premultiplied (colour, alpha).

        The depth test is per pixel, not per point. Testing a whole brush stamp
        on the depth under its centre is what the particles do, and for a 6 px
        confetto it does not show; a stroke runs along the edge of a shoulder for
        its whole length, and stamps whose centres fall just outside the
        silhouette would paint several pixels of ink across it. So every point
        lays its own depth into a z-buffer as it is stamped, and the split is
        decided pixel by pixel afterwards.
        """
        colour, alpha, zbuf = self._stamp_all(height, width)
        empty = (torch.zeros(1, 3, height, width, device=self.device),
                 torch.zeros(1, 1, height, width, device=self.device))
        if not self.count:
            return empty, empty
        if scene_depth is None:
            return (colour, alpha), empty

        near = (zbuf >= scene_depth[0, 0]).view(1, 1, height, width)
        return ((colour * near, alpha * near),
                (colour * ~near, alpha * ~near))

    def _stamp_all(self, height, width):
        colour = torch.zeros(1, 3, height, width, device=self.device)
        alpha = torch.zeros(1, 1, height, width, device=self.device)
        zbuf = torch.zeros(height, width, device=self.device)
        if not self.count:
            return colour, alpha, zbuf

        radii = self.radius[:self.count]
        # One brush per distinct width, which in practice means one or two.
        for value in radii.unique():
            sel = (radii == value).nonzero(as_tuple=True)[0]
            self._stamp(colour, alpha, zbuf, sel, float(value), height, width)

        # Overlapping stamps must not add up: ink is opaque, not additive, or a
        # slow hand burns a bright blob where a fast one draws a thin line.
        return colour.clamp(0, 1), alpha.clamp(0, 1), zbuf

    def _stamp(self, colour, alpha, zbuf, sel, radius, height, width):
        brush = disc(radius, self.device)
        size = brush.shape[0]
        half = size // 2
        m = sel.numel()

        p = self.pos[sel]
        c = self.color[sel]
        offs = torch.arange(size, device=self.device) - half
        ys = p[:, 1].round().long().view(m, 1) + offs.view(1, -1)
        xs = p[:, 0].round().long().view(m, 1) + offs.view(1, -1)

        keep = ((ys >= 0) & (ys < height))[:, :, None] & \
               ((xs >= 0) & (xs < width))[:, None, :]
        flat = (ys.clamp(0, height - 1)[:, :, None] * width
                + xs.clamp(0, width - 1)[:, None, :]).reshape(-1)
        # Opacity is per point, so a laser stroke can fade out where everything
        # else stays solid: the brush shape decides the edge, this decides how
        # much of the ink is still there.
        weight = brush.expand(m, size, size) * keep * self.opacity[sel].view(m, 1, 1)

        # scatter_reduce with amax, not scatter_add: the stamps within a stroke
        # overlap by design, and adding them would make every stroke a solid bar
        # with bright knots wherever the hand slowed down.
        alpha.view(-1).scatter_reduce_(0, flat, weight.reshape(-1), reduce="amax")
        for ch in range(3):
            premult = (weight * c[:, ch].view(m, 1, 1)).reshape(-1)
            colour[0, ch].view(-1).scatter_reduce_(0, flat, premult, reduce="amax")

        # Nearest ink wins the pixel, so two strokes crossing at different
        # depths are split at the crossing rather than as whole strokes.
        z = (p[:, 2].view(m, 1, 1) * (weight > 0)).reshape(-1)
        zbuf.view(-1).scatter_reduce_(0, flat, z, reduce="amax")


def _between(x0, y0, x1, y1, radius, include_first=False):
    """Points along a segment, close enough together that the stamps overlap."""
    span = float(np.hypot(x1 - x0, y1 - y0))
    steps = int(span / max(radius * 0.5, 1.0)) + 1
    t = np.linspace(0.0, 1.0, steps + 1)
    if not include_first:
        t = t[1:]
    return x0 + (x1 - x0) * t, y0 + (y1 - y0) * t


def _box(x0, y0, x1, y1, radius):
    """The four sides of a rectangle, as points."""
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    xs, ys = [], []
    for (ax, ay), (bx, by) in zip(corners, corners[1:]):
        sx, sy = _between(ax, ay, bx, by, radius, include_first=True)
        xs.append(sx)
        ys.append(sy)
    return np.concatenate(xs), np.concatenate(ys)


def over(top, bottom):
    """Premultiplied `top` over premultiplied `bottom`, both (colour, alpha)."""
    tc, ta = top
    bc, ba = bottom
    return tc + bc * (1.0 - ta), (ta + ba * (1.0 - ta)).clamp(0, 1)
