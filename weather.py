"""Weather: something falling through the picture, whatever the background is.

The flakes are shapes3d.Snow, the snow of the Christmas background, with other
numbers: rain is many, fast, and blurred into streaks; leaves and petals are
few, coloured and fluttering; fireflies do not fall at all. They live in the
same space as the 3D shapes and come in the same two layers, but here the far
layer goes into the background and the near one over the finished frame, so
you sit in the middle of it.

Snow can also settle: `Settled` lets it gather along the top of your outline
while you sit still, and takes it away again where you move.

And the sky, for the backgrounds with nothing floating in them: `stars` and
`aurora` are what shapes3d adds to its wall for shapes:stars and shapes:aurora.
"""

import math

import torch
import torch.nn.functional as F

# What differs from snow, per kind. See shapes3d.Snow for what each number is.
KINDS = {
    "snow": {},
    "rain": dict(COUNT=1100, FALL=11.0, SWAY=0.0, COLOURS=[(0.72, 0.80, 0.95)],
                 # Tall cells: scaling them up is what draws a drop out into a
                 # streak. A blur that long at this size costs 10 ms a frame.
                 SHRINK=((2, 12), (4, 8)), SIGMA=((0.45, 1.0), (0.45, 0.6)), GAIN=0.45),
    # shortcut: leaves and petals are soft discs of colour, not leaf shapes;
    # give them sprites (the atlas in particles.py) if they should read as
    # leaves up close.
    "leaves": dict(COUNT=90, FALL=0.8, SWAY=0.9, FLICKER=(0.7, 5.0),
                   COLOURS=[(0.95, 0.45, 0.10), (0.85, 0.20, 0.08), (0.98, 0.72, 0.15),
                            (0.60, 0.30, 0.10)],
                   SIGMA=((1.5, 1.5), (1.0, 1.0))),
    "petals": dict(COUNT=220, FALL=0.45, SWAY=0.55, FLICKER=(0.6, 4.0),
                   COLOURS=[(1.0, 0.78, 0.86), (1.0, 0.88, 0.92), (0.98, 0.66, 0.78)],
                   SIGMA=((1.1, 1.1), (0.8, 0.8))),
    "fireflies": dict(COUNT=60, FALL=0.0, SWAY=0.5, BOB=0.35, FLICKER=(1.0, 2.2),
                      COLOURS=[(0.85, 1.0, 0.40), (1.0, 0.92, 0.45)], GAIN=1.3),
}


class Weather:
    """One kind of weather, as the main loop uses it: step, then a far layer
    for the background and a near one for the finished frame."""

    def __init__(self, kind, device, seed=7):
        import shapes3d
        self.kind = kind
        self.flakes = type(kind.title(), (shapes3d.Snow,), KINDS[kind])(device, seed=seed)
        self._tan = math.tan(shapes3d.FOV / 2)

    def step(self, dt, height, width):
        self.flakes.step(dt, self._tan * width / height, self._tan)

    def layer(self, frame, near):
        """`frame` (1, 3, H, W) with the near or the far flakes over it."""
        focal = (frame.shape[-2] / 2) / self._tan
        return self.flakes.layer(frame[0].float(), focal, near).to(frame.dtype)[None]


class Settled:
    """Snow lying on you: along the top of your outline, deeper the longer you
    sit still, gone where you move.

    All of it from the matte, at a quarter of the frame. A pixel can hold snow
    when it is you and what is above it is not; how much of the `DEPTH` rows
    above it are empty is how fast it fills, so the cap starts as a line along
    the very top and grows downwards. Where the matte changed since the last
    frame it melts, fast, and so does snow that no longer has anything under it.
    """

    SHRINK = 4
    DEPTH = 0.035           # how deep it can get, as a fraction of the frame's height
    LIFT = 0.006            # and how far above your outline it sits
    GROW, MELT = 0.15, 3.0  # a second: seen after ~4 s of sitting still, gone in a third of one
    WHITE = (0.97, 0.98, 1.0)

    def __init__(self):
        self.amount = self._last = self._lumps = None

    def clear(self):
        self.amount = self._last = None

    def update(self, alpha, dt, fall=1.0):
        """Let it snow, or melt, for `dt` seconds on the matte `alpha` (1, 1, H, W).
        `fall` is how much is coming down: twice the snow settles twice as fast."""
        h = max(alpha.shape[-2] // self.SHRINK, 2)
        w = max(alpha.shape[-1] // self.SHRINK, 2)
        a = F.adaptive_avg_pool2d(alpha.float(), (h, w))
        if self.amount is None or self.amount.shape != a.shape:
            self.amount, self._last = torch.zeros_like(a), a
            # Not the same everywhere, or it is an outline and not a drift.
            g = torch.Generator().manual_seed(3)
            self._lumps = F.interpolate(0.45 + 0.55 * torch.rand(1, 1, h // 6 + 2, w // 6 + 2,
                                                                generator=g),
                                        size=(h, w), mode="bicubic").clamp(0.3, 1).to(a.device)
        depth, lift = max(round(self.DEPTH * h), 2), max(round(self.LIFT * h), 1)
        raised = F.pad(a[:, :, lift:], (0, 0, 0, lift))
        # For each pixel, how much of the `depth` rows above it are you.
        above = F.avg_pool2d(F.pad(raised, (0, 0, depth, 0)), (depth, 1), stride=1)[:, :, :-1]
        # The drop from above, not a product: a soft edge down your side is half
        # you all the way, and half of a half would slowly snow it in.
        ledge = (raised - above).clamp(min=0)
        moved = F.max_pool2d((a - self._last).abs(), 5, stride=1, padding=2)
        moved = ((moved - 0.15) / 0.3).clamp(0, 1)
        self._last = a
        melt = torch.maximum(moved, (ledge < 0.02).float())
        self.amount = (self.amount
                       + dt * (self.GROW * fall * self._lumps * ledge * (1 - moved)
                               - self.MELT * melt)).clamp(0, 1)

    def draw(self, frame):
        """`frame` (1, 3, H, W) with the snow that has settled over it."""
        if self.amount is None:
            return frame
        cover = ((self.amount - 0.5) / 0.25).clamp(0, 1)
        cover = cover * cover * (3 - 2 * cover)
        cover = F.interpolate(cover, size=frame.shape[-2:], mode="bilinear",
                              align_corners=False).to(frame.dtype)
        white = torch.tensor(self.WHITE, dtype=frame.dtype, device=frame.device).view(1, 3, 1, 1)
        return frame + (white - frame) * cover


# ---------------------------------------------------------------- the sky

_STARS = {}


def stars(height, width, t, device, seed=11):
    """(3, H, W) of stars to add to a dark wall: many faint and a few bright,
    from blue-white to warm, each twinkling at its own pace.

    The stars themselves are drawn once per size. Twinkling is a coarse grid
    of phases and rates laid over them, so a star keeps one rhythm and its
    neighbours another, and a frame costs a sine over a small image."""
    key = (height, width, str(device))
    if key not in _STARS:
        g = torch.Generator().manual_seed(seed)
        r = lambda *shape: torch.rand(*shape, generator=g)
        n = max(height * width // 900, 8)
        x, y = r(n) * (width - 1.001), r(n) * (height - 1.001)
        warm = r(n)[None]
        blue = torch.tensor([0.75, 0.85, 1.0]).view(3, 1)
        amber = torch.tensor([1.0, 0.9, 0.75]).view(3, 1)
        values = (blue + (amber - blue) * warm) * (0.25 + 2.5 * r(n) ** 5)
        x0, y0 = x.floor(), y.floor()
        fx, fy = x - x0, y - y0
        img = torch.zeros(3, height * width)
        for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                           (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            img.index_add_(1, ((y0 + dy) * width + x0 + dx).long(), values * wt)
        img = img.view(1, 3, height, width)
        # A little light round each: only the bright ones have enough to show.
        img = img + 2.0 * F.avg_pool2d(F.avg_pool2d(img, 5, 1, 2), 3, 1, 1)
        grid = (1, 1, max(height // 8, 1), max(width // 8, 1))
        _STARS[key] = (img[0].to(device), (r(*grid) * 2 * math.pi).to(device),
                       (1.5 + 4.0 * r(*grid)).to(device))
    img, phase, rate = _STARS[key]
    twinkle = 1 - 0.45 * (0.5 + 0.5 * torch.sin(rate * t + phase))
    return img * F.interpolate(twinkle, size=(height, width), mode="nearest")[0]


METEOR = 0.7            # seconds a shooting star lasts


def meteor(height, width, age, x, y, dx, dy, device):
    """(3, H, W) to add: a shooting star `age` seconds old, or None once it is out.

    It starts at (x, y), fractions of the frame, and goes along the unit
    vector (dx, dy) for a third of the frame's height: a head a pixel wide
    with a tail a tenth of the height long fading out behind it, the whole
    of it fading in and out. Far off and faint, not a firework."""
    if age >= METEOR:
        return None
    k = age / METEOR
    px = torch.arange(width, device=device).view(1, -1) - (x * width + dx * 0.33 * height * k)
    py = torch.arange(height, device=device).view(-1, 1) - (y * height + dy * 0.33 * height * k)
    behind = -(px * dx + py * dy)                      # along the tail, from the head
    across = px * dy - py * dx
    thick = max(height / 900, 0.7)
    glow = torch.exp(-(across / thick) ** 2) * (1 - behind / (0.1 * height)).clamp(0, 1) \
        * (1 + behind / thick).clamp(0, 1)
    colour = torch.tensor([0.85, 0.92, 1.0], device=device).view(3, 1, 1)
    return colour * glow * (0.6 * math.sin(math.pi * k))


def aurora(x, y, t):
    """(3, h, w) of northern lights to add to a dark wall, for `x` (1, w) and
    `y` (h, 1) in 0..1.

    Three curtains. Each has a wavering lower edge, bright and green, fades
    upwards into violet, and is combed into rays that drift along it."""
    low = torch.tensor([0.10, 0.95, 0.45], device=x.device).view(3, 1, 1)
    high = torch.tensor([0.45, 0.15, 0.75], device=x.device).view(3, 1, 1)
    out = 0
    for i, (foot, speed) in enumerate(((0.50, 0.11), (0.36, 0.07), (0.60, 0.05))):
        edge = foot + 0.08 * torch.sin(4.0 * x + 3 * speed * t + 2.1 * i) \
            + 0.035 * torch.sin(11.0 * x - 5 * speed * t + i) \
            + 0.012 * torch.sin(29.0 * x + 7 * speed * t + 3 * i)
        up = edge - y                                  # how far above the curtain's foot
        rise = (up / 0.05).clamp(0, 1)
        body = torch.exp(-up.clamp(min=0) / 0.2) * rise * rise * (3 - 2 * rise)
        # Two combs that do not line up, so no two rays are the same width.
        rays = 0.35 + 0.65 * (0.5 + 0.5 * torch.sin(
            43 * x + 6 * torch.sin(2.3 * x + 0.2 * t + i) + 0.5 * t)) \
            * (0.55 + 0.45 * torch.sin(17 * x - 0.3 * t + 2 * i))
        breath = 0.6 + 0.4 * math.sin(0.13 * t + 1.7 * i)
        out = out + (low + (high - low) * (up / 0.3).clamp(0, 1)) * (body * rays * breath * 0.5)
    return out
