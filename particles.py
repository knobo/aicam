"""GPU particle system with depth-aware compositing.

The point of difference: every particle carries a z coordinate that is compared
against the scene depth, so confetti can pass *behind* your shoulder and be
occluded by it, while other particles fall in front. macOS and Windows composite
their reactions as a flat layer over the video; they have no depth to work with.
"""

import math

import torch
import torch.nn.functional as F

SPRITE_SIZE = 64        # resolution each shape is baked at
PATCH_SIZE = 48         # size of the square stamped into the frame per particle


# ---------------------------------------------------------------- shapes

def _antialias(mask):
    """Rasterise hard, then soften by one pixel.

    Scaling an implicit curve by a constant to get a soft edge fails where the
    shape is thin: a heart's lower point never reaches full opacity and the
    shape comes out bloated. Filling the exact interior and blurring afterwards
    keeps the silhouette honest.
    """
    k = torch.tensor([0.25, 0.5, 0.25], device=mask.device)
    m = mask[None, None]
    m = F.conv2d(F.pad(m, (1, 1, 0, 0), mode="replicate"), k.view(1, 1, 1, 3))
    m = F.conv2d(F.pad(m, (0, 0, 1, 1), mode="replicate"), k.view(1, 1, 3, 1))
    return m[0, 0].clamp(0, 1)


def _heart(n, device):
    y, x = torch.meshgrid(
        torch.linspace(1.35, -1.15, n, device=device),
        torch.linspace(-1.25, 1.25, n, device=device),
        indexing="ij",
    )
    f = (x ** 2 + y ** 2 - 1) ** 3 - x ** 2 * y ** 3
    return _antialias((f < 0).to(torch.float32))


def _disc(n, device):
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, n, device=device),
        torch.linspace(-1, 1, n, device=device),
        indexing="ij",
    )
    return ((1.0 - torch.sqrt(x ** 2 + y ** 2)) * n / 6).clamp(0, 1)


def _rect(n, device):
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, n, device=device),
        torch.linspace(-1, 1, n, device=device),
        indexing="ij",
    )
    inside = (x.abs() < 0.75) & (y.abs() < 0.45)
    return inside.to(torch.float32)


def _star(n, device, points=5, inner=0.44):
    """Proper polygon star: radius alternates between the tips and the waist."""
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, n, device=device),
        torch.linspace(-1, 1, n, device=device),
        indexing="ij",
    )
    r = torch.sqrt(x ** 2 + y ** 2).clamp(min=1e-6)
    a = torch.atan2(y, x) + math.pi / 2
    wedge = (a % (2 * math.pi / points)) / (2 * math.pi / points)   # 0..1 per point
    edge = 1.0 - (1.0 - inner) * (1.0 - (2 * wedge - 1).abs())
    return _antialias((r < edge * 0.98).to(torch.float32))


def _balloon(n, device):
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, n, device=device),
        torch.linspace(-1, 1, n, device=device),
        indexing="ij",
    )
    body = ((1.0 - torch.sqrt((x / 0.72) ** 2 + ((y + 0.12) / 0.92) ** 2)) * n / 6).clamp(0, 1)
    knot = ((0.10 - torch.sqrt(x ** 2 + (y - 0.86) ** 2)) * n / 3).clamp(0, 1)
    return torch.maximum(body, knot)


SHAPES = {"disc": _disc, "heart": _heart, "rect": _rect, "star": _star, "balloon": _balloon}
SHAPE_IDS = {name: i for i, name in enumerate(SHAPES)}


# ---------------------------------------------------------------- system

class ParticleSystem:
    """Fixed-capacity pool. Dead slots are reused rather than reallocated."""

    def __init__(self, device, capacity=3000):
        self.device = device
        self.capacity = capacity
        z = lambda *s: torch.zeros(*s, device=device)

        self.pos = z(capacity, 3)        # x, y in pixels; z in depth units (1 = near)
        self.vel = z(capacity, 3)
        self.age = z(capacity)
        self.life = torch.ones(capacity, device=device)
        self.size = z(capacity)
        self.spin = z(capacity)
        self.angle = z(capacity)
        self.color = z(capacity, 3)
        self.shape = torch.zeros(capacity, dtype=torch.long, device=device)
        self.gravity = z(capacity)
        self.drag = z(capacity)
        self.fade = torch.ones(capacity, device=device)
        self.alive = torch.zeros(capacity, dtype=torch.bool, device=device)

        atlas = [SHAPES[name](SPRITE_SIZE, device) for name in SHAPES]
        self.atlas = torch.stack(atlas)[:, None]   # [S,1,64,64]

    @property
    def count(self):
        return int(self.alive.sum())

    def emit(self, n, **fields):
        """Activate up to n dead slots. Each field is a tensor of length n."""
        free = (~self.alive).nonzero(as_tuple=True)[0]
        if free.numel() == 0:
            return 0
        n = min(n, free.numel())
        idx = free[:n]
        for name, value in fields.items():
            getattr(self, name)[idx] = value[:n].to(getattr(self, name).dtype)
        self.age[idx] = 0.0
        self.alive[idx] = True
        return n

    def update(self, dt):
        a = self.alive
        if not bool(a.any()):
            return
        self.vel[a, 1] += self.gravity[a] * dt
        self.vel[a] *= (1.0 - self.drag[a] * dt).clamp(min=0).unsqueeze(1)
        self.pos[a] += self.vel[a] * dt
        self.angle[a] += self.spin[a] * dt
        self.age[a] += dt
        self.alive &= self.age < self.life

    def _opacity(self):
        """Fade in over the first 10% of life, out over the last 35%."""
        t = (self.age / self.life.clamp(min=1e-6)).clamp(0, 1)
        rise = (t / 0.10).clamp(0, 1)
        fall = ((1.0 - t) / 0.35).clamp(0, 1)
        return rise * fall * self.fade

    def render(self, height, width, scene_depth=None):
        """Rasterise into a front layer and a behind layer.

        `scene_depth` is the depth map (1 = near). A particle whose own z is
        smaller than the scene depth at its pixel sits further away than what is
        there, so it belongs behind and will be occluded by the subject.
        """
        empty = (torch.zeros(1, 3, height, width, device=self.device),
                 torch.zeros(1, 1, height, width, device=self.device))
        idx = self.alive.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return empty, empty

        pos = self.pos[idx]
        opacity = self._opacity()[idx]

        # Rotation and scale, baked into an affine sampling grid so each particle
        # can have an arbitrary angle without a per-shape texture per angle.
        n = idx.numel()
        scale = (PATCH_SIZE / self.size[idx].clamp(min=1.0))
        cos = torch.cos(self.angle[idx]) * scale
        sin = torch.sin(self.angle[idx]) * scale
        theta = torch.zeros(n, 2, 3, device=self.device)
        theta[:, 0, 0], theta[:, 0, 1] = cos, -sin
        theta[:, 1, 0], theta[:, 1, 1] = sin, cos

        grid = F.affine_grid(theta, (n, 1, PATCH_SIZE, PATCH_SIZE), align_corners=False)
        sprite = F.grid_sample(self.atlas[self.shape[idx]], grid,
                               align_corners=False, padding_mode="zeros")  # [n,1,P,P]
        sprite = sprite * opacity.view(n, 1, 1, 1)

        if scene_depth is not None:
            px = pos[:, 0].round().long().clamp(0, width - 1)
            py = pos[:, 1].round().long().clamp(0, height - 1)
            here = scene_depth[0, 0, py, px]
            behind = pos[:, 2] < here
        else:
            behind = torch.zeros(n, dtype=torch.bool, device=self.device)

        front_layer = self._splat(~behind, idx, pos, sprite, height, width)
        behind_layer = self._splat(behind, idx, pos, sprite, height, width)
        return front_layer, behind_layer

    def _splat(self, mask, idx, pos, sprite, height, width):
        colour = torch.zeros(1, 3, height, width, device=self.device)
        alpha = torch.zeros(1, 1, height, width, device=self.device)
        if not bool(mask.any()):
            return colour, alpha

        sel = mask.nonzero(as_tuple=True)[0]
        p = pos[sel]
        s = sprite[sel, 0]                       # [m,P,P]
        c = self.color[idx[sel]]                 # [m,3]
        m = sel.numel()

        half = PATCH_SIZE // 2
        offs = torch.arange(PATCH_SIZE, device=self.device) - half
        ys = (p[:, 1].round().long().view(m, 1) + offs.view(1, -1))    # [m,P]
        xs = (p[:, 0].round().long().view(m, 1) + offs.view(1, -1))

        valid_y = (ys >= 0) & (ys < height)
        valid_x = (xs >= 0) & (xs < width)
        keep = valid_y[:, :, None] & valid_x[:, None, :]               # [m,P,P]

        flat = (ys.clamp(0, height - 1)[:, :, None] * width
                + xs.clamp(0, width - 1)[:, None, :]).reshape(-1)
        weight = (s * keep).reshape(-1)

        alpha.view(-1).scatter_add_(0, flat, weight)
        for ch in range(3):
            premult = (s * keep * c[:, ch].view(m, 1, 1)).reshape(-1)
            colour[0, ch].view(-1).scatter_add_(0, flat, premult)

        alpha = alpha.clamp(0, 1)
        colour = colour.clamp(0, 3)
        return colour, alpha


def composite(base, subject, subject_alpha, front, behind):
    """background -> particles behind -> subject -> particles in front."""
    c_behind, a_behind = behind
    c_front, a_front = front

    out = base * (1.0 - a_behind) + c_behind
    out = subject * subject_alpha + out * (1.0 - subject_alpha)
    out = out * (1.0 - a_front) + c_front
    return out.clamp(0, 1)
