"""Image effects for aicam. Everything runs on the GPU and expects [1,3,H,W] in 0..1."""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- kernels

def gaussian_kernel1d(sigma, device, dtype):
    radius = max(1, int(3 * sigma))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def separable_blur(img, sigma):
    """Gaussian blur as two 1D convolutions."""
    k = gaussian_kernel1d(sigma, img.device, img.dtype)
    n = k.numel()
    pad = n // 2
    c = img.shape[1]
    kh = k.view(1, 1, 1, n).expand(c, 1, 1, n)
    kv = k.view(1, 1, n, 1).expand(c, 1, n, 1)
    img = F.conv2d(F.pad(img, (pad, pad, 0, 0), mode="reflect"), kh, groups=c)
    img = F.conv2d(F.pad(img, (0, 0, pad, pad), mode="reflect"), kv, groups=c)
    return img


def bokeh(img, strength):
    """Cheap heavy defocus: downscale, blur, upscale.

    Blurring at quarter scale looks the same as a large kernel at a fraction of
    the cost, and the interpolation softness on the way back up is free.
    """
    h, w = img.shape[-2:]
    small = F.interpolate(img, scale_factor=0.25, mode="bilinear", align_corners=False)
    small = separable_blur(small, max(1.0, strength / 4.0))
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)


def disc_kernel(radius, device, dtype):
    """Flat circular kernel - what gives bokeh a defined edge instead of mush.

    A Gaussian spreads a point of light into a soft blob. A lens spreads it into
    a hard-edged disc shaped by the aperture. That disc is what reads as "camera"
    rather than "filter".
    """
    r = max(1, int(round(radius)))
    n = 2 * r + 1
    y, x = torch.meshgrid(
        torch.arange(n, device=device, dtype=dtype) - r,
        torch.arange(n, device=device, dtype=dtype) - r,
        indexing="ij",
    )
    d = torch.sqrt(x ** 2 + y ** 2)
    k = (r + 0.5 - d).clamp(0, 1)     # soft rim, otherwise the disc staircases
    return k / k.sum()


def disc_blur(img, radius):
    k = disc_kernel(radius, img.device, img.dtype)
    n = k.shape[-1]
    c = img.shape[1]
    k = k.view(1, 1, n, n).expand(c, 1, n, n)
    return F.conv2d(F.pad(img, (n // 2,) * 4, mode="reflect"), k, groups=c)


# ---------------------------------------------------------------- depth of field

def highlight_boost(img, threshold, gain):
    """Lift the brightest points before blurring so they bloom into discs.

    Without this a highlight is simply averaged away. A lens preserves the
    energy: a small very bright spot becomes a large, still visible disc.
    """
    weights = torch.tensor([0.2126, 0.7152, 0.0722], device=img.device,
                           dtype=img.dtype).view(1, 3, 1, 1)
    luma = (img * weights).sum(1, keepdim=True)
    excess = (luma - threshold).clamp(min=0) / max(1e-3, 1.0 - threshold)
    return img * (1.0 + gain * excess ** 2)


def layered_bokeh(img, coc, max_radius, levels=4):
    """Per-pixel varying defocus driven by `coc` (0 = sharp, 1 = maximum).

    True varying blur needs per-pixel scatter. We approximate with a few
    pre-blurred layers and a triangular weighting between them - a standard
    real-time graphics trick, and indistinguishable in motion.
    """
    h, w = img.shape[-2:]
    small = F.interpolate(img, scale_factor=0.25, mode="bilinear", align_corners=False)

    stack = [small]
    for i in range(1, levels):
        radius = max_radius * (i / (levels - 1)) / 4.0    # /4 because we are at quarter scale
        stack.append(disc_blur(small, max(1.0, radius)))

    coc_small = F.interpolate(coc, size=small.shape[-2:], mode="bilinear", align_corners=False)
    pos = coc_small.clamp(0, 1) * (levels - 1)

    out = torch.zeros_like(small)
    for i, layer in enumerate(stack):
        out = out + layer * (1.0 - (pos - i).abs()).clamp(min=0)

    return F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)


def circle_of_confusion(depth, alpha, strength, focus_softness=0.15):
    """How defocused each pixel should be, from how far behind the subject it is.

    `depth` is MiDaS inverse depth: 1 = near, 0 = far. The focus plane is taken
    from the subject's own depth, read where the matte is confident, so it
    follows you if you lean forward or back.

    Note the clamp: nothing in *front* of the focus plane is blurred at all,
    which is why a book you hold up stays sharp.
    """
    solid = (alpha > 0.9).to(depth.dtype)
    if solid.sum() > 100:
        focus = (depth * solid).sum() / solid.sum()
    else:
        focus = depth.median()

    behind = (focus - depth).clamp(min=0)
    return ((behind / max(1e-3, focus_softness)).clamp(0, 1) * strength).clamp(0, 1)


# ---------------------------------------------------------------- studio light

def depth_normals(depth, strength=40.0):
    """Approximate surface normals from the depth map.

    Not geometrically correct - MiDaS gives relative, not metric depth - but
    good enough to tell which way a surface leans, which is all the lighting
    needs.
    """
    d = separable_blur(depth, 3.0)
    dx = F.pad(d[..., :, 1:] - d[..., :, :-1], (0, 1, 0, 0), mode="replicate")
    dy = F.pad(d[..., 1:, :] - d[..., :-1, :], (0, 0, 0, 1), mode="replicate")
    n = torch.cat([-dx * strength, -dy * strength, torch.ones_like(d)], dim=1)
    return n / n.norm(dim=1, keepdim=True).clamp(min=1e-6)


def studio_light(img, alpha, depth, key=0.25, rim=0.4, direction=(-0.5, -0.6, 0.6),
                 key_tint=(1.05, 1.0, 0.95), rim_tint=(0.95, 0.98, 1.10)):
    """A virtual key and rim light on the subject.

    The key light is Lambertian shading against the normals, which lifts one
    side of the face and gives it shape. The rim light sits on the silhouette
    where it faces the light, and is the trick that separates subject from
    background - it is why portrait photographers put a lamp behind the subject.
    """
    device, dtype = img.device, img.dtype
    L = torch.tensor(direction, device=device, dtype=dtype)
    L = L / L.norm()

    out = img

    if key > 0:
        n = depth_normals(depth)
        shade = (n * L.view(1, 3, 1, 1)).sum(1, keepdim=True).clamp(min=0)
        shade = separable_blur(shade, 8.0)      # a soft source, not a point light

        # Subtract the mean over the subject so the light *shapes* the face
        # rather than merely brightening it. Without this the whole face is
        # lifted equally and the result reads as washed out, not as lit.
        shade = shade - (shade * alpha).sum() / alpha.sum().clamp(min=1.0)

        tint = torch.tensor(key_tint, device=device, dtype=dtype).view(1, 3, 1, 1)
        out = out * (1.0 + key * shade * alpha * tint)

    if rim > 0:
        k = 9
        eroded = -F.max_pool2d(-alpha, k, stride=1, padding=k // 2)
        edge = (alpha - eroded).clamp(0, 1)

        ax = F.pad(alpha[..., :, 1:] - alpha[..., :, :-1], (0, 1, 0, 0), mode="replicate")
        ay = F.pad(alpha[..., 1:, :] - alpha[..., :-1, :], (0, 0, 0, 1), mode="replicate")
        facing = (-(ax * L[0] + ay * L[1])).clamp(min=0)
        facing = facing / facing.amax().clamp(min=1e-6)

        glow = separable_blur(edge * facing, 4.0)
        tint = torch.tensor(rim_tint, device=device, dtype=dtype).view(1, 3, 1, 1)
        out = out + rim * glow * tint

    return out.clamp(0, 1)


# ---------------------------------------------------------------- background replacement

def near_field_mask(coc, alpha, threshold=0.35, feather=9.0):
    """What to keep from the real image: the subject, plus anything in focus.

    A matte alone only knows "person" and "not person", so a book you hold up
    vanishes the moment segmentation stops counting it as part of you. Depth
    knows the book is nearer than the wall, so we keep everything at or in
    front of the focus plane and replace only what lies behind.
    """
    near = (1.0 - coc / max(1e-3, threshold)).clamp(0, 1)
    keep = (alpha + near * (1.0 - alpha)).clamp(0, 1)
    keep = separable_blur(keep, feather)      # the depth map is coarse at edges
    # Not decoration: the blur above can pull keep below alpha along the
    # silhouette, which would partly replace the edge of the subject.
    return torch.maximum(keep, alpha)


def replace_background(src, fgr, alpha, coc, bg_image, threshold=0.35):
    """Replace the background, leaving whatever is in focus standing."""
    keep = near_field_mask(coc, alpha, threshold)
    extra = (keep - alpha).clamp(min=0)       # in focus, but not the subject
    return fgr * alpha + src * extra + bg_image * (1.0 - alpha - extra)


def fill_behind(img, alpha, sigma=20.0, scale=0.25):
    """Guess what is behind the subject and paint it over them.

    Every plate in the blur modes is built from the camera frame, so it carries
    the subject too. Slide that plate and their ghost slides with it, one soft
    silhouette next to the sharp one. A normalised blur — the frame weighted by
    "not subject", divided by the same weights blurred — spreads the surrounding
    wall across the hole instead. It is only ever seen out of focus and behind a
    person, so a real inpainting model would be wasted here.
    """
    h, w = img.shape[-2:]
    small = F.interpolate(img, scale_factor=scale, mode="bilinear", align_corners=False)
    keep = 1.0 - F.interpolate(alpha, size=small.shape[-2:], mode="bilinear",
                               align_corners=False)
    filled = separable_blur(small * keep, sigma * scale) / \
        separable_blur(keep, sigma * scale).clamp(min=1e-3)
    filled = F.interpolate(filled, size=(h, w), mode="bilinear", align_corners=False)
    return img * (1.0 - alpha) + filled * alpha


def parallax(img, depth, dx, dy, pivot):
    """Reproject the frame for a virtual camera move, using the depth map.

    A backward warp: every output pixel samples the source displaced by its own
    disparity, so near things travel further than far ones and the picture gets
    real volume from a flat sensor. Whatever sits at `pivot` depth does not move
    at all, which is why the pivot belongs on the subject: the face stays nailed
    in place while the room slides behind it.

    Disocclusion is left to the sampler. Holes only open where the shift exceeds
    the depth edge it crosses, so keep the amplitude to a few percent of the
    frame and nobody ever sees one.
    """
    h, w = img.shape[-2:]
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, h, device=img.device, dtype=img.dtype),
        torch.linspace(-1, 1, w, device=img.device, dtype=img.dtype),
        indexing="ij")
    disp = depth[:, 0] - pivot                          # [N,H,W], 0 at the pivot
    grid = torch.stack((xs + dx * disp, ys + dy * disp), dim=-1)
    # align_corners=True is what makes linspace(-1, 1) the identity grid: with
    # False the same grid resamples a half pixel off and softens a still frame.
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border",
                         align_corners=True)


def subject_box(alpha, threshold=0.5):
    """Tightest box around the matte as (x0, y0, x1, y1) in 0..1, or None.

    One `any` per axis rather than a full nonzero: the reduction stays on the
    GPU and only four numbers cross back to the CPU.
    """
    mask = alpha[0, 0] > threshold
    rows, cols = mask.any(1), mask.any(0)
    if not bool(rows.any()):
        return None
    h, w = mask.shape
    ys = rows.nonzero()
    xs = cols.nonzero()
    return (xs[0].item() / w, ys[0].item() / h,
            (xs[-1].item() + 1) / w, (ys[-1].item() + 1) / h)


class AutoFramer:
    """Crop and zoom that follows the subject.

    The matte already knows where the person is, so this needs no tracker and no
    second model: a bounding box, an eased crop and one resize back to full size.

    Everything difficult is the damping. A crop that tracks the box frame by
    frame breathes with you and reads as a nervous operator, so the target only
    counts once it has left a dead zone, and the crop then eases towards it a
    fraction at a time. Standing still, the picture is perfectly still.
    """

    # Room around the subject, as a fraction of its size: a head cropped at the
    # hairline is worse framing than no framing at all.
    MARGIN = 0.55
    # Kept clear above the head, as a fraction of the crop. A webcam subject
    # runs from the hairline to the bottom edge of the frame, so it is usually
    # taller than the crop: centring on it slices the head off, and the top of
    # the box is the part worth keeping.
    HEADROOM = 0.08
    # How far the target may drift, in frame widths, before the crop follows,
    # and how much of the remaining distance is closed each frame.
    DEADZONE, SPEED = 0.04, 0.06

    def __init__(self):
        self.box = None     # the crop currently shown, (cx, cy, w, h) in 0..1

    def reset(self):
        """Forget the crop, so switching back on reframes from what is there now."""
        self.box = None

    def target(self, subject, aspect, zoom):
        """Where the crop wants to be for this subject box."""
        x0, y0, x1, y1 = subject
        w = (x1 - x0) * (1 + self.MARGIN)
        h = (y1 - y0) * (1 + self.MARGIN)
        # Widen to the output aspect rather than squeezing the picture into it.
        w = min(max(w, h * aspect, 1.0 / max(zoom, 1.0)), 1.0)
        h = min(w / aspect, 1.0)
        w = h * aspect
        cx = min(max((x0 + x1) / 2, w / 2), 1 - w / 2)
        cy = min((y0 + y1) / 2, y0 + h / 2 - self.HEADROOM * h)
        cy = min(max(cy, h / 2), 1 - h / 2)
        return (cx, cy, w, h)

    def advance(self, subject, aspect, zoom):
        """Step the crop one frame towards the subject. Returns the crop to use."""
        if subject is None:
            return self.box     # nobody in shot: hold the last framing
        want = self.target(subject, aspect, zoom)
        if self.box is None:
            self.box = want
        elif max(abs(a - b) for a, b in zip(want, self.box)) > self.DEADZONE:
            self.box = tuple(b + (a - b) * self.SPEED for a, b in zip(want, self.box))
        return self.box

    def apply(self, img, alpha, zoom):
        """Reframe `img` on the subject in `alpha`. Both are [1,3|1,H,W]."""
        h, w = img.shape[-2:]
        box = self.advance(subject_box(alpha), w / h, zoom)
        if box is None:
            return img
        cx, cy, bw, bh = box
        x0 = min(max(int((cx - bw / 2) * w), 0), w - 8)
        y0 = min(max(int((cy - bh / 2) * h), 0), h - 8)
        x1 = min(max(int((cx + bw / 2) * w), x0 + 8), w)
        y1 = min(max(int((cy + bh / 2) * h), y0 + 8), h)
        if (x1 - x0, y1 - y0) == (w, h):
            return img          # full frame: the resize would only cost softness
        return F.interpolate(img[:, :, y0:y1, x0:x1], size=(h, w),
                             mode="bilinear", align_corners=False)
