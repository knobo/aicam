"""Seasonal figures for the 3D background: Christmas and Easter.

Each figure is put together from the renderer's two kinds of piece, a convex
block of planes and an ellipsoid (which planes may clip), each in a material
of its own: a present is a lacquered box, two ribbon bands and a bow, a nisse
is a cone of a hat over a beard with a nose poking out. The style is the
painted wooden ornament: a few rounded parts, glossy, read at a glance.

Figures are drawn here with y up and z towards the camera, the way one would
sketch them, then turned into the renderer's frame (y down, z away) and scaled
so they reach a radius of 1, like every other shape.

`shapes:christmas` and `shapes:easter` float them; `shapes:season` picks by
the date and is the plain golden shapes the rest of the year.
"""

import datetime
import functools
import math

import numpy as np

import shapes3d as s3

# ---------------------------------------------------------------- materials
# (light end, dark end, metal), as in shapes3d.MATERIALS.

GOLD = ("#fff1b8", "#e0a12e", 1.0)
SILVER = ("#ffffff", "#9aa3ae", 1.0)
RED = ("#ff5a5f", "#b3101f", 0.0)
DEEP_RED = ("#e8323c", "#8c0a16", 0.0)
GREEN = ("#4fd98a", "#0f7a3d", 0.0)
PINE = ("#3fbf6a", "#0b4f27", 0.0)
BLUE = ("#6cb4ff", "#1748c0", 0.0)
PURPLE = ("#d9a6ff", "#7b2fc9", 0.0)
WHITE = ("#ffffff", "#dfe4ea", 0.0)
SNOW = ("#ffffff", "#e9eef5", 0.0)
COOKIE = ("#dd9152", "#94511f", 0.0)
DARK = ("#4a2c1c", "#1c0d06", 0.0)
SKIN = ("#ffcfb3", "#e39478", 0.0)
FUR = ("#b47a45", "#6b3f1d", 0.0)
ANTLER = ("#f4e2bd", "#b8945e", 0.0)
BOOT = ("#4b3426", "#20140c", 0.0)
SACK = ("#d2a676", "#7d5634", 0.0)
SHINY_RED = ("#ff6b6b", "#b3001b", 0.85)
SHINY_BLUE = ("#8cc8ff", "#1a4fd0", 0.85)
SHINY_GREEN = ("#7af0a8", "#0a8a45", 0.85)
SHINY_PURPLE = ("#e2b8ff", "#7d2fd0", 0.85)
YELLOW = ("#fff07a", "#f2b705", 0.0)
ORANGE = ("#ffbf5e", "#e86a00", 0.0)
BEAK = ("#ffb347", "#e25d00", 0.0)
PINK = ("#ffc2d6", "#f0729f", 0.0)
BUNNY = ("#f6efe6", "#bca78f", 0.0)
LEAF = ("#7be08f", "#1f8a3a", 0.0)
PASTELS = [("#ffc2d6", "#f27aa5", 0.0), ("#fff1a8", "#f2c94c", 0.0),
           ("#c7f5dc", "#4fc78a", 0.0), ("#e3d1ff", "#a37cf0", 0.0),
           ("#c9e6ff", "#5aa5f0", 0.0), ("#ffd9b8", "#f59a52", 0.0)]

WRAPS = [RED, GREEN, BLUE, GOLD, PURPLE, WHITE, SILVER]
RIBBONS = [GOLD, WHITE, RED, SILVER, GREEN]
BAUBLES = [SHINY_RED, GOLD, SHINY_BLUE, SHINY_GREEN, SILVER, SHINY_PURPLE]

# ---------------------------------------------------------------- pieces
#
# While a figure is being drawn a piece is a dict: planes n.x <= h, the bevel,
# the bevel's offsets, a material, and an ellipsoid (centre, rows) or None.

I3 = np.eye(3)
FAR = s3.FAR_PLANE


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _piece(n, h, bevel, material, hb=None, ellipsoid=None):
    n, h = np.asarray(n, float).reshape(-1, 3), np.asarray(h, float)
    return dict(n=n, h=h, hb=h.copy() if hb is None else np.asarray(hb, float),
                bevel=bevel, mat=material, ell=ellipsoid)


def moved(piece, rot=I3, at=(0, 0, 0)):
    """The piece turned by `rot` about the origin, then moved to `at`."""
    at = np.asarray(at, float)
    n = piece["n"] @ rot.T
    out = dict(piece, n=n, h=piece["h"] + n @ at, hb=piece["hb"] + n @ at)
    if piece["ell"] is not None:
        c, rows = piece["ell"]
        out["ell"] = (rot @ c + at, rows @ rot.T)
    return out


def box(at, half, material, rot=I3, bevel=0.03):
    half = np.asarray(half, float)
    n = np.concatenate([I3, -I3])
    return moved(_piece(n, np.concatenate([half, half]), bevel, material), rot, at)


def bar(p, q, thick, material, bevel=None):
    """A square rod from p to q."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    along = (q - p) / np.linalg.norm(q - p)
    side = np.cross(along, [0, 0, 1.0])
    if np.linalg.norm(side) < 1e-6:
        side = np.cross(along, [1.0, 0, 0])
    side /= np.linalg.norm(side)
    rot = np.stack([along, side, np.cross(along, side)], 1)
    return box((p + q) / 2, [np.linalg.norm(q - p) / 2 + thick / 2, thick / 2, thick / 2],
               material, rot, bevel if bevel is not None else thick * 0.35)


def frustum(base, r_bottom, r_top, height, material, sides=8, rot=I3, bevel=0.04):
    """An upright `sides`-gon tapering from r_bottom to r_top (0: a point)
    over `height`, standing on `base`. The radii are to the middle of a side."""
    n, h = [], []
    for k in range(sides):
        a = 2 * math.pi * (k + 0.5) / sides
        u = np.array([math.cos(a), 0.0, math.sin(a)])
        normal = height * u + (r_bottom - r_top) * np.array([0, 1.0, 0])
        normal /= np.linalg.norm(normal)
        n.append(normal)
        h.append(normal @ (r_bottom * u))
    n.append([0, -1.0, 0])
    h.append(0.0)
    if r_top > 0:
        n.append([0, 1.0, 0])
        h.append(height)
    return moved(_piece(n, h, bevel, material), rot, base)


def ellipsoid(at, radii, material, rot=I3, clip=(), cut=()):
    """An ellipsoid, optionally clipped by planes (normal, offset), n.x <= h,
    in the figure's frame. `clip` planes are surfaces and get the bevel; `cut`
    planes are seams inside the figure and do not."""
    at = np.asarray(at, float)
    rows = (rot / np.asarray(radii, float)).T          # row i: axis i over its radius
    n = [c[0] for c in clip] + [c[0] for c in cut]
    h = [c[1] for c in clip] + [c[1] for c in cut]
    hb = [c[1] for c in clip] + [FAR for _ in cut]
    if n:
        n = np.asarray(n, float)
        n = n / np.linalg.norm(n, axis=1, keepdims=True)
    return _piece(n if len(h) else np.zeros((0, 3)), h, 0.02, material, hb=hb,
                  ellipsoid=(at, rows))


def ball(at, r, material):
    return ellipsoid(at, (r, r, r), material)


def extruded(outline, z, half_depth, material, bevel=0.02):
    """A flat polygon (counter-clockwise, in xy) given thickness, in as few
    convex pieces as it takes."""
    outline = np.asarray(outline, float)
    parts = s3.convex_parts(outline)
    out = []
    for part in parts:
        n, h, bev, hb = s3._slab(part, half_depth, bevel,
                                 lambda p, q: s3._on_outline(p, q, outline))
        out.append(moved(_piece(n, h, bev, material, hb=hb), at=(0, 0, z)))
    return out


def star_outline(outer, inner, points=5, at=(0, 0)):
    pts = []
    for k in range(points * 2):
        a = math.pi / 2 + k * math.pi / points
        r = outer if k % 2 == 0 else inner
        pts.append((at[0] + r * math.cos(a), at[1] + r * math.sin(a)))
    return pts


# ---------------------------------------------------------------- figures

def present(variant):
    wrap = WRAPS[variant % len(WRAPS)]
    ribbon = RIBBONS[(variant * 3 + 1) % len(RIBBONS)]
    if ribbon == wrap:
        ribbon = RIBBONS[(variant * 3 + 2) % len(RIBBONS)]
    w, h, d = [(0.62, 0.5, 0.62), (0.45, 0.66, 0.45), (0.78, 0.32, 0.55)][variant % 3]
    band, lift = 0.12, 0.014
    tilt = math.radians(28)
    return [
        box((0, 0, 0), (w, h, d), wrap, bevel=0.035),
        box((0, 0, 0), (band, h + lift, d + lift), ribbon, bevel=0.02),
        box((0, 0, 0), (w + lift, h + lift, band), ribbon, bevel=0.02),
        ellipsoid((0.19, h + 0.1, 0), (0.22, 0.12, 0.08), ribbon, rot=rot_z(tilt)),
        ellipsoid((-0.19, h + 0.1, 0), (0.22, 0.12, 0.08), ribbon, rot=rot_z(-tilt)),
        ball((0, h + 0.06, 0), 0.085, ribbon),
    ]


def gingerbread(variant):
    cookie = []
    flat = 0.1
    cookie.append(ellipsoid((0, 0.62, 0), (0.3, 0.3, flat), COOKIE))
    cookie.append(ellipsoid((0, 0.04, 0), (0.36, 0.46, flat), COOKIE))
    for s in (1, -1):
        cookie.append(ellipsoid((s * 0.42, 0.24, 0), (0.36, 0.13, flat * 0.95), COOKIE,
                                rot=rot_z(s * math.radians(22))))
        cookie.append(ellipsoid((s * 0.21, -0.48, 0), (0.15, 0.38, flat * 0.95), COOKIE,
                                rot=rot_z(s * math.radians(14))))
    icing = []
    for s in (1, -1):
        icing.append(ball((s * 0.1, 0.68, 0.072), 0.042, DARK))                 # eyes
        icing.append(ellipsoid((s * 0.64, 0.32, 0.06), (0.03, 0.12, 0.03), WHITE,
                               rot=rot_z(s * math.radians(22))))              # cuffs
        icing.append(ellipsoid((s * 0.27, -0.72, 0.055), (0.12, 0.03, 0.03), WHITE,
                               rot=rot_z(s * math.radians(14))))
    for k, x in enumerate((-0.1, -0.035, 0.035, 0.1)):                            # smile
        icing.append(ball((x, 0.53 - 0.03 * (1 - abs(x) / 0.1), 0.075), 0.026, WHITE))
    buttons = [RED, GREEN, RED] if variant % 2 == 0 else [WHITE, RED, WHITE]
    for y, m in zip((0.22, 0.04, -0.14), buttons):
        icing.append(ball((0, y, 0.085), 0.05, m))
    return cookie + icing


def nisse(variant):
    coat = [RED, ("#8a97a8", "#46505e", 0.0)][variant % 2]
    lean = math.radians(9)
    hat_base, hat_h = np.array([0, 0.06, 0.0]), 0.95
    apex = hat_base + rot_z(-lean) @ np.array([0, hat_h, 0])
    return [
        frustum((0, -1.0, 0), 0.5, 0.2, 0.78, coat),
        ellipsoid((0, -0.3, 0.17), (0.34, 0.46, 0.22), SNOW),                 # beard
        ball((0, 0.0, 0.33), 0.115, SKIN),                                    # nose
        frustum(hat_base, 0.4, 0.0, hat_h, RED, rot=rot_z(-lean)),
        frustum(hat_base + [0, -0.05, 0], 0.43, 0.43, 0.12, SNOW, rot=rot_z(-lean)),
        ball(apex + [0, 0.02, 0], 0.11, SNOW),                                # pompom
        ellipsoid((0.4, -0.55, 0.14), (0.11, 0.12, 0.11), GREEN),             # mittens
        ellipsoid((-0.4, -0.55, 0.14), (0.11, 0.12, 0.11), GREEN),
        ellipsoid((0.2, -1.0, 0.12), (0.15, 0.08, 0.22), BOOT),
        ellipsoid((-0.2, -1.0, 0.12), (0.15, 0.08, 0.22), BOOT),
    ]


def reindeer(variant):
    parts = [
        ellipsoid((0, 0, 0), (0.55, 0.27, 0.25), FUR),                        # body
        ellipsoid((0.45, 0.26, 0), (0.12, 0.28, 0.12), FUR, rot=rot_z(math.radians(-35))),
        ellipsoid((0.63, 0.5, 0), (0.21, 0.14, 0.14), FUR, rot=rot_z(math.radians(-15))),
        ellipsoid((0.8, 0.44, 0), (0.12, 0.1, 0.11), FUR),                    # snout
        ball((0.92, 0.45, 0), 0.07, ("#ff6b6b", "#c4001a", 0.3) if variant % 2 == 0 else DARK),
        ball((-0.56, 0.13, 0), 0.08, SNOW),                                   # tail
    ]
    for z in (0.12, -0.12):
        parts.append(ball((0.68, 0.56, z), 0.035, DARK))                     # eyes
        parts.append(ellipsoid((0.54, 0.63, z * 1.15), (0.1, 0.035, 0.05), FUR,
                               rot=rot_z(math.radians(30))))                 # ears
    for z in (0.07, -0.07):
        root = (0.57, 0.62, z)
        top = (0.42, 1.02, z * 1.6)
        parts.append(bar(root, top, 0.05, ANTLER))
        parts.append(bar((0.5, 0.8, z * 1.3), (0.64, 0.96, z * 1.5), 0.04, ANTLER))
        parts.append(bar((0.45, 0.94, z * 1.5), (0.33, 1.06, z * 1.8), 0.04, ANTLER))
    for x in (0.33, -0.33):
        for z in (0.13, -0.13):
            parts.append(bar((x, -0.12, z), (x + 0.02, -0.78, z), 0.075, FUR))
    return parts


SLEIGH_SIDE = [(-0.85, -0.25), (0.45, -0.25), (0.7, -0.1), (0.82, 0.15), (0.78, 0.38),
               (0.62, 0.42), (0.58, 0.2), (0.45, 0.05), (-0.35, 0.05), (-0.55, 0.45),
               (-0.85, 0.5)]


def sleigh(variant):
    body = [RED, GREEN, BLUE][variant % 3]
    parts = []
    for z in (0.36, -0.36):
        parts += extruded(SLEIGH_SIDE, z, 0.04, body)
        parts.append(bar((-0.92, -0.55, z), (0.62, -0.55, z), 0.05, GOLD))
        parts.append(bar((0.62, -0.55, z), (0.86, -0.4, z), 0.05, GOLD))
        parts.append(bar((0.86, -0.4, z), (0.9, -0.2, z), 0.05, GOLD))
        parts.append(bar((-0.55, -0.55, z), (-0.55, -0.25, z), 0.045, GOLD))
        parts.append(bar((0.25, -0.55, z), (0.25, -0.25, z), 0.045, GOLD))
    parts.append(box((-0.17, -0.15, 0), (0.68, 0.1, 0.33), body))           # floor
    parts.append(box((-0.71, 0.2, 0), (0.1, 0.28, 0.32), GREEN if body != GREEN else RED))
    parts.append(box((-0.35, 0.09, 0), (0.28, 0.06, 0.32), GREEN if body != GREEN else RED))
    parts.append(ellipsoid((0.16, 0.24, 0), (0.24, 0.3, 0.22), SACK))      # the sack
    parts.append(ball((0.16, 0.55, 0), 0.07, SACK))
    return parts


def tree(variant):
    parts = [
        frustum((0, -0.55, 0), 0.7, 0.25, 0.6, PINE),
        frustum((0, -0.12, 0), 0.55, 0.17, 0.56, PINE),
        frustum((0, 0.3, 0), 0.4, 0.0, 0.58, PINE),
        box((0, -0.68, 0), (0.1, 0.14, 0.1), FUR),
    ]
    parts += extruded(star_outline(0.2, 0.085, at=(0, 0.98)), 0, 0.05, GOLD)
    colours = [SHINY_RED, GOLD, SHINY_BLUE, SILVER, SHINY_RED, GOLD]
    # On the cones' surface, not in it: the radius of the tier at that height
    # plus most of a bauble, so they hang off the branches.
    spots = [(-0.42, 0.68, 60), (-0.44, 0.68, 180), (-0.4, 0.67, 300),
             (0.0, 0.5, 120), (0.02, 0.49, 0), (0.42, 0.33, 240)]
    for (y, r, deg), m in zip(spots, colours[variant % 2:] + colours[:variant % 2]):
        a = math.radians(deg)
        parts.append(ball((r * math.cos(a), y, r * math.sin(a)), 0.075, m))
    return parts


def bauble(variant):
    shell = BAUBLES[variant % len(BAUBLES)]
    trim = WHITE if shell in (GOLD, SILVER) else GOLD
    r = 0.86
    big = r * 1.025
    return [
        ball((0, -0.08, 0), r, shell),
        ellipsoid((0, -0.08, 0), (big, big, big), trim,
                  clip=[((0, 1, 0), 0.02), ((0, -1, 0), 0.18)]),
        frustum((0, 0.68, 0), 0.17, 0.17, 0.18, GOLD, bevel=0.03),
    ]


def star(variant):
    return extruded(star_outline(1.0, 0.45), 0, 0.16, GOLD if variant % 2 == 0 else SILVER,
                    bevel=0.05)


def egg(variant):
    shell = PASTELS[variant % len(PASTELS)]
    other = PASTELS[(variant + 2) % len(PASTELS)]
    dots = PASTELS[(variant + 4) % len(PASTELS)]
    if variant % 7 == 6:
        shell, other, dots = GOLD, ("#ffffff", "#f0d9a0", 0.0), SHINY_RED
    top, bottom = (0.62, 0.92, 0.62), (0.62, 0.66, 0.62)
    grow = 1.03
    parts = [
        ellipsoid((0, 0, 0), top, shell, cut=[((0, -1, 0), 0.0)]),
        ellipsoid((0, 0, 0), bottom, shell, cut=[((0, 1, 0), 0.0)]),
    ]
    pattern = variant % 3
    if pattern in (0, 2):
        parts.append(ellipsoid((0, 0, 0), tuple(r * grow for r in top), other,
                               clip=[((0, 1, 0), 0.36), ((0, -1, 0), -0.22)]))
        parts.append(ellipsoid((0, 0, 0), tuple(r * grow for r in bottom), other,
                               clip=[((0, 1, 0), -0.2), ((0, -1, 0), 0.34)]))
    if pattern in (1, 2):
        for k in range(6):
            a = 2 * math.pi * k / 6 + 0.3
            parts.append(ball((0.61 * math.cos(a), 0.0, 0.61 * math.sin(a)), 0.075, dots))
    return parts


def chick(variant):
    parts = [
        ellipsoid((0, -0.2, 0), (0.48, 0.44, 0.44), YELLOW),
        ball((0, 0.36, 0.04), 0.31, YELLOW),
        frustum((0, 0, 0), 0.09, 0.0, 0.2, BEAK, sides=4,
                rot=rot_x(math.pi / 2)),
        ellipsoid((0.46, -0.16, 0), (0.12, 0.24, 0.28), YELLOW, rot=rot_z(math.radians(25))),
        ellipsoid((-0.46, -0.16, 0), (0.12, 0.24, 0.28), YELLOW, rot=rot_z(math.radians(-25))),
        ellipsoid((0.04, 0.7, 0), (0.05, 0.11, 0.05), YELLOW, rot=rot_z(math.radians(-20))),
        ellipsoid((-0.05, 0.69, 0), (0.05, 0.09, 0.05), YELLOW, rot=rot_z(math.radians(25))),
        ellipsoid((0, -0.5, 0), (0.6, 0.55, 0.6), WHITE, clip=[((0, 1, 0), -0.32)]),
    ]
    parts[2] = moved(parts[2], at=(0, 0.3, 0.31))
    for s in (1, -1):
        parts.append(ball((s * 0.12, 0.44, 0.27), 0.05, DARK))
    return parts


def bunny(variant):
    fur = [BUNNY, COOKIE][variant % 2]
    parts = [
        ellipsoid((0, -0.35, 0), (0.42, 0.45, 0.38), fur),
        ellipsoid((0, 0.2, 0.08), (0.3, 0.27, 0.27), fur),
        ball((0, -0.5, -0.38), 0.13, WHITE),                                  # tail
        ball((0, 0.17, 0.35), 0.05, PINK),                                    # nose
    ]
    for s in (1, -1):
        tilt = rot_z(s * math.radians(-10))
        parts.append(ellipsoid((s * 0.13, 0.68, 0), (0.1, 0.36, 0.05), fur, rot=tilt))
        parts.append(ellipsoid((s * 0.13, 0.66, 0.035), (0.06, 0.28, 0.03), PINK, rot=tilt))
        parts.append(ball((s * 0.12, 0.27, 0.3), 0.045, DARK))
        parts.append(ellipsoid((s * 0.15, -0.74, 0.3), (0.09, 0.07, 0.14), fur))
        parts.append(ellipsoid((s * 0.26, -0.78, 0.12), (0.12, 0.07, 0.22), fur))
    return parts


def daffodil(variant):
    centre = np.array([0, 0.38, 0.0])
    parts = [
        frustum((0, -1.0, 0), 0.045, 0.045, 1.35, LEAF),                     # stem
        ellipsoid((0.15, -0.55, 0), (0.06, 0.5, 0.03), LEAF, rot=rot_z(math.radians(-12))),
        ellipsoid((-0.15, -0.6, 0), (0.06, 0.45, 0.03), LEAF, rot=rot_z(math.radians(14))),
    ]
    for k in range(6):
        a = k * math.pi / 3
        out = rot_z(a) @ np.array([0, 0.24, 0])
        parts.append(ellipsoid(centre + out, (0.13, 0.27, 0.04), YELLOW, rot=rot_z(a)))
    trumpet = frustum((0, 0, 0), 0.13, 0.19, 0.24, ORANGE if variant % 2 == 0 else YELLOW,
                      rot=rot_x(math.pi / 2))
    parts.append(moved(trumpet, at=centre + [0, 0, 0.02]))
    return parts


def orange(variant):
    return [
        ellipsoid((0, -0.05, 0), (0.86, 0.8, 0.86), ORANGE),
        ellipsoid((0.2, 0.8, 0), (0.24, 0.05, 0.11), LEAF, rot=rot_z(math.radians(20))),
        ball((0, 0.76, 0), 0.06, FUR),
    ]


FIGURES = {
    "present": present, "gingerbread": gingerbread, "nisse": nisse,
    "reindeer": reindeer, "sleigh": sleigh, "tree": tree, "bauble": bauble,
    "star": star, "egg": egg, "chick": chick, "bunny": bunny,
    "daffodil": daffodil, "orange": orange,
}
# These stand up and sway; the rest tumble like the polyhedra.
UPRIGHT = {"nisse", "reindeer", "sleigh", "tree", "chick", "bunny", "daffodil"}

# Who comes to each season: everyone once, then the rest from `more`, so a
# swarm is never short of presents.
SEASONS = {
    "christmas": dict(cast=["present", "gingerbread", "nisse", "reindeer", "sleigh",
                            "tree", "bauble", "present", "star", "present"],
                      more=["present", "bauble", "gingerbread"]),
    "easter": dict(cast=["egg", "chick", "bunny", "daffodil", "egg", "orange", "egg",
                         "chick", "egg"],
                   more=["egg"]),
}


def is_figure(kind):
    return kind.split("/")[0] in FIGURES


def is_upright(kind):
    return kind.split("/")[0] in UPRIGHT


def cast(season, count=None, seed=7):
    """The kinds for a swarm, as `name/variant`, every variant different."""
    names = list(SEASONS[season]["cast"])
    more = SEASONS[season]["more"]
    count = count or len(names)
    while len(names) < count:
        names.append(more[len(names) % len(more)])
    names = names[:count]
    seen = {}
    out = []
    for name in names:
        seen[name] = seen.get(name, -1) + 1
        out.append(f"{name}/{(seed + seen[name] * 5) % 97}")
    return out


# ---------------------------------------------------------------- finishing

FLIP = np.diag([1.0, -1.0, -1.0])      # y up, z to camera -> the renderer's y down, z away


def _corners(piece):
    """Points on the piece's surface, enough to bound it."""
    if piece["ell"] is not None:
        c, rows = piece["ell"]
        axes = np.linalg.inv(rows)        # columns: the semi-axes
        return np.array([c + sx * axes[:, i] for i in range(3) for sx in (1, -1)])
    n, h = piece["n"], piece["h"]
    pts = []
    for a, b, c in __import__("itertools").combinations(range(len(h)), 3):
        m = n[[a, b, c]]
        if abs(np.linalg.det(m)) < 1e-9:
            continue
        x = np.linalg.solve(m, h[[a, b, c]])
        if np.all(n @ x <= h + 1e-6):
            pts.append(x)
    return np.array(pts)


@functools.lru_cache(maxsize=None)
def figure(kind):
    """The renderer's pieces for `name/variant`, centred and scaled to radius 1."""
    name, _, variant = kind.partition("/")
    parts = FIGURES[name](int(variant or 0))
    pts = np.concatenate([_corners(p) for p in parts])
    centre = (pts.max(0) + pts.min(0)) / 2
    scale = 1.0 / np.linalg.norm(pts - centre, axis=1).max()
    out = []
    for p in parts:
        p = moved(p, at=-centre)
        p = moved(p, rot=FLIP)
        assert len(p["h"]) <= s3.FACES, f"{name}: a piece with {len(p['h'])} planes"
        ell = None
        if p["ell"] is not None:
            c, rows = p["ell"]
            ell = (c * scale, rows / scale)
        out.append((p["n"], p["h"] * scale, p["bevel"] * scale, p["hb"] * scale,
                    p["mat"], ell))
    return out


# ---------------------------------------------------------------- the calendar

def easter(year):
    """Easter Sunday, by the anonymous Gregorian algorithm (Meeus/Jones/Butcher)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month = (h + m - 7 * n + 114) // 31
    day = (h + m - 7 * n + 114) % 31 + 1
    return datetime.date(year, month, day)


def first_advent(year):
    """The fourth Sunday before Christmas Day."""
    christmas = datetime.date(year, 12, 25)
    sunday_before = christmas - datetime.timedelta(days=(christmas.weekday() + 1) % 7 or 7)
    return sunday_before - datetime.timedelta(weeks=3)


def season_of(day=None):
    """christmas from the first Sunday of Advent to Epiphany, easter from
    two weeks before Easter Sunday to Easter Monday, otherwise None."""
    day = day or datetime.date.today()
    if day >= first_advent(day.year) or day <= datetime.date(day.year, 1, 6):
        return "christmas"
    e = easter(day.year)
    if e - datetime.timedelta(days=14) <= day <= e + datetime.timedelta(days=1):
        return "easter"
    return None
