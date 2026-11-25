"""A background of shiny 3D shapes that float around and keep clear of you.

Rendered on the GPU every frame, with no meshes and no raymarching. Every shape
is built from convex pieces, and a convex piece is just the intersection of a
handful of half spaces, so a ray hits it exactly where the last plane it enters
comes before the first plane it leaves: one division per plane per pixel.

A solid cube is one piece. A hollow one is its twelve edges, each a bar of its
own, and the ray takes the nearest bar it hits - which is how you see the back
of the frame through the front of it.

Each shape is rendered in a square patch around its own projection rather than
over the whole frame, so the cost follows how much screen the shapes cover, not
the number of shapes times the number of pixels.

The look comes from cheap things:

  bevelled normals  the normal is a softmax over the piece's face planes, so the
                    hard edges shade like rounded ones and catch a line of light;
                    on a bar the same trick makes it read as a polished rod
  softboxes         reflections look up a studio with rectangular lights in it,
                    the way product shots are lit; on black that is what makes
                    gold look like gold
  materials         metal tints its reflection with its own colour, lacquer
                    reflects white over a coloured base

The physics runs on the CPU, for a handful of bodies. The only thing it needs from
the GPU is the matte, pooled down to a 96x54 grid and blurred: its gradient
points away from you, so a shape that drifts towards you is pushed back out,
and when you move, the field rises under the shapes and kicks them.
"""

import itertools
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

PREFIX = "shapes"

# Vertical field of view of the virtual camera. Narrow, like a product lens:
# a wide one makes the near shapes bulge.
FOV = math.radians(38.0)
# Where the shapes live, and where the wall they cast shadows on is.
Z_NEAR, Z_FAR, Z_WALL = 9.0, 15.0, 18.0
# The matte is pooled to this grid before it crosses to the CPU.
FIELD_W, FIELD_H = 96, 54


def is_shapes_spec(path):
    return path == PREFIX or (path or "").startswith(PREFIX + ":")


def _hex(c):
    c = c.lstrip("#")
    return [int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


# Materials: the two ends of the gradient, and how metallic. Metal reflects in
# its own colour and has almost no diffuse; lacquer is a coloured base under a
# clear, white-reflecting coat. In between is candy paint.
GOLD = ("#fff1b8", "#e0a12e", 1.0)
MATERIALS = {
    "noir": [GOLD, ("#fff27a", "#ffb000", 0.35), ("#ff4a4a", "#a8001c", 0.45),
             GOLD, ("#5dffa0", "#00874a", 0.45)],
    "dusk": [("#ff5f6d", "#ffc371", 0.0), ("#5b5cff", "#c86cf0", 0.0),
             ("#00c6ff", "#0058ff", 0.0), ("#f7797d", "#b26cf0", 0.0),
             ("#3ee98a", "#2cd9e0", 0.0), ("#ffd27a", "#ff6f91", 0.0)],
}

PALETTES = {
    # backdrop inner and outer colour, two glows drifting behind the shapes,
    # and which material set the shapes are made of
    "noir":   dict(inner="#0d0c0b", outer="#000000", glows=("#2a1f0a", "#1f0a0a"),
                   glow=0.35, materials="noir"),
    "dusk":   dict(inner="#2c2452", outer="#0b0a18", glows=("#ff4f8b", "#3f6cff"),
                   glow=0.22, materials="dusk"),
    "studio": dict(inner="#f6f3ee", outer="#d3cdc4", glows=("#ffc6d6", "#bcdcff"),
                   glow=0.35, materials="dusk"),
    # The seasons float seasons.py's figures instead, and Christmas snows.
    "christmas": dict(inner="#1a2c44", outer="#050a14", glows=("#5a3410", "#0e3d2c"),
                      glow=0.4, materials="noir", season="christmas", snow=True,
                      sizes=(0.85, 1.1)),
    "easter": dict(inner="#fff6dc", outer="#f1d3a2", glows=("#ffd1e3", "#c9f2d6"),
                   glow=0.45, materials="dusk", season="easter", sizes=(0.7, 0.95)),
}
DEFAULT = "noir"
# Not a palette: whichever season it is, and the default between them.
BY_DATE = "season"


def palette_of(spec, day=None):
    name = spec.split(":", 1)[1] if ":" in spec else DEFAULT
    if name == BY_DATE:
        import seasons
        return seasons.season_of(day) or DEFAULT
    if name not in PALETTES:
        raise ValueError(f"unknown shapes palette {name!r} "
                         f"(have {', '.join(PALETTES)}, or {BY_DATE})")
    return name


# ---------------------------------------------------------------- geometry
#
# A piece is a set of outward normals and offsets, n.x <= h, padded to FACES
# planes with planes too far out to matter. Shapes are scaled so their
# circumradius is about 1.

FACES = 10
FAR_PLANE = 1e3


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _cube():
    n = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    return n, np.full(6, 1 / math.sqrt(3))


def _tetrahedron():
    n = _unit([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    return n, np.full(4, 1 / 3)


def _octahedron():
    n = _unit([[sx, sy, sz] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)])
    return n, np.full(8, 1 / math.sqrt(3))


def _prism():
    """A triangular prism: an equilateral triangle in xy, extruded along z."""
    rho, length = 0.85, 0.5
    sides = [[math.cos(a), math.sin(a), 0.0]
             for a in (math.pi / 2 + k * 2 * math.pi / 3 for k in range(3))]
    n = np.array(sides + [[0, 0, 1], [0, 0, -1]], float)
    return n, np.array([rho / 2] * 3 + [length, length])


SHAPES = {"cube": _cube, "tetrahedron": _tetrahedron,
          "octahedron": _octahedron, "prism": _prism}


def _edges(normals, offsets):
    """The polyhedron's edges, as the pairs of faces that meet in them.

    Every vertex lies on three planes; two faces share an edge when two
    vertices lie on both. Found from the planes alone, so a new shape needs
    nothing but its planes.
    """
    on = []
    for a, b, c in itertools.combinations(range(len(offsets)), 3):
        m = normals[[a, b, c]]
        if abs(np.linalg.det(m)) < 1e-6:
            continue
        x = np.linalg.solve(m, offsets[[a, b, c]])
        if np.all(normals @ x <= offsets + 1e-6):
            faces = frozenset(np.nonzero(np.abs(normals @ x - offsets) < 1e-6)[0])
            if faces not in on:
                on.append(faces)
    return [(i, j) for i, j in itertools.combinations(range(len(offsets)), 2)
            if sum(1 for faces in on if i in faces and j in faces) >= 2]


# ---------------------------------------------------------------- tiles
#
# The aperiodic monotiles: single shapes that tile the plane only without ever
# repeating (Smith, Myers, Kaplan and Goodman-Strauss, 2023). Both belong to
# one family, Tile(a, b), a 14-gon whose edges alternate between length a and
# b; the polygon below is the paper's definition traced as a turtle.
#
#   hat      Tile(1, sqrt 3). Two of its edges line up, so it has 13 sides,
#            and its area is exactly eight kites of the hexagonal kite grid.
#   spectre  Tile(1, 1), all fourteen edges equal. The straight-edged form,
#            which tiles aperiodically as long as mirror images are not used.
#
# tests/test_shapes.py checks both against those properties.

def tile_polygon(a, b):
    """Tile(a, b), counter-clockwise, from the origin."""
    s3 = math.sqrt(3) / 2
    ha, hb = a * s3, b * s3
    return np.array([
        (0, 0), (a, 0), (a + a / 2, -ha), (a + a / 2 + hb, -ha + b / 2),
        (a + a / 2 + hb, -ha + b + b / 2), (2 * a + a / 2 + hb, -ha + b + b / 2),
        (3 * a + hb, b + b / 2), (3 * a, 2 * b), (3 * a - hb, 2 * b - b / 2),
        (2 * a + a / 2 - hb, ha + 2 * b - b / 2), (a + a / 2 - hb, ha + 2 * b - b / 2),
        (a / 2 - hb, ha + 2 * b - b / 2), (-hb, 2 * b - b / 2), (0, b)], float)


TILES = {"hat": (1.0, math.sqrt(3)), "spectre": (1.0, 1.0)}
# Half the slab's thickness, relative to the tile's radius.
TILE_DEPTH = 0.16


def _area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(x @ np.roll(y, -1) - y @ np.roll(x, -1))


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex(points):
    n = len(points)
    return all(_cross(points[i], points[(i + 1) % n], points[(i + 2) % n]) >= -1e-9
               for i in range(n))


def convex_parts(poly, start=0):
    """A simple counter-clockwise polygon as a few convex ones.

    Ear clipping into triangles, then Hertel-Mehlhorn: merge two neighbours
    across their shared diagonal for as long as the merge stays convex. At
    most four times the optimum, and for a 14-gon a handful of pieces.
    `start` is where the ear clipping begins; different starts cut differently.
    """
    idx = list(range(len(poly)))
    idx = idx[start:] + idx[:start]
    tris = []
    while len(idx) > 3:
        for k in range(len(idx)):
            a, b, c = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            if _cross(poly[a], poly[b], poly[c]) <= 1e-12:
                continue                       # reflex or flat: not an ear
            if any(_cross(poly[a], poly[b], poly[p]) >= 0 and
                   _cross(poly[b], poly[c], poly[p]) >= 0 and
                   _cross(poly[c], poly[a], poly[p]) >= 0
                   for p in idx if p not in (a, b, c)):
                continue                       # another vertex inside it
            tris.append([a, b, c])
            idx.pop(k)
            break
    tris.append(idx)

    parts = tris
    merged = True
    while merged:
        merged = False
        for i, j in itertools.combinations(range(len(parts)), 2):
            P, Q = parts[i], parts[j]
            for k in range(len(P)):
                a, b = P[k], P[(k + 1) % len(P)]
                if any(Q[m] == b and Q[(m + 1) % len(Q)] == a for m in range(len(Q))):
                    m = next(m for m in range(len(Q)) if Q[m] == b)
                    ring = P[k + 1:] + P[:k + 1]            # b ... a
                    other = Q[m + 1:] + Q[:m + 1]           # a ... b
                    union = ring + other[1:-1]
                    if _convex(poly[union]):
                        parts[i] = union
                        parts.pop(j)
                        merged = True
                    break
            if merged:
                break
    return [poly[p] for p in parts]


def _on_outline(p, q, outline):
    """Whether the segment p-q lies along the polygon's own boundary."""
    mid = (p + q) / 2
    for a, b in zip(outline, np.roll(outline, -1, 0)):
        e = b - a
        t = np.clip((mid - a) @ e / (e @ e), 0, 1)
        if np.linalg.norm(a + t * e - mid) < 1e-6:
            return True
    return False


def _sides(points):
    """The polygon's corners, with corners that are not corners (180 degrees) left out."""
    return np.array([q for k, q in enumerate(points)
                     if abs(_cross(points[k - 1], q, points[(k + 1) % len(points)])) > 1e-9])


def _mixed(part, outline):
    """Whether a side of the part is partly the outline and partly a cut.

    One plane cannot be both: rounded, the cut half shows as a seam; left
    sharp, the outline half loses its bevel.
    """
    pts = _sides(part)
    for p, q in zip(pts, np.roll(pts, -1, 0)):
        on = {_on_outline(p + (q - p) * t, p + (q - p) * (t + 0.01), outline)
              for t in np.linspace(0.02, 0.97, 24)}
        if len(on) > 1:
            return True
    return False


def tile_parts(outline):
    """The fewest convex parts, among the decompositions with no mixed sides."""
    found = [parts for parts in (convex_parts(outline, start) for start in range(len(outline)))
             if not any(_mixed(part, outline) for part in parts)]
    if not found:
        raise ValueError("no decomposition without a side that is half cut")
    return min(found, key=len)


def _slab(points, depth, bevel, surface):
    """A convex polygon extruded to +-depth, as one piece.

    `surface(p, q)` says whether a side is part of the shape's own surface.
    Sides that are only cuts between pieces get their bevel offset pushed out
    of reach, so the softmax never rounds off a seam that is not there.
    """
    pts = _sides(points)
    n, h, hb = [], [], []
    for p, q in zip(pts, np.roll(pts, -1, 0)):
        e = q - p
        normal = np.array([e[1], -e[0], 0.0]) / np.linalg.norm(e)    # outward for CCW
        n.append(normal)
        h.append(normal[:2] @ p)
        hb.append(h[-1] + (0 if surface(p, q) else FAR_PLANE))
    for z in (1.0, -1.0):
        n.append([0, 0, z])
        h.append(depth)
        hb.append(depth)
    return np.array(n, float), np.array(h), bevel, np.array(hb)


def _tile_outline(kind):
    """The tile centred on its centroid and scaled to a radius of about 1."""
    poly = tile_polygon(*TILES[kind])
    x, y = poly[:, 0], poly[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    centre = np.array([(x + np.roll(x, -1)) @ cross, (y + np.roll(y, -1)) @ cross]) \
        / (6 * _area(poly))
    poly = poly - centre
    return poly / np.linalg.norm(poly, axis=1).max() * math.sqrt(1 - TILE_DEPTH ** 2)


def _tile_pieces(kind, hollow, width):
    outline = _tile_outline(kind)
    if not hollow:
        # A tighter bevel than the polyhedra get: the softmax has a tail, and
        # across a piece only a few bevels wide it tilts the whole top, each
        # piece its own way, so the cuts between them show.
        return [_slab(part, TILE_DEPTH, 0.015, lambda p, q: _on_outline(p, q, outline))
                for part in tile_parts(outline)]

    # The outline as a thin rod: one bar per edge, mitred into its neighbours.
    count = len(outline)
    inward = []
    for k in range(count):
        e = outline[(k + 1) % count] - outline[k]
        inward.append(np.array([-e[1], e[0]]) / np.linalg.norm(e))   # left of a CCW edge
    inner = []
    for k in range(count):
        # Where the two neighbouring edges, moved inward by `width`, cross.
        m = inward[k - 1] + inward[k]
        inner.append(outline[k] + m * width / max(m @ inward[k], 1e-6))
    bars = []
    for k in range(count):
        a, b = outline[k], outline[(k + 1) % count]
        along = (b - a) / np.linalg.norm(b - a)
        # The bar's outer and inner sides run along the edge; the mitres at
        # its ends are cuts.
        bars.append(_slab(np.array([a, b, inner[(k + 1) % count], inner[k]]), width / 2,
                          width * 0.5,
                          lambda p, q, along=along: abs(_cross((0, 0), q - p, along)) < 1e-9))
    return bars


def pieces(kind, hollow, width=0.07):
    """The convex pieces of a shape, each (normals, offsets, bevel, bevel offsets).

    The bevel offsets are the offsets again, except for planes that are cuts
    inside the shape, which are moved out of reach of the bevel.

    A hollow polyhedron is its frame: for each edge, the part of the solid that
    lies within `width` of both faces meeting there. The outer faces stay the
    solid's own, the other faces cap each piece, and where the pieces meet at a
    corner they overlap inside the solid, so the joints come out mitred without
    anyone working out a miter.
    """
    if kind in TILES:
        return _tile_pieces(kind, hollow, width)
    import seasons
    if seasons.is_figure(kind):
        return seasons.figure(kind)
    n, h = SHAPES[kind]()
    if not hollow:
        return [(n, h, 0.05, h)]
    out = []
    for i, j in _edges(n, h):
        rest = [f for f in range(len(h)) if f not in (i, j)]
        nn = np.concatenate([n[[i, j]], -n[[i, j]], n[rest]])
        hh = np.concatenate([h[[i, j]], width - h[[i, j]], h[rest]])
        out.append((nn, hh, width * 0.5, hh))
    return out


def _pad(piece):
    """Pad to FACES planes with planes so far out they never matter.

    Not a repeat of a real plane: that would intersect the same, but count
    twice in the bevel's softmax and pull the rounded normal towards it.

    Also fills in the two optional fields a piece may carry, so every piece
    comes out as (normals, offsets, bevel, bevel offsets, material, ellipsoid):
    material (low, high, metal) or None for the shape's own, and ellipsoid
    (centre, rows) or None for a piece that is planes only.
    """
    n, h, bevel, hb = piece[:4]
    material = piece[4] if len(piece) > 4 else None
    ellipsoid = piece[5] if len(piece) > 5 else None
    if len(h) == 0:                    # a whole ellipsoid: nothing clips it
        n, h, hb = np.array([[0.0, 0.0, 1.0]]), np.array([FAR_PLANE]), np.array([FAR_PLANE])
    extra = FACES - len(h)
    assert extra >= 0, f"a piece with {len(h)} planes; FACES is {FACES}"
    return (np.concatenate([n, np.repeat(n[:1], extra, 0)]),
            np.concatenate([h, np.full(extra, FAR_PLANE)]), bevel,
            np.concatenate([hb, np.full(extra, FAR_PLANE)]), material, ellipsoid)


def _rotation(axis_angle):
    """Rodrigues: a rotation matrix from an axis scaled by its angle."""
    theta = float(np.linalg.norm(axis_angle))
    if theta < 1e-9:
        return np.eye(3)
    k = axis_angle / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta) * kx + (1 - math.cos(theta)) * kx @ kx


def _random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ---------------------------------------------------------------- physics

class Swarm:
    """Positions, spins and the forces on them. Plain numpy: there are a handful."""

    # Spring towards each shape's wandering home, and how fast motion dies out.
    # Low damping is what lets a push turn into a bounce rather than a slide.
    HOME, DAMPING = 0.4, 2.0
    # Shapes pushing each other apart, the matte pushing them away from you,
    # and the extra shove when you move into one.
    CONTACT, BODY, KICK = 30.0, 20.0, 20.0

    def __init__(self, count=7, seed=7, aspect=16 / 9, hollow=0.45, cast=None,
                 sizes=(0.5, 0.85)):
        """`cast`, when given, is the kinds to float, one per shape, and
        `count` is its length; otherwise the tiles and polyhedra take turns."""
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.tan_y = math.tan(FOV / 2)
        self.tan_x = self.tan_y * aspect
        if cast is not None:
            import seasons
            count = len(cast)
            self.kind = list(cast)
            self.hollow = [False] * count
            # A nisse on his head is not festive. These stand, sway and turn,
            # and only the rest tumble.
            self.upright = np.array([seasons.is_upright(k) for k in cast])
        else:
            # The tiles first in the cycle, so even a small swarm has both.
            names = list(TILES) + list(SHAPES)
            self.kind = [names[i % len(names)] for i in range(count)]
            rng.shuffle(self.kind)
            self.hollow = list(rng.permutation(count) < round(count * hollow))
            self.upright = np.zeros(count, bool)
        self.material = list(rng.permutation(count))
        self.size = rng.uniform(*sizes, count)

        # Mostly towards the sides, where a person usually is not.
        z = rng.uniform(Z_NEAR + 0.5, Z_FAR - 0.5, count)
        u = np.array([(-1 if i % 2 else 1) * rng.uniform(0.35, 0.92) for i in range(count)])
        v = rng.uniform(-0.8, 0.8, count)
        self.pos = np.stack([u * z * self.tan_x, v * z * self.tan_y, z], 1)
        self.home = self.pos.copy()
        self.vel = np.zeros((count, 3))
        self.rot = np.stack([_random_rotation(rng) for _ in range(count)])
        self.spin = _unit(rng.normal(size=(count, 3))) * \
            rng.uniform(0.05, 0.14, count)[:, None]
        self.phase = rng.uniform(0, 2 * math.pi, (count, 3))
        self.freq = rng.uniform(0.015, 0.045, (count, 3))
        # Upright ones: which way they face, and how fast a knock turns them.
        # Facing the camera or side on, never edge on, mirrored at random.
        self.facing = rng.choice([0.0, math.pi], count) + rng.uniform(-0.5, 0.5, count)
        self.turn = np.zeros(count)
        self.yaw = np.zeros(count)
        self.t = 0.0
        self._field = None
        self._previous = None

    def observe(self, field):
        """The blurred matte as a (FIELD_H, FIELD_W) array in 0..1."""
        self._previous, self._field = self._field, field

    def screen(self):
        """Each centre in -1..1 screen units, x and y."""
        z = self.pos[:, 2]
        return self.pos[:, 0] / (z * self.tan_x), self.pos[:, 1] / (z * self.tan_y)

    def _body_force(self, dt):
        """Away from you, and how hard, per shape."""
        n = len(self.pos)
        if self._field is None:
            return np.zeros((n, 3)), np.zeros(n)
        field = self._field
        h, w = field.shape
        gy, gx = np.gradient(field)
        sx, sy = self.screen()
        fx = np.clip((sx * 0.5 + 0.5) * w - 0.5, 0, w - 1.001)
        fy = np.clip((sy * 0.5 + 0.5) * h - 0.5, 0, h - 1.001)
        x0, y0 = fx.astype(int), fy.astype(int)
        ax, ay = fx - x0, fy - y0

        def at(a):
            return ((a[y0, x0] * (1 - ax) + a[y0, x0 + 1] * ax) * (1 - ay)
                    + (a[y0 + 1, x0] * (1 - ax) + a[y0 + 1, x0 + 1] * ax) * ay)

        m, dx, dy = at(field), at(gx) * w / 2, at(gy) * h / 2
        norm = np.hypot(dx, dy)
        # Deep inside the matte the blurred gradient goes flat; push sideways
        # out of whichever side the shape is already on.
        flat = norm < 1e-3
        dx = np.where(flat, np.sign(sx + 1e-6), -dx / np.maximum(norm, 1e-6))
        dy = np.where(flat, 0.0, -dy / np.maximum(norm, 1e-6))

        push = self.BODY * np.clip((m - 0.015) / 0.25, 0, 1) ** 0.7
        if self._previous is not None and dt > 0:
            # You moving into a shape: the field rising under it. That is the kick.
            rising = np.maximum(at(field - self._previous), 0) / dt
            push = push + self.KICK * np.minimum(rising, 4.0)
        z = self.pos[:, 2]
        force = np.zeros((n, 3))
        force[:, 0] = push * dx * z * self.tan_x * 0.2
        force[:, 1] = push * dy * z * self.tan_y * 0.2
        return force, push

    def step(self, dt):
        dt = min(max(dt, 0.0), 0.05)
        self.t += dt
        n = len(self.pos)

        # The home itself wanders, slowly, so the shapes drift instead of hover.
        wander = np.sin(self.freq * 2 * math.pi * self.t + self.phase)
        target = self.home + wander * np.array([0.5, 0.35, 0.6])
        force = (target - self.pos) * self.HOME

        body, push = self._body_force(dt)
        force += body
        # A shove sets a shape spinning a little faster, as a knock would.
        self.spin += self.rng.normal(size=(n, 3)) * (push * dt * 0.006)[:, None]

        # Shapes do not pass through each other.
        delta = self.pos[:, None] - self.pos[None]
        dist = np.linalg.norm(delta, axis=-1) + np.eye(n)
        reach = (self.size[:, None] + self.size[None]) * 0.95
        overlap = np.clip(reach - dist, 0, None) * (1 - np.eye(n))
        force += (delta / dist[..., None] * overlap[..., None]).sum(1) * self.CONTACT

        # Soft walls: the sides of the view, and a slab in depth.
        sx, sy = self.screen()
        z = self.pos[:, 2]
        for axis, s, tan, limit in ((0, sx, self.tan_x, 1.05), (1, sy, self.tan_y, 1.0)):
            force[:, axis] -= np.sign(s) * np.clip(np.abs(s) - limit, 0, None) * z * tan * 40
        force[:, 2] += np.clip(Z_NEAR - z, 0, None) * 30 - np.clip(z - Z_FAR, 0, None) * 30

        self.vel += force * dt
        self.vel *= math.exp(-self.DAMPING * dt)
        self.pos += self.vel * dt

        # Spin settles back to a calm drift, not to a stop.
        speed = np.linalg.norm(self.spin, axis=1, keepdims=True)
        calm = np.clip(speed, 0.05, None)
        self.spin *= (calm + (speed - calm) * math.exp(-0.8 * dt)) / np.maximum(speed, 1e-9)

        # Upright: a knock turns them about the vertical, the turn dies away,
        # and they sway back towards facing where they faced. Only the sway
        # and a small rock, so they never end up edge on for long.
        self.turn += self.rng.normal(size=n) * push * dt * 0.08
        self.turn *= math.exp(-0.6 * dt)
        self.yaw += self.turn * dt - self.yaw * 0.15 * dt
        for i in range(n):
            if self.upright[i]:
                f, ph = self.freq[i], self.phase[i]
                sway = 0.45 * math.sin(2 * math.pi * f[0] * 1.6 * self.t + ph[0])
                rock_x = 0.10 * math.sin(2 * math.pi * f[1] * 3 * self.t + ph[1])
                rock_z = 0.12 * math.sin(2 * math.pi * f[2] * 3 * self.t + ph[2])
                self.rot[i] = _rotation(np.array([0.0, self.facing[i] + sway + self.yaw[i], 0.0])) \
                    @ _rotation(np.array([rock_x, 0.0, rock_z]))
            else:
                self.rot[i] = _rotation(self.spin[i] * dt) @ self.rot[i]


# ---------------------------------------------------------------- rendering
#
# The same shading exists twice: as plain torch, which runs anywhere and is
# what the tests check, and as one Triton kernel for CUDA. Run op by op, the
# torch version makes dozens of passes over patches of tens of megabytes and
# costs 30 ms a frame; torch.compile got that to 7 ms but took 13-40 s to
# start. The kernel keeps a pixel in registers from ray to colour.

KEY = tuple(_unit([-0.55, -0.75, -0.45]).tolist())


def _box(centre, tangent, half_w, half_h, intensity):
    c, t1 = _unit(centre), _unit(tangent)
    return [*c, *t1, *np.cross(c, t1), half_w, half_h, intensity]


# The softboxes the shapes reflect: big key upper left, a tall strip on the
# right, a wide one overhead. Each row: centre, two tangents, half sizes, power.
BOXES = [
    _box([-0.55, -0.65, -0.55], [0.7, -0.7, 0.0], 0.30, 0.16, 4.0),     # key, upper left
    _box([0.85, -0.15, -0.5], [0, 1, 0], 0.18, 0.5, 1.6),                # strip, right
    _box([0.0, -1.0, -0.15], [1, 0, 0], 0.55, 0.2, 1.2),                 # overhead
    _box([-0.95, 0.15, -0.25], [0, 1, 0], 0.15, 0.35, 0.8),              # kicker, low left
    _box([0.35, -0.45, 0.82], [1, 0, 0], 0.4, 0.2, 1.8),                 # back light, for rims
]
# Soft edges on the softboxes, wide enough that their reflection in a narrow
# rounded edge is a glow and not a white hairline.
BOX_EDGE = 0.3
# How much of the Fresnel rise at grazing angles is kept. All of it put a white
# line round every silhouette and along every bevel; physically right, but it
# reads as an outline drawn on.
GRAZING = 0.2
DOME = tuple(_unit([-0.3, -0.5, -0.8]).tolist())


def _tonemap(x):
    """Narkowicz's fit to ACES, then gamma: highlights roll off instead of clip."""
    x = x * 0.8
    y = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    return y.clamp(0, 1) ** (1 / 2.2)


def _environment(r):
    """The studio the shapes reflect: a dim gradient and soft-edged softboxes."""
    # Mostly dark, like a studio with the lights off between the softboxes:
    # polished metal is the contrast between what it reflects, and a bright
    # even surround turns gold into satin brass.
    sky = ((-r[..., 1]) * 0.5 + 0.5).clamp(0, 1)
    env = 0.015 + 0.2 * sky ** 4
    # And one big soft source behind the camera, so a face turned towards you
    # shades in a gradient instead of going black between the softboxes.
    dome = torch.tensor(DOME, dtype=r.dtype, device=r.device)
    env = env + 0.9 * (r * dome).sum(-1).clamp(0, 1) ** 4
    for row in BOXES:
        c, t1, t2 = (torch.tensor(row[i:i + 3], dtype=r.dtype, device=r.device)
                     for i in (0, 3, 6))
        a = (r * t1).sum(-1).abs()
        b = (r * t2).sum(-1).abs()
        box = (0.5 + (row[9] - a) / BOX_EDGE).clamp(0, 1) * \
            (0.5 + (row[10] - b) / BOX_EDGE).clamp(0, 1)
        env = env + row[11] * box * ((r * c).sum(-1) > 0)
    return env


def _patches(grid, x0, y0, focal, width, height, c, rot, size,
             normals, offsets, rounding, bevel, kind, centre, rows,
             colours, metal, axes):
    """Trace and shade a batch of shapes, each in an S x S patch at (x0, y0).

    `normals` and `offsets` are the pieces' planes already in world space;
    `rounding` the offsets the bevel sees, the same but for cuts between
    pieces, which are out of its reach; `bevel` how far from an edge each
    piece shades round, in world units.

    A piece whose `kind` is 1 is also inside an ellipsoid, {x : |A(x - centre)| <= 1}
    with A's rows in `rows`, so its planes only clip it: a whole ellipsoid has
    none that reach, an egg's half has one. Where the ellipsoid is the surface
    the ray enters through, the normal is the ellipsoid's own; where a plane
    is, the bevelled one.

    `colours` (K, P, 2, 3) and `metal` (K, P) are per piece, so a present and
    its ribbon are one shape in two materials.
    Returns colour (K, 3, S, S) and coverage (K, S, S).
    """
    k, s = c.shape[0], grid.shape[0]
    dx = ((x0.view(k, 1, 1) + grid.view(1, 1, s)) - width / 2) / focal
    dy = ((y0.view(k, 1, 1) + grid.view(1, s, 1)) - height / 2) / focal
    d = torch.stack([dx.expand(k, s, s), dy.expand(k, s, s),
                     torch.ones_like(dx).expand(k, s, s)], -1)
    d = d / d.norm(dim=-1, keepdim=True)                                   # K,S,S,3

    best = torch.full((k, s, s), 1e9, device=d.device)
    cover = torch.zeros_like(best)
    n = torch.zeros_like(d)
    lo = torch.zeros_like(d)
    hi = torch.zeros_like(d)
    shine = torch.zeros(k, s, s, 1, device=d.device)
    for p in range(normals.shape[1]):
        nrm, off = normals[:, p], offsets[:, p]
        denom = torch.einsum("kijc,kfc->kijf", d, nrm)
        denom = torch.where(denom.abs() < 1e-7, torch.full_like(denom, 1e-7), denom)
        t = off[:, None, None] / denom
        enter = torch.where(denom < 0, t, torch.full_like(t, -1e9)).amax(-1)
        leave = torch.where(denom > 0, t, torch.full_like(t, 1e9)).amin(-1)

        # The ellipsoid: |A(t d - centre)|^2 = 1 is a quadratic in t.
        ell = (kind[:, p] > 0.5).view(k, 1, 1)
        a = rows[:, p]                                                     # K,3,3
        qd = torch.einsum("kijc,krc->kijr", d, a)
        q0 = -torch.einsum("krc,kc->kr", a, centre[:, p])[:, None, None]
        # Not by the textbook b^2 - 4ac: at ten units away both terms are in
        # the thousands and their difference, the silhouette, drowns in float32
        # rounding. From the ray's closest approach instead, which is small.
        aa = (qd * qd).sum(-1).clamp(min=1e-12)
        mid = -(qd * q0).sum(-1) / aa
        miss = ((q0 + mid[..., None] * qd) ** 2).sum(-1)
        disc = 1 - miss
        root = (disc.clamp(min=0) / aa).sqrt()
        e_in = torch.where(disc > 0, mid - root, torch.ones_like(aa))
        e_out = torch.where(disc > 0, mid + root, torch.zeros_like(aa))
        curved = ell & (e_in >= enter)
        enter = torch.where(ell, torch.maximum(enter, e_in), enter)
        leave = torch.where(ell, torch.minimum(leave, e_out), leave)

        # Coverage from the chord: nothing at the silhouette, whole a pixel
        # inside. Antialiasing that needs no extra rays.
        here = ((leave - enter) * focal / (enter.clamp(min=1e-3) * 1.2)).clamp(0, 1)
        here = here * (enter > 0)
        nearer = (here > 0) & (enter < best)
        best = torch.where(nearer, enter, best)
        cover = torch.maximum(cover, here)

        # The bevelled normal of this piece, kept where it is the nearest.
        hit = d * enter.clamp(0, 1e3)[..., None]
        dist = (torch.einsum("kijc,kfc->kijf", hit, nrm) - rounding[:, p, None, None]) \
            / bevel[:, p].view(k, 1, 1, 1)
        m = torch.einsum("kijf,kfc->kijc", torch.softmax(dist, -1), nrm)
        rel = torch.einsum("krc,kijc->kijr", a, hit - centre[:, p, None, None])
        grad = torch.einsum("kijr,krc->kijc", rel, a)
        grad = grad / grad.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        m = torch.where(curved[..., None], grad, m)
        m = m / m.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        n = torch.where(nearer[..., None], m, n)
        lo = torch.where(nearer[..., None], colours[:, p, 0].view(k, 1, 1, 3), lo)
        hi = torch.where(nearer[..., None], colours[:, p, 1].view(k, 1, 1, 3), hi)
        shine = torch.where(nearer[..., None], metal[:, p].view(k, 1, 1, 1), shine)
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    hit = d * best.clamp(0, 1e3)[..., None]

    # Colour: a gradient along an axis fixed to the shape, so it turns with it.
    local = torch.einsum("kba,kijb->kija", rot, hit - c[:, None, None]) \
        / size.view(k, 1, 1, 1)
    g = ((local * axes[:, None, None]).sum(-1) * 0.6 + 0.5).clamp(0, 1)
    g = g * g * (3 - 2 * g)
    base = lo + (hi - lo) * g[..., None]
    metal = shine

    view = -d
    ndv = (n * view).sum(-1, keepdim=True).clamp(0, 1)
    key = torch.tensor(KEY, dtype=d.dtype, device=d.device)
    ndl = (n * key).sum(-1, keepdim=True)
    diffuse = 0.65 * ndl.clamp(min=0) + 0.35 * (ndl * 0.5 + 0.5) ** 2
    ambient = 0.15 + 0.2 * (-n[..., 1:2] * 0.5 + 0.5)
    colour = base * (ambient + diffuse * 1.05) * (1 - metal)

    # Reflection, by Schlick, with the grazing rise held down (see GRAZING).
    # Whatever is not metal has a clear coat on top that reflects white.
    r = d - 2 * (d * n).sum(-1, keepdim=True) * n
    env = _environment(r)[..., None]
    grazing = (1 - ndv) ** 5
    f0 = 0.04 + (base - 0.04) * metal
    colour = colour + env * (f0 + (1 - f0) * grazing * GRAZING)
    colour = colour + env * (0.04 + 0.96 * grazing * GRAZING) * 0.5 * (1 - metal)

    half = view + key
    half = half / half.norm(dim=-1, keepdim=True)
    spec = (n * half).sum(-1, keepdim=True).clamp(0, 1) ** 140 * 1.2
    colour = colour + spec * (1 + (base - 1) * metal)
    return _tonemap(colour).permute(0, 3, 1, 2), cover


# Per-shape record the kernel reads: x0 y0, centre, rotation (row major),
# size, gradient axis.
RECORD = 18
# After a piece's F planes: bevel, kind, ellipsoid centre and rows, low and
# high colour, metal.
PIECE_EXTRA = 21

try:
    import triton
    import triton.language as tl
except ImportError:            # CPU-only installs: the torch path still works
    triton = None

if triton is not None:
    @triton.jit
    def _pow(x, p):
        return tl.exp2(p * tl.log2(tl.maximum(x, 1e-12)))

    @triton.jit
    def _clamp01(x):
        return tl.minimum(tl.maximum(x, 0.0), 1.0)

    @triton.jit
    def _aces(x):
        x = x * 0.8
        y = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
        return _pow(_clamp01(y), 1.0 / 2.2)

    @triton.jit
    def _patch_kernel(colour_ptr, cover_ptr, shape_ptr, plane_ptr, box_ptr,
                      S, P, focal, half_w, half_h, kx, ky, kz, dmx, dmy, dmz, box_edge, grazing_gain,
                      F: tl.constexpr, NB: tl.constexpr, REC: tl.constexpr,
                      PREC: tl.constexpr, BLOCK: tl.constexpr):
        k = tl.program_id(0)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < S * S
        i = (offs // S).to(tl.float32)
        j = (offs % S).to(tl.float32)

        rec = shape_ptr + k * REC
        x0, y0 = tl.load(rec + 0), tl.load(rec + 1)
        cx, cy, cz = tl.load(rec + 2), tl.load(rec + 3), tl.load(rec + 4)
        r00, r01, r02 = tl.load(rec + 5), tl.load(rec + 6), tl.load(rec + 7)
        r10, r11, r12 = tl.load(rec + 8), tl.load(rec + 9), tl.load(rec + 10)
        r20, r21, r22 = tl.load(rec + 11), tl.load(rec + 12), tl.load(rec + 13)
        size = tl.load(rec + 14)
        gx, gy, gz = tl.load(rec + 15), tl.load(rec + 16), tl.load(rec + 17)

        dx = (x0 + j + 0.5 - half_w) / focal
        dy = (y0 + i + 0.5 - half_h) / focal
        inv = 1.0 / tl.sqrt(dx * dx + dy * dy + 1.0)
        dx, dy, dz = dx * inv, dy * inv, inv

        best = tl.full([BLOCK], 1e9, tl.float32)
        cover = tl.zeros([BLOCK], tl.float32)
        nx = tl.zeros([BLOCK], tl.float32)
        ny = tl.zeros([BLOCK], tl.float32)
        nz = tl.zeros([BLOCK], tl.float32)
        lr = tl.zeros([BLOCK], tl.float32)
        lg = tl.zeros([BLOCK], tl.float32)
        lb = tl.zeros([BLOCK], tl.float32)
        hr = tl.zeros([BLOCK], tl.float32)
        hg = tl.zeros([BLOCK], tl.float32)
        hb = tl.zeros([BLOCK], tl.float32)
        metal = tl.zeros([BLOCK], tl.float32)
        for p in range(0, P):
            base_ptr = plane_ptr + (k * P + p) * PREC
            extra = base_ptr + F * 5
            enter = tl.full([BLOCK], -1e9, tl.float32)
            leave = tl.full([BLOCK], 1e9, tl.float32)
            for f in tl.static_range(F):
                a = tl.load(base_ptr + f * 5)
                b = tl.load(base_ptr + f * 5 + 1)
                c = tl.load(base_ptr + f * 5 + 2)
                o = tl.load(base_ptr + f * 5 + 3)
                den = dx * a + dy * b + dz * c
                den = tl.where(tl.abs(den) < 1e-7, 1e-7, den)
                t = o / den
                enter = tl.where(den < 0, tl.maximum(enter, t), enter)
                leave = tl.where(den > 0, tl.minimum(leave, t), leave)

            # The ellipsoid, for a piece of kind 1: a quadratic in t.
            ell = tl.load(extra + 1) > 0.5
            ecx, ecy, ecz = tl.load(extra + 2), tl.load(extra + 3), tl.load(extra + 4)
            a0x, a0y, a0z = tl.load(extra + 5), tl.load(extra + 6), tl.load(extra + 7)
            a1x, a1y, a1z = tl.load(extra + 8), tl.load(extra + 9), tl.load(extra + 10)
            a2x, a2y, a2z = tl.load(extra + 11), tl.load(extra + 12), tl.load(extra + 13)
            q0x = -(a0x * ecx + a0y * ecy + a0z * ecz)
            q0y = -(a1x * ecx + a1y * ecy + a1z * ecz)
            q0z = -(a2x * ecx + a2y * ecy + a2z * ecz)
            qdx = a0x * dx + a0y * dy + a0z * dz
            qdy = a1x * dx + a1y * dy + a1z * dz
            qdz = a2x * dx + a2y * dy + a2z * dz
            aa = tl.maximum(qdx * qdx + qdy * qdy + qdz * qdz, 1e-12)
            mid = -(qdx * q0x + qdy * q0y + qdz * q0z) / aa
            mx_, my_, mz_ = q0x + mid * qdx, q0y + mid * qdy, q0z + mid * qdz
            disc = 1.0 - (mx_ * mx_ + my_ * my_ + mz_ * mz_)
            root = tl.sqrt(tl.maximum(disc, 0.0) / aa)
            e_in = tl.where(disc > 0, mid - root, 1.0)
            e_out = tl.where(disc > 0, mid + root, 0.0)
            curved = ell & (e_in >= enter)
            enter = tl.where(ell, tl.maximum(enter, e_in), enter)
            leave = tl.where(ell, tl.minimum(leave, e_out), leave)

            here = _clamp01((leave - enter) * focal / (tl.maximum(enter, 1e-3) * 1.2))
            here = tl.where(enter > 0, here, 0.0)
            nearer = (here > 0) & (enter < best)
            best = tl.where(nearer, enter, best)
            cover = tl.maximum(cover, here)

            # Bevelled normal: an online softmax over the planes' distances.
            bevel = tl.load(extra)
            te = tl.minimum(tl.maximum(enter, 0.0), 1e3)
            hx, hy, hz = dx * te, dy * te, dz * te
            top = tl.full([BLOCK], -1e30, tl.float32)
            total = tl.zeros([BLOCK], tl.float32)
            mx = tl.zeros([BLOCK], tl.float32)
            my = tl.zeros([BLOCK], tl.float32)
            mz = tl.zeros([BLOCK], tl.float32)
            for f in tl.static_range(F):
                a = tl.load(base_ptr + f * 5)
                b = tl.load(base_ptr + f * 5 + 1)
                c = tl.load(base_ptr + f * 5 + 2)
                o = tl.load(base_ptr + f * 5 + 4)
                dist = (hx * a + hy * b + hz * c - o) / bevel
                new = tl.maximum(top, dist)
                keep = tl.exp(top - new)
                w = tl.exp(dist - new)
                total = total * keep + w
                mx = mx * keep + w * a
                my = my * keep + w * b
                mz = mz * keep + w * c
                top = new
            mx, my, mz = mx / total, my / total, mz / total
            # The ellipsoid's normal: A^T A (x - centre).
            ux, uy, uz = hx - ecx, hy - ecy, hz - ecz
            r0 = a0x * ux + a0y * uy + a0z * uz
            r1 = a1x * ux + a1y * uy + a1z * uz
            r2 = a2x * ux + a2y * uy + a2z * uz
            ex = r0 * a0x + r1 * a1x + r2 * a2x
            ey = r0 * a0y + r1 * a1y + r2 * a2y
            ez = r0 * a0z + r1 * a1z + r2 * a2z
            inv = 1.0 / tl.maximum(tl.sqrt(ex * ex + ey * ey + ez * ez), 1e-9)
            mx = tl.where(curved, ex * inv, mx)
            my = tl.where(curved, ey * inv, my)
            mz = tl.where(curved, ez * inv, mz)
            inv = 1.0 / tl.maximum(tl.sqrt(mx * mx + my * my + mz * mz), 1e-6)
            nx = tl.where(nearer, mx * inv, nx)
            ny = tl.where(nearer, my * inv, ny)
            nz = tl.where(nearer, mz * inv, nz)
            lr = tl.where(nearer, tl.load(extra + 14), lr)
            lg = tl.where(nearer, tl.load(extra + 15), lg)
            lb = tl.where(nearer, tl.load(extra + 16), lb)
            hr = tl.where(nearer, tl.load(extra + 17), hr)
            hg = tl.where(nearer, tl.load(extra + 18), hg)
            hb = tl.where(nearer, tl.load(extra + 19), hb)
            metal = tl.where(nearer, tl.load(extra + 20), metal)

        inv = 1.0 / tl.maximum(tl.sqrt(nx * nx + ny * ny + nz * nz), 1e-6)
        nx, ny, nz = nx * inv, ny * inv, nz * inv
        te = tl.minimum(tl.maximum(best, 0.0), 1e3)
        rx, ry, rz = dx * te - cx, dy * te - cy, dz * te - cz

        # Gradient along an axis fixed to the shape: the hit in its own frame.
        lx = (r00 * rx + r10 * ry + r20 * rz) / size
        ly = (r01 * rx + r11 * ry + r21 * rz) / size
        lz = (r02 * rx + r12 * ry + r22 * rz) / size
        g = _clamp01((lx * gx + ly * gy + lz * gz) * 0.6 + 0.5)
        g = g * g * (3.0 - 2.0 * g)

        ndv = _clamp01(-(nx * dx + ny * dy + nz * dz))
        ndl = nx * kx + ny * ky + nz * kz
        hl = (ndl * 0.5 + 0.5)
        diffuse = 0.65 * tl.maximum(ndl, 0.0) + 0.35 * hl * hl
        light = (0.15 + 0.2 * (-ny * 0.5 + 0.5) + diffuse * 1.05) * (1.0 - metal)

        dn = dx * nx + dy * ny + dz * nz
        qx, qy, qz = dx - 2 * dn * nx, dy - 2 * dn * ny, dz - 2 * dn * nz
        sky = _clamp01(-qy * 0.5 + 0.5)
        env = 0.015 + 0.2 * _pow(sky, 4.0)
        env += 0.9 * _pow(_clamp01(qx * dmx + qy * dmy + qz * dmz), 4.0)
        for bx in tl.static_range(NB):
            row = box_ptr + bx * 12
            facing = qx * tl.load(row) + qy * tl.load(row + 1) + qz * tl.load(row + 2)
            u = tl.abs(qx * tl.load(row + 3) + qy * tl.load(row + 4) + qz * tl.load(row + 5))
            v = tl.abs(qx * tl.load(row + 6) + qy * tl.load(row + 7) + qz * tl.load(row + 8))
            box = _clamp01(0.5 + (tl.load(row + 9) - u) / box_edge) * \
                _clamp01(0.5 + (tl.load(row + 10) - v) / box_edge)
            env += tl.where(facing > 0, tl.load(row + 11) * box, 0.0)

        grazing = _pow(1.0 - ndv, 5.0)
        hx, hy, hz = kx - dx, ky - dy, kz - dz
        inv = 1.0 / tl.sqrt(hx * hx + hy * hy + hz * hz)
        spec = _pow(_clamp01((nx * hx + ny * hy + nz * hz) * inv), 140.0) * 1.2
        coat = env * (0.04 + 0.96 * grazing * grazing_gain) * 0.5 * (1.0 - metal)

        out = colour_ptr + k * 3 * S * S + offs
        for ch in tl.static_range(3):
            if ch == 0:
                lo, hi = lr, hr
            elif ch == 1:
                lo, hi = lg, hg
            else:
                lo, hi = lb, hb
            base = lo + (hi - lo) * g
            f0 = 0.04 + (base - 0.04) * metal
            col = base * light + env * (f0 + (1.0 - f0) * grazing * grazing_gain)
            col += coat + spec * (1.0 + (base - 1.0) * metal)
            tl.store(out + ch * S * S, _aces(col), mask=mask)
        tl.store(cover_ptr + k * S * S + offs, cover, mask=mask)


def _patches_triton(grid, x0, y0, focal, width, height, c, rot, size,
                    normals, offsets, rounding, bevel, kind, centre, rows,
                    colours, metal, axes):
    """`_patches` as one kernel. Same arguments, same results."""
    k, s = c.shape[0], grid.shape[0]
    pieces_ = normals.shape[1]
    record = torch.cat([x0[:, None], y0[:, None], c, rot.reshape(k, 9), size[:, None],
                        axes], 1).contiguous()
    # Per plane: normal, offset, rounding offset; per piece, then, the rest.
    planes = torch.cat([torch.cat([normals, offsets[..., None], rounding[..., None]], -1)
                        .reshape(k, pieces_, -1),
                        bevel[..., None], kind[..., None], centre,
                        rows.reshape(k, pieces_, 9), colours.reshape(k, pieces_, 6),
                        metal[..., None]], -1).contiguous()
    assert planes.shape[-1] == normals.shape[2] * 5 + PIECE_EXTRA
    colour = torch.empty(k, 3, s, s, device=c.device)
    cover = torch.empty(k, s, s, device=c.device)
    block = 256
    _patch_kernel[(k, triton.cdiv(s * s, block))](
        colour, cover, record, planes, _box_table(c.device),
        s, pieces_, focal, width / 2, height / 2, *KEY, *DOME, BOX_EDGE, GRAZING,
        F=normals.shape[2], NB=len(BOXES), REC=RECORD,
        PREC=normals.shape[2] * 5 + PIECE_EXTRA, BLOCK=block)
    return colour, cover


_BOX_TABLES = {}


def _box_table(device):
    if device not in _BOX_TABLES:
        _BOX_TABLES[device] = torch.tensor(BOXES, dtype=torch.float32, device=device)
    return _BOX_TABLES[device]


class Snow:
    """Flakes falling through the scene, in two layers either side of the shapes.

    Each flake is a point in the same space as the shapes. Drawing hundreds of
    soft discs one by one would be hundreds of small kernels; instead every
    flake is splatted, bilinearly so it moves smoothly, into a small image,
    which is blurred into discs and scaled up. The far layer is splatted at an
    eighth of the frame and the near one at a quarter, so near flakes come out
    bigger and softer, as they would through a lens focused on the shapes.
    """

    COUNT = 420
    FALL = 0.55              # world units a second; a calm evening, not a blizzard
    SPLIT = (Z_NEAR + Z_FAR) / 2

    def __init__(self, device, seed=7):
        self.device = device
        g = torch.Generator(device="cpu").manual_seed(seed)
        r = lambda *shape: torch.rand(*shape, generator=g)
        n = self.COUNT
        tan_y = math.tan(FOV / 2)
        z = 4.0 + r(n) * (Z_WALL - 4.5)
        self.pos = torch.stack([(r(n) * 2 - 1) * z * tan_y * 2.0,
                                (r(n) * 2 - 1) * z * tan_y * 1.1, z], 1).to(device)
        self.speed = (0.6 + r(n) * 0.8).to(device)
        self.phase = (r(n) * 2 * math.pi).to(device)
        self.bright = (0.55 + r(n) * 0.45).to(device)
        self.t = 0.0

    def step(self, dt, tan_x, tan_y):
        self.t += dt
        p = self.pos
        z = p[:, 2]
        p[:, 1] += self.FALL * self.speed * dt
        p[:, 0] += 0.18 * torch.sin(0.7 * self.t * self.speed + self.phase) * dt
        # Out at the bottom, in again at the top; out at a side, in at the other.
        bottom = z * tan_y * 1.1
        p[:, 1] = torch.where(p[:, 1] > bottom, p[:, 1] - 2 * bottom, p[:, 1])
        side = z * tan_x * 1.05
        p[:, 0] = torch.where(p[:, 0] > side, p[:, 0] - 2 * side, p[:, 0])
        p[:, 0] = torch.where(p[:, 0] < -side, p[:, 0] + 2 * side, p[:, 0])

    def layer(self, out, focal, near):
        """Lighten `out` (3, H, W) with the near or the far flakes."""
        height, width = out.shape[-2:]
        p = self.pos
        # Masked by weight rather than picked out: indexing by a mask waits
        # for the GPU to say how many there are, every frame.
        pick = (p[:, 2] < self.SPLIT) if near else (p[:, 2] >= self.SPLIT)
        bright = self.bright * pick
        shrink = 4 if near else 8
        h, w = max(height // shrink, 2), max(width // shrink, 2)
        u = (p[:, 0] / p[:, 2] * focal + width / 2) / shrink - 0.5
        v = (p[:, 1] / p[:, 2] * focal + height / 2) / shrink - 0.5
        # Nearer is bigger and brighter; the far ones fade into the wall.
        weight = bright * (9.0 / p[:, 2]).clamp(max=1.4) * (1.5 if near else 0.9)
        x0, y0 = u.floor(), v.floor()
        fx, fy = u - x0, v - y0
        img = torch.zeros(h * w, device=out.device)
        for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                           (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            x, y = (x0 + dx).long(), (y0 + dy).long()
            ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            img.index_add_(0, (y.clamp(0, h - 1) * w + x.clamp(0, w - 1)), weight * wt * ok)
        img = img.view(1, 1, h, w)
        k = torch.exp(-0.5 * (torch.arange(-2, 3, device=out.device) / 0.8) ** 2)
        k = (k / k.max()).view(1, 1, 1, -1)
        img = F.conv2d(F.pad(img, (2, 2, 0, 0)), k)
        img = F.conv2d(F.pad(img, (0, 0, 2, 2)), k.transpose(2, 3))
        img = F.interpolate(img.clamp_(0, 1), size=(height, width), mode="bilinear",
                            align_corners=False)[0]
        white = torch.tensor([0.97, 0.98, 1.0], device=out.device).view(3, 1, 1)
        return torch.lerp(out, white.expand_as(out), img)


class ShapesBackground:
    """A background source: `sample(height, width)` hands back one frame."""

    def __init__(self, spec, device, dtype, count=7, seed=7):
        self.path = spec
        self.device, self.dtype = device, dtype
        self.palette = PALETTES[palette_of(spec)]
        self._materials = MATERIALS[self.palette["materials"]]
        season = self.palette.get("season")
        if season:
            import seasons
            self.swarm = Swarm(seed=seed, cast=seasons.cast(season, seed=seed),
                               sizes=self.palette.get("sizes", (0.5, 0.85)))
        else:
            self.swarm = Swarm(count=count, seed=seed)
        self.snow = Snow(device, seed=seed) if self.palette.get("snow") else None
        self._build()
        self._clock = None
        self._dither = None
        self._trace = _patches_triton \
            if triton is not None and torch.device(device).type == "cuda" else _patches

    def _build(self):
        """Upload the pieces and materials of whatever the swarm holds."""
        device, sw, count = self.device, self.swarm, len(self.swarm.pos)
        materials = self._materials
        mats = [materials[m % len(materials)] for m in sw.material]
        # Shapes are batched by how many pieces they have: a solid cube is one
        # and a reindeer twenty, and padding every cube out to twenty would
        # make it pay for nineteen pieces that are not there.
        shaped = {i: [_pad(p) for p in pieces(sw.kind[i], sw.hollow[i])] for i in range(count)}
        buckets = {}
        for i in range(count):
            buckets.setdefault(1 << (len(shaped[i]) - 1).bit_length(), []).append(i)
        self._batches = []
        no_rows = np.zeros((3, 3))
        for _, idx in sorted(buckets.items()):
            parts = [shaped[i] for i in idx]
            most = max(len(p) for p in parts)
            parts = [p + [p[0]] * (most - len(p)) for p in parts]  # a repeat hits nothing new
            t = lambda a: torch.tensor(np.array(a), dtype=torch.float32, device=device)

            def material(q, i):
                lo, hi, metal = q[4] or mats[i]
                return (np.array([_hex(lo), _hex(hi)]) ** 2.2, metal)
            looks = [[material(q, i) for q in p] for p, i in zip(parts, idx)]
            self._batches.append(dict(
                index=idx,
                normals=t([[q[0] for q in p] for p in parts]),     # K,P,F,3
                offsets=t([[q[1] for q in p] for p in parts]),     # K,P,F
                bevel=t([[q[2] for q in p] for p in parts]),       # K,P
                rounding=t([[q[3] for q in p] for p in parts]),    # K,P,F
                kind=t([[float(q[5] is not None) for q in p] for p in parts]),     # K,P
                centre=t([[q[5][0] if q[5] else np.zeros(3) for q in p] for p in parts]),
                rows=t([[q[5][1] if q[5] else no_rows for q in p] for p in parts]),
                colours=t([[c for c, _ in look] for look in looks]),             # K,P,2,3
                metal=t([[m for _, m in look] for look in looks]),               # K,P
            ))

        # What the shadow on the wall is tinted with: the shape's main material.
        lin = np.array([[_hex(a), _hex(b)] for a, b, _ in
                        [self._main_material(i, shaped[i], mats[i]) for i in range(count)]]) ** 2.2
        self._colours = torch.tensor(lin, dtype=torch.float32, device=device)       # K,2,3
        axes = _unit(np.random.default_rng(count).normal(size=(count, 3)) + [0, 1.5, 0])
        self._axes = torch.tensor(axes, dtype=torch.float32, device=device)

    @staticmethod
    def _main_material(i, parts, own):
        """The material of the shape's first piece, its body as a rule."""
        return parts[0][4] or own

    # -- the render loop's side

    def observe(self, alpha):
        """Hand over this frame's matte, (1, 1, H, W) on the GPU."""
        a = F.adaptive_avg_pool2d(alpha.float(), (FIELD_H, FIELD_W))
        # Blurred by a few cells: the field has to reach out past your outline
        # far enough that a shape turns before its edge touches you.
        k = torch.exp(-0.5 * (torch.arange(-8, 9, device=a.device) / 3.0) ** 2)
        k = (k / k.sum()).view(1, 1, 1, -1)
        a = F.conv2d(F.pad(a, (8, 8, 0, 0), mode="replicate"), k)
        a = F.conv2d(F.pad(a, (0, 0, 8, 8), mode="replicate"), k.transpose(2, 3))
        self.swarm.observe(a[0, 0].cpu().numpy().astype(np.float64))

    def sample(self, height, width):
        now = time.monotonic()
        dt = 0.0 if self._clock is None else now - self._clock
        self._clock = now
        self.swarm.step(dt)
        if self.snow is not None:
            self.snow.step(min(dt, 0.05), self.swarm.tan_x, self.swarm.tan_y)
        return self.render(height, width).to(self.dtype)

    def close(self):
        pass

    # -- drawing

    def _backdrop(self, height, width, u, v, radius, z, tints, weight):
        """Gradient wall with drifting glows and soft tinted shadows, quarter res."""
        dev = self.device
        h, w = max(height // 4, 1), max(width // 4, 1)
        y = torch.linspace(0, 1, h, device=dev)[:, None]
        x = torch.linspace(0, 1, w, device=dev)[None]
        aspect = width / height
        r = torch.sqrt(((x - 0.5) * aspect) ** 2 + (y - 0.45) ** 2) / (0.5 * aspect)
        inner = torch.tensor(_hex(self.palette["inner"]), device=dev).view(3, 1, 1)
        outer = torch.tensor(_hex(self.palette["outer"]), device=dev).view(3, 1, 1)
        wall = inner + (outer - inner) * r.clamp(0, 1.2)[None] ** 1.4

        t = self.swarm.t
        for i, glow in enumerate(self.palette["glows"]):
            gx = 0.5 + 0.32 * math.sin(0.05 * t + i * 2.4)
            gy = 0.5 + 0.22 * math.cos(0.037 * t + i * 1.3)
            blob = torch.exp(-(((x - gx) * aspect) ** 2 + (y - gy) ** 2) / 0.18)
            col = torch.tensor(_hex(glow), device=dev).view(3, 1, 1)
            wall = wall + (col - wall) * (blob * self.palette["glow"])[None]

        # Light from the upper left, so each shadow falls down and to the right,
        # further and softer the further the shape floats in front of the wall.
        gap = Z_WALL - z
        rad = radius / height
        sx = (u / width + rad * gap * 0.09 / aspect).view(-1, 1, 1)
        sy = (v / height + rad * gap * 0.12).view(-1, 1, 1)
        sigma = (rad * (0.75 + 0.06 * gap)).view(-1, 1, 1)
        d2 = ((x[None] - sx) * aspect) ** 2 + (y[None] - sy) ** 2
        strength = torch.exp(-d2 / (2 * sigma ** 2)) * 0.42 * weight.view(-1, 1, 1)
        # Tinted: a coloured shape throws a shadow with some of its colour in it.
        keep = 1 - strength[:, None] * (1 - (tints * 0.55).view(-1, 3, 1, 1))
        wall = wall * keep.prod(0)

        wall = F.interpolate(wall[None], size=(height, width), mode="bilinear",
                             align_corners=False)[0]
        # Gradients this smooth band in eight bits; grain below one step hides it.
        if self._dither is None or self._dither.shape[-2:] != (height, width):
            g = torch.Generator(device="cpu").manual_seed(1)
            noise = torch.rand(1, height, width, generator=g) - \
                torch.rand(1, height, width, generator=g)
            self._dither = (noise / 255.0).to(dev)
        return wall + self._dither

    def _shade(self, batch, height, width, focal, c, rot, size):
        """Render one batch of shapes into patches: colour, coverage, corners."""
        idx = batch["index"]
        c, rot, size = c[idx], rot[idx], size[idx]
        u = c[:, 0] / c[:, 2] * focal + width / 2
        v = c[:, 1] / c[:, 2] * focal + height / 2
        radius = focal * size / (c[:, 2] - size).clamp(min=0.5)
        # One patch size per batch, rounded up so the shape of the batch, and
        # with it the allocator, changes rarely.
        s = int(math.ceil((float(radius.max()) * 2 + 6) / 32) * 32)
        x0, y0 = (u - s / 2).floor(), (v - s / 2).floor()
        # The pieces' planes in world space: rotate the normals, move the offsets.
        normals = torch.einsum("kab,kpfb->kpfa", rot, batch["normals"])     # K,P,F,3
        shift = (normals * c[:, None, None]).sum(-1)
        offsets = shift + batch["offsets"] * size[:, None, None]
        rounding = shift + batch["rounding"] * size[:, None, None]
        # An ellipsoid in the shape's frame, |a.(y - p)| <= 1 for y = R^T (x - c) / size,
        # is in the world |(R a / size).(x - c - size R p)| <= 1.
        centre = c[:, None] + size[:, None, None] * \
            torch.einsum("kab,kpb->kpa", rot, batch["centre"])
        rows = torch.einsum("kab,kprb->kpra", rot, batch["rows"]) / size.view(-1, 1, 1, 1)
        colour, cover = self._trace(
            torch.arange(s, device=self.device, dtype=torch.float32) + 0.5,
            x0, y0, float(focal), float(width), float(height), c, rot, size,
            normals, offsets, rounding, batch["bevel"] * size[:, None],
            batch["kind"], centre, rows, batch["colours"], batch["metal"],
            self._axes[idx])
        return colour, cover, x0, y0, s, u, v, radius

    def render(self, height, width):
        dev = self.device
        sw = self.swarm
        focal = (height / 2) / sw.tan_y
        sw.tan_x = sw.tan_y * width / height

        c = torch.tensor(sw.pos, dtype=torch.float32, device=dev)          # K,3
        rot = torch.tensor(sw.rot, dtype=torch.float32, device=dev)        # K,3,3
        size = torch.tensor(sw.size, dtype=torch.float32, device=dev)

        drawn = [(b["index"], *self._shade(b, height, width, focal, c, rot, size))
                 for b in self._batches]

        idx = [i for b in drawn for i in b[0]]
        u = torch.cat([b[6] for b in drawn])
        v = torch.cat([b[7] for b in drawn])
        radius = torch.cat([b[8] for b in drawn])
        tints = (self._colours[idx].mean(1)) ** (1 / 2.2)
        weight = torch.tensor([0.55 if sw.hollow[i] else 1.0 for i in idx], device=dev)
        out = self._backdrop(height, width, u, v, radius, c[idx, 2], tints, weight)
        if self.snow is not None:
            out = self.snow.layer(out, focal, near=False)

        # Far to near, each patch over what is already there. The physics keeps
        # shapes from passing through each other, so painter's order is enough.
        jobs = []
        for index, colour, cover, x0, y0, s, *_ in drawn:
            xs, ys = x0.long().tolist(), y0.long().tolist()
            for j, i in enumerate(index):
                jobs.append((sw.pos[i, 2], colour[j], cover[j], xs[j], ys[j], s))
        for _, colour, cover, x, y, s in sorted(jobs, key=lambda job: -job[0]):
            ax, ay = max(x, 0), max(y, 0)
            bx, by = min(x + s, width), min(y + s, height)
            if bx <= ax or by <= ay:
                continue
            a = cover[ay - y:by - y, ax - x:bx - x]
            col = colour[:, ay - y:by - y, ax - x:bx - x]
            region = out[:, ay:by, ax:bx]
            out[:, ay:by, ax:bx] = region + (col - region) * a
        if self.snow is not None:
            out = self.snow.layer(out, focal, near=True)
        return out.clamp(0, 1)[None]
