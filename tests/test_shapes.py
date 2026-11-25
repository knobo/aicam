"""The rendered 3D background: geometry, the push away from you, the kernel."""

import numpy as np
import pytest
import torch

import control
import gui
import history
import shapes3d


def one_shape(kind, hollow, spec="shapes", device="cpu"):
    """A background holding a single shape, face on, in the middle of the view."""
    bg = shapes3d.ShapesBackground(spec, device, torch.float32, count=1)
    sw = bg.swarm
    sw.kind, sw.hollow = [kind], [hollow]
    sw.pos[:] = [[0.0, 0.0, 10.0]]
    sw.rot[:] = np.eye(3)
    sw.size[:] = [1.0]
    bg._build()
    return bg


def test_palettes():
    assert shapes3d.palette_of("shapes") == shapes3d.DEFAULT
    assert shapes3d.palette_of("shapes:dusk") == "dusk"
    with pytest.raises(ValueError):
        shapes3d.palette_of("shapes:nope")
    assert shapes3d.is_shapes_spec("shapes:studio")
    assert not shapes3d.is_shapes_spec("shapes.png")


def test_the_panel_lists_every_palette():
    assert {shapes3d.palette_of(spec) for spec, *_ in gui.SHAPES} == set(shapes3d.PALETTES)


def test_specs_survive_the_command_line_and_the_history():
    assert control.normalise_source("shapes:dusk") == "shapes:dusk"
    assert history.kind_of("shapes") == "3d"


@pytest.mark.parametrize("kind,edges", [("cube", 12), ("tetrahedron", 6),
                                        ("octahedron", 12), ("prism", 9)])
def test_a_hollow_shape_is_one_piece_per_edge(kind, edges):
    parts = shapes3d.pieces(kind, hollow=True)
    assert len(parts) == edges
    assert all(len(piece[1]) <= shapes3d.FACES for piece in parts)


@pytest.mark.parametrize("kind", list(shapes3d.TILES) + list(shapes3d.SHAPES))
@pytest.mark.parametrize("hollow", [False, True])
def test_every_piece_fits_the_kernel(kind, hollow):
    assert all(len(piece[1]) <= shapes3d.FACES for piece in shapes3d.pieces(kind, hollow))


def coverage(bg, height=90, width=160):
    """How much of each pixel the shape covers, across the whole frame."""
    sw = bg.swarm
    focal = (height / 2) / sw.tan_y
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    _, cover, x0, y0, s, *_ = bg._shade(bg._batches[0], height, width, focal,
                                        t(sw.pos), t(sw.rot), t(sw.size))
    out = torch.zeros(height + 2 * s, width + 2 * s)
    x, y = int(x0[0]) + s, int(y0[0]) + s
    out[y:y + s, x:x + s] = cover[0]
    return out[s:s + height, s:s + width]


def test_a_solid_cube_covers_the_middle():
    cover = coverage(one_shape("cube", hollow=False))
    assert cover[45, 80] == 1
    assert cover[2, 2] == 0


def test_a_hollow_cube_shows_the_wall_through_its_middle():
    solid = coverage(one_shape("cube", hollow=False))
    frame = coverage(one_shape("cube", hollow=True))
    assert frame[45, 80] == 0             # straight through the middle
    assert float(frame.max()) == 1        # the frame is there
    assert not bool(((frame > 0) & (solid == 0)).any())   # within the solid's outline


def test_a_shape_inside_you_is_pushed_out_of_the_near_side():
    sw = shapes3d.Swarm(count=1)
    sw.pos[:] = [[-0.5, 0.0, 10.0]]       # a little left of centre
    sw.home = sw.pos.copy()
    sw.freq[:] = 0
    # You, filling the middle of the frame, fading off to either side.
    x = np.linspace(-1, 1, shapes3d.FIELD_W)
    field = np.tile(np.exp(-(x / 0.4) ** 2), (shapes3d.FIELD_H, 1))
    for _ in range(10):
        sw.observe(field)
        sw.step(1 / 30)
    assert sw.vel[0, 0] < 0               # out to the left, the way it was already leaning


def test_moving_into_a_shape_kicks_it():
    def speed_after(fields):
        sw = shapes3d.Swarm(count=1)
        sw.pos[:] = [[2.0, 0.0, 10.0]]
        sw.home = sw.pos.copy()
        sw.freq[:] = 0
        for f in fields:
            sw.observe(f)
            sw.step(1 / 30)
        return abs(sw.vel[0, 0])

    x = np.linspace(-1, 1, shapes3d.FIELD_W)
    still = np.tile(np.exp(-((x - 0.1) / 0.3) ** 2), (shapes3d.FIELD_H, 1))
    moved = np.tile(np.exp(-((x - 0.3) / 0.3) ** 2), (shapes3d.FIELD_H, 1))
    assert speed_after([still, moved]) > speed_after([moved, moved])


@pytest.mark.skipif(not torch.cuda.is_available() or shapes3d.triton is None,
                    reason="needs CUDA and Triton")
@pytest.mark.parametrize("spec", ["shapes", "shapes:christmas", "shapes:easter"])
def test_the_kernel_draws_what_the_reference_draws(spec):
    bg = shapes3d.ShapesBackground(spec, "cuda", torch.float32)
    bg.snow = None          # the same in both; only the shapes are on trial
    for _ in range(30):
        bg.swarm.step(1 / 30)
    fast = bg.render(270, 480)
    bg._trace = shapes3d._patches
    slow = bg.render(270, 480)
    # A handful of silhouette pixels may land either side of a tie in float.
    diff = (fast - slow).abs()
    assert int((diff > 2 / 255).sum()) < 20
    assert float(diff.mean()) < 1e-4


def test_the_hat_is_eight_kites_with_thirteen_sides():
    hat = shapes3d.tile_polygon(*shapes3d.TILES["hat"])
    # A kite of the hexagonal grid, with sides 1 and sqrt 3, has area sqrt 3.
    assert shapes3d._area(hat) == pytest.approx(8 * np.sqrt(3))
    turns = [shapes3d._cross(hat[k - 1], hat[k], hat[(k + 1) % len(hat)])
             for k in range(len(hat))]
    assert sum(abs(t) > 1e-9 for t in turns) == 13


def test_the_spectre_has_fourteen_equal_edges():
    spectre = shapes3d.tile_polygon(*shapes3d.TILES["spectre"])
    edges = np.linalg.norm(np.roll(spectre, -1, 0) - spectre, axis=1)
    assert len(edges) == 14
    assert edges == pytest.approx(np.ones(14))


@pytest.mark.parametrize("kind", list(shapes3d.TILES))
def test_a_tile_splits_into_convex_parts_that_fill_it_exactly(kind):
    outline = shapes3d._tile_outline(kind)
    parts = shapes3d.tile_parts(outline)
    assert all(shapes3d._convex(p) for p in parts)
    # No side that is half outline, half cut: that shows as a seam.
    assert not any(shapes3d._mixed(p, outline) for p in parts)
    assert sum(shapes3d._area(p) for p in parts) == pytest.approx(shapes3d._area(outline))
    assert len(parts) < len(outline) - 2      # merged, not left as triangles
