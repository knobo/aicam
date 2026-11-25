"""The seasonal figures: they fit the renderer, and the calendar picks them."""

import datetime

import numpy as np
import pytest
import torch

import seasons
import shapes3d

KINDS = [f"{name}/{v}" for name in seasons.FIGURES for v in range(3)]


@pytest.mark.parametrize("kind", KINDS)
def test_every_figure_fits_the_kernel_and_the_unit_ball(kind):
    parts = seasons.figure(kind)
    assert all(len(p[1]) <= shapes3d.FACES for p in parts)
    pts = np.concatenate([seasons._corners(dict(n=p[0], h=p[1], ell=p[5])) for p in parts])
    assert np.linalg.norm(pts, axis=1).max() == pytest.approx(1.0, abs=1e-6)


def test_a_figure_covers_its_middle_and_not_the_corner():
    bg = shapes3d.ShapesBackground("shapes:christmas", "cpu", torch.float32)
    sw = bg.swarm
    sw.kind, sw.hollow, sw.upright = ["bauble/0"], [False], np.array([False])
    sw.pos, sw.rot, sw.size = np.array([[0.0, 0.0, 10.0]]), np.eye(3)[None], np.array([1.0])
    sw.material = [0]
    bg._build()
    focal = (90 / 2) / sw.tan_y
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    _, cover, x0, y0, s, *_ = bg._shade(bg._batches[0], 90, 160, focal,
                                        t(sw.pos), t(sw.rot), t(sw.size))
    assert float(cover[0, s // 2, s // 2]) == 1      # a ball is solid in the middle
    assert float(cover[0, 0, 0]) == 0


@pytest.mark.parametrize("day,season", [
    ("2026-11-28", None), ("2026-11-29", "christmas"),     # first Sunday of Advent
    ("2027-01-06", "christmas"), ("2027-01-07", None),
    ("2027-03-13", None), ("2027-03-14", "easter"),        # Easter Sunday is 28 March
    ("2027-03-29", "easter"), ("2027-03-30", None),
])
def test_the_calendar(day, season):
    assert seasons.season_of(datetime.date.fromisoformat(day)) == season


def test_season_is_the_gold_shapes_out_of_season():
    assert shapes3d.palette_of("shapes:season", datetime.date(2026, 6, 1)) == shapes3d.DEFAULT
    assert shapes3d.palette_of("shapes:season", datetime.date(2026, 12, 24)) == "christmas"


def test_upright_figures_stay_upright_when_knocked():
    sw = shapes3d.Swarm(cast=["nisse/0", "present/0"])
    field = np.zeros((shapes3d.FIELD_H, shapes3d.FIELD_W))
    for k in range(300):
        # You sweeping across the frame and back, hitting everything.
        x = np.linspace(-1, 1, shapes3d.FIELD_W)
        field = np.tile(np.exp(-((x - np.sin(k / 20)) / 0.3) ** 2), (shapes3d.FIELD_H, 1))
        sw.observe(field)
        sw.step(1 / 30)
    up = sw.rot[0] @ np.array([0, -1.0, 0])         # the nisse's own up, in the world
    assert up @ np.array([0, -1.0, 0]) > 0.95
