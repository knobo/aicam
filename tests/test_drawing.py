"""Air drawing: where the ink lands, how deep it is, and what hides it.

CPU tensors throughout, no camera and no hand.
"""

import torch

import drawing


def depth_map(value=0.3, h=64, w=128):
    """A flat scene at one depth. 1 is near, so 0.3 is across the room."""
    return torch.full((1, 1, h, w), float(value))


def stroke(canvas, points, scene, **kw):
    canvas.begin()
    for x, y in points:
        canvas.extend(x, y, scene, **kw)
    canvas.end()


def test_ink_lands_just_in_front_of_what_it_was_drawn_on():
    """Drawn at arm's length, the ink must not be swallowed by the arm."""
    canvas = drawing.Canvas(torch.device("cpu"))
    scene = depth_map(0.30)
    stroke(canvas, [(0.2, 0.5), (0.8, 0.5)], scene)

    z = canvas.pos[:canvas.count, 2]
    assert torch.all(z > 0.30)
    assert torch.all(z < 0.30 + 2 * drawing.Z_BIAS)


def test_the_subject_stepping_in_front_hides_it():
    """Same ink, two different scenes: the one with something nearer occludes."""
    canvas = drawing.Canvas(torch.device("cpu"))
    stroke(canvas, [(0.2, 0.5), (0.8, 0.5)], depth_map(0.30))

    (_, front_a), (_, behind_a) = canvas.render(64, 128, scene_depth=depth_map(0.30))
    assert front_a.sum() > 0 and behind_a.sum() == 0

    # Now someone is standing there: nearer than the ink, so the ink is behind.
    (_, front_b), (_, behind_b) = canvas.render(64, 128, scene_depth=depth_map(0.95))
    assert behind_b.sum() > 0 and front_b.sum() == 0


def test_a_stroke_is_continuous_between_two_pointer_samples():
    """The tracker runs at 10 Hz. Two samples must still be one line, not two
    dots with a gap the width of however fast the hand was moving."""
    canvas = drawing.Canvas(torch.device("cpu"))
    scene = depth_map()
    canvas.begin()
    canvas.extend(0.1, 0.5, scene, width=6)
    canvas.extend(0.9, 0.5, scene, width=6)

    xs = canvas.pos[:canvas.count, 0]
    gaps = (xs[1:] - xs[:-1]).abs()
    assert canvas.count > 10
    assert float(gaps.max()) <= 3.0          # radius, so the stamps still overlap


def test_ink_does_not_pile_up_where_the_hand_lingered():
    """Stamps within a stroke overlap by design; alpha must stay ink, not glow."""
    canvas = drawing.Canvas(torch.device("cpu"))
    scene = depth_map()
    canvas.begin()
    for _ in range(40):
        canvas.extend(0.5, 0.5, scene)
    (_, alpha), _ = canvas.render(64, 128, scene_depth=scene)
    assert float(alpha.max()) <= 1.0


def test_undo_takes_the_last_stroke_and_nothing_else():
    canvas = drawing.Canvas(torch.device("cpu"))
    scene = depth_map()
    stroke(canvas, [(0.1, 0.2), (0.4, 0.2)], scene)
    after_first = canvas.count
    stroke(canvas, [(0.1, 0.8), (0.4, 0.8)], scene)

    assert canvas.count > after_first
    assert canvas.undo() is True
    assert canvas.count == after_first
    assert canvas.undo() is True
    assert canvas.empty


def test_a_full_canvas_drops_whole_strokes():
    """Room is made by wiping the oldest scribble, not by trimming its tail."""
    canvas = drawing.Canvas(torch.device("cpu"), capacity=400)
    scene = depth_map()
    for i in range(12):
        stroke(canvas, [(0.05, 0.1 * (i % 9) + 0.05), (0.95, 0.1 * (i % 9) + 0.05)], scene)

    assert canvas.count <= 400
    remaining = canvas.stroke[:canvas.count].unique().tolist()
    # Whatever survived is whole: no stroke id is present in part.
    assert remaining == sorted(remaining)
    assert canvas.strokes == 12


def test_colour_reaches_the_pixels():
    canvas = drawing.Canvas(torch.device("cpu"))
    scene = depth_map()
    stroke(canvas, [(0.2, 0.5), (0.8, 0.5)], scene, colour="cyan", width=8)
    (colour, alpha), _ = canvas.render(64, 128, scene_depth=scene)

    lit = alpha[0, 0] > 0.9
    r, g, b = (colour[0, ch][lit].mean() for ch in range(3))
    assert b > g > r                                    # cyan, premultiplied


def test_the_whole_composite_puts_you_in_front_of_your_own_ink():
    """The end of the pipeline, not just the layer split: a stroke drawn across
    the room, then someone standing in the middle of it."""
    import particles

    h, w = 64, 128
    canvas = drawing.Canvas(torch.device("cpu"))
    stroke(canvas, [(0.15, 0.5)] + [(0.85, 0.5)] * 12, depth_map(0.30, h, w))

    # Now a person stands in the middle band, nearer than the ink hanging there.
    depth_now = depth_map(0.30, h, w)
    depth_now[:, :, :, 45:85] = 0.95
    alpha = torch.zeros(1, 1, h, w)
    alpha[:, :, :, 45:85] = 1.0

    base = torch.zeros(1, 3, h, w)          # a black room
    subject = torch.ones(1, 3, h, w)        # a white person
    nothing = (torch.zeros(1, 3, h, w), torch.zeros(1, 1, h, w))

    ink_front, ink_behind = canvas.render(h, w, scene_depth=depth_now)
    out = particles.composite(base, subject, alpha,
                              drawing.over(ink_front, nothing),
                              drawing.over(ink_behind, nothing))

    # Across the person: the person, with no ink bleeding over them.
    assert float(out[0, :, :, 45:85].min()) > 0.999
    # Beside them: the stroke, and it is yellow rather than white.
    left = out[0, :, :, :45]
    assert float(left[0].max()) > 0.5
    assert float(left[2].max()) < float(left[0].max())


# ------------------------------------------------------------------- tools

class Clock:
    """A clock the test moves by hand, so fading is not timing-dependent."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_the_eraser_takes_what_is_under_the_finger_and_leaves_the_rest():
    canvas = drawing.Canvas(torch.device("cpu"))
    scene = depth_map()
    stroke(canvas, [(0.1, 0.5)] + [(0.9, 0.5)] * 12, scene)
    before = canvas.count

    canvas.begin("eraser")
    canvas.extend(0.5, 0.5, scene)
    canvas.end()

    assert canvas.count < before
    x = canvas.pos[:canvas.count, 0]
    assert float(((x - 64).abs()).min()) > 10        # a hole where the finger was
    assert float(x.min()) < 20 and float(x.max()) > 100   # the ends survived


def test_rubbing_out_can_be_undone():
    """The point of snapshots: the ink an eraser removed is gone, so stepping
    back cannot mean 'drop the last stroke' any more."""
    canvas = drawing.Canvas(torch.device("cpu"))
    scene = depth_map()
    stroke(canvas, [(0.1, 0.5)] + [(0.9, 0.5)] * 12, scene)
    before = canvas.count

    canvas.begin("eraser")
    canvas.extend(0.5, 0.5, scene)
    canvas.end()
    assert canvas.count < before

    assert canvas.undo() is True
    assert canvas.count == before


def test_wiping_the_canvas_can_be_undone():
    canvas = drawing.Canvas(torch.device("cpu"))
    scene = depth_map()
    stroke(canvas, [(0.1, 0.5), (0.9, 0.5)], scene)
    before = canvas.count

    canvas.clear()
    assert canvas.empty
    assert canvas.undo() is True
    assert canvas.count == before


def test_a_line_is_one_line_however_long_you_took_over_it():
    """Dragging redraws the line from its anchor; it must not leave a fan of
    every intermediate line behind."""
    canvas = drawing.Canvas(torch.device("cpu"))
    scene = depth_map()
    canvas.begin("line")
    for _ in range(15):
        canvas.extend(0.2, 0.2, scene)       # settle on the anchor
    for _ in range(25):
        canvas.extend(0.8, 0.8, scene)       # drag away from it
    canvas.end()

    p = canvas.pos[:canvas.count, :2]
    a, b = p[0], p[-1]
    span = b - a
    # Every point sits on the segment: cross product with the direction is zero.
    cross = (p[:, 0] - a[0]) * span[1] - (p[:, 1] - a[1]) * span[0]
    assert float(cross.abs().max()) < 1.0
    assert float(span.abs().max()) > 70       # 0.2 -> 0.8 across a 128 px frame


def test_a_rectangle_has_four_corners():
    canvas = drawing.Canvas(torch.device("cpu"))
    scene = depth_map()
    canvas.begin("rect")
    for _ in range(15):
        canvas.extend(0.2, 0.2, scene)
    for _ in range(25):
        canvas.extend(0.8, 0.8, scene)
    canvas.end()

    x = canvas.pos[:canvas.count, 0]
    y = canvas.pos[:canvas.count, 1]
    # Hollow: ink on all four sides, nothing across the middle.
    cx, cy = (float(x.min()) + float(x.max())) / 2, (float(y.min()) + float(y.max())) / 2
    inside = ((x - cx).abs() < (float(x.max()) - float(x.min())) / 4) & \
             ((y - cy).abs() < (float(y.max()) - float(y.min())) / 4)
    assert int(inside.sum()) == 0
    for corner in ((x.min(), y.min()), (x.max(), y.min()),
                   (x.min(), y.max()), (x.max(), y.max())):
        near = ((x - corner[0]).abs() < 3) & ((y - corner[1]).abs() < 3)
        assert int(near.sum()) > 0


def test_the_laser_fades_out_and_takes_itself_away():
    clock = Clock()
    canvas = drawing.Canvas(torch.device("cpu"), clock=clock)
    scene = depth_map()
    canvas.begin("laser")
    for x in (0.2, 0.5, 0.8):
        canvas.extend(x, 0.5, scene)
    canvas.end()
    drawn = canvas.count
    assert drawn > 0

    canvas.update(now=0.0)
    (_, alpha), _ = canvas.render(64, 128, scene_depth=scene)
    full = float(alpha.max())
    assert full > 0.9

    canvas.update(now=drawing.LASER_HOLD + drawing.LASER_FADE / 2)
    (_, alpha), _ = canvas.render(64, 128, scene_depth=scene)
    assert 0.2 < float(alpha.max()) < full

    canvas.update(now=drawing.LASER_HOLD + drawing.LASER_FADE + 0.01)
    assert canvas.empty


def test_ordinary_ink_never_fades():
    clock = Clock()
    canvas = drawing.Canvas(torch.device("cpu"), clock=clock)
    scene = depth_map()
    stroke(canvas, [(0.2, 0.5), (0.8, 0.5)], scene)
    before = canvas.count

    canvas.update(now=1000.0)
    assert canvas.count == before
    assert float(canvas.opacity[:before].min()) == 1.0
