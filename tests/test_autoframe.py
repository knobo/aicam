"""Auto-framing: the box maths and the damping. CPU tensors, no camera."""

import torch

import effects


def alpha_with(x0, y0, x1, y1, h=64, w=128):
    """A matte that is solid inside the given pixel box and empty outside."""
    a = torch.zeros(1, 1, h, w)
    a[:, :, y0:y1, x0:x1] = 1.0
    return a


def test_box_finds_the_subject():
    box = effects.subject_box(alpha_with(32, 16, 64, 48))
    assert box == (32 / 128, 16 / 64, 64 / 128, 48 / 64)


def test_no_subject_is_no_box():
    assert effects.subject_box(torch.zeros(1, 1, 64, 128)) is None


def test_crop_keeps_the_output_aspect():
    """A tall subject must widen the crop, never squash the picture."""
    cx, cy, w, h = effects.AutoFramer().target((0.4, 0.1, 0.6, 0.9), 2.0, 4.0)
    assert w == 2.0 * h


def test_crop_never_leaves_the_frame():
    for subject in [(0.0, 0.0, 0.1, 0.1), (0.9, 0.9, 1.0, 1.0)]:
        cx, cy, w, h = effects.AutoFramer().target(subject, 16 / 9, 4.0)
        assert cx - w / 2 >= -1e-9 and cx + w / 2 <= 1 + 1e-9
        assert cy - h / 2 >= -1e-9 and cy + h / 2 <= 1 + 1e-9


def test_zoom_is_capped():
    """A small subject far away must not be blown up past the limit."""
    _, _, w, _ = effects.AutoFramer().target((0.48, 0.48, 0.52, 0.52), 16 / 9, 1.25)
    assert w >= 1 / 1.25 - 1e-9


def test_a_subject_taller_than_the_crop_keeps_its_head():
    """Head to the bottom edge is the normal webcam framing, and centring on
    that box crops the face at the hairline."""
    subject = (0.35, 0.25, 0.75, 1.0)
    _, cy, _, h = effects.AutoFramer().target(subject, 16 / 9, 2.0)
    assert cy - h / 2 < subject[1]


def test_small_drift_does_not_move_the_crop():
    """Breathing is not a camera move."""
    f = effects.AutoFramer()
    first = f.advance((0.3, 0.2, 0.7, 0.9), 16 / 9, 2.0)
    same = f.advance((0.305, 0.2, 0.705, 0.9), 16 / 9, 2.0)
    assert same == first


def test_a_real_move_is_followed_but_eased():
    f = effects.AutoFramer()
    start = f.advance((0.0, 0.4, 0.2, 0.6), 16 / 9, 2.0)
    want = f.target((0.8, 0.4, 1.0, 0.6), 16 / 9, 2.0)
    step = f.advance((0.8, 0.4, 1.0, 0.6), 16 / 9, 2.0)
    assert start[0] < step[0] < want[0]      # moved towards it, did not jump


def test_an_empty_frame_holds_the_last_framing():
    f = effects.AutoFramer()
    held = f.advance((0.3, 0.2, 0.7, 0.9), 16 / 9, 2.0)
    assert f.advance(None, 16 / 9, 2.0) == held


def test_apply_returns_the_same_size_and_actually_crops():
    f = effects.AutoFramer()
    img = torch.rand(1, 3, 64, 128)
    out = f.apply(img, alpha_with(80, 16, 100, 48), zoom=2.0)
    assert out.shape == img.shape
    assert not torch.equal(out, img)


def test_apply_without_a_subject_leaves_the_frame_alone():
    img = torch.rand(1, 3, 64, 128)
    out = effects.AutoFramer().apply(img, torch.zeros(1, 1, 64, 128), zoom=2.0)
    assert torch.equal(out, img)
