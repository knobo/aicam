"""The depth warp behind --parallax. CPU only, no camera needed."""

import torch

import effects


def _ramp(width=32, height=16):
    """An image whose red channel is the x coordinate, so a shift is measurable."""
    x = torch.linspace(0, 1, width).view(1, 1, 1, width).expand(1, 3, height, width)
    return x.contiguous()


def test_pivot_depth_does_not_move():
    img = _ramp()
    depth = torch.full((1, 1, 16, 32), 0.5)
    out = effects.parallax(img, depth, 0.2, 0.1, pivot=0.5)
    assert torch.allclose(out, img, atol=1e-5)


def test_near_and_far_shift_in_opposite_directions():
    img = _ramp()
    depth = torch.full((1, 1, 16, 32), 0.9)
    near = effects.parallax(img, depth, 0.2, 0.0, pivot=0.5)
    far = effects.parallax(img, depth.fill_(0.1), 0.2, 0.0, pivot=0.5)

    middle = (slice(None), 0, 8, 16)
    assert near[middle] > img[middle]      # samples further right
    assert far[middle] < img[middle]


def test_shift_grows_with_disparity():
    img = _ramp()
    column = (slice(None), 0, 8, 16)
    small = effects.parallax(img, torch.full((1, 1, 16, 32), 0.6), 0.2, 0.0, 0.5)
    large = effects.parallax(img, torch.full((1, 1, 16, 32), 1.0), 0.2, 0.0, 0.5)
    assert large[column] - img[column] > small[column] - img[column]


def test_fill_behind_replaces_the_subject_with_its_surroundings():
    """The hole must take the colour of the wall, not keep the person."""
    img = torch.zeros(1, 3, 64, 64)
    img[:, 0] = 1.0                      # a red room
    alpha = torch.zeros(1, 1, 64, 64)
    alpha[..., 24:40, 24:40] = 1.0       # a subject in the middle
    img[:, :, 24:40, 24:40] = torch.tensor([0.0, 0.0, 1.0]).view(1, 3, 1, 1)   # in blue

    out = effects.fill_behind(img, alpha)
    middle = out[0, :, 32, 32]
    assert middle[0] > 0.8 and middle[2] < 0.2      # red wall, no blue left
    assert torch.allclose(out[0, :, 4, 4], img[0, :, 4, 4], atol=1e-4)   # rest untouched
