"""Overlay keying, layer placement and background sources.

Everything here runs on the CPU: no CUDA, no camera, no loopback device.
"""

import os
import time

import pytest
import torch

import overlays

DEVICE = torch.device("cpu")


def first_sample(source, height, width, timeout=5.0):
    """Wait for the decode thread to hand over a frame."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        got = source.sample(height, width)
        if got is not None:
            return got
        time.sleep(0.01)
    raise AssertionError("the decoder never produced a frame")


# ---------------------------------------------------------------- keying

def test_luma_alpha_follows_brightness(luma_clip):
    overlay = overlays.VideoOverlay(luma_clip, DEVICE, mode="luma")
    try:
        _, alpha = first_sample(overlay, 32, 64)
    finally:
        overlay.close()
    assert alpha[0, 0, 16, 8] > 0.9        # white half
    assert alpha[0, 0, 16, 56] < 0.1       # black half


def test_chroma_keys_out_the_green(green_clip):
    overlay = overlays.VideoOverlay(green_clip, DEVICE, mode="chroma")
    try:
        _, alpha = first_sample(overlay, 32, 64)
    finally:
        overlay.close()
    assert alpha[0, 0, 16, 8] < 0.1        # green half is keyed away
    assert alpha[0, 0, 16, 56] > 0.9       # white subject survives


def test_opaque_mode_is_fully_opaque(marked_clip):
    overlay = overlays.VideoOverlay(marked_clip, DEVICE, mode="opaque")
    try:
        _, alpha = first_sample(overlay, 32, 64)
    finally:
        overlay.close()
    assert torch.allclose(alpha, torch.ones_like(alpha))


def test_opaque_mode_covers_the_frame_without_stretching(marked_clip):
    # 64x32 sampled at 32x32: cover-cropping keeps the middle columns, so the
    # red block on the left quarter is cropped away. Stretching would keep it.
    overlay = overlays.VideoOverlay(marked_clip, DEVICE, mode="opaque")
    try:
        colour, _ = first_sample(overlay, 32, 32)
    finally:
        overlay.close()
    assert colour[0, 0, 16, 1] < 0.2       # no red at the left edge


# ---------------------------------------------------------------- layer

def test_overlay_sits_in_front_by_default(luma_clip):
    overlay = overlays.VideoOverlay(luma_clip, DEVICE)
    try:
        assert overlay.layer == "front"
    finally:
        overlay.close()


def test_overlay_can_be_placed_behind_the_subject(luma_clip):
    overlay = overlays.VideoOverlay(luma_clip, DEVICE, layer="back")
    try:
        assert overlay.layer == "back"
    finally:
        overlay.close()


def test_unknown_layer_is_rejected(luma_clip):
    with pytest.raises(ValueError):
        overlays.VideoOverlay(luma_clip, DEVICE, layer="sideways")


# ---------------------------------------------------------------- backgrounds

def test_image_background_is_cover_cropped(still_image):
    bg = overlays.open_background(still_image, DEVICE, torch.float32)
    try:
        frame = bg.sample(32, 32)
    finally:
        bg.close()
    assert frame.shape == (1, 3, 32, 32)
    assert frame[0, 0, 16, 1] < 0.2        # red left quarter cropped away


def test_image_background_returns_the_same_frame_every_time(still_image):
    bg = overlays.open_background(still_image, DEVICE, torch.float32)
    try:
        assert torch.equal(bg.sample(32, 64), bg.sample(32, 64))
    finally:
        bg.close()


def test_video_background_gives_a_full_frame(marked_clip):
    bg = overlays.open_background(marked_clip, DEVICE, torch.float32)
    try:
        frame = first_sample(bg, 32, 64)
    finally:
        bg.close()
    assert frame.shape == (1, 3, 32, 64)
    assert frame.dtype == torch.float32


def test_background_honours_the_requested_dtype(still_image):
    bg = overlays.open_background(still_image, DEVICE, torch.float16)
    try:
        assert bg.sample(32, 64).dtype == torch.float16
    finally:
        bg.close()


def test_missing_background_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        overlays.open_background(str(tmp_path / "nope.mp4"), DEVICE, torch.float32)


# ---------------------------------------------------------------- shutdown

def test_close_stops_the_decoder_before_releasing_the_capture(luma_clip):
    # Releasing the VideoCapture while the decode thread is inside read()
    # segfaults OpenCV, so close() has to wait for the thread first.
    overlay = overlays.VideoOverlay(luma_clip, DEVICE)
    first_sample(overlay, 32, 64)
    overlay.close()
    assert not overlay._thread.is_alive()


def test_backgrounds_remember_their_path(still_image, marked_clip):
    # The GUI shows what is loaded, so a background has to say where it came from.
    still = overlays.open_background(still_image, DEVICE, torch.float32)
    clip = overlays.open_background(marked_clip, DEVICE, torch.float32)
    try:
        assert still.path == still_image
        assert clip.path == marked_clip
    finally:
        still.close()
        clip.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="needs an X display")
def test_a_desktop_spec_opens_a_screen_grab():
    import desktop
    bg = overlays.open_background("desktop", DEVICE, torch.float32,
                                  width=320, height=180)
    try:
        assert isinstance(bg, desktop.DesktopBackground)
    finally:
        bg.close()


def test_a_desktop_spec_needs_a_frame_size():
    with pytest.raises(ValueError):
        overlays.open_background("desktop", DEVICE, torch.float32)


def test_a_yt_spec_is_resolved_before_it_is_opened(marked_clip, monkeypatch):
    import streams
    monkeypatch.setattr(streams, "resolve", lambda spec, **kw: marked_clip)
    bg = overlays.open_background("yt:https://youtu.be/abc", DEVICE, torch.float32)
    try:
        assert first_sample(bg, 32, 64) is not None
        assert bg.path == "yt:https://youtu.be/abc"   # the spec, not the signed URL
    finally:
        bg.close()


def test_an_exhausted_source_is_reopened_rather_than_rewound(marked_clip):
    # A signed stream URL cannot be seeked back to frame 0 once it ends; it has
    # to be resolved and opened again. Local files take the same path.
    calls = []

    def reopen():
        calls.append(1)
        return marked_clip

    overlay = overlays.VideoOverlay(marked_clip, DEVICE, mode="opaque", reopen=reopen)
    try:
        deadline = time.monotonic() + 10
        while not calls and time.monotonic() < deadline:
            overlay.sample(32, 64)
            time.sleep(0.02)
        assert calls, "the clip ran out and was never reopened"
    finally:
        overlay.close()
