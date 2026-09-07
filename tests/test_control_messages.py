"""Switching overlay, overlay layer and background on a running aicam.

These are the paths the GUI and aicamctl drive, so they run without CUDA.
"""

import os

import pytest
import torch

import aicam
import control
import overlays

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------- overlay

def test_overlay_message_opens_the_clip(luma_clip):
    overlay = aicam.apply_overlay_message({"overlay": luma_clip}, None, DEVICE)
    try:
        assert overlay.path == luma_clip
        assert overlay.layer == "front"
    finally:
        overlay.close()


def test_overlay_message_can_open_it_behind_the_subject(luma_clip):
    overlay = aicam.apply_overlay_message(
        {"overlay": luma_clip, "overlay_layer": "back"}, None, DEVICE)
    try:
        assert overlay.layer == "back"
    finally:
        overlay.close()


def test_null_overlay_removes_the_clip(luma_clip):
    overlay = overlays.VideoOverlay(luma_clip, DEVICE)
    assert aicam.apply_overlay_message({"overlay": None}, overlay, DEVICE) is None
    assert not overlay._thread.is_alive()      # the old clip was closed


def test_layer_alone_moves_the_running_clip_without_restarting_it(luma_clip):
    overlay = overlays.VideoOverlay(luma_clip, DEVICE)
    try:
        moved = aicam.apply_overlay_message({"overlay_layer": "back"}, overlay, DEVICE)
        assert moved is overlay                # same object: playback continues
        assert overlay.layer == "back"
    finally:
        overlay.close()


def test_layer_alone_with_no_clip_is_harmless():
    assert aicam.apply_overlay_message({"overlay_layer": "back"}, None, DEVICE) is None


def test_unknown_layer_is_rejected(luma_clip):
    overlay = overlays.VideoOverlay(luma_clip, DEVICE)
    try:
        with pytest.raises(ValueError):
            aicam.apply_overlay_message({"overlay_layer": "middle"}, overlay, DEVICE)
        assert overlay.layer == "front"        # unchanged
    finally:
        overlay.close()


# ---------------------------------------------------------------- background

def test_background_message_opens_an_image(still_image):
    bg = aicam.apply_background_message({"background": still_image}, None,
                                        DEVICE, torch.float32)
    try:
        assert bg.sample(32, 64).shape == (1, 3, 32, 64)
    finally:
        bg.close()


def test_background_message_opens_a_video(marked_clip):
    bg = aicam.apply_background_message({"background": marked_clip}, None,
                                        DEVICE, torch.float32)
    try:
        assert isinstance(bg, overlays.VideoBackground)
    finally:
        bg.close()


def test_null_background_falls_back_to_blur(still_image):
    bg = overlays.open_background(still_image, DEVICE, torch.float32)
    assert aicam.apply_background_message({"background": None}, bg,
                                          DEVICE, torch.float32) is None


def test_switching_background_closes_the_previous_video(marked_clip, still_image):
    old = overlays.open_background(marked_clip, DEVICE, torch.float32)
    new = aicam.apply_background_message({"background": still_image}, old,
                                         DEVICE, torch.float32)
    try:
        assert not old._clip._thread.is_alive()
    finally:
        new.close()


def test_missing_background_keeps_the_current_one(still_image, tmp_path):
    current = overlays.open_background(still_image, DEVICE, torch.float32)
    try:
        with pytest.raises(FileNotFoundError):
            aicam.apply_background_message({"background": str(tmp_path / "nope.mp4")},
                                           current, DEVICE, torch.float32)
        assert current.sample(32, 64) is not None    # still usable
    finally:
        current.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="needs an X display")
def test_background_message_opens_a_screen_grab(still_image):
    current = overlays.open_background(still_image, DEVICE, torch.float32)
    bg = aicam.apply_background_message({"background": "desktop:left"}, current,
                                        DEVICE, torch.float32, width=320, height=180)
    try:
        assert bg.grab.label      # resolved to a named monitor
    finally:
        bg.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="needs an X display")
def test_an_unknown_monitor_keeps_the_current_background(still_image):
    current = overlays.open_background(still_image, DEVICE, torch.float32)
    try:
        with pytest.raises(ValueError):
            aicam.apply_background_message({"background": "desktop:DP-9"}, current,
                                           DEVICE, torch.float32, width=320, height=180)
        assert current.sample(32, 64) is not None
    finally:
        current.close()


# ---------------------------------------------------------------- source strings

def test_a_file_argument_becomes_an_absolute_path():
    # aicam runs elsewhere, so a relative path from the caller's shell is useless.
    assert control.normalise_source("clips/beach.mp4").startswith("/")


def test_a_url_is_left_alone():
    url = "https://example.com/videoplayback?expire=1788192938&mime=video%2Fmp4"
    assert control.normalise_source(url) == url


def test_a_desktop_spec_is_left_alone():
    assert control.normalise_source("desktop:window:youtube") == "desktop:window:youtube"


def test_off_means_no_source_at_all():
    assert control.normalise_source("off") is None


# ---------------------------------------------------------------- history

def test_a_background_that_opened_is_remembered(still_image, tmp_path):
    import history
    path = str(tmp_path / "history.json")
    bg = aicam.apply_background_message({"background": still_image}, None,
                                        DEVICE, torch.float32, history_path=path)
    try:
        assert [e["source"] for e in history.load(path)] == [still_image]
    finally:
        bg.close()


def test_a_background_that_failed_is_not_remembered(tmp_path):
    import history
    path = str(tmp_path / "history.json")
    with pytest.raises(FileNotFoundError):
        aicam.apply_background_message({"background": str(tmp_path / "nope.mp4")},
                                       None, DEVICE, torch.float32, history_path=path)
    assert history.load(path) == []


def test_removing_the_background_remembers_nothing(tmp_path):
    import history
    path = str(tmp_path / "history.json")
    assert aicam.apply_background_message({"background": None}, None, DEVICE,
                                          torch.float32, history_path=path) is None
    assert history.load(path) == []


# ---------------------------------------------------------------- pasted urls

def test_a_youtube_link_becomes_a_stream_spec():
    assert control.spec_for_url("https://www.youtube.com/watch?v=abc") == \
        "yt:https://www.youtube.com/watch?v=abc"


def test_a_short_youtube_link_counts_too():
    assert control.spec_for_url("https://youtu.be/abc").startswith("yt:")


def test_a_direct_media_link_is_used_as_it_is():
    url = "https://example.com/clip.mp4"
    assert control.spec_for_url(url) == url


def test_a_spec_that_is_already_prefixed_is_left_alone():
    assert control.spec_for_url("yt:https://youtu.be/abc") == "yt:https://youtu.be/abc"


def test_surrounding_whitespace_from_a_paste_is_trimmed():
    assert control.spec_for_url("  https://youtu.be/abc\n") == "yt:https://youtu.be/abc"


def test_close_pipe_kills_a_process_that_ignores_eof():
    """A wedged ffmpeg must still end, or v4l2loopback stays broken."""
    import subprocess as sp
    import sys as _sys

    proc = sp.Popen([_sys.executable, "-c", "import time; time.sleep(30)"],
                    stdin=sp.PIPE)
    aicam.close_pipe(proc, timeout=0.5)
    assert proc.poll() is not None


def test_close_pipe_lets_a_well_behaved_process_exit_on_eof():
    import subprocess as sp
    import sys as _sys

    proc = sp.Popen([_sys.executable, "-c", "import sys; sys.stdin.read()"],
                    stdin=sp.PIPE)
    aicam.close_pipe(proc, timeout=5)
    assert proc.returncode == 0
