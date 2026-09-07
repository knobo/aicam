"""Fixtures for the overlay tests.

The clips are generated with ffmpeg so the tests carry no binary fixtures and
still exercise the real decode path in VideoOverlay.
"""

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _encode(path, filter_complex, duration=0.5, fps=10):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"color=c=black:s=64x32:d={duration}:r={fps}",
         "-vf", filter_complex, "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return str(path)


@pytest.fixture(scope="session")
def marked_clip(tmp_path_factory):
    """64x32, red block over the left quarter, black elsewhere.

    Cover-cropping this to a square keeps the middle 32 columns, which drops the
    block entirely; stretching would keep it. That difference is the test.
    """
    path = tmp_path_factory.mktemp("clips") / "marked.mp4"
    return _encode(path, "drawbox=x=0:y=0:w=16:h=32:color=red:t=fill")


@pytest.fixture(scope="session")
def luma_clip(tmp_path_factory):
    """64x32, white block on the left half, black on the right."""
    path = tmp_path_factory.mktemp("clips") / "luma.mp4"
    return _encode(path, "drawbox=x=0:y=0:w=32:h=32:color=white:t=fill")


@pytest.fixture(scope="session")
def green_clip(tmp_path_factory):
    """64x32, pure green on the left half, white subject on the right."""
    path = tmp_path_factory.mktemp("clips") / "green.mp4"
    return _encode(path, "drawbox=x=0:y=0:w=32:h=32:color=#00FF00:t=fill,"
                         "drawbox=x=32:y=0:w=32:h=32:color=white:t=fill")


@pytest.fixture(scope="session")
def still_image(tmp_path_factory):
    """64x32 image, red over the left quarter - the marked clip as one frame."""
    path = tmp_path_factory.mktemp("stills") / "still.png"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=c=black:s=64x32:d=1",
         "-vf", "drawbox=x=0:y=0:w=16:h=32:color=red:t=fill",
         "-frames:v", "1", str(path)],
        check=True,
    )
    return str(path)


@pytest.fixture(autouse=True)
def config_home(tmp_path, monkeypatch):
    """No test may touch the real ~/.config/aicam - one of them did, once."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return tmp_path / "config"
