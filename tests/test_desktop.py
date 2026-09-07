"""Screen capture: enumerating monitors and windows, and resolving a spec.

The parsing and resolution are pure functions over captured tool output, so
they run anywhere. The one test that touches a real X display skips itself
without one.
"""

import os

import pytest

import desktop

# Real output from `xrandr --listmonitors` on a two-head X11 desktop.
XRANDR = """Monitors: 2
 0: +*HDMI-1-0 3840/597x2160/336+1680+0  HDMI-1-0
 1: +HDMI-4 1680/459x1050/296+0+0  HDMI-4
"""

GEOMETRY = """WINDOW=94371850
X=2061
Y=84
WIDTH=2739
HEIGHT=1659
SCREEN=0
"""


# ---------------------------------------------------------------- monitors

def test_monitors_are_parsed_with_geometry():
    monitors = desktop.parse_monitors(XRANDR)
    assert [m.name for m in monitors] == ["HDMI-4", "HDMI-1-0"]   # sorted left to right
    right = monitors[1]
    assert (right.width, right.height, right.x, right.y) == (3840, 2160, 1680, 0)


def test_primary_monitor_is_marked():
    monitors = desktop.parse_monitors(XRANDR)
    assert [m.primary for m in monitors] == [False, True]


def test_no_monitors_is_an_error():
    with pytest.raises(RuntimeError):
        desktop.parse_monitors("Monitors: 0\n")


# ---------------------------------------------------------------- windows

def test_window_geometry_is_parsed():
    window = desktop.parse_window_geometry(GEOMETRY)
    assert window == {"id": 94371850, "x": 2061, "y": 84, "width": 2739, "height": 1659}


# ---------------------------------------------------------------- specs

def test_plain_desktop_covers_every_monitor():
    grab = desktop.resolve_spec("desktop", desktop.parse_monitors(XRANDR), [])
    # The bounding box of 1680x1050 at +0+0 and 3840x2160 at +1680+0.
    assert (grab.x, grab.y, grab.width, grab.height) == (0, 0, 5520, 2160)


def test_a_monitor_can_be_named():
    grab = desktop.resolve_spec("desktop:HDMI-4", desktop.parse_monitors(XRANDR), [])
    assert (grab.x, grab.y, grab.width, grab.height) == (0, 0, 1680, 1050)


def test_left_and_right_follow_the_x_offset():
    monitors = desktop.parse_monitors(XRANDR)
    assert desktop.resolve_spec("desktop:left", monitors, []).width == 1680
    assert desktop.resolve_spec("desktop:right", monitors, []).width == 3840


def test_monitor_names_are_case_insensitive():
    grab = desktop.resolve_spec("desktop:hdmi-4", desktop.parse_monitors(XRANDR), [])
    assert grab.width == 1680


def test_unknown_monitor_names_the_ones_there_are():
    with pytest.raises(ValueError) as exc:
        desktop.resolve_spec("desktop:DP-9", desktop.parse_monitors(XRANDR), [])
    assert "HDMI-4" in str(exc.value)


def test_a_window_is_matched_on_its_title():
    windows = [desktop.Window(id=17, title="YouTube - Chromium", x=5, y=6,
                              width=1280, height=720)]
    grab = desktop.resolve_spec("desktop:window:youtube",
                                desktop.parse_monitors(XRANDR), windows)
    assert grab.window_id == 17
    assert (grab.width, grab.height) == (1280, 720)


def test_an_unmatched_window_title_is_an_error():
    with pytest.raises(ValueError):
        desktop.resolve_spec("desktop:window:spreadsheet",
                             desktop.parse_monitors(XRANDR), [])


def test_desktop_specs_are_recognised():
    assert desktop.is_desktop_spec("desktop")
    assert desktop.is_desktop_spec("desktop:left")
    assert not desktop.is_desktop_spec("/home/user/clips/beach.mp4")
    assert not desktop.is_desktop_spec("desktopchairs.png")


# ---------------------------------------------------------------- ffmpeg command

def test_a_region_grab_asks_x11grab_for_that_corner():
    grab = desktop.Grab(x=1680, y=0, width=3840, height=2160)
    cmd = desktop.build_command(grab, 1920, 1080, 30, display=":0")
    assert "x11grab" in cmd
    assert ":0+1680,0" in cmd
    assert "-window_id" not in cmd
    assert "scale=1920:1080" in " ".join(cmd)    # scaled before it hits the pipe


def test_a_window_grab_asks_x11grab_for_that_window():
    grab = desktop.Grab(x=5, y=6, width=1280, height=720, window_id=17)
    cmd = desktop.build_command(grab, 1920, 1080, 30, display=":0")
    assert cmd[cmd.index("-window_id") + 1] == "17"
    # Inside a window, its desktop position is not an offset: x11grab rejects
    # anything but 0,0 there with "outside the screen size".
    assert ":0+0,0" in cmd


# ---------------------------------------------------------------- live capture

@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="needs an X display")
def test_capturing_the_real_desktop_yields_a_frame():
    import torch
    source = desktop.DesktopBackground("desktop", torch.device("cpu"), torch.float32,
                                       width=320, height=180, fps=10)
    try:
        frame = None
        for _ in range(200):
            frame = source.sample(180, 320)
            if frame is not None:
                break
            import time
            time.sleep(0.05)
        assert frame is not None, "no frame arrived from x11grab"
        assert frame.shape == (1, 3, 180, 320)
        assert float(frame.max()) > 0.0        # not a black rectangle
    finally:
        source.close()


def test_wayland_says_so_instead_of_going_black(monkeypatch):
    import torch
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.delenv("DISPLAY", raising=False)
    with pytest.raises(RuntimeError) as exc:
        desktop.DesktopBackground("desktop", torch.device("cpu"), torch.float32,
                                  width=320, height=180, fps=10)
    assert "wayland" in str(exc.value).lower()


def test_listing_screens_does_not_drag_in_torch():
    # The Tk panel enumerates monitors and windows. It has no business loading
    # CUDA to fill a dropdown, so the heavy imports stay inside the capture.
    import subprocess
    import sys
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, os; sys.path.insert(0, os.getcwd()); import desktop; "
         "print('torch' in sys.modules, 'cv2' in sys.modules)"],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
    assert out.stdout.strip() == "False False", out.stdout + out.stderr
