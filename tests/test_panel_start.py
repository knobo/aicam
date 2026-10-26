"""What the panel shows between pressing Start and aicam answering.

Stands in a short script for aicam.py, so it needs neither CUDA nor a camera,
only a display for Tk.
"""

import os
import subprocess
import sys
import time

import pytest

tk = pytest.importorskip("tkinter")

import gui  # noqa: E402

needs_display = pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="no X display")


@pytest.fixture
def panel(tmp_path, monkeypatch):
    monkeypatch.setattr(gui, "LOG", str(tmp_path / "aicam.log"))
    root = tk.Tk()
    p = gui.Panel(root, socket_path="/nonexistent/aicam.sock")
    yield p
    if p.proc is not None:
        p.proc.kill()
        p.proc.wait()
    root.destroy()


def launch(panel, script):
    """Run script the way Start runs aicam.py."""
    with open(gui.LOG, "w") as log:
        panel.proc = subprocess.Popen([sys.executable, "-c", script], stderr=log)
    panel.launching = True


def test_every_stage_is_something_aicam_prints():
    with open(os.path.join(os.path.dirname(gui.__file__), "aicam.py")) as f:
        source = f.read()
    assert [s for s in gui.STAGES if s not in source] == []


@needs_display
def test_progress_keeps_the_furthest_stage(panel):
    launch(panel, "import sys, time\n"
                  "sys.stderr.write('aicam: loading rvm\\naicam: loaded MiDaS_small\\n"
                  "Downloading control socket\\naicam: camera gave 640x480\\n"
                  "Using cache found in hub\\n')\n"
                  "sys.stderr.flush(); time.sleep(5)")
    deadline = time.monotonic() + 5
    while "camera gave" not in open(gui.LOG).read() and time.monotonic() < deadline:
        time.sleep(0.02)
    panel.refresh()
    # Lines aicam did not write as aicam: neither count as a stage nor show.
    assert panel.progress["value"] == 2
    assert panel.stage["text"] == "camera gave 640x480"


@needs_display
def test_a_failed_launch_says_why(panel):
    launch(panel, "import sys\nsys.exit('aicam: ffmpeg could not open /dev/video10.\\n"
                  "the loopback device is\\nwedged. Reload the module:\\n  sudo modprobe')")
    panel.proc.wait()
    panel.refresh()
    assert panel.stage["text"] == ("failed: the loopback device is\n"
                                   "wedged. Reload the module:\nsudo modprobe")
    assert panel.stage.winfo_manager() == "grid"
    assert panel.run_button["text"] == "Start"
