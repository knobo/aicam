"""The control panel in a window too short for it.

Needs a display, so it is skipped where there is none. Everything else about
the panel is tested through the messages it sends; this is the one part that
only exists as geometry.
"""

import os

import pytest

tk = pytest.importorskip("tkinter")
from tkinter import ttk  # noqa: E402

import gui  # noqa: E402

pytestmark = pytest.mark.skipif(not os.environ.get("DISPLAY"),
                                reason="no X display")

TALL, SHORT = "420x1200", "420x480"


@pytest.fixture
def panel():
    root = tk.Tk()
    # A socket nothing answers on: the panel must not depend on a running aicam.
    p = gui.Panel(root, socket_path="/nonexistent/aicam.sock")
    settle(root)
    yield p
    root.destroy()


def settle(root, times=6):
    for _ in range(times):
        root.update()


def resize(panel, geometry):
    panel.root.geometry(geometry)
    settle(panel.root)


def test_a_short_window_gets_a_scrollbar(panel):
    resize(panel, SHORT)
    assert panel._bar.winfo_ismapped()


def test_a_tall_enough_window_does_not(panel):
    resize(panel, TALL)
    assert not panel._bar.winfo_ismapped()


def test_the_wheel_scrolls_the_panel_rather_than_the_control_under_it(panel):
    """ttk gives the wheel to a combobox, which would change the tool as you
    scrolled past it. In a panel that scrolls, that is a trap."""
    resize(panel, SHORT)
    before = panel.tool.get()

    for _ in range(3):
        panel.tool_box.event_generate("<Button-5>", x=5, y=5)
    settle(panel.root)

    assert panel._view.yview()[0] > 0.0
    assert panel.tool.get() == before


def test_the_wheel_does_nothing_when_it_all_fits(panel):
    resize(panel, TALL)
    panel.run_button.event_generate("<Button-5>", x=5, y=5)
    settle(panel.root)
    assert panel._view.yview()[0] == 0.0


def test_growing_the_window_brings_the_top_back(panel):
    """Scrolled to the bottom, then made tall: the controls must not stay parked
    above the top of the window with no way to reach them."""
    resize(panel, SHORT)
    panel._view.yview_moveto(1.0)
    settle(panel.root)
    assert panel._view.yview()[0] > 0.0

    resize(panel, TALL)
    assert panel._view.yview()[0] == 0.0
