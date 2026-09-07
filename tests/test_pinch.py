"""Pinch tracking and the floating window's geometry. No camera, no CUDA."""

from types import SimpleNamespace

import pytest

import desktop
import gestures


def hand(thumb, index, wrist=(0.5, 0.9), knuckle=(0.5, 0.7)):
    """Enough landmarks for the pinch maths: 21 points, only four of them real."""
    lm = [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
    lm[gestures.WRIST] = SimpleNamespace(x=wrist[0], y=wrist[1])
    lm[gestures.PIPS["index"]] = SimpleNamespace(x=knuckle[0], y=knuckle[1])
    lm[gestures.TIPS["thumb"]] = SimpleNamespace(x=thumb[0], y=thumb[1])
    lm[gestures.TIPS["index"]] = SimpleNamespace(x=index[0], y=index[1])
    return lm


def test_fingertips_together_is_a_pinch():
    x, y, closed = gestures.pinch(hand((0.40, 0.50), (0.42, 0.50)))
    assert closed
    assert x == pytest.approx(0.41) and y == pytest.approx(0.50)


def test_fingertips_apart_is_not():
    assert not gestures.pinch(hand((0.30, 0.50), (0.50, 0.50)))[2]


def test_pinch_holds_through_the_hysteresis_band():
    """A gap that would not start a pinch must not drop one already held."""
    middling = hand((0.40, 0.50), (0.46, 0.50))
    assert not gestures.pinch(middling, was_pinching=False)[2]
    assert gestures.pinch(middling, was_pinching=True)[2]


def test_pinch_is_scale_free():
    """The same gesture twice as far from the camera is still a pinch."""
    far = hand((0.45, 0.50), (0.46, 0.50), wrist=(0.5, 0.7), knuckle=(0.5, 0.6))
    assert gestures.pinch(far)[2]


def test_panel_geometry_keeps_the_window_aspect(monkeypatch):
    monkeypatch.setattr(desktop, "list_monitors", lambda: [])
    monkeypatch.setattr(desktop, "list_windows", lambda: [])
    monkeypatch.setattr(desktop, "resolve_spec",
                        lambda *_: desktop.Grab(0, 0, 1600, 900, label="editor"))
    monkeypatch.setattr(desktop, "DesktopBackground", lambda *a, **k: None)

    panel = desktop.FloatingWindow("desktop:window:editor", None, None, 1920, 1080)
    assert panel.width == 728 and panel.height == 408      # 16:9, 38% of the frame
    assert panel.label == "editor"


def test_move_to_eases_towards_the_hand():
    panel = desktop.FloatingWindow.__new__(desktop.FloatingWindow)
    panel.x = panel.y = 0.0
    panel.move_to(1.0, 1.0, smoothing=0.5)
    assert panel.x == pytest.approx(0.5)
    for _ in range(20):
        panel.move_to(1.0, 1.0, smoothing=0.5)
    assert panel.x == pytest.approx(1.0, abs=1e-3)


def test_draw_pastes_the_panel_without_touching_the_plate(monkeypatch):
    """The plate can be a cached background tensor, so draw must not write into it."""
    import torch

    monkeypatch.setattr(desktop, "resolve_spec",
                        lambda *_: desktop.Grab(0, 0, 1600, 900, label="editor"))
    monkeypatch.setattr(desktop, "list_monitors", lambda: [])
    monkeypatch.setattr(desktop, "list_windows", lambda: [])
    monkeypatch.setattr(desktop, "DesktopBackground", lambda *a, **k: None)

    panel = desktop.FloatingWindow("desktop:window:editor", None, None, 320, 180)
    panel.source = SimpleNamespace(
        sample=lambda h, w: torch.ones(1, 3, h, w) * 0.5, close=lambda: None)

    plate = torch.zeros(1, 3, 180, 320)
    out = panel.draw(plate)

    cx, cy = int(panel.x * 320), int(panel.y * 180)
    assert plate.max() == 0                      # the original is untouched
    assert out[0, 0, cy, cx] == pytest.approx(0.5)                  # panel centre
    assert out[0, 0, cy - panel.height // 2, cx] == pytest.approx(0.92)    # its edge
    assert out[0, 0, 5, 5] == 0                  # the plate elsewhere


def test_draw_survives_a_capture_that_has_no_frame_yet():
    panel = desktop.FloatingWindow.__new__(desktop.FloatingWindow)
    panel.width, panel.height = 64, 36
    panel.source = SimpleNamespace(sample=lambda h, w: None)
    plate = object()
    assert panel.draw(plate) is plate


class FakeDetector:
    """The tracking half of GestureDetector, without mediapipe or a camera."""
    _track = gestures.GestureDetector._track

    def __init__(self):
        self.pointer, self._pinch_streak = None, 0
        self.pinch_on, self.pinch_gap = None, None


def test_a_pinch_must_be_held_before_it_grabs():
    d = FakeDetector()
    closed = [hand((0.40, 0.50), (0.41, 0.50))]
    for _ in range(gestures.PINCH_CONFIRM - 1):
        d._track(closed)
        assert not d.pointer[2]          # not yet: a hand passing through the pose
    d._track(closed)
    assert d.pointer[2]


def test_a_resting_hand_never_grabs():
    """Curled fingers on a keyboard sit around 0.4 of hand size apart."""
    d = FakeDetector()
    resting = [hand((0.44, 0.50), (0.50, 0.50), wrist=(0.5, 0.9), knuckle=(0.5, 0.75))]
    for _ in range(20):
        d._track(resting)
    assert not d.pointer[2]


def test_a_dropped_hand_releases():
    d = FakeDetector()
    for _ in range(5):
        d._track([hand((0.40, 0.50), (0.41, 0.50))])
    assert d.pointer[2]
    for _ in range(gestures.PINCH_RELEASE):
        d._track([hand((0.30, 0.50), (0.50, 0.50))])
    assert not d.pointer[2]


class FakeGestures:
    """The debouncing half of GestureDetector, without mediapipe."""
    _consider = gestures.GestureDetector._consider

    def __init__(self, confirm=8, steadiness=0.06, cooldown=4.0):
        self.confirm, self.steadiness, self.cooldown = confirm, steadiness, cooldown
        self._streak, self._last_fired, self._events = (None, 0, None), {}, {}
        self._lock = __import__("threading").Lock()
        self._events = []

    def feed(self, gesture, where, times):
        for _ in range(times):
            self._consider(gesture, where)
        return self._events


def test_a_pose_held_still_fires_once():
    d = FakeGestures(confirm=8)
    assert d.feed("victory", (0.5, 0.5), 7) == []
    assert d.feed("victory", (0.5, 0.5), 1) == ["victory"]
    assert d.feed("victory", (0.5, 0.5), 20) == ["victory"]      # not again


def test_scratching_your_head_fires_nothing():
    """A hand in motion runs through poses; none of them is a gesture."""
    d = FakeGestures(confirm=8)
    for i in range(40):
        pose = ["victory", "thumbs_up", "open_palm"][i % 3]
        d._consider(pose, (0.5 + 0.02 * i, 0.5))
    assert d._events == []


def test_a_held_pose_that_drifts_away_starts_over():
    d = FakeGestures(confirm=8)
    d.feed("victory", (0.5, 0.5), 7)
    d.feed("victory", (0.7, 0.5), 1)          # same pose, somewhere else
    assert d._events == []
    assert d.feed("victory", (0.7, 0.5), 7) == ["victory"]
