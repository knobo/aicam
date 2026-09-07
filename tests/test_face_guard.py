"""The face gate: a hand on the face is not a gesture. No camera, no model."""

from types import SimpleNamespace

import gestures


def palm_at(x, y):
    """21 landmarks all in one spot: enough for the palm-centre test."""
    return [SimpleNamespace(x=x, y=y) for _ in range(21)]


FACE = (0.40, 0.20, 0.60, 0.50)     # a head, upper middle of the frame


def test_hand_on_the_face_is_ignored():
    assert gestures.on_face(palm_at(0.50, 0.35), FACE)


def test_hand_beside_the_face_is_not():
    assert not gestures.on_face(palm_at(0.75, 0.35), FACE)


def test_hand_at_the_chin_still_counts_as_on_the_face():
    """Just below the box: the margin has to cover a fist under the jaw."""
    assert gestures.on_face(palm_at(0.50, 0.53), FACE)


def test_no_face_found_gates_nothing():
    assert not gestures.on_face(palm_at(0.50, 0.35), None)
