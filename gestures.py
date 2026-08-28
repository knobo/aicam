"""Hand gesture recognition, off the critical path.

MediaPipe's hand landmarker costs 18-20 ms per frame on this machine, which does
not fit in a 33 ms budget alongside matting, depth and bokeh. It is also pure
CPU work, so it runs on its own thread against the most recent frame and the
render loop never waits for it. Hands do not move fast enough for the resulting
lag to matter.
"""

import itertools
import threading
import time

import cv2
import numpy as np

WRIST = 0
TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIPS = {"thumb": 2, "index": 6, "middle": 10, "ring": 14, "pinky": 18}

# Which scene each gesture fires. Kept here so the mapping is one obvious table.
GESTURE_SCENES = {
    "thumbs_up": "confetti",
    "heart": "hearts",
    "open_palms": "fireworks",
    "victory": "balloons",
    "thumbs_down": "rain",
}


def _extended(lm, finger):
    """A finger counts as extended when its tip is further from the wrist than
    its middle joint. Scale-free, so it works at any distance from the camera."""
    w = np.array([lm[WRIST].x, lm[WRIST].y])
    tip = np.array([lm[TIPS[finger]].x, lm[TIPS[finger]].y])
    pip = np.array([lm[PIPS[finger]].x, lm[PIPS[finger]].y])
    return np.linalg.norm(tip - w) > np.linalg.norm(pip - w) * 1.15


def classify_hand(lm):
    """One hand's landmarks -> a coarse pose name."""
    fingers = {f: _extended(lm, f) for f in TIPS}
    curled = [f for f in ("index", "middle", "ring", "pinky") if not fingers[f]]

    if fingers["thumb"] and len(curled) == 4:
        # Thumb clearly above or below the wrist decides up from down.
        if lm[TIPS["thumb"]].y < lm[WRIST].y - 0.08:
            return "thumbs_up"
        if lm[TIPS["thumb"]].y > lm[WRIST].y + 0.08:
            return "thumbs_down"
        return None
    if fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
        return "victory"
    if all(fingers.values()):
        return "open_palm"
    return None


def classify_frame(hands):
    """All hands in one frame -> a single gesture name, or None."""
    if not hands:
        return None

    poses = [classify_hand(h) for h in hands]

    if len(hands) == 2:
        # Heart: both hands near each other with thumbs and index tips meeting.
        a, b = hands
        thumb_gap = abs(a[TIPS["thumb"]].x - b[TIPS["thumb"]].x) + \
                    abs(a[TIPS["thumb"]].y - b[TIPS["thumb"]].y)
        index_gap = abs(a[TIPS["index"]].x - b[TIPS["index"]].x) + \
                    abs(a[TIPS["index"]].y - b[TIPS["index"]].y)
        if thumb_gap < 0.12 and index_gap < 0.16:
            return "heart"
        if poses.count("open_palm") == 2:
            return "open_palms"

    for pose in poses:
        if pose in ("thumbs_up", "thumbs_down", "victory"):
            return pose
    return None


class GestureDetector:
    """Background hand tracking with debouncing.

    A gesture must be seen on `confirm` consecutive detections before it fires,
    and the same gesture cannot fire again within `cooldown` seconds. Without
    both, a hand passing through a pose on its way somewhere else sets off
    confetti in the middle of a meeting.
    """

    def __init__(self, model_path, downscale=0.5, confirm=3, cooldown=4.0,
                 interval=0.10, num_hands=2):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._timestamps = itertools.count(1)

        self.downscale = downscale
        self.confirm = confirm
        self.cooldown = cooldown
        self.interval = interval

        self._latest = None
        self._lock = threading.Lock()
        self._events = []
        self._stop = threading.Event()
        self._streak = (None, 0)
        self._last_fired = {}
        self.hands_visible = 0

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame_bgr):
        """Hand over the newest frame. Older unprocessed frames are dropped."""
        with self._lock:
            self._latest = frame_bgr

    def poll(self):
        """Gesture names recognised since the last call."""
        with self._lock:
            out, self._events = self._events, []
        return out

    def _run(self):
        while not self._stop.is_set():
            with self._lock:
                frame = self._latest
                self._latest = None
            if frame is None:
                time.sleep(0.01)
                continue

            small = cv2.resize(frame, None, fx=self.downscale, fy=self.downscale,
                               interpolation=cv2.INTER_AREA)
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                   data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            try:
                result = self._landmarker.detect_for_video(image, next(self._timestamps))
            except Exception:
                continue

            self.hands_visible = len(result.hand_landmarks)
            self._consider(classify_frame(result.hand_landmarks))
            time.sleep(self.interval)

    def _consider(self, gesture):
        name, streak = self._streak
        streak = streak + 1 if gesture == name else 1
        self._streak = (gesture, streak)

        if gesture is None or streak != self.confirm:
            return
        now = time.monotonic()
        if now - self._last_fired.get(gesture, -1e9) < self.cooldown:
            return
        self._last_fired[gesture] = now
        with self._lock:
            self._events.append(gesture)

    def close(self):
        """Stop the worker before closing the landmarker.

        MediaPipe dispatches its teardown through a thread pool. Letting the
        object be finalised at interpreter shutdown instead raises "cannot
        schedule new futures after shutdown", so close it explicitly while the
        pool is still alive - and only once the worker is out of detect().
        """
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self._landmarker.close()
        except Exception:
            pass
