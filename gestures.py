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


# Fingertip gap as a fraction of hand size, with hysteresis: a pinch that
# toggles on the same number it releases on flickers, and a flickering grab
# drops the window. Measured against a hand resting on a keyboard, which sits
# around 0.4 with the fingers curled - a deliberate pinch has to be tighter
# than an idle one, or the panel pulls itself out of the screen unasked.
PINCH_ON, PINCH_OFF = 0.22, 0.42

# And it has to be held: 3 detections at 10 Hz is a third of a second, short
# enough to feel instant and long enough that a hand passing through the pose
# on its way to the mouse never grabs anything.
PINCH_CONFIRM, PINCH_RELEASE = 3, 2


def hand_scale(lm):
    """Wrist to index knuckle: how big this hand is on screen, so pinch
    thresholds mean the same thing near the camera and far from it."""
    return np.hypot(lm[PIPS["index"]].x - lm[WRIST].x,
                    lm[PIPS["index"]].y - lm[WRIST].y)


def pinch_ratio(lm):
    """Fingertip gap over hand size: the one number a pinch is judged on."""
    thumb, index = lm[TIPS["thumb"]], lm[TIPS["index"]]
    gap = np.hypot(thumb.x - index.x, thumb.y - index.y)
    return gap / max(hand_scale(lm), 1e-3)


def pinch(lm, was_pinching=False, on=None):
    """(x, y, pinching) for one hand: where the fingertips meet, and whether.

    `on` overrides the grab threshold, because no fixed number fits every hand
    and every camera angle; the release threshold follows it.
    """
    on = PINCH_ON if on is None else on
    thumb, index = lm[TIPS["thumb"]], lm[TIPS["index"]]
    limit = on * (PINCH_OFF / PINCH_ON) if was_pinching else on
    return ((thumb.x + index.x) / 2, (thumb.y + index.y) / 2,
            pinch_ratio(lm) < limit)


# Hands wander over the face all day: a chin resting on a fist is a thumbs-up,
# glasses pushed up is a victory, and none of it was meant for the camera. A
# pose shown on purpose is held out clear of the face, so a hand sitting on top
# of the face is dropped before it is classified. The box is grown a little
# because a hand at the chin overlaps the face without its palm being on it.
FACE_MARGIN = 1.25


def on_face(lm, box):
    """Is this hand's palm inside the face box? Box is (x0, y0, x1, y1), normalised."""
    if box is None:
        return False
    x = sum(p.x for p in lm) / len(lm)
    y = sum(p.y for p in lm) / len(lm)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return (abs(x - cx) < (box[2] - box[0]) / 2 * FACE_MARGIN and
            abs(y - cy) < (box[3] - box[1]) / 2 * FACE_MARGIN)


def anchor(hands):
    """The wrist of the first hand: what "held still" is measured against."""
    return (hands[0][WRIST].x, hands[0][WRIST].y) if hands else None


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

    A gesture must be held still on `confirm` consecutive detections before it
    fires, and the same gesture cannot fire again within `cooldown` seconds.

    Held and still are both needed. Scratching your head runs a hand through
    victory, thumbs-up and open-palm on the way, each for a tenth of a second,
    and every one of them used to fire. A deliberate gesture is a pose someone
    stops and shows you: it lasts the better part of a second and the hand sits
    still while it lasts. Waiting for that costs nothing, because the person
    doing it on purpose is already waiting.
    """

    def __init__(self, model_path, downscale=0.5, hold=0.8, cooldown=4.0,
                 interval=0.10, num_hands=2, pinch_on=None, steadiness=0.06,
                 face_guard=True):
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
        # In detections, from the seconds the caller asked for: the interval is
        # this class's business, not the caller's.
        self.confirm = max(round(hold / interval), 1)
        self.cooldown = cooldown
        # How far the wrist may drift, in frame widths, before the pose counts
        # as a movement rather than a gesture.
        self.steadiness = steadiness
        self.interval = interval

        self._latest = None
        self._lock = threading.Lock()
        self._events = []
        self._stop = threading.Event()
        self._streak = (None, 0, None)
        self._last_fired = {}
        self.hands_visible = 0
        # Read straight off the render loop every frame: one tuple assignment is
        # atomic, so this needs no lock and never blocks the pipeline.
        self.pointer = None
        self.pinch_on = pinch_on
        # OpenCV ships the cascade, so this costs no new model file and no GPU.
        self._cascade = None
        if face_guard:
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self._cascade = None if cascade.empty() else cascade
        self.face_box = None
        self._face_countdown = 0
        # The live measurement, so a threshold can be tuned against a real hand
        # instead of guessed: watch it in `aicamctl state` while you pinch.
        self.pinch_gap = None
        self._pinch_streak = 0

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
            self._find_face(small)
            self._track(result.hand_landmarks)
            # Pinching keeps every hand: it has its own hysteresis, and a drag
            # already under way should not die because it crossed the face.
            free = [h for h in result.hand_landmarks if not on_face(h, self.face_box)]
            self._consider(classify_frame(free), anchor(free))
            time.sleep(self.interval)

    def _find_face(self, small):
        """Refresh `face_box` about once a second - a head does not move fast,
        and the cascade costs more than the rest of this thread put together.

        An empty search keeps the previous box on purpose: the commonest reason
        to find no face is a hand held in front of it, which is exactly the case
        the box exists for.
        """
        if self._cascade is None:
            return
        self._face_countdown -= 1
        if self._face_countdown > 0:
            return
        self._face_countdown = round(1.0 / self.interval)

        tiny = cv2.cvtColor(cv2.resize(small, (240, 135)), cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(tiny, 1.2, 5, minSize=(24, 24))
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            th, tw = tiny.shape[:2]
            self.face_box = (x / tw, y / th, (x + w) / tw, (y + h) / th)

    def _track(self, hands):
        """Keep `pointer` current: where the pinch is, and whether it counts yet."""
        if not hands:
            self._pinch_streak = 0
            self.pointer = self.pinch_gap = None
            return

        held = bool(self.pointer and self.pointer[2])
        self.pinch_gap = round(float(pinch_ratio(hands[0])), 2)
        x, y, closed = pinch(hands[0], was_pinching=held, on=self.pinch_on)
        # Streak counts up while closed and down while open, so both the grab
        # and the release need to be meant.
        self._pinch_streak = max(self._pinch_streak + 1, 1) if closed \
            else min(self._pinch_streak - 1, -1)
        if self._pinch_streak >= PINCH_CONFIRM:
            held = True
        elif self._pinch_streak <= -PINCH_RELEASE:
            held = False
        self.pointer = (x, y, held)

    def _consider(self, gesture, where=None):
        name, streak, origin = self._streak
        moved = origin is None or where is None or \
            np.hypot(where[0] - origin[0], where[1] - origin[1]) > self.steadiness
        if gesture != name or moved:
            # A new pose, or the same one carried somewhere else: start over from
            # here rather than counting a hand in transit towards the confirm.
            self._streak = (gesture, 1, where)
            return
        streak += 1
        self._streak = (gesture, streak, origin)

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
