"""What the camera shows while aicam is still loading its models.

The loopback device is opened first and fed from here, so "AI Cam" turns up in
Teams and Meet at once instead of ten seconds later - and the meeting sees a
calm card saying the camera is on its way, not a black rectangle or nothing.

Drawn with numpy and OpenCV on the CPU, from its own thread, at a low rate:
the models are loading on the GPU meanwhile, and this must not slow them down.
The card is drawn once; each frame only redraws the stage line and the bar.
"""

import threading
import time

import cv2
import numpy as np

BG = (14, 18, 24)               # RGB, like the panel's dark HUD
ACCENT = (64, 200, 230)
SOFT = (150, 160, 172)
RATE = 10                       # frames a second; nobody needs more of a card


def _text(img, text, centre_y, scale, colour, thickness):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (img.shape[1] - w) // 2
    cv2.putText(img, text, (x, centre_y + h // 2), font, scale, colour,
                thickness, cv2.LINE_AA)


def card(width, height):
    """The still part: a dark field, a vignette, the name and 'starting'."""
    y, x = np.ogrid[:height, :width]
    r = np.hypot((x - width / 2) / width, (y - height / 2) / height)
    shade = np.clip(1.25 - r * 1.1, 0.55, 1.0)[..., None]
    img = (np.array(BG, np.float32) * shade).astype(np.uint8)
    unit = height / 1080
    _text(img, "AI CAM", int(height * 0.42), 3.2 * unit, ACCENT, max(1, int(7 * unit)))
    _text(img, "camera starting...", int(height * 0.53), 1.3 * unit, SOFT,
          max(1, int(2 * unit)))
    return img


def frame(base, stage, t):
    """The card with this moment's stage line and a bar sweeping under it."""
    img = base.copy()
    height, width = img.shape[:2]
    unit = height / 1080
    if stage:
        _text(img, stage, int(height * 0.60), 0.9 * unit, SOFT, max(1, int(2 * unit)))
    # A sweep, not a percentage: how long CUDA and the hub take is not ours to say.
    bar_w, bar_y = int(width * 0.24), int(height * 0.66)
    left = (width - bar_w) // 2
    thick = max(2, int(4 * unit))
    cv2.rectangle(img, (left, bar_y), (left + bar_w, bar_y + thick),
                  tuple(c // 3 for c in ACCENT), -1)
    span = bar_w // 4
    pos = int((0.5 - 0.5 * np.cos(t * 2.2)) * (bar_w - span))
    cv2.rectangle(img, (left + pos, bar_y), (left + pos + span, bar_y + thick), ACCENT, -1)
    return img


class BootScreen:
    """Feed `sink` (an ffmpeg taking rgb24 on stdin) until stop()."""

    def __init__(self, sink, width, height, rate=RATE):
        self.sink, self.rate = sink, rate
        self.base = card(width, height)
        self.stage = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="bootscreen", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def say(self, stage):
        self.stage = stage

    def stop(self):
        """Stop writing, and return once the last frame is out: from here on
        the pipeline writes to the sink, and two writers would interleave."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()

    def _run(self):
        start = time.monotonic()
        while not self._stop.is_set():
            img = frame(self.base, self.stage, time.monotonic() - start)
            try:
                self.sink.stdin.write(img.tobytes())
                self.sink.stdin.flush()
            except (OSError, ValueError):
                return          # ffmpeg is gone; main() will find out and say so
            self._stop.wait(1 / self.rate)
