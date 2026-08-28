"""Mixing a video file into the stream.

Two keying modes cover almost all stock footage:

  luma    the clip is bright material on black (fireworks, sparks, smoke). The
          brightness *is* the alpha, and it composites additively, so nothing
          has to be cut out by hand.
  chroma  the clip is on a green or blue screen; distance from the key colour
          in chrominance gives the alpha.

Like everything else here the overlay carries a z, so it can sit behind you and
be occluded rather than pasted flat on top.
"""

import queue
import threading

import cv2
import torch
import torch.nn.functional as F


class VideoOverlay:
    """Decodes on a background thread; the render loop never waits on I/O."""

    def __init__(self, path, device, mode="luma", key=(0.0, 1.0, 0.0),
                 tolerance=0.35, softness=0.12, gain=1.0, loop=True, z=None):
        self.path = path
        self.device = device
        self.mode = mode
        self.tolerance = tolerance
        self.softness = softness
        self.gain = gain
        self.loop = loop
        self.z = z

        self.key = torch.tensor(key, device=device).view(1, 3, 1, 1)

        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"cannot open overlay video {path!r}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

        self._queue = queue.Queue(maxsize=4)
        self._stop = threading.Event()
        self.finished = False
        self._current = None

        self._thread = threading.Thread(target=self._decode, daemon=True)
        self._thread.start()

    def _decode(self):
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                if not self.loop:
                    self._queue.put(None)
                    return
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            try:
                self._queue.put(frame, timeout=0.5)
            except queue.Full:
                continue

    def _next_frame(self):
        try:
            frame = self._queue.get_nowait()
        except queue.Empty:
            return self._current
        if frame is None:
            self.finished = True
            return None
        self._current = frame
        return frame

    def sample(self, height, width):
        """Returns (colour [1,3,H,W], alpha [1,1,H,W]) or None when exhausted."""
        frame = self._next_frame()
        if frame is None:
            return None

        t = torch.from_numpy(frame).to(self.device).permute(2, 0, 1)[None]
        t = t.flip(1).float() / 255.0                     # BGR -> RGB
        if t.shape[-2:] != (height, width):
            t = F.interpolate(t, size=(height, width), mode="bilinear", align_corners=False)

        if self.mode == "luma":
            luma = (t * torch.tensor([0.2126, 0.7152, 0.0722],
                                     device=t.device).view(1, 3, 1, 1)).sum(1, keepdim=True)
            alpha = (luma * self.gain).clamp(0, 1)
            return t * alpha, alpha                        # premultiplied, additive look

        # chroma: measure distance from the key in colour only, ignoring brightness,
        # so shadows and highlights on the subject do not punch holes in it.
        def chroma(x):
            m = x.mean(1, keepdim=True).clamp(min=1e-4)
            return x / m

        dist = (chroma(t) - chroma(self.key)).abs().sum(1, keepdim=True)
        alpha = ((dist - self.tolerance) / max(1e-3, self.softness)).clamp(0, 1)
        alpha = (alpha * self.gain).clamp(0, 1)
        return t * alpha, alpha

    def close(self):
        self._stop.set()
        self._cap.release()


def composite_over(base, layer):
    """layer is (premultiplied colour, alpha)."""
    if layer is None:
        return base
    colour, alpha = layer
    return (base * (1.0 - alpha) + colour).clamp(0, 1)
