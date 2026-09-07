"""Mixing a video file into the stream.

Two keying modes cover almost all stock footage:

  luma    the clip is bright material on black (fireworks, sparks, smoke). The
          brightness *is* the alpha, and it composites additively, so nothing
          has to be cut out by hand.
  chroma  the clip is on a green or blue screen; distance from the key colour
          in chrominance gives the alpha.
  opaque  the clip is ordinary footage with no alpha at all. It covers whatever
          it is composited onto, which is what a background video wants.

Like everything else here the overlay carries a z, so it can sit behind you and
be occluded rather than pasted flat on top. It also carries a layer: "front"
pastes it over the finished frame, "back" mixes it into the background before
you are composited on top, so you cover it.

The same decoder backs `open_background`, which turns an image or a video file
into something the render loop can ask for one frame per iteration.
"""

import queue
import threading

import cv2
import torch
import torch.nn.functional as F


LAYERS = ("front", "back")


def fit_cover(t, height, width):
    """Scale to cover the frame and crop the overflow, never stretching."""
    h, w = t.shape[-2:]
    if (h, w) == (height, width):
        return t
    scale = max(height / h, width / w)
    t = F.interpolate(t, size=(max(height, round(h * scale)), max(width, round(w * scale))),
                      mode="bilinear", align_corners=False)
    top = (t.shape[-2] - height) // 2
    left = (t.shape[-1] - width) // 2
    return t[..., top:top + height, left:left + width]


class VideoOverlay:
    """Decodes on a background thread; the render loop never waits on I/O."""

    def __init__(self, path, device, mode="luma", key=(0.0, 1.0, 0.0),
                 tolerance=0.35, softness=0.12, gain=1.0, loop=True, z=None,
                 layer="front", dtype=torch.float32, reopen=None):
        if layer not in LAYERS:
            raise ValueError(f"layer must be one of {LAYERS}, not {layer!r}")
        self.path = path
        self.layer = layer
        self.dtype = dtype
        self.device = device
        self.mode = mode
        self.tolerance = tolerance
        self.softness = softness
        self.gain = gain
        self.loop = loop
        self.z = z
        # Called for a fresh path when the source runs out. A signed stream URL
        # cannot be rewound to frame 0 the way a file can; it has to be reopened.
        self._reopen = reopen

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
                if self._reopen is None:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                try:
                    fresh = cv2.VideoCapture(self._reopen())
                except Exception:
                    self._queue.put(None)              # the stream is gone for good
                    return
                self._cap.release()
                self._cap = fresh
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
        t = t.flip(1).to(self.dtype) / 255.0              # BGR -> RGB

        if self.mode == "opaque":
            # A background fills the frame, so it is cropped to cover rather than
            # stretched. Keyed overlays keep the old stretch: they are usually
            # framed deliberately and cropping them would cut the effect.
            t = fit_cover(t, height, width)
            return t, torch.ones_like(t[:, :1])

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
        # Releasing the capture while the decode thread is inside read() segfaults
        # OpenCV, so the thread has to be gone first. It only ever blocks in a
        # put() with a timeout, so it notices the flag within half a second.
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._cap.release()


def composite_over(base, layer):
    """layer is (premultiplied colour, alpha)."""
    if layer is None:
        return base
    colour, alpha = layer
    return (base * (1.0 - alpha) + colour).clamp(0, 1)


# ---------------------------------------------------------------- backgrounds


def desktop_module():
    """Imported on use: desktop.py needs fit_cover from here, so a module-level
    import in the other direction would be circular."""
    import desktop
    return desktop


class StillBackground:
    """An image background. Loaded once, handed back unchanged every frame."""

    def __init__(self, image, device, dtype, path=None):
        self.path = path
        t = torch.from_numpy(image).to(device).permute(2, 0, 1)[None]
        self._full = t.flip(1).to(dtype) / 255.0          # BGR -> RGB
        self._cache = None

    def sample(self, height, width):
        if self._cache is None or self._cache.shape[-2:] != (height, width):
            self._cache = fit_cover(self._full, height, width)
        return self._cache

    def close(self):
        pass


class VideoBackground:
    """A video background: one opaque frame per call, looping forever."""

    def __init__(self, source, device, dtype, path=None, reopen=None):
        self.path = path or source
        self._clip = VideoOverlay(source, device, mode="opaque", loop=True,
                                  dtype=dtype, reopen=reopen)

    def sample(self, height, width):
        got = self._clip.sample(height, width)
        return None if got is None else got[0]

    def close(self):
        self._clip.close()


def open_background(path, device, dtype, width=None, height=None, fps=30):
    """Open an image, a video or the screen as a background source.

    The file type is settled by trying to decode it, not by its extension, so a
    .png that is really a clip - or the reverse - still works. A screen grab is
    named rather than decoded: `desktop`, `desktop:HDMI-4`, `desktop:left`,
    `desktop:window:youtube`.
    """
    import streams

    if streams.is_stream_spec(path):
        # The signed URL is short-lived, so the spec is what we remember and
        # what we resolve again each time the stream runs out.
        return VideoBackground(streams.resolve(path), device, dtype, path=path,
                               reopen=lambda: streams.resolve(path))

    if desktop_module().is_desktop_spec(path):
        if not width or not height:
            raise ValueError("a desktop background needs the frame size")
        return desktop_module().DesktopBackground(path, device, dtype,
                                                  width=width, height=height, fps=fps)

    image = cv2.imread(path)
    if image is not None:
        return StillBackground(image, device, dtype, path=path)
    try:
        return VideoBackground(path, device, dtype)
    except FileNotFoundError:
        raise FileNotFoundError(f"cannot open background {path!r} as an image or a video")
