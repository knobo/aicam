"""Using the screen itself as a background.

A monitor, a window, or the whole desktop is captured with ffmpeg's x11grab and
read as raw frames, so putting a fullscreen video - a YouTube tab, a slide deck,
anything at all - behind you is a matter of pointing at the screen it plays on.

The capture is scaled by ffmpeg before it reaches the pipe. Raw 4K in bgr24 is
745 MB/s; asking for the render size up front makes it 186 MB/s at 1080p, and
the scaling happens where it is cheap.

X11 only. Wayland hands screen capture to a portal and PipeWire, which is a
different mechanism altogether, so we say that plainly instead of handing back
a black rectangle.
"""

import os
import queue
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass

# torch and overlays are imported inside DesktopBackground rather than here: the
# Tk panel enumerates monitors and windows from this module, and it has no
# business loading CUDA to fill a dropdown.

MONITOR_LINE = re.compile(
    r"^\s*\d+:\s+\+(?P<primary>\*?)(?P<name>\S+)\s+"
    r"(?P<width>\d+)/\d+x(?P<height>\d+)/\d+\+(?P<x>\d+)\+(?P<y>\d+)")


@dataclass
class Monitor:
    name: str
    width: int
    height: int
    x: int
    y: int
    primary: bool = False


@dataclass
class Window:
    id: int
    title: str
    x: int
    y: int
    width: int
    height: int


@dataclass
class Grab:
    """A region of the X display, or one window on it."""
    x: int
    y: int
    width: int
    height: int
    window_id: int = None
    label: str = ""


# ---------------------------------------------------------------- enumeration

def parse_monitors(text):
    """Read `xrandr --listmonitors` output, ordered left to right."""
    found = []
    for line in text.splitlines():
        m = MONITOR_LINE.match(line)
        if m:
            found.append(Monitor(name=m["name"],
                                 width=int(m["width"]), height=int(m["height"]),
                                 x=int(m["x"]), y=int(m["y"]),
                                 primary=bool(m["primary"])))
    if not found:
        raise RuntimeError("no monitors found; is this an X11 session?")
    return sorted(found, key=lambda m: m.x)


def parse_window_geometry(text):
    """Read `xdotool getwindowgeometry --shell` output."""
    values = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
    return {"id": int(values["WINDOW"]), "x": int(values["X"]), "y": int(values["Y"]),
            "width": int(values["WIDTH"]), "height": int(values["HEIGHT"])}


def _run(*args):
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


def list_monitors():
    """The monitors of the running X session, left to right."""
    return parse_monitors(_run("xrandr", "--listmonitors"))


def list_windows():
    """Visible, titled windows of the running X session.

    Windows that vanish between the search and the query are skipped rather
    than raising: the list is a snapshot of something the user is still using.
    """
    if not shutil.which("xdotool"):
        return []
    try:
        ids = _run("xdotool", "search", "--onlyvisible", "--name", ".").split()
    except subprocess.CalledProcessError:
        return []

    windows = []
    for wid in ids:
        try:
            title = _run("xdotool", "getwindowname", wid).strip()
            geometry = parse_window_geometry(
                _run("xdotool", "getwindowgeometry", "--shell", wid))
        except (subprocess.CalledProcessError, KeyError, ValueError):
            continue
        if not title or geometry["width"] < 32 or geometry["height"] < 32:
            continue
        windows.append(Window(id=geometry["id"], title=title,
                              x=geometry["x"], y=geometry["y"],
                              width=geometry["width"], height=geometry["height"]))
    return windows


# ---------------------------------------------------------------- specs

def is_desktop_spec(path):
    """True for `desktop`, `desktop:HDMI-4`, `desktop:window:youtube` and friends."""
    return path == "desktop" or path.startswith("desktop:")


def resolve_spec(spec, monitors, windows):
    """Turn a desktop spec into the region or window to grab.

    Kept free of subprocesses so the whole vocabulary can be tested against a
    fixed set of monitors and windows.
    """
    _, _, rest = spec.partition(":")

    if not rest:
        left = min(m.x for m in monitors)
        top = min(m.y for m in monitors)
        return Grab(x=left, y=top,
                    width=max(m.x + m.width for m in monitors) - left,
                    height=max(m.y + m.height for m in monitors) - top,
                    label="whole desktop")

    if rest == "window" or rest.startswith("window:"):
        wanted = rest.partition(":")[2].strip().lower()
        for window in windows:
            if not wanted or wanted in window.title.lower():
                return Grab(x=window.x, y=window.y,
                            width=window.width, height=window.height,
                            window_id=window.id, label=window.title)
        raise ValueError(f"no visible window matching {wanted!r}")

    if rest.lower() in ("left", "right"):
        monitor = monitors[0] if rest.lower() == "left" else monitors[-1]
    else:
        by_name = {m.name.lower(): m for m in monitors}
        if rest.lower() not in by_name:
            raise ValueError(f"unknown monitor {rest!r}; this session has "
                             f"{', '.join(m.name for m in monitors)}")
        monitor = by_name[rest.lower()]
    return Grab(x=monitor.x, y=monitor.y, width=monitor.width, height=monitor.height,
                label=monitor.name)


def build_command(grab, width, height, fps, display):
    """The ffmpeg invocation that puts scaled raw frames on stdout."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-f", "x11grab", "-framerate", str(fps),
           "-video_size", f"{grab.width}x{grab.height}", "-draw_mouse", "0"]
    origin = f"{grab.x},{grab.y}"
    if grab.window_id is not None:
        # With -window_id the capture area is the window's own coordinate space,
        # so its position on the desktop is not an offset into it: pass the
        # window's origin and ffmpeg refuses with "outside the screen size".
        cmd += ["-window_id", str(grab.window_id)]
        origin = "0,0"
    cmd += ["-i", f"{display}+{origin}",
            "-vf", f"scale={width}:{height}",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    return cmd


# ---------------------------------------------------------------- the source

class DesktopBackground:
    """One ffmpeg screen grab, read on a thread, sampled like any background."""

    def __init__(self, spec, device, dtype, width, height, fps=30, display=None):
        display = display or os.environ.get("DISPLAY")
        if not display:
            session = os.environ.get("XDG_SESSION_TYPE", "unknown")
            raise RuntimeError(
                f"screen capture needs an X display, and this is a {session} session. "
                "On Wayland, capture goes through the desktop portal and PipeWire, "
                "which aicam does not speak yet; log in to an Xorg session, or point "
                "--background at a file instead.")

        self.path = spec
        self.device = device
        self.dtype = dtype
        self.width = width
        self.height = height
        self.fps = fps
        self.display = display

        import torch                                  # see the note at the top

        self.grab = resolve_spec(spec, list_monitors(), list_windows())
        self._torch = torch
        self._frame_bytes = width * height * 3
        self._queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._cache = None

        self._proc = None
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _spawn(self):
        return subprocess.Popen(
            build_command(self.grab, self.width, self.height, self.fps, self.display),
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=self._frame_bytes)

    def _read(self):
        while not self._stop.is_set():
            self._proc = self._spawn()
            while not self._stop.is_set():
                raw = self._proc.stdout.read(self._frame_bytes)
                if len(raw) < self._frame_bytes:
                    break              # the grab died: a resized or closed window
                try:
                    self._queue.put(raw, timeout=0.5)
                except queue.Full:
                    continue
            self._kill()
            if self._stop.is_set():
                return
            # A window that was resized has new geometry; a monitor keeps its own.
            try:
                self.grab = resolve_spec(self.path, list_monitors(), list_windows())
            except (ValueError, RuntimeError):
                return                 # the window is gone for good; keep the last frame

    def _kill(self):
        # close() and the reader thread both end up here, so the process is taken
        # out of the field first: whoever loses the race finds nothing to kill.
        proc, self._proc = self._proc, None
        if proc is None:
            return
        proc.kill()
        proc.wait(timeout=2)
        proc.stdout.close()

    def sample(self, height, width):
        """The most recent screen frame, or None until the first one lands."""
        try:
            raw = self._queue.get_nowait()
        except queue.Empty:
            return self._cache if self._cache is None or \
                self._cache.shape[-2:] == (height, width) else None

        import overlays                               # see the note at the top

        torch = self._torch
        t = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        t = t.view(self.height, self.width, 3).to(self.device)
        t = t.permute(2, 0, 1)[None].flip(1).to(self.dtype) / 255.0     # BGR -> RGB
        self._cache = overlays.fit_cover(t, height, width)
        return self._cache

    def close(self):
        self._stop.set()
        self._kill()
        self._thread.join(timeout=2.0)


class FloatingWindow:
    """One captured window, drawn as a panel you can hold in the frame.

    The same x11grab as `DesktopBackground`, only asked for panel-sized frames
    instead of full-frame ones, so the scaling still happens inside ffmpeg where
    it is cheap. Position is normalised, which keeps it independent of the
    render resolution and lets the hand tracker drive it directly.

    It is composited into the plate, behind the subject: you pass in front of it
    and it reads as held rather than pasted on top of you.
    """

    BORDER = 3

    def __init__(self, spec, device, dtype, frame_width, frame_height,
                 fps=30, scale=0.38):
        grab = resolve_spec(spec, list_monitors(), list_windows())
        self.label = grab.label
        self.width = int(frame_width * scale) // 2 * 2
        self.height = max(int(self.width * grab.height / grab.width) // 2 * 2, 2)
        self.source = DesktopBackground(spec, device, dtype,
                                        self.width, self.height, fps=fps)
        # Off to one side, not dead centre: a panel opened from the panel or the
        # command line should be readable next to you, not behind your head.
        self.x, self.y = 0.72, 0.55
        self.held = False

    def move_to(self, x, y, smoothing=0.7):
        """The tracker runs at 10 Hz and jitters; the panel must not.

        An exponential follow costs nothing and turns a twitching landmark into
        a hand movement. Higher smoothing is calmer and lags further behind.
        """
        self.x += (x - self.x) * (1.0 - smoothing)
        self.y += (y - self.y) * (1.0 - smoothing)

    def draw(self, base):
        """Paste the panel into a [1,3,H,W] plate. Returns a new tensor."""
        frame = self.source.sample(self.height, self.width)
        if frame is None:
            return base

        h, w = base.shape[-2:]
        x0 = min(max(int(self.x * w - self.width / 2), 0), w - self.width)
        y0 = min(max(int(self.y * h - self.height / 2), 0), h - self.height)

        # Never in place: the plate can be a background source's cached tensor,
        # and writing into that would smear the panel across every later frame.
        out = base.clone()
        out[..., y0:y0 + self.height, x0:x0 + self.width] = frame
        b = self.BORDER
        # A bright edge is what makes it read as an object in the room rather
        # than a hole cut in the picture.
        for view in (out[..., y0:y0 + b, x0:x0 + self.width],
                     out[..., y0 + self.height - b:y0 + self.height, x0:x0 + self.width],
                     out[..., y0:y0 + self.height, x0:x0 + b],
                     out[..., y0:y0 + self.height, x0 + self.width - b:x0 + self.width]):
            view.fill_(0.92)
        return out

    def close(self):
        self.source.close()
