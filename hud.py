"""The panel's HUD widgets, each one a Canvas that draws itself.

ttk can be themed dark but cannot be made to look like an instrument: no glow,
no segmented bars, no chamfered corners. So the controls the user looks at are
drawn here, and the plain ttk widgets that stay (combobox, entry, scrollbar)
are only recoloured by `setup`.

Every clickable widget here still behaves like the control it replaced: it
takes focus with Tab, and Space or Return does what a click does. A control
that is pretty but cannot be reached from the keyboard is a regression.
"""

import math
import time
import tkinter as tk
from tkinter import font as tkfont, ttk

BG = "#05080c"
FIELD = "#0a1419"
CYAN = "#3ef0ff"
DIM = "#0f4a52"
TRACK = "#0d2a30"
AMBER = "#ffb238"
RED = "#ff3b4e"
TEXT = "#bfefff"
SOFT = "#7fb3bd"
MUTED = "#5e8a93"
BRIGHT = "#e6fbff"
OK = "#6effa0"
DEAD = "#333b40"

# Stacked budget segments, in pipeline order. Neighbours differ in hue as well
# as brightness, so two short stages next to each other still read as two.
SEGMENT_COLOURS = [CYAN, OK, AMBER, "#c08bff", "#ff7ac8", "#7f9cff", "#2bb8c6", SOFT]

FAMILY = "DejaVu Sans Mono"
# Design pixels to screen pixels. Xft renders fonts at its own DPI (192 on a
# HiDPI GNOME session) while Tk lays out at 96, so a canvas drawn in plain
# pixels gets text twice the size of the boxes around it. Everything here is
# drawn in design units and scaled by S; setup() measures it from the font.
S = 1.0


def px(n):
    """Design pixels to screen pixels, for geometry outside the canvases."""
    return round(n * S)


_FONTS = {}


def font(size, weight="normal"):
    """A named font, made once per root.

    Named rather than a tuple, so there is one font to keep open (below).
    Negative sizes are pixels: the layout
    is measured in pixels, and a point size would grow with the desktop's
    scaling and break it.
    """
    key = (size, weight)
    if key not in _FONTS:
        _FONTS[key] = tkfont.Font(family=FAMILY, size=-size, weight=weight)
        # Held by a label that is never shown. Tk frees a font when the last
        # widget or item using it goes, so a canvas that deletes all its
        # items would reopen every font through fontconfig on each repaint -
        # some 5 ms a font.
        tk.Label(font=_FONTS[key])
    return _FONTS[key]


def mix(a, b, t):
    """Colour a moved a fraction t towards b. Tk has no alpha; this stands in."""
    t = min(max(t, 0.0), 1.0)
    pa = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    pb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(pa, pb))


def num(value):
    """A finite number, or None for anything else an old or odd aicam sends."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def setup(root):
    """Pick the font and paint the stock widgets dark.

    Done per root rather than in main(): the tests build their own Tk and
    would otherwise get white comboboxes, which is exactly what this avoids.
    """
    global FAMILY, S
    have = set(tkfont.families(root))
    FAMILY = next((f for f in ("Fira Code", "DM Mono", "DejaVu Sans Mono") if f in have),
                  "Courier")
    _FONTS.clear()          # named fonts belong to the root they were made in
    probe = tkfont.Font(root, family=FAMILY, size=-100)
    # A monospace digit is 0.6 em in all three fonts; how wide a 100 px one
    # actually comes out says how much Xft scales.
    S = max(1.0, probe.measure("0") / 60)
    root.configure(bg=BG)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")        # the only stock theme that takes colours
    except tk.TclError:
        pass
    style.configure(".", background=BG, foreground=TEXT, fieldbackground=FIELD,
                    bordercolor=DIM, lightcolor=BG, darkcolor=BG, troughcolor=BG,
                    arrowcolor=CYAN, selectbackground=DIM, selectforeground=BRIGHT,
                    insertcolor=CYAN, font=font(12))
    style.map("TCombobox",
              fieldbackground=[("readonly", FIELD)], foreground=[("readonly", TEXT)],
              selectbackground=[("readonly", FIELD)], selectforeground=[("readonly", CYAN)],
              background=[("active", DIM), ("!active", BG)],
              bordercolor=[("focus", CYAN), ("!focus", DIM)])
    style.map("TEntry", bordercolor=[("focus", CYAN), ("!focus", DIM)])
    style.configure("Vertical.TScrollbar", background=DIM, troughcolor=BG,
                    bordercolor=BG, arrowcolor=CYAN, gripcount=0)
    style.map("Vertical.TScrollbar", background=[("active", "#1a6a75")])
    # The combobox drop-down and the background menus are plain Tk widgets
    # built later; the option database is how they get the colours too.
    for pattern, value in (("*TCombobox*Listbox.background", FIELD),
                           ("*TCombobox*Listbox.foreground", TEXT),
                           ("*TCombobox*Listbox.selectBackground", DIM),
                           ("*TCombobox*Listbox.selectForeground", BRIGHT),
                           ("*Menu.background", FIELD), ("*Menu.foreground", TEXT),
                           ("*Menu.activeBackground", DIM),
                           ("*Menu.activeForeground", BRIGHT),
                           ("*Menu.disabledForeground", MUTED),
                           ("*Menu.relief", "flat")):
        root.option_add(pattern, value)
    root.option_add("*TCombobox*Listbox.font", font(12))
    root.option_add("*Menu.font", font(12))


class Widget(tk.Canvas):
    """A canvas that redraws on resize and can carry options of its own.

    The extra options (OPTIONS) go through configure/cget like real ones, so
    the panel can say `button.configure(text=..., state=...)` and the tests can
    read `button["text"]` as they did when it was a ttk.Button.
    """

    OPTIONS = ()

    def __init__(self, parent, height, focus=False, **options):
        self._opts = options
        super().__init__(parent, height=px(height), width=10, bg=BG, bd=0,
                         highlightthickness=1 if focus else 0,
                         highlightbackground=BG, highlightcolor=CYAN,
                         takefocus=1 if focus else 0)
        self.bind("<Configure>", lambda _e: self.draw())

    def configure(self, cnf=None, **kw):
        kw = {**(cnf or {}), **kw}
        mine = {k: kw.pop(k) for k in list(kw) if k in self.OPTIONS}
        if not kw and not mine:
            return super().configure()          # the query form: all options
        result = super().configure(**kw) if kw else None
        # Redrawn only on a real change: the panel sets these on every poll.
        changed = {k: v for k, v in mine.items() if self._opts.get(k) != v}
        if changed:
            self._opts.update(changed)
            self.draw()
        return result

    config = configure

    def cget(self, key):
        return self._opts[key] if key in self.OPTIONS else super().cget(key)

    __getitem__ = cget

    def size(self):
        """The canvas in design units."""
        return self.winfo_width() / S, self.winfo_height() / S

    def text_width(self, item):
        x0, _y0, x1, _y1 = self.bbox(item)
        return (x1 - x0) / S        # fonts are already at screen size

    def draw(self):
        self.delete("all")
        self.paint()
        self.scale("all", 0, 0, S, S)
        self.tick(time.monotonic())

    def paint(self):
        raise NotImplementedError

    def tick(self, t):
        """Animate between paints; most widgets have nothing to animate."""

    def activate_on(self, action):
        """Click, Space and Return all do `action` - the keyboard half of a button."""
        for sequence in ("<Button-1>", "<space>", "<Return>", "<KP_Enter>"):
            self.bind(sequence, lambda _e: action())


class Button(Widget):
    """A chamfered button; with a glyph it is a reaction tile.

    It flashes when pressed, and `flash()` lets the panel light it for a scene
    that started some other way - a gesture, or aicamctl.
    """

    OPTIONS = ("text", "state", "color")

    def __init__(self, parent, text, command, glyph=None, color=CYAN, height=22, size=11):
        super().__init__(parent, height, focus=True, text=text, state="normal", color=color)
        self.command, self.glyph, self.text_size = command, glyph, size
        self._glow, self._hover = 0.0, False
        self.activate_on(self.invoke)
        self.bind("<Enter>", lambda _e: self._set_hover(True))
        self.bind("<Leave>", lambda _e: self._set_hover(False))

    def _set_hover(self, on):
        self._hover = on
        self.draw()

    def invoke(self):
        if self._opts["state"] != "disabled":
            self.flash()
            self.command()

    def flash(self):
        idle = self._glow == 0
        self._glow = 1.1
        if idle:
            self._fade()

    def _fade(self):
        # Its own short timer, running only while it is lit; the panel's
        # animation loop need not know which tiles are fading.
        self._glow = max(0.0, self._glow - 0.1)
        self.draw()
        if self._glow > 0:
            self.after(50, self._fade)

    def paint(self):
        w, h = self.size()
        if w < 8:
            return
        enabled = self._opts["state"] != "disabled"
        c = self._opts["color"] if enabled else mix(BG, self._opts["color"], 0.45)
        glow = min(self._glow, 1.0)
        fill = mix(BG, c, 0.05 + 0.4 * glow + (0.05 if self._hover and enabled else 0))
        edge = c if (self._hover and enabled) or glow else mix(DIM, c, 0.35)
        k = 7 if h > 30 else 5
        self.create_polygon(k, 1, w - 2, 1, w - 2, h - 2 - k, w - 2 - k, h - 2,
                            1, h - 2, 1, 1 + k, fill=fill, outline=edge)
        text = self._opts["text"]
        if self.glyph:
            self.create_text(w / 2, h * 0.4, text=self.glyph, fill=c, font=font(20))
            self.create_text(w / 2, h * 0.78, text=text, fill=SOFT if enabled else c,
                             font=font(9))
        else:
            self.create_text(w / 2, h / 2, text=text, fill=c,
                             font=font(self.text_size, "bold"))


class Toggle(Widget):
    """A system switch: label on the left, ■ or □ on the right."""

    def __init__(self, parent, text, variable, command):
        super().__init__(parent, 21, focus=True)
        self.text, self.variable, self.command = text, variable, command
        variable.trace_add("write", lambda *_: self.draw())
        self.activate_on(self.toggle)

    def toggle(self):
        self.variable.set(not self.variable.get())
        self.command()

    def paint(self):
        w, h = self.size()
        if w < 8:
            return
        on = bool(self.variable.get())
        self.create_rectangle(1, 1, w - 2, h - 2, outline=CYAN if on else DIM,
                              fill=mix(BG, CYAN, 0.08) if on else BG)
        self.create_text(7, h / 2, anchor="w", text=self.text,
                         fill=CYAN if on else SOFT, font=font(10))
        self.create_text(w - 7, h / 2, anchor="e", text="■" if on else "□",
                         fill=CYAN if on else "#2f4a52", font=font(12))


class Slider(Widget):
    """A segmented bar with the value printed beside it.

    Click or drag sets the value; when focused the arrow keys step it by one
    segment, Home and End jump to the ends. `command` runs only on the user's
    input, never when the panel writes the variable from aicam's state.
    """

    SEGMENTS, LABEL, VALUE = 20, 128, 44

    def __init__(self, parent, text, lo, hi, variable, command):
        super().__init__(parent, 16, focus=True)
        self.text, self.lo, self.hi = text, lo, hi
        self.variable, self.command = variable, command
        self.dragging = False
        variable.trace_add("write", lambda *_: self.draw())
        self.bind("<Button-1>", self._press)
        self.bind("<B1-Motion>", lambda e: self._point(e.x / S))
        self.bind("<ButtonRelease-1>", self._release)
        step = (hi - lo) / self.SEGMENTS
        for keys, delta in ((("<Left>", "<Down>"), -step), (("<Right>", "<Up>"), step),
                            (("<Home>",), lo - hi), (("<End>",), hi - lo)):
            for key in keys:
                self.bind(key, lambda _e, d=delta: self._set(self.variable.get() + d))

    def _press(self, event):
        self.focus_set()
        self.dragging = True
        self._point(event.x / S)

    def _release(self, _event):
        self.dragging = False

    def _track(self):
        return self.LABEL, self.winfo_width() / S - self.VALUE

    def _point(self, x):
        x0, x1 = self._track()
        if x1 > x0:
            self._set(self.lo + (x - x0) / (x1 - x0) * (self.hi - self.lo))

    def _set(self, value):
        value = min(max(value, self.lo), self.hi)
        if abs(value - self.variable.get()) > 1e-9:
            self.variable.set(value)
            self.command()

    def paint(self):
        w, h = self.size()
        x0, x1 = self._track()
        if x1 - x0 < 20:
            return
        value = self.variable.get()
        frac = (value - self.lo) / (self.hi - self.lo)
        lit = round(frac * self.SEGMENTS)
        self.create_text(3, h / 2, anchor="w", text=self.text, fill=SOFT, font=font(10))
        gap = 2
        seg = (x1 - x0 - 6 - gap * (self.SEGMENTS - 1)) / self.SEGMENTS
        for i in range(self.SEGMENTS):
            x = x0 + i * (seg + gap)
            self.create_rectangle(x, h / 2 - 4, x + seg, h / 2 + 3, width=0,
                                  fill=CYAN if i < lit else TRACK)
        shown = f"{value:.0f}" if self.hi - self.lo >= 10 else f"{value:.2f}"
        self.create_text(w - 4, h / 2, anchor="e", text=shown, fill=BRIGHT, font=font(11))


class Section(Widget):
    """A section title with a rule that fades out to the right."""

    def __init__(self, parent, text):
        super().__init__(parent, 16)
        self.text = text

    def paint(self):
        w, h = self.size()
        item = self.create_text(2, h / 2, anchor="w", text=self.text, fill=MUTED,
                                font=font(10))
        x = 2 + self.text_width(item) + 8
        steps = 10
        for i in range(steps):
            a = x + (w - x) * i / steps
            self.create_line(a, h / 2, a + (w - x) / steps, h / 2,
                             fill=mix(DIM, BG, i / steps))


class Header(Widget):
    """Corner brackets, scanlines, a sweeping band, a pulsing dot and the state.

    The state has to be readable from across the room: is the camera live or
    not. Static parts are drawn on show/resize; `tick` only moves the band and
    recolours the dot, which is what lets it run at 15 fps for next to nothing.
    """

    # colour, tag text, pulse period in seconds (None: steady), blink
    MODES = {
        "LIVE": (RED, "● CAM LIVE", 1.2, False),
        "MUTED": (AMBER, "CAM MUTED", 2.4, False),
        "BOOTING": (AMBER, "BOOTING", 0.5, True),
        "OFFLINE": (DEAD, "OFFLINE", None, False),
        "FAILED": (RED, "FAULT", None, False),
    }
    BAND = 16           # strips in the sweeping band

    def __init__(self, parent):
        super().__init__(parent, 64)
        self.mode, self.info = "OFFLINE", "connecting…"
        self._shutter = 0.0

    def show(self, mode, info):
        if mode != self.mode:
            self.mode, self.info = mode, info
            self.draw()
        elif info != self.info:         # the uptime, every poll: just the text
            self.info = info
            self.itemconfigure("info", text=info)

    def paint(self):
        w, h = self.size()
        if w < 60:
            return
        colour, tag, _period, _blink = self.MODES[self.mode]
        top, bottom = 6, 42
        self.create_rectangle(8, top + 1, w - 8, bottom, width=0, tags=("shutter",),
                              fill=BG)
        for y in range(top + 1, bottom, 3):
            self.create_line(8, y, w - 8, y, fill="#07141a")
        for i in range(self.BAND):
            bell = math.sin(math.pi * (i + 0.5) / self.BAND)
            self.create_rectangle(0, top + 1, 0, bottom - 1, width=0, tags=("band",),
                                  fill=mix(BG, CYAN, 0.16 * bell))
        self.create_rectangle(7, top, w - 8, bottom, outline=DIM)
        for x, y, dx, dy in ((1, 1, 1, 1), (w - 2, 1, -1, 1),
                             (1, h - 2, 1, -1), (w - 2, h - 2, -1, -1)):
            self.create_line(x, y + 12 * dy, x, y, x + 12 * dx, y, fill=CYAN, width=2 * S)
        cy = (top + bottom) / 2
        self.create_oval(14, cy - 9, 32, cy + 9, width=0, tags=("glow",),
                         fill=mix(BG, colour, 0.2))
        self.create_oval(18, cy - 5, 28, cy + 5, width=0, tags=("dot",), fill=colour)
        self.create_text(40, cy, anchor="w", text="AICAM", fill=CYAN, font=font(20, "bold"))
        tag_colour = MUTED if self.mode == "OFFLINE" else colour
        label = self.create_text(w - 20, cy, anchor="e", text=tag, fill=tag_colour,
                                 font=font(12, "bold"))
        x0 = w - 20 - self.text_width(label)
        self.create_rectangle(x0 - 8, cy - 11, w - 14, cy + 11, outline=tag_colour)
        self.create_text(9, bottom + 12, anchor="w", text=self.info, fill=MUTED, tags=("info",),
                         font=font(10))

    def shutter(self):
        """A white flash that fades over half a second: the picture was taken."""
        self._shutter = 1.0

    def tick(self, t):
        w, _h = self.size()
        if w < 60:
            return
        if self._shutter > 0:
            self._shutter = max(0.0, self._shutter - 0.12)
            self.itemconfigure("shutter", fill=mix(BG, "#ffffff", 0.85 * self._shutter))
        # The band crosses the box in three seconds, and is clipped by hand to
        # the box: a canvas cannot clip, and a band poking out looks broken.
        left, right = 8, w - 8
        width = (right - left) * 0.4
        start = left - width + (t % 3.0) / 3.0 * (right - left + width)
        strip = width / self.BAND
        for i, item in enumerate(self.find_withtag("band")):
            a = max(start + i * strip, left)
            b = min(start + (i + 1) * strip, right)
            self.coords(item, a * S, 7 * S, max(a, b) * S, 41 * S)

        colour, _tag, period, blink = self.MODES[self.mode]
        if period is None:
            level = 1.0 if self.mode != "OFFLINE" else 0.0
        elif blink:
            level = 1.0 if (t % period) < period / 2 else 0.15
        else:
            level = 0.5 + 0.5 * math.cos(2 * math.pi * t / period)
        self.itemconfigure("dot", fill=mix(mix(BG, colour, 0.35), colour, level))
        self.itemconfigure("glow", fill=mix(BG, colour, 0.28 * level))


def clock(seconds):
    s = int(seconds)
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}"


class Telemetry(Widget):
    """FPS ring and history, hands, particles, GPU, and the frame budget.

    Redrawn whole on every poll (twice a second); a few dozen canvas items is
    nothing next to keeping each one's state in sync. Anything aicam does not
    send - an older aicam, no nvidia-smi - shows as a dash.
    """

    def __init__(self, parent):
        super().__init__(parent, 250)
        self.state, self.history = None, []

    def show(self, state, history):
        history = list(history)
        if (state, history) != (self.state, self.history):     # offline: no change
            self.state, self.history = state, history
            self.draw()

    def _stat(self, x, y, key):
        self.create_line(x, y, x, y + 34, fill=CYAN, width=2 * S)
        self.create_text(x + 7, y + 5, anchor="w", text=key, fill=MUTED, font=font(9))

    def _bar(self, y, label, frac, colour, value, under=None):
        """One GPU row: label, a thin bar, the value. `under` is a second,
        dimmer fill drawn first - the whole of VRAM in use, behind aicam's."""
        w, _h = self.size()
        x0, x1 = 50, w - 128
        self.create_text(3, y, anchor="w", text=label, fill=SOFT, font=font(10))
        self.create_rectangle(x0, y - 3, x1, y + 4, width=0, fill=TRACK)
        for part, fill in ((under, "#1b6d77"), (frac, colour)):
            if part is not None:
                part = min(max(part, 0.0), 1.0)
                self.create_rectangle(x0, y - 3, x0 + (x1 - x0) * part, y + 4,
                                      width=0, fill=fill)
        self.create_text(w - 3, y, anchor="e", text=value, fill=BRIGHT, font=font(10))

    def paint(self):
        w, _h = self.size()
        if w < 200:
            return
        s = self.state or {}

        # FPS ring against 30.
        fps = num(s.get("fps"))
        cx, cy, r = 50, 52, 42
        ring = CYAN if fps is None or fps >= 24 else AMBER if fps >= 15 else RED
        self.create_oval(cx - r, cy - r, cx + r, cy + r, outline=TRACK, width=7 * S)
        if fps:
            self.create_arc(cx - r, cy - r, cx + r, cy + r, start=90, style="arc",
                            extent=-359.9 * min(fps / 30, 1), outline=ring, width=7 * S)
        self.create_oval(cx - r + 9, cy - r + 9, cx + r - 9, cy + r - 9,
                         outline=TRACK, dash=(2, 4))
        self.create_text(cx, cy - 3, text=f"{fps:.1f}" if fps is not None else "—",
                         fill=ring, font=font(22, "bold"))
        self.create_text(cx, cy + 17, text="FPS", fill=MUTED, font=font(9))

        # History, hands, particles.
        x0 = 108
        self._stat(x0, 6, "FPS HISTORY")
        top, bottom = 18, 40
        history = [v for v in (num(v) for v in self.history) if v is not None]
        # From a little under the worst sample, not from zero: a dip from 30
        # to 27 is the thing worth seeing, and from zero it is a flat line.
        floor = max(0.0, min(history + [30.0]) - 4)
        scale = max([32.0] + history) - floor
        y30 = bottom - (bottom - top) * (30 - floor) / scale
        self.create_line(x0 + 8, y30, w - 3, y30, fill=TRACK, dash=(2, 3))
        if len(history) > 1:
            step = (w - 3 - x0 - 8) / 49
            points = []
            for i, v in enumerate(history[-50:]):
                points += [x0 + 8 + i * step, bottom - (bottom - top) * (v - floor) / scale]
            self.create_line(*points, fill=CYAN, width=1.5 * S)
        else:
            self.create_text(x0 + 8, (top + bottom) / 2, anchor="w", text="—",
                             fill=BRIGHT, font=font(12))

        self._stat(x0, 52, "HANDS")
        hands = num(s.get("hands"))
        for i in range(2):
            lit = hands is not None and hands > i
            self.create_text(x0 + 8 + i * 30, 72, anchor="w", text="✋",
                             fill=CYAN if lit else "#12303a", font=font(22))
        pcol = x0 + max(74, (w - x0) * 0.42)
        self._stat(pcol, 52, "PARTICLES · MODE")
        particles = num(s.get("particles"))
        mode = s.get("mode")
        self.create_text(pcol + 8, 72, anchor="w", fill=BRIGHT, font=font(12),
                         text=f"{int(particles) if particles is not None else '—'} · "
                              f"{str(mode).upper() if mode else '—'}")

        # GPU.
        y = 104
        gpu = s.get("gpu") if isinstance(s.get("gpu"), dict) else {}
        name = gpu.get("name") if isinstance(gpu.get("name"), str) else None
        self.create_text(3, y, anchor="w", fill=MUTED, font=font(9),
                         text=f"GPU · {name.upper()}" if name else "GPU · —")
        util, temp, power = num(gpu.get("util")), num(gpu.get("temp")), num(gpu.get("power"))
        used, total, mine = num(gpu.get("mem_used")), num(gpu.get("mem_total")), num(s.get("vram"))
        self._bar(y + 16, "UTIL", util / 100 if util is not None else None, CYAN,
                  f"{util:.0f}%" if util is not None else "—")
        heat = None if temp is None else CYAN if temp < 70 else AMBER if temp < 83 else RED
        self._bar(y + 32, "TEMP", temp / 100 if temp is not None else None, heat,
                  (f"{temp:.0f}°C" if temp is not None else "—") +
                  (f"  {power:.0f} W" if power is not None else ""))
        if total:
            vram = (f"{mine / 1024:.1f} · " if mine is not None else "") + \
                   (f"{used / 1024:.1f}" if used is not None else "—") + f"/{total / 1024:.0f}G"
        else:
            vram = "—"
        self._bar(y + 48, "VRAM", mine / total if total and mine is not None else None,
                  CYAN, vram, under=used / total if total and used is not None else None)
        self.create_text(50, y + 60, anchor="w", fill=MUTED, font=font(8),
                         text="aicam · all in use / total" if total else "")

        # Frame budget: one stacked bar against the 33.3 ms of a 30 fps frame.
        # `capture` includes waiting for the camera - at 30 fps it soaks up all
        # the slack - so it is drawn hatched as "wait" and left out of the
        # number, which is what aicam itself costs per frame.
        y = 180
        budget = s.get("budget") if isinstance(s.get("budget"), dict) else {}
        stages = [(str(k), v) for k, v in ((k, num(v)) for k, v in budget.items())
                  if k != "total" and v is not None and v >= 0]
        total_ms = num(budget.get("total")) or (sum(v for _k, v in stages) or None)
        wait = dict(stages).get("capture", 0.0)
        cost = total_ms - wait if total_ms is not None else None
        self.create_text(3, y, anchor="w", text="FRAME BUDGET", fill=MUTED, font=font(9))
        over = total_ms is not None and total_ms > 33.4
        self.create_text(w - 3, y, anchor="e", font=font(10), fill=RED if over else BRIGHT,
                         text=f"cost {cost:.1f} / 33.3 ms" if cost is not None
                         else "— / 33.3 ms")
        bar_top, bar_bottom, left, right = y + 10, y + 22, 3, w - 3
        scale = max(40.0, (total_ms or 0) * 1.05)
        self.create_rectangle(left, bar_top, right, bar_bottom, width=0, fill=TRACK)

        def colour(i, key):
            return MUTED if key == "capture" else SEGMENT_COLOURS[i % len(SEGMENT_COLOURS)]

        x = left
        for i, (key, ms) in enumerate(stages):
            end = min(x + (right - left) * ms / scale, right)
            if key == "capture":
                self.create_rectangle(x, bar_top, end, bar_bottom, outline=DIM)
                for hx in range(int(x) - 12, int(end), 4):    # hatching, clipped
                    a, b = max(hx, x), min(hx + 12, end)
                    if b > a:
                        self.create_line(a, bar_bottom - (a - hx), b, bar_bottom - (b - hx),
                                         fill=DIM)
            else:
                self.create_rectangle(x, bar_top, end, bar_bottom, width=0,
                                      fill=colour(i, key))
            x = end
        mark = left + (right - left) * 33.3 / scale
        self.create_line(mark, bar_top - 4, mark, bar_bottom + 4, fill=RED, width=2 * S)

        # Legend, wrapped to the width.
        lx, ly = left, bar_bottom + 12
        for i, (key, ms) in enumerate(stages):
            name = "wait/capture" if key == "capture" else key
            item = self.create_text(lx + 11, ly, anchor="w", text=f"{name} {ms:.1f}",
                                    fill=SOFT, font=font(9))
            width = 11 + self.text_width(item) + 10
            if lx > left and lx + width > right:
                lx, ly = left, ly + 14
                self.coords(item, lx + 11, ly)
            self.create_rectangle(lx, ly - 3, lx + 7, ly + 4, width=0, fill=colour(i, key))
            lx += width
        height = px(ly + 10)
        if int(self["height"]) != height:
            self.configure(height=height)


class BootLog(Widget):
    """The boot sequence: a segmented bar and a [ OK ] / [ ▸▸ ] / [ -- ] log.

    `value` is how many stages are done, the same number the old progress bar
    took. With `fault` set the log turns red and the stage it died in says so.
    """

    OPTIONS = ("value",)

    def __init__(self, parent, labels):
        super().__init__(parent, 30 + 15 * len(labels), value=0)
        self.labels = labels
        self.fault = False

    def paint(self):
        w, h = self.size()
        if w < 60:
            return
        done, n = int(self._opts["value"]), len(self.labels)
        lit = RED if self.fault else AMBER
        title = f"FAULT · STAGE {min(done + 1, n)}/{n}" if self.fault else \
            f"BOOT SEQUENCE {done}/{n}"
        self.create_text(3, 7, anchor="w", text=title, fill=lit if self.fault else MUTED,
                         font=font(9))
        gap, skew = 4, 4
        seg = (w - 6 - skew - gap * (n - 1)) / n
        for i in range(n):
            x = 3 + i * (seg + gap)
            current = i == done and not self.fault
            self.create_polygon(x + skew, 16, x + seg + skew, 16, x + seg, 26, x, 26,
                                fill=lit if i < done else TRACK,
                                tags=("current",) if current else ())
        for i, label in enumerate(self.labels):
            y = 38 + 15 * i
            if i < done:
                mark, colour = "[ OK ]", OK
            elif i == done and self.fault:
                mark, colour = "[FAIL]", RED
            elif i == done:
                mark, colour, label = "[ ▸▸ ]", AMBER, label + "…"
            else:
                mark, colour = "[ -- ]", "#3a4a50"
            self.create_text(3, y, anchor="w", text=f"{mark} {label}", fill=colour,
                             font=font(11))

    def tick(self, t):
        level = 0.5 + 0.5 * math.cos(2 * math.pi * t / 0.5)
        self.itemconfigure("current", fill=mix(TRACK, AMBER, level))


class EventLog(Widget):
    """The last few things aicam did, terminal style, newest at the bottom."""

    ROWS = 5

    def __init__(self, parent):
        super().__init__(parent, 6 + 15 * self.ROWS)
        self.events = None

    def show(self, events):
        if events != self.events:
            self.events = events
            self.draw()

    def paint(self):
        events = [e for e in self.events if isinstance(e, dict)] \
            if isinstance(self.events, list) else None
        if not events:
            self.create_text(3, 10, anchor="w", text="—" if events is None else "quiet",
                             fill=MUTED, font=font(11))
            return
        rows = events[-self.ROWS:]
        for i, event in enumerate(rows):
            y = 10 + 15 * (self.ROWS - len(rows) + i)
            t = num(event.get("t"))
            stamp = time.strftime("%H:%M:%S", time.localtime(t)) if t else "--:--:--"
            newest = i == len(rows) - 1
            self.create_text(3, y, anchor="w", text=stamp, fill=MUTED, font=font(11))
            self.create_text(64, y, anchor="w", text=("▸ " if newest else "  ") +
                             str(event.get("text", "")),
                             fill=CYAN if newest else TEXT, font=font(11))
