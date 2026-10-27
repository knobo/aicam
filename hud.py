"""The panel's HUD widgets, each one a Canvas that draws itself.

ttk can be themed dark but cannot be made to look like an instrument: no glow,
no segmented bars, no chamfered corners. So the controls the user looks at are
drawn here, and the plain ttk widgets that stay (combobox, entry, scrollbar)
are only recoloured by `setup`.

Every clickable widget here still behaves like the control it replaced: it
takes focus with Tab, and Space or Return does what a click does. A control
that is pretty but cannot be reached from the keyboard is a regression.
"""

import colorsys
import json
import math
import os
import subprocess
import time
import tkinter as tk
from tkinter import font as tkfont, ttk

import history

# The palettes. Every colour a widget draws with is one of these, by name, so
# switching theme is: rebind the module globals below, then redraw. A theme is
# a mode, dark or light, and an accent. Dark ice is the HUD as designed; light
# ice is the same instrument in daylight - deep teal where the dark one glows
# cyan, status colours darkened until they read as text on near-white (WCAG AA,
# 4.5:1 or better, against BG).
#
# Within a palette every value is distinct: `apply_theme` recolours the plain
# Tk widgets by looking up which name their old colour had.
ICE = {
    "dark": {
        "BG": "#05080c", "FIELD": "#0a1419", "CYAN": "#3ef0ff", "DIM": "#0f4a52",
        "TRACK": "#0d2a30", "AMBER": "#ffb238", "RED": "#ff3b4e", "TEXT": "#bfefff",
        "SOFT": "#7fb3bd", "MUTED": "#5e8a93", "BRIGHT": "#e6fbff", "OK": "#6effa0",
        "DEAD": "#333b40",
        "OFF": "#2f4a52",           # a toggle's empty box
        "SCAN": "#07141a",          # the header's scanlines
        "SHARE": "#1b6d77",         # all of VRAM in use, behind aicam's share
        "HAND_OFF": "#12303a",      # an unlit hand
        "PENDING": "#3a4a50",       # a boot stage not reached yet
        "ACTIVE": "#1a6a75",        # the scrollbar under the pointer
        "FLASH": "#ffffff",         # the shutter
        "PURPLE": "#c08bff", "PINK": "#ff7ac8", "BLUE": "#7f9cff", "TEAL": "#2bb8c6",
    },
    "light": {
        "BG": "#f2f6f8", "FIELD": "#fcfeff", "CYAN": "#006f7c", "DIM": "#9fc7cd",
        "TRACK": "#d6e4e8", "AMBER": "#955300", "RED": "#c0122c", "TEXT": "#1b3a42",
        "SOFT": "#3c626b", "MUTED": "#4f727a", "BRIGHT": "#06242c", "OK": "#0d7340",
        "DEAD": "#b3bec2",
        "OFF": "#93abb1",
        "SCAN": "#e7eff2",
        "SHARE": "#8fc4cc",
        "HAND_OFF": "#c8d7db",
        "PENDING": "#97a9ae",
        "ACTIVE": "#6fa8b1",
        # White would vanish on BG; a cyan flash still says "taken".
        "FLASH": "#1fc3d4",
        "PURPLE": "#7a43cc", "PINK": "#c2307f", "BLUE": "#3d5ccc", "TEAL": "#178a96",
    },
}
# The names that carry the accent; the status colours (RED, AMBER, OK) and the
# budget segments keep their hue in every theme, so LIVE is red everywhere.
TINTED = ("BG", "FIELD", "CYAN", "DIM", "TRACK", "TEXT", "SOFT", "MUTED", "BRIGHT",
          "DEAD", "OFF", "SCAN", "SHARE", "HAND_OFF", "PENDING", "ACTIVE", "FLASH")
# accent: (hue in degrees, saturation scale, status hues moved out of its way).
# Where a status colour would be the accent's own colour it gets another hue:
# phosphor's OK goes cyan, nostromo's MUTED amber goes yellow.
ACCENTS = {
    "ice": None,
    "phosphor": (125, 1.0, {"OK": 185}),
    "nostromo": (28, 1.0, {"AMBER": 62}),
    "synthwave": (285, 1.0, {}),
    "arctic": (214, 0.45, {}),
}
DEFAULT_ACCENT = "ice"


def luminance(colour):
    """WCAG relative luminance."""
    c = [int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast(a, b):
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


# Read as text, so they are held to 4.5:1 against BG whatever their hue.
TEXT_NAMES = ("CYAN", "TEXT", "SOFT", "MUTED", "BRIGHT", "AMBER", "RED", "OK")


def _rehue(colour, hue, sat=1.0, bg=None, need=4.6):
    """colour turned to another hue, keeping its lightness and saturation.

    Hues differ in luminance at the same lightness - green is brighter than
    cyan, violet darker - so with `bg` given the colour is then moved away from
    the background, just far enough to keep `need` contrast with it.
    """
    _h, light, s = colorsys.rgb_to_hls(*(int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)))

    def at(lightness):
        return "#%02x%02x%02x" % tuple(round(x * 255) for x in
                                       colorsys.hls_to_rgb(hue / 360, lightness, s * sat))
    if bg is None or contrast(at(light), bg) >= need:
        return at(light)
    lo, hi = light, 1.0 if luminance(bg) < 0.5 else 0.0     # lo fails, hi passes
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if contrast(at(mid), bg) >= need else (mid, hi)
    return at(hi)


def _palette(accent, mode):
    base = ICE[mode]
    if ACCENTS[accent] is None:
        return dict(base)
    hue, sat, status = ACCENTS[accent]
    palette = dict(base)
    palette["BG"] = bg = _rehue(base["BG"], hue, sat)
    for k in TINTED[1:]:
        palette[k] = _rehue(base[k], hue, sat, bg if k in TEXT_NAMES else None)
    for k, h in status.items():
        palette[k] = _rehue(base[k], h, 1.0, bg)
    return palette


PALETTES = {(accent, mode): _palette(accent, mode) for accent in ACCENTS for mode in ICE}
MODE, ACCENT = "dark", DEFAULT_ACCENT
globals().update(PALETTES[ACCENT, MODE])


def _segments():
    # Stacked budget segments, in pipeline order. Neighbours differ in hue as
    # well as brightness, so two short stages next to each other still read as two.
    return [CYAN, OK, AMBER, PURPLE, PINK, BLUE, TEAL, SOFT]


SEGMENT_COLOURS = _segments()

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


def config_path():
    """~/.config/aicam/panel.json, beside the background history."""
    return os.path.join(os.path.dirname(history.default_path()), "panel.json")


def saved_theme(path=None):
    """(mode, accent) as last picked; None for a mode never picked.

    {"theme": "light"} is what the first version of the switch wrote.
    """
    try:
        with open(path or config_path()) as f:
            saved = json.load(f)
        mode, accent = saved.get("mode", saved.get("theme")), saved.get("accent")
    except (OSError, ValueError, AttributeError):     # missing, corrupt, not a dict
        mode = accent = None
    return (mode if mode in ICE else None,
            accent if accent in ACCENTS else DEFAULT_ACCENT)


def desktop_theme():
    """What GNOME is set to; dark when it cannot be asked.

    Dark is the fallback because it is what the HUD was designed as. The
    timeout is there because this runs before the window shows.
    """
    try:
        out = subprocess.run(["gsettings", "get", "org.gnome.desktop.interface",
                              "color-scheme"], capture_output=True, text=True,
                             timeout=1).stdout
    except (OSError, subprocess.SubprocessError):
        return "dark"
    return "light" if "prefer-light" in out or "default" in out else "dark"


def initial_theme():
    mode, accent = saved_theme()
    return mode or desktop_theme(), accent


def save_theme(mode, accent, path=None):
    """Remember the user's pick. Never raises: this is a nicety."""
    path = path or config_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"mode": mode, "accent": accent}, f)
    except OSError:
        pass


def setup(root):
    """Pick the font and the theme, and paint the stock widgets to match.

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
    _use(*initial_theme())
    _style(root)


def _use(mode, accent):
    global MODE, ACCENT, SEGMENT_COLOURS
    MODE, ACCENT = mode, accent
    globals().update(PALETTES[accent, mode])
    SEGMENT_COLOURS = _segments()


def apply_theme(root, mode, accent):
    """Switch theme in a running panel, with no restart.

    The canvases draw from the module globals, so they only need a redraw.
    The plain Tk widgets (frames, labels, the combobox drop-down, which Tk
    builds out of sight of Python) hold the colour they were made with; each
    is moved to the same name's colour in the new palette.
    """
    back = {PALETTES[ACCENT, MODE][k]: v for k, v in PALETTES[accent, mode].items()}
    _use(mode, accent)
    _style(root)

    def recolour(path):
        for entry in root.tk.splitlist(root.tk.call(path, "configure")):
            entry = root.tk.splitlist(entry)
            if len(entry) == 5 and str(entry[4]) in back:     # 2 long: an alias
                root.tk.call(path, "configure", entry[0], back[str(entry[4])])
        for child in root.tk.splitlist(root.tk.call("winfo", "children", path)):
            recolour(str(child))

    def redraw(widget):
        if isinstance(widget, Widget):
            if widget._opts.get("color") in back:
                widget._opts["color"] = back[widget._opts["color"]]
            widget.draw()
        for child in widget.winfo_children():
            redraw(child)

    recolour(str(root))
    redraw(root)


def _style(root):
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
    style.map("Vertical.TScrollbar", background=[("active", ACTIVE)])
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

    def __init__(self, parent, text, command, glyph=None, color=None, height=22, size=11):
        # None, not CYAN: a default is fixed when the class is defined, and
        # would stay the dark theme's cyan.
        super().__init__(parent, height, focus=True, text=text, state="normal",
                         color=color or CYAN)
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
                         fill=CYAN if on else OFF, font=font(12))


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
    # The colour by name, looked up when drawn: the theme can change under it.
    MODES = {
        "LIVE": ("RED", "● CAM LIVE", 1.2, False),
        "MUTED": ("AMBER", "CAM MUTED", 2.4, False),
        "BOOTING": ("AMBER", "BOOTING", 0.5, True),
        "OFFLINE": ("DEAD", "OFFLINE", None, False),
        "FAILED": ("RED", "FAULT", None, False),
    }
    BAND = 16           # strips in the sweeping band
    HOLD = 0.5          # seconds on the theme switch that open the picker

    def __init__(self, parent):
        super().__init__(parent, 64)
        self.mode, self.info = "OFFLINE", "connecting…"
        self._shutter = 0.0
        # The theme switch is the glyph at the right of the title row. The
        # header takes focus for it and draws its own focus ring round the
        # glyph: a ring round the whole header would not say what Space does.
        #
        # A click flips dark and light; holding it opens the accent picker,
        # and a ring fills round the glyph meanwhile so the hold is found by
        # accident rather than only by reading the README. Right-click,
        # Shift+Return and the Menu key open the picker at once.
        self.configure(takefocus=1)
        self._focused = False
        self._held = None           # when the switch was pressed, while it is
        self._glyph = (0, 0)
        for sequence in ("<space>", "<Return>", "<KP_Enter>"):
            self.bind(sequence, lambda _e: self.toggle_theme())
        for sequence in ("<Shift-Return>", "<Shift-KP_Enter>", "<KeyPress-Menu>"):
            self.bind(sequence, lambda _e: self.pick_theme())
        self.tag_bind("theme", "<ButtonPress-1>", lambda _e: self.press())
        self.tag_bind("theme", "<Button-3>", lambda _e: self.pick_theme())
        # On the canvas, not the glyph: a release a few pixels off it still
        # ends the press.
        self.bind("<ButtonRelease-1>", lambda _e: self.release())
        self.bind("<FocusIn>", lambda _e: self._set_focus(True))
        self.bind("<FocusOut>", lambda _e: self._set_focus(False))

    def _set_focus(self, on):
        self._focused = on
        self.draw()

    def press(self):
        self._held = time.monotonic()

    def release(self):
        """A release before the hold ran out was a click."""
        if self._held is not None:
            self._held = None
            self.toggle_theme()

    def toggle_theme(self):
        self.set_theme("light" if MODE == "dark" else "dark", ACCENT)

    def set_theme(self, mode, accent):
        apply_theme(self.winfo_toplevel(), mode, accent)
        save_theme(mode, accent)

    def pick_theme(self):
        """The accents as a menu under the switch, the current one ticked."""
        self._held = None
        self.itemconfigure("hold", state="hidden")
        # Made fresh each time, so it takes the theme's colours from the
        # option database.
        menu = tk.Menu(self, tearoff=0)
        # ■ and □ as on the system switches; Tk's radio dot is a speck at this size.
        for name in ACCENTS:
            menu.add_command(label=("■ " if name == ACCENT else "□ ") + name,
                             command=lambda n=name: self.set_theme(MODE, n))
        menu.update_idletasks()         # or it has no width yet to align by
        # Right-aligned under the switch, which sits at the window's edge.
        gx, cy = self._glyph
        menu.tk_popup(self.winfo_rootx() + round((gx + 11) * S) - menu.winfo_reqwidth(),
                      self.winfo_rooty() + round((cy + 14) * S))

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
        name, tag, _period, _blink = self.MODES[self.mode]
        colour = globals()[name]
        top, bottom = 6, 42
        self.create_rectangle(8, top + 1, w - 8, bottom, width=0, tags=("shutter",),
                              fill=BG)
        for y in range(top + 1, bottom, 3):
            self.create_line(8, y, w - 8, y, fill=SCAN)
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
        label = self.create_text(w - 50, cy, anchor="e", text=tag, fill=tag_colour,
                                 font=font(12, "bold"))
        x0 = w - 50 - self.text_width(label)
        self.create_rectangle(x0 - 8, cy - 11, w - 44, cy + 11, outline=tag_colour)
        # The theme switch shows where a press takes you: the sun in the dark.
        gx = w - 26
        self.create_rectangle(gx - 11, cy - 11, gx + 11, cy + 11, fill=BG, tags=("theme",),
                              outline=CYAN if self._focused else DIM,
                              width=2 * S if self._focused else 1)
        self.create_text(gx, cy, text="☀" if MODE == "dark" else "☾", fill=CYAN,
                         font=font(14), tags=("theme",))
        self.create_arc(gx - 15, cy - 15, gx + 15, cy + 15, style="arc", start=90, extent=0,
                        outline=CYAN, width=2 * S, state="hidden", tags=("hold",))
        self._glyph = (gx, cy)
        self.create_text(9, bottom + 12, anchor="w", text=self.info, fill=MUTED, tags=("info",),
                         font=font(10))

    def shutter(self):
        """A white flash that fades over half a second: the picture was taken."""
        self._shutter = 1.0

    def tick(self, t):
        if self._held is not None:
            held = (t - self._held) / self.HOLD
            if held >= 1:
                self.pick_theme()
            else:
                self.itemconfigure("hold", state="normal", extent=-359.9 * max(held, 0.01))
        w, _h = self.size()
        if w < 60:
            return
        if self._shutter > 0:
            self._shutter = max(0.0, self._shutter - 0.12)
            self.itemconfigure("shutter", fill=mix(BG, FLASH, 0.85 * self._shutter))
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

        name, _tag, period, blink = self.MODES[self.mode]
        colour = globals()[name]
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
        for part, fill in ((under, SHARE), (frac, colour)):
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
                             fill=CYAN if lit else HAND_OFF, font=font(22))
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
                mark, colour = "[ -- ]", PENDING
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
