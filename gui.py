#!/usr/bin/env python3
"""Tk control panel for a running aicam.

Gestures are convenient when they work and useless when they don't, so every
effect needs a button too. This talks to the same control socket as `aicamctl`,
which means the panel can be opened, closed and reopened without disturbing the
pipeline.
"""

import collections
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tkinter as tk
from tkinter import filedialog, ttk

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)

import control  # noqa: E402
import hud  # noqa: E402

BACKGROUNDS = os.path.join(HERE, "backgrounds")
IMAGE_OR_VIDEO = ("*.jpg", "*.jpeg", "*.png", "*.webp",
                  "*.mp4", "*.mov", "*.mkv", "*.webm", "*.gif", "*.avi")
VIDEO = ("*.mp4", "*.mov", "*.mkv", "*.webm", "*.gif", "*.avi")

SCENES = ["confetti", "hearts", "fireworks", "balloons", "rain"]
SCENE_GLYPHS = {"confetti": "✦", "hearts": "♥", "fireworks": "✺",
                "balloons": "◍", "rain": "⁞"}
INKS = ["white", "yellow", "pink", "cyan", "green", "orange"]
TOOLS = ["pen", "laser", "line", "rect", "eraser"]
SLIDERS = [
    ("aperture", 0, 90, "Depth of field"),
    ("light", 0, 1, "Key light"),
    ("rim", 0, 1, "Rim light"),
    ("highlights", 0, 2, "Bokeh highlights"),
    ("blur", 0, 100, "Blur (blur mode)"),
    ("parallax", 0, 6, "Parallax (3D)"),
    ("frame_zoom", 1, 2.5, "Auto-frame zoom"),
    ("ink_width", 1, 20, "Ink width"),
]
# What aicam prints on stderr as it starts, in order, so the panel can tell how
# far a launch has come. Lines missing (no gestures) just skip a step.
STAGES = ["loading", "loaded MiDaS", "gestures", "control socket", "Ctrl-C to stop"]
# What the boot log calls each of them.
BOOT_LABELS = ["matting model", "MiDaS depth", "hand tracking", "control socket",
               "camera → loopback"]
# aicam's stderr when the panel starts it. A file, not a pipe: aicam outlives
# the panel, and a pipe nobody reads any more kills it on its next print.
LOG = os.path.join(os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir(), "aicam.log")


class Panel:
    def __init__(self, root, socket_path=None):
        self.root = root
        self.socket_path = socket_path
        self.connected = False
        self._suppress = False
        self.proc = None
        self.launching = False
        self.failed = False
        self.snapshot = None
        self.fps_history = collections.deque(maxlen=50)

        hud.setup(root)
        root.title("aicam")
        root.minsize(hud.px(380), hud.px(300))

        # The header and the buttons below stay put; only the middle scrolls.
        # Whether the camera is live, and the button that changes it, must
        # never be a scroll away. ENGAGE is made first so it is first on Tab.
        foot = tk.Frame(root, bg=hud.BG)
        foot.grid(row=2, column=0, columnspan=2, sticky="ew", padx=hud.px(12),
                  pady=(hud.px(4), hud.px(8)))
        foot.columnconfigure(1, weight=1)
        self.run_button = hud.Button(foot, "▶ ENGAGE", self.toggle_run,
                                     height=36, size=15)
        self.run_button.grid(row=1, column=1, sticky="ew")
        self.capture_button = hud.Button(foot, "◉ CAPTURE", self.capture,
                                         color=hud.BRIGHT, height=36, size=13)
        self.capture_button.configure(width=hud.px(130), state="disabled")
        self.capture_button.grid(row=1, column=0, sticky="ew", padx=(0, hud.px(6)))
        self.snapshot_link = hud.Button(foot, "", self.open_snapshot, color=hud.SOFT,
                                        height=20, size=10)
        self.snapshot_link.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        self.snapshot_link.grid_remove()

        self.header = hud.Header(root)
        self.header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=hud.px(8),
                         pady=(hud.px(8), 0))

        body = self._scrolling_body(root)
        body.columnconfigure(0, weight=1)
        self._row = 0

        # Telemetry and the boot sequence share a row: one or the other.
        self.telemetry = hud.Telemetry(body)
        self.telemetry.grid(row=0, column=0, sticky="ew", pady=(4, 0))
        self.boot = tk.Frame(body, bg=hud.BG)
        self.boot.grid(row=0, column=0, sticky="ew", pady=(4, 0))
        self.boot.columnconfigure(0, weight=1)
        self.progress = hud.BootLog(self.boot, BOOT_LABELS)
        self.progress.grid(row=0, column=0, sticky="ew")
        # aicam's own last line under the log, and on failure its last words.
        self.stage = tk.Label(self.boot, bg=hud.BG, fg=hud.MUTED, font=hud.font(11),
                              justify="left", anchor="w", wraplength=340)
        self.stage.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        self.boot.bind("<Configure>",
                       lambda e: self.stage.configure(wraplength=max(e.width - 8, 100)))
        self.boot.grid_remove()
        self._row = 1

        box = self._section(body, "REACTIONS")
        self.tiles = {}
        for i, name in enumerate(SCENES):
            tile = hud.Button(box, name.upper(), lambda n=name: self.trigger(n),
                              glyph=SCENE_GLYPHS[name], height=46)
            tile.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 4, 0))
            box.columnconfigure(i, weight=1, uniform="tile")
            self.tiles[name] = tile
        hud.Button(box, "✕ CLEAR ALL", self.clear, color=hud.AMBER).grid(
            row=1, column=0, columnspan=len(SCENES), sticky="ew", pady=(4, 0))

        box = self._section(body, "SYSTEMS")
        self.preview = tk.BooleanVar()
        self.muted = tk.BooleanVar()
        self.mirror = tk.BooleanVar()
        self.gestures = tk.BooleanVar()
        self.auto_frame = tk.BooleanVar()
        self.draw = tk.BooleanVar()
        for i, (text, key, var) in enumerate([
                ("PREVIEW", "preview", self.preview), ("MUTE", "mute", self.muted),
                ("MIRROR", "mirror", self.mirror), ("GESTURES", "gestures", self.gestures),
                ("AUTO-FRAME", "auto_frame", self.auto_frame),
                ("PINCH DRAW", "draw", self.draw)]):
            hud.Toggle(box, text, var, lambda k=key, v=var: self.send({k: v.get()})).grid(
                row=i // 3, column=i % 3, sticky="ew",
                padx=(0 if i % 3 == 0 else 4, 0), pady=(0, 3))
            box.columnconfigure(i % 3, weight=1, uniform="toggle")

        box = self._section(body, "DRAW IN THE AIR")
        box.columnconfigure((1, 3), weight=1)
        hud.Button(box, "UNDO", lambda: self.send({"undo": True})).grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=(0, 2), pady=(0, 4))
        hud.Button(box, "ERASE ALL", lambda: self.send({"erase": True})).grid(
            row=0, column=2, columnspan=2, sticky="ew", padx=(2, 0), pady=(0, 4))
        self.tool = tk.StringVar(value="pen")
        self.ink = tk.StringVar(value="yellow")
        self._label(box, "TOOL").grid(row=1, column=0, sticky="w", padx=(0, 6))
        self.tool_box = tools = self._combo(box, self.tool, TOOLS)
        tools.grid(row=1, column=1, sticky="ew", padx=(0, 8))
        tools.bind("<<ComboboxSelected>>",
                   lambda _e: self.send({"tool": self.tool.get()}))
        self._label(box, "INK").grid(row=1, column=2, sticky="w", padx=(0, 6))
        self.ink_box = inks = self._combo(box, self.ink, INKS)
        inks.grid(row=1, column=3, sticky="ew")
        inks.bind("<<ComboboxSelected>>",
                  lambda _e: self.send({"ink": self.ink.get()}))

        box = self._section(body, "OPTICS")
        box.columnconfigure(0, weight=1)
        self.vars = {}
        self.sliders = {}
        for i, (key, lo, hi, label) in enumerate(SLIDERS):
            var = tk.DoubleVar()
            self.vars[key] = var
            self.sliders[key] = hud.Slider(box, label.upper(), lo, hi, var,
                                           lambda k=key: self.on_slide(k))
            self.sliders[key].grid(row=i, column=0, sticky="ew")

        box = self._section(body, "OVERLAY")
        box.columnconfigure((1, 3), weight=1)
        self.overlay_mode = tk.StringVar(value="luma")
        self._label(box, "MODE").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self._combo(box, self.overlay_mode, ["luma", "chroma"]).grid(
            row=0, column=1, sticky="ew", padx=(0, 8))
        # Sent on its own, so moving a clip between layers does not restart it.
        self.overlay_layer = tk.StringVar(value="front")
        self._label(box, "LAYER").grid(row=0, column=2, sticky="w", padx=(0, 6))
        self.layer_box = layers = self._combo(box, self.overlay_layer, ["front", "back"])
        layers.grid(row=0, column=3, sticky="ew")
        layers.bind("<<ComboboxSelected>>", lambda _e: self.send(
            {"overlay_layer": self.overlay_layer.get()}))
        hud.Button(box, "CHOOSE VIDEO…", self.pick_overlay).grid(
            row=1, column=0, columnspan=3, sticky="ew", padx=(0, 4), pady=(4, 0))
        hud.Button(box, "REMOVE", lambda: self.send({"overlay": None})).grid(
            row=1, column=3, sticky="ew", pady=(4, 0))

        box = self._section(body, "BACKGROUND")
        box.columnconfigure((0, 1, 2), weight=1, uniform="bg")
        self.background_label = tk.Label(box, text="none", bg=hud.BG, fg=hud.TEXT,
                                         font=hud.font(11), anchor="w")
        self.background_label.grid(row=0, column=0, columnspan=3, sticky="ew",
                                   pady=(0, 4))
        for row, buttons in enumerate([
                [("BUNDLED…", self.pick_bundled, 1),
                 ("IMAGE/VIDEO…", self.pick_background, 1),
                 ("REMOVE", lambda: self.send({"background": None}), 1)],
                [("HOLD A WINDOW…", lambda: self.pick_desktop(hold=True), 2),
                 ("DROP", lambda: self.send({"window": None}), 1)],
                [("SCREEN OR WINDOW…", self.pick_desktop, 2),
                 ("RECENT…", self.pick_recent, 1)]], start=1):
            column = 0
            for text, command, span in buttons:
                hud.Button(box, text, command).grid(
                    row=row, column=column, columnspan=span, sticky="ew",
                    padx=(0 if column == 0 else 4, 0), pady=(0, 4))
                column += span
        self._label(box, "STREAM A URL · YouTube or a direct video link").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(1, 1))
        self.url = tk.StringVar()
        entry = ttk.Entry(box, textvariable=self.url, font=hud.font(12))
        entry.grid(row=5, column=0, columnspan=2, sticky="ew", padx=(0, 4))
        entry.bind("<Return>", lambda _e: self.use_url())
        hud.Button(box, "PASTE URL", self.use_url).grid(row=5, column=2, sticky="ew")

        box = self._section(body, "EVENT LOG")
        box.columnconfigure(0, weight=1)
        self.events = hud.EventLog(box)
        self.events.grid(row=0, column=0, sticky="ew")

        # Everything exists by now. From the root down, so the header and the
        # buttons under the scrolling part scroll it too.
        self.wheel_scrolling(root)
        self.refresh()
        self.animate()

    def _section(self, body, title):
        """A titled block in the scrolling body; returns the frame to fill."""
        hud.Section(body, title).grid(row=self._row, column=0, sticky="ew",
                                      pady=(hud.px(7), hud.px(3)))
        box = tk.Frame(body, bg=hud.BG)
        box.grid(row=self._row + 1, column=0, sticky="ew")
        self._row += 2
        return box

    @staticmethod
    def _label(parent, text):
        return tk.Label(parent, text=text, bg=hud.BG, fg=hud.SOFT, font=hud.font(10))

    @staticmethod
    def _combo(parent, variable, values):
        return ttk.Combobox(parent, textvariable=variable, values=values,
                            state="readonly", width=8, font=hud.font(12))

    # ---------------------------------------------------------------- layout

    def _scrolling_body(self, root):
        """Put the controls in a canvas that scrolls when the window is short.

        The panel has grown past the height of a laptop screen, and a control
        you cannot reach is worse than one that is a scroll away. ENGAGE and
        the header sit outside it and never scroll away; the background
        buttons, at the bottom, are what this is for. The scrollbar only appears when there is something to
        scroll to; at full height it would just be a stripe of nothing.
        """
        view = tk.Canvas(root, highlightthickness=0, takefocus=0, bg=hud.BG, height=1)
        bar = ttk.Scrollbar(root, orient="vertical", command=view.yview)
        view.configure(yscrollcommand=bar.set)
        view.grid(row=1, column=0, sticky="nsew", padx=(hud.px(12), 0))
        root.rowconfigure(1, weight=1)
        root.columnconfigure(0, weight=1)

        body = tk.Frame(view, bg=hud.BG, padx=0, pady=0)
        window = view.create_window((0, 0), window=body, anchor="nw")
        self._view, self._bar, self._body = view, bar, body

        def fit(_event=None):
            view.configure(scrollregion=view.bbox("all"))
            needed = body.winfo_reqheight() > view.winfo_height()
            if needed and not bar.winfo_ismapped():
                bar.grid(row=1, column=1, sticky="ns")
            elif not needed and bar.winfo_ismapped():
                bar.grid_remove()
                view.yview_moveto(0)        # or the top stays scrolled away

        # The inner frame is told how wide to be, rather than being left at its
        # requested width: a canvas window does not stretch on its own, and a
        # panel whose buttons stop short of the edge looks broken.
        body.bind("<Configure>", fit)
        # 12 px short of the edge, so the panel keeps its right margin.
        view.bind("<Configure>", lambda e: (view.itemconfigure(window, width=e.width - hud.px(12)),
                                            fit()))

        # Sized once the window manager has placed it: until then there is no
        # telling which monitor it landed on, and on a multi-head session that
        # is the whole question.
        root.bind("<Map>", self._on_map)
        return body

    def _on_map(self, _event=None):
        self.root.unbind("<Map>")
        self.root.after_idle(self._fit_to_screen)

    def wheel_scrolling(self, widget):
        """Bind the wheel on every control, not only on the canvas.

        Binding the canvas and relying on Enter/Leave is the usual trick, but
        the canvas is covered by the controls: the pointer is hardly ever over
        it. Binding each widget also settles what a wheel over a combobox does -
        ttk hands it to the box, so scrolling past the Tool or Ink list would
        quietly change the tool. Here it scrolls the panel and the value stays
        where it was, which is why the handler answers "break".
        """
        for sequence in ("<Button-4>", "<Button-5>", "<MouseWheel>"):
            widget.bind(sequence, self._wheel)
        for child in widget.winfo_children():
            self.wheel_scrolling(child)

    def _wheel(self, event):
        # X11 sends buttons 4 and 5; everything else sends <MouseWheel>.
        if self._bar.winfo_ismapped():
            up = getattr(event, "num", 0) == 4 or getattr(event, "delta", 0) > 0
            self._view.yview_scroll(-2 if up else 2, "units")
        return "break"

    def _fit_to_screen(self):
        """Open tall enough for the whole panel, unless the screen is smaller.

        Asked of the controls, not of the window: a canvas reports its own
        default size rather than the size of what is inside it, so the window
        would open 264 px tall and immediately need scrolling it did not need.
        The fixed header and footer come on top of the scrolling part.
        """
        self.root.update_idletasks()
        wanted = (self.root.winfo_reqheight() - self._view.winfo_reqheight()
                  + self._body.winfo_reqheight())
        room = self._monitor_height() - 120             # panels, title bar, margin
        height = min(wanted, room)
        width = max(self._body.winfo_reqwidth() + hud.px(24), hud.px(420))
        if height < wanted:
            width += self._bar.winfo_reqwidth()         # room for the scrollbar
        self.root.geometry(f"{width}x{height}")

    def _monitor_height(self):
        """The height of the monitor this window is on.

        Not `winfo_screenheight`: on a multi-head X11 session that is the whole
        virtual desktop - 2160 for a 4K screen beside a 1050 one - so the panel
        would open 2040 tall on the small screen and run off the bottom, which
        is the one case the scrolling is here for.
        """
        x = self.root.winfo_x() + self.root.winfo_width() // 2
        y = self.root.winfo_y()
        try:
            import desktop      # cheap: it shells out to xrandr, no torch
            for m in desktop.list_monitors():
                if m.x <= x < m.x + m.width and m.y <= y < m.y + m.height:
                    return m.height
        except Exception:
            pass                # no xrandr, or not X11: the virtual size will do
        return self.root.winfo_screenheight()

    # ---------------------------------------------------------------- actions

    def send(self, message):
        try:
            reply = control.send(message, self.socket_path)
            self.connected = True
            return reply
        # OSError, not only ConnectionError: a stalled aicam times out instead,
        # and an exception here would end the refresh poll for good.
        except OSError:
            self.connected = False
            return None

    def toggle_run(self):
        """Start aicam if nothing answers, otherwise ask it to stop.

        Stopping goes through the socket, never a kill: aicam has to close its
        ffmpeg cleanly or v4l2loopback is left wedged for the next run.
        """
        if self.connected:
            self.send({"quit": True})
            return
        here = os.path.dirname(os.path.realpath(__file__))
        # The venv has mediapipe; the interpreter running the panel may not.
        venv = os.path.join(here, ".venv", "bin", "python")
        cmd = [venv if os.path.exists(venv) else sys.executable,
               os.path.join(here, "aicam.py")]
        if self.socket_path:
            cmd += ["--socket", self.socket_path]
        with open(LOG, "w") as log:
            self.proc = subprocess.Popen(cmd, stderr=log)
        self.launching = True
        self.failed = self.progress.fault = False
        self.progress.configure(value=0)
        self.stage.configure(text="starting Python and CUDA…", foreground=hud.MUTED)
        self._show_boot(True)
        self.run_button.configure(text="◌ SPINNING UP", state="disabled", color=hud.AMBER)
        self.header.show("BOOTING", "starting aicam…")

    def _show_boot(self, on):
        """The boot sequence takes the telemetry's place, and gives it back."""
        (self.boot if on else self.telemetry).grid()
        (self.telemetry if on else self.boot).grid_remove()

    def capture(self):
        """Save the frame the meeting sees; aicam writes it, the header flashes."""
        if self.send({"snapshot": True}) is not None:
            self.header.shutter()

    def open_snapshot(self):
        if self.snapshot and shutil.which("xdg-open"):
            subprocess.Popen(["xdg-open", self.snapshot])

    def _log_lines(self):
        # Text mode reads the fps counter's \r as a line end too.
        try:
            with open(LOG, errors="replace") as f:
                return [line.strip() for line in f if line.strip()]
        except OSError:
            return []

    def animate(self):
        """Move the header's band and pulse its dot, about 15 times a second.

        Only the header and the boot log animate; everything else is redrawn
        by refresh(). Nothing is drawn while the window is minimised, so the
        panel costs nothing during a meeting it is not shown in.
        """
        if self.root.state() not in ("iconic", "withdrawn"):
            t = time.monotonic()
            self.header.tick(t)
            if self.launching:
                self.progress.tick(t)
        self.root.after(66, self.animate)

    def _aicam_lines(self):
        # torch.hub and timm chatter on stderr too; only aicam's own lines
        # say where it is.
        return [line for line in self._log_lines() if line.startswith("aicam:")]

    @staticmethod
    def _stage_reached(lines):
        return max((i + 1 for i, stage in enumerate(STAGES)
                    if any(stage in line for line in lines)), default=0)

    def trigger(self, name):
        self.send({"trigger": name})

    def clear(self):
        self.send({"clear": True})

    def on_slide(self, key):
        if self._suppress:
            return
        self.send({"set": {key: round(self.vars[key].get(), 3)}})

    def ask_file(self, title, kind, patterns, on_pick, initialdir=BACKGROUNDS):
        """Ask for a file with the desktop's own chooser; Tk's if there is none.

        Tk's dialog has no sidebar, no bookmarks, no thumbnails and no recent
        files, and on this desktop it is not the dialog every other application
        shows. zenity (GNOME) and kdialog (KDE) are the ones that put the real
        chooser on screen; both are a hard dependency of their desktop, so in
        practice the fallback is for a bare window manager.

        The chooser runs as a child process polled from the Tk loop instead of
        being waited on, so fps keeps ticking and Start/Stop keeps working while
        it is open.
        """
        chooser = shutil.which("zenity") or shutil.which("kdialog")
        if not chooser:
            path = filedialog.askopenfilename(
                title=title, initialdir=initialdir,
                filetypes=[(kind, " ".join(patterns)), ("All files", "*")])
            if path:
                on_pick(path)
            return

        if chooser.endswith("kdialog"):
            cmd = [chooser, "--title", title, "--getopenfilename",
                   initialdir, f"{' '.join(patterns)}|{kind}"]
        else:
            # The trailing separator is what makes zenity read this as a folder
            # to open in rather than a file to preselect.
            cmd = [chooser, "--file-selection", "--title", title,
                   "--filename", initialdir + os.sep,
                   f"--file-filter={kind} | {' '.join(patterns)}",
                   "--file-filter=All files | *"]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)

        def poll():
            if proc.poll() is None:
                self.root.after(120, poll)
                return
            path = proc.stdout.read().strip()   # empty if it was cancelled
            if path:
                on_pick(path)
        poll()

    def pick_overlay(self):
        # Overlay clips are not project files, so start where the user's are.
        self.ask_file("Overlay video", "Video", VIDEO,
                      initialdir=os.path.expanduser("~"),
                      on_pick=lambda path: self.send({
                          "overlay": path,
                          "overlay_mode": self.overlay_mode.get(),
                          "overlay_layer": self.overlay_layer.get()}))

    def pick_background(self):
        self.ask_file("Background image or video", "Image or video",
                      IMAGE_OR_VIDEO,
                      lambda path: self.send({"background": path}))

    def pick_bundled(self):
        """Whatever is in backgrounds/, one click away.

        A file dialog is the right tool for something elsewhere on disk. For the
        pictures that ship with aicam, or the ones you dropped in the folder
        once, it is three clicks too many.
        """
        menu = tk.Menu(self.root, tearoff=0)
        suffixes = tuple(pattern[1:] for pattern in IMAGE_OR_VIDEO)
        try:
            names = sorted((name for name in os.listdir(BACKGROUNDS)
                            if name.lower().endswith(suffixes)), key=str.lower)
        except OSError:
            names = []

        if names:
            for name in names:
                menu.add_command(
                    label=os.path.splitext(name)[0],
                    command=lambda n=name: self.send(
                        {"background": os.path.join(BACKGROUNDS, n)}))
        else:
            menu.add_command(label="backgrounds/ is empty", state="disabled")

        if shutil.which("xdg-open"):
            menu.add_separator()
            menu.add_command(label="Open the folder…", command=self.open_backgrounds)
        menu.tk_popup(self.root.winfo_pointerx(), self.root.winfo_pointery())

    def open_backgrounds(self):
        os.makedirs(BACKGROUNDS, exist_ok=True)
        subprocess.Popen(["xdg-open", BACKGROUNDS])

    def use_url(self):
        """A YouTube link pasted into the box; anything else a direct media URL."""
        text = self.url.get().strip()
        if not text:
            return
        if self.send({"background": control.spec_for_url(text)}) is not None:
            self.url.set("")

    def pick_recent(self):
        import history        # cheap: plain json, no torch

        menu = tk.Menu(self.root, tearoff=0)
        entries = history.load()
        if not entries:
            menu.add_command(label="nothing used yet", state="disabled")
        for entry in entries:
            menu.add_command(
                label=f"{entry['kind']:8} {history.label(entry['source'])}",
                command=lambda src=entry["source"]: self.send({"background": src}))
        menu.tk_popup(self.root.winfo_pointerx(), self.root.winfo_pointery())

    def pick_desktop(self, hold=False):
        """A menu of monitors and windows, posted where the mouse is.

        `hold` sends the choice to the floating panel instead of the background,
        so the same list serves "put it behind me" and "put it in my hand".
        """
        key = "window" if hold else "background"
        import desktop        # cheap: it keeps torch and cv2 out of the panel

        menu = tk.Menu(self.root, tearoff=0)
        try:
            monitors = desktop.list_monitors()
        except Exception as exc:
            menu.add_command(label=f"no screens: {exc}", state="disabled")
            monitors = []

        if monitors:
            menu.add_command(label="Whole desktop",
                             command=lambda: self.send({key: "desktop"}))
            for m in monitors:
                where = " (left)" if m is monitors[0] else \
                    " (right)" if m is monitors[-1] else ""
                menu.add_command(
                    label=f"{m.name}  {m.width}x{m.height}{where}",
                    command=lambda n=m.name: self.send({key: f"desktop:{n}"}))

        windows = desktop.list_windows()
        if windows:
            menu.add_separator()
            for w in sorted(windows, key=lambda w: w.title.lower())[:25]:
                title = w.title if len(w.title) <= 50 else w.title[:47] + "…"
                # Matched by title, so the grab survives the window being reopened.
                menu.add_command(
                    label=title,
                    command=lambda t=w.title: self.send(
                        {key: f"desktop:window:{t}"}))

        menu.tk_popup(self.root.winfo_pointerx(), self.root.winfo_pointery())

    # ---------------------------------------------------------------- polling

    def refresh(self):
        reply = self.send({"query": "state"})
        if reply and reply.get("ok"):
            state = reply.get("state", {})
            # Do not fight the user: only adopt values while the slider is idle.
            self._suppress = True
            for key, var in self.vars.items():
                value = hud.num(state.get(key))
                if value is not None and not self.sliders[key].dragging \
                        and abs(var.get() - value) > 1e-6:
                    var.set(value)
            self._suppress = False

            # Same rule as the sliders: do not overwrite a box the user is in.
            self.run_button.configure(text="■ DISENGAGE", state="normal", color=hud.RED)
            self.capture_button.configure(state="normal")
            if self.launching or self.failed:
                self.launching = self.failed = False
                self._show_boot(False)
            self.preview.set(bool(state.get("preview")))
            self.muted.set(bool(state.get("muted")))
            self.mirror.set(bool(state.get("mirror")))
            self.gestures.set(bool(state.get("gestures")))
            self.auto_frame.set(bool(state.get("auto_frame")))
            if self.root.focus_get() is not self.layer_box:
                self.overlay_layer.set(state.get("overlay_layer", "front"))
            self.draw.set(bool(state.get("draw")))
            if self.root.focus_get() is not self.ink_box:
                self.ink.set(state.get("ink", "yellow"))
            if self.root.focus_get() is not self.tool_box:
                self.tool.set(state.get("tool", "pen"))
            background = state.get("background")
            self.background_label.configure(
                text=background if background and background.startswith("desktop")
                else os.path.basename(background) if background else "none")

            fps = hud.num(state.get("fps"))
            if fps is not None:
                self.fps_history.append(fps)
            self.header.show("MUTED" if state.get("muted") else "LIVE", self._info(state))
            self.telemetry.show(state, self.fps_history)
            self.events.show(state.get("events"))
            scenes = state.get("scenes")
            for name in scenes if isinstance(scenes, list) else ():
                if name in self.tiles:
                    self.tiles[name].flash()
            snapshot = state.get("last_snapshot")
            if isinstance(snapshot, str) and snapshot != self.snapshot:
                self.snapshot = snapshot
                self.snapshot_link.configure(text=f"▣ {os.path.basename(snapshot)}  ↗")
                self.snapshot_link.grid()
        else:
            self.capture_button.configure(state="disabled")
            alive = self.proc is not None and self.proc.poll() is None
            if alive and self.launching:
                # Still booting while a launch we made is loading its model.
                lines = self._aicam_lines()
                self.progress.configure(value=self._stage_reached(lines))
                if lines:
                    self.stage.configure(text=lines[-1].removeprefix("aicam: "))
                self.header.show("BOOTING", "starting aicam…")
            elif not alive:
                if self.proc is not None:
                    self.launching = False
                    if self.proc.returncode == 0:
                        self._show_boot(False)
                    else:
                        # Its last words say why: no CUDA, wedged loopback, out of
                        # memory. Some reasons take more than one line.
                        # It may have died between two polls; the log still
                        # says how far it got, and so which stage failed.
                        self.failed = self.progress.fault = True
                        self.progress.configure(
                            value=self._stage_reached(self._aicam_lines()))
                        self.stage.configure(
                            text="failed: " + "\n".join(self._log_lines()[-3:]),
                            foreground=hud.RED)
                        self._show_boot(True)
                    self.proc = None
                self.run_button.configure(text="▶ ENGAGE", state="normal", color=hud.CYAN)
                if self.failed:
                    self.header.show("FAILED", "aicam did not start - see below")
                else:
                    self.header.show("OFFLINE", "no aicam running")
                self.telemetry.show(None, ())
        self.root.after(500, self.refresh)

    @staticmethod
    def _info(state):
        """`resnet50 · 1920x1080 → /dev/video10 · up 00:12:31`, dashes for gaps."""
        def text(key):
            value = state.get(key)
            return str(value) if value else "—"
        uptime = hud.num(state.get("uptime"))
        if uptime is None and not any(state.get(k) for k in ("variant", "camera", "output")):
            return "—"           # an aicam from before these fields
        up = hud.clock(uptime) if uptime is not None else "—"
        return f"{text('variant')} · {text('camera')} → {text('output')} · up {up}"


def main():
    import argparse
    p = argparse.ArgumentParser(description="control panel for aicam")
    p.add_argument("--socket", default=None)
    a = p.parse_args()

    # className sets WM_CLASS to ("aicam", "Aicam"); without it Tk says "Tk"
    # and the window matches no desktop entry, so it gets no icon in the dock.
    root = tk.Tk(className="aicam")
    Panel(root, a.socket)
    root.mainloop()


if __name__ == "__main__":
    main()
