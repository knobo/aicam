#!/usr/bin/env python3
"""Tk control panel for a running aicam.

Gestures are convenient when they work and useless when they don't, so every
effect needs a button too. This talks to the same control socket as `aicamctl`,
which means the panel can be opened, closed and reopened without disturbing the
pipeline.
"""

import os
import shutil
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, ttk

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)

import control  # noqa: E402

BACKGROUNDS = os.path.join(HERE, "backgrounds")
IMAGE_OR_VIDEO = ("*.jpg", "*.jpeg", "*.png", "*.webp",
                  "*.mp4", "*.mov", "*.mkv", "*.webm", "*.gif", "*.avi")
VIDEO = ("*.mp4", "*.mov", "*.mkv", "*.webm", "*.gif", "*.avi")

SCENES = ["confetti", "hearts", "fireworks", "balloons", "rain"]
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


class Panel:
    def __init__(self, root, socket_path=None):
        self.root = root
        self.socket_path = socket_path
        self.connected = False
        self._suppress = False

        root.title("aicam")
        root.minsize(380, 220)

        frame = self._scrolling_body(root)
        frame.columnconfigure((0, 1, 2), weight=1)

        row = 0
        self.proc = None
        self.run_button = ttk.Button(frame, text="Start", command=self.toggle_run)
        self.run_button.grid(row=row, column=0, sticky="ew", padx=2, pady=(0, 8))
        self.preview = tk.BooleanVar()
        ttk.Checkbutton(frame, text="Preview window", variable=self.preview,
                        command=lambda: self.send({"preview": self.preview.get()})).grid(
            row=row, column=1, sticky="w", padx=2, pady=(0, 8))
        self.mirror = tk.BooleanVar()
        self.muted = tk.BooleanVar()
        ttk.Checkbutton(frame, text="Mute camera", variable=self.muted,
                        command=lambda: self.send({"mute": self.muted.get()})).grid(
            row=row, column=2, sticky="w", padx=2, pady=(0, 8))
        row += 1
        self.auto_frame = tk.BooleanVar()
        ttk.Checkbutton(frame, text="Auto-frame", variable=self.auto_frame,
                        command=lambda: self.send(
                            {"auto_frame": self.auto_frame.get()})).grid(
            row=row, column=2, sticky="w", padx=2, pady=(0, 8))
        self.gestures = tk.BooleanVar()
        ttk.Checkbutton(frame, text="Gestures", variable=self.gestures,
                        command=lambda: self.send({"gestures": self.gestures.get()})).grid(
            row=row, column=0, sticky="w", padx=2, pady=(0, 8))
        ttk.Checkbutton(frame, text="Mirror me", variable=self.mirror,
                        command=lambda: self.send({"mirror": self.mirror.get()})).grid(
            row=row, column=1, sticky="w", padx=2, pady=(0, 8))
        row += 1

        ttk.Label(frame, text="Reactions", font=("", 11, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1
        for i, name in enumerate(SCENES):
            ttk.Button(frame, text=name.capitalize(),
                       command=lambda n=name: self.trigger(n)).grid(
                row=row + i // 3, column=i % 3, sticky="ew", padx=2, pady=2)
        row += (len(SCENES) + 2) // 3

        ttk.Button(frame, text="Clear all", command=self.clear).grid(
            row=row, column=0, columnspan=3, sticky="ew", padx=2, pady=(6, 10))
        row += 1

        ttk.Separator(frame).grid(row=row, column=0, columnspan=3, sticky="ew", pady=6)
        row += 1
        ttk.Label(frame, text="Draw in the air", font=("", 11, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1

        self.draw = tk.BooleanVar()
        ttk.Checkbutton(frame, text="Pinch to draw", variable=self.draw,
                        command=lambda: self.send({"draw": self.draw.get()})).grid(
            row=row, column=0, sticky="w", padx=2)
        ttk.Button(frame, text="Undo", command=lambda: self.send({"undo": True})).grid(
            row=row, column=1, sticky="ew", padx=2)
        ttk.Button(frame, text="Erase all",
                   command=lambda: self.send({"erase": True})).grid(
            row=row, column=2, sticky="ew", padx=2)
        row += 1

        self.tool = tk.StringVar(value="pen")
        ttk.Label(frame, text="Tool").grid(row=row, column=0, sticky="w")
        self.tool_box = tools = ttk.Combobox(frame, textvariable=self.tool, values=TOOLS,
                                             state="readonly", width=8)
        tools.grid(row=row, column=1, columnspan=2, sticky="ew", padx=2)
        tools.bind("<<ComboboxSelected>>",
                   lambda _e: self.send({"tool": self.tool.get()}))
        row += 1

        self.ink = tk.StringVar(value="yellow")
        ttk.Label(frame, text="Ink").grid(row=row, column=0, sticky="w")
        self.ink_box = inks = ttk.Combobox(frame, textvariable=self.ink, values=INKS,
                                           state="readonly", width=8)
        inks.grid(row=row, column=1, columnspan=2, sticky="ew", padx=2)
        inks.bind("<<ComboboxSelected>>",
                  lambda _e: self.send({"ink": self.ink.get()}))
        row += 1

        ttk.Separator(frame).grid(row=row, column=0, columnspan=3, sticky="ew", pady=6)
        row += 1
        ttk.Label(frame, text="Look", font=("", 11, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1

        self.vars = {}
        for key, lo, hi, label in SLIDERS:
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
            var = tk.DoubleVar()
            self.vars[key] = var
            scale = ttk.Scale(frame, from_=lo, to=hi, variable=var,
                              command=lambda _v, k=key: self.on_slide(k))
            scale.grid(row=row, column=1, columnspan=2, sticky="ew", padx=(6, 0))
            row += 1

        ttk.Separator(frame).grid(row=row, column=0, columnspan=3, sticky="ew", pady=6)
        row += 1
        ttk.Label(frame, text="Overlay", font=("", 11, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1

        self.overlay_mode = tk.StringVar(value="luma")
        ttk.Combobox(frame, textvariable=self.overlay_mode, values=["luma", "chroma"],
                     state="readonly", width=8).grid(row=row, column=0, sticky="ew", padx=2)
        ttk.Button(frame, text="Choose video…", command=self.pick_overlay).grid(
            row=row, column=1, sticky="ew", padx=2)
        ttk.Button(frame, text="Remove", command=lambda: self.send({"overlay": None})).grid(
            row=row, column=2, sticky="ew", padx=2)
        row += 1

        # Sent on its own, so moving a clip between layers does not restart it.
        self.overlay_layer = tk.StringVar(value="front")
        ttk.Label(frame, text="Layer").grid(row=row, column=0, sticky="w")
        self.layer_box = layers = ttk.Combobox(frame, textvariable=self.overlay_layer,
                              values=["front", "back"], state="readonly", width=8)
        layers.grid(row=row, column=1, columnspan=2, sticky="ew", padx=2)
        layers.bind("<<ComboboxSelected>>", lambda _e: self.send(
            {"overlay_layer": self.overlay_layer.get()}))
        row += 1

        ttk.Separator(frame).grid(row=row, column=0, columnspan=3, sticky="ew", pady=6)
        row += 1
        ttk.Label(frame, text="Background", font=("", 11, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1

        self.background_label = ttk.Label(frame, text="none", foreground="#888")
        self.background_label.grid(row=row, column=0, columnspan=3, sticky="w",
                                   pady=(0, 4))
        row += 1
        ttk.Button(frame, text="Bundled…", command=self.pick_bundled).grid(
            row=row, column=0, sticky="ew", padx=2)
        ttk.Button(frame, text="Image/video…", command=self.pick_background).grid(
            row=row, column=1, sticky="ew", padx=2)
        ttk.Button(frame, text="Remove",
                   command=lambda: self.send({"background": None})).grid(
            row=row, column=2, sticky="ew", padx=2)
        row += 1

        ttk.Button(frame, text="Hold a window…",
                   command=lambda: self.pick_desktop(hold=True)).grid(
            row=row, column=0, columnspan=2, sticky="ew", padx=2)
        ttk.Button(frame, text="Drop", command=lambda: self.send({"window": None})).grid(
            row=row, column=2, sticky="ew", padx=2)
        row += 1

        ttk.Button(frame, text="Screen or window…", command=self.pick_desktop).grid(
            row=row, column=0, columnspan=2, sticky="ew", padx=2)
        ttk.Button(frame, text="Recent…", command=self.pick_recent).grid(
            row=row, column=2, sticky="ew", padx=2)
        row += 1

        ttk.Label(frame, text="Stream a URL as the background").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(10, 0))
        row += 1
        ttk.Label(frame, text="a YouTube link, or a direct link to a video file",
                  foreground="#888").grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1

        self.url = tk.StringVar()
        entry = ttk.Entry(frame, textvariable=self.url)
        entry.grid(row=row, column=0, columnspan=2, sticky="ew", padx=2, pady=(4, 0))
        entry.bind("<Return>", lambda _e: self.use_url())
        ttk.Button(frame, text="Paste URL", command=self.use_url).grid(
            row=row, column=2, sticky="ew", padx=2, pady=(4, 0))
        row += 1

        self.status = ttk.Label(frame, text="connecting…", foreground="#888")
        self.status.grid(row=row, column=0, columnspan=3, sticky="w", pady=(12, 0))

        self.wheel_scrolling(self._view)      # everything exists by now
        self.refresh()

    # ---------------------------------------------------------------- layout

    def _scrolling_body(self, root):
        """Put the controls in a canvas that scrolls when the window is short.

        The panel has grown past the height of a laptop screen, and a control
        you cannot reach is worse than one that is a scroll away - particularly
        Start/Stop, which is at the top, and the background buttons, which are
        at the bottom. The scrollbar only appears when there is something to
        scroll to; at full height it would just be a stripe of nothing.
        """
        view = tk.Canvas(root, highlightthickness=0, takefocus=0)
        bar = ttk.Scrollbar(root, orient="vertical", command=view.yview)
        view.configure(yscrollcommand=bar.set)
        view.grid(row=0, column=0, sticky="nsew")
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)

        body = ttk.Frame(view, padding=12)
        window = view.create_window((0, 0), window=body, anchor="nw")
        self._view, self._bar, self._body = view, bar, body

        def fit(_event=None):
            view.configure(scrollregion=view.bbox("all"))
            needed = body.winfo_reqheight() > view.winfo_height()
            if needed and not bar.winfo_ismapped():
                bar.grid(row=0, column=1, sticky="ns")
            elif not needed and bar.winfo_ismapped():
                bar.grid_remove()
                view.yview_moveto(0)        # or the top stays scrolled away

        # The inner frame is told how wide to be, rather than being left at its
        # requested width: a canvas window does not stretch on its own, and a
        # panel whose buttons stop short of the edge looks broken.
        body.bind("<Configure>", fit)
        view.bind("<Configure>", lambda e: (view.itemconfigure(window, width=e.width),
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
        """
        self.root.update_idletasks()
        wanted = self._body.winfo_reqheight()
        room = self._monitor_height() - 120             # panels, title bar, margin
        height = min(wanted, room)
        width = max(self._body.winfo_reqwidth(), 380)
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
        except ConnectionError:
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
        self.proc = subprocess.Popen(cmd)
        self.run_button.configure(text="Starting…", state="disabled")

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
                if key in state and abs(var.get() - float(state[key])) > 1e-6:
                    var.set(float(state[key]))
            self._suppress = False

            # Same rule as the sliders: do not overwrite a box the user is in.
            self.run_button.configure(text="Stop", state="normal")
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

            hands = state.get("hands", 0)
            self.status.configure(
                text=f"{state.get('fps', 0):.1f} fps   ·   {state.get('particles', 0)} particles"
                     f"   ·   {hands} hand{'' if hands == 1 else 's'}   ·   {state.get('mode', '?')}",
                foreground="#2a7")
        else:
            # Still "Starting…" while a launch we made is loading its model.
            if self.proc is None or self.proc.poll() is not None:
                self.proc = None
                self.run_button.configure(text="Start", state="normal")
            self.status.configure(text="no aicam running", foreground="#c33")
        self.root.after(500, self.refresh)


def main():
    import argparse
    p = argparse.ArgumentParser(description="control panel for aicam")
    p.add_argument("--socket", default=None)
    a = p.parse_args()

    # className sets WM_CLASS to ("aicam", "Aicam"); without it Tk says "Tk"
    # and the window matches no desktop entry, so it gets no icon in the dock.
    root = tk.Tk(className="aicam")
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    Panel(root, a.socket)
    root.mainloop()


if __name__ == "__main__":
    main()
