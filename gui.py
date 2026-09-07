#!/usr/bin/env python3
"""Tk control panel for a running aicam.

Gestures are convenient when they work and useless when they don't, so every
effect needs a button too. This talks to the same control socket as `aicamctl`,
which means the panel can be opened, closed and reopened without disturbing the
pipeline.
"""

import os
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, ttk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import control  # noqa: E402

SCENES = ["confetti", "hearts", "fireworks", "balloons", "rain"]
SLIDERS = [
    ("aperture", 0, 90, "Depth of field"),
    ("light", 0, 1, "Key light"),
    ("rim", 0, 1, "Rim light"),
    ("highlights", 0, 2, "Bokeh highlights"),
    ("blur", 0, 100, "Blur (blur mode)"),
    ("parallax", 0, 6, "Parallax (3D)"),
    ("frame_zoom", 1, 2.5, "Auto-frame zoom"),
]


class Panel:
    def __init__(self, root, socket_path=None):
        self.root = root
        self.socket_path = socket_path
        self.connected = False
        self._suppress = False

        root.title("aicam")
        root.minsize(380, 0)

        frame = ttk.Frame(root, padding=12)
        frame.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
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
        self.background_label.grid(row=row, column=0, sticky="w")
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

        self.url = tk.StringVar()
        entry = ttk.Entry(frame, textvariable=self.url)
        entry.grid(row=row, column=0, columnspan=2, sticky="ew", padx=2, pady=(4, 0))
        entry.bind("<Return>", lambda _e: self.use_url())
        ttk.Button(frame, text="Paste URL", command=self.use_url).grid(
            row=row, column=2, sticky="ew", padx=2, pady=(4, 0))
        row += 1

        self.status = ttk.Label(frame, text="connecting…", foreground="#888")
        self.status.grid(row=row, column=0, columnspan=3, sticky="w", pady=(12, 0))

        self.refresh()

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
        here = os.path.dirname(os.path.abspath(__file__))
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

    def pick_overlay(self):
        path = filedialog.askopenfilename(
            title="Overlay video",
            filetypes=[("Video", "*.mp4 *.mov *.mkv *.webm *.gif *.avi"), ("All files", "*")])
        if path:
            self.send({"overlay": path,
                       "overlay_mode": self.overlay_mode.get(),
                       "overlay_layer": self.overlay_layer.get()})

    def pick_background(self):
        path = filedialog.askopenfilename(
            title="Background image or video",
            filetypes=[("Image or video", "*.jpg *.jpeg *.png *.webp "
                                          "*.mp4 *.mov *.mkv *.webm *.gif *.avi"),
                       ("All files", "*")])
        if path:
            self.send({"background": path})

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

    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    Panel(root, a.socket)
    root.mainloop()


if __name__ == "__main__":
    main()
