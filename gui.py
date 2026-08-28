#!/usr/bin/env python3
"""Tk control panel for a running aicam.

Gestures are convenient when they work and useless when they don't, so every
effect needs a button too. This talks to the same control socket as `aicamctl`,
which means the panel can be opened, closed and reopened without disturbing the
pipeline.
"""

import os
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
            self.send({"overlay": path, "overlay_mode": self.overlay_mode.get()})

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

            hands = state.get("hands", 0)
            self.status.configure(
                text=f"{state.get('fps', 0):.1f} fps   ·   {state.get('particles', 0)} particles"
                     f"   ·   {hands} hand{'' if hands == 1 else 's'}   ·   {state.get('mode', '?')}",
                foreground="#2a7")
        else:
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
