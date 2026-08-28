"""Reaction scenes: what to emit, where, and for how long.

Each scene spreads its particles in z around the subject's own depth, so some
pass in front and some behind. That spread is the whole illusion — a scene where
every particle shares one z looks like a sticker.
"""

import math

import torch

from particles import SHAPE_IDS

CONFETTI_COLOURS = [
    (0.96, 0.26, 0.36), (0.99, 0.76, 0.18), (0.30, 0.78, 0.47),
    (0.25, 0.55, 0.96), (0.79, 0.38, 0.95), (0.99, 0.51, 0.24),
]
HEART_COLOURS = [(0.95, 0.20, 0.36), (0.99, 0.42, 0.55), (0.85, 0.14, 0.42)]
BALLOON_COLOURS = CONFETTI_COLOURS
FIREWORK_COLOURS = [
    (1.0, 0.85, 0.45), (1.0, 0.45, 0.35), (0.55, 0.80, 1.0),
    (0.85, 0.55, 1.0), (0.60, 1.0, 0.70),
]


def _pick(colours, n, device):
    table = torch.tensor(colours, device=device)
    return table[torch.randint(len(colours), (n,), device=device)]


class Scene:
    """A named effect with a fixed lifetime, driven by the main loop."""

    name = "scene"
    duration = 3.0

    def __init__(self, device):
        self.device = device
        self.t = 0.0
        self._emitted = 0.0

    def advance(self, dt):
        self.t += dt
        return self.t < self.duration

    def spawn(self, ps, dt, width, height, focus_z):
        raise NotImplementedError

    def _budget(self, rate, dt):
        """Turn a continuous rate into whole particles without losing fractions."""
        self._emitted += rate * dt
        n = int(self._emitted)
        self._emitted -= n
        return n


class Confetti(Scene):
    name = "confetti"
    duration = 4.0

    def spawn(self, ps, dt, width, height, focus_z):
        n = self._budget(320 if self.t < 1.2 else 0, dt)
        if n <= 0:
            return
        d = self.device
        r = lambda *s: torch.rand(*s, device=d)
        ps.emit(
            n,
            pos=torch.stack([r(n) * width,
                             -r(n) * height * 0.2,
                             focus_z + (r(n) - 0.55) * 0.30], 1),
            vel=torch.stack([(r(n) - 0.5) * 220, 120 + r(n) * 180, torch.zeros(n, device=d)], 1),
            life=2.4 + r(n) * 1.6,
            size=14 + r(n) * 16,
            spin=(r(n) - 0.5) * 14,
            angle=r(n) * 6.28,
            color=_pick(CONFETTI_COLOURS, n, d),
            shape=torch.full((n,), SHAPE_IDS["rect"], device=d),
            gravity=torch.full((n,), 260.0, device=d),
            drag=torch.full((n,), 0.55, device=d),
        )


class Hearts(Scene):
    name = "hearts"
    duration = 4.0

    def spawn(self, ps, dt, width, height, focus_z):
        n = self._budget(26 if self.t < 3.0 else 0, dt)
        if n <= 0:
            return
        d = self.device
        r = lambda *s: torch.rand(*s, device=d)
        ps.emit(
            n,
            pos=torch.stack([(0.2 + r(n) * 0.6) * width,
                             height * (1.02 + r(n) * 0.1),
                             focus_z + (r(n) - 0.5) * 0.26], 1),
            vel=torch.stack([(r(n) - 0.5) * 70, -(150 + r(n) * 130), torch.zeros(n, device=d)], 1),
            life=2.8 + r(n) * 1.4,
            size=26 + r(n) * 30,
            spin=(r(n) - 0.5) * 1.6,
            angle=(r(n) - 0.5) * 0.7,
            color=_pick(HEART_COLOURS, n, d),
            shape=torch.full((n,), SHAPE_IDS["heart"], device=d),
            gravity=torch.full((n,), -22.0, device=d),   # keeps drifting up
            drag=torch.full((n,), 0.85, device=d),
        )


class Fireworks(Scene):
    name = "fireworks"
    duration = 4.5

    def __init__(self, device):
        super().__init__(device)
        self._next = 0.0

    def spawn(self, ps, dt, width, height, focus_z):
        if self.t < self._next or self.t > self.duration - 0.8:
            return
        self._next = self.t + 0.35 + float(torch.rand(1)) * 0.4

        d = self.device
        n = 150
        r = lambda *s: torch.rand(*s, device=d)
        cx = (0.12 + float(torch.rand(1)) * 0.76) * width
        cy = (0.08 + float(torch.rand(1)) * 0.42) * height
        # Bursts sit behind the subject, which is what makes the occlusion read.
        cz = focus_z - 0.10 - float(torch.rand(1)) * 0.18
        colour = _pick(FIREWORK_COLOURS, 1, d).expand(n, 3)

        a = r(n) * 6.283
        speed = 180 + r(n) * 320
        ps.emit(
            n,
            pos=torch.stack([torch.full((n,), cx, device=d),
                             torch.full((n,), cy, device=d),
                             torch.full((n,), cz, device=d)], 1),
            vel=torch.stack([torch.cos(a) * speed, torch.sin(a) * speed,
                             torch.zeros(n, device=d)], 1),
            life=0.9 + r(n) * 0.7,
            size=7 + r(n) * 7,
            spin=torch.zeros(n, device=d),
            angle=torch.zeros(n, device=d),
            color=colour,
            # Embers read as points of light, not as star glyphs.
            shape=torch.full((n,), SHAPE_IDS["disc"], device=d),
            gravity=torch.full((n,), 130.0, device=d),
            drag=torch.full((n,), 1.5, device=d),
            fade=torch.full((n,), 1.5, device=d),        # embers burn brighter
        )


class Balloons(Scene):
    name = "balloons"
    duration = 5.0

    def spawn(self, ps, dt, width, height, focus_z):
        n = self._budget(9 if self.t < 3.2 else 0, dt)
        if n <= 0:
            return
        d = self.device
        r = lambda *s: torch.rand(*s, device=d)
        ps.emit(
            n,
            pos=torch.stack([r(n) * width,
                             height * (1.05 + r(n) * 0.15),
                             focus_z + (r(n) - 0.5) * 0.34], 1),
            vel=torch.stack([(r(n) - 0.5) * 40, -(90 + r(n) * 70), torch.zeros(n, device=d)], 1),
            life=3.6 + r(n) * 1.4,
            size=54 + r(n) * 46,
            spin=(r(n) - 0.5) * 0.5,
            angle=(r(n) - 0.5) * 0.3,
            color=_pick(BALLOON_COLOURS, n, d),
            shape=torch.full((n,), SHAPE_IDS["balloon"], device=d),
            gravity=torch.full((n,), -12.0, device=d),
            drag=torch.full((n,), 0.5, device=d),
        )


class Rain(Scene):
    name = "rain"
    duration = 6.0

    def spawn(self, ps, dt, width, height, focus_z):
        n = self._budget(220 if self.t < 5.0 else 0, dt)
        if n <= 0:
            return
        d = self.device
        r = lambda *s: torch.rand(*s, device=d)
        ps.emit(
            n,
            pos=torch.stack([r(n) * width * 1.2 - width * 0.1,
                             -r(n) * height * 0.15,
                             focus_z + (r(n) - 0.65) * 0.34], 1),
            vel=torch.stack([torch.full((n,), -60.0, device=d),
                             700 + r(n) * 400, torch.zeros(n, device=d)], 1),
            life=1.2 + r(n) * 0.6,
            size=5 + r(n) * 4,
            spin=torch.zeros(n, device=d),
            angle=torch.full((n,), 1.48, device=d),
            color=torch.tensor([[0.62, 0.74, 0.92]], device=d).expand(n, 3),
            shape=torch.full((n,), SHAPE_IDS["rect"], device=d),
            gravity=torch.full((n,), 300.0, device=d),
            drag=torch.zeros(n, device=d),
            fade=torch.full((n,), 0.6, device=d),
        )


SCENES = {c.name: c for c in [Confetti, Hearts, Fireworks, Balloons, Rain]}
