#!/usr/bin/env python3
"""Render the bundled backgrounds.

They are generated, not photographed: nothing to license, a few dozen kilobytes
each, and the point of a call background is to sit still behind a face. A stock
photo of an office competes with you; a gradient does not.

Everything is rendered in float and dithered before it is quantised. Smooth
gradients are exactly the case 8-bit JPEG handles worst - without the dither the
vignettes come out in visible rings, which is the one artefact these images
cannot afford.

    ./make-backgrounds.py            # writes into backgrounds/
"""

import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
OUT = os.path.join(HERE, "backgrounds")
W, H = 1920, 1080

rng = np.random.default_rng(7)          # fixed, so a re-run reproduces the set


def grid():
    """Normalised coordinates, and the radius from centre at 16:9."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    x = x / W - 0.5
    y = y / H - 0.5
    return x, y, np.hypot(x * (W / H), y)


def save(name, rgb):
    """Dither, quantise, write. rgb is float in 0..1, shape (H, W, 3)."""
    noise = rng.uniform(-0.5, 0.5, rgb.shape).astype(np.float32) / 255.0
    out = np.clip(rgb + noise, 0, 1) * 255.0
    path = os.path.join(OUT, name)
    cv2.imwrite(path, out.astype(np.uint8)[:, :, ::-1],
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"{name:24} {os.path.getsize(path) / 1024:5.0f} kB")


def vignette(inner, outer, falloff=1.35):
    """Radial blend between two colours: the studio softbox look."""
    _, _, r = grid()
    t = np.clip(r / 0.72, 0, 1) ** falloff
    return np.array(inner) * (1 - t[..., None]) + np.array(outer) * t[..., None]


def warm_studio():
    return vignette((0.42, 0.39, 0.36), (0.13, 0.115, 0.10))


def dusk():
    """Vertical gradient with a low glow, like a window after sunset."""
    x, y, _ = grid()
    t = np.clip(y + 0.5, 0, 1)[..., None]
    sky = np.array((0.10, 0.12, 0.24)) * (1 - t) + np.array((0.22, 0.17, 0.28)) * t
    # Wide and low, and amber rather than pink: a glow with any magenta in it
    # reads as a lens flare, and a flare behind a face looks like a mistake.
    glow = np.exp(-(((x + 0.22) ** 2) * 3.5 + ((y - 0.36) ** 2) * 11))[..., None]
    return sky + glow * np.array((0.20, 0.12, 0.05))


def bokeh_night():
    """Out-of-focus lights on a dark wall.

    Drawn large and downscaled rather than blurred: a disc with a soft edge is
    what an out-of-focus point light actually looks like, and supersampling gets
    that edge for free.
    """
    scale = 2
    big = np.zeros((H * scale, W * scale, 3), np.float32)
    big += np.array((0.045, 0.05, 0.065))

    palette = [(1.00, 0.78, 0.45), (0.95, 0.62, 0.35),
               (0.55, 0.80, 0.95), (0.85, 0.85, 1.00)]
    for _ in range(46):
        cx = rng.integers(0, W * scale)
        cy = rng.integers(0, H * scale)
        radius = int(rng.integers(26, 150) * scale / 2)
        colour = palette[rng.integers(0, len(palette))]
        alpha = float(rng.uniform(0.10, 0.42))
        disc = np.zeros_like(big)
        cv2.circle(disc, (int(cx), int(cy)), radius, colour, -1, cv2.LINE_AA)
        # A real bokeh disc is brighter at its rim than in the middle.
        cv2.circle(disc, (int(cx), int(cy)), max(radius - 3, 1),
                   tuple(c * 0.82 for c in colour), -1, cv2.LINE_AA)
        big += disc * alpha

    small = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (0, 0), 9)


def paper():
    """A light background, for anyone the dark ones make into a silhouette."""
    base = vignette((0.94, 0.93, 0.90), (0.74, 0.73, 0.71), falloff=1.8)
    _, y, _ = grid()
    return base - (y[..., None] + 0.5) * 0.03


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    save("studio-warm.jpg", warm_studio())
    save("dusk.jpg", dusk())
    save("bokeh-night.jpg", bokeh_night())
    save("paper.jpg", paper())
