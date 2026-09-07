#!/usr/bin/env python3
"""aicam - a GPU-accelerated virtual webcam for Linux.

Reads a real camera, separates you from the background with RobustVideoMatting
on CUDA, applies depth-aware effects, and writes the result to a v4l2loopback
device that Teams, Meet, Zoom or anything else sees as an ordinary webcam.

Copyright (C) 2026 Knut Boehmer
Licensed under the GNU General Public License v3.0 or later; see LICENSE.
"""

import argparse
import math
import os
import re
import signal
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

# Run via shebang from any directory, so our own modules must be found explicitly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import control  # noqa: E402
import desktop  # noqa: E402
import effects  # noqa: E402
import history  # noqa: E402
import overlays  # noqa: E402
import particles  # noqa: E402
from control import ControlServer  # noqa: E402
from depth import DepthEstimator  # noqa: E402
from scenes import SCENES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TUNABLE = {"aperture", "light", "rim", "highlights", "blur", "keep_near", "downsample",
           "parallax", "parallax_period", "frame_zoom"}


# ---------------------------------------------------------------- input and output

def open_camera(device, width, height, fps):
    # This OpenCV build only accepts an index on the V4L2 backend, not a path.
    m = re.fullmatch(r"/dev/video(\d+)", device)
    source = int(m.group(1)) if m else device
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    if not cap.isOpened():
        # Raised, not exited: muting closes the camera and unmuting reopens it,
        # and losing that race to another app must not take the pipeline down.
        raise RuntimeError(f"could not open {device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    # Do NOT set CAP_PROP_BUFFERSIZE=1 here. It looks like a latency win, but a
    # one-frame queue makes the consumer miss every other frame and halves the
    # rate to exactly 15 fps. Measured: default and 4 give 29.9, 1 gives 15.0.
    # The GPU drains the queue anyway (~140 fps), so the default costs nothing.
    return cap, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def open_sink(device, width, height, fps):
    """ffmpeg takes raw RGB on stdin and negotiates the v4l2 format for us."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-f", "v4l2", "-pix_fmt", "yuv420p", device,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def close_pipe(proc, timeout=5):
    """Shut an ffmpeg down gently.

    A killed ffmpeg leaves v4l2loopback wedged (VIDIOC_G_FMT: Invalid argument
    on the next run), so let it see EOF, then SIGTERM, and only then give up.
    """
    if proc is None or proc.poll() is not None:
        return
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    except OSError:
        pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def open_preview(width, height, fps):
    """Preview through ffplay.

    cv2.imshow is not an option: opencv-python-headless is built with GUI: NONE
    and raises "function is not implemented" regardless of what you do.
    """
    cmd = [
        "ffplay", "-hide_banner", "-loglevel", "error",
        "-fflags", "nobuffer", "-flags", "low_delay",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{width}x{height}", "-framerate", str(fps),
        "-window_title", "aicam", "-i", "-",
    ]
    try:
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)
    except FileNotFoundError:
        # Toggled live as well as at startup, so a missing ffplay must not be fatal.
        print("aicam: preview needs ffplay (apt install ffmpeg)", file=sys.stderr)
        return None


# ------------------------------------------------------- live control messages

def apply_overlay_message(msg, overlay, device):
    """Apply an overlay / overlay_layer message and return the overlay to use.

    A layer on its own moves the clip that is already playing, keeping the same
    object so the footage does not jump back to its first frame.
    """
    if "overlay" not in msg:
        if "overlay_layer" in msg and overlay is not None:
            layer = msg["overlay_layer"]
            if layer not in overlays.LAYERS:
                raise ValueError(f"unknown overlay layer {layer!r}")
            overlay.layer = layer
        return overlay

    fresh = None
    if msg["overlay"]:
        # Opened before the old one is closed, so a bad path leaves you with the
        # clip you already had rather than nothing at all.
        fresh = overlays.VideoOverlay(msg["overlay"], device,
                                      mode=msg.get("overlay_mode", "luma"),
                                      layer=msg.get("overlay_layer", "front"))
    if overlay is not None:
        overlay.close()
    return fresh


def apply_background_message(msg, background, device, dtype,
                             width=None, height=None, fps=30, history_path=None):
    """Apply a background message and return the background source to use.

    The frame size travels with the message because a screen grab has to be
    scaled to it by ffmpeg rather than by us.
    """
    fresh = overlays.open_background(msg["background"], device, dtype,
                                     width=width, height=height, fps=fps) \
        if msg["background"] else None
    if fresh is not None:
        # Only after it opened: a typo never reaches the recent list.
        history.remember(msg["background"], history_path)
    if background is not None:
        background.close()
    return fresh


def open_panel(spec, panel, device, dtype, width, height, fps):
    """Pull a window out of the screen and into the frame, or drop the one held.

    Reopening the same window is a no-op rather than a restart: the pinch that
    grabs a panel already on screen must not blink it.
    """
    if panel is not None and (spec is None or spec == panel.source.path):
        if spec is None:
            panel.close()
            return None
        return panel
    if panel is not None:
        panel.close()
    if spec is None:
        return None
    return desktop.FloatingWindow(spec, device, dtype, width, height, fps=fps)


# ---------------------------------------------------------------- arguments

def build_parser():
    p = argparse.ArgumentParser(description="GPU-accelerated virtual webcam")
    p.add_argument("--input", default="/dev/video0")
    p.add_argument("--output", default="/dev/video10")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=30)

    p.add_argument("--mode", default="blur", choices=["blur", "cinematic"],
                   help="blur = uniform defocus; cinematic = depth-graded bokeh and studio light")
    p.add_argument("--background", metavar="IMAGE|VIDEO|desktop",
                   help="replace the background with an image, a video file, or the "
                        "screen: desktop, desktop:HDMI-4, desktop:left, desktop:right, "
                        "desktop:window:<title>")
    p.add_argument("--blur", type=float, default=40.0, help="blur mode: background defocus")

    p.add_argument("--aperture", type=float, default=45.0,
                   help="cinematic: maximum bokeh radius, i.e. how shallow the focus is")
    # Off by default. The effect is built for point lights - lamps, fairy lights -
    # which bloom into discs. A large bright area such as a window or an open door
    # simply floods and turns white.
    p.add_argument("--highlights", type=float, default=0.0,
                   help="cinematic: bloom point lights into bokeh discs (try 0.8 in the evening)")
    p.add_argument("--keep-near", type=float, default=0.35, metavar="THRESHOLD",
                   help="cinematic + --background: how much in front of you survives replacement")
    p.add_argument("--light", type=float, default=0.25, help="cinematic: virtual key light (0 = off)")
    p.add_argument("--rim", type=float, default=0.4, help="cinematic: rim light (0 = off)")

    p.add_argument("--parallax", type=float, default=0.0, metavar="PERCENT",
                   help="virtual camera move through the depth map, as a percentage "
                        "of frame width (0 = off, 2-4 is plenty)")
    p.add_argument("--parallax-period", type=float, default=12.0, metavar="SECONDS",
                   help="how long one parallax orbit takes")

    p.add_argument("--overlay", metavar="VIDEO", help="mix in a video file")
    p.add_argument("--overlay-mode", default="luma", choices=["luma", "chroma"],
                   help="luma = bright-on-black footage; chroma = green screen")
    p.add_argument("--overlay-z", type=float, default=None,
                   help="place the overlay at this depth so you can occlude it")
    p.add_argument("--overlay-layer", default="front", choices=list(overlays.LAYERS),
                   help="front = over the finished frame; back = behind you, "
                        "mixed into the background so you cover it")

    p.add_argument("--gestures", action=argparse.BooleanOptionalAction, default=True,
                   help="trigger reactions with hand gestures (needs mediapipe)")
    p.add_argument("--gesture-hold", type=float, default=0.8, metavar="SECONDS",
                   help="how long a gesture must be held still before it fires")
    p.add_argument("--auto-frame", action=argparse.BooleanOptionalAction, default=False,
                   help="crop and zoom to follow you around the frame")
    p.add_argument("--frame-zoom", type=float, default=1.6, metavar="FACTOR",
                   help="auto-frame: how far in it may crop (1 = no zoom)")
    p.add_argument("--gesture-face-guard", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="ignore gestures made with the hand over the face")
    p.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=False,
                   help="flip the camera left to right, so reaching right moves "
                        "right on screen; backgrounds and held windows are not "
                        "flipped, so text in them stays readable")
    p.add_argument("--pinch", type=float, default=None, metavar="RATIO",
                   help="how tight a pinch has to be, as fingertip gap over hand "
                        "size; watch `pinch_gap` in aicamctl state to tune it")
    p.add_argument("--gesture-model", default=os.path.join(HERE, "models", "hand_landmarker.task"))
    p.add_argument("--particles", type=int, default=3000, help="particle pool size")
    p.add_argument("--socket", default=None, help="control socket path")
    p.add_argument("--no-control", action="store_true", help="do not open a control socket")

    p.add_argument("--downsample", type=float, default=0.25,
                   help="resolution the matting model works at, as a fraction of full")
    # Measured at 1080p with downsample 0.25: resnet50 142 fps, mobilenetv3 139.
    # The network runs at 480x270 where the GPU has slack, so the larger model is
    # free here and gives cleaner edges. mobilenetv3 is for weaker machines.
    p.add_argument("--variant", default="resnet50", choices=["resnet50", "mobilenetv3"])
    # Measured on an RTX 4070 SUPER at 1080p: fp32 ~149 fps, fp16 ~130 fps. The
    # model is too small for half precision to pay for its conversion cost.
    p.add_argument("--fp16", action="store_true", help="half precision (measured slower, halves VRAM)")
    p.add_argument("--preview", action="store_true", help="show a window while running")
    p.add_argument("--stats", action="store_true", help="print frame rate")
    return p


# ---------------------------------------------------------------- main

def main():
    args = build_parser().parse_args()

    if not args.no_control:
        try:
            control.send({"query": "state"}, args.socket, timeout=1.0)
            sys.exit("aicam: already running (aicamctl stop, or --no-control)")
        except ConnectionError:
            pass    # nobody there, the socket is ours

    if not os.path.exists(args.output):
        sys.exit(
            f"aicam: {args.output} does not exist. The loopback module is not loaded:\n"
            f"  sudo modprobe v4l2loopback devices=1 "
            f"video_nr={args.output.rsplit('video', 1)[-1]} "
            f'card_label="AI Cam" exclusive_caps=1'
        )
    if not torch.cuda.is_available():
        sys.exit("aicam: no CUDA device found")

    device = torch.device("cuda")
    dtype = torch.float16 if args.fp16 else torch.float32

    print(f"aicam: loading {args.variant} on {torch.cuda.get_device_name(0)}", file=sys.stderr)
    matting = torch.hub.load("PeterL1n/RobustVideoMatting", args.variant, trust_repo=True)
    matting = matting.eval().to(device, dtype)

    depth_model = DepthEstimator(device, dtype)
    print("aicam: loaded MiDaS_small for depth", file=sys.stderr)

    detector = None
    if args.gestures:
        try:
            from gestures import GestureDetector, GESTURE_SCENES
            detector = GestureDetector(args.gesture_model, pinch_on=args.pinch,
                                       hold=args.gesture_hold,
                                       face_guard=args.gesture_face_guard)
            print(f"aicam: gestures active ({', '.join(GESTURE_SCENES)})", file=sys.stderr)
        except ImportError:
            print("aicam: mediapipe not available, gestures off "
                  "(use .venv/bin/python, or --no-gestures)", file=sys.stderr)
        except Exception as exc:
            print(f"aicam: gestures off ({exc})", file=sys.stderr)

    try:
        cap, width, height = open_camera(args.input, args.width, args.height, args.fps)
    except RuntimeError as exc:
        sys.exit(f"aicam: {exc}")
    if (width, height) != (args.width, args.height):
        print(f"aicam: camera gave {width}x{height}", file=sys.stderr)

    try:
        background_source = overlays.open_background(
            args.background, device, dtype,
            width=width, height=height, fps=args.fps) if args.background else None
        if background_source is not None:
            history.remember(args.background)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        sys.exit(f"aicam: {exc}")

    sink = open_sink(args.output, width, height, args.fps)
    # ffmpeg takes over a second to fail against a wedged loopback device, so a
    # shorter check lets it through and we crash later on the first write.
    time.sleep(1.5)
    if sink.poll() is not None:
        sys.exit(
            f"aicam: ffmpeg could not open {args.output}.\n"
            "If it printed VIDIOC_G_FMT: Invalid argument, the loopback device is\n"
            "wedged. Reload the module:\n"
            "  sudo modprobe -r v4l2loopback && sudo modprobe v4l2loopback"
        )

    preview = open_preview(width, height, args.fps) if args.preview else None
    server = None if args.no_control else ControlServer(args.socket)
    if server:
        print(f"aicam: control socket at {server.path}", file=sys.stderr)

    pool = particles.ParticleSystem(device, capacity=args.particles)
    active = []
    overlay = None
    if args.overlay:
        overlay = overlays.VideoOverlay(args.overlay, device, mode=args.overlay_mode,
                                        z=args.overlay_z, layer=args.overlay_layer)

    settings = {k: getattr(args, k) for k in TUNABLE}
    print(f"aicam: {args.input} -> {args.output} ({width}x{height}), Ctrl-C to stop",
          file=sys.stderr)

    # RVM is recurrent: the state carried between frames is what keeps the matte
    # steady over time instead of shimmering around hair and shoulders.
    rec = [None] * 4
    stop = False
    muted = False
    panel = None            # the window being held in the frame, if any
    mirror = args.mirror
    # Reactions can be switched off while the hand tracker keeps running: the
    # pinch pointer is a pointing device, not a party trick, and a false
    # confetti is a different annoyance from losing the ability to drag a window.
    gestures_on = detector is not None
    auto_frame = args.auto_frame
    framer = effects.AutoFramer()
    was_pinching = False

    # Muting releases the camera outright, so the lamp goes out - a mute you
    # cannot see is not a mute. Nothing arrives from the device then, and the
    # loopback still has to be fed, so the loop runs on black frames and paces
    # itself. Note the cost: with the camera closed no hand is visible, so only
    # the panel and aicamctl can unmute again.
    blank = np.zeros((height, width, 3), np.uint8)

    def set_muted(on):
        nonlocal cap, muted
        if on == muted:
            return
        if on:
            cap.release()
            cap = None
        else:
            try:
                cap = open_camera(args.input, args.width, args.height, args.fps)[0]
            except RuntimeError as exc:
                print(f"aicam: {exc}, staying muted", file=sys.stderr)
                return
        muted = on

    def handle_signal(*_):
        nonlocal stop
        stop = True

    try:
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
    except ValueError:
        pass    # not the main thread; the caller owns shutdown

    def start(name):
        if name in SCENES:
            active.append(SCENES[name](device))
            return True
        return False

    frames, fps_now, t0 = 0, 0.0, time.monotonic()
    last = time.monotonic()

    try:
        with torch.no_grad():
            while not stop:
                if cap is not None:
                    ok, frame = cap.read()
                    if not ok:
                        print("aicam: lost the camera feed", file=sys.stderr)
                        break
                else:
                    # The camera used to do the pacing. Sleep out whatever is
                    # left of the frame budget instead, or ffmpeg is handed
                    # black frames as fast as the GPU can make them.
                    frame = blank
                    idle = 1.0 / args.fps - (time.monotonic() - last)
                    if idle > 0:
                        time.sleep(idle)

                now = time.monotonic()
                dt = min(now - last, 0.1)      # clamp so a hiccup cannot teleport particles
                last = now

                # Flipped here, before anything reads it: matting, depth and the
                # hand tracker all then work in the same coordinates as the output,
                # so a hand moving right moves right on screen. The background and
                # any held window are composited later and stay unflipped, which is
                # what keeps their text readable.
                if mirror:
                    frame = cv2.flip(frame, 1)

                if detector is not None:
                    detector.submit(frame)
                    for gesture in detector.poll():
                        if not gestures_on:
                            continue    # drained anyway, so nothing fires late
                        from gestures import GESTURE_SCENES
                        start(GESTURE_SCENES.get(gesture, ""))

                if server is not None:
                    for msg in server.drain():
                        if msg.get("quit"):
                            stop = True
                        if "mirror" in msg:
                            mirror = bool(msg["mirror"])
                        if "auto_frame" in msg:
                            auto_frame = bool(msg["auto_frame"])
                            framer.reset()
                        if "gestures" in msg:
                            gestures_on = bool(msg["gestures"]) and detector is not None
                        if "mute" in msg:
                            set_muted(bool(msg["mute"]))
                        if "preview" in msg:
                            if msg["preview"] and (preview is None or preview.poll() is not None):
                                preview = open_preview(width, height, args.fps)
                            elif not msg["preview"] and preview is not None:
                                close_pipe(preview)
                                preview = None
                        if "trigger" in msg:
                            start(msg["trigger"])
                        if msg.get("clear"):
                            active.clear()
                            pool.alive[:] = False
                        if "set" in msg:
                            for k, v in msg["set"].items():
                                if k in TUNABLE:
                                    settings[k] = float(v)
                        if "overlay" in msg or "overlay_layer" in msg:
                            try:
                                overlay = apply_overlay_message(msg, overlay, device)
                            except (FileNotFoundError, ValueError) as exc:
                                print(f"aicam: {exc}", file=sys.stderr)
                        if "window" in msg:
                            try:
                                panel = open_panel(msg["window"], panel, device, dtype,
                                                   width, height, args.fps)
                            except (ValueError, RuntimeError) as exc:
                                print(f"aicam: {exc}", file=sys.stderr)
                        if "background" in msg:
                            try:
                                background_source = apply_background_message(
                                    msg, background_source, device, dtype,
                                    width=width, height=height, fps=args.fps)
                            # A bad path, an unknown monitor, a Wayland session: say so
                            # and keep running on the background that already works.
                            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                                print(f"aicam: {exc}", file=sys.stderr)

                # Pinch to hold. Grabbing an empty hand pulls out whatever screen
                # is already the background, so the trick needs no setup at all.
                pointer = detector.pointer if detector is not None else None
                pinching = bool(pointer and pointer[2])
                if pinching and not was_pinching:
                    # Any deliberate pinch takes the panel, wherever it is: making
                    # the user find a rectangle with their fingertips is a game,
                    # not a feature. The smoothing turns the jump into a glide.
                    if panel is not None:
                        panel.held = True
                    elif background_source is not None and \
                            desktop.is_desktop_spec(background_source.path):
                        try:
                            panel = open_panel(background_source.path, None, device,
                                               dtype, width, height, args.fps)
                            panel.x, panel.y = pointer[0], pointer[1]
                            panel.held = True
                        except (ValueError, RuntimeError) as exc:
                            print(f"aicam: {exc}", file=sys.stderr)
                elif not pinching and panel is not None:
                    panel.held = False
                was_pinching = pinching

                if panel is not None and panel.held and pointer is not None:
                    panel.move_to(pointer[0], pointer[1])

                src = torch.from_numpy(frame).to(device, non_blocking=True)
                src = src.permute(2, 0, 1)[None].flip(1).to(dtype) / 255.0     # BGR -> RGB

                fgr, pha, *rec = matting(src, *rec, downsample_ratio=settings["downsample"])
                d = depth_model(src)

                # A video background hands over a new frame here; a still one hands
                # back the same tensor every time.
                background = background_source.sample(height, width) \
                    if background_source is not None else None

                # One virtual camera move per frame, applied to the plate only: the
                # subject is composited afterwards so its edge cannot smear into
                # the background it is sliding over.
                pdx = pdy = 0.0
                if settings["parallax"] > 0:
                    # Half the amplitude vertically: an orbit, not a circle, which
                    # is what a camera on a real dolly does.
                    ang = 2 * math.pi * time.monotonic() / max(settings["parallax_period"], 0.1)
                    amp = settings["parallax"] / 50.0     # percent of width -> grid units
                    pdx, pdy = amp * math.sin(ang), amp * 0.5 * math.cos(ang)

                solid = (pha > 0.9).to(d.dtype)
                focus = float((d * solid).sum() / solid.sum()) if float(solid.sum()) > 100 \
                    else float(d.median())

                if background is not None and (pdx or pdy):
                    # Flat depth: a backdrop is a plane, so it pans rigidly. The
                    # subject is drawn from the unwarped camera frame and stays put.
                    background = effects.parallax(background, torch.zeros_like(d),
                                                  pdx, pdy, focus)

                # Only worth paying for when the plate is about to move; standing
                # still, the subject covers their own ghost exactly.
                plate = effects.fill_behind(src, pha) if (pdx or pdy) else src

                if muted:
                    # Camera off: the background is the whole picture. The matte is
                    # still computed, so unmuting is instant and depth keeps working
                    # for the particles.
                    base = background if background is not None else torch.zeros_like(src)
                    subject, subject_alpha = src, torch.zeros_like(pha)
                else:
                    if args.mode == "cinematic":
                        coc = effects.circle_of_confusion(d, pha, strength=1.0)
                        if background is not None:
                            base = effects.replace_background(src, fgr, pha, coc, background,
                                                              threshold=settings["keep_near"])
                            subject_alpha = torch.zeros_like(pha)   # already composited
                            subject = base
                        else:
                            bg = plate
                            if settings["highlights"] > 0:
                                bg = effects.highlight_boost(bg, 0.85, settings["highlights"])
                            base = effects.layered_bokeh(bg, coc, settings["aperture"])
                            subject, subject_alpha = fgr, pha
                    elif background is not None:
                        base, subject, subject_alpha = background, fgr, pha
                    else:
                        base = effects.bokeh(plate, settings["blur"])
                        subject, subject_alpha = fgr, pha

                if background is None and (pdx or pdy):
                    base = effects.parallax(base, d, pdx, pdy, focus)

                # After the camera move: the panel is in your hand, not in the room.
                if panel is not None:
                    base = panel.draw(base)

                # Sampled once per frame whichever layer it sits on. The layer is read
                # here too, because a clip that ends is closed on the spot.
                clip, clip_layer = None, None
                if overlay is not None:
                    clip, clip_layer = overlay.sample(height, width), overlay.layer
                    if overlay.finished:
                        overlay.close()
                        overlay = None

                if clip_layer == "back":
                    # Into the background before you are composited, so you cover it
                    # and particles behind you still pass in front of it.
                    base = overlays.composite_over(base, clip)

                for scene in list(active):
                    scene.spawn(pool, dt, width, height, focus)
                    if not scene.advance(dt):
                        active.remove(scene)
                pool.update(dt)

                front, behind = pool.render(height, width, scene_depth=d)
                out = particles.composite(base, subject, subject_alpha, front, behind)

                if clip_layer == "front":
                    out = overlays.composite_over(out, clip)

                if args.mode == "cinematic" and not muted and \
                        (settings["light"] > 0 or settings["rim"] > 0):
                    out = effects.studio_light(out, pha, d,
                                               key=settings["light"], rim=settings["rim"])

                # Last of all, so the particles, the overlay and the held panel
                # are all carried by the same camera move.
                if auto_frame:
                    out = framer.apply(out, pha, settings["frame_zoom"])

                out = (out.clamp(0, 1) * 255).round().to(torch.uint8)
                out = out[0].permute(1, 2, 0).contiguous().cpu().numpy()

                try:
                    sink.stdin.write(out.tobytes())
                except BrokenPipeError:
                    print("aicam: ffmpeg closed the pipe. If the loopback device is wedged,\n"
                          "  sudo modprobe -r v4l2loopback && sudo modprobe v4l2loopback",
                          file=sys.stderr)
                    break

                if preview is not None and preview.poll() is None:
                    try:
                        preview.stdin.write(out.tobytes())
                    except BrokenPipeError:
                        preview = None      # window closed; the stream carries on

                frames += 1
                if frames % 15 == 0:
                    elapsed = time.monotonic() - t0
                    fps_now = frames / elapsed
                    if args.stats:
                        print(f"\raicam: {fps_now:5.1f} fps  {pool.count:4d} particles",
                              end="", file=sys.stderr)
                    frames, t0 = 0, time.monotonic()
                    if server is not None:
                        server.state = {
                            "fps": round(fps_now, 1),
                            "particles": pool.count,
                            "hands": detector.hands_visible if detector else 0,
                            "mode": args.mode,
                            "preview": preview is not None and preview.poll() is None,
                            "muted": muted,
                            "mirror": mirror,
                            "gestures": gestures_on,
                            "auto_frame": auto_frame,
                            "window": panel.label if panel else None,
                            "pinch": pinching,
                            "pinch_gap": detector.pinch_gap if detector else None,
                            "scenes": [s.name for s in active],
                            "overlay": overlay.path if overlay else None,
                            "overlay_layer": overlay.layer if overlay else args.overlay_layer,
                            "background": background_source.path if background_source else None,
                            **settings,
                        }

    finally:
        if cap is not None:
            cap.release()
        if detector is not None:
            detector.close()
        if overlay is not None:
            overlay.close()
        if panel is not None:
            panel.close()
        if background_source is not None:
            background_source.close()
        if server is not None:
            server.close()
        close_pipe(sink)
        close_pipe(preview)
    print("\naicam: stopped", file=sys.stderr)


if __name__ == "__main__":
    main()
