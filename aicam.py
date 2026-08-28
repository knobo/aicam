#!/usr/bin/env python3
"""aicam - a GPU-accelerated virtual webcam for Linux.

Reads a real camera, separates you from the background with RobustVideoMatting
on CUDA, applies depth-aware effects, and writes the result to a v4l2loopback
device that Teams, Meet, Zoom or anything else sees as an ordinary webcam.

Copyright (C) 2026 Knut Boehmer
Licensed under the GNU General Public License v3.0 or later; see LICENSE.
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import time

import cv2
import torch

# Run via shebang from any directory, so our own modules must be found explicitly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import effects  # noqa: E402
import overlays  # noqa: E402
import particles  # noqa: E402
from control import ControlServer  # noqa: E402
from depth import DepthEstimator  # noqa: E402
from scenes import SCENES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TUNABLE = {"aperture", "light", "rim", "highlights", "blur", "keep_near", "downsample"}


# ---------------------------------------------------------------- input and output

def open_camera(device, width, height, fps):
    # This OpenCV build only accepts an index on the V4L2 backend, not a path.
    m = re.fullmatch(r"/dev/video(\d+)", device)
    source = int(m.group(1)) if m else device
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"aicam: could not open {device}")
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
        sys.exit("aicam: --preview needs ffplay (apt install ffmpeg)")


def load_background(path, h, w, device, dtype):
    """Load a background image, cropped to cover the frame without stretching."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"aicam: could not read background image {path!r}")
    ih, iw = img.shape[:2]
    scale = max(w / iw, h / ih)
    img = cv2.resize(img, (round(iw * scale), round(ih * scale)), interpolation=cv2.INTER_AREA)
    y, x = (img.shape[0] - h) // 2, (img.shape[1] - w) // 2
    img = img[y:y + h, x:x + w]
    t = torch.from_numpy(img).to(device).permute(2, 0, 1)[None]
    return t.flip(1).to(dtype) / 255.0


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
    p.add_argument("--background", metavar="IMAGE", help="replace the background with this image")
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

    p.add_argument("--overlay", metavar="VIDEO", help="mix in a video file")
    p.add_argument("--overlay-mode", default="luma", choices=["luma", "chroma"],
                   help="luma = bright-on-black footage; chroma = green screen")
    p.add_argument("--overlay-z", type=float, default=None,
                   help="place the overlay at this depth so you can occlude it")

    p.add_argument("--gestures", action=argparse.BooleanOptionalAction, default=True,
                   help="trigger reactions with hand gestures (needs mediapipe)")
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
            detector = GestureDetector(args.gesture_model)
            print(f"aicam: gestures active ({', '.join(GESTURE_SCENES)})", file=sys.stderr)
        except ImportError:
            print("aicam: mediapipe not available, gestures off "
                  "(use .venv/bin/python, or --no-gestures)", file=sys.stderr)
        except Exception as exc:
            print(f"aicam: gestures off ({exc})", file=sys.stderr)

    cap, width, height = open_camera(args.input, args.width, args.height, args.fps)
    if (width, height) != (args.width, args.height):
        print(f"aicam: camera gave {width}x{height}", file=sys.stderr)

    background = load_background(args.background, height, width, device, dtype) \
        if args.background else None

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
                                        z=args.overlay_z)

    settings = {k: getattr(args, k) for k in TUNABLE}
    print(f"aicam: {args.input} -> {args.output} ({width}x{height}), Ctrl-C to stop",
          file=sys.stderr)

    # RVM is recurrent: the state carried between frames is what keeps the matte
    # steady over time instead of shimmering around hair and shoulders.
    rec = [None] * 4
    stop = False

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

    with torch.no_grad():
        while not stop:
            ok, frame = cap.read()
            if not ok:
                print("aicam: lost the camera feed", file=sys.stderr)
                break

            now = time.monotonic()
            dt = min(now - last, 0.1)      # clamp so a hiccup cannot teleport particles
            last = now

            if detector is not None:
                detector.submit(frame)
                for gesture in detector.poll():
                    from gestures import GESTURE_SCENES
                    start(GESTURE_SCENES.get(gesture, ""))

            if server is not None:
                for msg in server.drain():
                    if "trigger" in msg:
                        start(msg["trigger"])
                    if msg.get("clear"):
                        active.clear()
                        pool.alive[:] = False
                    if "set" in msg:
                        for k, v in msg["set"].items():
                            if k in TUNABLE:
                                settings[k] = float(v)
                    if "overlay" in msg:
                        if overlay is not None:
                            overlay.close()
                            overlay = None
                        if msg["overlay"]:
                            try:
                                overlay = overlays.VideoOverlay(
                                    msg["overlay"], device,
                                    mode=msg.get("overlay_mode", "luma"))
                            except FileNotFoundError as exc:
                                print(f"aicam: {exc}", file=sys.stderr)

            src = torch.from_numpy(frame).to(device, non_blocking=True)
            src = src.permute(2, 0, 1)[None].flip(1).to(dtype) / 255.0     # BGR -> RGB

            fgr, pha, *rec = matting(src, *rec, downsample_ratio=settings["downsample"])
            d = depth_model(src)

            solid = (pha > 0.9).to(d.dtype)
            focus = float((d * solid).sum() / solid.sum()) if float(solid.sum()) > 100 \
                else float(d.median())

            if args.mode == "cinematic":
                coc = effects.circle_of_confusion(d, pha, strength=1.0)
                if background is not None:
                    base = effects.replace_background(src, fgr, pha, coc, background,
                                                      threshold=settings["keep_near"])
                    subject_alpha = torch.zeros_like(pha)   # already composited
                    subject = base
                else:
                    bg = src
                    if settings["highlights"] > 0:
                        bg = effects.highlight_boost(bg, 0.85, settings["highlights"])
                    base = effects.layered_bokeh(bg, coc, settings["aperture"])
                    subject, subject_alpha = fgr, pha
            elif background is not None:
                base, subject, subject_alpha = background, fgr, pha
            else:
                base = effects.bokeh(src, settings["blur"])
                subject, subject_alpha = fgr, pha

            for scene in list(active):
                scene.spawn(pool, dt, width, height, focus)
                if not scene.advance(dt):
                    active.remove(scene)
            pool.update(dt)

            front, behind = pool.render(height, width, scene_depth=d)
            out = particles.composite(base, subject, subject_alpha, front, behind)

            if overlay is not None:
                out = overlays.composite_over(out, overlay.sample(height, width))
                if overlay.finished:
                    overlay.close()
                    overlay = None

            if args.mode == "cinematic" and (settings["light"] > 0 or settings["rim"] > 0):
                out = effects.studio_light(out, pha, d,
                                           key=settings["light"], rim=settings["rim"])

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
                        "scenes": [s.name for s in active],
                        **settings,
                    }

    cap.release()
    if detector is not None:
        detector.close()
    if overlay is not None:
        overlay.close()
    if server is not None:
        server.close()
    if sink.stdin:
        sink.stdin.close()
    sink.wait(timeout=5)
    if preview is not None and preview.poll() is None:
        preview.stdin.close()
        preview.wait(timeout=5)
    print("\naicam: stopped", file=sys.stderr)


if __name__ == "__main__":
    main()
