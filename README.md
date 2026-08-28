# aicam

A virtual webcam for Linux with depth-aware background effects and reactions,
running on your GPU. Teams, Meet, Zoom and anything else that speaks V4L2 sees
it as an ordinary camera.

macOS and Windows composite their reactions as a flat layer over the video.
aicam estimates a depth map, so **fireworks go off behind you and are occluded
by your shoulder** while confetti falls in front. Same reason a book you hold up
stays sharp when the background is blurred: it is nearer than the focus plane,
and the pipeline knows that.

```
/dev/video0              aicam                              /dev/video10
your webcam         ->   matting + depth on CUDA        ->   "AI Cam"    -> browser -> Teams
1920x1080 MJPG           depth-graded bokeh                  v4l2loopback
                         particles with a z coordinate
                         video overlays
                         hand gestures (CPU, off-thread)
```

## Requirements

- Linux with `v4l2loopback` (ships with `linux-modules-$(uname -r)` on Ubuntu 24.04)
- An NVIDIA GPU with CUDA. Measured on an RTX 4070 SUPER; anything from a 2060 up
  should hold 30 fps.
- `ffmpeg` (also provides `ffplay`, used for the preview)
- Python 3.10+

## Install

```sh
git clone <your fork> aicam && cd aicam
python3 -m venv --system-site-packages .venv
.venv/bin/pip install torch opencv-python-headless numpy timm

# Gestures. --no-deps keeps mediapipe from replacing opencv-python-headless
# with opencv-contrib-python, which would shadow the build everything else uses.
.venv/bin/pip install --no-deps mediapipe
.venv/bin/pip install absl-py flatbuffers protobuf attrs sounddevice matplotlib

./setup-models.sh          # hand landmarker; the other models self-download
```

Then the loopback device, once (survives reboot):

```sh
sudo tee /etc/modprobe.d/v4l2loopback.conf >/dev/null <<'CONF'
options v4l2loopback devices=1 video_nr=10 card_label="AI Cam" exclusive_caps=1
CONF
echo v4l2loopback | sudo tee /etc/modules-load.d/v4l2loopback.conf >/dev/null
sudo modprobe v4l2loopback
```

`exclusive_caps=1` is required. Without it the device advertises itself as both
input and output at once, and Chrome refuses to list it.

## Use

```sh
./aicam                                   # uniform background blur
./aicam --mode cinematic                  # depth-graded bokeh + studio light
./aicam --mode cinematic --preview        # with an ffplay window
./aicam --background backgrounds/studio.jpg
```

Then pick **AI Cam** as your camera. The real camera must be free when aicam
starts; Chrome does not release `/dev/video0` until every tab that used it is
closed.

### Cinematic mode

Defocus increases gradually with distance instead of switching at the silhouette,
and a virtual key and rim light are applied to the subject.

| flag | |
|---|---|
| `--aperture 45` | how shallow the focus is. 30 is discreet, 60 is obvious. |
| `--light 0.25` | key light. Shapes the subject rather than just brightening it. |
| `--rim 0.4` | rim light on the silhouette; what separates you from the background. |
| `--highlights 0` | bloom point lights into bokeh discs. See below. |
| `--keep-near 0.35` | with `--background`: how much in front of you survives replacement. |

`--highlights` is off by default because it is built for *point* lights — lamps,
fairy lights, candles. A large bright area such as a window or an open door has
too much area: it floods and turns white. Try `--highlights 0.8` in the evening.

### Background replacement that lets through what you hold up

```sh
./aicam --mode cinematic --background backgrounds/studio.jpg
```

In blur mode `--background` replaces everything outside the matte, so a book you
hold up vanishes the moment segmentation stops counting it as part of you. In
cinematic mode *depth* decides: everything at or in front of the focus plane is
kept from the real image, and only what lies behind is replaced.

### Reactions

Five scenes: `confetti`, `hearts`, `fireworks`, `balloons`, `rain`. Each spreads
its particles in z around your own depth, so some pass in front of you and some
behind.

Trigger them three ways:

```sh
./aicamctl confetti           # command line
./gui.py                      # Tk control panel with a button per scene
```

...or with a gesture, if `mediapipe` is installed:

| gesture | scene |
|---|---|
| thumbs up | confetti |
| both hands forming a heart | hearts |
| both palms open | fireworks |
| victory sign | balloons |
| thumbs down | rain |

A gesture must be held for three consecutive detections before it fires, and the
same gesture will not fire again for four seconds. Gestures are never reliable
enough to be the only route, which is why the panel and the CLI exist.

### Video overlays

```sh
./aicam --overlay sparks.mp4 --overlay-mode luma
```

`luma` treats brightness as alpha — right for stock footage of fireworks, sparks
or smoke on black, which composites additively with nothing to cut out. `chroma`
keys out a green or blue screen. The GUI has a file picker for both.

### Control socket

Everything is reachable at runtime over a unix socket:

```sh
./aicamctl state
./aicamctl fireworks
./aicamctl set aperture=70 light=0.4
./aicamctl clear
```

## Performance

RTX 4070 SUPER, 1080p, `--downsample 0.25`, whole chain including transfer back
to the CPU.

| | ms/frame | fps | VRAM |
|---|---:|---:|---:|
| `--mode blur` | 7.1 | 142 | 375 MiB |
| `--mode cinematic` | 17.6 | 57 | 511 MiB |
| cinematic + particles + gestures | 27 | 37 | ~700 MiB |

Hand tracking costs 18-20 ms but runs on its own CPU thread, so it does not
enter the frame budget.

## Licence

GPL-3.0-or-later. See [LICENSE](LICENSE).

aicam uses [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting)
(GPL-3.0), which is why this project is GPL rather than permissive.
[MiDaS](https://github.com/isl-org/MiDaS) is MIT and
[MediaPipe](https://github.com/google-ai-edge/mediapipe) is Apache-2.0. No model
weights are redistributed here; they are downloaded on first run.

## Troubleshooting

**"Device or resource busy"** — something else is streaming from the camera.
Chrome holds it until every tab that used it is closed:

```sh
fuser -v /dev/video0
```

`cameractrls` in that list is fine; it holds the device for control, not for
streaming, and steals no frames.

**Exactly half the frame rate (15 fps instead of 30)** — this is not the camera
and not poor lighting. It is `CAP_PROP_BUFFERSIZE = 1` in OpenCV: with a
one-frame queue the consumer misses every other frame. Check against ffmpeg,
which bypasses OpenCV:

```sh
ffmpeg -f v4l2 -input_format mjpeg -video_size 1920x1080 -framerate 30 \
    -i /dev/video0 -t 5 -f null -
```

**`ioctl(VIDIOC_G_FMT): Invalid argument` when aicam starts** — the loopback
device is wedged. Reload the module:

```sh
sudo modprobe -r v4l2loopback && sudo modprobe v4l2loopback
```

It seems to happen when a consumer is still attached as the producer stops, so
close the Teams tab and the preview *before* stopping aicam.

Note that `v4l2-ctl -d /dev/video10 --get-fmt-video` prints **the same error when
everything is fine**, as long as no producer is attached: with `exclusive_caps=1`
the capture side has no format until something writes to it. Do not use it as a
health check — test with a write instead:

```sh
ffmpeg -f lavfi -i testsrc=size=1920x1080:rate=30 -t 2 \
    -f v4l2 -pix_fmt yuv420p /dev/video10
```

Silence means the device is healthy.

**"AI Cam" does not appear in Teams** — check the module loaded with
`exclusive_caps=1`, and restart Chrome after `modprobe`; it enumerates cameras at
startup.

**`--preview` crashes with "The function is not implemented"** — you have an old
`aicam.py` that used `cv2.imshow`. `opencv-python-headless` is built with
`GUI: NONE`; the preview goes through ffplay now.
