# Design notes and measurements

How aicam is put together, and what was actually measured while building it.
Several of the conclusions are the opposite of what seems obvious, so they are
recorded with the numbers that settled them.

## The problem

Teams on Linux has no background effects. macOS users get Portrait mode, Center
Stage, Studio Light and Reactions in the client. The way out is a *virtual*
camera: a program reads the real camera, does something to the image, and writes
the result to a device that everything else sees as an ordinary webcam.

## Data flow

    /dev/video0              aicam.py                            /dev/video10
    webcam            -->    +--------------------------+   -->  "AI Cam"     --> Chrome --> Teams
    1920x1080 MJPG           | OpenCV: read + MJPG decode|        v4l2loopback
    30 fps                   | CUDA: matting (RVM)       |        1920x1080 YU12
                             | CUDA: depth (MiDaS)       |
                             | CUDA: bokeh, lighting     |
                             | CUDA: particles           |
                             | ffmpeg: RGB -> YUV420     |
                             +--------------------------+
                                   |            |
                                   |            +-- ffplay preview (optional)
                                   +-- gesture thread (CPU, MediaPipe)
                                   +-- control socket thread

Everything between the upload and the download stays on the GPU: one transfer up
(uint8 BGR) and one down (uint8 RGB) per frame.

## Matting

[RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) via
`torch.hub`. The model is *recurrent*: it carries four hidden states
(`rec[0..3]`) from frame to frame.

That is the whole point. A model that sees each frame in isolation produces a
matte that shivers around hair and shoulders, and that shiver is exactly what
makes cheap background filters look cheap. RVM sees the previous frame and holds
the edge still.

`downsample_ratio` sets the resolution the network runs at. 0.25 means 480x270
for 1080p input, which the authors recommend for HD. The result is upsampled and
refined against the full-resolution frame.

## Depth

`MiDaS_small` via `torch.hub`, 3.0 ms at 256x256. `DPT_Hybrid` measured 18.6 ms
and would have eaten more than half the budget on its own, without the larger
map mattering here: what is needed is the big surfaces, not the detail.

Unlike RVM the model is not recurrent, so two nearly identical frames give
slightly different maps. Without smoothing that shows up as defocus pulsing in
the background. An exponential moving average on the low-resolution map removes
it at no cost.

## Cinematic mode

**Layered defocus.** True per-pixel varying blur needs scatter. We approximate
with four pre-blurred layers and triangular weighting between them, driven by a
*circle of confusion* computed from how far behind the subject each pixel is.
The focus plane is read from the subject's own depth where the matte is
confident, so it follows you if you lean forward.

The kernel is a flat disc, not a Gaussian. A lens spreads a point of light into
a hard-edged disc shaped by the aperture; a Gaussian spreads it into a soft
blob. The disc shape is what reads as "camera" rather than "filter".

**Studio light.** Approximate normals from the depth gradient give Lambertian
shading, plus a rim light on the silhouette where it faces the light.

## Depth-aware compositing

This is the part macOS and Windows do not do. Their reactions are a flat layer
over the video. Every aicam particle carries a z, compared against the depth map
at its own pixel:

    here   = depth[particle.y, particle.x]
    behind = particle.z < here          # further away than whatever is there

Particles are rasterised into two layers and composited in order:

    background -> overlay (back) -> particles behind -> subject
               -> particles in front -> overlay (front)

so confetti passes behind your shoulder and is occluded by it. A video overlay
picks one of the two ends of that chain: `front` pastes it over the finished
frame, `back` mixes it into the background before you are composited, so you
cover it while particles between you and the wall still pass in front of it.
There is deliberately no slot for it *between* the particle layers - a clip is
flat footage with no depth of its own, so the only honest choices are all the
way in front or all the way behind. Each scene
spreads its particles in z *around the subject's own depth*, which is what makes
some go in front and some behind. A scene where every particle shares one z
looks like a sticker.

Rendering is one `grid_sample` per particle for rotation and scale, then a
`scatter_add` splat into the frame. 1.2 ms for a few hundred particles at 1080p.

## The screen as a background

A monitor or a window is captured with ffmpeg's `x11grab` and read as raw frames
off a pipe, which is the same shape as the video decoder: a thread fills a small
queue, the render loop never waits on I/O.

The scaling is the whole design decision. Raw 4K in bgr24 is 745 MB/s down the
pipe; asking ffmpeg for the render size instead makes it 186 MB/s at 1080p, and
the resampling happens in the process that already has the frame. Measured time
to the first frame is around 0.2 s whether the grab is one 1680x1050 monitor or
the full 5520x2160 desktop.

Window grabs use `-window_id`, so the capture follows the window as it moves. A
resize changes the frame format and ends the grab; the reader notices the short
read, re-resolves the window's geometry and respawns. You get a hiccup rather
than a dead background.

X11 only, deliberately. Wayland routes screen capture through the desktop portal
and PipeWire, which is a different mechanism rather than a different flag, and
guessing at it would produce a black rectangle with no explanation.

## Background replacement that understands depth

A matte only knows "person" and "not person". Hold up a book and it vanishes the
moment segmentation stops counting it as part of you — and it is unpredictable
when that happens.

The depth map knows something the matte does not: the book is nearer than the
wall. So the keep mask is the union of the subject and everything in focus:

    near = (1 - coc / threshold).clamp(0, 1)      # 1 at and in front of focus
    keep = (alpha + near * (1 - alpha)).clamp(0, 1)
    keep = blur(keep, 9)                          # the depth map is coarse at edges
    keep = max(keep, alpha)                       # the subject is always kept

That last line is not decoration: the blur above can pull `keep` below `alpha`
along the silhouette, which would partly replace the edge of the subject.

Verified by setting a patch of the depth map nearer than the focus plane: `keep`
comes out 1.000 inside the patch and 0.000 in the wall behind, and the pixels
inside the patch are identical to the original. With nothing held up, 0.4% of
the frame is kept beyond the matte.

## Gestures

MediaPipe's hand landmarker costs 18-20 ms per frame here, and input resolution
barely moves it — it is palm detection that dominates. That does not fit in a
33 ms budget alongside matting, depth and bokeh.

It is also pure CPU work, so it runs on its own thread against the most recent
frame, and the render loop never waits for it. Hands do not move fast enough for
the resulting lag to matter.

A gesture must be seen on three consecutive detections before it fires, and the
same gesture cannot fire again within four seconds. Without both, a hand passing
through a pose on its way somewhere else sets off confetti in a meeting.

## Drawing in the air

The pinch that holds a window already gives a fingertip position every frame, so
ink costs no new model and no new tracking. What makes it worth building is the
depth map: each point of a stroke keeps the depth of the fingertip that drew it,
so the ink hangs in the room rather than on the glass, and you can walk in front
of it.

Two things had to be right.

**The depth test is per pixel, not per point.** The particles test one depth
under each sprite's centre, which is invisible for a 6 px confetto. A stroke is
different: it runs *along* an edge for its whole length, so stamps whose centres
fall just outside a shoulder would paint several pixels of ink across it. Every
point lays its own depth into a z-buffer as it is stamped and the front/behind
split is decided pixel by pixel afterwards. Same cost, no bleed.

**Stamps combine with max, not sum.** A stroke is a dense line of overlapping
brush marks. Added together, a hand that slowed down burns a bright knot and a
steady one draws a bar; `scatter_reduce(amax)` gives ink that is opaque and even
whatever the hand did.

The tracker runs at 10 Hz against a 30 fps loop. Rather than interpolating
between two tracker samples, the pointer is low-passed every frame and the gap
between consecutive positions is filled with stamps: the smoothing *is* the
interpolation, and it doubles as the fix for a fingertip that jitters by a pixel
or two when a hand is trying to hold still.

**Undo had to become a snapshot.** It started as "drop the last stroke", which
is the obvious implementation and cannot survive an eraser: the ink a rub-out
removed is gone, so there is no stroke to drop. Each action now keeps a copy of
the canvas as it was before it - 0.04 ms and only as large as the ink that
existed at the time - and one Undo means the same thing whether the last thing
you did was draw, rub out, or wipe the lot. Wiping being undoable is the part
that gets used: *Erase all* is the easiest button to hit by mistake.

Measured on the RTX 4070 SUPER at 1080p, with the rest of the pipeline running:

| | ms/frame |
|---|---:|
| 1 000 points on the canvas | 0.97 |
| 5 000 points | 1.08 |
| 12 000 points | 1.84 |
| 24 000 points (the cap) | 2.26 |
| adding to a stroke, per frame while drawing | 0.22 |
| an eraser sweep over a full canvas | 0.06 |
| ageing the laser, full canvas of it | 0.34 |
| ageing when nothing is fading | 0.02 |
| the undo snapshot, once per stroke | 0.04 |

Nothing is paid when the canvas is empty. A full canvas is 2.3 ms of the 6 ms
that were left, which is what set the cap: past that the wall is illegible
anyway, and the oldest stroke is dropped whole rather than trimmed, because
losing the tail of a scribble looks like a bug and losing the scribble looks
like a wipe.

## Measurements

RTX 4070 SUPER, 1080p, `downsample_ratio=0.25`, whole chain including the
transfer back to the CPU. 60 frames after warm-up.

| | ms/frame | fps | VRAM |
|---|---:|---:|---:|
| resnet50, fp32 (default) | 7.1 | 142 | 375 MiB |
| mobilenetv3, fp32 | 7.2 | 139 | 265 MiB |
| mobilenetv3, fp16 | 7.7 | 130 | 131 MiB |
| cinematic | 17.6 | 57 | 511 MiB |
| cinematic + particles + gestures | 27 | 37 | ~700 MiB |

End to end against the real camera: **28.7 fps**, which is the camera's ceiling.

## Things that turned out the opposite of the assumption

**fp16 is slower than fp32.** 130 against 139 fps. The model is 3.7 M parameters
— too small to earn back the conversion cost, and it does not saturate the
tensor cores. `--fp16` remains, but only if you need the VRAM elsewhere.

**resnet50 is free.** 142 against 139 fps despite 7x the parameters. At
`downsample_ratio=0.25` the network runs at 480x270 where the GPU has plenty of
slack; the time goes on upsampling to full resolution and the transfer back,
which is identical for both. Hence the large model is the default.

**`CAP_PROP_BUFFERSIZE = 1` halves the frame rate.** It was added as a latency
optimisation and produced exactly 15.0 fps instead of 30. With a one-frame queue
the consumer misses every other frame. Measured: default 29.9 — `4` 29.9 — `1`
15.0. ffmpeg outside OpenCV gave 30, which is what revealed the fault was ours.

The lesson generalises: when a rate is *exactly* half, suspect the queue, not
the source. The first suspect was auto-exposure lowering the frame rate in poor
light; it was off (`exposure_dynamic_framerate = 0`).

**`VIDIOC_G_FMT: Invalid argument` usually means nothing.** With
`exclusive_caps=1` the capture side of the loopback device has no format until a
producer has written to it, so `v4l2-ctl --get-fmt-video` fails when everything
is fine. It is not a health check. Test with a write.

**`--highlights` only works on point lights.** The idea was that lights behind
you would bloom into bokeh discs. They do — if they are *points*. A large bright
area, such as an open door in daylight, has too much area: it floods and turns
white. Hence off by default.

**The key light had to be made zero-mean.** The first version only added light
(`out * (1 + key * shade)`), and the result was a washed-out face, not a lit one.
Subtracting the mean over the subject makes the light *shape*: one side lifts,
the other drops.

**The first `--aperture` was three times too low.** 14 was chosen because it
sounded like a lot. It left the background still readable — a worse separation
than the plain `--blur 40` it was meant to replace. Measured side by side on the
same frame, it landed on 45.

**The first heart sprite was bloated and the star was a flower.** Scaling an
implicit curve by a constant to get a soft edge fails where the shape is thin:
the heart's lower point never reached full opacity. Filling the exact interior
and blurring by one pixel afterwards keeps the silhouette honest.

## Deliberate choices

**ffmpeg as the sink** rather than `VIDIOC_S_FMT` ioctls from Python. It
negotiates the v4l2 format itself and is already installed. The device ends up
1920x1080 YU12.

**ffplay for the preview** rather than `cv2.imshow`. The installed package is
`opencv-python-headless`, built with `GUI: NONE`; `cv2.imshow` raises "function
is not implemented" no matter what. ffplay ships with ffmpeg, so it costs no new
dependency.

**Bokeh at quarter scale.** Same look as a large kernel for a fraction of the
cost, and the interpolation softness on the way back up is free.

**One unix socket for all control.** The Tk panel, `aicamctl` and any future
hotkey daemon all speak the same protocol, so the pipeline only ever handles one
kind of message, and front ends can come and go without disturbing it.

## Compared with the OBS route

OBS Studio with `obs-backgroundremoval` does much of this without code. It was
installed and tested, but **runs on the CPU**: the flatpak contains only
`libonnxruntime_providers_shared.so`, not `libonnxruntime_providers_cuda.so`, so
the CUDA provider cannot load regardless of what the GUI offers. Verified with
`strings` against the library rather than assumed.

It is worth keeping installed as a fallback, and it has four low-light models
(`zero_dce`, `uretinex_net`, `tbefn`, `semantic_guided_llie`) that aicam does not.

**Auto-framing off the matte.** A face tracker would be a third model in a
budget that has six spare milliseconds. The alpha channel already carries the
subject's outline every frame, so the bounding box is a reduction on data that
has been paid for, and the framing is one crop and one resize. Measured: 30 fps
with it on, the same as with it off.

## Where there is room left

At 30 fps roughly 27 of 33 ms are used with everything on. Still unbuilt:
low-light denoising, background stylisation, gaze correction.
