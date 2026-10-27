# HUD panel and richer telemetry

## Intent

The Tk panel (`gui.py`) works but looks like a settings dialog. The goal is a
sci-fi HUD / cockpit look that is fun to have open, and that makes two things
obvious at a glance: **whether the camera is live**, and **how the pipeline is
doing**. The user picked direction "B · HUD / Cockpit" from three mockups.

To give the HUD something real to show, aicam reports more about itself over the
control socket: GPU load and temperature, where each frame's milliseconds go,
and a log of recent events.

## Goals

- Every control the panel has today survives, with the same behaviour: start /
  stop, preview, mute, mirror, gestures, auto-frame, reactions, clear, draw
  (pinch, undo, erase, tool, ink), all sliders, overlay (mode, choose, remove,
  layer), background (bundled, file, remove, hold a window, drop, screen or
  window, recent, URL). Scrolling, the multi-monitor sizing and the desktop
  file chooser stay.
- Camera state is unmistakable: LIVE, MUTED, BOOTING, FAILED, OFFLINE.
- Startup shows as a boot sequence; a failed start shows why, as today.
- Live telemetry: fps, fps history, hands, particles, mode, GPU, frame budget,
  event log.

## Non-goals

- No web UI, no new GUI toolkit, no ttkbootstrap.
- No new pip dependency, in the panel or in aicam.
- No change to existing messages or fields; only additions.

## Anti-goals

- The panel must not cost noticeable CPU during a meeting. Animation is cheap
  and stops when the window is minimised.
- Telemetry must not cost the render loop frame rate. Nothing new may block the
  loop or add a CUDA synchronise.
- A control that is pretty but cannot be reached or operated is a regression.

## Part 1 – protocol (`aicam.py`, maybe a small new module)

New fields in the `{"query": "state"}` reply, i.e. in `server.state`. Old
clients ignore them.

| field | type | meaning |
|---|---|---|
| `gpu` | object or `null` | `{"name", "util", "temp", "power", "mem_used", "mem_total"}`: % / °C / W / MiB / MiB. `null` when `nvidia-smi` is missing or fails. |
| `vram` | int | MiB aicam itself holds (`torch.cuda.memory_reserved`). |
| `budget` | object | ms per pipeline stage for a recent frame, keys in pipeline order, plus `"total"`. Stage names are the builder's call (e.g. `capture`, `matting`, `depth`, `look`, `particles`, `ink`, `transfer`); the panel draws whatever keys arrive, in order. |
| `uptime` | float | seconds since aicam started. |
| `camera` | str | e.g. `"1920x1080"`. |
| `output` | str | the loopback device, e.g. `"/dev/video10"`. |
| `variant` | str | matting model, e.g. `"resnet50"`. |
| `events` | list | last 10 of `{"t": unix_time, "text": str}`, oldest first. |

**GPU**: a daemon thread runs
`nvidia-smi --query-gpu=name,utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total --format=csv,noheader,nounits`
every 2 s and keeps the latest parse. Measured on this machine: the command
works; `torch.cuda.utilization()` does not (needs `pynvml`, not installed).

**Budget**: `torch.cuda.Event(enable_timing=True)` recorded between stages,
read with `elapsed_time` only after the frame's existing `.cpu()` transfer has
synchronised. CPU-side stages (capture/decode, the ffmpeg write) by
`time.perf_counter`. Measuring only the frames that feed a state update (every
15th) is fine.

**Events**: append on at least: a gesture firing a scene, a trigger or clear
from the socket, mute on/off, background / overlay / window changed or
removed, draw on/off, camera feed lost, preview opened/closed. Short text,
e.g. `"gesture: hearts"`, `"background: dusk.jpg"`.

Tests: the parsing of `nvidia-smi` output (including a missing binary and junk
output) and the event ring are testable without CUDA; add them in the style of
`tests/`.

## Part 2 – panel (`gui.py`)

Reference mockup: direction B in
`/tmp/claude-1000/-home-knobo-prog-aicam/d333e631-bfb3-4082-acf4-9ddf444a34f2/scratchpad/.superpowers/brainstorm/1259460-1793030203/content/directions.html`
(open it in a browser, look at the `.B` CSS and markup).

Palette: background `#05080c`, cyan `#3ef0ff`, dim cyan `#0f4a52`, amber
`#ffb238`, red `#ff3b4e`, text `#bfefff`, muted `#5e8a93`, ok green `#6effa0`.
Font: `Fira Code` if installed, else `DM Mono`, else `DejaVu Sans Mono`
(check `tkinter.font.families()`).

From top to bottom:

1. **Header** (Canvas): corner brackets, faint scanlines, a light band sweeping
   across, a pulsing dot and a state tag:
   LIVE = red pulsing dot + `● CAM LIVE`; MUTED = amber + `CAM MUTED`;
   BOOTING = fast amber blink + `BOOTING`; OFFLINE = dead grey dot + `OFFLINE`;
   FAILED = red steady + `FAULT`. Below it one info line:
   `resnet50 · 1920x1080 → /dev/video10 · up 00:12:31`.
2. **Telemetry** (when live): fps ring against 30, fps sparkline of the last
   ~50 samples, two hand glyphs lit by `hands`, particles · mode. GPU block:
   util bar, temperature bar coloured cyan→amber→red (70 °C / 83 °C), power,
   VRAM bar (aicam's own share marked inside the total). **Frame budget**: one
   stacked horizontal bar, one segment per `budget` key, against a 33.3 ms
   marker, with a small legend.
3. **Boot sequence** (while launching, replaces telemetry): segmented bar for
   `STAGES`, and a log `[ OK ]` / `[ ▸▸ ]` / `[ -- ]` per stage, with the last
   `aicam:` line under it. On failure: FAULT, the log in red with the last
   lines, as today's failed-start message.
4. **Reactions**: tiles with glyphs (✦ confetti, ♥ hearts, ✺ fireworks,
   ◍ balloons, ⁞ rain) and a Clear tile/button; a tile flashes when pressed
   and when that scene appears in `state["scenes"]`.
5. **Systems**: the toggles as `■`/`□` boxes (preview, mute, mirror, gestures,
   auto-frame, pinch to draw).
6. **Draw / Optics / Overlay / Background**: same controls as today, restyled.
   Sliders become segmented bars with the value printed beside; click and drag
   set the value, and when focused the arrow keys step it.
7. **Event log**: last events, terminal style, newest at the bottom.
8. **ENGAGE / DISENGAGE** button (start / stop), big, at the bottom or top –
   builder's call, but it must be reachable without scrolling at the default
   window height if at all possible.

Behaviour:

- Animation via `root.after` at ~15 fps; skipped while `root.state()` is
  `"iconic"` or `"withdrawn"`. The 500 ms `refresh()` poll stays as is.
- Every missing new field (old aicam) degrades to a dash, never an exception.
- Plain `ttk` widgets that stay (Combobox, Entry, Scrollbar, menus) get a dark
  `ttk.Style` on `clam` so nothing white is left.
- Every clickable custom widget takes keyboard focus (Tab) and activates on
  Space/Return.

Existing tests in `tests/test_panel_*.py` must still pass; adapt them only
where a widget changed shape, never by dropping what they check.

## Left to the builder

Exact geometry, sizes, glyphs, which module the custom Canvas widgets live in
(`gui.py` or a new `hud.py`), stage names in `budget`, wording of events.
