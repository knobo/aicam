"""What the panel shows about the pipeline besides the picture: GPU load, where
each frame's milliseconds go, and what happened lately.

None of it may cost frame rate. The GPU is asked through nvidia-smi on a thread
of its own, because torch.cuda.utilization() needs pynvml, which is not
installed. Frame timing reads CUDA events only after the frame's own download
has synchronised, so measuring adds no wait of its own.
"""

import collections
import subprocess
import threading
import time

import torch

QUERY = ["nvidia-smi",
         "--query-gpu=name,utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total",
         "--format=csv,noheader,nounits"]
FIELDS = ("name", "util", "temp", "power", "mem_used", "mem_total")


def _number(text):
    # A field the card does not report reads "[N/A]" (power on some laptops);
    # that is one dash in the panel, not a reason to drop the whole reading.
    try:
        value = float(text)
    except ValueError:
        return None
    return int(value) if value.is_integer() else value


def parse_gpu(output):
    """The first GPU in nvidia-smi's CSV, as a dict, or None if it is not that."""
    lines = output.strip().splitlines()
    parts = [p.strip() for p in lines[0].split(",")] if lines else []
    if len(parts) != len(FIELDS):
        return None
    gpu = dict(zip(FIELDS, [parts[0]] + [_number(p) for p in parts[1:]]))
    return gpu if gpu["mem_total"] is not None else None


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True,
                          check=True, timeout=5).stdout


def query_gpu(run=_run):
    """One reading, or None when nvidia-smi is missing, fails or talks nonsense."""
    try:
        return parse_gpu(run(QUERY))
    except (OSError, subprocess.SubprocessError):
        return None


class GpuPoller:
    """Keeps the latest reading, refreshed every two seconds on a daemon thread.

    nvidia-smi takes tens of milliseconds to answer, which is a whole frame;
    the render loop only ever reads `latest`.
    """

    def __init__(self, period=2.0, run=_run):
        self.latest = None
        self._period, self._run = period, run
        threading.Thread(target=self._poll, daemon=True).start()

    def _poll(self):
        while True:
            self.latest = query_gpu(self._run)
            time.sleep(self._period)


class EventLog:
    """The last few things that happened, oldest first, for the panel to scroll."""

    def __init__(self, size=10):
        self._items = collections.deque(maxlen=size)

    def add(self, text):
        self._items.append({"t": round(time.time(), 2), "text": text})

    def recent(self):
        return list(self._items)


class Budget:
    """Milliseconds per pipeline stage, for the frames it is told to measure.

    Every lap stamps both clocks. A "cpu" stage is read from perf_counter: the
    camera read and the pipe write happen on the host. A "gpu" stage is read
    from CUDA events, because the host only queues that work and would time the
    queueing. The events are read in `finish`, after the frame's .cpu() has
    synchronised, so they are complete and reading them waits for nothing.
    """

    def __init__(self):
        self.ms = {}
        self._laps = None

    def start(self, measure):
        self._laps = [(None, "cpu", time.perf_counter(), None)] if measure else None

    def lap(self, name, clock="gpu"):
        """The stage called `name` ends here."""
        if self._laps is None:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._laps.append((name, clock, time.perf_counter(), event))

    def finish(self):
        laps, self._laps = self._laps, None
        if not laps:
            return
        # The event after the download is recorded behind the sync, not before
        # it. It is done long before the write to ffmpeg returns, but should it
        # not be, this frame is skipped rather than waited for.
        if not all(e.query() for _, clock, _, e in laps if e is not None and clock == "gpu"):
            return
        ms = {}
        for (_, _, t0, e0), (name, clock, t1, e1) in zip(laps, laps[1:]):
            ms[name] = round(e0.elapsed_time(e1) if clock == "gpu" else (t1 - t0) * 1000, 2)
        ms["total"] = round(sum(ms.values()), 2)
        self.ms = ms
