"""The preview must never hold up the stream, even when ffplay stops reading."""

import subprocess
import time

import numpy as np

import aicam


def test_stalled_preview_does_not_block():
    # `sleep` never reads stdin, like ffplay behind a covered window.
    preview = aicam.Preview(subprocess.Popen(["sleep", "30"], stdin=subprocess.PIPE))
    frame = np.zeros((1080, 1920, 3), np.uint8)     # far bigger than a pipe buffer
    t0 = time.monotonic()
    for _ in range(30):
        preview.show(frame)
    assert time.monotonic() - t0 < 0.5
    preview.close()
    assert preview.poll() is not None
