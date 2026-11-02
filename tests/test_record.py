"""Recording what aicam sees - none of it needs CUDA."""

import json
import subprocess
import time

import numpy as np
import torch

import aicam


def test_a_path_without_one_given_lands_in_the_videos_dir(tmp_path):
    path = aicam.new_recording_path("alpha", True, run=lambda: f"{tmp_path}/Videoer\n")
    assert path.startswith(f"{tmp_path}/Videoer/aicam-") and path.endswith(".webm")
    assert aicam.new_recording_path("green", "/x/y.mp4") == "/x/y.mp4"


def test_green_keeps_the_subject_and_paints_the_rest():
    fgr = torch.full((1, 3, 2, 2), 0.5)
    pha = torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]])
    out = aicam.record_frame("green", fgr, pha)
    assert out[0, :, 0, 0].tolist() == [0.5, 0.5, 0.5]
    assert out[0, :, 0, 1].tolist() == [0.0, 1.0, 0.0]
    assert out[0, :, 1, 0].tolist() == [0.25, 0.75, 0.25]


def test_alpha_carries_the_matte_as_a_fourth_channel():
    fgr, pha = torch.rand(1, 3, 2, 2), torch.rand(1, 1, 2, 2)
    out = aicam.record_frame("alpha", fgr, pha)
    assert out.shape == (1, 4, 2, 2) and torch.equal(out[:, 3:], pha)


def _probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=codec_name,pix_fmt,width,height",
                          "-show_entries", "stream_tags=alpha_mode",
                          "-count_frames", "-show_entries", "stream=nb_read_frames",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)["streams"][0]


def _record(mode, path, channels):
    warnings = []
    rec = aicam.Recorder(mode, str(path), 64, 32, 10, warnings.append)
    frame = np.zeros((32, 64, channels), np.uint8)
    for _ in range(10):
        rec.write(frame)
        time.sleep(0.1)
    rec.close()
    assert warnings == []
    return _probe(path)


def test_alpha_recording_is_vp9_with_alpha(tmp_path):
    info = _record("alpha", tmp_path / "a.webm", 4)
    assert info["codec_name"] == "vp9" and info.get("tags", {}).get("alpha_mode") == "1"
    assert (info["width"], info["height"]) == (64, 32)


def test_green_recording_is_h264_at_the_speed_it_happened(tmp_path):
    info = _record("green", tmp_path / "g.mp4", 3)
    assert info["codec_name"] == "h264"
    # One second of wall clock at 10 fps, give or take the first and last frame.
    assert 8 <= int(info["nb_read_frames"]) <= 12
