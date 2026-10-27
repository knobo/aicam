"""Where `aicamctl snapshot` puts the picture - none of it needs CUDA."""

import subprocess

import cv2
import numpy as np

import aicam


def test_the_desktop_pictures_dir_is_used(tmp_path):
    assert aicam.pictures_dir(run=lambda: f"{tmp_path}/Bilder\n") == f"{tmp_path}/Bilder"
    assert (tmp_path / "Bilder").is_dir()


def test_no_xdg_user_dir_means_home_pictures(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    def missing():
        raise FileNotFoundError("xdg-user-dir")

    def fails():
        raise subprocess.CalledProcessError(1, "xdg-user-dir")

    for run in (missing, fails, lambda: ""):
        assert aicam.pictures_dir(run=run) == f"{tmp_path}/Pictures"


def test_a_pictures_dir_that_cannot_be_made_means_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "file").write_text("")
    assert aicam.pictures_dir(run=lambda: f"{tmp_path}/file/Pictures") == str(tmp_path)


def test_a_snapshot_never_overwrites_one_from_the_same_second(tmp_path):
    names = []
    for _ in range(3):
        with aicam.open_new_snapshot(str(tmp_path), now=0) as f:
            names.append(f.name.rsplit("/", 1)[1])
    stem = names[0][:-len(".png")]
    assert stem.startswith("aicam-") and len(stem) == len("aicam-YYYYmmdd-HHMMSS")
    assert names == [f"{stem}.png", f"{stem}-2.png", f"{stem}-3.png"]


def test_the_png_is_the_rgb_frame(tmp_path):
    rgb = np.zeros((4, 6, 3), np.uint8)
    rgb[..., 0] = 255                                   # pure red
    saved = []
    aicam.save_snapshot(rgb, str(tmp_path / "x.png"), saved.append, pytest_fail)
    assert saved == [str(tmp_path / "x.png")]
    assert (cv2.imread(saved[0])[..., ::-1] == rgb).all()


def test_a_default_snapshot_lands_in_the_pictures_dir(tmp_path):
    saved = []
    aicam.save_snapshot(np.zeros((4, 6, 3), np.uint8), True, saved.append, pytest_fail,
                        run=lambda: str(tmp_path))
    assert saved[0].startswith(f"{tmp_path}/aicam-") and saved[0].endswith(".png")


def test_a_failed_write_is_said_not_raised(tmp_path):
    said = []
    aicam.save_snapshot(np.zeros((4, 6, 3), np.uint8), str(tmp_path / "no" / "x.png"),
                        pytest_fail, said.append)
    assert said and said[0].startswith("snapshot failed:")


def pytest_fail(*args):
    raise AssertionError(f"unexpected call: {args}")
