"""Startup defaults: the file, and how aicam's parser takes it."""

import json

import pytest

import settings


def test_round_trip_keeps_known_keys_of_the_right_type(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"mirror": True, "aperture": 60, "mode": "cinematic",
                                "background": "shapes:dusk", "bogus": 1,
                                "gestures": "yes", "blur": True}))
    assert settings.load(path) == {"mirror": True, "aperture": 60.0,
                                   "mode": "cinematic", "background": "shapes:dusk"}


def test_a_missing_or_corrupt_file_is_no_defaults(tmp_path):
    assert settings.load(tmp_path / "absent.json") == {}
    (tmp_path / "bad.json").write_text("{not json")
    assert settings.load(tmp_path / "bad.json") == {}
    (tmp_path / "list.json").write_text("[1, 2]")
    assert settings.load(tmp_path / "list.json") == {}


def test_update_merges_and_none_restores_the_builtin_default(tmp_path):
    path = tmp_path / "settings.json"
    settings.update({"mirror": True, "preview": True}, path)
    assert settings.update({"preview": None, "background": None}, path) == \
        {"mirror": True, "background": None}


def test_the_command_line_beats_the_file():
    aicam = pytest.importorskip("aicam")
    parser = aicam.build_parser()
    used = settings.apply(parser, {"mirror": True, "preview": True, "mode": "nonsense"})
    assert "mode" not in used                 # not a choice aicam has
    args = parser.parse_args(["--no-mirror"])
    assert (args.mirror, args.preview, args.mode) == (False, True, "blur")


def test_panel_defaults_match_aicam():
    aicam = pytest.importorskip("aicam")
    args = aicam.build_parser().parse_args([])
    assert {k: getattr(args, k) for k in settings.DEFAULTS} == settings.DEFAULTS
    dests = {a.dest for a in aicam.build_parser()._actions}
    assert set(settings.KEYS) <= dests
