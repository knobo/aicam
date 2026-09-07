"""The list of recently used backgrounds, kept in ~/.config/aicam."""

import json
import os

import history


def test_it_lives_under_the_config_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert history.default_path() == str(tmp_path / "aicam" / "history.json")


def test_nothing_remembered_yet_is_an_empty_list(tmp_path):
    assert history.load(str(tmp_path / "history.json")) == []


def test_a_remembered_source_comes_back(tmp_path, still_image):
    path = str(tmp_path / "history.json")
    history.remember(still_image, path)
    assert [e["source"] for e in history.load(path)] == [still_image]


def test_the_newest_is_first(tmp_path, still_image, marked_clip):
    path = str(tmp_path / "history.json")
    history.remember(still_image, path)
    history.remember(marked_clip, path)
    assert [e["source"] for e in history.load(path)] == [marked_clip, still_image]


def test_using_something_again_moves_it_to_the_front_without_duplicating_it(
        tmp_path, still_image, marked_clip):
    path = str(tmp_path / "history.json")
    history.remember(still_image, path)
    history.remember(marked_clip, path)
    history.remember(still_image, path)
    assert [e["source"] for e in history.load(path)] == [still_image, marked_clip]


def test_the_list_is_capped(tmp_path):
    path = str(tmp_path / "history.json")
    for i in range(20):
        history.remember(f"desktop:screen{i}", path)
    entries = history.load(path)
    assert len(entries) == history.LIMIT
    assert entries[0]["source"] == "desktop:screen19"


def test_files_that_are_gone_are_dropped(tmp_path, still_image):
    path = str(tmp_path / "history.json")
    missing = str(tmp_path / "deleted.mp4")
    missing_file = open(missing, "w")
    missing_file.close()
    history.remember(missing, path)
    history.remember(still_image, path)
    os.unlink(missing)
    assert [e["source"] for e in history.load(path)] == [still_image]


def test_screens_and_streams_survive_because_they_are_not_files(tmp_path):
    path = str(tmp_path / "history.json")
    history.remember("desktop:left", path)
    history.remember("yt:https://youtu.be/abc", path)
    history.remember("https://example.com/clip.mp4", path)
    assert len(history.load(path)) == 3


def test_a_corrupt_file_is_not_a_crash(tmp_path):
    path = str(tmp_path / "history.json")
    with open(path, "w") as f:
        f.write("{not json at all")
    assert history.load(path) == []


def test_each_entry_says_what_kind_of_source_it_is(tmp_path, still_image):
    path = str(tmp_path / "history.json")
    history.remember("desktop:left", path)
    history.remember("yt:https://youtu.be/abc", path)
    history.remember(still_image, path)
    kinds = {e["source"]: e["kind"] for e in history.load(path)}
    assert kinds["desktop:left"] == "desktop"
    assert kinds["yt:https://youtu.be/abc"] == "stream"
    assert kinds[still_image] == "file"


def test_labels_are_short_enough_for_a_menu(tmp_path, still_image):
    long_url = "yt:https://www.youtube.com/watch?v=" + "a" * 200
    assert history.label(still_image) == os.path.basename(still_image)
    assert history.label("desktop:left") == "desktop:left"
    assert len(history.label(long_url)) <= 60


def test_writing_leaves_valid_json_behind(tmp_path):
    path = str(tmp_path / "history.json")
    history.remember("desktop:left", path)
    with open(path) as f:
        assert isinstance(json.load(f), list)


def test_picking_by_number_uses_the_listed_order(tmp_path, still_image):
    path = str(tmp_path / "history.json")
    history.remember("desktop:left", path)
    history.remember(still_image, path)
    assert history.pick(1, path) == still_image      # 1 is the most recent
    assert history.pick(2, path) == "desktop:left"


def test_picking_past_the_end_says_so(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        history.pick(3, str(tmp_path / "history.json"))
