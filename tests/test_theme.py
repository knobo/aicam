"""Which theme the panel opens in, that the palettes hold up, and the switch."""

import colorsys
import os
import subprocess
import time

import pytest

import hud

needs_display = pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="no X display")


def test_every_accent_comes_in_both_modes():
    assert set(hud.PALETTES) == {(a, m) for a in hud.ACCENTS for m in ("dark", "light")}


def test_every_palette_has_the_same_names():
    names = [set(p) for p in hud.PALETTES.values()]
    assert all(n == names[0] for n in names)


def test_dark_ice_is_the_hud_as_designed():
    assert hud.PALETTES["ice", "dark"]["CYAN"] == "#3ef0ff"
    assert hud.PALETTES["ice", "dark"]["BG"] == "#05080c"


@pytest.mark.parametrize("key", sorted(hud.PALETTES))
def test_text_reads_against_the_background(key):
    palette = hud.PALETTES[key]
    low = {n: round(hud.contrast(palette[n], palette["BG"]), 2) for n in hud.TEXT_NAMES}
    assert {n: c for n, c in low.items() if c < 4.5} == {}


def hue(colour):
    return colorsys.rgb_to_hls(*(int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)))[0] * 360


@pytest.mark.parametrize("key", sorted(hud.PALETTES))
def test_status_colours_are_not_the_accent(key):
    # LIVE red, MUTED amber and OK green must not read as decoration.
    palette = hud.PALETTES[key]
    for name in ("RED", "AMBER", "OK"):
        apart = abs(hue(palette[name]) - hue(palette["CYAN"])) % 360
        assert min(apart, 360 - apart) >= 30, name


def test_colours_are_distinct_within_a_palette():
    # apply_theme recolours plain widgets by looking their colour's name up.
    for palette in hud.PALETTES.values():
        assert len(set(palette.values())) == len(palette)


def gsettings(monkeypatch, stdout=None, error=None):
    def run(*_a, **_k):
        if error:
            raise error
        return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")
    monkeypatch.setattr(hud.subprocess, "run", run)


def test_gnome_scheme_picks_the_theme(monkeypatch):
    for out, theme in (("'prefer-dark'\n", "dark"), ("'prefer-light'\n", "light"),
                       ("'default'\n", "light"), ("", "dark")):
        gsettings(monkeypatch, out)
        assert hud.initial_theme() == (theme, "ice")


def test_no_gsettings_means_dark(monkeypatch):
    gsettings(monkeypatch, error=FileNotFoundError("gsettings"))
    assert hud.initial_theme() == ("dark", "ice")
    gsettings(monkeypatch, error=subprocess.TimeoutExpired("gsettings", 1))
    assert hud.initial_theme() == ("dark", "ice")


def test_saved_choice_wins(monkeypatch):
    gsettings(monkeypatch, "'prefer-dark'\n")
    hud.save_theme("light", "nostromo")
    assert hud.initial_theme() == ("light", "nostromo")


def test_first_version_file_and_unknown_accent(monkeypatch):
    gsettings(monkeypatch, "'prefer-dark'\n")
    hud.save_theme("dark", "ice")               # makes the folder
    for saved, theme in (('{"theme": "light"}', ("light", "ice")),
                         ('{"mode": "light", "accent": "plaid"}', ("light", "ice"))):
        with open(hud.config_path(), "w") as f:
            f.write(saved)
        assert hud.initial_theme() == theme


def test_corrupt_file_falls_back_to_gnome(monkeypatch):
    gsettings(monkeypatch, "'prefer-light'\n")
    path = hud.config_path()
    hud.save_theme("dark", "ice")           # makes the folder
    for junk in ("{not json", "[1, 2]", '{"mode": "purple"}'):
        with open(path, "w") as f:
            f.write(junk)
        assert hud.initial_theme() == ("light", "ice")


@pytest.fixture
def header(monkeypatch):
    tk = pytest.importorskip("tkinter")
    gsettings(monkeypatch, "'prefer-dark'\n")
    root = tk.Tk()
    hud.setup(root)
    h = hud.Header(root)
    h.grid(sticky="ew")
    root.update()
    picked = []
    monkeypatch.setattr(h, "pick_theme", lambda: (picked.append(1),
                                                  setattr(h, "_held", None)))
    h.picked = picked
    yield h
    root.destroy()


@needs_display
def test_a_click_toggles(header):
    header.press()
    header.tick(time.monotonic() + 0.1)
    header.release()
    assert (hud.MODE, header.picked) == ("light", [])
    assert hud.saved_theme() == ("light", "ice")


@needs_display
def test_a_hold_picks_and_does_not_toggle(header):
    header.press()
    header.tick(time.monotonic() + 0.6)
    header.release()
    assert (hud.MODE, header.picked) == ("dark", [1])
