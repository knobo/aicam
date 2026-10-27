"""Which theme the panel opens in, and that the palettes agree. No display needed."""

import subprocess

import hud


def test_every_palette_has_the_same_names():
    names = [set(p) for p in hud.PALETTES.values()]
    assert all(n == names[0] for n in names)


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
        assert hud.initial_theme() == theme


def test_no_gsettings_means_dark(monkeypatch):
    gsettings(monkeypatch, error=FileNotFoundError("gsettings"))
    assert hud.initial_theme() == "dark"
    gsettings(monkeypatch, error=subprocess.TimeoutExpired("gsettings", 1))
    assert hud.initial_theme() == "dark"


def test_saved_choice_wins(monkeypatch):
    gsettings(monkeypatch, "'prefer-dark'\n")
    hud.save_theme("light")
    assert hud.initial_theme() == "light"


def test_corrupt_file_falls_back_to_gnome(monkeypatch):
    gsettings(monkeypatch, "'prefer-light'\n")
    path = hud.config_path()
    hud.save_theme("dark")                  # makes the folder
    for junk in ("{not json", "[1, 2]", '{"theme": "purple"}'):
        with open(path, "w") as f:
            f.write(junk)
        assert hud.initial_theme() == "light"
