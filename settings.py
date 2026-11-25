"""Startup defaults: what aicam starts with when no flag says otherwise.

A flat JSON object in ~/.config/aicam/settings.json, keyed by the argparse
dest of each flag (`auto_frame`, not `--auto-frame`). aicam.py hands it to the
parser as defaults before parsing, so a flag on the command line still wins:
`mirror: true` in the file and `--no-mirror` on the line gives no mirror.

The panel starts aicam with no flags at all, so for the panel this file *is*
the configuration.

Only keys in KEYS are read, and only with a value of the right type. A typo or
a stale key from an older version is ignored rather than refusing to start the
camera ten seconds before a meeting.
"""

import json
import os
import tempfile

import history

# What may be set, and what type its value must be. The panel's STARTUP section
# edits the first group; "use current" fills in the rest from a running aicam.
KEYS = {
    "preview": bool,
    "mirror": bool,
    "gestures": bool,
    "auto_frame": bool,
    "gesture_face_guard": bool,
    "mode": str,
    "background": (str, type(None)),
    "aperture": float,
    "light": float,
    "rim": float,
    "highlights": float,
    "blur": float,
    "parallax": float,
    "frame_zoom": float,
    "ink_width": float,
    "ink": str,
    "tool": str,
}
# The ones the panel shows as toggles, in the order it shows them.
TOGGLES = [("preview", "PREVIEW"), ("mirror", "MIRROR"), ("gestures", "GESTURES"),
           ("auto_frame", "AUTO-FRAME"), ("gesture_face_guard", "FACE GUARD")]
# aicam.py's own defaults for what the panel shows, so a key missing from the
# file is drawn the way aicam will behave. Spelled out rather than read from
# build_parser(): that would load torch into the panel. tests/test_settings.py
# keeps the two in step.
DEFAULTS = {"preview": False, "mirror": False, "gestures": True,
            "auto_frame": False, "gesture_face_guard": True,
            "mode": "blur", "background": None}


def default_path():
    return os.path.join(os.path.dirname(history.default_path()), "settings.json")


def clean(raw):
    """Keep the keys aicam knows, with values of the type it expects."""
    if not isinstance(raw, dict):
        return {}
    kept = {}
    for key, value in raw.items():
        kind = KEYS.get(key)
        if kind is None:
            continue
        if kind is float:
            # JSON has no int/float distinction worth keeping; bool is an int
            # in Python and is not a slider position.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            value = float(value)
        elif kind is bool:
            if not isinstance(value, bool):
                continue
        elif not isinstance(value, kind):
            continue
        if key == "background" and value == "":
            value = None
        kept[key] = value
    return kept


def load(path=None):
    """The saved defaults; {} when there are none or the file is unreadable."""
    try:
        with open(path or default_path()) as f:
            return clean(json.load(f))
    except (OSError, ValueError):
        return {}


def save(values, path=None):
    """Write the defaults. Never raises: the camera must still start."""
    path = path or default_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(clean(values), f, indent=2, sort_keys=True)
        os.replace(temp, path)
        return True
    except OSError:
        return False


def update(changes, path=None):
    """Merge changes into the file; a value of None for a non-background key
    removes it, so aicam's own default applies again."""
    values = load(path)
    for key, value in changes.items():
        if value is None and key != "background":
            values.pop(key, None)
        else:
            values[key] = value
    save(values, path)
    return load(path)


def apply(parser, values):
    """Make the saved values the parser's defaults.

    A choice the parser would reject (a mode that no longer exists, say) is
    dropped here: argparse does not check defaults against `choices`, and the
    pipeline would fail much later and much less clearly.
    """
    dests = {a.dest: a for a in parser._actions}
    usable = {}
    for key, value in values.items():
        action = dests.get(key)
        if action is None:
            continue
        if action.choices is not None and value not in action.choices:
            continue
        usable[key] = value
    parser.set_defaults(**usable)
    return usable
