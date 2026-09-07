"""Recently used backgrounds, so switching back is one click.

A flat list in ~/.config/aicam/history.json, newest first. Only sources that
actually opened are written, so a typo never ends up in the menu.

Files that have since been deleted are dropped when the list is read; screens
and streams are names rather than paths and cannot be checked that way, so they
stay until they fall off the end.
"""

import json
import os
import tempfile
import time

LIMIT = 12


def default_path():
    config = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(config, "aicam", "history.json")


def kind_of(source):
    """desktop, stream, url or file - what the menu needs to know."""
    if source.startswith("desktop"):
        return "desktop"
    if source.startswith("yt:"):
        return "stream"
    if "://" in source:
        return "url"
    return "file"


def label(source):
    """A name short enough for a menu entry."""
    if kind_of(source) == "file":
        return os.path.basename(source)
    return source if len(source) <= 60 else source[:57] + "…"


def load(path=None):
    """The remembered sources, newest first, minus anything gone missing."""
    path = path or default_path()
    try:
        with open(path) as f:
            entries = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []
    if not isinstance(entries, list):
        return []

    alive = []
    for entry in entries:
        if not isinstance(entry, dict) or "source" not in entry:
            continue
        if entry.get("kind") == "file" and not os.path.exists(entry["source"]):
            continue
        alive.append(entry)
    return alive[:LIMIT]


def remember(source, path=None):
    """Put a source at the front of the list. Never raises: this is a nicety."""
    path = path or default_path()
    entries = [e for e in load(path) if e.get("source") != source]
    entries.insert(0, {"source": source, "kind": kind_of(source),
                       "used": time.strftime("%Y-%m-%d %H:%M:%S")})
    entries = entries[:LIMIT]

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Written to a neighbour and renamed, so an interrupted write cannot
        # leave half a file where the list used to be.
        fd, temp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(entries, f, indent=2)
        os.replace(temp, path)
    except OSError:
        pass
    return entries


def pick(index, path=None):
    """The nth remembered source, counting from 1 as the listing prints it."""
    entries = load(path)
    if index < 1 or index > len(entries):
        raise ValueError(f"there is no history entry {index}; "
                         f"{len(entries)} remembered")
    return entries[index - 1]["source"]
