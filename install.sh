#!/usr/bin/env bash
# Desktop integration for aicam, per user, no root.
#
# Nothing is copied: the venv, the model weights and the backgrounds live in
# this checkout and are far too big to duplicate. The launchers in ~/.local/bin
# and the desktop entry point straight back here, so `git pull` is the upgrade
# and `./install.sh --uninstall` leaves the checkout untouched.
set -euo pipefail

here="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
data="${XDG_DATA_HOME:-$HOME/.local/share}"
bindir="$HOME/.local/bin"
appdir="$data/applications"
icondir="$data/icons/hicolor/scalable/apps"
unitdir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

desktop="$appdir/aicam.desktop"
icon="$icondir/aicam.svg"
unit="$unitdir/aicam.service"
launchers=(aicam aicam-gui aicamctl)

usage() {
    cat <<USAGE
usage: ./install.sh [--service] [--uninstall]

  (no options)  launchers in $bindir, desktop entry and icon
  --service     also a systemd --user unit, so aicam starts with the session
  --uninstall   remove everything this script installed
USAGE
}

action=install
service=0
while [ $# -gt 0 ]; do
    case "$1" in
        --service)   service=1 ;;
        --uninstall) action=uninstall ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "install.sh: unknown option $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

# Only ever remove a file that points back at this checkout: someone may have a
# second clone installed, and clobbering its entry would be rude.
ours() { [ -e "$1" ] && grep -qF "$here" "$1"; }

if [ "$action" = uninstall ]; then
    for name in "${launchers[@]}"; do
        link="$bindir/$name"
        if [ -L "$link" ] && [ "$(readlink -f "$link")" = "$here/$name" ]; then
            rm -f "$link"; echo "removed $link"
        fi
    done
    # if-blocks, not `a && b`: under `set -e` a false test would end the script
    if ours "$desktop"; then rm -f "$desktop"; echo "removed $desktop"; fi
    if [ -e "$icon" ]; then rm -f "$icon"; echo "removed $icon"; fi
    if ours "$unit"; then
        systemctl --user disable --now aicam.service >/dev/null 2>&1 || true
        rm -f "$unit"; echo "removed $unit"
        systemctl --user daemon-reload >/dev/null 2>&1 || true
    fi
    command -v update-desktop-database >/dev/null && update-desktop-database "$appdir" || true
    exit 0
fi

mkdir -p "$bindir" "$appdir" "$icondir"

for name in "${launchers[@]}"; do
    ln -sfn "$here/$name" "$bindir/$name"
    echo "$bindir/$name -> $here/$name"
done

install -m 644 "$here/icons/aicam.svg" "$icon"
echo "$icon"

sed -e "s|@EXEC@|$here/aicam-gui|g" -e "s|@HERE@|$here|g" \
    "$here/aicam.desktop.in" > "$desktop"
chmod 644 "$desktop"
echo "$desktop"

if command -v desktop-file-validate >/dev/null; then
    desktop-file-validate "$desktop"
fi
command -v update-desktop-database >/dev/null && update-desktop-database "$appdir" || true

if [ "$service" = 1 ]; then
    mkdir -p "$unitdir"
    sed -e "s|@EXEC@|$here/aicam|g" "$here/aicam.service.in" > "$unit"
    echo "$unit"
    systemctl --user daemon-reload
    echo "enable it with: systemctl --user enable --now aicam.service"
fi

case ":$PATH:" in
    *":$bindir:"*) ;;
    *) echo "note: $bindir is not on your PATH; the menu entry works regardless." ;;
esac

echo "done. 'aicam' is now in the application menu."
