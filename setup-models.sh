#!/usr/bin/env bash
# Fetch the model weights that are not bundled (they are not ours to ship).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$here/models"
url=https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
echo "fetching hand_landmarker.task"
curl -fsSL -o "$here/models/hand_landmarker.task" "$url"
echo "done. RobustVideoMatting and MiDaS download themselves on first run."
