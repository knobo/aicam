"""Turning `yt:<url>` into a stream the decoder can open.

yt-dlp does the work; we only pick the format and hand the resulting URL to the
same VideoBackground that plays a local file.

Two facts shape this. The format has to be H.264: left to itself yt-dlp picks
AV1, which the OpenCV ffmpeg build cannot decode, and you get a black background
with megabytes of `Missing Sequence Header` on stderr rather than an error. And
the URL it hands back is signed and short-lived - six hours in the case measured
- so it is resolved again whenever the stream ends rather than being kept.
"""

import subprocess

PREFIX = "yt:"

# Video only, H.264, at most 720p: the background is scaled to the frame anyway,
# and a smaller stream starts faster.
FORMAT = "bv[vcodec^=avc1][height<=720]/bv[vcodec^=avc1]/b[vcodec^=avc1]"


def is_stream_spec(path):
    """True for `yt:https://...`."""
    return path.startswith(PREFIX)


def url_of(spec):
    return spec[len(PREFIX):] if is_stream_spec(spec) else spec


def build_command(url):
    return ["yt-dlp", "-g", "-f", FORMAT, url]


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True,
                          check=True, timeout=120).stdout


def resolve(spec, run=_run):
    """Ask yt-dlp for a playable URL. Raises RuntimeError with the reason."""
    url = url_of(spec)
    try:
        output = run(build_command(url))
    except FileNotFoundError:
        raise RuntimeError("yt-dlp is not installed; try: pipx install yt-dlp")
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip().splitlines()
        raise RuntimeError(f"yt-dlp could not open {url}: "
                           f"{detail[-1] if detail else 'no reason given'}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"yt-dlp timed out on {url}")

    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"yt-dlp found no playable video for {url}")
    return lines[-1]
