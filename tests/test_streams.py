"""Resolving `yt:<url>` into something the decoder can open."""

import os

import pytest

import streams


def fake_run(output):
    def run(cmd):
        return output
    return run


def test_stream_specs_are_recognised():
    assert streams.is_stream_spec("yt:https://www.youtube.com/watch?v=abc")
    assert not streams.is_stream_spec("https://example.com/clip.mp4")
    assert not streams.is_stream_spec("/home/user/clips/beach.mp4")


def test_the_url_is_taken_from_the_spec():
    cmd = streams.build_command("https://www.youtube.com/watch?v=abc")
    assert cmd[-1] == "https://www.youtube.com/watch?v=abc"
    assert "-g" in cmd


def test_h264_is_requested_explicitly():
    # Left to itself yt-dlp picks AV1, which the OpenCV ffmpeg build cannot
    # decode: a black background and megabytes of errors on stderr.
    assert "avc1" in " ".join(streams.build_command("https://youtu.be/abc"))


def test_the_last_line_of_output_is_the_stream():
    # yt-dlp prints one URL per stream; with a video-only format that is the last.
    url = streams.resolve("yt:https://youtu.be/abc",
                          run=fake_run("https://a.example/audio\nhttps://b.example/video\n"))
    assert url == "https://b.example/video"


def test_an_empty_answer_is_an_error():
    with pytest.raises(RuntimeError):
        streams.resolve("yt:https://youtu.be/abc", run=fake_run("\n"))


def test_a_failing_resolver_names_yt_dlp():
    def boom(cmd):
        raise FileNotFoundError("yt-dlp")
    with pytest.raises(RuntimeError) as exc:
        streams.resolve("yt:https://youtu.be/abc", run=boom)
    assert "yt-dlp" in str(exc.value)


@pytest.mark.skipif(not os.environ.get("AICAM_NETWORK_TESTS"),
                    reason="set AICAM_NETWORK_TESTS=1 to hit YouTube")
def test_resolving_a_real_video():
    url = streams.resolve("yt:https://www.youtube.com/watch?v=aqz-KE-bpKQ")
    assert url.startswith("http")
