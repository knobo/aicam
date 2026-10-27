"""GPU readings and the event log, which the panel shows - none of it needs CUDA."""

import subprocess

import telemetry


def fake_run(output):
    def run(cmd):
        return output
    return run


def test_a_reading_is_parsed_into_numbers():
    gpu = telemetry.query_gpu(run=fake_run("NVIDIA GeForce RTX 4070 SUPER, 9, 31, 9.73, 3813, 12282\n"))
    assert gpu == {"name": "NVIDIA GeForce RTX 4070 SUPER", "util": 9, "temp": 31,
                   "power": 9.73, "mem_used": 3813, "mem_total": 12282}


def test_only_the_first_gpu_is_reported():
    gpu = telemetry.parse_gpu("A, 1, 30, 10, 100, 1000\nB, 2, 40, 20, 200, 2000\n")
    assert gpu["name"] == "A"


def test_a_field_the_card_does_not_report_is_a_dash_not_a_failure():
    gpu = telemetry.parse_gpu("Laptop GPU, 5, 45, [N/A], 500, 4096\n")
    assert gpu["power"] is None and gpu["mem_total"] == 4096


def test_no_nvidia_smi_is_no_gpu():
    def missing(cmd):
        raise FileNotFoundError("nvidia-smi")
    assert telemetry.query_gpu(run=missing) is None


def test_a_failing_nvidia_smi_is_no_gpu():
    def fails(cmd):
        raise subprocess.CalledProcessError(9, cmd, stderr="NVIDIA-SMI has failed")
    assert telemetry.query_gpu(run=fails) is None


def test_a_hung_nvidia_smi_is_no_gpu():
    def hangs(cmd):
        raise subprocess.TimeoutExpired(cmd, 5)
    assert telemetry.query_gpu(run=hangs) is None


def test_junk_is_no_gpu():
    for junk in ["", "\n", "Failed to initialize NVML: Driver/library version mismatch\n",
                 "a, b, c, d, e, f\n"]:
        assert telemetry.query_gpu(run=fake_run(junk)) is None, junk


def test_events_come_oldest_first_with_a_time():
    log = telemetry.EventLog()
    log.add("gesture: hearts")
    log.add("background: dusk.jpg")
    events = log.recent()
    assert [e["text"] for e in events] == ["gesture: hearts", "background: dusk.jpg"]
    assert events[0]["t"] <= events[1]["t"]


def test_only_the_last_ten_events_are_kept():
    log = telemetry.EventLog()
    for i in range(25):
        log.add(f"trigger: {i}")
    assert [e["text"] for e in log.recent()] == [f"trigger: {i}" for i in range(15, 25)]

