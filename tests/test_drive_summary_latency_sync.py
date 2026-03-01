from pathlib import Path

import h5py
import numpy as np

from tools.drive_summary_core import analyze_recording_summary


def _write_latency_recording(
    path: Path,
    *,
    include_e2e: bool,
    traj_offset_s: float,
    control_offset_s: float,
    irregular_indices: tuple[int, ...] = (),
    irregular_gap_s: float = 0.22,
) -> None:
    n = 121
    t = np.linspace(0.0, 4.0, n, dtype=np.float64)
    for idx in irregular_indices:
        if 0 < idx < n:
            t[idx:] += float(irregular_gap_s)
    zeros = np.zeros(n, dtype=np.float32)

    with h5py.File(path, "w") as f:
        f.create_dataset("vehicle/timestamps", data=t)
        f.create_dataset("vehicle/speed", data=np.full(n, 5.0, dtype=np.float32))
        f.create_dataset("camera/timestamps", data=t)
        f.create_dataset("trajectory/timestamps", data=t + traj_offset_s)
        f.create_dataset("control/timestamps", data=t + control_offset_s)
        f.create_dataset("control/steering", data=zeros)
        f.create_dataset("control/lateral_error", data=zeros)
        f.create_dataset("control/heading_error", data=zeros)
        f.create_dataset("control/total_error", data=zeros)
        f.create_dataset("control/pid_integral", data=zeros)
        f.create_dataset("control/emergency_stop", data=np.zeros(n, dtype=np.int8))
        f.create_dataset("ground_truth/left_lane_line_x", data=np.full(n, -3.5, dtype=np.float32))
        f.create_dataset("ground_truth/right_lane_line_x", data=np.full(n, 3.5, dtype=np.float32))

        if include_e2e:
            f.create_dataset("control/e2e_latency_ms", data=np.linspace(24.0, 42.0, n, dtype=np.float32))
            mode_dtype = h5py.string_dtype(encoding="utf-8", length=64)
            f.create_dataset(
                "control/e2e_latency_mode",
                data=np.array(["input_ready_to_command_sent"] * n, dtype=mode_dtype),
            )


def test_latency_sync_available_and_passing(tmp_path: Path) -> None:
    recording = tmp_path / "latency_sync_pass.h5"
    _write_latency_recording(
        recording,
        include_e2e=True,
        traj_offset_s=0.005,
        control_offset_s=0.004,
    )

    summary = analyze_recording_summary(recording)
    latency_sync = summary["latency_sync"]
    assert latency_sync["e2e"]["availability"] == "available"
    assert latency_sync["e2e"]["pass"] is True
    assert latency_sync["sync_alignment"]["availability"] == "available"
    assert latency_sync["sync_alignment"]["pass"] is True
    assert latency_sync["cadence"]["availability"] == "available"
    assert latency_sync["cadence"]["pass"] is True
    assert latency_sync["cadence"]["tuning_valid"] is True
    assert latency_sync["cadence"]["status"] == "good"
    assert latency_sync["overall"]["status"] == "good"


def test_latency_sync_legacy_e2e_unavailable(tmp_path: Path) -> None:
    recording = tmp_path / "latency_sync_legacy.h5"
    _write_latency_recording(
        recording,
        include_e2e=False,
        traj_offset_s=0.006,
        control_offset_s=0.006,
    )

    summary = analyze_recording_summary(recording)
    latency_sync = summary["latency_sync"]
    assert latency_sync["e2e"]["availability"] == "unavailable"
    assert latency_sync["e2e"]["pass"] is None
    assert latency_sync["sync_alignment"]["availability"] == "available"
    assert latency_sync["cadence"]["availability"] == "available"


def test_latency_sync_detects_alignment_failure(tmp_path: Path) -> None:
    recording = tmp_path / "latency_sync_fail.h5"
    _write_latency_recording(
        recording,
        include_e2e=True,
        traj_offset_s=0.080,
        control_offset_s=0.090,
    )

    summary = analyze_recording_summary(recording)
    latency_sync = summary["latency_sync"]
    assert latency_sync["sync_alignment"]["pass"] is False
    assert latency_sync["overall"]["status"] == "poor"
    assert latency_sync["overall"]["issues"]


def test_latency_sync_detects_cadence_failure(tmp_path: Path) -> None:
    recording = tmp_path / "latency_sync_cadence_fail.h5"
    _write_latency_recording(
        recording,
        include_e2e=True,
        traj_offset_s=0.004,
        control_offset_s=0.004,
        irregular_indices=(6, 12, 18, 24, 30),
        irregular_gap_s=0.22,
    )

    summary = analyze_recording_summary(recording)
    latency_sync = summary["latency_sync"]
    cadence = latency_sync["cadence"]
    assert cadence["availability"] == "available"
    assert cadence["pass"] is False
    assert cadence["tuning_valid"] is False
    assert cadence["status"] == "poor"
    assert "irregular_rate_max" in cadence["limits"]
    assert "severe_irregular_rate_max" in cadence["limits"]
    assert latency_sync["overall"]["status"] == "poor"
