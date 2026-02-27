from pathlib import Path

import h5py
import numpy as np

from tools.drive_summary_core import analyze_recording_summary


def _write_base_recording(path: Path, n: int = 60) -> np.ndarray:
    t = np.linspace(0.0, 5.9, n, dtype=np.float64)
    zeros = np.zeros(n, dtype=np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("vehicle/timestamps", data=t)
        f.create_dataset("vehicle/speed", data=np.full(n, 4.0, dtype=np.float32))
        f.create_dataset("control/steering", data=zeros)
        f.create_dataset("control/lateral_error", data=zeros)
        f.create_dataset("control/heading_error", data=zeros)
        f.create_dataset("control/total_error", data=zeros)
        f.create_dataset("control/pid_integral", data=zeros)
        f.create_dataset("control/emergency_stop", data=np.zeros(n, dtype=np.int8))
        f.create_dataset("ground_truth/left_lane_line_x", data=np.full(n, -3.5, dtype=np.float32))
        f.create_dataset("ground_truth/right_lane_line_x", data=np.full(n, 3.5, dtype=np.float32))
    return t


def test_chassis_ground_good(tmp_path: Path) -> None:
    recording = tmp_path / "chassis_ground_good.h5"
    _write_base_recording(recording)
    with h5py.File(recording, "a") as f:
        n = len(f["vehicle/timestamps"])
        f.create_dataset("vehicle/chassis_ground_min_clearance_m", data=np.full(n, 0.08, dtype=np.float32))
        f.create_dataset(
            "vehicle/chassis_ground_effective_min_clearance_m", data=np.full(n, 0.08, dtype=np.float32)
        )
        f.create_dataset("vehicle/chassis_ground_clearance_m", data=np.full(n, 0.12, dtype=np.float32))
        f.create_dataset("vehicle/chassis_ground_penetration_m", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("vehicle/chassis_ground_contact", data=np.zeros(n, dtype=np.int8))
        f.create_dataset("vehicle/wheel_grounded_count", data=np.full(n, 4, dtype=np.int8))
        f.create_dataset("vehicle/wheel_colliders_ready", data=np.ones(n, dtype=np.int8))
        f.create_dataset("vehicle/force_fallback_active", data=np.zeros(n, dtype=np.int8))

    summary = analyze_recording_summary(recording)
    chassis = summary["chassis_ground"]
    assert chassis["availability"] == "available"
    assert chassis["health"] == "GOOD"
    assert float(chassis["contact_rate_pct"]) == 0.0
    assert float(chassis["penetration_max_m"]) == 0.0


def test_chassis_ground_poor_when_contact_and_penetration(tmp_path: Path) -> None:
    recording = tmp_path / "chassis_ground_poor.h5"
    t = _write_base_recording(recording)
    with h5py.File(recording, "a") as f:
        n = len(f["vehicle/timestamps"])
        contact = np.zeros(n, dtype=np.int8)
        penetration = np.zeros(n, dtype=np.float32)
        # Ignore first second; fail after transient window.
        bad_idx = t >= 1.0
        contact[bad_idx] = 1
        penetration[bad_idx] = 0.02
        clearance = np.full(n, 0.10, dtype=np.float32)
        clearance[bad_idx] = -0.01
        f.create_dataset("vehicle/chassis_ground_min_clearance_m", data=np.full(n, 0.08, dtype=np.float32))
        f.create_dataset(
            "vehicle/chassis_ground_effective_min_clearance_m", data=np.full(n, 0.08, dtype=np.float32)
        )
        f.create_dataset("vehicle/chassis_ground_clearance_m", data=clearance)
        f.create_dataset("vehicle/chassis_ground_penetration_m", data=penetration)
        f.create_dataset("vehicle/chassis_ground_contact", data=contact)
        f.create_dataset("vehicle/wheel_grounded_count", data=np.full(n, 4, dtype=np.int8))
        f.create_dataset("vehicle/wheel_colliders_ready", data=np.ones(n, dtype=np.int8))
        f.create_dataset("vehicle/force_fallback_active", data=np.zeros(n, dtype=np.int8))

    summary = analyze_recording_summary(recording)
    chassis = summary["chassis_ground"]
    assert chassis["availability"] == "available"
    assert chassis["health"] == "POOR"
    assert float(chassis["contact_rate_pct"]) > 0.5
    assert float(chassis["penetration_max_m"]) > 0.01


def test_chassis_ground_unavailable_for_legacy_recording(tmp_path: Path) -> None:
    recording = tmp_path / "chassis_ground_legacy.h5"
    _write_base_recording(recording)

    summary = analyze_recording_summary(recording)
    chassis = summary["chassis_ground"]
    assert chassis["availability"] == "unavailable"
    assert chassis["health"] == "UNKNOWN"
