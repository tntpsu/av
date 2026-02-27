from pathlib import Path

import h5py
import numpy as np

from tools.drive_summary_core import analyze_recording_summary


def _write_hotspot_recording(path: Path, *, with_contact: bool) -> None:
    n = 20
    t = np.arange(n, dtype=np.float64) * 0.033
    t[8:] += 0.24  # Inject one large timestamp gap before the main spike.

    speed = np.zeros(n, dtype=np.float32)
    speed[:8] = np.linspace(0.0, 1.4, 8, dtype=np.float32)
    speed[8] = 3.0
    speed[9:] = 3.0 + np.arange(n - 9, dtype=np.float32) * 0.08

    chassis_contact = np.zeros(n, dtype=np.int8)
    chassis_pen = np.zeros(n, dtype=np.float32)
    chassis_clear = np.full(n, 0.12, dtype=np.float32)
    if with_contact:
        chassis_contact[:] = 1
        chassis_pen[:] = 0.02
        chassis_clear[:] = -0.005

    with h5py.File(path, "w") as f:
        f.create_dataset("vehicle/timestamps", data=t)
        f.create_dataset("vehicle/speed", data=speed)
        f.create_dataset("vehicle/chassis_ground_contact", data=chassis_contact)
        f.create_dataset("vehicle/chassis_ground_penetration_m", data=chassis_pen)
        f.create_dataset("vehicle/chassis_ground_clearance_m", data=chassis_clear)

        f.create_dataset("control/steering", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/lateral_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/heading_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/total_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/pid_integral", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/emergency_stop", data=np.zeros(n, dtype=np.int8))
        f.create_dataset("control/throttle", data=np.full(n, 0.16, dtype=np.float32))
        f.create_dataset("control/brake", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/target_speed_final", data=np.full(n, 4.0, dtype=np.float32))
        f.create_dataset("control/longitudinal_accel_capped", data=np.zeros(n, dtype=np.int8))
        f.create_dataset("control/longitudinal_jerk_capped", data=np.zeros(n, dtype=np.int8))

        # Ground-truth boundaries keep safety metrics stable and in-lane.
        f.create_dataset("ground_truth/left_lane_line_x", data=np.full(n, -3.5, dtype=np.float32))
        f.create_dataset("ground_truth/right_lane_line_x", data=np.full(n, 3.5, dtype=np.float32))


def test_hotspot_attribution_detects_timestamp_gap_artifact(tmp_path: Path) -> None:
    recording = tmp_path / "hotspot_gap.h5"
    _write_hotspot_recording(recording, with_contact=False)

    summary = analyze_recording_summary(recording)
    hotspot = summary["comfort"]["hotspot_attribution"]

    assert hotspot["availability"] == "available"
    entries = hotspot.get("entries", [])
    assert entries
    assert any(e.get("attribution") == "timestamp_gap_derivative_artifact" for e in entries)


def test_hotspot_attribution_prioritizes_ground_contact(tmp_path: Path) -> None:
    recording = tmp_path / "hotspot_contact.h5"
    _write_hotspot_recording(recording, with_contact=True)

    summary = analyze_recording_summary(recording)
    hotspot = summary["comfort"]["hotspot_attribution"]

    assert hotspot["availability"] == "available"
    entries = hotspot.get("entries", [])
    assert entries
    assert any(e.get("attribution") == "ground_contact_or_penetration" for e in entries)
