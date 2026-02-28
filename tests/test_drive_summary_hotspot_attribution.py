from pathlib import Path

import h5py
import numpy as np

from tools.drive_summary_core import analyze_recording_summary


def _write_hotspot_recording(
    path: Path,
    *,
    with_contact: bool = False,
    with_timestamp_gap: bool = False,
    with_command_step: bool = False,
    with_limiter_transition: bool = False,
    with_physics_spike: bool = False,
) -> None:
    n = 64
    t = np.arange(n, dtype=np.float64) * 0.033
    if with_timestamp_gap:
        t[22:] += 0.26

    speed = np.linspace(0.5, 3.0, n, dtype=np.float32)
    throttle = np.full(n, 0.14, dtype=np.float32)
    brake = np.zeros(n, dtype=np.float32)
    accel_cap = np.zeros(n, dtype=np.int8)
    jerk_cap = np.zeros(n, dtype=np.int8)

    if with_command_step:
        throttle[30:] = 1.0
        speed[30:] += 1.8

    if with_limiter_transition:
        accel_cap[28:40] = 1
        jerk_cap[28:40] = 1
        speed[28:40] += np.linspace(0.0, 1.2, 12, dtype=np.float32)

    if with_physics_spike:
        speed[34] += 5.0
        speed[35] -= 4.5
        throttle[:] = 0.0
        brake[:] = 0.0

    chassis_contact = np.zeros(n, dtype=np.int8)
    chassis_pen = np.zeros(n, dtype=np.float32)
    chassis_clear = np.full(n, 0.12, dtype=np.float32)
    if with_contact:
        chassis_contact[30:36] = 1
        chassis_pen[30:36] = 0.02
        chassis_clear[30:36] = -0.005

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
        f.create_dataset("control/throttle", data=throttle)
        f.create_dataset("control/brake", data=brake)
        f.create_dataset("control/target_speed_final", data=np.full(n, 4.0, dtype=np.float32))
        f.create_dataset("control/longitudinal_accel_capped", data=accel_cap)
        f.create_dataset("control/longitudinal_jerk_capped", data=jerk_cap)

        f.create_dataset("ground_truth/left_lane_line_x", data=np.full(n, -3.5, dtype=np.float32))
        f.create_dataset("ground_truth/right_lane_line_x", data=np.full(n, 3.5, dtype=np.float32))


def _attributions(summary: dict) -> list[str]:
    hotspot = summary["comfort"]["hotspot_attribution"]
    entries = hotspot.get("entries", [])
    return [str(e.get("attribution", "")) for e in entries]


def test_hotspot_attribution_detects_timestamp_gap_artifact(tmp_path: Path) -> None:
    recording = tmp_path / "hotspot_gap.h5"
    _write_hotspot_recording(recording, with_timestamp_gap=True)

    summary = analyze_recording_summary(recording)
    hotspot = summary["comfort"]["hotspot_attribution"]
    entries = hotspot.get("entries", [])

    assert hotspot["availability"] == "available"
    assert entries
    assert any(e.get("timestamp_irregular_nearby") for e in entries)
    assert "timestamp_gap_derivative_artifact" in _attributions(summary)


def test_hotspot_attribution_prioritizes_ground_contact(tmp_path: Path) -> None:
    recording = tmp_path / "hotspot_contact.h5"
    _write_hotspot_recording(recording, with_contact=True, with_timestamp_gap=True)

    summary = analyze_recording_summary(recording)
    hotspot = summary["comfort"]["hotspot_attribution"]
    entries = hotspot.get("entries", [])

    assert hotspot["availability"] == "available"
    assert entries
    assert "ground_contact_or_penetration" in _attributions(summary)


def test_hotspot_attribution_classifies_commanded_step(tmp_path: Path) -> None:
    recording = tmp_path / "hotspot_commanded.h5"
    _write_hotspot_recording(recording, with_command_step=True)

    summary = analyze_recording_summary(recording)
    attrs = _attributions(summary)
    assert "commanded_longitudinal_step" in attrs


def test_hotspot_attribution_classifies_limiter_transition(tmp_path: Path) -> None:
    recording = tmp_path / "hotspot_limiter.h5"
    _write_hotspot_recording(recording, with_limiter_transition=True)

    summary = analyze_recording_summary(recording)
    hotspot = summary["comfort"]["hotspot_attribution"]
    entries = hotspot.get("entries", [])
    attrs = _attributions(summary)

    assert "longitudinal_limiter_transition" in attrs
    assert any(bool(e.get("limiter_transition")) for e in entries)


def test_hotspot_attribution_classifies_physics_spike(tmp_path: Path) -> None:
    recording = tmp_path / "hotspot_physics.h5"
    _write_hotspot_recording(recording, with_physics_spike=True)

    summary = analyze_recording_summary(recording)
    attrs = _attributions(summary)
    assert "physics_or_speed_estimation_spike" in attrs


def test_hotspot_attribution_rollups_present(tmp_path: Path) -> None:
    recording = tmp_path / "hotspot_rollup.h5"
    _write_hotspot_recording(
        recording,
        with_contact=True,
        with_command_step=True,
        with_limiter_transition=True,
    )

    summary = analyze_recording_summary(recording)
    hotspot = summary["comfort"]["hotspot_attribution"]

    assert isinstance(hotspot.get("counts_by_attribution"), dict)
    assert hotspot.get("high_confidence_rate") is not None
    assert "commanded_vs_measured_mismatch_rate" in hotspot
