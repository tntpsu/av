from pathlib import Path

import h5py
import numpy as np

from tools.drive_summary_core import analyze_recording_summary


def _write_minimal_recording(path: Path) -> None:
    n = 12
    t = np.linspace(0.0, 1.1, n)
    with h5py.File(path, "w") as f:
        f.create_dataset("vehicle/timestamps", data=t)
        f.create_dataset("vehicle/speed", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/steering", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/lateral_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/heading_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/total_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/pid_integral", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/emergency_stop", data=np.zeros(n, dtype=np.int8))
        f.create_dataset("ground_truth/left_lane_line_x", data=np.full(n, -3.5, dtype=np.float32))
        f.create_dataset("ground_truth/right_lane_line_x", data=np.full(n, 3.5, dtype=np.float32))


def test_drive_summary_contract_keys(tmp_path: Path) -> None:
    recording = tmp_path / "contract_recording.h5"
    _write_minimal_recording(recording)

    summary = analyze_recording_summary(recording)

    expected_top_level_keys = {
        "summary_schema_version",
        "executive_summary",
        "path_tracking",
        "layer_scores",
        "layer_score_breakdown",
        "control_mode",
        "control_smoothness",
        "speed_control",
        "comfort",
        "control_stability",
        "perception_quality",
        "trajectory_quality",
        "turn_bias",
        "alignment_summary",
        "latency_sync",
        "chassis_ground",
        "system_health",
        "safety",
        "recommendations",
        "config",
        "time_series",
    }

    assert expected_top_level_keys.issubset(summary.keys())
    assert summary["summary_schema_version"] == "v1"
    latency_sync = summary.get("latency_sync", {})
    assert {"schema_version", "e2e", "sync_alignment", "overall"}.issubset(latency_sync.keys())
    chassis_ground = summary.get("chassis_ground", {})
    assert {"schema_version", "availability", "health", "limits"}.issubset(chassis_ground.keys())
    comfort = summary.get("comfort", {})
    assert {"metric_roles", "hotspot_attribution"}.issubset(comfort.keys())
    speed_control = summary.get("speed_control", {})
    assert {
        "curve_cap_active_rate",
        "pre_turn_arm_lead_frames_p50",
        "pre_turn_arm_lead_frames_p95",
        "overspeed_into_curve_rate",
        "turn_infeasible_rate_when_curve_cap_active",
    }.issubset(speed_control.keys())
