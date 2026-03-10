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
        "curve_intent_diagnostics",
        "curve_local_contract",
        "local_curve_reference",
        "turn_in_owner",
        "curve_turn_events",
        "curve_straight_segments",
        "speed_control",
        "comfort",
        "control_stability",
        "perception_quality",
        "trajectory_quality",
        "turn_bias",
        "alignment_summary",
        "latency_sync",
        "chassis_ground",
        "curvature_contract_health",
        "first_fault_chain",
        "system_health",
        "safety",
        "recommendations",
        "config",
        "time_series",
    }

    assert expected_top_level_keys.issubset(summary.keys())
    assert summary["summary_schema_version"] == "v1"
    latency_sync = summary.get("latency_sync", {})
    assert {"schema_version", "e2e", "sync_alignment", "cadence", "overall"}.issubset(latency_sync.keys())
    cadence = latency_sync.get("cadence", {})
    assert {"availability", "status", "stats_ms", "limits", "pass", "tuning_valid"}.issubset(
        cadence.keys()
    )
    chassis_ground = summary.get("chassis_ground", {})
    assert {"schema_version", "availability", "health", "limits"}.issubset(chassis_ground.keys())
    comfort = summary.get("comfort", {})
    assert {"metric_roles", "hotspot_attribution"}.issubset(comfort.keys())
    hotspot = comfort.get("hotspot_attribution", {})
    assert {
        "counts_by_attribution",
        "high_confidence_rate",
        "commanded_vs_measured_mismatch_rate",
    }.issubset(hotspot.keys())
    speed_control = summary.get("speed_control", {})
    assert {
        "curve_cap_active_rate",
        "pre_turn_arm_lead_frames_p50",
        "pre_turn_arm_lead_frames_p95",
        "overspeed_into_curve_rate",
        "turn_infeasible_rate_when_curve_cap_active",
        "cap_tracking_error_p95_mps",
        "frames_above_cap_1p0mps_rate",
        "cap_recovery_frames_p95",
        "hard_ceiling_applied_rate",
    }.issubset(speed_control.keys())
    contract_health = summary.get("curvature_contract_health", {})
    assert {
        "availability",
        "curvature_source_divergence_p95",
        "curvature_source_diverged_rate",
        "curvature_map_authority_lost_rate",
        "curve_intent_commit_streak_max_frames",
        "feasibility_violation_rate",
        "feasibility_backstop_active_rate",
        "map_health_untrusted_rate",
        "track_mismatch_rate",
        "curvature_contract_consistency_rate",
        "telemetry_completeness_rate_curvature_contract",
        "telemetry_completeness_rate_feasibility",
        "limits",
    }.issubset(contract_health.keys())
    first_fault_chain = summary.get("first_fault_chain", {})
    assert {
        "first_divergence_frame",
        "first_infeasible_frame",
        "first_speed_above_feasibility_frame",
        "first_boundary_breach_frame",
    }.issubset(first_fault_chain.keys())
    curve_local_contract = summary.get("curve_local_contract", {})
    assert {
        "availability",
        "curve_local_contract_available",
        "curve_preview_far_active_straight_rate",
        "curve_local_active_straight_rate",
        "curve_local_arm_ready_straight_rate",
        "curve_local_commit_ready_straight_rate",
        "curve_local_path_sustain_active_straight_rate",
        "straight_summary_source",
        "straight_summary_vs_segment_rate_delta_pct",
        "curve_local_commit_streak_max_frames",
        "curve_local_arm_without_ready_count",
        "curve_local_commit_without_ready_count",
        "curve_local_commit_without_distance_ready_count",
        "curve_local_reentry_without_gate_count",
        "curve_local_watchdog_pingpong_count",
        "curve_local_latched_straight_count",
        "pp_curve_local_floor_breach_count",
        "curve_lookahead_collapse_violation_count",
        "limits",
    }.issubset(curve_local_contract.keys())
    turn_in_owner = summary.get("turn_in_owner", {})
    assert {
        "availability",
        "owner_mode",
        "entry_weight_source",
        "fallback_active_rate",
        "owner_commit_band_clamp_active_rate",
        "owner_commit_progress_p50",
        "curve_local_commit_without_ready_count",
        "curve_local_arm_without_ready_count",
        "steering_onset_minus_curve_start_frames_p50",
    }.issubset(turn_in_owner.keys())
    assert {
        "reference_lookahead_entry_shorten_slew_m_per_frame",
        "pp_floor_rescue_delta_max_m",
    }.issubset(curve_local_contract.get("limits", {}).keys())
    local_curve_reference = summary.get("local_curve_reference", {})
    assert {
        "availability",
        "mode",
        "active_rate",
        "shadow_only_rate",
        "valid_rate",
        "fallback_active_rate",
        "source_mode",
        "fallback_reason_mode",
        "blend_weight_p50",
        "progress_weight_p50",
        "planner_delta_p50_m",
        "planner_delta_p95_m",
        "target_distance_p50_m",
        "arc_curvature_p50",
        "turn_event_count",
        "limits",
    }.issubset(local_curve_reference.keys())
    assert isinstance(summary.get("curve_turn_events", []), list)
    assert isinstance(summary.get("curve_straight_segments", []), list)
