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
        "transport_contract",
        "speed_intent",
        "run_intent",
        "highway_mild_curve_contract",
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
    transport_contract = summary.get("transport_contract", {})
    assert {
        "schema_version",
        "availability",
        "packet_mode",
        "consume_policy",
        "packet_completeness_rate",
        "fallback_active_rate",
        "packet_queue_depth",
        "payload_queue_depth",
        "payload_selected_age_ms",
        "payload_selected_fresh_rate",
        "payload_warn_age_exceeded_rate",
        "payload_selection_source_mode",
        "selection_result_mode",
        "payload_selection_fallback_reason_mode",
        "payload_server_queue_depth_after_select",
        "payload_server_oldest_age_ms_after_select",
        "join_source_mode",
        "join_key_present_rate",
        "join_failure_reason_mode",
        "join_failure_side_mode",
        "selected_failure_contract_reason_mode",
        "selected_failure_source_stage_mode",
        "source_key_present_camera_rate",
        "source_key_present_vehicle_rate",
        "selected_packet_key_present_rate",
        "timeout_event_delta_rate",
        "source_bundle_close_reason_mode",
        "source_bundle_deadline_ms",
        "source_bundle_age_ms",
        "source_bundle_inflight_count",
        "source_bundle_vehicle_state_built_rate",
        "source_bundle_vehicle_state_enqueued_rate",
        "source_bundle_vehicle_state_sent_rate",
        "source_bundle_camera_requested_rate",
        "source_camera_request_attempted_rate",
        "source_camera_request_accepted_rate",
        "source_camera_request_rejected_reason_mode",
        "source_camera_request_skipped_reason_mode",
        "source_camera_request_disposition_mode",
        "source_camera_request_attempt_age_ms",
        "source_camera_request_accept_age_ms",
        "source_camera_request_queue_depth",
        "active_camera_eligible_rate",
        "debug_unbundled_capture_rate",
        "camera_capture_contract_reason_mode",
        "source_bundle_camera_sent_rate",
        "source_bundle_aborted_before_vehicle_send_rate",
        "source_bundle_abort_reason_mode",
        "source_vehicle_send_blocked_by_camera_request_rate",
        "source_bundle_superseded_before_send_rate",
        "active_camera_excluded_event_rate",
        "active_camera_excluded_reason_mode",
        "unbundled_camera_entered_active_path_rate",
        "join_wait_ms",
        "packet_superseded_camera_count",
        "packet_superseded_vehicle_count",
        "skipped_unity_frames",
        "post_jump_cooldown_active_rate",
        "teleport_guard_suppressed_rate",
        "teleport_continuity_suspect_rate",
        "teleport_guard_reason_mode",
        "effective_reference_velocity_drop_count",
        "limits",
    }.issubset(transport_contract.keys())
    speed_intent = summary.get("speed_intent", {})
    assert {
        "schema_version",
        "availability",
        "desired_target_speed_mps",
        "post_limits_target_speed_mps",
        "governor_target_speed_mps",
        "acc_target_speed_mps",
        "planner_target_speed_applied_mps",
        "final_target_speed_mps",
        "final_longitudinal_owner_mode",
        "reference_velocity_source_mode",
        "planner_reference_speed_mps",
        "effective_reference_speed_mps",
        "brake_episode_counts_by_reason",
    }.issubset(speed_intent.keys())
    run_intent = summary.get("run_intent", {})
    assert {
        "schema_version",
        "availability",
        "mode",
        "recording_type",
        "track_id",
        "run_target_speed_mps",
        "road_speed_limit_expected_mps",
        "lead_following_active",
        "acc_state_mode",
        "final_longitudinal_owner_mode",
        "lead_collision_override_rate_pct",
    }.issubset(run_intent.keys())
    highway_mild_curve_contract = summary.get("highway_mild_curve_contract", {})
    assert {
        "schema_version",
        "availability",
        "issue_detected",
        "high_error_frame_count",
        "mild_curve_present_on_high_error_rate",
        "curve_recognition_inactive_on_high_error_rate",
        "long_lookahead_on_high_error_rate",
        "reference_geometry_mismatch_on_high_error_rate",
        "underactivated_tracking_on_high_error_rate",
        "poor_perception_overlap_on_high_error_rate",
        "transport_fallback_overlap_on_high_error_rate",
        "mpc_feasible_on_high_error_rate",
        "curve_intent_state_mode_on_high_error",
        "curve_local_state_mode_on_high_error",
        "lateral_error_abs_m",
        "road_frame_lane_center_offset_abs_m",
        "reference_point_curvature_abs",
        "pp_lookahead_distance_m",
        "reference_lookahead_target_m",
        "limits",
    }.issubset(highway_mild_curve_contract.keys())
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
