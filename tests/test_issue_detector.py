from pathlib import Path
import json

import h5py
import numpy as np

from tools.debug_visualizer.backend.issue_detector import detect_issues


def test_detects_straight_sign_mismatch_issue(tmp_path: Path) -> None:
    recording_path = tmp_path / "issue_sign_mismatch.h5"
    n_frames = 12
    timestamps = np.linspace(0.0, 1.1, n_frames)

    steering = np.zeros(n_frames, dtype=np.float32)
    total_error_scaled = np.zeros(n_frames, dtype=np.float32)
    is_straight = np.ones(n_frames, dtype=np.int8)
    steering_before_limits = np.zeros(n_frames, dtype=np.float32)
    feedback_steering = np.zeros(n_frames, dtype=np.float32)
    num_lanes = np.full(n_frames, 2, dtype=np.int8)

    # Create a sustained mismatch (5 frames)
    total_error_scaled[4:9] = 0.1
    steering[4:9] = -0.1
    steering_before_limits[4:9] = -0.1
    feedback_steering[4:9] = 0.2

    with h5py.File(recording_path, "w") as f:
        f.create_dataset("vehicle/timestamps", data=timestamps)
        f.create_dataset("control/steering", data=steering)
        f.create_dataset("control/lateral_error", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/total_error_scaled", data=total_error_scaled)
        f.create_dataset("control/is_straight", data=is_straight)
        f.create_dataset("control/steering_before_limits", data=steering_before_limits)
        f.create_dataset("control/feedback_steering", data=feedback_steering)
        f.create_dataset("perception/num_lanes_detected", data=num_lanes)
        f.create_dataset("perception/left_lane_line_x", data=np.zeros(n_frames, dtype=np.float32))

    issues_data = detect_issues(recording_path)
    issues = issues_data.get("issues", [])
    mismatch_issues = [i for i in issues if i.get("type") == "straight_sign_mismatch"]

    assert len(mismatch_issues) == 1
    assert mismatch_issues[0]["root_cause"] == "rate_or_jerk_limit"


def test_detects_right_lane_low_visibility(tmp_path: Path) -> None:
    recording_path = tmp_path / "issue_right_lane_visibility.h5"
    n_frames = 8
    timestamps = np.linspace(0.0, 0.7, n_frames)

    right_pts = [
        [[520.0, 200.0]] * 8,
        [[530.0, 200.0]] * 8,
        [[635.0, 200.0]] * 3,  # low visibility (few points near edge)
        [[636.0, 200.0]] * 3,
        [[637.0, 200.0]] * 3,
        [[520.0, 200.0]] * 8,
        [[520.0, 200.0]] * 8,
        [[520.0, 200.0]] * 8,
    ]

    with h5py.File(recording_path, "w") as f:
        f.create_dataset("vehicle/timestamps", data=timestamps)
        f.create_dataset("control/steering", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/lateral_error", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("perception/num_lanes_detected", data=np.full(n_frames, 2, dtype=np.int8))
        f.create_dataset("perception/left_lane_line_x", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("perception/right_lane_line_x", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset(
            "perception/fit_points_right",
            data=np.array([json.dumps(p).encode("utf-8") for p in right_pts], dtype="S1000"),
        )

    issues_data = detect_issues(recording_path)
    issues = issues_data.get("issues", [])
    visibility_issues = [i for i in issues if i.get("type") == "right_lane_low_visibility"]

    assert len(visibility_issues) == 1


def test_detects_highway_mild_curve_underactivation(tmp_path: Path) -> None:
    recording_path = tmp_path / "issue_mild_curve_underactivation.h5"
    n_frames = 20
    timestamps = np.linspace(0.0, 1.9, n_frames)
    lateral_error = np.zeros(n_frames, dtype=np.float32)
    lateral_error[6:14] = 0.82
    ref_curvature = np.zeros(n_frames, dtype=np.float32)
    ref_curvature[6:14] = 0.002

    with h5py.File(recording_path, "w") as f:
        f.create_dataset("vehicle/timestamps", data=timestamps)
        f.create_dataset("control/steering", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/lateral_error", data=lateral_error)
        f.create_dataset("control/pp_lookahead_distance", data=np.full(n_frames, 9.5, dtype=np.float32))
        f.create_dataset("control/reference_lookahead_target", data=np.full(n_frames, 14.5, dtype=np.float32))
        f.create_dataset(
            "control/curve_intent_state",
            data=np.array([b"STRAIGHT"] * n_frames, dtype="S16"),
        )
        f.create_dataset(
            "control/curve_local_state",
            data=np.array([b"STRAIGHT"] * n_frames, dtype="S16"),
        )
        f.create_dataset("control/sync_packet_fallback_active", data=np.zeros(n_frames, dtype=np.int8))
        f.create_dataset("trajectory/reference_point_curvature", data=ref_curvature)
        f.create_dataset("vehicle/road_frame_lane_center_offset", data=np.full(n_frames, 0.03, dtype=np.float32))
        f.create_dataset("perception/confidence", data=np.ones(n_frames, dtype=np.float32))
        f.create_dataset("perception/num_lanes_detected", data=np.full(n_frames, 2, dtype=np.int8))
        f.create_dataset("perception/left_lane_line_x", data=np.zeros(n_frames, dtype=np.float32))

    issues_data = detect_issues(recording_path)
    issues = issues_data.get("issues", [])
    mild_curve_issues = [
        i for i in issues if i.get("type") == "highway_mild_curve_underactivation"
    ]

    assert len(mild_curve_issues) == 1
    assert mild_curve_issues[0]["deep_link_target"] == "summary-section-highway-mild-curve"


def test_detects_curve_sustain_collapse_rearm_cycle(tmp_path: Path) -> None:
    recording_path = tmp_path / "issue_rearm_cycle.h5"
    n_frames = 20
    timestamps = np.linspace(0.0, 1.9, n_frames)
    lateral_error = np.zeros(n_frames, dtype=np.float32)
    lateral_error[5:15] = 0.82
    ref_curvature = np.zeros(n_frames, dtype=np.float32)
    ref_curvature[5:15] = 0.002
    states = np.array([b"STRAIGHT"] * n_frames, dtype="S16")
    blockers = np.array([b"none"] * n_frames, dtype="S32")
    for idx in range(5, 15):
        states[idx] = b"ENTRY" if idx % 2 == 0 else b"REARM"
        blockers[idx] = b"state_hold" if idx % 2 == 1 else b"none"

    with h5py.File(recording_path, "w") as f:
        f.create_dataset("vehicle/timestamps", data=timestamps)
        f.create_dataset("control/steering", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/lateral_error", data=lateral_error)
        f.create_dataset("control/pp_lookahead_distance", data=np.full(n_frames, 9.5, dtype=np.float32))
        f.create_dataset("control/reference_lookahead_target", data=np.full(n_frames, 14.0, dtype=np.float32))
        f.create_dataset("control/curve_local_state", data=states)
        f.create_dataset("control/curve_intent_state", data=np.array([b"ENTRY"] * n_frames, dtype="S16"))
        f.create_dataset("control/curve_activation_blocker_mode", data=blockers)
        f.create_dataset("control/curve_local_arm_phase_deficit", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/curve_local_sustain_phase_raw", data=np.full(n_frames, 0.05, dtype=np.float32))
        f.create_dataset("control/sync_packet_fallback_active", data=np.zeros(n_frames, dtype=np.int8))
        f.create_dataset("trajectory/reference_point_curvature", data=ref_curvature)
        f.create_dataset("vehicle/road_frame_lane_center_offset", data=np.full(n_frames, 0.03, dtype=np.float32))
        f.create_dataset("perception/confidence", data=np.ones(n_frames, dtype=np.float32))
        f.create_dataset("perception/num_lanes_detected", data=np.full(n_frames, 2, dtype=np.int8))
        f.create_dataset("perception/left_lane_line_x", data=np.zeros(n_frames, dtype=np.float32))

    issues = detect_issues(recording_path).get("issues", [])
    rearm_issues = [i for i in issues if i.get("type") == "curve_sustain_collapse_rearm_cycle"]
    assert len(rearm_issues) == 1
    assert rearm_issues[0]["deep_link_target"] == "summary-section-highway-mild-curve"


def test_detects_mpc_curvature_bias_cancellation(tmp_path: Path) -> None:
    recording_path = tmp_path / "issue_mpc_bias_cancel.h5"
    n_frames = 20
    timestamps = np.linspace(0.0, 1.9, n_frames)
    lateral_error = np.zeros(n_frames, dtype=np.float32)
    lateral_error[5:15] = 0.80
    ref_curvature = np.zeros(n_frames, dtype=np.float32)
    ref_curvature[5:15] = 0.002
    mpc_kappa = np.zeros(n_frames, dtype=np.float32)
    mpc_kappa[5:15] = 0.0004
    bias = np.zeros(n_frames, dtype=np.float32)
    bias[5:15] = -0.0016

    with h5py.File(recording_path, "w") as f:
        f.create_dataset("vehicle/timestamps", data=timestamps)
        f.create_dataset("control/steering", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/lateral_error", data=lateral_error)
        f.create_dataset("control/pp_lookahead_distance", data=np.full(n_frames, 7.0, dtype=np.float32))
        f.create_dataset("control/reference_lookahead_target", data=np.full(n_frames, 10.0, dtype=np.float32))
        f.create_dataset("control/curve_local_state", data=np.array([b"ENTRY"] * n_frames, dtype="S16"))
        f.create_dataset("control/curve_intent_state", data=np.array([b"ENTRY"] * n_frames, dtype="S16"))
        f.create_dataset("control/sync_packet_fallback_active", data=np.zeros(n_frames, dtype=np.int8))
        f.create_dataset("control/mpc_feasible", data=np.ones(n_frames, dtype=np.int8))
        f.create_dataset("control/mpc_fallback_active", data=np.zeros(n_frames, dtype=np.int8))
        f.create_dataset("control/mpc_kappa_ref", data=mpc_kappa)
        f.create_dataset("control/mpc_kappa_bias_correction", data=bias)
        f.create_dataset("trajectory/reference_point_curvature", data=ref_curvature)
        f.create_dataset("vehicle/road_frame_lane_center_offset", data=np.full(n_frames, 0.04, dtype=np.float32))
        f.create_dataset("perception/confidence", data=np.ones(n_frames, dtype=np.float32))
        f.create_dataset("perception/num_lanes_detected", data=np.full(n_frames, 2, dtype=np.int8))
        f.create_dataset("perception/left_lane_line_x", data=np.zeros(n_frames, dtype=np.float32))

    issues = detect_issues(recording_path).get("issues", [])
    bias_issues = [i for i in issues if i.get("type") == "mpc_curvature_bias_cancellation"]
    assert len(bias_issues) == 1
    assert bias_issues[0]["deep_link_target"] == "summary-section-highway-mild-curve"


def test_detects_mpc_gt_cross_track_semantic_mismatch(tmp_path: Path) -> None:
    recording_path = tmp_path / "issue_mpc_gt_semantic_mismatch.h5"
    n_frames = 20
    timestamps = np.linspace(0.0, 1.9, n_frames)
    lateral_error = np.zeros(n_frames, dtype=np.float32)
    lateral_error[5:15] = 0.82
    gt_used = np.zeros(n_frames, dtype=np.float32)
    gt_used[5:15] = 0.86
    gt_at_car = np.zeros(n_frames, dtype=np.float32)
    gt_lookahead = np.zeros(n_frames, dtype=np.float32)
    gt_lookahead[5:15] = 0.86

    with h5py.File(recording_path, "w") as f:
        f.create_dataset("vehicle/timestamps", data=timestamps)
        f.create_dataset("control/steering", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/lateral_error", data=lateral_error)
        f.create_dataset("control/pp_lookahead_distance", data=np.full(n_frames, 9.0, dtype=np.float32))
        f.create_dataset("control/reference_lookahead_target", data=np.full(n_frames, 14.0, dtype=np.float32))
        f.create_dataset("control/mpc_gt_cross_track_m", data=gt_used)
        f.create_dataset("control/mpc_gt_cross_track_at_car_m", data=gt_at_car)
        f.create_dataset("control/mpc_gt_cross_track_lookahead_m", data=gt_lookahead)
        f.create_dataset("trajectory/reference_point_curvature", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/curve_intent_state", data=np.array([b"STRAIGHT"] * n_frames, dtype="S16"))
        f.create_dataset("control/curve_local_state", data=np.array([b"STRAIGHT"] * n_frames, dtype="S16"))
        f.create_dataset("control/sync_packet_fallback_active", data=np.zeros(n_frames, dtype=np.int8))
        f.create_dataset("perception/confidence", data=np.ones(n_frames, dtype=np.float32))
        f.create_dataset("perception/num_lanes_detected", data=np.full(n_frames, 2, dtype=np.int8))
        f.create_dataset("perception/left_lane_line_x", data=np.zeros(n_frames, dtype=np.float32))

    issues = detect_issues(recording_path).get("issues", [])
    semantic_issues = [
        i for i in issues if i.get("type") == "mpc_gt_cross_track_semantic_mismatch"
    ]
    assert len(semantic_issues) == 1
    assert semantic_issues[0]["deep_link_target"] == "summary-section-mpc-gt-cross-track"


def test_detects_mpc_gt_cross_track_absolute_coordinate_mismatch(tmp_path: Path) -> None:
    recording_path = tmp_path / "issue_mpc_gt_absolute_mismatch.h5"
    n_frames = 20
    timestamps = np.linspace(0.0, 1.9, n_frames)
    lateral_error = np.zeros(n_frames, dtype=np.float32)
    lateral_error[5:15] = 0.82
    gt_at_car = np.zeros(n_frames, dtype=np.float32)
    gt_at_car[5:15] = 12.0
    gt_lookahead = np.zeros(n_frames, dtype=np.float32)
    gt_lookahead[5:15] = 0.92
    road_offset = np.zeros(n_frames, dtype=np.float32)
    road_offset[5:15] = 0.03

    with h5py.File(recording_path, "w") as f:
        f.create_dataset("vehicle/timestamps", data=timestamps)
        f.create_dataset("vehicle/road_frame_lane_center_offset", data=road_offset)
        f.create_dataset("control/steering", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/lateral_error", data=lateral_error)
        f.create_dataset("control/mpc_gt_cross_track_m", data=gt_at_car)
        f.create_dataset("control/mpc_gt_cross_track_at_car_m", data=gt_at_car)
        f.create_dataset("control/mpc_gt_cross_track_lookahead_m", data=gt_lookahead)
        f.create_dataset("trajectory/reference_point_curvature", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/curve_intent_state", data=np.array([b"STRAIGHT"] * n_frames, dtype="S16"))
        f.create_dataset("control/curve_local_state", data=np.array([b"STRAIGHT"] * n_frames, dtype="S16"))
        f.create_dataset("control/sync_packet_fallback_active", data=np.zeros(n_frames, dtype=np.int8))
        f.create_dataset("perception/confidence", data=np.ones(n_frames, dtype=np.float32))
        f.create_dataset("perception/num_lanes_detected", data=np.full(n_frames, 2, dtype=np.int8))
        f.create_dataset("perception/left_lane_line_x", data=np.zeros(n_frames, dtype=np.float32))

    issues = detect_issues(recording_path).get("issues", [])
    absolute_issues = [
        i for i in issues if i.get("type") == "mpc_gt_cross_track_absolute_coordinate_mismatch"
    ]
    assert len(absolute_issues) == 1
    assert absolute_issues[0]["deep_link_target"] == "summary-section-mpc-gt-cross-track"
