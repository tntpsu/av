from pathlib import Path

import h5py
import numpy as np

from tools.debug_visualizer.backend.summary_analyzer import analyze_recording_summary


def test_out_of_lane_requires_sustained_frames(tmp_path: Path) -> None:
    recording_path = tmp_path / "summary_out_of_lane.h5"
    n_frames = 20
    timestamps = np.linspace(0.0, 1.9, n_frames)

    gt_left = np.full(n_frames, -3.5, dtype=np.float32)
    gt_right = np.full(n_frames, 3.5, dtype=np.float32)

    # Single-frame out-of-lane spike (should be ignored)
    gt_left[3] = 0.2

    # Sustained out-of-lane event (10 frames)
    gt_right[10:20] = -0.2

    with h5py.File(recording_path, "w") as f:
        f.create_dataset("vehicle/timestamps", data=timestamps)
        f.create_dataset("vehicle/speed", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/steering", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/lateral_error", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/heading_error", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/total_error", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/pid_integral", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/emergency_stop", data=np.zeros(n_frames, dtype=np.int8))
        f.create_dataset("ground_truth/left_lane_line_x", data=gt_left)
        f.create_dataset("ground_truth/right_lane_line_x", data=gt_right)

    summary = analyze_recording_summary(recording_path)
    safety = summary["safety"]

    assert safety["out_of_lane_events"] == 1
    assert safety["out_of_lane_events_list"][0]["start_frame"] == 10
    assert safety["out_of_lane_events_list"][0]["duration_frames"] == 10
    assert safety["out_of_lane_time"] == 50.0


def test_failure_banner_uses_boundary_mask(tmp_path: Path) -> None:
    recording_path = tmp_path / "summary_failure_boundary.h5"
    n_frames = 25
    timestamps = np.linspace(0.0, 2.4, n_frames)

    gt_left = np.full(n_frames, -3.5, dtype=np.float32)
    gt_right = np.full(n_frames, 3.5, dtype=np.float32)
    gt_center = np.full(n_frames, 0.0, dtype=np.float32)

    # Create a sustained out-of-lane event using boundaries (frames 10-19).
    gt_left[10:20] = 0.2
    # Make gt_center look benign so we prove we use boundaries.
    gt_center[10:20] = 0.0

    with h5py.File(recording_path, "w") as f:
        f.create_dataset("vehicle/timestamps", data=timestamps)
        f.create_dataset("vehicle/speed", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/steering", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/lateral_error", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/heading_error", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/total_error", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/pid_integral", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/emergency_stop", data=np.zeros(n_frames, dtype=np.int8))
        f.create_dataset("ground_truth/left_lane_line_x", data=gt_left)
        f.create_dataset("ground_truth/right_lane_line_x", data=gt_right)
        f.create_dataset("ground_truth/lane_center_x", data=gt_center)

    summary = analyze_recording_summary(recording_path)
    safety = summary["safety"]
    assert safety["out_of_lane_events"] == 1


def test_straight_sign_mismatch_events(tmp_path: Path) -> None:
    recording_path = tmp_path / "summary_sign_mismatch.h5"
    n_frames = 10
    timestamps = np.linspace(0.0, 0.9, n_frames)

    steering = np.zeros(n_frames, dtype=np.float32)
    steering_before_limits = np.zeros(n_frames, dtype=np.float32)
    feedback_steering = np.zeros(n_frames, dtype=np.float32)
    total_error_scaled = np.zeros(n_frames, dtype=np.float32)
    is_straight = np.ones(n_frames, dtype=np.int8)
    num_lanes = np.full(n_frames, 2, dtype=np.int8)

    # Create a sustained sign mismatch (4 frames)
    total_error_scaled[3:7] = 0.1
    steering[3:7] = -0.1
    steering_before_limits[3:7] = -0.1
    feedback_steering[3:7] = 0.2

    with h5py.File(recording_path, "w") as f:
        f.create_dataset("vehicle/timestamps", data=timestamps)
        f.create_dataset("vehicle/speed", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/steering", data=steering)
        f.create_dataset("control/steering_before_limits", data=steering_before_limits)
        f.create_dataset("control/feedback_steering", data=feedback_steering)
        f.create_dataset("control/lateral_error", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/heading_error", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/total_error", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/total_error_scaled", data=total_error_scaled)
        f.create_dataset("control/pid_integral", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/emergency_stop", data=np.zeros(n_frames, dtype=np.int8))
        f.create_dataset("control/is_straight", data=is_straight)
        f.create_dataset("perception/num_lanes_detected", data=num_lanes)

    summary = analyze_recording_summary(recording_path)
    control_stability = summary["control_stability"]
    assert control_stability["straight_sign_mismatch_events"] == 1
    assert control_stability["straight_sign_mismatch_rate"] == 100.0
    events = control_stability["straight_sign_mismatch_events_list"]
    assert events[0]["root_cause"] == "rate_or_jerk_limit"


def test_alignment_summary_handles_empty_arrays(tmp_path: Path) -> None:
    recording_path = tmp_path / "summary_alignment_empty.h5"

    with h5py.File(recording_path, "w") as f:
        f.create_dataset("vehicle/timestamps", data=np.array([0.0]))
        f.create_dataset("vehicle/speed", data=np.array([0.0], dtype=np.float32))
        f.create_dataset("control/steering", data=np.array([0.0], dtype=np.float32))
        f.create_dataset("control/lateral_error", data=np.array([0.0], dtype=np.float32))
        f.create_dataset("control/heading_error", data=np.array([0.0], dtype=np.float32))
        f.create_dataset("control/total_error", data=np.array([0.0], dtype=np.float32))
        f.create_dataset("control/pid_integral", data=np.array([0.0], dtype=np.float32))
        f.create_dataset("control/emergency_stop", data=np.array([0], dtype=np.int8))
        f.create_dataset("perception/left_lane_line_x", data=np.array([], dtype=np.float32))
        f.create_dataset("perception/right_lane_line_x", data=np.array([], dtype=np.float32))
        f.create_dataset("ground_truth/lane_center_x", data=np.array([], dtype=np.float32))

    summary = analyze_recording_summary(recording_path)
    alignment_summary = summary.get("alignment_summary") or {}
    assert alignment_summary.get("perception_vs_gt_p95_abs", 0.0) == 0.0
