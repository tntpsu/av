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
