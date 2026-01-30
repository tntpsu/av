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
