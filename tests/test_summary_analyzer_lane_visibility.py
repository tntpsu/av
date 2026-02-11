from pathlib import Path
import json

import h5py
import numpy as np

from tools.debug_visualizer.backend.summary_analyzer import analyze_recording_summary


def test_lane_visibility_metrics(tmp_path: Path) -> None:
    recording_path = tmp_path / "summary_lane_visibility.h5"
    n_frames = 6
    timestamps = np.linspace(0.0, 0.5, n_frames)

    with h5py.File(recording_path, "w") as f:
        f.create_dataset("vehicle/timestamps", data=timestamps)
        f.create_dataset("vehicle/speed", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/steering", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/lateral_error", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/heading_error", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/total_error", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/pid_integral", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/emergency_stop", data=np.zeros(n_frames, dtype=np.int8))
        f.create_dataset("perception/left_lane_line_x", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("perception/right_lane_line_x", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("perception/num_lanes_detected", data=np.array([2, 2, 1, 1, 2, 2], dtype=np.int8))

        left_pts = [[[100.0, 200.0]] * 8 for _ in range(n_frames)]
        right_pts = [
            [[520.0, 200.0]] * 8,
            [[530.0, 200.0]] * 8,
            [[635.0, 200.0]] * 3,  # low visibility (few points near edge)
            [[636.0, 200.0]] * 3,  # low visibility
            [[520.0, 200.0]] * 8,
            [[520.0, 200.0]] * 8,
        ]

        f.create_dataset(
            "perception/fit_points_left",
            data=np.array([json.dumps(p).encode("utf-8") for p in left_pts], dtype="S1000"),
        )
        f.create_dataset(
            "perception/fit_points_right",
            data=np.array([json.dumps(p).encode("utf-8") for p in right_pts], dtype="S1000"),
        )

    summary = analyze_recording_summary(recording_path)
    perception = summary["perception_quality"]

    assert perception["single_lane_rate"] > 0.0
    assert perception["right_lane_low_visibility_rate"] > 0.0
