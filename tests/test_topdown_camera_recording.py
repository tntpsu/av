from pathlib import Path

import h5py
import numpy as np

from data.formats.data_format import CameraFrame, ControlCommand, RecordingFrame, VehicleState
from data.recorder import DataRecorder


def test_recorder_writes_topdown_camera_frames(tmp_path: Path) -> None:
    recorder = DataRecorder(str(tmp_path), recording_name="topdown_camera_test")
    try:
        vehicle_state = VehicleState(
            timestamp=0.0,
            position=np.zeros(3),
            rotation=np.array([0.0, 0.0, 0.0, 1.0]),
            velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
            speed=0.0,
            steering_angle=0.0,
            motor_torque=0.0,
            brake_torque=0.0,
        )
        front_frame = CameraFrame(
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=0.0,
            frame_id=0,
        )
        topdown_frame = CameraFrame(
            image=np.ones((480, 640, 3), dtype=np.uint8) * 255,
            timestamp=0.0,
            frame_id=42,
            camera_id="top_down",
        )
        frame = RecordingFrame(
            timestamp=0.0,
            frame_id=0,
            camera_frame=front_frame,
            camera_topdown_frame=topdown_frame,
            vehicle_state=vehicle_state,
            control_command=ControlCommand(timestamp=0.0, steering=0.0, throttle=0.0, brake=0.0),
            perception_output=None,
            trajectory_output=None,
            unity_feedback=None,
        )

        recorder._flush_frames([frame])
        recorder.h5_file.flush()
    finally:
        recorder.close()

    with h5py.File(tmp_path / "topdown_camera_test.h5", "r") as h5_file:
        assert "camera/topdown_images" in h5_file
        assert "camera/topdown_timestamps" in h5_file
        assert "camera/topdown_frame_ids" in h5_file
        assert h5_file["camera/topdown_images"].shape[0] == 1
        assert int(h5_file["camera/topdown_frame_ids"][0]) == 42
