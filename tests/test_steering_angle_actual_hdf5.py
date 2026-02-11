from pathlib import Path

import h5py
import numpy as np

from data.formats.data_format import CameraFrame, ControlCommand, RecordingFrame, VehicleState
from data.recorder import DataRecorder


def test_recorder_writes_steering_angle_actual(tmp_path: Path) -> None:
    recorder = DataRecorder(str(tmp_path), recording_name="steering_angle_actual_test")
    try:
        vehicle_state = VehicleState(
            timestamp=0.0,
            position=np.zeros(3),
            rotation=np.array([0.0, 0.0, 0.0, 1.0]),
            velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
            speed=5.0,
            steering_angle=12.0,
            steering_angle_actual=10.5,
            motor_torque=0.0,
            brake_torque=0.0,
        )
        frame = RecordingFrame(
            timestamp=0.0,
            frame_id=0,
            camera_frame=CameraFrame(image=np.zeros((1, 1, 3), dtype=np.uint8), timestamp=0.0, frame_id=0),
            vehicle_state=vehicle_state,
            control_command=ControlCommand(timestamp=0.0, steering=0.0, throttle=0.0, brake=0.0),
            perception_output=None,
            trajectory_output=None,
            unity_feedback=None,
        )

        recorder._write_vehicle_states([frame])
        recorder.h5_file.flush()
    finally:
        recorder.close()

    with h5py.File(tmp_path / "steering_angle_actual_test.h5", "r") as h5_file:
        assert "vehicle/steering_angle_actual" in h5_file
        assert float(h5_file["vehicle/steering_angle_actual"][0]) == 10.5
