from pathlib import Path

import h5py
import numpy as np
import pytest

from data.formats.data_format import (
    CameraFrame,
    ControlCommand,
    RecordingFrame,
    UnityFeedback,
    VehicleState,
)
from data.recorder import DataRecorder


def test_recorder_writes_vehicle_chassis_ground_metrics(tmp_path: Path) -> None:
    recorder = DataRecorder(str(tmp_path), recording_name="chassis_ground_vehicle_test")
    try:
        vehicle_state = VehicleState(
            timestamp=0.0,
            position=np.zeros(3),
            rotation=np.array([0.0, 0.0, 0.0, 1.0]),
            velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
            speed=2.0,
            steering_angle=0.0,
            motor_torque=0.0,
            brake_torque=0.0,
            chassis_ground_min_clearance_m=0.08,
            chassis_ground_effective_min_clearance_m=0.08,
            chassis_ground_clearance_m=0.11,
            chassis_ground_penetration_m=0.0,
            chassis_ground_contact=False,
            wheel_grounded_count=4,
            wheel_colliders_ready=True,
            force_fallback_active=False,
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

    with h5py.File(tmp_path / "chassis_ground_vehicle_test.h5", "r") as h5_file:
        assert float(h5_file["vehicle/chassis_ground_min_clearance_m"][0]) == pytest.approx(0.08, rel=1e-6)
        assert float(h5_file["vehicle/chassis_ground_effective_min_clearance_m"][0]) == pytest.approx(0.08, rel=1e-6)
        assert float(h5_file["vehicle/chassis_ground_clearance_m"][0]) == pytest.approx(0.11, rel=1e-6)
        assert float(h5_file["vehicle/chassis_ground_penetration_m"][0]) == pytest.approx(0.0, rel=1e-6)
        assert int(h5_file["vehicle/chassis_ground_contact"][0]) == 0
        assert int(h5_file["vehicle/wheel_grounded_count"][0]) == 4
        assert int(h5_file["vehicle/wheel_colliders_ready"][0]) == 1
        assert int(h5_file["vehicle/force_fallback_active"][0]) == 0


def test_recorder_writes_unity_feedback_chassis_ground_metrics(tmp_path: Path) -> None:
    recorder = DataRecorder(str(tmp_path), recording_name="chassis_ground_feedback_test")
    try:
        vehicle_state = VehicleState(
            timestamp=0.0,
            position=np.zeros(3),
            rotation=np.array([0.0, 0.0, 0.0, 1.0]),
            velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
            speed=2.0,
            steering_angle=0.0,
            motor_torque=0.0,
            brake_torque=0.0,
        )
        feedback = UnityFeedback(
            timestamp=0.0,
            chassis_ground_min_clearance_m=0.08,
            chassis_ground_effective_min_clearance_m=0.08,
            chassis_ground_clearance_m=-0.01,
            chassis_ground_penetration_m=0.02,
            chassis_ground_contact=True,
            wheel_grounded_count=3,
            wheel_colliders_ready=True,
            force_fallback_active=True,
        )
        frame = RecordingFrame(
            timestamp=0.0,
            frame_id=0,
            camera_frame=CameraFrame(image=np.zeros((1, 1, 3), dtype=np.uint8), timestamp=0.0, frame_id=0),
            vehicle_state=vehicle_state,
            control_command=ControlCommand(timestamp=0.0, steering=0.0, throttle=0.0, brake=0.0),
            perception_output=None,
            trajectory_output=None,
            unity_feedback=feedback,
        )
        recorder._write_unity_feedback([frame])
        recorder.h5_file.flush()
    finally:
        recorder.close()

    with h5py.File(tmp_path / "chassis_ground_feedback_test.h5", "r") as h5_file:
        assert float(h5_file["unity_feedback/chassis_ground_min_clearance_m"][0]) == pytest.approx(0.08, rel=1e-6)
        assert float(h5_file["unity_feedback/chassis_ground_effective_min_clearance_m"][0]) == pytest.approx(0.08, rel=1e-6)
        assert float(h5_file["unity_feedback/chassis_ground_clearance_m"][0]) == pytest.approx(-0.01, rel=1e-6)
        assert float(h5_file["unity_feedback/chassis_ground_penetration_m"][0]) == pytest.approx(0.02, rel=1e-6)
        assert int(h5_file["unity_feedback/chassis_ground_contact"][0]) == 1
        assert int(h5_file["unity_feedback/wheel_grounded_count"][0]) == 3
        assert int(h5_file["unity_feedback/wheel_colliders_ready"][0]) == 1
        assert int(h5_file["unity_feedback/force_fallback_active"][0]) == 1
