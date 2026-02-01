from pathlib import Path

import h5py
import numpy as np
import pytest

from data.formats.data_format import CameraFrame, ControlCommand, RecordingFrame, VehicleState
from data.recorder import DataRecorder


def test_recorder_writes_road_frame_metrics(tmp_path: Path) -> None:
    recorder = DataRecorder(str(tmp_path), recording_name="road_frame_metrics_test")
    try:
        vehicle_state = VehicleState(
            timestamp=0.0,
            position=np.zeros(3),
            rotation=np.array([0.0, 0.0, 0.0, 1.0]),
            velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
            speed=5.0,
            steering_angle=0.0,
            motor_torque=0.0,
            brake_torque=0.0,
            road_frame_lateral_offset=0.25,
            road_heading_deg=45.0,
            car_heading_deg=50.0,
            heading_delta_deg=5.0,
            road_frame_lane_center_offset=0.1,
            road_frame_lane_center_error=0.15,
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

    with h5py.File(tmp_path / "road_frame_metrics_test.h5", "r") as h5_file:
        assert "vehicle/road_frame_lateral_offset" in h5_file
        assert "vehicle/road_heading_deg" in h5_file
        assert "vehicle/car_heading_deg" in h5_file
        assert "vehicle/heading_delta_deg" in h5_file
        assert "vehicle/road_frame_lane_center_offset" in h5_file
        assert "vehicle/road_frame_lane_center_error" in h5_file
        assert float(h5_file["vehicle/road_frame_lateral_offset"][0]) == pytest.approx(0.25, rel=1e-6)
        assert float(h5_file["vehicle/road_heading_deg"][0]) == pytest.approx(45.0, rel=1e-6)
        assert float(h5_file["vehicle/car_heading_deg"][0]) == pytest.approx(50.0, rel=1e-6)
        assert float(h5_file["vehicle/heading_delta_deg"][0]) == pytest.approx(5.0, rel=1e-6)
        assert float(h5_file["vehicle/road_frame_lane_center_offset"][0]) == pytest.approx(0.1, rel=1e-6)
        assert float(h5_file["vehicle/road_frame_lane_center_error"][0]) == pytest.approx(0.15, rel=1e-6)
