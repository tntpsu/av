from pathlib import Path

import h5py
import numpy as np
import pytest

from data.formats.data_format import CameraFrame, ControlCommand, RecordingFrame
from data.recorder import DataRecorder


def test_recorder_handles_none_distance_to_next_curve_start(tmp_path: Path) -> None:
    recorder = DataRecorder(str(tmp_path), recording_name="control_command_none_distance_test")
    try:
        frame = RecordingFrame(
            timestamp=0.0,
            frame_id=0,
            camera_frame=CameraFrame(
                image=np.zeros((1, 1, 3), dtype=np.uint8),
                timestamp=0.0,
                frame_id=0,
            ),
            vehicle_state=None,
            control_command=ControlCommand(
                timestamp=0.0,
                steering=0.0,
                throttle=0.0,
                brake=0.0,
                distance_to_next_curve_start_m=None,
            ),
            perception_output=None,
            trajectory_output=None,
            unity_feedback=None,
        )

        recorder._write_control_commands([frame])
        recorder.h5_file.flush()
    finally:
        recorder.close()

    with h5py.File(tmp_path / "control_command_none_distance_test.h5", "r") as h5_file:
        value = float(h5_file["control/distance_to_next_curve_start_m"][0])
        assert np.isnan(value)


def test_recorder_writes_curve_anticipation_fields(tmp_path: Path) -> None:
    recorder = DataRecorder(str(tmp_path), recording_name="control_command_curve_anticipation_test")
    try:
        frame = RecordingFrame(
            timestamp=0.0,
            frame_id=0,
            camera_frame=CameraFrame(
                image=np.zeros((1, 1, 3), dtype=np.uint8),
                timestamp=0.0,
                frame_id=0,
            ),
            vehicle_state=None,
            control_command=ControlCommand(
                timestamp=0.0,
                steering=0.0,
                throttle=0.0,
                brake=0.0,
                curve_anticipation_score=0.62,
                curve_anticipation_score_raw=0.71,
                curve_anticipation_active=True,
                curve_anticipation_source="trajectory_shadow",
                curve_anticipation_term_curvature=0.12,
                curve_anticipation_term_heading=0.54,
                curve_anticipation_term_far_rise=0.31,
            ),
            perception_output=None,
            trajectory_output=None,
            unity_feedback=None,
        )

        recorder._write_control_commands([frame])
        recorder.h5_file.flush()
    finally:
        recorder.close()

    with h5py.File(tmp_path / "control_command_curve_anticipation_test.h5", "r") as h5_file:
        assert float(h5_file["control/curve_anticipation_score"][0]) == pytest.approx(0.62, rel=1e-6)
        assert float(h5_file["control/curve_anticipation_score_raw"][0]) == pytest.approx(0.71, rel=1e-6)
        assert int(h5_file["control/curve_anticipation_active"][0]) == 1
        source = h5_file["control/curve_anticipation_source"][0]
        if isinstance(source, bytes):
            source = source.decode("utf-8")
        assert str(source) == "trajectory_shadow"
        assert float(h5_file["control/curve_anticipation_term_curvature"][0]) == pytest.approx(0.12, rel=1e-6)
        assert float(h5_file["control/curve_anticipation_term_heading"][0]) == pytest.approx(0.54, rel=1e-6)
        assert float(h5_file["control/curve_anticipation_term_far_rise"][0]) == pytest.approx(0.31, rel=1e-6)


def test_recorder_writes_curve_phase_scheduler_fields(tmp_path: Path) -> None:
    recorder = DataRecorder(str(tmp_path), recording_name="control_command_curve_phase_scheduler_test")
    try:
        frame = RecordingFrame(
            timestamp=0.0,
            frame_id=0,
            camera_frame=CameraFrame(
                image=np.zeros((1, 1, 3), dtype=np.uint8),
                timestamp=0.0,
                frame_id=0,
            ),
            vehicle_state=None,
            control_command=ControlCommand(
                timestamp=0.0,
                steering=0.0,
                throttle=0.0,
                brake=0.0,
                curve_scheduler_mode="phase_active",
                curve_phase=0.61,
                curve_phase_raw=0.73,
                curve_phase_state="COMMIT",
                curve_phase_rearm_event=False,
                curve_phase_entry_frames=5,
                curve_phase_rearm_hold_frames=0,
                curve_phase_term_preview=0.82,
                curve_phase_term_path=0.42,
                curve_phase_term_rise=0.31,
                curve_phase_curvature_rise_abs=0.0024,
                curve_intent=0.61,
                curve_intent_raw=0.73,
                curve_intent_state="COMMIT",
                curve_intent_term_preview=0.82,
                curve_intent_term_path=0.42,
                curve_intent_term_rise=0.31,
                curve_intent_confidence=0.48,
                curve_intent_speed_guardrail_active=True,
                curve_intent_speed_guardrail_cap_mps=7.0,
                curve_intent_speed_guardrail_confidence=0.42,
                reference_lookahead_target=10.2,
                reference_lookahead_after_slew=9.8,
                reference_lookahead_active=9.8,
            ),
            perception_output=None,
            trajectory_output=None,
            unity_feedback=None,
        )

        recorder._write_control_commands([frame])
        recorder.h5_file.flush()
    finally:
        recorder.close()

    with h5py.File(tmp_path / "control_command_curve_phase_scheduler_test.h5", "r") as h5_file:
        mode = h5_file["control/curve_scheduler_mode"][0]
        state = h5_file["control/curve_phase_state"][0]
        if isinstance(mode, bytes):
            mode = mode.decode("utf-8")
        if isinstance(state, bytes):
            state = state.decode("utf-8")
        assert str(mode) == "phase_active"
        assert float(h5_file["control/curve_phase"][0]) == pytest.approx(0.61, rel=1e-6)
        assert float(h5_file["control/curve_phase_raw"][0]) == pytest.approx(0.73, rel=1e-6)
        assert str(state) == "COMMIT"
        assert int(h5_file["control/curve_phase_rearm_event"][0]) == 0
        assert int(h5_file["control/curve_phase_entry_frames"][0]) == 5
        assert int(h5_file["control/curve_phase_rearm_hold_frames"][0]) == 0
        assert float(h5_file["control/curve_phase_term_preview"][0]) == pytest.approx(0.82, rel=1e-6)
        assert float(h5_file["control/curve_phase_term_path"][0]) == pytest.approx(0.42, rel=1e-6)
        assert float(h5_file["control/curve_phase_term_rise"][0]) == pytest.approx(0.31, rel=1e-6)
        assert float(h5_file["control/curve_phase_curvature_rise_abs"][0]) == pytest.approx(
            0.0024, rel=1e-6
        )
        intent_state = h5_file["control/curve_intent_state"][0]
        if isinstance(intent_state, bytes):
            intent_state = intent_state.decode("utf-8")
        assert float(h5_file["control/curve_intent"][0]) == pytest.approx(0.61, rel=1e-6)
        assert float(h5_file["control/curve_intent_raw"][0]) == pytest.approx(0.73, rel=1e-6)
        assert str(intent_state) == "COMMIT"
        assert float(h5_file["control/curve_intent_term_preview"][0]) == pytest.approx(0.82, rel=1e-6)
        assert float(h5_file["control/curve_intent_term_path"][0]) == pytest.approx(0.42, rel=1e-6)
        assert float(h5_file["control/curve_intent_term_rise"][0]) == pytest.approx(0.31, rel=1e-6)
        assert float(h5_file["control/curve_intent_confidence"][0]) == pytest.approx(0.48, rel=1e-6)
        assert int(h5_file["control/curve_intent_speed_guardrail_active"][0]) == 1
        assert float(h5_file["control/curve_intent_speed_guardrail_cap_mps"][0]) == pytest.approx(
            7.0, rel=1e-6
        )
        assert float(h5_file["control/curve_intent_speed_guardrail_confidence"][0]) == pytest.approx(
            0.42, rel=1e-6
        )
        assert float(h5_file["control/reference_lookahead_target"][0]) == pytest.approx(10.2, rel=1e-6)
        assert float(h5_file["control/reference_lookahead_after_slew"][0]) == pytest.approx(9.8, rel=1e-6)
        assert float(h5_file["control/reference_lookahead_active"][0]) == pytest.approx(9.8, rel=1e-6)
