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
                curve_phase_term_time=0.77,
                curve_phase_curvature_rise_abs=0.0024,
                curve_preview_far_upcoming=True,
                curve_preview_far_phase=0.86,
                curve_local_phase=0.44,
                curve_local_phase_raw=0.52,
                curve_local_state="ENTRY",
                curve_local_phase_source="distance_time_path_gate",
                curve_local_entry_driver="distance",
                curve_local_entry_severity=0.82,
                curve_local_entry_on_effective=0.40,
                curve_local_phase_distance_start_effective_m=10.0,
                curve_local_phase_time_start_effective_s=1.6,
                curve_local_arm_ready=True,
                curve_local_time_ready=False,
                curve_local_in_curve_now=False,
                curve_local_commit_ready=True,
                curve_local_commit_driver="distance",
                curve_local_arm_phase_raw=0.42,
                curve_local_sustain_phase_raw=0.86,
                curve_local_path_sustain_active=True,
                curve_local_distance_ready=True,
                curve_local_distance_horizon_m=7.5,
                curve_local_time_horizon_s=1.6,
                curve_local_reentry_ready=True,
                curve_local_rearm_cooldown_active=False,
                curve_local_force_straight_active=False,
                curve_local_commit_streak_frames=3,
                curve_intent=0.61,
                curve_intent_raw=0.73,
                curve_intent_state="COMMIT",
                curve_intent_term_preview=0.82,
                curve_intent_term_path=0.42,
                curve_intent_term_rise=0.31,
                curve_intent_confidence=0.48,
                curve_intent_watchdog_triggered=True,
                curve_intent_speed_guardrail_active=True,
                curve_intent_speed_guardrail_cap_mps=7.0,
                curve_intent_speed_guardrail_confidence=0.42,
                reference_lookahead_target=10.2,
                reference_lookahead_target_pre_entry_guard=9.8,
                reference_lookahead_after_slew=9.8,
                reference_lookahead_active=9.8,
                reference_lookahead_owner_nominal_target=9.4,
                reference_lookahead_owner_commit_band_target=5.6,
                reference_lookahead_owner_entry_progress=0.62,
                reference_lookahead_owner_commit_distance_progress=0.55,
                reference_lookahead_owner_commit_phase_progress=0.28,
                reference_lookahead_owner_commit_progress=0.55,
                reference_lookahead_owner_commit_distance_start_effective_m=7.5,
                reference_lookahead_owner_commit_band_clamp_active=True,
                reference_lookahead_owner_commit_band_clamp_delta_m=0.45,
                reference_lookahead_local_gate_weight=0.66,
                reference_lookahead_owner_mode="phase_active",
                reference_lookahead_entry_weight_source="curve_local_phase",
                reference_lookahead_fallback_active=False,
                reference_lookahead_entry_shorten_guard_active=True,
                reference_lookahead_entry_shorten_guard_delta_m=0.35,
                local_curve_reference_mode="shadow",
                local_curve_reference_active=True,
                local_curve_reference_shadow_only=True,
                local_curve_reference_valid=True,
                local_curve_reference_source="road_curvature",
                local_curve_reference_fallback_active=False,
                local_curve_reference_fallback_reason="",
                local_curve_reference_blend_weight=0.58,
                local_curve_reference_progress_weight=0.72,
                local_curve_reference_arc_curvature_abs=0.009,
                local_curve_reference_target_x=0.82,
                local_curve_reference_target_y=5.1,
                local_curve_reference_target_heading=0.14,
                local_curve_reference_target_distance_m=5.4,
                local_curve_reference_vs_planner_delta_m=0.63,
                local_curve_reference_curve_direction_sign=1.0,
                local_curve_reference_curve_progress_ratio=0.18,
                local_curve_reference_distance_to_curve_start_m=4.5,
                time_to_next_curve_start_s=1.25,
                pp_curve_local_floor_active=True,
                pp_curve_local_floor_m=4.7,
                pp_curve_local_lookahead_pre_floor=3.0,
                pp_curve_local_lookahead_post_floor=4.7,
                pp_curve_local_shorten_slew_active=True,
                pp_curve_local_shorten_delta_m=0.4,
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
        assert float(h5_file["control/curve_phase_term_time"][0]) == pytest.approx(0.77, rel=1e-6)
        assert float(h5_file["control/curve_phase_curvature_rise_abs"][0]) == pytest.approx(
            0.0024, rel=1e-6
        )
        assert int(h5_file["control/curve_preview_far_upcoming"][0]) == 1
        assert float(h5_file["control/curve_preview_far_phase"][0]) == pytest.approx(0.86, rel=1e-6)
        assert float(h5_file["control/curve_local_phase"][0]) == pytest.approx(0.44, rel=1e-6)
        assert float(h5_file["control/curve_local_phase_raw"][0]) == pytest.approx(0.52, rel=1e-6)
        local_state = h5_file["control/curve_local_state"][0]
        if isinstance(local_state, bytes):
            local_state = local_state.decode("utf-8")
        assert str(local_state) == "ENTRY"
        local_source = h5_file["control/curve_local_phase_source"][0]
        if isinstance(local_source, bytes):
            local_source = local_source.decode("utf-8")
        assert str(local_source) == "distance_time_path_gate"
        local_entry_driver = h5_file["control/curve_local_entry_driver"][0]
        if isinstance(local_entry_driver, bytes):
            local_entry_driver = local_entry_driver.decode("utf-8")
        assert str(local_entry_driver) == "distance"
        assert float(h5_file["control/curve_local_entry_severity"][0]) == pytest.approx(0.82, rel=1e-6)
        assert float(h5_file["control/curve_local_entry_on_effective"][0]) == pytest.approx(
            0.40, rel=1e-6
        )
        assert float(
            h5_file["control/curve_local_phase_distance_start_effective_m"][0]
        ) == pytest.approx(10.0, rel=1e-6)
        assert float(
            h5_file["control/curve_local_phase_time_start_effective_s"][0]
        ) == pytest.approx(1.6, rel=1e-6)
        assert int(h5_file["control/curve_local_arm_ready"][0]) == 1
        assert int(h5_file["control/curve_local_time_ready"][0]) == 0
        assert int(h5_file["control/curve_local_in_curve_now"][0]) == 0
        assert int(h5_file["control/curve_local_commit_ready"][0]) == 1
        local_commit_driver = h5_file["control/curve_local_commit_driver"][0]
        if isinstance(local_commit_driver, bytes):
            local_commit_driver = local_commit_driver.decode("utf-8")
        assert str(local_commit_driver) == "distance"
        assert float(h5_file["control/curve_local_arm_phase_raw"][0]) == pytest.approx(0.42, rel=1e-6)
        assert float(h5_file["control/curve_local_sustain_phase_raw"][0]) == pytest.approx(
            0.86, rel=1e-6
        )
        assert int(h5_file["control/curve_local_path_sustain_active"][0]) == 1
        assert int(h5_file["control/curve_local_distance_ready"][0]) == 1
        assert float(h5_file["control/curve_local_distance_horizon_m"][0]) == pytest.approx(
            7.5, rel=1e-6
        )
        assert float(h5_file["control/curve_local_time_horizon_s"][0]) == pytest.approx(
            1.6, rel=1e-6
        )
        assert int(h5_file["control/curve_local_reentry_ready"][0]) == 1
        assert int(h5_file["control/curve_local_rearm_cooldown_active"][0]) == 0
        assert int(h5_file["control/curve_local_force_straight_active"][0]) == 0
        assert int(h5_file["control/curve_local_commit_streak_frames"][0]) == 3
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
        assert int(h5_file["control/curve_intent_watchdog_triggered"][0]) == 1
        assert int(h5_file["control/curve_intent_speed_guardrail_active"][0]) == 1
        assert float(h5_file["control/curve_intent_speed_guardrail_cap_mps"][0]) == pytest.approx(
            7.0, rel=1e-6
        )
        assert float(h5_file["control/curve_intent_speed_guardrail_confidence"][0]) == pytest.approx(
            0.42, rel=1e-6
        )
        assert float(h5_file["control/reference_lookahead_target"][0]) == pytest.approx(10.2, rel=1e-6)
        assert float(h5_file["control/reference_lookahead_target_pre_entry_guard"][0]) == pytest.approx(
            9.8, rel=1e-6
        )
        assert float(h5_file["control/reference_lookahead_after_slew"][0]) == pytest.approx(9.8, rel=1e-6)
        assert float(h5_file["control/reference_lookahead_active"][0]) == pytest.approx(9.8, rel=1e-6)
        assert float(h5_file["control/reference_lookahead_owner_nominal_target"][0]) == pytest.approx(
            9.4, rel=1e-6
        )
        assert float(h5_file["control/reference_lookahead_owner_commit_band_target"][0]) == pytest.approx(
            5.6, rel=1e-6
        )
        assert float(h5_file["control/reference_lookahead_owner_entry_progress"][0]) == pytest.approx(
            0.62, rel=1e-6
        )
        assert float(
            h5_file["control/reference_lookahead_owner_commit_distance_progress"][0]
        ) == pytest.approx(0.55, rel=1e-6)
        assert float(
            h5_file["control/reference_lookahead_owner_commit_phase_progress"][0]
        ) == pytest.approx(0.28, rel=1e-6)
        assert float(h5_file["control/reference_lookahead_owner_commit_progress"][0]) == pytest.approx(
            0.55, rel=1e-6
        )
        assert float(
            h5_file["control/reference_lookahead_owner_commit_distance_start_effective_m"][0]
        ) == pytest.approx(7.5, rel=1e-6)
        assert int(h5_file["control/reference_lookahead_owner_commit_band_clamp_active"][0]) == 1
        assert float(
            h5_file["control/reference_lookahead_owner_commit_band_clamp_delta_m"][0]
        ) == pytest.approx(0.45, rel=1e-6)
        assert float(h5_file["control/reference_lookahead_local_gate_weight"][0]) == pytest.approx(
            0.66, rel=1e-6
        )
        owner_mode = h5_file["control/reference_lookahead_owner_mode"][0]
        if isinstance(owner_mode, bytes):
            owner_mode = owner_mode.decode("utf-8")
        assert str(owner_mode) == "phase_active"
        weight_source = h5_file["control/reference_lookahead_entry_weight_source"][0]
        if isinstance(weight_source, bytes):
            weight_source = weight_source.decode("utf-8")
        assert str(weight_source) == "curve_local_phase"
        assert float(h5_file["control/reference_lookahead_fallback_active"][0]) == pytest.approx(
            0.0, rel=1e-6
        )
        assert float(h5_file["control/reference_lookahead_entry_shorten_guard_active"][0]) == pytest.approx(
            1.0, rel=1e-6
        )
        assert float(h5_file["control/reference_lookahead_entry_shorten_guard_delta_m"][0]) == pytest.approx(
            0.35, rel=1e-6
        )
        local_arc_mode = h5_file["control/local_curve_reference_mode"][0]
        if isinstance(local_arc_mode, bytes):
            local_arc_mode = local_arc_mode.decode("utf-8")
        assert str(local_arc_mode) == "shadow"
        assert float(h5_file["control/local_curve_reference_active"][0]) == pytest.approx(
            1.0, rel=1e-6
        )
        assert float(h5_file["control/local_curve_reference_shadow_only"][0]) == pytest.approx(
            1.0, rel=1e-6
        )
        assert float(h5_file["control/local_curve_reference_valid"][0]) == pytest.approx(
            1.0, rel=1e-6
        )
        local_arc_source = h5_file["control/local_curve_reference_source"][0]
        if isinstance(local_arc_source, bytes):
            local_arc_source = local_arc_source.decode("utf-8")
        assert str(local_arc_source) == "road_curvature"
        assert float(h5_file["control/local_curve_reference_fallback_active"][0]) == pytest.approx(
            0.0, rel=1e-6
        )
        assert float(h5_file["control/local_curve_reference_blend_weight"][0]) == pytest.approx(
            0.58, rel=1e-6
        )
        assert float(h5_file["control/local_curve_reference_progress_weight"][0]) == pytest.approx(
            0.72, rel=1e-6
        )
        assert float(h5_file["control/local_curve_reference_arc_curvature_abs"][0]) == pytest.approx(
            0.009, rel=1e-6
        )
        assert float(h5_file["control/local_curve_reference_target_x"][0]) == pytest.approx(
            0.82, rel=1e-6
        )
        assert float(h5_file["control/local_curve_reference_target_y"][0]) == pytest.approx(
            5.1, rel=1e-6
        )
        assert float(h5_file["control/local_curve_reference_target_heading"][0]) == pytest.approx(
            0.14, rel=1e-6
        )
        assert float(h5_file["control/local_curve_reference_target_distance_m"][0]) == pytest.approx(
            5.4, rel=1e-6
        )
        assert float(h5_file["control/local_curve_reference_vs_planner_delta_m"][0]) == pytest.approx(
            0.63, rel=1e-6
        )
        assert float(h5_file["control/local_curve_reference_curve_direction_sign"][0]) == pytest.approx(
            1.0, rel=1e-6
        )
        assert float(h5_file["control/local_curve_reference_curve_progress_ratio"][0]) == pytest.approx(
            0.18, rel=1e-6
        )
        assert float(h5_file["control/local_curve_reference_distance_to_curve_start_m"][0]) == pytest.approx(
            4.5, rel=1e-6
        )
        assert float(h5_file["control/time_to_next_curve_start_s"][0]) == pytest.approx(1.25, rel=1e-6)
        assert float(h5_file["control/pp_curve_local_floor_active"][0]) == pytest.approx(1.0, rel=1e-6)
        assert float(h5_file["control/pp_curve_local_floor_m"][0]) == pytest.approx(4.7, rel=1e-6)
        assert float(h5_file["control/pp_curve_local_lookahead_pre_floor"][0]) == pytest.approx(
            3.0, rel=1e-6
        )
        assert float(h5_file["control/pp_curve_local_lookahead_post_floor"][0]) == pytest.approx(
            4.7, rel=1e-6
        )
        assert float(h5_file["control/pp_curve_local_shorten_slew_active"][0]) == pytest.approx(
            1.0, rel=1e-6
        )
        assert float(h5_file["control/pp_curve_local_shorten_delta_m"][0]) == pytest.approx(
            0.4, rel=1e-6
        )


def test_recorder_writes_transition_telemetry_fields(tmp_path: Path) -> None:
    recorder = DataRecorder(str(tmp_path), recording_name="control_command_transition_telemetry_test")
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
                throttle=0.2,
                brake=0.0,
                longitudinal_limiter_transition_active=True,
                longitudinal_limiter_state_code=2.0,
                longitudinal_accel_cmd_raw=0.45,
                longitudinal_accel_cmd_smoothed=0.31,
                steering_rate_limit_effective=0.22,
                steering_rate_limit_effective_raw=0.30,
                steering_rate_limit_effective_smoothed=0.24,
                steering_rate_limit_transition_active=True,
                speed_governor_active_limiter="curve_cap_tracking",
                speed_governor_active_limiter_code=6,
                speed_governor_cap_tracking_active=True,
                speed_governor_cap_tracking_error_mps=0.18,
                speed_governor_cap_tracking_mode="catch_up",
                speed_governor_cap_tracking_mode_code=1,
                speed_governor_cap_tracking_recovery_frames=7,
                speed_governor_cap_tracking_hard_ceiling_applied=False,
            ),
            perception_output=None,
            trajectory_output=None,
            unity_feedback=None,
        )

        recorder._write_control_commands([frame])
        recorder.h5_file.flush()
    finally:
        recorder.close()

    with h5py.File(tmp_path / "control_command_transition_telemetry_test.h5", "r") as h5_file:
        assert int(h5_file["control/longitudinal_limiter_transition_active"][0]) == 1
        assert float(h5_file["control/longitudinal_limiter_state_code"][0]) == pytest.approx(
            2.0, rel=1e-6
        )
        assert float(h5_file["control/longitudinal_accel_cmd_raw"][0]) == pytest.approx(
            0.45, rel=1e-6
        )
        assert float(h5_file["control/longitudinal_accel_cmd_smoothed"][0]) == pytest.approx(
            0.31, rel=1e-6
        )
        assert float(h5_file["control/steering_rate_limit_effective_raw"][0]) == pytest.approx(
            0.30, rel=1e-6
        )
        assert float(
            h5_file["control/steering_rate_limit_effective_smoothed"][0]
        ) == pytest.approx(0.24, rel=1e-6)
        assert int(h5_file["control/steering_rate_limit_transition_active"][0]) == 1
        assert float(h5_file["control/speed_governor_active_limiter_code"][0]) == pytest.approx(
            6.0, rel=1e-6
        )
        limiter_label = h5_file["control/speed_governor_active_limiter"][0]
        if isinstance(limiter_label, bytes):
            limiter_label = limiter_label.decode("utf-8")
        assert str(limiter_label) == "curve_cap_tracking"
        assert int(h5_file["control/speed_governor_cap_tracking_active"][0]) == 1
        assert float(h5_file["control/speed_governor_cap_tracking_error_mps"][0]) == pytest.approx(
            0.18, rel=1e-6
        )
        mode = h5_file["control/speed_governor_cap_tracking_mode"][0]
        if isinstance(mode, bytes):
            mode = mode.decode("utf-8")
        assert str(mode) == "catch_up"
        assert float(h5_file["control/speed_governor_cap_tracking_mode_code"][0]) == pytest.approx(
            1.0, rel=1e-6
        )
        assert int(h5_file["control/speed_governor_cap_tracking_recovery_frames"][0]) == 7
        assert int(h5_file["control/speed_governor_cap_tracking_hard_ceiling_applied"][0]) == 0


def test_recorder_writes_curvature_contract_and_backstop_fields(tmp_path: Path) -> None:
    recorder = DataRecorder(str(tmp_path), recording_name="control_command_curvature_contract_test")
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
                path_curvature_source_used="curvature_primary_abs",
                path_curvature_primary_abs=0.011,
                path_curvature_lane_abs=0.010,
                speed_governor_feasibility_backstop_active=True,
                speed_governor_feasibility_backstop_speed=7.4,
                curvature_primary_abs=0.011,
                curvature_primary_source="map_track",
                curvature_map_abs=0.011,
                curvature_lane_context_abs=0.010,
                curvature_preview_abs=0.012,
                curvature_source_diverged=False,
                curvature_map_authority_lost=False,
                curvature_source_divergence_abs=0.001,
                curvature_selection_reason="map_ok",
                map_health_ok=True,
                track_match_ok=True,
                map_segment_lookup_success_rate=99.5,
                map_teleport_skip_count=0,
                map_odometer_jump_rate=0.001,
                curvature_contract_consistent_controller=True,
                curvature_contract_consistent_governor=True,
                curvature_contract_consistent_intent=True,
                curvature_contract_consistent_all=True,
                curvature_contract_mismatch_reason="none",
            ),
            perception_output=None,
            trajectory_output=None,
            unity_feedback=None,
        )
        recorder._write_control_commands([frame])
        recorder.h5_file.flush()
    finally:
        recorder.close()

    with h5py.File(tmp_path / "control_command_curvature_contract_test.h5", "r") as h5_file:
        src = h5_file["control/path_curvature_source_used"][0]
        if isinstance(src, bytes):
            src = src.decode("utf-8")
        assert str(src) == "curvature_primary_abs"
        assert float(h5_file["control/path_curvature_primary_abs"][0]) == pytest.approx(0.011, rel=1e-6)
        assert float(h5_file["control/path_curvature_lane_abs"][0]) == pytest.approx(0.010, rel=1e-6)
        assert int(h5_file["control/speed_governor_feasibility_backstop_active"][0]) == 1
        assert float(h5_file["control/speed_governor_feasibility_backstop_speed"][0]) == pytest.approx(7.4, rel=1e-6)
        psrc = h5_file["control/curvature_primary_source"][0]
        if isinstance(psrc, bytes):
            psrc = psrc.decode("utf-8")
        assert str(psrc) == "map_track"
        assert float(h5_file["control/curvature_source_divergence_abs"][0]) == pytest.approx(0.001, rel=1e-6)
        assert int(h5_file["control/curvature_map_authority_lost"][0]) == 0
        assert int(h5_file["control/map_health_ok"][0]) == 1
        assert int(h5_file["control/track_match_ok"][0]) == 1
        assert float(h5_file["control/map_segment_lookup_success_rate"][0]) == pytest.approx(99.5, rel=1e-6)
        assert int(h5_file["control/map_teleport_skip_count"][0]) == 0
        assert int(h5_file["control/curvature_contract_consistent_all"][0]) == 1
