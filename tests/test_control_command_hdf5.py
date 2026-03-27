from pathlib import Path

import h5py
import numpy as np
import pytest

from data.formats.data_format import CameraFrame, ControlCommand, RecordingFrame, VehicleState
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
                curve_activation_blocker_mode="arm_phase_below_entry_threshold",
                curve_local_arm_phase_deficit=0.11,
                curve_local_arm_effect_score=0.37,
                curve_local_arm_effect_heading_term=0.21,
                curve_local_arm_effect_lateral_shift_term=0.29,
                curve_local_arm_effect_time_support_term=0.08,
                curve_local_dynamic_sustain_effect_score=0.24,
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
                mpc_kappa_bias_correction=-0.0014,
                mpc_kappa_bias_ema=0.093,
                mpc_kappa_bias_guard_active=True,
                mpc_kappa_bias_guard_limit=0.0010,
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
        blocker_mode = h5_file["control/curve_activation_blocker_mode"][0]
        if isinstance(blocker_mode, bytes):
            blocker_mode = blocker_mode.decode("utf-8")
        assert str(blocker_mode) == "arm_phase_below_entry_threshold"
        assert float(h5_file["control/curve_local_arm_phase_deficit"][0]) == pytest.approx(
            0.11, rel=1e-6
        )
        assert float(h5_file["control/curve_local_arm_effect_score"][0]) == pytest.approx(
            0.37, rel=1e-6
        )
        assert float(
            h5_file["control/curve_local_arm_effect_heading_term"][0]
        ) == pytest.approx(0.21, rel=1e-6)
        assert float(
            h5_file["control/curve_local_arm_effect_lateral_shift_term"][0]
        ) == pytest.approx(0.29, rel=1e-6)
        assert float(
            h5_file["control/curve_local_arm_effect_time_support_term"][0]
        ) == pytest.approx(0.08, rel=1e-6)
        assert float(
            h5_file["control/curve_local_dynamic_sustain_effect_score"][0]
        ) == pytest.approx(0.24, rel=1e-6)
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
        assert float(h5_file["control/mpc_kappa_bias_correction"][0]) == pytest.approx(
            -0.0014, rel=1e-6
        )
        assert float(h5_file["control/mpc_kappa_bias_ema"][0]) == pytest.approx(0.093, rel=1e-6)
        assert int(h5_file["control/mpc_kappa_bias_guard_active"][0]) == 1
        assert float(h5_file["control/mpc_kappa_bias_guard_limit"][0]) == pytest.approx(
            0.0010, rel=1e-6
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


def test_recorder_writes_sync_packet_and_post_jump_fields(tmp_path: Path) -> None:
    recorder = DataRecorder(str(tmp_path), recording_name="sync_packet_shadow_fields_test")
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
            sync_packet_mode="packet_shadow",
            sync_packet_schema_version=1,
            sync_packet_id=7,
            sync_packet_unity_frame_count=222,
            sync_packet_consume_policy="freshest_within_budget",
            sync_packet_complete=True,
            sync_packet_fallback_active=False,
            sync_packet_fallback_reason_code="none",
            sync_packet_queue_depth=3,
            sync_packet_drop_count=1,
            sync_packet_payload_queue_depth=2,
            sync_packet_payload_drop_count=4,
            sync_packet_orphan_camera_count=2,
            sync_packet_orphan_vehicle_count=4,
            sync_packet_timeout_count=5,
            sync_packet_skipped_unity_frames=6,
            sync_packet_age_ms=11.0,
            sync_packet_payload_oldest_age_ms=13.5,
            sync_packet_payload_bytes=921600,
            sync_packet_payload_fallback_reason_code="none",
            sync_packet_payload_selected_age_ms=42.0,
            sync_packet_payload_selected_fresh=True,
            sync_packet_payload_warn_age_exceeded=False,
            sync_packet_payload_stale_drop_count=3,
            sync_packet_payload_drained_count=3,
            sync_packet_payload_max_drained_age_ms=211.0,
            sync_packet_payload_selection_source="server_selector",
            sync_packet_payload_selection_fallback_active=False,
            sync_packet_payload_selection_fallback_reason_code="none",
            sync_packet_payload_server_queue_depth_after_select=0,
            sync_packet_payload_server_oldest_age_ms_after_select=np.nan,
            sync_packet_selection_result="complete",
            sync_packet_join_source="packet_key",
            sync_packet_join_key_present=True,
            sync_packet_join_failure_reason_code="none",
            sync_packet_join_failure_side_code="none",
            sync_packet_selected_failure_contract_reason_code="none",
            sync_packet_selected_failure_source_stage_code="none",
            sync_packet_source_key_present_camera=True,
            sync_packet_source_key_present_vehicle=True,
            sync_packet_selected_packet_key="avbridge:7:128",
            sync_packet_timeout_event_delta=0,
            sync_packet_coherence_pass=True,
            sync_packet_coherence_reason_code="coherent",
            sync_packet_complete_but_incoherent=False,
            sync_packet_front_vehicle_time_delta_budget_exceeded=False,
            sync_packet_front_vehicle_frame_delta_budget_exceeded=False,
            sync_packet_join_wait_budget_exceeded=False,
            sync_packet_component_age_budget_exceeded=False,
            sync_packet_source_context_queue_depth=2,
            sync_packet_source_context_dropped_stale_count=1,
            sync_packet_source_context_missing_count=0,
            sync_packet_source_context_frame_delta=0.0,
            sync_packet_source_context_time_delta_ms=4.0,
            sync_packet_source_bundle_close_reason="open",
            sync_packet_source_bundle_deadline_ms=100.0,
            sync_packet_source_bundle_age_ms=12.0,
            sync_packet_source_bundle_inflight_count=2,
            sync_packet_source_bundle_vehicle_state_built=True,
            sync_packet_source_bundle_vehicle_state_enqueued=True,
            sync_packet_source_bundle_vehicle_state_sent=True,
            sync_packet_source_bundle_camera_requested=True,
            sync_packet_source_camera_request_attempted=True,
            sync_packet_source_camera_request_accepted=True,
            sync_packet_source_camera_request_rejected_reason="",
            sync_packet_source_camera_request_skipped_reason="camera_capture_missing",
            sync_packet_source_camera_request_disposition_code="skipped_camera_capture_missing",
            sync_packet_source_camera_request_attempt_age_ms=6.0,
            sync_packet_source_camera_request_accept_age_ms=4.0,
            sync_packet_source_camera_request_queue_depth=1,
            sync_packet_source_bundle_active_transport_eligible=False,
            sync_packet_source_bundle_debug_unbundled_capture=True,
            sync_packet_camera_capture_contract_reason="no_source_bundle_context",
            sync_packet_source_bundle_camera_sent=True,
            sync_packet_source_bundle_aborted_before_vehicle_send=True,
            sync_packet_source_bundle_abort_reason="bundle_aborted_camera_request_not_attempted",
            sync_packet_source_vehicle_send_blocked_by_camera_request=True,
            sync_packet_source_bundle_superseded_before_send=False,
            sync_packet_active_camera_excluded_event_delta=2,
            sync_packet_active_camera_excluded_reason_code="no_source_bundle_context",
            sync_packet_unbundled_camera_entered_active_path_event_delta=0,
            sync_packet_join_wait_ms=18.0,
            sync_packet_key_match_count=9,
            sync_packet_unity_fallback_count=1,
            sync_packet_superseded_camera_count=2,
            sync_packet_superseded_vehicle_count=3,
            sync_packet_packet_superseded_camera_count=0,
            sync_packet_packet_superseded_vehicle_count=1,
            sync_front_age_ms=7.0,
            sync_vehicle_age_ms=9.0,
            sync_front_vehicle_frame_delta=0.0,
            sync_front_vehicle_time_delta_ms=4.0,
            sync_packet_missing_front=False,
            sync_packet_missing_vehicle=False,
            lead_collision_detected=True,
            lead_collision_override_active=True,
            acc_active=1.0,
            acc_target_gap_m=22.0,
            acc_gap_error_m=-21.9,
            acc_ttc_s=999.0,
            acc_state_code="COLLAPSED_GAP_STOP",
            acc_target_speed_mps=0.0,
            acc_request_estop=True,
            acc_safety_mode_code="collapsed_gap_stop",
        )
        frame = RecordingFrame(
            timestamp=0.0,
            frame_id=0,
            camera_frame=CameraFrame(
                image=np.zeros((1, 1, 3), dtype=np.uint8),
                timestamp=0.0,
                frame_id=0,
            ),
            vehicle_state=vehicle_state,
            control_command=ControlCommand(
                timestamp=0.0,
                steering=0.0,
                throttle=0.0,
                brake=0.0,
                teleport_detected=True,
                teleport_jump_m=2.5,
                teleport_expected_motion_m=1.2,
                teleport_motion_ratio=2.08,
                teleport_guard_suppressed=False,
                teleport_continuity_suspect=True,
                teleport_guard_reason_code="true_discontinuity",
                teleport_dynamic_threshold_m=3.4,
                teleport_hard_override_threshold_m=8.0,
                teleport_effective_dt_s=0.15,
                teleport_unity_dt_s=0.15,
                post_jump_cooldown_active=True,
                post_jump_cooldown_frames_remaining=17,
                post_jump_reason_code="teleport_guard",
                governor_target_speed_mps=15.0,
                acc_target_speed_mps=0.0,
                planner_target_speed_applied_mps=0.0,
                final_longitudinal_target_mps=0.0,
                final_longitudinal_owner_code="acc_collapsed_gap_stop",
                reference_velocity_source_code="post_jump_cooldown",
                reference_velocity_effective=7.5,
            ),
            perception_output=None,
            trajectory_output=None,
            unity_feedback=None,
        )

        recorder._write_vehicle_states([frame])
        recorder._write_control_commands([frame])
        recorder.h5_file.flush()
    finally:
        recorder.close()

    with h5py.File(tmp_path / "sync_packet_shadow_fields_test.h5", "r") as h5_file:
        mode = h5_file["vehicle/sync_packet_mode"][0]
        if isinstance(mode, bytes):
            mode = mode.decode("utf-8")
        assert str(mode) == "packet_shadow"
        assert int(h5_file["vehicle/sync_packet_schema_version"][0]) == 1
        assert int(h5_file["vehicle/sync_packet_id"][0]) == 7
        assert int(h5_file["vehicle/sync_packet_complete"][0]) == 1
        consume_policy = h5_file["vehicle/sync_packet_consume_policy"][0]
        if isinstance(consume_policy, bytes):
            consume_policy = consume_policy.decode("utf-8")
        assert str(consume_policy) == "freshest_within_budget"
        assert float(h5_file["vehicle/sync_packet_age_ms"][0]) == pytest.approx(11.0, rel=1e-6)
        assert int(h5_file["vehicle/sync_packet_payload_queue_depth"][0]) == 2
        assert int(h5_file["vehicle/sync_packet_payload_drop_count"][0]) == 4
        assert float(h5_file["vehicle/sync_packet_payload_oldest_age_ms"][0]) == pytest.approx(
            13.5, rel=1e-6
        )
        assert int(h5_file["vehicle/sync_packet_payload_bytes"][0]) == 921600
        payload_reason = h5_file["vehicle/sync_packet_payload_fallback_reason_code"][0]
        if isinstance(payload_reason, bytes):
            payload_reason = payload_reason.decode("utf-8")
        assert str(payload_reason) == "none"
        assert float(h5_file["vehicle/sync_packet_payload_selected_age_ms"][0]) == pytest.approx(
            42.0, rel=1e-6
        )
        assert int(h5_file["vehicle/sync_packet_payload_selected_fresh"][0]) == 1
        assert int(h5_file["vehicle/sync_packet_payload_warn_age_exceeded"][0]) == 0
        assert int(h5_file["vehicle/sync_packet_payload_stale_drop_count"][0]) == 3
        assert int(h5_file["vehicle/sync_packet_payload_drained_count"][0]) == 3
        assert float(h5_file["vehicle/sync_packet_payload_max_drained_age_ms"][0]) == pytest.approx(
            211.0, rel=1e-6
        )
        selection_source = h5_file["vehicle/sync_packet_payload_selection_source"][0]
        if isinstance(selection_source, bytes):
            selection_source = selection_source.decode("utf-8")
        assert str(selection_source) == "server_selector"
        assert int(h5_file["vehicle/sync_packet_payload_selection_fallback_active"][0]) == 0
        selection_reason = h5_file["vehicle/sync_packet_payload_selection_fallback_reason_code"][0]
        if isinstance(selection_reason, bytes):
            selection_reason = selection_reason.decode("utf-8")
        assert str(selection_reason) == "none"
        assert int(h5_file["vehicle/sync_packet_payload_server_queue_depth_after_select"][0]) == 0
        selection_result = h5_file["vehicle/sync_packet_selection_result"][0]
        if isinstance(selection_result, bytes):
            selection_result = selection_result.decode("utf-8")
        assert str(selection_result) == "complete"
        join_source = h5_file["vehicle/sync_packet_join_source"][0]
        if isinstance(join_source, bytes):
            join_source = join_source.decode("utf-8")
        assert str(join_source) == "packet_key"
        assert int(h5_file["vehicle/sync_packet_join_key_present"][0]) == 1
        join_failure_reason = h5_file["vehicle/sync_packet_join_failure_reason_code"][0]
        if isinstance(join_failure_reason, bytes):
            join_failure_reason = join_failure_reason.decode("utf-8")
        assert str(join_failure_reason) == "none"
        join_failure_side = h5_file["vehicle/sync_packet_join_failure_side_code"][0]
        if isinstance(join_failure_side, bytes):
            join_failure_side = join_failure_side.decode("utf-8")
        assert str(join_failure_side) == "none"
        selected_failure_reason = h5_file[
            "vehicle/sync_packet_selected_failure_contract_reason_code"
        ][0]
        if isinstance(selected_failure_reason, bytes):
            selected_failure_reason = selected_failure_reason.decode("utf-8")
        assert str(selected_failure_reason) == "none"
        selected_failure_stage = h5_file[
            "vehicle/sync_packet_selected_failure_source_stage_code"
        ][0]
        if isinstance(selected_failure_stage, bytes):
            selected_failure_stage = selected_failure_stage.decode("utf-8")
        assert str(selected_failure_stage) == "none"
        assert int(h5_file["vehicle/sync_packet_source_key_present_camera"][0]) == 1
        assert int(h5_file["vehicle/sync_packet_source_key_present_vehicle"][0]) == 1
        selected_packet_key = h5_file["vehicle/sync_packet_selected_packet_key"][0]
        if isinstance(selected_packet_key, bytes):
            selected_packet_key = selected_packet_key.decode("utf-8")
        assert str(selected_packet_key) == "avbridge:7:128"
        assert int(h5_file["vehicle/sync_packet_timeout_event_delta"][0]) == 0
        assert int(h5_file["vehicle/sync_packet_coherence_pass"][0]) == 1
        coherence_reason = h5_file["vehicle/sync_packet_coherence_reason_code"][0]
        if isinstance(coherence_reason, bytes):
            coherence_reason = coherence_reason.decode("utf-8")
        assert str(coherence_reason) == "coherent"
        assert int(h5_file["vehicle/sync_packet_complete_but_incoherent"][0]) == 0
        assert int(
            h5_file["vehicle/sync_packet_front_vehicle_time_delta_budget_exceeded"][0]
        ) == 0
        assert int(
            h5_file["vehicle/sync_packet_front_vehicle_frame_delta_budget_exceeded"][0]
        ) == 0
        assert int(h5_file["vehicle/sync_packet_join_wait_budget_exceeded"][0]) == 0
        assert int(
            h5_file["vehicle/sync_packet_component_age_budget_exceeded"][0]
        ) == 0
        assert int(h5_file["vehicle/sync_packet_source_context_queue_depth"][0]) == 2
        assert int(
            h5_file["vehicle/sync_packet_source_context_dropped_stale_count"][0]
        ) == 1
        assert int(h5_file["vehicle/sync_packet_source_context_missing_count"][0]) == 0
        assert float(h5_file["vehicle/sync_packet_source_context_frame_delta"][0]) == pytest.approx(
            0.0, rel=1e-6
        )
        assert float(
            h5_file["vehicle/sync_packet_source_context_time_delta_ms"][0]
        ) == pytest.approx(4.0, rel=1e-6)
        bundle_close_reason = h5_file["vehicle/sync_packet_source_bundle_close_reason"][0]
        if isinstance(bundle_close_reason, bytes):
            bundle_close_reason = bundle_close_reason.decode("utf-8")
        assert str(bundle_close_reason) == "open"
        assert float(h5_file["vehicle/sync_packet_source_bundle_deadline_ms"][0]) == pytest.approx(
            100.0, rel=1e-6
        )
        assert float(h5_file["vehicle/sync_packet_source_bundle_age_ms"][0]) == pytest.approx(
            12.0, rel=1e-6
        )
        assert int(h5_file["vehicle/sync_packet_source_bundle_inflight_count"][0]) == 2
        assert int(h5_file["vehicle/sync_packet_source_bundle_vehicle_state_built"][0]) == 1
        assert int(h5_file["vehicle/sync_packet_source_bundle_vehicle_state_enqueued"][0]) == 1
        assert int(h5_file["vehicle/sync_packet_source_bundle_vehicle_state_sent"][0]) == 1
        assert int(h5_file["vehicle/sync_packet_source_bundle_camera_requested"][0]) == 1
        assert int(h5_file["vehicle/sync_packet_source_camera_request_attempted"][0]) == 1
        assert int(h5_file["vehicle/sync_packet_source_camera_request_accepted"][0]) == 1
        rejected_reason = h5_file["vehicle/sync_packet_source_camera_request_rejected_reason"][0]
        if isinstance(rejected_reason, bytes):
            rejected_reason = rejected_reason.decode("utf-8")
        assert str(rejected_reason) == ""
        skipped_reason = h5_file["vehicle/sync_packet_source_camera_request_skipped_reason"][0]
        if isinstance(skipped_reason, bytes):
            skipped_reason = skipped_reason.decode("utf-8")
        assert str(skipped_reason) == "camera_capture_missing"
        disposition = h5_file["vehicle/sync_packet_source_camera_request_disposition_code"][0]
        if isinstance(disposition, bytes):
            disposition = disposition.decode("utf-8")
        assert str(disposition) == "skipped_camera_capture_missing"
        assert float(h5_file["vehicle/sync_packet_source_camera_request_attempt_age_ms"][0]) == pytest.approx(
            6.0, rel=1e-6
        )
        assert float(h5_file["vehicle/sync_packet_source_camera_request_accept_age_ms"][0]) == pytest.approx(
            4.0, rel=1e-6
        )
        assert int(h5_file["vehicle/sync_packet_source_camera_request_queue_depth"][0]) == 1
        assert int(h5_file["vehicle/sync_packet_source_bundle_active_transport_eligible"][0]) == 0
        assert int(h5_file["vehicle/sync_packet_source_bundle_debug_unbundled_capture"][0]) == 1
        capture_contract_reason = h5_file["vehicle/sync_packet_camera_capture_contract_reason"][0]
        if isinstance(capture_contract_reason, bytes):
            capture_contract_reason = capture_contract_reason.decode("utf-8")
        assert str(capture_contract_reason) == "no_source_bundle_context"
        assert int(h5_file["vehicle/sync_packet_source_bundle_camera_sent"][0]) == 1
        assert int(h5_file["vehicle/sync_packet_source_bundle_aborted_before_vehicle_send"][0]) == 1
        abort_reason = h5_file["vehicle/sync_packet_source_bundle_abort_reason"][0]
        if isinstance(abort_reason, bytes):
            abort_reason = abort_reason.decode("utf-8")
        assert str(abort_reason) == "bundle_aborted_camera_request_not_attempted"
        assert int(
            h5_file["vehicle/sync_packet_source_vehicle_send_blocked_by_camera_request"][0]
        ) == 1
        assert int(h5_file["vehicle/sync_packet_source_bundle_superseded_before_send"][0]) == 0
        assert int(h5_file["vehicle/sync_packet_active_camera_excluded_event_delta"][0]) == 2
        excluded_reason = h5_file["vehicle/sync_packet_active_camera_excluded_reason_code"][0]
        if isinstance(excluded_reason, bytes):
            excluded_reason = excluded_reason.decode("utf-8")
        assert str(excluded_reason) == "no_source_bundle_context"
        assert int(
            h5_file["vehicle/sync_packet_unbundled_camera_entered_active_path_event_delta"][0]
        ) == 0
        assert float(h5_file["vehicle/sync_packet_join_wait_ms"][0]) == pytest.approx(
            18.0, rel=1e-6
        )
        assert int(h5_file["vehicle/sync_packet_key_match_count"][0]) == 9
        assert int(h5_file["vehicle/sync_packet_unity_fallback_count"][0]) == 1
        assert int(h5_file["vehicle/sync_packet_superseded_camera_count"][0]) == 2
        assert int(h5_file["vehicle/sync_packet_superseded_vehicle_count"][0]) == 3
        assert int(h5_file["vehicle/sync_packet_packet_superseded_camera_count"][0]) == 0
        assert int(h5_file["vehicle/sync_packet_packet_superseded_vehicle_count"][0]) == 1
        assert int(h5_file["vehicle/lead_collision_detected"][0]) == 1
        assert int(h5_file["vehicle/lead_collision_override_active"][0]) == 1
        acc_state = h5_file["vehicle/acc_state_code"][0]
        if isinstance(acc_state, bytes):
            acc_state = acc_state.decode("utf-8")
        assert str(acc_state) == "COLLAPSED_GAP_STOP"
        assert float(h5_file["vehicle/acc_target_speed_mps"][0]) == pytest.approx(0.0, rel=1e-6)
        assert int(h5_file["vehicle/acc_request_estop"][0]) == 1
        assert int(h5_file["control/post_jump_cooldown_active"][0]) == 1
        assert int(h5_file["control/post_jump_cooldown_frames_remaining"][0]) == 17
        reason = h5_file["control/post_jump_reason_code"][0]
        if isinstance(reason, bytes):
            reason = reason.decode("utf-8")
        assert str(reason) == "teleport_guard"
        assert float(h5_file["control/reference_velocity_effective"][0]) == pytest.approx(
            7.5, rel=1e-6
        )
        assert float(h5_file["control/teleport_expected_motion_m"][0]) == pytest.approx(
            1.2, rel=1e-6
        )
        assert float(h5_file["control/teleport_motion_ratio"][0]) == pytest.approx(
            2.08, rel=1e-6
        )
        assert int(h5_file["control/teleport_guard_suppressed"][0]) == 0
        assert int(h5_file["control/teleport_continuity_suspect"][0]) == 1
        guard_reason = h5_file["control/teleport_guard_reason_code"][0]
        if isinstance(guard_reason, bytes):
            guard_reason = guard_reason.decode("utf-8")
        assert str(guard_reason) == "true_discontinuity"
        assert float(h5_file["control/governor_target_speed_mps"][0]) == pytest.approx(15.0, rel=1e-6)
        assert float(h5_file["control/planner_target_speed_applied_mps"][0]) == pytest.approx(0.0, rel=1e-6)
        assert float(h5_file["control/final_longitudinal_target_mps"][0]) == pytest.approx(0.0, rel=1e-6)
        owner_code = h5_file["control/final_longitudinal_owner_code"][0]
        if isinstance(owner_code, bytes):
            owner_code = owner_code.decode("utf-8")
        assert str(owner_code) == "acc_collapsed_gap_stop"
        ref_src = h5_file["control/reference_velocity_source_code"][0]
        if isinstance(ref_src, bytes):
            ref_src = ref_src.decode("utf-8")
        assert str(ref_src) == "post_jump_cooldown"
        assert float(h5_file["control/teleport_dynamic_threshold_m"][0]) == pytest.approx(
            3.4, rel=1e-6
        )
        assert float(
            h5_file["control/teleport_hard_override_threshold_m"][0]
        ) == pytest.approx(8.0, rel=1e-6)
        assert float(h5_file["control/teleport_effective_dt_s"][0]) == pytest.approx(
            0.15, rel=1e-6
        )
        assert float(h5_file["control/teleport_unity_dt_s"][0]) == pytest.approx(
            0.15, rel=1e-6
        )


def test_recorder_writes_mpc_gt_cross_track_contract_fields(tmp_path: Path) -> None:
    recorder = DataRecorder(str(tmp_path), recording_name="control_command_mpc_gt_contract_test")
    try:
        frame = RecordingFrame(
            timestamp=0.0,
            frame_id=0,
            camera_frame=CameraFrame(
                image=np.zeros((1, 1, 3), dtype=np.uint8),
                timestamp=0.0,
                frame_id=0,
            ),
            vehicle_state=VehicleState(
                timestamp=0.0,
                position=np.zeros(3, dtype=np.float32),
                rotation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                velocity=np.zeros(3, dtype=np.float32),
                angular_velocity=np.zeros(3, dtype=np.float32),
                speed=0.0,
                steering_angle=0.0,
                motor_torque=0.0,
                brake_torque=0.0,
                ground_truth_left_lane_line_x=-3.0,
                ground_truth_right_lane_line_x=3.0,
                ground_truth_lane_center_x=0.82,
                ground_truth_lane_center_x_lookahead=0.82,
                ground_truth_lane_center_x_at_car=0.04,
            ),
            control_command=ControlCommand(
                timestamp=0.0,
                steering=0.0,
                throttle=0.0,
                brake=0.0,
                mpc_gt_cross_track_m=0.04,
                mpc_gt_cross_track_at_car_m=0.04,
                mpc_gt_cross_track_lookahead_m=0.82,
                mpc_gt_cross_track_source_code="at_car",
            ),
            perception_output=None,
            trajectory_output=None,
            unity_feedback=None,
        )

        recorder.record_frame(frame)
        recorder.close()
    finally:
        if getattr(recorder, "h5_file", None) is not None:
            recorder.close()

    with h5py.File(tmp_path / "control_command_mpc_gt_contract_test.h5", "r") as h5_file:
        assert float(h5_file["control/mpc_gt_cross_track_m"][0]) == pytest.approx(0.04, rel=1e-6)
        assert float(h5_file["control/mpc_gt_cross_track_at_car_m"][0]) == pytest.approx(0.04, rel=1e-6)
        assert float(h5_file["control/mpc_gt_cross_track_lookahead_m"][0]) == pytest.approx(0.82, rel=1e-6)
        source = h5_file["control/mpc_gt_cross_track_source_code"][0]
        if isinstance(source, bytes):
            source = source.decode("utf-8")
        assert str(source) == "at_car"
        assert float(h5_file["ground_truth/lane_center_x_at_car"][0]) == pytest.approx(0.04, rel=1e-6)
