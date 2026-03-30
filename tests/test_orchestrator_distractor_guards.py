import pytest

from av_stack.orchestrator import AVStack


def _make_stack(trajectory_config: dict) -> AVStack:
    stack = AVStack.__new__(AVStack)
    stack.trajectory_config = trajectory_config

    stack._local_curve_reference_distractor_guard_active = False
    stack._local_curve_reference_distractor_guard_on_counter = 0
    stack._local_curve_reference_distractor_guard_off_counter = 0
    stack._local_curve_reference_distractor_guard_dwell_frames = 0
    stack._local_curve_reference_distractor_guard_trigger_raw_delta_m = float("nan")
    stack._local_curve_reference_distractor_guard_last_exit_raw_delta_m = float("nan")

    stack._reference_distractor_guard_active = False
    stack._reference_distractor_guard_on_counter = 0
    stack._reference_distractor_guard_off_counter = 0
    stack._reference_distractor_guard_dwell_frames = 0
    stack._reference_distractor_guard_trigger_center_error_m = float("nan")
    stack._reference_distractor_guard_last_exit_center_error_m = float("nan")
    stack._reference_distractor_input_guard_active = False
    stack._reference_distractor_input_guard_on_counter = 0
    stack._reference_distractor_input_guard_off_counter = 0
    stack._reference_distractor_input_guard_dwell_frames = 0
    stack._reference_distractor_input_guard_trigger_center_error_m = float("nan")
    stack._reference_distractor_input_guard_last_exit_center_error_m = float("nan")
    return stack


def test_reference_distractor_guard_anchors_to_at_car_center_for_wrong_target_straight() -> None:
    stack = _make_stack(
        {
            "reference_distractor_guard_enabled": True,
            "reference_distractor_guard_on_frames": 1,
            "reference_distractor_guard_off_frames": 2,
            "reference_distractor_guard_wrong_target_use_at_car_center": True,
            "reference_distractor_guard_wrong_target_center_error_on_m": 0.20,
            "reference_distractor_guard_wrong_target_center_error_full_m": 0.45,
            "reference_distractor_guard_wrong_target_width_error_on_m": 0.25,
            "reference_distractor_guard_wrong_target_width_error_full_m": 0.50,
            "reference_distractor_guard_wrong_target_weight_min": 0.20,
            "reference_distractor_guard_wrong_target_blend_on": 0.90,
            "reference_distractor_guard_wrong_target_blend_full": 1.0,
        }
    )

    result = stack._update_reference_distractor_guard(
        vehicle_state_dict={
            "radar_fwd_candidate_present": 1.0,
            "radar_fwd_reject_reason": "wrong_lane",
            "groundTruthLaneCenterXLookahead": 0.78,
            "groundTruthLaneCenterXAtCar": 0.0,
        },
        reference_point={"method": "lane_positions", "perception_center_x": 0.82, "x": 0.82, "y": 16.0, "heading": 0.0},
        left_lane_line_x=-1.6,
        right_lane_line_x=1.6,
        current_speed_mps=12.0,
        current_path_curvature=0.0,
        curve_local_state="STRAIGHT",
    )

    assert result["active"] is True
    assert result["expected_center_x_m"] == pytest.approx(0.0, rel=1e-6)
    assert result["guard_blend_weight"] >= 0.90
    assert result["reference_override"]["method"] == "distractor_guarded_map_lane"


def test_reference_distractor_input_guard_builds_synthetic_ego_lane_and_suppresses_coeffs() -> None:
    stack = _make_stack(
        {
            "target_lane_width_m": 3.6,
            "reference_distractor_input_guard_enabled": True,
            "reference_distractor_input_guard_on_frames": 1,
            "reference_distractor_input_guard_off_frames": 2,
            "reference_distractor_input_guard_speed_on": 8.5,
            "reference_distractor_input_guard_speed_full": 12.5,
            "reference_distractor_input_guard_center_error_on_m": 0.20,
            "reference_distractor_input_guard_center_error_full_m": 0.45,
            "reference_distractor_input_guard_width_error_on_m": 0.25,
            "reference_distractor_input_guard_width_error_full_m": 0.50,
            "reference_distractor_input_guard_weight_min": 0.20,
            "reference_distractor_input_guard_suppress_lane_coeffs": True,
            "reference_distractor_input_guard_seed_center_history": True,
        }
    )

    result = stack._update_reference_distractor_input_guard(
        vehicle_state_dict={
            "radar_fwd_candidate_present": 1.0,
            "radar_fwd_reject_reason": "wrong_lane",
            "groundTruthEgoLaneCenterXAtCar": 0.15,
        },
        left_lane_line_x=-1.10,
        right_lane_line_x=2.05,
        current_speed_mps=12.0,
        current_path_curvature=0.0,
        curve_local_state="STRAIGHT",
    )

    assert result["active"] is True
    assert result["expected_center_x_m"] == pytest.approx(0.15, rel=1e-6)
    assert result["synthetic_left_lane_x_m"] == pytest.approx(-1.65, rel=1e-6)
    assert result["synthetic_right_lane_x_m"] == pytest.approx(1.95, rel=1e-6)
    assert result["synthetic_lane_width_m"] == pytest.approx(3.6, rel=1e-6)
    assert result["suppress_lane_coeffs"] is True
    assert result["center_history_seeded"] is True


def test_local_curve_reference_distractor_guard_uses_active_state_thresholds() -> None:
    stack = _make_stack(
        {
            "local_curve_reference_distractor_guard_enabled": True,
            "local_curve_reference_distractor_guard_on_frames": 1,
            "local_curve_reference_distractor_guard_off_frames": 2,
            "local_curve_reference_distractor_guard_speed_on": 12.5,
            "local_curve_reference_distractor_guard_speed_full": 14.5,
            "local_curve_reference_distractor_guard_curvature_on": 0.0015,
            "local_curve_reference_distractor_guard_curvature_full": 0.0020,
            "local_curve_reference_distractor_guard_raw_delta_on_m": 0.90,
            "local_curve_reference_distractor_guard_raw_delta_full_m": 2.20,
            "local_curve_reference_distractor_guard_weight_min": 0.35,
            "local_curve_reference_distractor_guard_active_raw_delta_on_m": 0.60,
            "local_curve_reference_distractor_guard_active_raw_delta_full_m": 1.20,
            "local_curve_reference_distractor_guard_active_weight_min": 0.20,
            "local_curve_reference_distractor_guard_active_blend_floor_on": 0.38,
            "local_curve_reference_distractor_guard_active_blend_floor_full": 0.62,
            "local_curve_reference_distractor_guard_active_max_blend_on": 0.42,
            "local_curve_reference_distractor_guard_active_max_blend_full": 0.72,
            "local_curve_reference_distractor_guard_active_ignore_progress_gate": True,
            "local_curve_reference_distractor_guard_active_unwind_start": 0.45,
            "local_curve_reference_distractor_guard_active_unwind_end": 0.75,
        }
    )

    result = stack._update_local_curve_reference_distractor_guard(
        vehicle_state_dict={
            "radar_fwd_candidate_present": 1.0,
            "radar_fwd_reject_reason": "opposite_direction",
        },
        local_curve_reference={
            "local_curve_reference_valid": True,
            "local_curve_reference_fallback_active": False,
            "local_curve_reference_raw_delta_m": 1.10,
            "local_curve_reference_arc_curvature_abs": 0.0020,
        },
        current_speed_mps=14.0,
        curve_local_state="ENTRY",
        current_curve_progress_ratio=0.40,
        road_curvature_abs=0.0020,
        preview_curvature_abs=0.0020,
    )

    assert result["active"] is True
    assert result["reason"] == "distractor_opposite_direction"
    assert result["trigger_weight"] >= 0.20
    assert result["guard_blend_floor"] >= 0.38
    assert result["config_override"]["local_curve_reference_max_blend"] >= 0.42
