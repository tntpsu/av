"""
Tests for control stack.
"""

import pytest
import numpy as np
from control.pid_controller import PIDController, LateralController, LongitudinalController, VehicleController


def test_pid_controller():
    """Test PID controller."""
    pid = PIDController(kp=1.0, ki=0.1, kd=0.5, output_limit=(-1.0, 1.0))
    
    # Test step response
    error = 1.0
    output = pid.update(error, dt=0.1)
    
    assert output is not None
    assert -1.0 <= output <= 1.0


def test_lateral_controller():
    """Test lateral controller."""
    controller = LateralController(kp=1.0, kd=0.5)
    
    reference_point = {
        'x': 0.0,
        'y': 10.0,
        'heading': 0.0,
        'velocity': 10.0
    }
    
    steering = controller.compute_steering(
        current_heading=0.0,
        reference_point=reference_point
    )
    
    assert -1.0 <= steering <= 1.0


def test_lateral_controller_metadata_includes_steering_terms():
    """Metadata should include feedforward/feedback steering breakdown."""
    controller = LateralController(kp=1.0, kd=0.2)
    reference_point = {
        'x': 0.3,
        'y': 10.0,
        'heading': 0.0,
        'velocity': 10.0,
        'curvature': 0.01,
    }
    meta = controller.compute_steering(
        current_heading=0.0,
        reference_point=reference_point,
        return_metadata=True,
    )

    assert 'feedforward_steering' in meta
    assert 'feedback_steering' in meta
    assert 'total_error_scaled' in meta
    assert isinstance(meta['feedforward_steering'], float)
    assert isinstance(meta['feedback_steering'], float)
    assert isinstance(meta['total_error_scaled'], float)


def test_lateral_controller_straight_proxy_alias_matches_legacy_flag():
    """Control straight proxy should match legacy is_straight exactly."""
    controller = LateralController(kp=1.0, kd=0.2, straight_curvature_threshold=0.01)
    reference_point = {
        'x': 0.1,
        'y': 10.0,
        'heading': 0.012,
        'velocity': 10.0,
        'curvature': 0.0,
    }
    meta = controller.compute_steering(
        current_heading=0.0,
        reference_point=reference_point,
        return_metadata=True,
    )

    assert 'is_straight' in meta
    assert 'is_control_straight_proxy' in meta
    assert meta['is_control_straight_proxy'] == meta['is_straight']


def test_lateral_controller_reports_road_straight_validity_separately():
    """Road-straight signal should track curvature validity independently of proxy."""
    controller = LateralController(
        kp=1.0,
        kd=0.2,
        straight_curvature_threshold=0.01,
        road_straight_hold_invalid_frames=0,
    )

    # No curvature provided -> road curvature invalid, proxy still evaluated from fallback.
    meta_missing = controller.compute_steering(
        current_heading=0.0,
        reference_point={
            'x': 0.0,
            'y': 10.0,
            'heading': 0.02,  # fallback proxy says not straight
            'velocity': 10.0,
        },
        return_metadata=True,
    )
    assert meta_missing['road_curvature_valid'] is False
    assert meta_missing['is_road_straight'] is None
    assert meta_missing['is_control_straight_proxy'] is False

    # Curvature provided (even zero) with no "missing" source -> valid road curvature.
    meta_zero_curv = controller.compute_steering(
        current_heading=0.0,
        reference_point={
            'x': 0.0,
            'y': 10.0,
            'heading': 0.02,
            'velocity': 10.0,
            'curvature': 0.0,
        },
        return_metadata=True,
    )
    assert bool(meta_zero_curv['road_curvature_valid']) is True
    assert meta_zero_curv['road_curvature_abs'] == pytest.approx(0.0)
    assert meta_zero_curv['is_road_straight'] is True
    assert meta_zero_curv['is_control_straight_proxy'] is False

    # Explicit missing source keeps road signal unknown while proxy still works.
    meta_missing_source = controller.compute_steering(
        current_heading=0.0,
        reference_point={
            'x': 0.0,
            'y': 10.0,
            'heading': 0.02,
            'velocity': 10.0,
            'curvature': 0.0,
            'curvature_source': 'missing',
        },
        return_metadata=True,
    )
    assert bool(meta_missing_source['road_curvature_valid']) is False
    assert meta_missing_source['is_road_straight'] is None
    assert meta_missing_source['is_control_straight_proxy'] is False


def test_lateral_controller_curve_upcoming_gates_curve_entry_schedule():
    """Curve-entry schedule should arm when curve_at_car rises (after distance wait)."""
    controller = LateralController(
        kp=1.0,
        kd=0.2,
        curve_entry_schedule_enabled=True,
        curve_entry_schedule_frames=8,
        curve_upcoming_enter_threshold=0.010,
        curve_upcoming_exit_threshold=0.008,
        curve_upcoming_on_frames=2,
        curve_upcoming_off_frames=1,
    )
    base_ref = {'x': 0.0, 'y': 0.05, 'velocity': 10.0}

    m0 = controller.compute_steering(
        current_heading=0.0,
        reference_point={**base_ref, 'heading': 0.011, 'curvature': 0.011},
        current_speed=10.0,
        return_metadata=True,
    )
    assert m0['curve_upcoming'] is False
    assert m0['curve_at_car'] is False
    assert m0['curve_entry_schedule_triggered'] is False

    m1 = controller.compute_steering(
        current_heading=0.0,
        reference_point={**base_ref, 'heading': 0.012, 'curvature': 0.012},
        current_speed=10.0,
        return_metadata=True,
    )
    assert m1['curve_upcoming'] is True
    assert m1['curve_at_car'] is True
    assert m1['curve_entry_schedule_triggered'] is True


def test_lateral_controller_curve_upcoming_gates_curve_commit_trigger():
    """Curve-commit trigger should wait for curve_at_car stabilization."""
    controller = LateralController(
        kp=1.0,
        kd=0.2,
        curve_entry_schedule_enabled=False,
        curve_commit_mode_enabled=True,
        curve_commit_mode_max_frames=12,
        curve_upcoming_enter_threshold=0.010,
        curve_upcoming_exit_threshold=0.008,
        curve_upcoming_on_frames=2,
        curve_upcoming_off_frames=1,
    )
    base_ref = {'x': 0.0, 'y': 0.05, 'velocity': 10.0}

    m0 = controller.compute_steering(
        current_heading=0.0,
        reference_point={**base_ref, 'heading': 0.011, 'curvature': 0.011},
        current_speed=10.0,
        return_metadata=True,
    )
    assert m0['curve_upcoming'] is False
    assert m0['curve_at_car'] is False
    assert m0['curve_commit_mode_triggered'] is False

    m1 = controller.compute_steering(
        current_heading=0.0,
        reference_point={**base_ref, 'heading': 0.012, 'curvature': 0.012},
        current_speed=10.0,
        return_metadata=True,
    )
    assert m1['curve_upcoming'] is True
    assert m1['curve_at_car'] is True
    assert m1['curve_commit_mode_triggered'] is True


def test_lateral_controller_curve_at_car_waits_out_reference_distance():
    """curve_at_car should remain false until traveled distance reaches initial reference y."""
    controller = LateralController(
        kp=1.0,
        kd=0.2,
        curve_upcoming_enter_threshold=0.010,
        curve_upcoming_exit_threshold=0.008,
        curve_upcoming_on_frames=1,
        curve_upcoming_off_frames=1,
    )
    ref = {'x': 0.0, 'y': 4.6, 'heading': 0.02, 'velocity': 10.0, 'curvature': 0.0}

    m0 = controller.compute_steering(
        current_heading=0.0,
        reference_point=ref,
        current_speed=10.0,
        return_metadata=True,
    )
    assert m0['curve_upcoming'] is True
    assert m0['curve_at_car'] is False
    assert m0['curve_at_car_distance_remaining_m'] == pytest.approx(4.6 - (10.0 * 0.033), rel=1e-2)

    # Advance ~4.6m at 10m/s with dt=0.033s internal step.
    for _ in range(13):
        m = controller.compute_steering(
            current_heading=0.0,
            reference_point=ref,
            current_speed=10.0,
            return_metadata=True,
        )
    assert m['curve_at_car'] is True


def test_curve_commit_mode_does_not_exit_on_proxy_straight_alone():
    """Commit mode should not hand off solely because proxy becomes straight."""
    controller = LateralController(
        kp=1.0,
        kd=0.2,
        curve_entry_schedule_enabled=False,
        curve_commit_mode_enabled=True,
        curve_commit_mode_max_frames=6,
        curve_upcoming_enter_threshold=0.010,
        curve_upcoming_exit_threshold=0.008,
        curve_upcoming_on_frames=1,
        curve_upcoming_off_frames=1,
    )
    # Trigger commit.
    m0 = controller.compute_steering(
        current_heading=0.0,
        reference_point={'x': 0.0, 'y': 0.05, 'heading': 0.02, 'velocity': 10.0, 'curvature': 0.02},
        current_speed=10.0,
        return_metadata=True,
    )
    assert m0['curve_commit_mode_triggered'] is True

    # Next frame: proxy goes straight (heading near zero). Commit should persist by countdown.
    m1 = controller.compute_steering(
        current_heading=0.0,
        reference_point={'x': 0.0, 'y': 0.05, 'heading': 0.0, 'velocity': 10.0, 'curvature': 0.0},
        current_speed=10.0,
        return_metadata=True,
    )
    assert m1['is_straight'] is True
    assert m1['curve_commit_mode_active'] is True
    assert m1['curve_commit_mode_handoff_triggered'] is False
    assert int(m1['curve_commit_mode_frames_remaining']) > 0


def test_curve_commit_mode_exit_requires_consecutive_handoff_frames():
    """Commit handoff should require configured consecutive qualifying frames."""
    controller = LateralController(
        kp=1.0,
        kd=0.2,
        curve_entry_schedule_enabled=False,
        curve_commit_mode_enabled=True,
        curve_commit_mode_max_frames=10,
        curve_commit_mode_exit_consecutive_frames=3,
    )
    controller._curve_commit_mode_frames_remaining = 6
    ref = {'x': 0.3, 'y': 8.0, 'heading': 0.0, 'velocity': 8.0, 'curvature': 0.01}

    # Force three qualifying handoff candidates in a row.
    m0 = m1 = m2 = None
    for i in range(3):
        controller._last_error_magnitude = 1.0  # keeps "error_recovering_commit" true
        m = controller.compute_steering(
            current_heading=0.0,
            reference_point=ref,
            current_speed=8.0,
            dt=0.033,
            return_metadata=True,
        )
        if i == 0:
            m0 = m
        elif i == 1:
            m1 = m
        else:
            m2 = m

    assert m0['curve_commit_mode_handoff_triggered'] is False
    assert m1['curve_commit_mode_handoff_triggered'] is False
    assert m2['curve_commit_mode_handoff_triggered'] is True


def test_curve_entry_schedule_handoff_respects_min_hold_frames():
    """Entry schedule handoff should not occur before configured hold window."""
    controller = LateralController(
        kp=1.0,
        kd=0.2,
        curve_entry_schedule_enabled=True,
        curve_entry_schedule_frames=24,
        curve_entry_schedule_min_hold_frames=4,
        curve_entry_schedule_handoff_transfer_ratio=0.0,
    )
    controller._curve_entry_schedule_frames_remaining = 6
    controller._curve_entry_schedule_elapsed_frames = 0
    ref = {'x': 0.2, 'y': 8.0, 'heading': 0.0, 'velocity': 8.0, 'curvature': 0.01}

    m0 = m1 = m2 = m3 = None
    for i in range(4):
        controller._last_error_magnitude = 1.0
        m = controller.compute_steering(
            current_heading=0.0,
            reference_point=ref,
            current_speed=8.0,
            dt=0.033,
            return_metadata=True,
        )
        if i == 0:
            m0 = m
        elif i == 1:
            m1 = m
        elif i == 2:
            m2 = m
        else:
            m3 = m

    assert m0['curve_entry_schedule_handoff_triggered'] is False
    assert m1['curve_entry_schedule_handoff_triggered'] is False
    assert m2['curve_entry_schedule_handoff_triggered'] is False
    assert m3['curve_entry_schedule_handoff_triggered'] is True


def test_curve_entry_schedule_handoff_waits_for_curve_progress_ratio():
    """Distance-track handoff should wait until configured in-curve progress ratio."""
    controller = LateralController(
        kp=1.0,
        kd=0.2,
        curve_phase_use_distance_track=True,
        curve_phase_track_name='s_loop',
        curve_entry_schedule_enabled=True,
        curve_entry_schedule_frames=24,
        curve_entry_schedule_min_hold_frames=1,
        curve_entry_schedule_min_curve_progress_ratio=0.20,
        curve_entry_schedule_handoff_transfer_ratio=0.0,
    )
    ref = {'x': 0.2, 'y': 4.7, 'heading': 0.0, 'velocity': 8.0, 'curvature': 0.01}
    total_len = 452.0353755551324

    # Trigger inside the first curve.
    m0 = controller.compute_steering(
        current_heading=0.0,
        reference_point=ref,
        current_speed=8.0,
        road_center_reference_t=25.2 / total_len,
        dt=0.033,
        return_metadata=True,
    )
    assert m0['curve_entry_schedule_triggered'] is True

    # Early in curve (~8% progress): hold elapsed but handoff still blocked by progress gate.
    m1 = controller.compute_steering(
        current_heading=0.0,
        reference_point=ref,
        current_speed=8.0,
        road_center_reference_t=30.0 / total_len,
        dt=0.033,
        return_metadata=True,
    )
    assert m1['curve_entry_schedule_handoff_triggered'] is False

    # Deeper into curve (~24% progress): handoff can now occur.
    m2 = controller.compute_steering(
        current_heading=0.0,
        reference_point=ref,
        current_speed=8.0,
        road_center_reference_t=40.0 / total_len,
        dt=0.033,
        return_metadata=True,
    )
    assert m2['curve_entry_schedule_handoff_triggered'] is True


def test_dynamic_curve_authority_boosts_rate_limit_in_curve():
    """Dynamic authority should raise steering-rate limit when deficit exists in-curve."""
    common = dict(
        kp=1.0,
        kd=0.2,
        curve_entry_schedule_enabled=False,
        curve_commit_mode_enabled=False,
        curve_upcoming_enter_threshold=0.005,
        curve_upcoming_exit_threshold=0.004,
        curve_upcoming_on_frames=1,
        curve_upcoming_off_frames=1,
        dynamic_curve_rate_boost_gain=1.0,
        dynamic_curve_rate_boost_max=0.20,
    )
    controller_dynamic = LateralController(
        **common,
        dynamic_curve_authority_enabled=True,
    )
    controller_static = LateralController(
        **common,
        dynamic_curve_authority_enabled=False,
    )
    ref = {'x': 1.5, 'y': 0.0, 'heading': 0.2, 'velocity': 8.0, 'curvature': 0.02}

    m_dyn = controller_dynamic.compute_steering(
        current_heading=0.0,
        reference_point=ref,
        current_speed=8.0,
        dt=0.033,
        return_metadata=True,
    )
    m_static = controller_static.compute_steering(
        current_heading=0.0,
        reference_point=ref,
        current_speed=8.0,
        dt=0.033,
        return_metadata=True,
    )

    assert m_dyn['curve_at_car'] is True
    assert m_dyn['dynamic_curve_authority_active'] is True
    assert float(m_dyn['dynamic_curve_rate_boost']) > 0.0
    assert float(m_dyn['steering_rate_limit_effective']) > float(
        m_static['steering_rate_limit_effective']
    )


def test_curve_entry_schedule_dynamic_fallback_waits_for_deficit_streak():
    """With dynamic fallback mode, schedule should trigger only after deficit streak threshold."""
    controller = LateralController(
        kp=1.0,
        kd=0.2,
        curve_entry_schedule_enabled=True,
        curve_entry_schedule_frames=12,
        curve_entry_schedule_min_hold_frames=10,
        curve_entry_schedule_fallback_only_when_dynamic=True,
        curve_entry_schedule_fallback_deficit_frames=2,
        curve_entry_schedule_fallback_rate_deficit_min=0.0,
        dynamic_curve_authority_enabled=True,
        dynamic_curve_rate_boost_gain=0.0,
        curve_commit_mode_enabled=False,
        curve_upcoming_enter_threshold=0.005,
        curve_upcoming_exit_threshold=0.004,
        curve_upcoming_on_frames=1,
        curve_upcoming_off_frames=1,
    )
    ref = {'x': 1.5, 'y': 0.0, 'heading': 0.2, 'velocity': 8.0, 'curvature': 0.02}

    m0 = controller.compute_steering(
        current_heading=0.0,
        reference_point=ref,
        current_speed=8.0,
        dt=0.033,
        return_metadata=True,
    )
    m1 = controller.compute_steering(
        current_heading=0.0,
        reference_point=ref,
        current_speed=8.0,
        dt=0.033,
        return_metadata=True,
    )
    m2 = controller.compute_steering(
        current_heading=0.0,
        reference_point=ref,
        current_speed=8.0,
        dt=0.033,
        return_metadata=True,
    )

    assert m0['curve_entry_schedule_triggered'] is False
    assert m1['curve_entry_schedule_triggered'] is False
    assert m2['curve_entry_schedule_triggered'] is True


def test_curve_entry_schedule_dynamic_fallback_triggers_once_per_curve_segment():
    """Dynamic fallback should not retrigger schedule repeatedly while staying in same curve."""
    controller = LateralController(
        kp=1.0,
        kd=0.2,
        curve_entry_schedule_enabled=True,
        curve_entry_schedule_frames=2,
        curve_entry_schedule_min_hold_frames=10,
        curve_entry_schedule_fallback_only_when_dynamic=True,
        curve_entry_schedule_fallback_deficit_frames=1,
        curve_entry_schedule_fallback_rate_deficit_min=0.0,
        dynamic_curve_authority_enabled=True,
        dynamic_curve_rate_boost_gain=0.0,
        curve_commit_mode_enabled=False,
        curve_upcoming_enter_threshold=0.005,
        curve_upcoming_exit_threshold=0.004,
        curve_upcoming_on_frames=1,
        curve_upcoming_off_frames=1,
    )
    ref = {'x': 1.5, 'y': 0.0, 'heading': 0.2, 'velocity': 8.0, 'curvature': 0.02}

    triggers = 0
    for _ in range(8):
        m = controller.compute_steering(
            current_heading=0.0,
            reference_point=ref,
            current_speed=8.0,
            dt=0.033,
            return_metadata=True,
        )
        if m['curve_entry_schedule_triggered']:
            triggers += 1

    assert triggers == 1


def test_dynamic_curve_boost_cap_scales_with_speed_and_comfort():
    """Dynamic boost cap should increase at low speed and shut down past comfort envelope."""
    base = dict(
        kp=1.0,
        kd=0.2,
        curve_entry_schedule_enabled=False,
        curve_commit_mode_enabled=False,
        curve_upcoming_enter_threshold=0.005,
        curve_upcoming_exit_threshold=0.004,
        curve_upcoming_on_frames=1,
        curve_upcoming_off_frames=1,
        dynamic_curve_authority_enabled=True,
        dynamic_curve_rate_boost_gain=1.0,
        dynamic_curve_rate_boost_max=0.30,
        dynamic_curve_speed_low_mps=4.0,
        dynamic_curve_speed_high_mps=10.0,
        dynamic_curve_speed_boost_max_scale=1.4,
        dynamic_curve_comfort_lat_accel_comfort_max_g=0.18,
        dynamic_curve_comfort_lat_accel_peak_max_g=0.25,
        dynamic_curve_comfort_lat_jerk_comfort_max_gps=0.30,
    )
    ref = {'x': 1.5, 'y': 0.0, 'heading': 0.2, 'velocity': 8.0, 'curvature': 0.02}

    low_speed = LateralController(**base).compute_steering(
        current_heading=0.0,
        reference_point=ref,
        current_speed=4.0,
        dt=0.033,
        return_metadata=True,
    )
    high_speed = LateralController(**base).compute_steering(
        current_heading=0.0,
        reference_point=ref,
        current_speed=10.0,
        dt=0.033,
        return_metadata=True,
    )
    assert low_speed['dynamic_curve_speed_scale'] > high_speed['dynamic_curve_speed_scale']
    assert (
        low_speed['dynamic_curve_rate_boost_cap_effective']
        > high_speed['dynamic_curve_rate_boost_cap_effective']
    )

    over_comfort_ref = {'x': 1.5, 'y': 0.0, 'heading': 0.2, 'velocity': 10.0, 'curvature': 0.03}
    over_comfort = LateralController(**base).compute_steering(
        current_heading=0.0,
        reference_point=over_comfort_ref,
        current_speed=10.0,
        dt=0.033,
        return_metadata=True,
    )
    assert over_comfort['dynamic_curve_lateral_accel_est_g'] > 0.25
    assert over_comfort['dynamic_curve_comfort_scale'] == 0.0
    assert over_comfort['dynamic_curve_rate_boost_cap_effective'] == 0.0
    assert over_comfort['dynamic_curve_rate_boost'] == 0.0


def test_dynamic_curve_jerk_soft_penalty_does_not_hard_disable_boost():
    """A single high jerk frame should soft-dampen boost, not hard-disable it."""
    controller = LateralController(
        kp=1.0,
        kd=0.2,
        curve_entry_schedule_enabled=False,
        curve_commit_mode_enabled=False,
        dynamic_curve_authority_enabled=True,
        dynamic_curve_rate_boost_gain=1.0,
        dynamic_curve_rate_boost_max=0.30,
        dynamic_curve_comfort_lat_accel_comfort_max_g=0.18,
        dynamic_curve_comfort_lat_accel_peak_max_g=0.25,
        dynamic_curve_comfort_lat_jerk_comfort_max_gps=0.30,
        dynamic_curve_lat_jerk_smoothing_alpha=0.25,
        dynamic_curve_lat_jerk_soft_start_ratio=0.60,
        dynamic_curve_lat_jerk_soft_floor_scale=0.35,
        curve_upcoming_enter_threshold=0.005,
        curve_upcoming_exit_threshold=0.004,
        curve_upcoming_on_frames=1,
        curve_upcoming_off_frames=1,
    )
    # Prime history at low curvature.
    controller.compute_steering(
        current_heading=0.0,
        reference_point={'x': 0.5, 'y': 0.0, 'heading': 0.0, 'velocity': 8.0, 'curvature': 0.001},
        current_speed=8.0,
        dt=0.033,
        return_metadata=True,
    )
    # Sudden curvature jump creates high raw jerk estimate.
    m = controller.compute_steering(
        current_heading=0.0,
        reference_point={'x': 1.5, 'y': 0.0, 'heading': 0.2, 'velocity': 8.0, 'curvature': 0.02},
        current_speed=8.0,
        dt=0.033,
        return_metadata=True,
    )
    assert m['dynamic_curve_lateral_jerk_est_gps'] > m['dynamic_curve_lateral_jerk_est_smoothed_gps']
    assert 0.0 < float(m['dynamic_curve_comfort_jerk_penalty']) <= 1.0
    assert float(m['dynamic_curve_rate_boost_cap_effective']) > 0.0


def test_dynamic_curve_hard_clip_boost_relaxes_clip_under_comfort_headroom():
    """Dynamic authority should be able to lift hard clip limit when comfort allows."""
    base = dict(
        kp=1.0,
        kd=0.2,
        max_steering=0.2,
        curve_entry_schedule_enabled=False,
        curve_commit_mode_enabled=False,
        dynamic_curve_authority_enabled=True,
        dynamic_curve_rate_boost_gain=1.0,
        dynamic_curve_rate_boost_max=0.30,
        dynamic_curve_hard_clip_boost_max=0.12,
        dynamic_curve_comfort_lat_accel_comfort_max_g=0.18,
        dynamic_curve_comfort_lat_accel_peak_max_g=0.25,
        dynamic_curve_comfort_lat_jerk_comfort_max_gps=0.30,
        curve_upcoming_enter_threshold=0.005,
        curve_upcoming_exit_threshold=0.004,
        curve_upcoming_on_frames=1,
        curve_upcoming_off_frames=1,
    )
    ref = {'x': 2.0, 'y': 0.0, 'heading': 0.4, 'velocity': 8.0, 'curvature': 0.02}
    with_clip_controller = LateralController(**base)
    with_clip_controller.last_steering = -0.5
    with_clip_boost = with_clip_controller.compute_steering(
        current_heading=0.0,
        reference_point=ref,
        current_speed=8.0,
        dt=0.033,
        return_metadata=True,
    )
    no_clip_controller = LateralController(
        **{**base, 'dynamic_curve_hard_clip_boost_gain': 0.0}
    )
    no_clip_controller.last_steering = -0.5
    no_clip_boost = no_clip_controller.compute_steering(
        current_heading=0.0,
        reference_point=ref,
        current_speed=8.0,
        dt=0.033,
        return_metadata=True,
    )
    assert with_clip_boost['dynamic_curve_authority_active'] is True
    assert float(with_clip_boost['dynamic_curve_hard_clip_limit_effective']) > 0.2
    assert float(with_clip_boost['dynamic_curve_hard_clip_boost']) > 0.0
    assert float(with_clip_boost['dynamic_curve_hard_clip_limit_effective']) > float(
        no_clip_boost['dynamic_curve_hard_clip_limit_effective']
    )


def test_dynamic_curve_entry_governor_applies_stale_floor_and_scales_jerk_limit():
    """Entry governor should keep minimum authority under stale input and scale jerk with rate."""
    base = dict(
        kp=1.0,
        kd=0.2,
        curve_entry_schedule_enabled=True,
        curve_entry_schedule_frames=12,
        curve_entry_schedule_min_rate=0.22,
        curve_entry_schedule_min_jerk=0.14,
        dynamic_curve_authority_enabled=True,
        dynamic_curve_entry_governor_enabled=True,
        dynamic_curve_entry_governor_gain=1.2,
        dynamic_curve_entry_governor_max_scale=1.8,
        dynamic_curve_entry_governor_stale_floor_scale=1.2,
        curve_upcoming_enter_threshold=0.005,
        curve_upcoming_exit_threshold=0.004,
        curve_upcoming_on_frames=1,
        curve_upcoming_off_frames=1,
    )
    ref = {'x': 1.6, 'y': 0.0, 'heading': 0.25, 'velocity': 8.0, 'curvature': 0.02}

    with_governor = LateralController(**base).compute_steering(
        current_heading=0.0,
        reference_point=ref,
        current_speed=8.0,
        dt=0.033,
        using_stale_perception=True,
        return_metadata=True,
    )
    no_governor = LateralController(
        **{**base, 'dynamic_curve_entry_governor_enabled': False}
    ).compute_steering(
        current_heading=0.0,
        reference_point=ref,
        current_speed=8.0,
        dt=0.033,
        using_stale_perception=True,
        return_metadata=True,
    )

    assert with_governor['dynamic_curve_entry_governor_active'] is True
    assert float(with_governor['dynamic_curve_entry_governor_scale']) >= 1.2
    assert float(with_governor['dynamic_curve_entry_governor_jerk_scale']) >= 1.2
    assert float(with_governor['steering_rate_limit_effective']) > float(
        no_governor['steering_rate_limit_effective']
    )


def test_dynamic_curve_entry_governor_exclusive_mode_disables_entry_assist_stacking():
    """Exclusive governor mode should prevent entry-assist stacking while governor is active."""
    controller = LateralController(
        kp=1.0,
        kd=0.2,
        curve_entry_schedule_enabled=True,
        curve_entry_schedule_frames=10,
        curve_entry_schedule_min_rate=0.22,
        curve_entry_schedule_min_jerk=0.14,
        curve_entry_assist_enabled=True,
        curve_entry_assist_error_min=0.05,
        curve_entry_assist_heading_error_min=0.01,
        curve_entry_assist_curvature_min=0.001,
        curve_entry_assist_rate_boost=1.4,
        curve_entry_assist_jerk_boost=1.4,
        dynamic_curve_authority_enabled=True,
        dynamic_curve_entry_governor_enabled=True,
        dynamic_curve_entry_governor_exclusive_mode=True,
        curve_upcoming_enter_threshold=0.005,
        curve_upcoming_exit_threshold=0.004,
        curve_upcoming_on_frames=1,
        curve_upcoming_off_frames=1,
    )
    m = controller.compute_steering(
        current_heading=0.0,
        reference_point={'x': 1.4, 'y': 0.0, 'heading': 0.22, 'velocity': 8.0, 'curvature': 0.02},
        current_speed=8.0,
        dt=0.033,
        using_stale_perception=True,
        return_metadata=True,
    )
    assert m['dynamic_curve_entry_governor_active'] is True
    assert m['curve_entry_assist_active'] is False


def test_dynamic_curve_entry_governor_anticipatory_phase_starts_before_curve_at_car():
    """Anticipatory mode should raise authority while curve is upcoming (before at_car)."""
    base = dict(
        kp=1.0,
        kd=0.2,
        curve_phase_use_distance_track=True,
        curve_phase_track_name='s_loop',
        dynamic_curve_authority_enabled=True,
        dynamic_curve_entry_governor_enabled=True,
        dynamic_curve_entry_governor_exclusive_mode=True,
        dynamic_curve_entry_governor_gain=1.2,
        dynamic_curve_entry_governor_upcoming_phase_weight=0.6,
        dynamic_curve_entry_governor_anticipatory_enabled=True,
        dynamic_curve_authority_precurve_enabled=True,
        dynamic_curve_authority_precurve_scale=0.9,
    )
    ref = {'x': 1.4, 'y': 4.7, 'heading': 0.22, 'velocity': 8.0, 'curvature': 0.02}
    road_t_precurve = 20.5 / 452.0353755551324

    anticipatory = LateralController(**base).compute_steering(
        current_heading=0.0,
        reference_point=ref,
        current_speed=8.0,
        road_center_reference_t=road_t_precurve,
        dt=0.033,
        return_metadata=True,
    )
    baseline = LateralController(
        **{
            **base,
            'dynamic_curve_entry_governor_anticipatory_enabled': False,
            'dynamic_curve_authority_precurve_enabled': False,
        }
    ).compute_steering(
        current_heading=0.0,
        reference_point=ref,
        current_speed=8.0,
        road_center_reference_t=road_t_precurve,
        dt=0.033,
        return_metadata=True,
    )

    assert anticipatory['curve_upcoming'] is True
    assert anticipatory['curve_at_car'] is False
    assert anticipatory['dynamic_curve_entry_governor_anticipatory_active'] is True
    assert anticipatory['dynamic_curve_authority_active'] is True
    assert anticipatory['dynamic_curve_entry_governor_active'] is True
    assert float(anticipatory['steering_rate_limit_effective']) > float(
        baseline['steering_rate_limit_effective']
    )


def test_dynamic_curve_single_owner_mode_disables_aux_entry_layers_and_applies_floors():
    """Single-owner mode should disable schedule/commit/assist and enforce rate/jerk floors."""
    controller = LateralController(
        kp=1.0,
        kd=0.2,
        curve_phase_use_distance_track=True,
        curve_phase_track_name='s_loop',
        curve_entry_schedule_enabled=True,
        curve_entry_schedule_frames=12,
        curve_commit_mode_enabled=True,
        curve_entry_assist_enabled=True,
        curve_entry_assist_error_min=0.05,
        curve_entry_assist_heading_error_min=0.01,
        curve_entry_assist_curvature_min=0.001,
        dynamic_curve_authority_enabled=True,
        dynamic_curve_entry_governor_enabled=True,
        dynamic_curve_single_owner_mode=True,
        dynamic_curve_single_owner_min_rate=0.22,
        dynamic_curve_single_owner_min_jerk=0.6,
    )
    ref = {'x': 1.2, 'y': 4.7, 'heading': 0.2, 'velocity': 8.0, 'curvature': 0.02}
    m = controller.compute_steering(
        current_heading=0.0,
        reference_point=ref,
        current_speed=8.0,
        road_center_reference_t=20.5 / 452.0353755551324,
        dt=0.033,
        return_metadata=True,
    )
    assert m['curve_upcoming'] is True
    assert m['curve_at_car'] is False
    assert m['dynamic_curve_single_owner_active'] is True
    assert m['curve_entry_schedule_active'] is False
    assert m['curve_commit_mode_active'] is False
    assert m['curve_entry_assist_active'] is False
    assert float(m['steering_rate_limit_effective']) >= 0.22
    assert float(m['steering_jerk_limit_effective']) >= 0.6


def test_turn_feasibility_governor_reports_margin_and_speed_limit():
    """Feasibility telemetry should report comfort-envelope headroom consistently."""
    controller = LateralController(
        kp=1.0,
        kd=0.2,
        dynamic_curve_comfort_lat_accel_comfort_max_g=0.18,
        dynamic_curve_comfort_lat_accel_peak_max_g=0.25,
        turn_feasibility_governor_enabled=True,
        turn_feasibility_curvature_min=0.002,
        turn_feasibility_guardband_g=0.015,
        turn_feasibility_use_peak_bound=True,
    )
    curvature = 0.02
    speed = 8.0
    m = controller.compute_steering(
        current_heading=0.0,
        reference_point={'x': 1.2, 'y': 0.0, 'heading': 0.2, 'velocity': speed, 'curvature': curvature},
        current_speed=speed,
        dt=0.033,
        return_metadata=True,
    )
    expected_required = (speed * speed * curvature) / 9.81
    expected_selected = 0.25
    expected_effective = expected_selected - 0.015
    expected_speed_limit = np.sqrt((expected_effective * 9.81) / curvature)
    assert m['turn_feasibility_active'] is True
    assert np.isclose(float(m['turn_feasibility_required_lat_accel_g']), expected_required, atol=1e-6)
    assert np.isclose(float(m['turn_feasibility_selected_limit_g']), expected_selected, atol=1e-6)
    assert np.isclose(float(m['turn_feasibility_margin_g']), expected_effective - expected_required, atol=1e-6)
    assert np.isclose(float(m['turn_feasibility_speed_limit_mps']), expected_speed_limit, atol=1e-6)


def test_turn_feasibility_governor_is_telemetry_only_no_actuation_change():
    """Turning feasibility telemetry must not alter steering behavior in Phase 1."""
    base = dict(
        kp=1.0,
        kd=0.2,
        dynamic_curve_authority_enabled=True,
        dynamic_curve_entry_governor_enabled=True,
        turn_feasibility_curvature_min=0.002,
        turn_feasibility_guardband_g=0.015,
        turn_feasibility_use_peak_bound=True,
    )
    ref = {'x': 1.2, 'y': 0.0, 'heading': 0.2, 'velocity': 8.0, 'curvature': 0.02}
    with_telemetry = LateralController(**{**base, 'turn_feasibility_governor_enabled': True}).compute_steering(
        current_heading=0.0,
        reference_point=ref,
        current_speed=8.0,
        dt=0.033,
        return_metadata=True,
    )
    without_telemetry = LateralController(
        **{**base, 'turn_feasibility_governor_enabled': False}
    ).compute_steering(
        current_heading=0.0,
        reference_point=ref,
        current_speed=8.0,
        dt=0.033,
        return_metadata=True,
    )
    assert np.isclose(
        float(with_telemetry['steering_rate_limit_effective']),
        float(without_telemetry['steering_rate_limit_effective']),
        atol=1e-9,
    )
    assert np.isclose(
        float(with_telemetry['steering']),
        float(without_telemetry['steering']),
        atol=1e-9,
    )


def test_curve_unwind_policy_scales_limits_and_decays_integral():
    """Unwind policy should reduce post-curve authority and decay integral deterministically."""
    base = dict(
        kp=1.0,
        ki=0.2,
        kd=0.2,
        curve_upcoming_enter_threshold=0.5,
        curve_upcoming_exit_threshold=0.4,
        curve_unwind_frames=6,
        curve_unwind_rate_scale_start=0.9,
        curve_unwind_rate_scale_end=0.6,
        curve_unwind_jerk_scale_start=0.9,
        curve_unwind_jerk_scale_end=0.6,
        curve_unwind_integral_decay=0.5,
        steering_jerk_limit=0.2,
    )
    ref = {'x': 0.8, 'y': 0.0, 'heading': 0.0, 'velocity': 8.0, 'curvature': 0.0}

    with_unwind = LateralController(
        **{**base, 'curve_unwind_policy_enabled': True}
    )
    without_unwind = LateralController(
        **{**base, 'curve_unwind_policy_enabled': False}
    )
    for c in (with_unwind, without_unwind):
        c._prev_curve_at_car = True
        c.pid.integral = 0.2

    m_with = with_unwind.compute_steering(
        current_heading=0.0,
        reference_point=ref,
        current_speed=8.0,
        dt=0.033,
        return_metadata=True,
    )
    m_without = without_unwind.compute_steering(
        current_heading=0.0,
        reference_point=ref,
        current_speed=8.0,
        dt=0.033,
        return_metadata=True,
    )

    assert m_with['curve_unwind_active'] is True
    assert int(m_with['curve_unwind_frames_remaining']) >= 0
    assert float(m_with['curve_unwind_rate_scale']) < 1.0
    assert float(m_with['curve_unwind_jerk_scale']) < 1.0
    assert np.isclose(float(m_with['curve_unwind_integral_decay_applied']), 0.5, atol=1e-9)
    assert float(m_with['steering_rate_limit_effective']) < float(
        m_without['steering_rate_limit_effective']
    )
    assert abs(with_unwind.pid.integral) < abs(without_unwind.pid.integral)


def test_distance_track_curve_phase_arms_upcoming_before_curve_start():
    """Distance-track phase should arm upcoming before start and at_car at curve start."""
    controller = LateralController(
        kp=1.0,
        kd=0.2,
        curve_phase_use_distance_track=True,
        curve_phase_track_name='s_loop',
        curve_entry_schedule_enabled=True,
        curve_entry_schedule_frames=8,
    )
    base_ref = {'x': 0.0, 'y': 4.7, 'heading': 0.0, 'velocity': 8.0, 'curvature': 0.0}

    # ~20.5m along track, first curve at 25m and lookahead=4.7: upcoming should be true.
    m0 = controller.compute_steering(
        current_heading=0.0,
        reference_point=base_ref,
        current_speed=8.0,
        road_center_reference_t=20.5 / 452.0353755551324,
        return_metadata=True,
    )
    assert m0['curve_upcoming'] is True
    assert m0['curve_at_car'] is False
    assert m0['curve_entry_schedule_triggered'] is False

    # Slightly into curve start -> at_car true and schedule triggers.
    m1 = controller.compute_steering(
        current_heading=0.0,
        reference_point=base_ref,
        current_speed=8.0,
        road_center_reference_t=25.2 / 452.0353755551324,
        return_metadata=True,
    )
    assert m1['curve_upcoming'] is True
    assert m1['curve_at_car'] is True
    assert m1['curve_entry_schedule_triggered'] is True


def test_lateral_controller_road_straight_hysteresis_and_invalid_hold():
    """Road-straight should use enter/exit hysteresis and hold state on invalid gaps."""
    controller = LateralController(
        kp=1.0,
        kd=0.2,
        straight_curvature_threshold=0.01,
        road_curve_enter_threshold=0.012,
        road_curve_exit_threshold=0.008,
        road_straight_hold_invalid_frames=2,
    )
    base_ref = {'x': 0.0, 'y': 10.0, 'heading': 0.0, 'velocity': 10.0}

    m0 = controller.compute_steering(
        current_heading=0.0,
        reference_point={**base_ref, 'curvature': 0.007},
        return_metadata=True,
    )
    assert m0['is_road_straight'] is True

    # Between exit and enter thresholds: hold previous straight state.
    m1 = controller.compute_steering(
        current_heading=0.0,
        reference_point={**base_ref, 'curvature': 0.010},
        return_metadata=True,
    )
    assert m1['is_road_straight'] is True

    # Exceed enter threshold: transition to curve.
    m2 = controller.compute_steering(
        current_heading=0.0,
        reference_point={**base_ref, 'curvature': 0.013},
        return_metadata=True,
    )
    assert m2['is_road_straight'] is False

    # Back into hysteresis band: hold curve state.
    m3 = controller.compute_steering(
        current_heading=0.0,
        reference_point={**base_ref, 'curvature': 0.010},
        return_metadata=True,
    )
    assert m3['is_road_straight'] is False

    # Below exit threshold: transition back to straight.
    m4 = controller.compute_steering(
        current_heading=0.0,
        reference_point={**base_ref, 'curvature': 0.007},
        return_metadata=True,
    )
    assert m4['is_road_straight'] is True

    # Missing curvature: hold prior state for configured invalid-frame hold.
    m5 = controller.compute_steering(
        current_heading=0.0,
        reference_point={**base_ref},
        return_metadata=True,
    )
    m6 = controller.compute_steering(
        current_heading=0.0,
        reference_point={**base_ref},
        return_metadata=True,
    )
    m7 = controller.compute_steering(
        current_heading=0.0,
        reference_point={**base_ref},
        return_metadata=True,
    )
    assert m5['is_road_straight'] is True
    assert m6['is_road_straight'] is True
    assert m7['is_road_straight'] is None


def test_lateral_controller_sign_flip_override_on_straight():
    """Sign-flip override should activate on straight when error reverses."""
    controller = LateralController(
        kp=0.8,
        kd=0.0,
        deadband=0.0,
        max_steering=1.0,
        steering_smoothing_alpha=1.0,
    )
    ref = {'x': 1.0, 'y': 10.0, 'heading': 0.0, 'velocity': 10.0, 'curvature': 0.0}
    controller.compute_steering(current_heading=0.0, reference_point=ref)

    ref['x'] = -1.0
    meta = controller.compute_steering(current_heading=0.0, reference_point=ref, return_metadata=True)
    assert meta['straight_sign_flip_override_active'] is True


def test_lateral_controller_steering_jerk_limit():
    """Steering jerk limiter should bound change in steering rate."""
    controller = LateralController(
        kp=1.0,
        kd=0.0,
        deadband=0.0,
        max_steering=1.0,
        steering_smoothing_alpha=1.0,
        steering_jerk_limit=0.5,
    )
    reference_point = {
        'x': 0.0,
        'y': 10.0,
        'heading': 0.0,
        'velocity': 10.0
    }
    controller.compute_steering(current_heading=0.0, reference_point=reference_point)
    last_rate = getattr(controller, 'last_steering_rate', 0.0)

    reference_point['x'] = 1.0
    controller.compute_steering(current_heading=0.0, reference_point=reference_point)
    new_rate = controller.last_steering_rate

    dt = 0.033
    max_rate_delta = 0.5 * dt
    assert abs(new_rate - last_rate) <= max_rate_delta + 1e-6


def test_lateral_controller_curvature_transition_smoothing():
    """Curvature smoothing should damp abrupt curvature changes."""
    controller = LateralController(
        kp=0.0,
        kd=0.0,
        deadband=0.0,
        max_steering=1.0,
        steering_smoothing_alpha=1.0,
        curvature_smoothing_alpha=0.7,
        curvature_transition_threshold=0.0,
        curvature_transition_alpha=0.2,
    )
    reference_point = {
        'x': 0.0,
        'y': 10.0,
        'heading': 0.0,
        'velocity': 10.0,
        'curvature': 0.0,
    }
    controller.compute_steering(
        current_heading=0.0,
        reference_point=reference_point,
        return_metadata=True,
    )

    reference_point['curvature'] = 0.2
    meta = controller.compute_steering(
        current_heading=0.0,
        reference_point=reference_point,
        return_metadata=True,
    )
    assert meta['curve_feedforward_curvature_used'] < meta['path_curvature_raw']


def test_lateral_controller_steering_direction():
    """
    CRITICAL TEST: Verify steering direction is correct.
    
    This test prevents the steering direction inversion bug.
    The car should steer TOWARD the reference, not away from it.
    
    Lesson Learned: We previously added/removed/added negation based on
    theoretical reasoning. This test ensures we catch direction issues
    empirically. See DEVELOPMENT_GUIDELINES.md for details.
    
    Note: This test uses lower kp to avoid saturation masking the direction.
    """
    # Use lower kp to avoid saturation at max_steering limit
    controller = LateralController(kp=0.5, kd=0.1, deadband=0.0, max_steering=0.5)
    
    # Test case 1: Car is RIGHT of reference
    # After coordinate system fix: lateral_error = ref_x (directly)
    # If ref_x = -1.0 (reference is 1.0m LEFT of vehicle center)
    # Then lateral_error = -1.0 (negative)
    # Car is RIGHT of ref → should steer LEFT (negative steering)
    # Standard PID: negative error → negative output → negative steering ✅
    reference_point_right = {
        'x': -1.0,  # Reference 1.0m LEFT of vehicle center (vehicle frame)
        'y': 10.0,
        'heading': 0.0,
        'velocity': 10.0
    }
    
    steering_right = controller.compute_steering(
        current_heading=0.0,
        reference_point=reference_point_right
    )
    
    # Car is RIGHT of ref (ref_x = -1.0) → lateral_error = -1.0 → should steer LEFT (negative)
    # Standard PID: negative error → negative output → negative steering ✅
    assert steering_right < 0, (
        f"Car RIGHT of ref (ref_x=-1.0) should steer LEFT (negative), got {steering_right:.3f}. "
        f"This indicates steering direction is inverted!"
    )
    
    # Test case 2: Car is LEFT of reference
    # After coordinate system fix: lateral_error = ref_x (directly)
    # If ref_x = 1.0 (reference is 1.0m RIGHT of vehicle center)
    # Then lateral_error = 1.0 (positive)
    # Car is LEFT of ref → should steer RIGHT (positive steering)
    # Standard PID: positive error → positive output → positive steering ✅
    # NOTE: Create a fresh controller to avoid rate limiting from previous test case
    controller_left = LateralController(kp=0.5, kd=0.1, deadband=0.0, max_steering=0.5)
    reference_point_left = {
        'x': 1.0,  # Reference 1.0m RIGHT of vehicle center (vehicle frame)
        'y': 10.0,
        'heading': 0.0,
        'velocity': 10.0
    }
    
    steering_left = controller_left.compute_steering(
        current_heading=0.0,
        reference_point=reference_point_left
    )
    
    # Car is LEFT of ref (ref_x = 1.0) → lateral_error = 1.0 → should steer RIGHT (positive)
    # Standard PID: positive error → positive output → positive steering ✅
    assert steering_left > 0, (
        f"Car LEFT of ref (ref_x=1.0) should steer RIGHT (positive), got {steering_left:.3f}. "
        f"This indicates steering direction is inverted!"
    )
    
    # Test case 3: Car is at reference
    # After coordinate system fix: lateral_error = ref_x (directly)
    # If ref_x = 0.0 (reference is at vehicle center)
    # Then lateral_error = 0.0 (zero)
    # Car at ref → should have minimal steering
    # NOTE: Create a fresh controller to avoid PID state from previous test cases
    controller_center = LateralController(kp=0.5, kd=0.1, deadband=0.0, max_steering=0.5)
    reference_point_center = {
        'x': 0.0,  # Reference at vehicle center (vehicle frame)
        'y': 10.0,
        'heading': 0.0,
        'velocity': 10.0
    }
    
    steering_center = controller_center.compute_steering(
        current_heading=0.0,
        reference_point=reference_point_center
    )
    
    # Car at ref (ref_x = 0.0) → lateral_error = 0.0 → should have minimal steering
    assert abs(steering_center) < 0.1, (
        f"Car at ref (ref_x=0.0) should have minimal steering, got {steering_center:.3f}"
    )
    
    print(f"✅ Steering direction test passed:")
    print(f"   Car RIGHT of ref: steering = {steering_right:.3f} (should be < 0) ✓")
    print(f"   Car LEFT of ref: steering = {steering_left:.3f} (should be > 0) ✓")
    print(f"   Car at center: steering = {steering_center:.3f} (should be ~0) ✓")


def test_longitudinal_controller():
    """Test longitudinal controller."""
    controller = LongitudinalController(target_speed=10.0)
    
    throttle, brake = controller.compute_control(current_speed=5.0)
    
    assert 0.0 <= throttle <= 1.0
    assert 0.0 <= brake <= 1.0


def test_longitudinal_rate_limit():
    """Test longitudinal throttle/brake rate limiting."""
    controller = LongitudinalController(
        target_speed=10.0,
        throttle_rate_limit=0.05,
        brake_rate_limit=0.10
    )

    throttle1, brake1 = controller.compute_control(current_speed=0.0, reference_velocity=10.0)
    throttle2, brake2 = controller.compute_control(current_speed=0.0, reference_velocity=10.0)

    assert throttle2 - throttle1 <= 0.05 + 1e-6
    assert brake2 - brake1 <= 0.10 + 1e-6


def test_longitudinal_accel_feedforward():
    """Feedforward accel should influence throttle/brake when PID gains are zero."""
    controller = LongitudinalController(
        kp=0.0,
        ki=0.0,
        kd=0.0,
        target_speed=10.0,
        max_accel=2.0,
        max_decel=2.0,
        max_jerk=100.0,
        max_jerk_min=100.0,
        max_jerk_max=100.0,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        throttle_smoothing_alpha=1.0,
        speed_smoothing_alpha=0.0,
        speed_for_jerk_alpha=0.0,
        speed_error_to_accel_gain=0.0,
        accel_mode_threshold=0.0,
        decel_mode_threshold=0.0,
        speed_error_accel_threshold=0.0,
        speed_error_brake_threshold=0.0,
        mode_switch_min_time=0.0,
        coast_throttle_max=1.0,
        accel_tracking_enabled=False
    )

    throttle, brake = controller.compute_control(
        current_speed=5.0,
        reference_velocity=10.0,
        reference_accel=1.0,
        dt=0.1
    )
    assert throttle == pytest.approx(0.5, rel=1e-3)
    assert brake == pytest.approx(0.0, rel=1e-6)


def test_longitudinal_accel_tracking_error_scale_reduces_throttle():
    base = LongitudinalController(
        max_accel=2.0,
        max_decel=2.0,
        speed_error_deadband=0.0,
        speed_error_gain_under=0.1,
        speed_error_gain_over=0.1,
        min_throttle_when_accel=0.0,
        min_throttle_hold=0.0,
        min_throttle_hold_speed=0.0,
        accel_tracking_enabled=True,
        accel_tracking_error_scale=1.0,
        accel_target_smoothing_alpha=0.0,
        continuous_accel_control=True,
        startup_speed_threshold=0.0,
        low_speed_accel_limit=0.0,
    )
    scaled = LongitudinalController(
        max_accel=2.0,
        max_decel=2.0,
        speed_error_deadband=0.0,
        speed_error_gain_under=0.1,
        speed_error_gain_over=0.1,
        min_throttle_when_accel=0.0,
        min_throttle_hold=0.0,
        min_throttle_hold_speed=0.0,
        accel_tracking_enabled=True,
        accel_tracking_error_scale=0.2,
        accel_target_smoothing_alpha=0.0,
        continuous_accel_control=True,
        startup_speed_threshold=0.0,
        low_speed_accel_limit=0.0,
    )

    throttle_base, _ = base.compute_control(
        current_speed=0.0,
        reference_velocity=5.0,
        dt=0.1,
        reference_accel=0.0,
    )
    throttle_scaled, _ = scaled.compute_control(
        current_speed=0.0,
        reference_velocity=5.0,
        dt=0.1,
        reference_accel=0.0,
    )

    assert throttle_scaled < throttle_base

def test_longitudinal_continuous_accel_control():
    """Continuous accel control should apply brake when overspeeding."""
    controller = LongitudinalController(
        kp=0.0,
        ki=0.0,
        kd=0.0,
        target_speed=10.0,
        max_accel=2.0,
        max_decel=2.0,
        max_jerk=100.0,
        max_jerk_min=100.0,
        max_jerk_max=100.0,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        throttle_smoothing_alpha=1.0,
        speed_smoothing_alpha=0.0,
        speed_for_jerk_alpha=0.0,
        min_throttle_when_accel=0.0,
        min_throttle_hold=0.0,
        min_throttle_hold_speed=0.0,
        speed_error_to_accel_gain=1.0,
        accel_mode_threshold=0.0,
        decel_mode_threshold=0.0,
        speed_error_accel_threshold=0.0,
        speed_error_brake_threshold=0.0,
        mode_switch_min_time=0.0,
        coast_throttle_max=0.0,
        accel_tracking_enabled=False,
        continuous_accel_control=True,
        continuous_accel_deadband=0.0,
    )

    throttle, brake = controller.compute_control(
        current_speed=10.0,
        reference_velocity=5.0,
        reference_accel=0.0,
        dt=0.1,
    )

    assert throttle == pytest.approx(0.0, rel=1e-6)
    assert brake >= 0.4


def test_longitudinal_startup_ramp_caps_accel():
    """Startup ramp should cap accel during initial low-speed phase."""
    controller = LongitudinalController(
        kp=0.0,
        ki=0.0,
        kd=0.0,
        target_speed=10.0,
        max_accel=2.0,
        max_decel=2.0,
        max_jerk=100.0,
        max_jerk_min=100.0,
        max_jerk_max=100.0,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        throttle_smoothing_alpha=1.0,
        speed_smoothing_alpha=0.0,
        speed_for_jerk_alpha=0.0,
        speed_error_to_accel_gain=1.0,
        accel_mode_threshold=0.0,
        decel_mode_threshold=0.0,
        speed_error_accel_threshold=0.0,
        speed_error_brake_threshold=0.0,
        mode_switch_min_time=0.0,
        coast_throttle_max=0.0,
        accel_tracking_enabled=False,
        continuous_accel_control=True,
        continuous_accel_deadband=0.0,
        startup_ramp_seconds=4.0,
        startup_accel_limit=1.0,
        startup_speed_threshold=2.0,
    )

    # First step: ramp is 0.25, accel cap is 0.25 m/s^2 => throttle <= 0.125
    throttle, brake = controller.compute_control(
        current_speed=0.5,
        reference_velocity=10.0,
        reference_accel=2.0,
        dt=1.0,
    )

    assert brake == pytest.approx(0.0, rel=1e-6)
    assert throttle <= 0.125 + 1e-3


def test_longitudinal_startup_throttle_cap_applies():
    """Startup throttle cap should limit throttle during startup window."""
    controller = LongitudinalController(
        kp=0.0,
        ki=0.0,
        kd=0.0,
        target_speed=10.0,
        max_accel=2.0,
        max_decel=2.0,
        max_jerk=100.0,
        max_jerk_min=100.0,
        max_jerk_max=100.0,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        throttle_smoothing_alpha=1.0,
        speed_smoothing_alpha=0.0,
        speed_for_jerk_alpha=0.0,
        min_throttle_when_accel=0.0,
        min_throttle_hold=0.0,
        min_throttle_hold_speed=0.0,
        speed_error_to_accel_gain=1.0,
        accel_mode_threshold=0.0,
        decel_mode_threshold=0.0,
        speed_error_accel_threshold=0.0,
        speed_error_brake_threshold=0.0,
        mode_switch_min_time=0.0,
        coast_throttle_max=0.0,
        accel_tracking_enabled=False,
        continuous_accel_control=True,
        continuous_accel_deadband=0.0,
        startup_ramp_seconds=1.0,
        startup_accel_limit=10.0,
        startup_speed_threshold=2.0,
        startup_throttle_cap=0.2,
        startup_disable_accel_feedforward=True,
        low_speed_accel_limit=10.0,
        low_speed_speed_threshold=2.0,
    )

    throttle, brake = controller.compute_control(
        current_speed=0.5,
        reference_velocity=10.0,
        reference_accel=2.0,
        dt=0.1,
    )

    assert brake == pytest.approx(0.0, rel=1e-6)
    assert throttle <= 0.2 + 1e-3


def test_longitudinal_low_speed_accel_limit_applies():
    """Low-speed accel limit should cap throttle at low speeds."""
    controller = LongitudinalController(
        kp=0.0,
        ki=0.0,
        kd=0.0,
        target_speed=10.0,
        max_accel=2.0,
        max_decel=2.0,
        max_jerk=100.0,
        max_jerk_min=100.0,
        max_jerk_max=100.0,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        throttle_smoothing_alpha=1.0,
        speed_smoothing_alpha=0.0,
        speed_for_jerk_alpha=0.0,
        min_throttle_when_accel=0.0,
        min_throttle_hold=0.0,
        min_throttle_hold_speed=0.0,
        speed_error_to_accel_gain=1.0,
        accel_mode_threshold=0.0,
        decel_mode_threshold=0.0,
        speed_error_accel_threshold=0.0,
        speed_error_brake_threshold=0.0,
        mode_switch_min_time=0.0,
        coast_throttle_max=0.0,
        accel_tracking_enabled=False,
        continuous_accel_control=True,
        continuous_accel_deadband=0.0,
        startup_ramp_seconds=1.0,
        startup_accel_limit=10.0,
        startup_speed_threshold=2.0,
        startup_throttle_cap=1.0,
        startup_disable_accel_feedforward=True,
        low_speed_accel_limit=0.6,
        low_speed_speed_threshold=2.0,
    )

    throttle, brake = controller.compute_control(
        current_speed=0.5,
        reference_velocity=10.0,
        reference_accel=2.0,
        dt=0.1,
    )

    assert brake == pytest.approx(0.0, rel=1e-6)
    assert throttle <= 0.3 + 1e-3


def test_longitudinal_speed_error_deadband_zeroes_small_error():
    """Deadband should zero small speed error contributions."""
    controller = LongitudinalController(
        kp=0.0,
        ki=0.0,
        kd=0.0,
        target_speed=10.0,
        max_accel=2.0,
        max_decel=2.0,
        max_jerk=100.0,
        max_jerk_min=100.0,
        max_jerk_max=100.0,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        throttle_smoothing_alpha=1.0,
        speed_smoothing_alpha=0.0,
        speed_for_jerk_alpha=0.0,
        min_throttle_when_accel=0.0,
        min_throttle_hold=0.0,
        min_throttle_hold_speed=0.0,
        speed_error_to_accel_gain=0.0,
        speed_error_deadband=1.0,
        speed_error_gain_under=1.0,
        speed_error_gain_over=1.0,
        accel_mode_threshold=0.0,
        decel_mode_threshold=0.0,
        speed_error_accel_threshold=0.0,
        speed_error_brake_threshold=0.0,
        mode_switch_min_time=0.0,
        coast_throttle_max=0.0,
        accel_tracking_enabled=False,
        continuous_accel_control=True,
        continuous_accel_deadband=0.0,
    )

    throttle, brake = controller.compute_control(
        current_speed=9.6,
        reference_velocity=10.0,
        reference_accel=0.0,
        dt=0.1,
    )

    assert throttle == pytest.approx(0.0, rel=1e-6)
    assert brake == pytest.approx(0.0, rel=1e-6)


def test_longitudinal_overspeed_accel_zero_clamps_positive_accel():
    """Overspeed accel clamp should zero positive accel when above target."""
    controller = LongitudinalController(
        kp=0.0,
        ki=0.0,
        kd=0.0,
        target_speed=10.0,
        max_accel=2.0,
        max_decel=2.0,
        max_jerk=100.0,
        max_jerk_min=100.0,
        max_jerk_max=100.0,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        throttle_smoothing_alpha=1.0,
        speed_smoothing_alpha=0.0,
        speed_for_jerk_alpha=0.0,
        min_throttle_when_accel=0.0,
        min_throttle_hold=0.0,
        min_throttle_hold_speed=0.0,
        speed_error_to_accel_gain=1.0,
        speed_error_deadband=0.0,
        speed_error_gain_under=1.0,
        speed_error_gain_over=1.0,
        overspeed_accel_zero_threshold=0.2,
        accel_mode_threshold=0.0,
        decel_mode_threshold=0.0,
        speed_error_accel_threshold=0.0,
        speed_error_brake_threshold=0.0,
        mode_switch_min_time=0.0,
        coast_throttle_max=0.0,
        accel_tracking_enabled=False,
        continuous_accel_control=True,
        continuous_accel_deadband=0.0,
    )

    throttle, brake = controller.compute_control(
        current_speed=10.5,
        reference_velocity=10.0,
        reference_accel=0.0,
        dt=0.1,
    )

    assert throttle == pytest.approx(0.0, rel=1e-6)
    assert brake > 0.0


def test_longitudinal_continuous_accel_respects_jerk_limit():
    controller = LongitudinalController(
        max_accel=2.0,
        max_decel=2.0,
        max_jerk=0.5,
        max_jerk_min=0.5,
        max_jerk_max=0.5,
        speed_error_deadband=0.0,
        speed_error_gain_under=0.0,
        speed_error_gain_over=0.0,
        min_throttle_when_accel=0.0,
        min_throttle_hold=0.0,
        min_throttle_hold_speed=0.0,
        accel_tracking_enabled=True,
        accel_target_smoothing_alpha=0.0,
        continuous_accel_control=True,
        startup_speed_threshold=0.0,
        low_speed_accel_limit=0.0,
    )

    throttle1, brake1 = controller.compute_control(
        current_speed=0.0,
        reference_velocity=5.0,
        dt=0.1,
        reference_accel=0.0,
    )
    accel1 = (throttle1 * controller.max_accel) - (brake1 * controller.max_decel)
    throttle2, brake2 = controller.compute_control(
        current_speed=0.0,
        reference_velocity=5.0,
        dt=0.1,
        reference_accel=2.0,
    )
    accel2 = (throttle2 * controller.max_accel) - (brake2 * controller.max_decel)

    assert accel2 - accel1 <= controller.max_jerk * 0.1 + 1e-6


def test_longitudinal_continuous_accel_pid_tracks_reference_accel():
    def _make_controller(accel_pid_kp: float) -> LongitudinalController:
        return LongitudinalController(
            max_accel=3.0,
            max_decel=3.0,
            max_jerk=0.0,
            throttle_rate_limit=0.0,
            brake_rate_limit=0.0,
            speed_error_deadband=0.0,
            speed_error_gain_under=0.0,
            speed_error_gain_over=0.0,
            min_throttle_when_accel=0.0,
            min_throttle_hold=0.0,
            min_throttle_hold_speed=0.0,
            accel_tracking_enabled=True,
            accel_pid_kp=accel_pid_kp,
            accel_pid_ki=0.0,
            accel_pid_kd=0.0,
            accel_target_smoothing_alpha=0.0,
            continuous_accel_control=True,
            startup_speed_threshold=0.0,
            low_speed_accel_limit=0.0,
        )

    controller_pid = _make_controller(accel_pid_kp=1.0)
    controller_no = _make_controller(accel_pid_kp=0.0)

    controller_pid.compute_control(
        current_speed=0.0,
        reference_velocity=0.0,
        dt=0.1,
        reference_accel=0.0,
    )
    controller_no.compute_control(
        current_speed=0.0,
        reference_velocity=0.0,
        dt=0.1,
        reference_accel=0.0,
    )

    throttle_pid, _ = controller_pid.compute_control(
        current_speed=0.0,
        reference_velocity=0.0,
        dt=0.1,
        reference_accel=1.0,
    )
    throttle_no, _ = controller_no.compute_control(
        current_speed=0.0,
        reference_velocity=0.0,
        dt=0.1,
        reference_accel=1.0,
    )

    assert throttle_pid > throttle_no


def test_longitudinal_throttle_curve_gamma_shapes_output():
    controller_linear = LongitudinalController(
        max_accel=2.0,
        max_decel=2.0,
        max_jerk=0.0,
        throttle_rate_limit=0.0,
        brake_rate_limit=0.0,
        speed_error_deadband=0.0,
        speed_error_gain_under=0.0,
        speed_error_gain_over=0.0,
        min_throttle_when_accel=0.0,
        min_throttle_hold=0.0,
        min_throttle_hold_speed=0.0,
        accel_tracking_enabled=True,
        accel_pid_kp=0.0,
        accel_pid_ki=0.0,
        accel_pid_kd=0.0,
        accel_target_smoothing_alpha=0.0,
        continuous_accel_control=True,
        startup_speed_threshold=0.0,
        low_speed_accel_limit=0.0,
        throttle_curve_gamma=1.0,
    )
    controller_gamma = LongitudinalController(
        max_accel=2.0,
        max_decel=2.0,
        max_jerk=0.0,
        throttle_rate_limit=0.0,
        brake_rate_limit=0.0,
        speed_error_deadband=0.0,
        speed_error_gain_under=0.0,
        speed_error_gain_over=0.0,
        min_throttle_when_accel=0.0,
        min_throttle_hold=0.0,
        min_throttle_hold_speed=0.0,
        accel_tracking_enabled=True,
        accel_pid_kp=0.0,
        accel_pid_ki=0.0,
        accel_pid_kd=0.0,
        accel_target_smoothing_alpha=0.0,
        continuous_accel_control=True,
        startup_speed_threshold=0.0,
        low_speed_accel_limit=0.0,
        throttle_curve_gamma=1.5,
    )

    controller_linear.compute_control(
        current_speed=0.0,
        reference_velocity=0.0,
        dt=0.1,
        reference_accel=1.0,
    )
    controller_gamma.compute_control(
        current_speed=0.0,
        reference_velocity=0.0,
        dt=0.1,
        reference_accel=1.0,
    )
    throttle_linear, _ = controller_linear.compute_control(
        current_speed=0.0,
        reference_velocity=0.0,
        dt=0.1,
        reference_accel=1.0,
    )
    throttle_gamma, _ = controller_gamma.compute_control(
        current_speed=0.0,
        reference_velocity=0.0,
        dt=0.1,
        reference_accel=1.0,
    )

    assert throttle_gamma < throttle_linear


def test_longitudinal_exclusive_throttle_brake_prefers_throttle():
    controller = LongitudinalController(
        max_accel=2.0,
        max_decel=2.0,
        max_jerk=0.0,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        speed_error_deadband=0.0,
        speed_error_gain_under=0.0,
        speed_error_gain_over=0.0,
        min_throttle_when_accel=0.0,
        min_throttle_hold=0.0,
        min_throttle_hold_speed=0.0,
        accel_tracking_enabled=True,
        accel_pid_kp=0.0,
        accel_pid_ki=0.0,
        accel_pid_kd=0.0,
        accel_target_smoothing_alpha=0.0,
        continuous_accel_control=True,
        startup_speed_threshold=0.0,
        low_speed_accel_limit=0.0,
        low_speed_speed_threshold=0.0,
    )

    controller.compute_control(
        current_speed=1.0,
        reference_velocity=0.0,
        dt=0.1,
        reference_accel=-1.0,
    )
    throttle, brake = controller.compute_control(
        current_speed=1.0,
        reference_velocity=2.0,
        dt=0.1,
        reference_accel=1.0,
    )

    assert throttle > 0.0
    assert brake == pytest.approx(0.0)


def test_longitudinal_accel_cap():
    """Acceleration cap should trigger when measured accel exceeds limits."""
    controller = LongitudinalController(
        kp=0.0,
        ki=0.0,
        kd=0.0,
        target_speed=10.0,
        max_accel=0.5,
        max_decel=0.5,
        max_jerk=100.0,
        speed_for_jerk_alpha=0.0
    )

    controller.compute_control(
        current_speed=0.0,
        reference_velocity=10.0,
        reference_accel=1.0,
        dt=0.1
    )
    controller.compute_control(
        current_speed=1.0,
        reference_velocity=10.0,
        reference_accel=1.0,
        dt=0.1
    )

    assert controller.last_accel_capped is True


def test_longitudinal_mode_dwell_prevents_flip():
    """Mode dwell should prevent rapid accel/brake switching."""
    controller = LongitudinalController(
        target_speed=10.0,
        max_speed=20.0,
        overspeed_brake_max=0.0,
        speed_error_accel_threshold=0.1,
        speed_error_brake_threshold=-0.1,
        accel_mode_threshold=0.1,
        decel_mode_threshold=0.1,
        mode_switch_min_time=0.5,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        speed_smoothing_alpha=0.0,
        speed_for_jerk_alpha=0.0,
        speed_error_to_accel_gain=0.5,
        coast_throttle_max=0.2,
        coast_hold_seconds=0.5
    )

    throttle1, brake1 = controller.compute_control(
        current_speed=9.0,
        reference_velocity=10.0,
        dt=0.1
    )
    assert controller.longitudinal_mode == "accel"
    assert throttle1 >= 0.0
    assert brake1 == pytest.approx(0.0)

    throttle2, brake2 = controller.compute_control(
        current_speed=10.5,
        reference_velocity=10.0,
        dt=0.1
    )
    assert controller.longitudinal_mode == "accel"
    assert brake2 == pytest.approx(0.0)


def test_longitudinal_coast_no_brake_near_target():
    """Near-target speed should coast without braking."""
    controller = LongitudinalController(
        target_speed=10.0,
        speed_error_accel_threshold=0.2,
        speed_error_brake_threshold=-0.2,
        accel_mode_threshold=0.2,
        decel_mode_threshold=0.2,
        coast_throttle_kp=0.2,
        coast_throttle_max=0.1,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        speed_smoothing_alpha=0.0,
        speed_for_jerk_alpha=0.0,
        accel_tracking_enabled=False
    )


def test_longitudinal_accel_tracking_uses_reference_accel():
    """Accel tracking should respond to reference_accel even when speed error is zero."""
    controller = LongitudinalController(
        kp=0.0,
        ki=0.0,
        kd=0.0,
        target_speed=10.0,
        max_accel=2.0,
        max_decel=2.0,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        speed_smoothing_alpha=0.0,
        speed_for_jerk_alpha=0.0,
        accel_tracking_enabled=True,
        accel_pid_kp=1.0,
        accel_pid_ki=0.0,
        accel_pid_kd=0.0,
        accel_mode_threshold=0.0,
        speed_error_accel_threshold=0.0,
        mode_switch_min_time=0.0
    )

    throttle, brake = controller.compute_control(
        current_speed=10.0,
        reference_velocity=10.0,
        reference_accel=1.0,
        dt=0.1
    )
    assert throttle > 0.0
    assert brake == pytest.approx(0.0)


def test_longitudinal_straight_throttle_cap():
    """Straight throttle cap should limit throttle on low curvature."""
    controller = LongitudinalController(
        kp=0.0,
        ki=0.0,
        kd=0.0,
        target_speed=10.0,
        max_accel=2.0,
        max_decel=2.0,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        speed_smoothing_alpha=0.0,
        speed_for_jerk_alpha=0.0,
        accel_tracking_enabled=True,
        accel_pid_kp=1.0,
        accel_pid_ki=0.0,
        accel_pid_kd=0.0,
        accel_mode_threshold=0.0,
        speed_error_accel_threshold=0.0,
        mode_switch_min_time=0.0,
        straight_throttle_cap=0.3,
        straight_curvature_threshold=0.003
    )

    throttle, brake = controller.compute_control(
        current_speed=5.0,
        reference_velocity=10.0,
        reference_accel=1.5,
        current_curvature=0.0,
        dt=0.1
    )
    assert brake == pytest.approx(0.0)
    assert throttle <= 0.3 + 1e-6


def test_longitudinal_coast_hold_forces_coast():
    """Coast hold should keep controller in coast even if accel is requested."""
    controller = LongitudinalController(
        target_speed=10.0,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        speed_smoothing_alpha=0.0,
        speed_for_jerk_alpha=0.0,
        accel_tracking_enabled=True,
        accel_pid_kp=1.0,
        accel_pid_ki=0.0,
        accel_pid_kd=0.0,
        accel_mode_threshold=0.0,
        speed_error_accel_threshold=0.0,
        mode_switch_min_time=0.0,
        coast_hold_seconds=0.5
    )

    controller.longitudinal_mode = "coast"
    controller.coast_hold_remaining = 0.3
    throttle, brake = controller.compute_control(
        current_speed=9.0,
        reference_velocity=10.0,
        reference_accel=1.0,
        dt=0.1
    )
    assert controller.longitudinal_mode == "coast"
    assert brake == pytest.approx(0.0)


def test_longitudinal_min_throttle_hold_applies():
    """Min throttle hold should prevent drift when under target at low speed."""
    controller = LongitudinalController(
        target_speed=10.0,
        min_throttle_hold=0.2,
        min_throttle_hold_speed=4.0,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        speed_smoothing_alpha=0.0,
        speed_for_jerk_alpha=0.0,
        accel_tracking_enabled=True,
        accel_pid_kp=1.0,
        accel_pid_ki=0.0,
        accel_pid_kd=0.0,
        accel_mode_threshold=0.0,
        speed_error_accel_threshold=0.0,
        mode_switch_min_time=0.0
    )

    throttle, brake = controller.compute_control(
        current_speed=1.0,
        reference_velocity=10.0,
        reference_accel=1.0,
        dt=0.1
    )
    assert brake == pytest.approx(0.0)
    assert throttle >= 0.2

    throttle, brake = controller.compute_control(
        current_speed=9.95,
        reference_velocity=10.0,
        dt=0.1
    )
    assert brake == pytest.approx(0.0)
    assert 0.0 <= throttle <= 0.3


def test_vehicle_controller():
    """Test combined vehicle controller."""
    controller = VehicleController()
    
    current_state = {
        'heading': 0.0,
        'speed': 10.0,
        'position': np.array([0.0, 0.0])
    }
    
    reference_point = {
        'x': 0.0,
        'y': 10.0,
        'heading': 0.0,
        'velocity': 10.0
    }
    
    commands = controller.compute_control(current_state, reference_point)
    
    assert 'steering' in commands
    assert 'throttle' in commands
    assert 'brake' in commands
    assert -1.0 <= commands['steering'] <= 1.0
    assert 0.0 <= commands['throttle'] <= 1.0
    assert 0.0 <= commands['brake'] <= 1.0

