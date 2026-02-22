import pytest

from control.pid_controller import LateralController
from trajectory.utils import (
    compute_curve_anticipation_score,
    compute_curve_phase_scheduler,
    compute_reference_lookahead,
)


def test_reference_lookahead_dynamic_scaling() -> None:
    config = {
        "dynamic_reference_lookahead": True,
        "reference_lookahead_min": 4.0,
        "reference_lookahead_scale_min": 0.7,
        "reference_lookahead_speed_min": 4.0,
        "reference_lookahead_speed_max": 10.0,
        "reference_lookahead_curvature_min": 0.002,
        "reference_lookahead_curvature_max": 0.015,
    }

    lookahead = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.001,
        config=config,
    )

    assert lookahead == pytest.approx(6.3, rel=1e-3)


def test_reference_lookahead_no_scaling_when_disabled() -> None:
    config = {
        "dynamic_reference_lookahead": False,
    }

    lookahead = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.001,
        config=config,
    )

    assert lookahead == pytest.approx(9.0, rel=1e-6)


def test_reference_lookahead_tight_curve_scale() -> None:
    config = {
        "dynamic_reference_lookahead": True,
        "reference_lookahead_min": 3.0,
        "reference_lookahead_scale_min": 0.7,
        "reference_lookahead_speed_min": 8.0,
        "reference_lookahead_speed_max": 12.0,
        "reference_lookahead_curvature_min": 0.02,
        "reference_lookahead_curvature_max": 0.03,
        "reference_lookahead_tight_curvature_threshold": 0.02,
        "reference_lookahead_tight_scale": 0.8,
    }

    lookahead = compute_reference_lookahead(
        base_lookahead=10.0,
        current_speed=4.0,
        path_curvature=0.03,
        config=config,
    )

    assert lookahead == pytest.approx(8.0, rel=1e-3)


def test_reference_lookahead_speed_table_interpolates() -> None:
    config = {
        "dynamic_reference_lookahead": True,
        "reference_lookahead_min": 2.5,
        "reference_lookahead_speed_table": [
            {"speed_mps": 0.0, "lookahead_m": 7.0},
            {"speed_mps": 5.0, "lookahead_m": 10.0},
            {"speed_mps": 10.0, "lookahead_m": 16.0},
        ],
    }

    lookahead = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=7.5,
        path_curvature=0.001,
        config=config,
    )

    assert lookahead == pytest.approx(13.0, rel=1e-3)


def test_reference_lookahead_speed_table_uses_tight_curve_scale() -> None:
    config = {
        "dynamic_reference_lookahead": True,
        "reference_lookahead_min": 2.5,
        "reference_lookahead_speed_table": [
            {"speed_mps": 0.0, "lookahead_m": 7.0},
            {"speed_mps": 10.0, "lookahead_m": 16.0},
        ],
        "reference_lookahead_tight_curvature_threshold": 0.02,
        "reference_lookahead_tight_scale": 0.75,
    }

    lookahead = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.03,
        config=config,
    )

    assert lookahead == pytest.approx(12.0, rel=1e-3)


def test_reference_lookahead_speed_table_overrides_legacy_scaling() -> None:
    config = {
        "dynamic_reference_lookahead": True,
        "reference_lookahead_min": 2.5,
        "reference_lookahead_scale_min": 0.2,
        "reference_lookahead_speed_min": 0.0,
        "reference_lookahead_speed_max": 10.0,
        "reference_lookahead_curvature_min": 0.0,
        "reference_lookahead_curvature_max": 0.01,
        "reference_lookahead_speed_table": [
            {"speed_mps": 0.0, "lookahead_m": 7.0},
            {"speed_mps": 10.0, "lookahead_m": 16.0},
        ],
    }

    lookahead = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.0,
        config=config,
    )

    assert lookahead == pytest.approx(16.0, rel=1e-3)


def test_reference_lookahead_tight_curve_blend_band_smooths_transition() -> None:
    config = {
        "dynamic_reference_lookahead": True,
        "reference_lookahead_min": 2.5,
        "reference_lookahead_speed_table": [
            {"speed_mps": 10.0, "lookahead_m": 16.0},
        ],
        "reference_lookahead_tight_curvature_threshold": 0.02,
        "reference_lookahead_tight_scale": 0.5,
        "reference_lookahead_tight_blend_band": 0.01,
    }

    lookahead_below = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.014,
        config=config,
    )
    lookahead_mid = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.020,
        config=config,
    )
    lookahead_above = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.026,
        config=config,
    )

    assert lookahead_below == pytest.approx(16.0, rel=1e-3)
    assert lookahead_mid == pytest.approx(12.0, rel=1e-3)
    assert lookahead_above == pytest.approx(8.0, rel=1e-3)
    assert lookahead_below > lookahead_mid > lookahead_above


def test_reference_lookahead_dual_table_blends_by_curvature() -> None:
    config = {
        "dynamic_reference_lookahead": True,
        "reference_lookahead_min": 2.5,
        "reference_lookahead_speed_table_straight": [
            {"speed_mps": 10.0, "lookahead_m": 16.0},
        ],
        "reference_lookahead_speed_table_curve": [
            {"speed_mps": 10.0, "lookahead_m": 12.0},
        ],
        "reference_lookahead_curve_blend_curvature_min": 0.01,
        "reference_lookahead_curve_blend_curvature_max": 0.02,
    }

    lookahead_straight = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.005,
        config=config,
    )
    lookahead_mid = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.015,
        config=config,
    )
    lookahead_curve = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.025,
        config=config,
    )

    assert lookahead_straight == pytest.approx(16.0, rel=1e-3)
    assert lookahead_mid == pytest.approx(14.0, rel=1e-3)
    assert lookahead_curve == pytest.approx(12.0, rel=1e-3)
    assert lookahead_straight > lookahead_mid > lookahead_curve


def test_reference_lookahead_dual_table_missing_falls_back_to_single_table() -> None:
    config = {
        "dynamic_reference_lookahead": True,
        "reference_lookahead_min": 2.5,
        "reference_lookahead_speed_table": [
            {"speed_mps": 0.0, "lookahead_m": 7.0},
            {"speed_mps": 10.0, "lookahead_m": 16.0},
        ],
        "reference_lookahead_speed_table_straight": [
            {"speed_mps": 0.0, "lookahead_m": 6.0},
            {"speed_mps": 10.0, "lookahead_m": 14.0},
        ],
        # Curve table intentionally omitted to enforce fallback behavior.
    }

    lookahead = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.01,
        config=config,
    )

    assert lookahead == pytest.approx(16.0, rel=1e-3)


def test_reference_lookahead_dual_table_applies_tight_scale_after_blend() -> None:
    config = {
        "dynamic_reference_lookahead": True,
        "reference_lookahead_min": 2.5,
        "reference_lookahead_speed_table_straight": [
            {"speed_mps": 10.0, "lookahead_m": 16.0},
        ],
        "reference_lookahead_speed_table_curve": [
            {"speed_mps": 10.0, "lookahead_m": 12.0},
        ],
        "reference_lookahead_curve_blend_curvature_min": 0.01,
        "reference_lookahead_curve_blend_curvature_max": 0.02,
        "reference_lookahead_tight_curvature_threshold": 0.02,
        "reference_lookahead_tight_scale": 0.5,
    }

    lookahead = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.03,
        config=config,
    )

    # curve-table lookahead (12.0) then tight-scale (0.5) => 6.0
    assert lookahead == pytest.approx(6.0, rel=1e-3)


def test_reference_lookahead_entry_table_uses_preview_and_distance_signals() -> None:
    config = {
        "dynamic_reference_lookahead": True,
        "reference_lookahead_min": 2.5,
        "reference_lookahead_speed_table": [
            {"speed_mps": 10.0, "lookahead_m": 16.0},
        ],
        "reference_lookahead_speed_table_entry": [
            {"speed_mps": 10.0, "lookahead_m": 12.0},
        ],
        "reference_lookahead_entry_preview_curvature_min": 0.006,
        "reference_lookahead_entry_preview_curvature_max": 0.015,
        "reference_lookahead_entry_distance_start_m": 20.0,
        "reference_lookahead_entry_distance_end_m": 6.0,
    }

    lookahead_far = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.001,
        path_curvature_preview=0.020,
        distance_to_curve_start_m=25.0,
        config=config,
    )
    lookahead_mid = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.001,
        path_curvature_preview=0.020,
        distance_to_curve_start_m=13.0,
        config=config,
    )
    lookahead_near = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.001,
        path_curvature_preview=0.020,
        distance_to_curve_start_m=4.0,
        config=config,
    )

    assert lookahead_far == pytest.approx(16.0, rel=1e-3)
    assert lookahead_mid == pytest.approx(14.0, rel=1e-3)
    assert lookahead_near == pytest.approx(12.0, rel=1e-3)
    assert lookahead_far > lookahead_mid > lookahead_near


def test_reference_lookahead_entry_table_falls_back_when_preview_missing() -> None:
    config = {
        "dynamic_reference_lookahead": True,
        "reference_lookahead_min": 2.5,
        "reference_lookahead_speed_table": [
            {"speed_mps": 10.0, "lookahead_m": 16.0},
        ],
        "reference_lookahead_speed_table_entry": [
            {"speed_mps": 10.0, "lookahead_m": 12.0},
        ],
        "reference_lookahead_entry_preview_curvature_min": 0.006,
        "reference_lookahead_entry_preview_curvature_max": 0.015,
        "reference_lookahead_entry_distance_start_m": 20.0,
        "reference_lookahead_entry_distance_end_m": 6.0,
    }

    lookahead = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.001,
        path_curvature_preview=None,
        distance_to_curve_start_m=4.0,
        config=config,
    )
    assert lookahead == pytest.approx(16.0, rel=1e-3)


def test_reference_lookahead_entry_table_requires_distance_without_preview_only_fallback() -> None:
    config = {
        "dynamic_reference_lookahead": True,
        "reference_lookahead_min": 2.5,
        "reference_lookahead_speed_table": [
            {"speed_mps": 10.0, "lookahead_m": 16.0},
        ],
        "reference_lookahead_speed_table_entry": [
            {"speed_mps": 10.0, "lookahead_m": 12.0},
        ],
        "reference_lookahead_entry_preview_curvature_min": 0.006,
        "reference_lookahead_entry_preview_curvature_max": 0.015,
        "reference_lookahead_entry_distance_start_m": 20.0,
        "reference_lookahead_entry_distance_end_m": 6.0,
    }

    lookahead = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.001,
        path_curvature_preview=0.020,
        distance_to_curve_start_m=None,
        config=config,
    )
    assert lookahead == pytest.approx(16.0, rel=1e-3)


def test_reference_lookahead_entry_table_preview_only_fallback_uses_preview_without_distance() -> None:
    config = {
        "dynamic_reference_lookahead": True,
        "reference_lookahead_min": 2.5,
        "reference_lookahead_speed_table": [
            {"speed_mps": 10.0, "lookahead_m": 16.0},
        ],
        "reference_lookahead_speed_table_entry": [
            {"speed_mps": 10.0, "lookahead_m": 12.0},
        ],
        "reference_lookahead_entry_preview_curvature_min": 0.006,
        "reference_lookahead_entry_preview_curvature_max": 0.015,
        "reference_lookahead_entry_distance_start_m": 20.0,
        "reference_lookahead_entry_distance_end_m": 6.0,
        "reference_lookahead_entry_preview_only_fallback": True,
    }

    lookahead = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.001,
        path_curvature_preview=0.020,
        distance_to_curve_start_m=None,
        config=config,
    )
    assert lookahead == pytest.approx(12.0, rel=1e-3)


def test_reference_lookahead_entry_table_applies_tight_scale_then_min_clamp() -> None:
    config = {
        "dynamic_reference_lookahead": True,
        "reference_lookahead_min": 3.0,
        "reference_lookahead_speed_table": [
            {"speed_mps": 10.0, "lookahead_m": 16.0},
        ],
        "reference_lookahead_speed_table_entry": [
            {"speed_mps": 10.0, "lookahead_m": 4.0},
        ],
        "reference_lookahead_entry_preview_curvature_min": 0.006,
        "reference_lookahead_entry_preview_curvature_max": 0.015,
        "reference_lookahead_entry_distance_start_m": 20.0,
        "reference_lookahead_entry_distance_end_m": 6.0,
        "reference_lookahead_tight_curvature_threshold": 0.02,
        "reference_lookahead_tight_scale": 0.5,
    }

    lookahead = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.03,
        path_curvature_preview=0.02,
        distance_to_curve_start_m=4.0,
        config=config,
    )
    assert lookahead == pytest.approx(3.0, rel=1e-3)


def test_reference_lookahead_entry_table_ignores_anticipation_in_shadow_mode() -> None:
    config = {
        "dynamic_reference_lookahead": True,
        "reference_lookahead_min": 2.5,
        "reference_lookahead_speed_table": [
            {"speed_mps": 10.0, "lookahead_m": 16.0},
        ],
        "reference_lookahead_speed_table_entry": [
            {"speed_mps": 10.0, "lookahead_m": 12.0},
        ],
        "curve_anticipation_enabled": True,
        "curve_anticipation_shadow_only": True,
        "curve_anticipation_single_owner_mode": True,
        "curve_anticipation_entry_weight_on": 0.60,
        "curve_anticipation_entry_weight_full": 0.90,
    }

    lookahead = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.001,
        path_curvature_preview=0.001,
        distance_to_curve_start_m=30.0,
        anticipation_score=0.95,
        anticipation_active=True,
        config=config,
    )
    assert lookahead == pytest.approx(16.0, rel=1e-3)


def test_reference_lookahead_entry_table_single_owner_uses_anticipation() -> None:
    config = {
        "dynamic_reference_lookahead": True,
        "reference_lookahead_min": 2.5,
        "reference_lookahead_speed_table": [
            {"speed_mps": 10.0, "lookahead_m": 16.0},
        ],
        "reference_lookahead_speed_table_entry": [
            {"speed_mps": 10.0, "lookahead_m": 12.0},
        ],
        "curve_anticipation_enabled": True,
        "curve_anticipation_shadow_only": False,
        "curve_anticipation_single_owner_mode": True,
        "curve_anticipation_entry_weight_on": 0.60,
        "curve_anticipation_entry_weight_full": 0.90,
    }

    lookahead = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.001,
        path_curvature_preview=0.001,
        distance_to_curve_start_m=30.0,
        anticipation_score=0.95,
        anticipation_active=True,
        config=config,
    )
    assert lookahead == pytest.approx(12.0, rel=1e-3)


def test_reference_lookahead_entry_table_single_owner_falls_back_when_inactive() -> None:
    config = {
        "dynamic_reference_lookahead": True,
        "reference_lookahead_min": 2.5,
        "reference_lookahead_speed_table": [
            {"speed_mps": 10.0, "lookahead_m": 16.0},
        ],
        "reference_lookahead_speed_table_entry": [
            {"speed_mps": 10.0, "lookahead_m": 12.0},
        ],
        "reference_lookahead_entry_preview_curvature_min": 0.006,
        "reference_lookahead_entry_preview_curvature_max": 0.015,
        "reference_lookahead_entry_distance_start_m": 20.0,
        "reference_lookahead_entry_distance_end_m": 6.0,
        "curve_anticipation_enabled": True,
        "curve_anticipation_shadow_only": False,
        "curve_anticipation_single_owner_mode": True,
        "curve_anticipation_entry_weight_on": 0.60,
        "curve_anticipation_entry_weight_full": 0.90,
    }

    lookahead = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.001,
        path_curvature_preview=0.020,
        distance_to_curve_start_m=4.0,
        anticipation_score=0.10,
        anticipation_active=False,
        config=config,
    )
    assert lookahead == pytest.approx(12.0, rel=1e-3)


def test_reference_lookahead_entry_table_phase_active_uses_curve_phase_weight() -> None:
    config = {
        "dynamic_reference_lookahead": True,
        "curve_scheduler_mode": "phase_active",
        "reference_lookahead_min": 2.5,
        "reference_lookahead_speed_table": [
            {"speed_mps": 10.0, "lookahead_m": 16.0},
        ],
        "reference_lookahead_speed_table_entry": [
            {"speed_mps": 10.0, "lookahead_m": 12.0},
        ],
    }

    result = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.001,
        path_curvature_preview=0.001,
        distance_to_curve_start_m=30.0,
        curve_scheduler_mode="phase_active",
        curve_phase=0.75,
        config=config,
        return_diagnostics=True,
    )
    assert isinstance(result, dict)
    assert float(result["lookahead"]) == pytest.approx(13.0, rel=1e-3)
    assert result["curve_scheduler_mode"] == "phase_active"
    assert float(result["curve_phase"]) == pytest.approx(0.75, rel=1e-6)


def test_reference_lookahead_entry_table_phase_shadow_keeps_binary_output() -> None:
    config = {
        "dynamic_reference_lookahead": True,
        "curve_scheduler_mode": "phase_shadow",
        "reference_lookahead_min": 2.5,
        "reference_lookahead_speed_table": [
            {"speed_mps": 10.0, "lookahead_m": 16.0},
        ],
        "reference_lookahead_speed_table_entry": [
            {"speed_mps": 10.0, "lookahead_m": 12.0},
        ],
        "reference_lookahead_entry_preview_curvature_min": 0.006,
        "reference_lookahead_entry_preview_curvature_max": 0.015,
        "reference_lookahead_entry_distance_start_m": 20.0,
        "reference_lookahead_entry_distance_end_m": 6.0,
    }

    result = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.001,
        path_curvature_preview=0.001,
        distance_to_curve_start_m=30.0,
        curve_scheduler_mode="phase_shadow",
        curve_phase=1.0,
        config=config,
        return_diagnostics=True,
    )
    assert isinstance(result, dict)
    assert float(result["lookahead"]) == pytest.approx(16.0, rel=1e-3)
    assert result["curve_scheduler_mode"] == "binary"
    assert float(result["curve_phase"]) == pytest.approx(0.0, rel=1e-6)


def test_reference_lookahead_slew_limit_caps_per_frame_change() -> None:
    config = {
        "dynamic_reference_lookahead": True,
        "reference_lookahead_min": 2.5,
        "reference_lookahead_speed_table": [
            {"speed_mps": 10.0, "lookahead_m": 16.0},
        ],
        "reference_lookahead_slew_rate_m_per_frame": 0.5,
    }

    lookahead = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.001,
        config=config,
        previous_lookahead=10.0,
    )
    assert lookahead == pytest.approx(10.5, rel=1e-3)


def test_curve_phase_scheduler_rearms_for_second_curve_entry() -> None:
    config = {
        "curve_phase_preview_curvature_min": 0.003,
        "curve_phase_preview_curvature_max": 0.015,
        "curve_phase_path_curvature_min": 0.003,
        "curve_phase_path_curvature_max": 0.015,
        "curve_phase_rise_min": 0.0005,
        "curve_phase_rise_max": 0.006,
        "curve_phase_ema_alpha": 1.0,
        "curve_phase_on": 0.45,
        "curve_phase_off": 0.30,
        "curve_phase_commit_on": 0.55,
        "curve_phase_commit_min_frames": 2,
        "curve_phase_rearm_hold_frames": 2,
    }
    state = {
        "curve_phase": 0.0,
        "curve_phase_state": "STRAIGHT",
        "curve_phase_entry_frames": 0,
        "curve_phase_rearm_hold_frames": 0,
    }

    def step(preview_abs: float, path_abs: float, rise_abs: float) -> dict:
        nonlocal state
        diag = compute_curve_phase_scheduler(
            preview_curvature_abs=preview_abs,
            path_curvature_abs=path_abs,
            curvature_rise_abs=rise_abs,
            previous_phase=float(state["curve_phase"]),
            previous_state=str(state["curve_phase_state"]),
            previous_entry_frames=int(state["curve_phase_entry_frames"]),
            previous_rearm_hold_frames=int(state["curve_phase_rearm_hold_frames"]),
            config=config,
        )
        state = diag
        return diag

    first_entry = step(0.014, 0.008, 0.004)
    first_commit = step(0.014, 0.010, 0.002)
    rearm = step(0.0, 0.0, 0.0)
    second_entry = step(0.014, 0.010, 0.003)

    assert first_entry["curve_phase_state"] == "ENTRY"
    assert first_commit["curve_phase_state"] == "COMMIT"
    assert rearm["curve_phase_state"] == "REARM"
    assert second_entry["curve_phase_state"] == "ENTRY"
    assert second_entry["curve_phase_rearm_event"] is True


def test_curve_phase_scheduler_can_arm_from_time_to_curve_signal() -> None:
    config = {
        "curve_phase_preview_curvature_min": 0.003,
        "curve_phase_preview_curvature_max": 0.015,
        "curve_phase_path_curvature_min": 0.003,
        "curve_phase_path_curvature_max": 0.015,
        "curve_phase_rise_min": 0.0005,
        "curve_phase_rise_max": 0.006,
        "curve_phase_time_to_curve_min_s": 0.8,
        "curve_phase_time_to_curve_max_s": 3.0,
        "curve_phase_confidence_min": 0.35,
        "curve_phase_confidence_max": 0.80,
        "curve_phase_confidence_floor_scale": 0.6,
        "curve_phase_ema_alpha": 1.0,
        "curve_phase_on": 0.45,
        "curve_phase_off": 0.30,
        "curve_phase_commit_on": 0.55,
        "curve_phase_commit_min_frames": 2,
    }

    diag = compute_curve_phase_scheduler(
        preview_curvature_abs=0.0005,
        path_curvature_abs=0.0005,
        curvature_rise_abs=0.0,
        time_to_curve_s=1.0,
        preview_confidence=0.9,
        previous_phase=0.0,
        previous_state="STRAIGHT",
        previous_entry_frames=0,
        previous_rearm_hold_frames=0,
        config=config,
    )

    assert diag["curve_phase_state"] == "ENTRY"
    assert float(diag["curve_phase_term_time"]) > 0.8
    assert float(diag["curve_phase_confidence_scale"]) > 0.95


def test_curve_phase_scheduler_scales_down_low_confidence_time_trigger() -> None:
    config = {
        "curve_phase_preview_curvature_min": 0.003,
        "curve_phase_preview_curvature_max": 0.015,
        "curve_phase_path_curvature_min": 0.003,
        "curve_phase_path_curvature_max": 0.015,
        "curve_phase_rise_min": 0.0005,
        "curve_phase_rise_max": 0.006,
        "curve_phase_time_to_curve_min_s": 0.8,
        "curve_phase_time_to_curve_max_s": 3.0,
        "curve_phase_confidence_min": 0.35,
        "curve_phase_confidence_max": 0.80,
        "curve_phase_confidence_floor_scale": 0.2,
        "curve_phase_ema_alpha": 1.0,
        "curve_phase_on": 0.45,
        "curve_phase_off": 0.30,
        "curve_phase_commit_on": 0.55,
        "curve_phase_commit_min_frames": 2,
    }

    diag = compute_curve_phase_scheduler(
        preview_curvature_abs=0.0005,
        path_curvature_abs=0.0005,
        curvature_rise_abs=0.0,
        time_to_curve_s=1.0,
        preview_confidence=0.0,
        previous_phase=0.0,
        previous_state="STRAIGHT",
        previous_entry_frames=0,
        previous_rearm_hold_frames=0,
        config=config,
    )

    assert diag["curve_phase_state"] == "STRAIGHT"
    assert float(diag["curve_phase_confidence_scale"]) == pytest.approx(0.2, rel=1e-6)


def test_curve_anticipation_score_uses_max_term() -> None:
    config = {
        "curve_anticipation_curvature_min": 0.0015,
        "curve_anticipation_curvature_max": 0.010,
        "curve_anticipation_heading_delta_min_rad": 0.10,
        "curve_anticipation_heading_delta_max_rad": 0.30,
        "curve_anticipation_far_rise_min": 0.20,
        "curve_anticipation_far_rise_max": 0.70,
    }
    score = compute_curve_anticipation_score(
        path_curvature_abs=0.001,
        preview_curvature_abs=0.001,
        heading_delta_abs=0.20,
        far_geometry_rise_abs=0.10,
        config=config,
    )
    assert score["term_curvature"] == pytest.approx(0.0, abs=1e-6)
    assert score["term_heading"] == pytest.approx(0.5, rel=1e-3)
    assert score["term_far_rise"] == pytest.approx(0.0, abs=1e-6)
    assert score["score_raw"] == pytest.approx(0.5, rel=1e-3)


def test_curve_anticipation_score_prefers_strong_curvature_evidence() -> None:
    config = {
        "curve_anticipation_curvature_min": 0.0015,
        "curve_anticipation_curvature_max": 0.010,
        "curve_anticipation_heading_delta_min_rad": 0.10,
        "curve_anticipation_heading_delta_max_rad": 0.30,
        "curve_anticipation_far_rise_min": 0.20,
        "curve_anticipation_far_rise_max": 0.70,
    }
    score = compute_curve_anticipation_score(
        path_curvature_abs=0.020,
        preview_curvature_abs=0.002,
        heading_delta_abs=0.05,
        far_geometry_rise_abs=0.10,
        config=config,
    )
    assert score["term_curvature"] == pytest.approx(1.0, rel=1e-6)
    assert score["term_heading"] == pytest.approx(0.0, abs=1e-6)
    assert score["term_far_rise"] == pytest.approx(0.0, abs=1e-6)
    assert score["score_raw"] == pytest.approx(1.0, rel=1e-6)


def test_stanley_mode_uses_heading_and_crosstrack() -> None:
    controller = LateralController(
        control_mode="stanley",
        stanley_k=1.5,
        stanley_soft_speed=2.0,
        stanley_heading_weight=1.0,
    )

    reference_point = {
        "x": 0.5,
        "y": 10.0,
        "heading": 0.1,
        "curvature": 0.0,
    }

    result = controller.compute_steering(
        0.0,
        reference_point,
        current_speed=8.0,
        return_metadata=True,
    )

    assert result["control_mode"] == "stanley"
    assert result["stanley_heading_term"] != 0.0
    assert result["stanley_crosstrack_term"] != 0.0
