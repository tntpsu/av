import pytest

from control.pid_controller import LateralController
from trajectory.utils import (
    build_local_curve_reference,
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
    assert result["reference_lookahead_owner_mode"] == "phase_active"
    assert result["reference_lookahead_entry_weight_source"] == "curve_local_phase"
    assert result["reference_lookahead_fallback_active"] is False


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
    assert result["reference_lookahead_owner_mode"] == "binary"
    assert result["reference_lookahead_entry_weight_source"] == "preview_distance"
    assert result["reference_lookahead_fallback_active"] is False


def test_reference_lookahead_phase_active_labels_binary_fallback_when_phase_invalid() -> None:
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
        "reference_lookahead_entry_preview_curvature_min": 0.006,
        "reference_lookahead_entry_preview_curvature_max": 0.015,
        "reference_lookahead_entry_distance_start_m": 20.0,
        "reference_lookahead_entry_distance_end_m": 6.0,
    }

    result = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.001,
        path_curvature_preview=0.020,
        distance_to_curve_start_m=4.0,
        curve_scheduler_mode="phase_active",
        curve_phase=float("nan"),
        config=config,
        return_diagnostics=True,
    )

    assert isinstance(result, dict)
    assert float(result["lookahead"]) == pytest.approx(12.0, rel=1e-3)
    assert result["curve_scheduler_mode"] == "phase_active"
    assert result["reference_lookahead_owner_mode"] == "phase_active"
    assert result["reference_lookahead_entry_weight_source"] == "preview_distance_fallback"
    assert result["reference_lookahead_fallback_active"] is True


def test_reference_lookahead_phase_active_shortens_for_high_severity_local_entry() -> None:
    config = {
        "dynamic_reference_lookahead": True,
        "curve_scheduler_mode": "phase_active",
        "reference_lookahead_min": 2.5,
        "reference_lookahead_speed_table": [
            {"speed_mps": 0.0, "lookahead_m": 7.0},
            {"speed_mps": 10.0, "lookahead_m": 16.0},
        ],
        "reference_lookahead_speed_table_entry": [
            {"speed_mps": 0.0, "lookahead_m": 6.0},
            {"speed_mps": 10.0, "lookahead_m": 12.0},
        ],
        "reference_lookahead_phase_active_entry_scale_min": 0.8,
        "reference_lookahead_phase_active_entry_scale_severity_on": 0.35,
        "reference_lookahead_phase_active_entry_scale_severity_full": 0.85,
    }

    result = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.001,
        path_curvature_preview=0.020,
        distance_to_curve_start_m=3.0,
        curve_scheduler_mode="phase_active",
        curve_phase=1.0,
        curve_local_state="ENTRY",
        curve_local_entry_severity=1.0,
        config=config,
        return_diagnostics=True,
    )

    assert isinstance(result, dict)
    assert float(result["lookahead"]) == pytest.approx(9.6, rel=1e-3)


def test_reference_lookahead_phase_active_does_not_shorten_without_local_entry_state() -> None:
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
        "reference_lookahead_phase_active_entry_scale_min": 0.8,
        "reference_lookahead_phase_active_entry_scale_severity_on": 0.35,
        "reference_lookahead_phase_active_entry_scale_severity_full": 0.85,
    }

    result = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.001,
        path_curvature_preview=0.020,
        distance_to_curve_start_m=3.0,
        curve_scheduler_mode="phase_active",
        curve_phase=1.0,
        curve_local_state="STRAIGHT",
        curve_local_entry_severity=1.0,
        config=config,
        return_diagnostics=True,
    )

    assert isinstance(result, dict)
    assert float(result["lookahead"]) == pytest.approx(12.0, rel=1e-3)


def test_reference_lookahead_phase_active_commit_table_shortens_commit_state() -> None:
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
        "reference_lookahead_speed_table_commit": [
            {"speed_mps": 10.0, "lookahead_m": 8.0},
        ],
        "reference_lookahead_phase_active_commit_phase_on": 0.75,
        "reference_lookahead_phase_active_commit_phase_full": 0.95,
    }

    entry_diag = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.01,
        curve_scheduler_mode="phase_active",
        curve_phase=0.90,
        curve_local_state="ENTRY",
        config=config,
        return_diagnostics=True,
    )
    commit_diag = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.01,
        curve_scheduler_mode="phase_active",
        curve_phase=0.90,
        curve_local_state="COMMIT",
        config=config,
        return_diagnostics=True,
    )

    assert float(entry_diag["lookahead"]) == pytest.approx(12.4, rel=1e-3)
    assert float(commit_diag["lookahead"]) < float(entry_diag["lookahead"])
    assert float(commit_diag["lookahead"]) == pytest.approx(8.6875, rel=1e-3)
    assert float(commit_diag["reference_lookahead_owner_commit_band_target"]) == pytest.approx(
        8.0, rel=1e-3
    )
    assert float(commit_diag["reference_lookahead_owner_commit_progress"]) == pytest.approx(
        0.84375, rel=1e-3
    )
    assert bool(commit_diag["reference_lookahead_owner_commit_band_clamp_active"]) is False


def test_reference_lookahead_phase_active_commit_table_falls_back_when_missing() -> None:
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
        "reference_lookahead_phase_active_commit_phase_on": 0.75,
        "reference_lookahead_phase_active_commit_phase_full": 0.95,
    }

    diag = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.01,
        curve_scheduler_mode="phase_active",
        curve_phase=0.90,
        curve_local_state="COMMIT",
        config=config,
        return_diagnostics=True,
    )

    assert float(diag["lookahead"]) == pytest.approx(12.4, rel=1e-3)




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

    def step(
        preview_abs: float,
        path_abs: float,
        rise_abs: float,
        *,
        distance_to_curve_start_m: float,
        time_to_curve_s: float,
    ) -> dict:
        nonlocal state
        diag = compute_curve_phase_scheduler(
            preview_curvature_abs=preview_abs,
            path_curvature_abs=path_abs,
            curvature_rise_abs=rise_abs,
            distance_to_curve_start_m=distance_to_curve_start_m,
            time_to_curve_s=time_to_curve_s,
            previous_phase=float(state["curve_phase"]),
            previous_state=str(state["curve_phase_state"]),
            previous_entry_frames=int(state["curve_phase_entry_frames"]),
            previous_rearm_hold_frames=int(state["curve_phase_rearm_hold_frames"]),
            config=config,
        )
        state = diag
        return diag

    first_entry = step(0.014, 0.008, 0.004, distance_to_curve_start_m=4.0, time_to_curve_s=0.5)
    first_commit = step(0.014, 0.010, 0.002, distance_to_curve_start_m=2.0, time_to_curve_s=0.25)
    rearm = step(0.0, 0.0, 0.0, distance_to_curve_start_m=20.0, time_to_curve_s=4.0)
    second_entry = step(0.014, 0.010, 0.003, distance_to_curve_start_m=4.0, time_to_curve_s=0.5)

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
        "curve_local_phase_time_start_s": 3.0,
        "curve_local_phase_time_end_s": 0.8,
        "curve_phase_ema_alpha": 1.0,
        "curve_phase_on": 0.45,
        "curve_phase_off": 0.30,
        "curve_phase_commit_on": 0.55,
        "curve_phase_commit_min_frames": 2,
        "curve_local_phase_use_time_term": True,
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
        "curve_local_phase_time_start_s": 3.0,
        "curve_local_phase_time_end_s": 0.8,
        "curve_phase_ema_alpha": 1.0,
        "curve_phase_on": 0.45,
        "curve_phase_off": 0.30,
        "curve_phase_commit_on": 0.55,
        "curve_phase_commit_min_frames": 2,
        "curve_local_phase_use_time_term": True,
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


def test_curve_phase_scheduler_keeps_far_preview_but_gates_local_phase_when_curve_is_far() -> None:
    config = {
        "curve_phase_preview_curvature_min": 0.003,
        "curve_phase_preview_curvature_max": 0.015,
        "curve_phase_path_curvature_min": 0.003,
        "curve_phase_path_curvature_max": 0.015,
        "curve_phase_rise_min": 0.0005,
        "curve_phase_rise_max": 0.006,
        "curve_local_phase_distance_start_m": 8.0,
        "curve_local_phase_distance_end_m": 1.5,
        "curve_local_phase_time_start_s": 1.2,
        "curve_local_phase_time_end_s": 0.25,
        "curve_phase_ema_alpha": 1.0,
        "curve_phase_on": 0.45,
        "curve_phase_off": 0.30,
        "curve_phase_commit_on": 0.55,
        "curve_phase_commit_min_frames": 2,
    }

    diag = compute_curve_phase_scheduler(
        preview_curvature_abs=0.014,
        path_curvature_abs=0.0005,
        curvature_rise_abs=0.003,
        distance_to_curve_start_m=20.0,
        time_to_curve_s=3.0,
        preview_confidence=1.0,
        previous_phase=0.0,
        previous_state="STRAIGHT",
        previous_entry_frames=0,
        previous_rearm_hold_frames=0,
        config=config,
    )

    assert diag["curve_preview_far_upcoming"] is True
    assert float(diag["curve_local_gate_weight"]) == pytest.approx(0.0, abs=1e-6)
    assert float(diag["curve_local_phase"]) == pytest.approx(0.0, abs=1e-6)
    assert str(diag["curve_local_state"]).upper() == "STRAIGHT"


def test_curve_phase_scheduler_far_preview_high_while_local_phase_stays_low() -> None:
    config = {
        "curve_phase_preview_curvature_min": 0.003,
        "curve_phase_preview_curvature_max": 0.015,
        "curve_phase_path_curvature_min": 0.003,
        "curve_phase_path_curvature_max": 0.015,
        "curve_phase_rise_min": 0.0005,
        "curve_phase_rise_max": 0.006,
        "curve_phase_time_to_curve_min_s": 0.8,
        "curve_phase_time_to_curve_max_s": 3.0,
        "curve_local_phase_distance_start_m": 8.0,
        "curve_local_phase_distance_end_m": 1.5,
        "curve_local_phase_time_start_s": 1.2,
        "curve_local_phase_time_end_s": 0.25,
        "curve_phase_ema_alpha": 1.0,
        "curve_phase_on": 0.45,
        "curve_phase_off": 0.30,
        "curve_phase_commit_on": 0.55,
        "curve_phase_commit_min_frames": 2,
    }

    diag = compute_curve_phase_scheduler(
        preview_curvature_abs=0.014,
        path_curvature_abs=0.0005,
        curvature_rise_abs=0.0,
        distance_to_curve_start_m=18.0,
        time_to_curve_s=3.0,
        preview_confidence=1.0,
        previous_phase=0.0,
        previous_state="STRAIGHT",
        previous_entry_frames=0,
        previous_rearm_hold_frames=0,
        config=config,
    )

    assert diag["curve_preview_far_upcoming"] is True
    assert float(diag["curve_preview_far_phase"]) > 0.8
    assert float(diag["curve_local_gate_weight"]) == pytest.approx(0.0, abs=1e-6)
    assert float(diag["curve_local_phase"]) == pytest.approx(0.0, abs=1e-6)
    assert str(diag["curve_local_state"]).upper() == "STRAIGHT"
    assert diag["curve_local_reentry_ready"] is False


def test_curve_phase_scheduler_rearm_does_not_immediately_reenter_without_local_gate() -> None:
    config = {
        "curve_phase_preview_curvature_min": 0.003,
        "curve_phase_preview_curvature_max": 0.015,
        "curve_phase_path_curvature_min": 0.003,
        "curve_phase_path_curvature_max": 0.015,
        "curve_phase_rise_min": 0.0005,
        "curve_phase_rise_max": 0.006,
        "curve_phase_time_to_curve_min_s": 0.8,
        "curve_phase_time_to_curve_max_s": 3.0,
        "curve_local_phase_distance_start_m": 8.0,
        "curve_local_phase_distance_end_m": 1.5,
        "curve_local_phase_time_start_s": 1.2,
        "curve_local_phase_time_end_s": 0.25,
        "curve_phase_ema_alpha": 0.0,
        "curve_phase_on": 0.45,
        "curve_phase_off": 0.30,
        "curve_phase_commit_on": 0.55,
        "curve_phase_commit_min_frames": 2,
        "curve_local_phase_reentry_gate_min": 0.10,
    }

    diag = compute_curve_phase_scheduler(
        preview_curvature_abs=0.014,
        path_curvature_abs=0.0005,
        curvature_rise_abs=0.0,
        distance_to_curve_start_m=20.0,
        time_to_curve_s=3.0,
        preview_confidence=1.0,
        previous_phase=0.8,
        previous_state="REARM",
        previous_entry_frames=0,
        previous_rearm_hold_frames=2,
        config=config,
    )

    assert str(diag["curve_local_state"]).upper() in {"REARM", "STRAIGHT"}
    assert diag["curve_local_reentry_ready"] is False


def test_curve_phase_scheduler_reentry_occurs_when_local_gate_becomes_positive() -> None:
    config = {
        "curve_phase_preview_curvature_min": 0.003,
        "curve_phase_preview_curvature_max": 0.015,
        "curve_phase_path_curvature_min": 0.003,
        "curve_phase_path_curvature_max": 0.015,
        "curve_phase_rise_min": 0.0005,
        "curve_phase_rise_max": 0.006,
        "curve_phase_time_to_curve_min_s": 0.8,
        "curve_phase_time_to_curve_max_s": 3.0,
        "curve_local_phase_distance_start_m": 8.0,
        "curve_local_phase_distance_end_m": 1.5,
        "curve_local_phase_time_start_s": 1.2,
        "curve_local_phase_time_end_s": 0.25,
        "curve_phase_ema_alpha": 0.0,
        "curve_phase_on": 0.45,
        "curve_phase_off": 0.30,
        "curve_phase_commit_on": 0.55,
        "curve_phase_commit_min_frames": 2,
        "curve_local_phase_reentry_gate_min": 0.10,
    }

    diag = compute_curve_phase_scheduler(
        preview_curvature_abs=0.014,
        path_curvature_abs=0.0005,
        curvature_rise_abs=0.003,
        distance_to_curve_start_m=4.0,
        time_to_curve_s=0.8,
        preview_confidence=1.0,
        previous_phase=0.8,
        previous_state="REARM",
        previous_entry_frames=0,
        previous_rearm_hold_frames=2,
        config=config,
    )

    assert diag["curve_local_reentry_ready"] is True
    assert str(diag["curve_local_state"]).upper() == "ENTRY"


def test_curve_phase_scheduler_tight_curve_arms_earlier_than_gentle_curve() -> None:
    config = {
        "curve_phase_preview_curvature_min": 0.003,
        "curve_phase_preview_curvature_max": 0.015,
        "curve_phase_path_curvature_min": 0.003,
        "curve_phase_path_curvature_max": 0.015,
        "curve_phase_rise_min": 0.0005,
        "curve_phase_rise_max": 0.006,
        "curve_local_entry_severity_curvature_min": 0.003,
        "curve_local_entry_severity_curvature_max": 0.012,
        "curve_local_phase_distance_start_m": 8.0,
        "curve_local_phase_distance_start_tight_m": 10.0,
        "curve_local_phase_distance_end_m": 1.5,
        "curve_local_phase_time_start_s": 1.2,
        "curve_local_phase_time_start_tight_s": 1.6,
        "curve_local_phase_time_end_s": 0.25,
        "curve_local_entry_on_base": 0.45,
        "curve_local_entry_on_tight": 0.40,
        "curve_phase_off": 0.30,
        "curve_phase_commit_on": 0.55,
        "curve_phase_commit_min_frames": 2,
        "curve_phase_ema_alpha": 1.0,
    }

    gentle = compute_curve_phase_scheduler(
        preview_curvature_abs=0.004,
        path_curvature_abs=0.0005,
        curvature_rise_abs=0.0003,
        distance_to_curve_start_m=5.0,
        time_to_curve_s=1.0,
        preview_confidence=1.0,
        previous_phase=0.0,
        previous_state="STRAIGHT",
        previous_entry_frames=0,
        previous_rearm_hold_frames=0,
        config=config,
    )
    tight = compute_curve_phase_scheduler(
        preview_curvature_abs=0.014,
        path_curvature_abs=0.0005,
        curvature_rise_abs=0.003,
        distance_to_curve_start_m=5.0,
        time_to_curve_s=1.0,
        preview_confidence=1.0,
        previous_phase=0.0,
        previous_state="STRAIGHT",
        previous_entry_frames=0,
        previous_rearm_hold_frames=0,
        config=config,
    )

    assert str(gentle["curve_local_state"]).upper() == "STRAIGHT"
    assert str(tight["curve_local_state"]).upper() == "ENTRY"
    assert float(tight["curve_local_entry_severity"]) > float(gentle["curve_local_entry_severity"])
    assert float(tight["curve_local_entry_on_effective"]) < float(gentle["curve_local_entry_on_effective"])
    assert float(tight["curve_local_phase_distance_start_effective_m"]) > float(
        gentle["curve_local_phase_distance_start_effective_m"]
    )


def test_curve_phase_scheduler_reports_effective_entry_horizons_and_driver() -> None:
    config = {
        "curve_phase_preview_curvature_min": 0.003,
        "curve_phase_preview_curvature_max": 0.015,
        "curve_phase_path_curvature_min": 0.003,
        "curve_phase_path_curvature_max": 0.015,
        "curve_phase_rise_min": 0.0005,
        "curve_phase_rise_max": 0.006,
        "curve_local_entry_severity_curvature_min": 0.003,
        "curve_local_entry_severity_curvature_max": 0.012,
        "curve_local_phase_distance_start_m": 8.0,
        "curve_local_phase_distance_start_tight_m": 10.0,
        "curve_local_phase_distance_end_m": 1.5,
        "curve_local_phase_time_start_s": 1.2,
        "curve_local_phase_time_start_tight_s": 1.6,
        "curve_local_phase_time_end_s": 0.25,
        "curve_local_entry_on_base": 0.45,
        "curve_local_entry_on_tight": 0.40,
        "curve_phase_off": 0.30,
        "curve_phase_commit_on": 0.55,
        "curve_phase_commit_min_frames": 2,
        "curve_phase_ema_alpha": 1.0,
    }

    diag = compute_curve_phase_scheduler(
        preview_curvature_abs=0.014,
        path_curvature_abs=0.0005,
        curvature_rise_abs=0.003,
        distance_to_curve_start_m=5.0,
        time_to_curve_s=1.4,
        preview_confidence=1.0,
        previous_phase=0.0,
        previous_state="STRAIGHT",
        previous_entry_frames=0,
        previous_rearm_hold_frames=0,
        config=config,
    )

    assert float(diag["curve_local_entry_severity"]) > 0.9
    assert float(diag["curve_local_phase_distance_start_effective_m"]) == pytest.approx(10.0, rel=1e-6)
    assert float(diag["curve_local_phase_time_start_effective_s"]) == pytest.approx(1.6, rel=1e-6)
    assert float(diag["curve_local_entry_on_effective"]) == pytest.approx(0.40, rel=1e-6)
    assert str(diag["curve_local_entry_driver"]) in {"distance", "mixed"}


def test_curve_phase_scheduler_blocks_path_only_arm_when_curve_is_far() -> None:
    config = {
        "curve_phase_preview_curvature_min": 0.003,
        "curve_phase_preview_curvature_max": 0.015,
        "curve_phase_path_curvature_min": 0.003,
        "curve_phase_path_curvature_max": 0.015,
        "curve_phase_rise_min": 0.0005,
        "curve_phase_rise_max": 0.006,
        "curve_local_phase_distance_start_m": 8.0,
        "curve_local_phase_distance_end_m": 1.5,
        "curve_local_phase_time_start_s": 1.2,
        "curve_local_phase_time_end_s": 0.25,
        "curve_phase_ema_alpha": 1.0,
        "curve_phase_on": 0.45,
        "curve_phase_off": 0.30,
        "curve_phase_commit_on": 0.55,
        "curve_phase_commit_min_frames": 2,
    }

    diag = compute_curve_phase_scheduler(
        preview_curvature_abs=0.0,
        path_curvature_abs=0.02,
        curvature_rise_abs=0.0,
        distance_to_curve_start_m=16.0,
        time_to_curve_s=4.0,
        preview_confidence=1.0,
        previous_phase=0.0,
        previous_state="STRAIGHT",
        previous_entry_frames=0,
        previous_rearm_hold_frames=0,
        config=config,
    )

    assert diag["curve_local_arm_ready"] is False
    assert diag["curve_local_distance_ready"] is False
    assert diag["curve_local_in_curve_now"] is False
    assert str(diag["curve_local_state"]).upper() == "STRAIGHT"
    assert float(diag["curve_local_phase"]) == pytest.approx(0.0, abs=1e-6)
    assert float(diag["curve_local_gate_weight"]) == pytest.approx(0.0, abs=1e-6)
    assert float(diag["curve_local_arm_phase_raw"]) == pytest.approx(0.0, abs=1e-6)
    assert float(diag["curve_local_sustain_phase_raw"]) > 0.0


def test_curve_phase_scheduler_allows_path_sustain_once_curve_is_local() -> None:
    config = {
        "curve_phase_preview_curvature_min": 0.003,
        "curve_phase_preview_curvature_max": 0.015,
        "curve_phase_path_curvature_min": 0.003,
        "curve_phase_path_curvature_max": 0.015,
        "curve_phase_rise_min": 0.0005,
        "curve_phase_rise_max": 0.006,
        "curve_local_phase_distance_start_m": 8.0,
        "curve_local_phase_distance_end_m": 1.5,
        "curve_local_phase_time_start_s": 1.2,
        "curve_local_phase_time_end_s": 0.25,
        "curve_phase_ema_alpha": 1.0,
        "curve_phase_on": 0.45,
        "curve_phase_off": 0.30,
        "curve_phase_commit_on": 0.55,
        "curve_phase_commit_min_frames": 2,
    }

    diag = compute_curve_phase_scheduler(
        preview_curvature_abs=0.0,
        path_curvature_abs=0.02,
        curvature_rise_abs=0.0,
        distance_to_curve_start_m=0.0,
        time_to_curve_s=0.0,
        preview_confidence=1.0,
        previous_phase=0.0,
        previous_state="STRAIGHT",
        previous_entry_frames=0,
        previous_rearm_hold_frames=0,
        config=config,
    )

    assert diag["curve_local_arm_ready"] is True
    assert diag["curve_local_in_curve_now"] is True
    assert diag["curve_local_commit_ready"] is True
    assert str(diag["curve_local_entry_driver"]) in {"in_curve", "mixed"}
    assert str(diag["curve_local_state"]).upper() == "ENTRY"
    assert bool(diag["curve_local_path_sustain_active"]) is True


def test_curve_phase_scheduler_entry_can_arm_before_commit_ready() -> None:
    config = {
        "curve_phase_preview_curvature_min": 0.003,
        "curve_phase_preview_curvature_max": 0.015,
        "curve_phase_path_curvature_min": 0.003,
        "curve_phase_path_curvature_max": 0.015,
        "curve_phase_rise_min": 0.0005,
        "curve_phase_rise_max": 0.006,
        "curve_local_phase_distance_start_m": 8.0,
        "curve_local_phase_distance_end_m": 1.5,
        "curve_local_commit_distance_ready_m": 3.0,
        "curve_local_commit_time_ready_s": 0.60,
        "curve_phase_ema_alpha": 1.0,
        "curve_phase_on": 0.45,
        "curve_phase_off": 0.30,
        "curve_phase_commit_on": 0.55,
        "curve_phase_commit_min_frames": 2,
    }

    diag = compute_curve_phase_scheduler(
        preview_curvature_abs=0.014,
        path_curvature_abs=0.010,
        curvature_rise_abs=0.003,
        distance_to_curve_start_m=4.0,
        time_to_curve_s=0.7,
        preview_confidence=1.0,
        previous_phase=0.0,
        previous_state="STRAIGHT",
        previous_entry_frames=0,
        previous_rearm_hold_frames=0,
        config=config,
    )

    assert diag["curve_local_arm_ready"] is True
    assert diag["curve_local_commit_ready"] is False
    assert diag["curve_local_distance_ready"] is False
    assert str(diag["curve_local_state"]).upper() == "ENTRY"


def test_curve_phase_scheduler_does_not_report_time_ready_without_finite_ttc() -> None:
    config = {
        "curve_phase_preview_curvature_min": 0.003,
        "curve_phase_preview_curvature_max": 0.015,
        "curve_phase_path_curvature_min": 0.003,
        "curve_phase_path_curvature_max": 0.015,
        "curve_phase_rise_min": 0.0005,
        "curve_phase_rise_max": 0.006,
        "curve_local_phase_distance_start_m": 8.0,
        "curve_local_phase_distance_end_m": 1.5,
        "curve_local_phase_time_start_s": 1.2,
        "curve_local_phase_time_end_s": 0.25,
        "curve_local_phase_use_time_term": True,
        "curve_phase_ema_alpha": 1.0,
        "curve_phase_on": 0.45,
        "curve_phase_off": 0.30,
        "curve_phase_commit_on": 0.55,
        "curve_phase_commit_min_frames": 2,
    }

    diag = compute_curve_phase_scheduler(
        preview_curvature_abs=0.014,
        path_curvature_abs=0.010,
        curvature_rise_abs=0.003,
        distance_to_curve_start_m=16.0,
        time_to_curve_s=float("nan"),
        preview_confidence=1.0,
        previous_phase=0.0,
        previous_state="STRAIGHT",
        previous_entry_frames=0,
        previous_rearm_hold_frames=0,
        config=config,
    )

    assert diag["curve_local_time_ready"] is False
    assert diag["curve_local_arm_ready"] is False


def test_curve_phase_scheduler_commit_requires_commit_ready() -> None:
    config = {
        "curve_phase_preview_curvature_min": 0.003,
        "curve_phase_preview_curvature_max": 0.015,
        "curve_phase_path_curvature_min": 0.003,
        "curve_phase_path_curvature_max": 0.015,
        "curve_phase_rise_min": 0.0005,
        "curve_phase_rise_max": 0.006,
        "curve_local_phase_distance_start_m": 8.0,
        "curve_local_phase_distance_end_m": 1.5,
        "curve_local_commit_distance_ready_m": 3.0,
        "curve_local_commit_time_ready_s": 0.60,
        "curve_phase_ema_alpha": 1.0,
        "curve_phase_on": 0.45,
        "curve_phase_off": 0.30,
        "curve_phase_commit_on": 0.55,
        "curve_phase_commit_min_frames": 2,
    }

    first = compute_curve_phase_scheduler(
        preview_curvature_abs=0.014,
        path_curvature_abs=0.010,
        curvature_rise_abs=0.003,
        distance_to_curve_start_m=4.0,
        time_to_curve_s=0.8,
        preview_confidence=1.0,
        previous_phase=0.0,
        previous_state="STRAIGHT",
        previous_entry_frames=0,
        previous_rearm_hold_frames=0,
        config=config,
    )
    second = compute_curve_phase_scheduler(
        preview_curvature_abs=0.014,
        path_curvature_abs=0.010,
        curvature_rise_abs=0.003,
        distance_to_curve_start_m=4.0,
        time_to_curve_s=0.8,
        preview_confidence=1.0,
        previous_phase=float(first["curve_phase"]),
        previous_state=str(first["curve_local_state"]),
        previous_entry_frames=int(first["curve_phase_entry_frames"]),
        previous_rearm_hold_frames=int(first["curve_phase_rearm_hold_frames"]),
        config=config,
    )
    third = compute_curve_phase_scheduler(
        preview_curvature_abs=0.014,
        path_curvature_abs=0.010,
        curvature_rise_abs=0.003,
        distance_to_curve_start_m=2.5,
        time_to_curve_s=0.4,
        preview_confidence=1.0,
        previous_phase=float(second["curve_phase"]),
        previous_state=str(second["curve_local_state"]),
        previous_entry_frames=int(second["curve_phase_entry_frames"]),
        previous_rearm_hold_frames=int(second["curve_phase_rearm_hold_frames"]),
        config=config,
    )

    assert str(first["curve_local_state"]).upper() == "ENTRY"
    assert second["curve_local_commit_ready"] is False
    assert str(second["curve_local_state"]).upper() == "ENTRY"
    assert third["curve_local_commit_ready"] is True
    assert str(third["curve_local_commit_driver"]) in {"distance", "time", "mixed", "in_curve"}
    assert str(third["curve_local_state"]).upper() == "COMMIT"


def test_reference_lookahead_returns_local_gate_weight_diagnostics() -> None:
    config = {
        "dynamic_reference_lookahead": True,
        "reference_lookahead_min": 3.0,
        "reference_lookahead_speed_table": [
            {"speed_mps": 0.0, "lookahead_m": 8.0},
            {"speed_mps": 8.0, "lookahead_m": 8.0},
        ],
        "reference_lookahead_speed_table_entry": [
            {"speed_mps": 0.0, "lookahead_m": 4.0},
            {"speed_mps": 8.0, "lookahead_m": 4.0},
        ],
        "curve_scheduler_mode": "phase_active",
    }

    diag = compute_reference_lookahead(
        base_lookahead=8.0,
        current_speed=6.0,
        path_curvature=0.002,
        config=config,
        curve_phase=0.5,
        local_gate_weight=0.25,
        return_diagnostics=True,
    )

    assert float(diag["reference_lookahead_local_gate_weight"]) == pytest.approx(0.25, rel=1e-6)


def test_reference_lookahead_entry_shorten_guard_limits_entry_contraction() -> None:
    config = {
        "dynamic_reference_lookahead": True,
        "reference_lookahead_min": 2.5,
        "reference_lookahead_speed_table": [
            {"speed_mps": 0.0, "lookahead_m": 8.0},
            {"speed_mps": 8.0, "lookahead_m": 8.0},
        ],
        "reference_lookahead_speed_table_entry": [
            {"speed_mps": 0.0, "lookahead_m": 3.0},
            {"speed_mps": 8.0, "lookahead_m": 3.0},
        ],
        "curve_scheduler_mode": "phase_active",
        "reference_lookahead_slew_rate_shorten_m_per_frame": 0.30,
        "reference_lookahead_entry_shorten_slew_m_per_frame": 0.20,
        "reference_lookahead_entry_shorten_state_min": "ENTRY",
        "reference_lookahead_entry_shorten_state_max": "ENTRY",
        "reference_lookahead_entry_shorten_gate_min": 0.20,
        "reference_lookahead_entry_shorten_distance_min_m": 0.01,
        "reference_lookahead_entry_shorten_distance_max_m": 8.0,
    }

    diag = compute_reference_lookahead(
        base_lookahead=8.0,
        current_speed=6.0,
        path_curvature=0.0,
        config=config,
        previous_lookahead=6.0,
        curve_phase=1.0,
        curve_local_state="ENTRY",
        local_gate_weight=0.40,
        distance_to_curve_start_m=4.0,
        return_diagnostics=True,
    )

    assert float(diag["reference_lookahead_target_pre_entry_guard"]) == pytest.approx(5.7, rel=1e-6)
    assert bool(diag["reference_lookahead_entry_shorten_guard_active"]) is True
    assert float(diag["reference_lookahead_entry_shorten_guard_delta_m"]) == pytest.approx(0.1, rel=1e-6)
    assert float(diag["lookahead"]) == pytest.approx(5.8, rel=1e-6)


def test_reference_lookahead_entry_shorten_guard_ignores_nonlocal_state() -> None:
    config = {
        "dynamic_reference_lookahead": True,
        "reference_lookahead_min": 2.5,
        "reference_lookahead_speed_table": [
            {"speed_mps": 0.0, "lookahead_m": 8.0},
            {"speed_mps": 8.0, "lookahead_m": 8.0},
        ],
        "reference_lookahead_speed_table_entry": [
            {"speed_mps": 0.0, "lookahead_m": 3.0},
            {"speed_mps": 8.0, "lookahead_m": 3.0},
        ],
        "curve_scheduler_mode": "phase_active",
        "reference_lookahead_slew_rate_shorten_m_per_frame": 0.30,
        "reference_lookahead_entry_shorten_slew_m_per_frame": 0.20,
        "reference_lookahead_entry_shorten_state_min": "ENTRY",
        "reference_lookahead_entry_shorten_state_max": "ENTRY",
        "reference_lookahead_entry_shorten_gate_min": 0.20,
        "reference_lookahead_entry_shorten_distance_min_m": 0.01,
        "reference_lookahead_entry_shorten_distance_max_m": 8.0,
    }

    diag = compute_reference_lookahead(
        base_lookahead=8.0,
        current_speed=6.0,
        path_curvature=0.0,
        config=config,
        previous_lookahead=6.0,
        curve_phase=1.0,
        curve_local_state="STRAIGHT",
        local_gate_weight=0.40,
        distance_to_curve_start_m=4.0,
        return_diagnostics=True,
    )

    assert bool(diag["reference_lookahead_entry_shorten_guard_active"]) is False
    assert float(diag["reference_lookahead_entry_shorten_guard_delta_m"]) == pytest.approx(0.0, abs=1e-6)
    assert float(diag["reference_lookahead_target_pre_entry_guard"]) == pytest.approx(5.7, rel=1e-6)
    assert float(diag["lookahead"]) == pytest.approx(5.7, rel=1e-6)


def test_reference_lookahead_entry_shorten_guard_ignores_commit_state() -> None:
    config = {
        "dynamic_reference_lookahead": True,
        "reference_lookahead_min": 2.5,
        "reference_lookahead_speed_table": [
            {"speed_mps": 0.0, "lookahead_m": 8.0},
            {"speed_mps": 8.0, "lookahead_m": 8.0},
        ],
        "reference_lookahead_speed_table_entry": [
            {"speed_mps": 0.0, "lookahead_m": 3.0},
            {"speed_mps": 8.0, "lookahead_m": 3.0},
        ],
        "curve_scheduler_mode": "phase_active",
        "reference_lookahead_slew_rate_shorten_m_per_frame": 0.30,
        "reference_lookahead_entry_shorten_slew_m_per_frame": 0.20,
        "reference_lookahead_entry_shorten_state_min": "ENTRY",
        "reference_lookahead_entry_shorten_state_max": "ENTRY",
        "reference_lookahead_entry_shorten_gate_min": 0.20,
        "reference_lookahead_entry_shorten_distance_min_m": 0.01,
        "reference_lookahead_entry_shorten_distance_max_m": 8.0,
    }

    diag = compute_reference_lookahead(
        base_lookahead=8.0,
        current_speed=6.0,
        path_curvature=0.0,
        config=config,
        previous_lookahead=6.0,
        curve_phase=1.0,
        curve_local_state="COMMIT",
        local_gate_weight=0.40,
        distance_to_curve_start_m=4.0,
        return_diagnostics=True,
    )

    assert bool(diag["reference_lookahead_entry_shorten_guard_active"]) is False
    assert float(diag["lookahead"]) == pytest.approx(5.7, rel=1e-6)


def test_reference_lookahead_entry_shorten_guard_disables_at_curve_start() -> None:
    config = {
        "dynamic_reference_lookahead": True,
        "reference_lookahead_min": 2.5,
        "reference_lookahead_speed_table": [
            {"speed_mps": 0.0, "lookahead_m": 8.0},
            {"speed_mps": 8.0, "lookahead_m": 8.0},
        ],
        "reference_lookahead_speed_table_entry": [
            {"speed_mps": 0.0, "lookahead_m": 3.0},
            {"speed_mps": 8.0, "lookahead_m": 3.0},
        ],
        "curve_scheduler_mode": "phase_active",
        "reference_lookahead_slew_rate_shorten_m_per_frame": 0.30,
        "reference_lookahead_entry_shorten_slew_m_per_frame": 0.20,
        "reference_lookahead_entry_shorten_state_min": "ENTRY",
        "reference_lookahead_entry_shorten_state_max": "ENTRY",
        "reference_lookahead_entry_shorten_gate_min": 0.20,
        "reference_lookahead_entry_shorten_distance_min_m": 0.01,
        "reference_lookahead_entry_shorten_distance_max_m": 8.0,
    }

    diag = compute_reference_lookahead(
        base_lookahead=8.0,
        current_speed=6.0,
        path_curvature=0.0,
        config=config,
        previous_lookahead=6.0,
        curve_phase=1.0,
        curve_local_state="ENTRY",
        local_gate_weight=0.40,
        distance_to_curve_start_m=0.0,
        return_diagnostics=True,
    )

    assert bool(diag["reference_lookahead_entry_shorten_guard_active"]) is False
    assert float(diag["lookahead"]) == pytest.approx(5.7, rel=1e-6)


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


def test_local_curve_reference_builds_signed_shadow_target() -> None:
    config = {
        "local_curve_reference_mode": "shadow",
        "local_curve_reference_target_distance_table": [
            {"speed_mps": 0.0, "target_distance_m": 5.0},
            {"speed_mps": 10.0, "target_distance_m": 7.0},
        ],
        "local_curve_reference_entry_phase_on": 0.2,
        "local_curve_reference_entry_phase_full": 0.6,
        "local_curve_reference_commit_phase_on": 0.6,
        "local_curve_reference_commit_phase_full": 0.9,
        "local_curve_reference_curvature_min": 0.003,
        "local_curve_reference_curvature_max": 0.015,
        "local_curve_reference_tight_distance_scale_min": 0.8,
    }

    result = build_local_curve_reference(
        current_speed_mps=6.0,
        curve_local_state="ENTRY",
        curve_local_phase=0.6,
        curve_local_entry_severity=0.8,
        distance_to_curve_start_m=4.0,
        current_curve_progress_ratio=None,
        road_curvature_abs=0.01,
        preview_curvature_abs=0.01,
        base_reference_point={"x": 0.0, "y": 6.0, "heading": 0.0, "curvature": 0.01},
        config=config,
    )

    assert result["local_curve_reference_valid"] is True
    assert result["local_curve_reference_active"] is True
    assert result["local_curve_reference_source"] == "road_curvature"
    assert float(result["local_curve_reference_target_x"]) > 0.0
    assert float(result["local_curve_reference_target_y"]) > 0.0
    assert float(result["local_curve_reference_target_distance_m"]) < 7.0


def test_local_curve_reference_tightens_distance_with_higher_curvature() -> None:
    config = {
        "local_curve_reference_mode": "shadow",
        "local_curve_reference_target_distance_table": [
            {"speed_mps": 0.0, "target_distance_m": 5.0},
            {"speed_mps": 10.0, "target_distance_m": 7.0},
        ],
        "local_curve_reference_entry_phase_on": 0.2,
        "local_curve_reference_entry_phase_full": 0.6,
        "local_curve_reference_curvature_min": 0.003,
        "local_curve_reference_curvature_max": 0.015,
        "local_curve_reference_tight_distance_scale_min": 0.7,
    }

    base_ref = {"x": 0.0, "y": 6.0, "heading": 0.0, "curvature": 0.01}
    loose = build_local_curve_reference(
        current_speed_mps=6.0,
        curve_local_state="ENTRY",
        curve_local_phase=0.6,
        curve_local_entry_severity=0.2,
        distance_to_curve_start_m=4.0,
        current_curve_progress_ratio=None,
        road_curvature_abs=0.004,
        preview_curvature_abs=0.004,
        base_reference_point=base_ref,
        config=config,
    )
    tight = build_local_curve_reference(
        current_speed_mps=6.0,
        curve_local_state="ENTRY",
        curve_local_phase=0.6,
        curve_local_entry_severity=0.9,
        distance_to_curve_start_m=4.0,
        current_curve_progress_ratio=None,
        road_curvature_abs=0.012,
        preview_curvature_abs=0.012,
        base_reference_point=base_ref,
        config=config,
    )

    assert float(tight["local_curve_reference_target_distance_m"]) < float(
        loose["local_curve_reference_target_distance_m"]
    )


def test_local_curve_reference_falls_back_when_table_missing() -> None:
    result = build_local_curve_reference(
        current_speed_mps=6.0,
        curve_local_state="ENTRY",
        curve_local_phase=0.6,
        curve_local_entry_severity=0.8,
        distance_to_curve_start_m=4.0,
        current_curve_progress_ratio=None,
        road_curvature_abs=0.01,
        preview_curvature_abs=0.01,
        base_reference_point={"x": 0.0, "y": 6.0, "heading": 0.0, "curvature": 0.01},
        config={"local_curve_reference_mode": "shadow"},
    )

    assert result["local_curve_reference_valid"] is False
    assert result["local_curve_reference_fallback_active"] is True
    assert result["local_curve_reference_fallback_reason"] == "distance_table_missing"


def test_local_curve_reference_uses_signed_preview_direction() -> None:
    config = {
        "local_curve_reference_mode": "shadow",
        "local_curve_reference_target_distance_table": [
            {"speed_mps": 0.0, "lookahead_m": 4.5},
            {"speed_mps": 8.0, "lookahead_m": 5.0},
        ],
        "local_curve_reference_entry_phase_on": 0.2,
        "local_curve_reference_entry_phase_full": 0.6,
        "local_curve_reference_curvature_min": 0.003,
        "local_curve_reference_curvature_max": 0.015,
        "local_curve_reference_tight_distance_scale_min": 0.8,
    }

    result = build_local_curve_reference(
        current_speed_mps=6.0,
        curve_local_state="ENTRY",
        curve_local_phase=0.75,
        curve_local_entry_severity=0.8,
        distance_to_curve_start_m=3.0,
        current_curve_progress_ratio=None,
        road_curvature_abs=None,
        road_curvature_signed=None,
        preview_curvature_abs=0.01,
        preview_curvature_signed=-0.01,
        base_reference_point={"x": 0.15, "y": 5.5, "heading": 0.09, "curvature": 0.01},
        config=config,
    )

    assert result["local_curve_reference_valid"] is True
    assert float(result["local_curve_reference_curve_direction_sign"]) == pytest.approx(-1.0)
    assert float(result["local_curve_reference_target_x"]) < 0.0


def test_local_curve_reference_commit_unwinds_with_curve_progress() -> None:
    config = {
        "local_curve_reference_mode": "shadow",
        "local_curve_reference_target_distance_table": [
            {"speed_mps": 0.0, "lookahead_m": 4.5},
            {"speed_mps": 8.0, "lookahead_m": 5.0},
        ],
        "local_curve_reference_entry_phase_on": 0.2,
        "local_curve_reference_entry_phase_full": 0.6,
        "local_curve_reference_commit_phase_on": 0.6,
        "local_curve_reference_commit_phase_full": 0.9,
        "local_curve_reference_unwind_progress_start": 0.35,
        "local_curve_reference_unwind_progress_end": 0.70,
        "local_curve_reference_curvature_min": 0.003,
        "local_curve_reference_curvature_max": 0.015,
        "local_curve_reference_tight_distance_scale_min": 0.8,
    }

    committed = build_local_curve_reference(
        current_speed_mps=6.0,
        curve_local_state="COMMIT",
        curve_local_phase=1.0,
        curve_local_entry_severity=1.0,
        distance_to_curve_start_m=0.0,
        current_curve_progress_ratio=0.10,
        road_curvature_abs=0.01,
        road_curvature_signed=0.01,
        preview_curvature_abs=0.01,
        preview_curvature_signed=0.01,
        base_reference_point={"x": 0.02, "y": 5.5, "heading": 0.02, "curvature": 0.01},
        config=config,
    )
    unwound = build_local_curve_reference(
        current_speed_mps=6.0,
        curve_local_state="COMMIT",
        curve_local_phase=1.0,
        curve_local_entry_severity=1.0,
        distance_to_curve_start_m=0.0,
        current_curve_progress_ratio=0.60,
        road_curvature_abs=0.01,
        road_curvature_signed=0.01,
        preview_curvature_abs=0.01,
        preview_curvature_signed=0.01,
        base_reference_point={"x": 0.02, "y": 5.5, "heading": 0.02, "curvature": 0.01},
        config=config,
    )

    assert float(unwound["local_curve_reference_blend_weight"]) < float(
        committed["local_curve_reference_blend_weight"]
    )


# ── Bounded mode tests ──────────────────────────────────────────────


def _bounded_config(**overrides):
    """Helper: minimal config for bounded mode tests."""
    cfg = {
        "local_curve_reference_mode": "bounded",
        "local_curve_reference_delta_cap_m": 0.8,
        "local_curve_reference_max_blend": 0.7,
        "local_curve_reference_target_distance_table": [
            {"speed_mps": 0.0, "target_distance_m": 5.0},
            {"speed_mps": 10.0, "target_distance_m": 7.0},
        ],
        "local_curve_reference_entry_phase_on": 0.2,
        "local_curve_reference_entry_phase_full": 0.6,
        "local_curve_reference_commit_phase_on": 0.6,
        "local_curve_reference_commit_phase_full": 0.9,
        "local_curve_reference_curvature_min": 0.003,
        "local_curve_reference_curvature_max": 0.015,
        "local_curve_reference_tight_distance_scale_min": 0.8,
        "local_curve_reference_unwind_progress_start": 0.35,
        "local_curve_reference_unwind_progress_end": 0.70,
    }
    cfg.update(overrides)
    return cfg


def test_bounded_mode_caps_blend_weight() -> None:
    """Blend weight must not exceed max_blend in bounded mode."""
    config = _bounded_config(local_curve_reference_max_blend=0.5)
    result = build_local_curve_reference(
        current_speed_mps=6.0,
        curve_local_state="ENTRY",
        curve_local_phase=1.0,  # would give blend=1.0 without cap
        curve_local_entry_severity=0.8,
        distance_to_curve_start_m=2.0,
        current_curve_progress_ratio=None,
        road_curvature_abs=0.012,
        preview_curvature_abs=0.012,
        base_reference_point={"x": 0.0, "y": 6.0, "heading": 0.0, "curvature": 0.012},
        config=config,
    )
    assert result["local_curve_reference_valid"] is True
    assert result["local_curve_reference_active"] is True
    assert float(result["local_curve_reference_blend_weight"]) <= 0.5 + 1e-6


def test_bounded_mode_caps_delta_magnitude() -> None:
    """Applied correction must not exceed delta_cap_m."""
    config = _bounded_config(
        local_curve_reference_delta_cap_m=0.3,
        local_curve_reference_max_blend=1.0,  # allow full blend to test cap only
    )
    result = build_local_curve_reference(
        current_speed_mps=6.0,
        curve_local_state="ENTRY",
        curve_local_phase=1.0,
        curve_local_entry_severity=1.0,
        distance_to_curve_start_m=1.0,
        current_curve_progress_ratio=None,
        road_curvature_abs=0.015,
        preview_curvature_abs=0.015,
        base_reference_point={"x": 0.0, "y": 6.0, "heading": 0.0, "curvature": 0.015},
        config=config,
    )
    assert result["local_curve_reference_valid"] is True
    assert result["local_curve_reference_cap_active"] is True
    assert float(result["local_curve_reference_capped_delta_m"]) <= 0.3 + 1e-6
    # Raw delta should be larger than the cap
    assert float(result["local_curve_reference_raw_delta_m"]) > 0.3


def test_bounded_mode_no_cap_when_delta_small() -> None:
    """Cap should not activate when correction is within bounds."""
    config = _bounded_config(
        local_curve_reference_delta_cap_m=5.0,  # very large cap
        local_curve_reference_max_blend=1.0,
    )
    result = build_local_curve_reference(
        current_speed_mps=6.0,
        curve_local_state="ENTRY",
        curve_local_phase=0.4,  # moderate blend
        curve_local_entry_severity=0.3,
        distance_to_curve_start_m=4.0,
        current_curve_progress_ratio=None,
        road_curvature_abs=0.005,
        preview_curvature_abs=0.005,
        base_reference_point={"x": 0.0, "y": 6.0, "heading": 0.0, "curvature": 0.005},
        config=config,
    )
    assert result["local_curve_reference_valid"] is True
    assert result["local_curve_reference_cap_active"] is False
    # Capped delta should equal vs_planner_delta
    assert float(result["local_curve_reference_capped_delta_m"]) == pytest.approx(
        float(result["local_curve_reference_vs_planner_delta_m"]), abs=1e-6
    )


def test_bounded_mode_heading_scales_with_delta_cap() -> None:
    """When delta cap fires, heading correction should be proportionally reduced."""
    config = _bounded_config(
        local_curve_reference_delta_cap_m=0.2,
        local_curve_reference_max_blend=1.0,
    )
    # Run with tiny cap
    capped = build_local_curve_reference(
        current_speed_mps=6.0,
        curve_local_state="ENTRY",
        curve_local_phase=1.0,
        curve_local_entry_severity=1.0,
        distance_to_curve_start_m=1.0,
        current_curve_progress_ratio=None,
        road_curvature_abs=0.015,
        preview_curvature_abs=0.015,
        base_reference_point={"x": 0.0, "y": 6.0, "heading": 0.0, "curvature": 0.015},
        config=config,
    )
    # Run with no cap (shadow mode, same geometry)
    uncapped_config = dict(config)
    uncapped_config["local_curve_reference_mode"] = "shadow"
    uncapped = build_local_curve_reference(
        current_speed_mps=6.0,
        curve_local_state="ENTRY",
        curve_local_phase=1.0,
        curve_local_entry_severity=1.0,
        distance_to_curve_start_m=1.0,
        current_curve_progress_ratio=None,
        road_curvature_abs=0.015,
        preview_curvature_abs=0.015,
        base_reference_point={"x": 0.0, "y": 6.0, "heading": 0.0, "curvature": 0.015},
        config=uncapped_config,
    )
    # Heading should be smaller in bounded mode
    assert abs(float(capped["local_curve_reference_target_heading"])) < abs(
        float(uncapped["local_curve_reference_target_heading"])
    )


def test_bounded_mode_returns_new_diagnostic_fields() -> None:
    """Bounded mode must populate raw_delta, capped_delta, and cap_active."""
    config = _bounded_config()
    result = build_local_curve_reference(
        current_speed_mps=6.0,
        curve_local_state="ENTRY",
        curve_local_phase=0.5,
        curve_local_entry_severity=0.5,
        distance_to_curve_start_m=3.0,
        current_curve_progress_ratio=None,
        road_curvature_abs=0.01,
        preview_curvature_abs=0.01,
        base_reference_point={"x": 0.0, "y": 6.0, "heading": 0.0, "curvature": 0.01},
        config=config,
    )
    assert "local_curve_reference_raw_delta_m" in result
    assert "local_curve_reference_capped_delta_m" in result
    assert "local_curve_reference_cap_active" in result
    assert isinstance(result["local_curve_reference_raw_delta_m"], float)
    assert isinstance(result["local_curve_reference_capped_delta_m"], float)
    assert isinstance(result["local_curve_reference_cap_active"], bool)


def test_bounded_mode_shadow_still_works_unchanged() -> None:
    """Shadow mode should NOT apply blend cap or delta cap."""
    config = _bounded_config()
    config["local_curve_reference_mode"] = "shadow"
    config["local_curve_reference_max_blend"] = 0.3  # should be ignored in shadow
    config["local_curve_reference_delta_cap_m"] = 0.1  # should be ignored in shadow
    result = build_local_curve_reference(
        current_speed_mps=6.0,
        curve_local_state="ENTRY",
        curve_local_phase=1.0,
        curve_local_entry_severity=1.0,
        distance_to_curve_start_m=1.0,
        current_curve_progress_ratio=None,
        road_curvature_abs=0.015,
        preview_curvature_abs=0.015,
        base_reference_point={"x": 0.0, "y": 6.0, "heading": 0.0, "curvature": 0.015},
        config=config,
    )
    # Blend should NOT be capped at 0.3 — shadow ignores max_blend
    assert float(result["local_curve_reference_blend_weight"]) > 0.3 + 0.1
    assert result["local_curve_reference_cap_active"] is False


def test_bounded_mode_entry_only_unwind_fades_in_curve() -> None:
    """In bounded mode, correction should fade to zero as curve_progress_ratio increases."""
    config = _bounded_config(
        local_curve_reference_max_blend=1.0,
        local_curve_reference_delta_cap_m=5.0,
        local_curve_reference_bounded_unwind_start=0.0,
        local_curve_reference_bounded_unwind_end=0.25,
    )
    common = dict(
        current_speed_mps=6.0,
        curve_local_state="COMMIT",
        curve_local_phase=1.0,
        curve_local_entry_severity=1.0,
        distance_to_curve_start_m=0.0,
        road_curvature_abs=0.012,
        preview_curvature_abs=0.012,
        base_reference_point={"x": 0.0, "y": 6.0, "heading": 0.0, "curvature": 0.012},
        config=config,
    )

    # At curve start (progress=0): full correction
    at_start = build_local_curve_reference(
        **common, current_curve_progress_ratio=0.0,
    )
    # Mid-curve (progress=0.15): partial correction
    mid_curve = build_local_curve_reference(
        **common, current_curve_progress_ratio=0.15,
    )
    # In-curve (progress=0.30): fully unwound
    in_curve = build_local_curve_reference(
        **common, current_curve_progress_ratio=0.30,
    )

    start_blend = float(at_start["local_curve_reference_blend_weight"])
    mid_blend = float(mid_curve["local_curve_reference_blend_weight"])
    in_blend = float(in_curve["local_curve_reference_blend_weight"])

    # Start should have meaningful correction
    assert start_blend > 0.3
    # Mid should be reduced
    assert mid_blend < start_blend
    # In-curve should be near zero
    assert in_blend < 0.05


def test_bounded_mode_entry_phase_has_full_correction() -> None:
    """During ENTRY phase, bounded correction is not unwound (only COMMIT unwinds)."""
    config = _bounded_config(
        local_curve_reference_max_blend=1.0,
        local_curve_reference_delta_cap_m=5.0,
        local_curve_reference_bounded_unwind_start=0.0,
        local_curve_reference_bounded_unwind_end=0.25,
    )

    result = build_local_curve_reference(
        current_speed_mps=6.0,
        curve_local_state="ENTRY",
        curve_local_phase=0.8,
        curve_local_entry_severity=0.8,
        distance_to_curve_start_m=3.0,
        current_curve_progress_ratio=None,
        road_curvature_abs=0.012,
        preview_curvature_abs=0.012,
        base_reference_point={"x": 0.0, "y": 6.0, "heading": 0.0, "curvature": 0.012},
        config=config,
    )

    # ENTRY should have strong correction — no entry_fade applied
    assert float(result["local_curve_reference_blend_weight"]) > 0.5
    assert float(result["local_curve_reference_vs_planner_delta_m"]) > 0.1


# ---------------------------------------------------------------------------
# Curve phase scheduler: local-relevance sustain gating
# ---------------------------------------------------------------------------

def _make_scheduler_config() -> dict:
    """Minimal config sufficient to drive compute_curve_phase_scheduler."""
    return {
        "curve_phase_preview_curvature_min": 0.003,
        "curve_phase_preview_curvature_max": 0.015,
        "curve_phase_path_curvature_min": 0.003,
        "curve_phase_path_curvature_max": 0.015,
        "curve_phase_rise_min": 0.0005,
        "curve_phase_rise_max": 0.006,
        "curve_phase_ema_alpha": 0.45,
        "curve_phase_on": 0.45,
        "curve_local_entry_on_base": 0.45,
        "curve_local_entry_on_tight": 0.40,
        "curve_phase_off": 0.30,
        "curve_phase_commit_on": 0.55,
        "curve_phase_commit_min_frames": 4,
        "curve_phase_entry_floor": 0.15,
        "curve_phase_commit_floor": 0.30,
        "curve_phase_rearm_hold_frames": 4,
        "curve_local_phase_distance_start_m": 8.0,
        "curve_local_phase_distance_start_tight_m": 10.0,
        "curve_local_phase_distance_end_m": 1.5,
        "curve_local_phase_time_start_s": 1.2,
        "curve_local_phase_time_start_tight_s": 1.6,
        "curve_local_phase_time_end_s": 0.25,
        "curve_local_in_curve_distance_eps_m": 0.25,
        "curve_local_in_curve_time_eps_s": 0.05,
        "curve_local_in_curve_path_min": 0.70,
        "curve_local_commit_distance_ready_m": 3.0,
        "curve_local_commit_time_ready_s": 0.60,
        "curve_local_phase_reentry_gate_min": 0.10,
        "curve_local_phase_reentry_path_min": 0.15,
        "curve_local_entry_severity_curvature_min": 0.003,
        "curve_local_entry_severity_curvature_max": 0.012,
        "curve_phase_confidence_min": 0.35,
        "curve_phase_confidence_max": 0.80,
        "curve_phase_confidence_floor_scale": 0.6,
        "curve_local_phase_use_time_term": False,
    }


def _run_scheduler(config: dict, frames: list[dict], initial_state: dict | None = None) -> list[dict]:
    """Run compute_curve_phase_scheduler for a sequence of frames, returning all results."""
    state = initial_state or {
        "curve_phase": 0.0,
        "curve_phase_state": "STRAIGHT",
        "curve_phase_entry_frames": 0,
        "curve_phase_rearm_hold_frames": 0,
    }
    results = []
    for frame in frames:
        result = compute_curve_phase_scheduler(
            preview_curvature_abs=frame.get("preview_curvature_abs", 0.025),
            path_curvature_abs=frame.get("path_curvature_abs", 0.0),
            curvature_rise_abs=frame.get("curvature_rise_abs", 0.0),
            distance_to_curve_start_m=frame.get("distance_to_curve_start_m", None),
            time_to_curve_s=frame.get("time_to_curve_s", None),
            previous_phase=state["curve_phase"],
            previous_state=state["curve_phase_state"],
            previous_entry_frames=state["curve_phase_entry_frames"],
            previous_rearm_hold_frames=state["curve_phase_rearm_hold_frames"],
            config=config,
        )
        state = {
            "curve_phase": result["curve_phase"],
            "curve_phase_state": result["curve_phase_state"],
            "curve_phase_entry_frames": result["curve_phase_entry_frames"],
            "curve_phase_rearm_hold_frames": result["curve_phase_rearm_hold_frames"],
        }
        results.append(result)
    return results


def test_commit_exits_on_straight_when_far_preview_only() -> None:
    """
    COMMIT must exit to REARM/STRAIGHT when the vehicle is on a straight
    and the next curve is far enough away that local_commit_ready is False —
    even when far preview curvature is high (next curve is visible ahead).

    This is the s_loop trap: preview is always high, but COMMIT must not
    persist across inter-curve straights.
    """
    config = _make_scheduler_config()

    # --- Phase 1: approach and enter a curve (builds phase, reaches COMMIT) ---
    approach_frames = [
        {"preview_curvature_abs": 0.025, "path_curvature_abs": 0.001,
         "distance_to_curve_start_m": max(0.1, 5.0 - i * 0.4)}
        for i in range(20)
    ]
    results = _run_scheduler(config, approach_frames)
    # After approach + in-curve, should have reached COMMIT at some point
    states_seen = {r["curve_local_state"] for r in results}
    assert "COMMIT" in states_seen or "ENTRY" in states_seen, \
        "Should have entered ENTRY or COMMIT during approach"

    commit_state = {
        "curve_phase": results[-1]["curve_phase"],
        "curve_phase_state": results[-1]["curve_phase_state"],
        "curve_phase_entry_frames": results[-1]["curve_phase_entry_frames"],
        "curve_phase_rearm_hold_frames": results[-1]["curve_phase_rearm_hold_frames"],
    }

    # --- Phase 2: straight segment — far preview still high, next curve 8m away ---
    # distance_to_curve_start = 8m means local_commit_ready is False (> 3m threshold)
    # local_arm_ready is at edge (distance_start_m=8.0 in config)
    straight_frames = [
        {"preview_curvature_abs": 0.025, "path_curvature_abs": 0.001,
         "distance_to_curve_start_m": 8.0}
        for _ in range(15)
    ]
    straight_results = _run_scheduler(config, straight_frames, initial_state=commit_state)

    final_state = straight_results[-1]["curve_local_state"]
    assert final_state in ("STRAIGHT", "REARM"), (
        f"COMMIT persisted on straight for 15 frames with far preview only. "
        f"Final state: {final_state}, phase: {straight_results[-1]['curve_local_phase']:.3f}. "
        f"Far preview should NOT sustain COMMIT across straights."
    )


def test_commit_sustains_when_physically_in_curve() -> None:
    """
    COMMIT must sustain when the vehicle is physically inside a curve
    (high path curvature, distance past curve start).

    The sustain gate change should only affect far-preview-only scenarios.
    When in_curve_now is True (distance ≤ 0.25m) or path curvature is high,
    local_commit_ready is True and full sustain applies.
    """
    config = _make_scheduler_config()

    # Start from a settled COMMIT state
    in_commit_state = {
        "curve_phase": 0.65,
        "curve_phase_state": "COMMIT",
        "curve_phase_entry_frames": 8,
        "curve_phase_rearm_hold_frames": 0,
    }

    # Simulate 20 frames of being in the curve:
    # - distance ≤ 0.25m (in_curve_now = True → local_commit_ready = True)
    # - high path curvature
    in_curve_frames = [
        {"preview_curvature_abs": 0.025, "path_curvature_abs": 0.022,
         "curvature_rise_abs": 0.0, "distance_to_curve_start_m": 0.0}
        for _ in range(20)
    ]
    results = _run_scheduler(config, in_curve_frames, initial_state=in_commit_state)

    # All frames should remain in COMMIT
    non_commit = [r for r in results if r["curve_local_state"] != "COMMIT"]
    assert len(non_commit) == 0, (
        f"COMMIT exited while physically in curve. "
        f"Non-commit frames: {[(r['curve_local_state'], r['curve_local_phase']) for r in non_commit]}"
    )


def test_phase_decays_faster_on_straight_after_gating() -> None:
    """
    With the sustain gating fix, phase must decay materially faster on a
    straight (distance > commit_distance_ready_m) compared to what
    far-preview-only sustain would produce.

    Specifically: starting from phase=0.50 in COMMIT state, on a straight
    where distance_to_curve_start = 8.0m (> commit threshold of 3.0m),
    phase should drop below 0.30 within 10 frames.
    """
    config = _make_scheduler_config()

    in_commit_state = {
        "curve_phase": 0.50,
        "curve_phase_state": "COMMIT",
        "curve_phase_entry_frames": 5,
        "curve_phase_rearm_hold_frames": 0,
    }

    # Straight: far preview high, path curvature near zero, next curve 8m away
    straight_frames = [
        {"preview_curvature_abs": 0.025, "path_curvature_abs": 0.001,
         "distance_to_curve_start_m": 8.0}
        for _ in range(10)
    ]
    results = _run_scheduler(config, straight_frames, initial_state=in_commit_state)

    # Phase must have exited COMMIT (dropped below the OFF threshold or lost commit_ready)
    final_state = results[-1]["curve_local_state"]
    final_phase = results[-1]["curve_local_phase"]
    assert final_state in ("STRAIGHT", "REARM"), (
        f"Phase did not decay on straight within 10 frames. "
        f"final_state={final_state}, final_phase={final_phase:.3f}. "
        f"Phase trace: {[r['curve_local_phase'] for r in results]}"
    )
