import pytest

from control.pid_controller import LateralController
from trajectory.utils import compute_reference_lookahead


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
