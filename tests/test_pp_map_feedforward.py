import numpy as np
import pytest

from control.pid_controller import LateralController


def _pp_controller(**kwargs) -> LateralController:
    defaults = dict(
        kp=1.0,
        ki=0.0,
        kd=0.1,
        control_mode="pure_pursuit",
        pp_feedback_gain=0.0,
        max_steering=1.0,
        pp_max_steering_rate=5.0,
        pp_max_steering_jerk=0.0,
        pp_map_ff_enabled=True,
        pp_map_ff_gain=1.0,
        pp_map_ff_wheelbase_m=2.5,
        pp_map_ff_curvature_min=0.005,
        pp_map_ff_curvature_max_clip=0.08,
    )
    defaults.update(kwargs)
    return LateralController(**defaults)


def _ref_with_primary_curvature(curv: float) -> dict:
    return {
        "x": 1.0,
        "y": 8.0,
        "heading": np.sign(curv) * 0.02 if curv != 0.0 else 0.0,
        "velocity": 10.0,
        "curvature": curv,
        "curvature_primary_abs": abs(curv),
        "curvature_source": "primary_contract",
        "curvature_primary_source": "map_track",
    }


def test_map_ff_applies_for_primary_curvature_source() -> None:
    ctrl = _pp_controller()
    curv = 0.025
    meta = ctrl.compute_steering(
        0.0, _ref_with_primary_curvature(curv), current_speed=10.0, return_metadata=True
    )
    expected = (2.5 * curv) / np.radians(30.0)
    assert meta["pp_map_ff_applied"] == pytest.approx(expected, rel=0.01)


def test_map_ff_disabled_for_non_primary_source() -> None:
    ctrl = _pp_controller()
    meta = ctrl.compute_steering(
        0.0,
        {"x": 1.0, "y": 8.0, "heading": 0.02, "velocity": 10.0, "curvature": 0.025},
        current_speed=10.0,
        return_metadata=True,
    )
    assert meta["pp_map_ff_applied"] == pytest.approx(0.0, abs=1e-8)


def test_map_ff_respects_min_curvature_gate() -> None:
    ctrl = _pp_controller(pp_map_ff_curvature_min=0.01)
    meta = ctrl.compute_steering(
        0.0, _ref_with_primary_curvature(0.005), current_speed=10.0, return_metadata=True
    )
    assert meta["pp_map_ff_applied"] == pytest.approx(0.0, abs=1e-8)


def test_map_ff_disabled_flag() -> None:
    ctrl = _pp_controller(pp_map_ff_enabled=False)
    meta = ctrl.compute_steering(
        0.0, _ref_with_primary_curvature(0.025), current_speed=10.0, return_metadata=True
    )
    assert meta["pp_map_ff_applied"] == pytest.approx(0.0, abs=1e-8)


def test_map_ff_clips_large_curvature() -> None:
    ctrl = _pp_controller(pp_map_ff_curvature_max_clip=0.08)
    meta = ctrl.compute_steering(
        0.0, _ref_with_primary_curvature(0.2), current_speed=10.0, return_metadata=True
    )
    expected = (2.5 * 0.08) / np.radians(30.0)
    assert meta["pp_map_ff_applied"] == pytest.approx(expected, rel=0.01)
    assert abs(meta["steering"]) <= 1.0 + 1e-6


def test_map_ff_negative_curvature_sign() -> None:
    ctrl = _pp_controller()
    meta = ctrl.compute_steering(
        0.0, _ref_with_primary_curvature(-0.025), current_speed=10.0, return_metadata=True
    )
    assert meta["pp_map_ff_applied"] < 0.0


def test_map_ff_wheelbase_clamps_at_init() -> None:
    ctrl = _pp_controller(pp_map_ff_wheelbase_m=0.0)
    assert ctrl.pp_map_ff_wheelbase_m == pytest.approx(0.5, rel=1e-6)
    meta = ctrl.compute_steering(
        0.0, _ref_with_primary_curvature(0.025), current_speed=10.0, return_metadata=True
    )
    assert np.isfinite(meta["pp_map_ff_applied"])


def test_map_ff_and_speed_norm_both_apply() -> None:
    ctrl = _pp_controller(
        pp_speed_norm_enabled=True,
        pp_speed_norm_reference_mps=12.0,
        pp_speed_norm_min_scale=0.7,
    )
    meta = ctrl.compute_steering(
        0.0, _ref_with_primary_curvature(0.025), current_speed=15.0, return_metadata=True
    )
    # Quadratic formula: (12/15)² = 0.64, clamped to min_scale=0.70
    assert meta["pp_speed_norm_scale"] == pytest.approx(0.70, rel=1e-6)
    assert meta["pp_map_ff_applied"] > 0.0
    assert abs(meta["steering"]) <= ctrl.max_steering + 1e-6
