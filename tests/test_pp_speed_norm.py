import pytest

from control.pid_controller import LateralController


def _pp_controller(**kwargs) -> LateralController:
    defaults = dict(
        kp=1.0,
        ki=0.0,
        kd=0.1,
        control_mode="pure_pursuit",
        pp_feedback_gain=0.0,
        max_steering=0.8,
        pp_max_steering_rate=5.0,
        pp_max_steering_jerk=0.0,
        pp_speed_norm_enabled=True,
        pp_speed_norm_reference_mps=12.0,
        pp_speed_norm_min_scale=0.70,
    )
    defaults.update(kwargs)
    return LateralController(**defaults)


def _ref(x: float = 2.0, y: float = 8.0) -> dict:
    return {"x": x, "y": y, "heading": 0.0, "velocity": 10.0, "curvature": 0.0}


def test_speed_norm_at_reference_speed() -> None:
    ctrl = _pp_controller()
    meta = ctrl.compute_steering(0.0, _ref(), current_speed=12.0, return_metadata=True)
    assert meta["pp_speed_norm_scale"] == pytest.approx(1.0, rel=1e-6)


def test_speed_norm_clamps_to_min_scale() -> None:
    ctrl = _pp_controller()
    meta = ctrl.compute_steering(0.0, _ref(), current_speed=24.0, return_metadata=True)
    assert meta["pp_speed_norm_scale"] == pytest.approx(0.70, rel=1e-6)


def test_speed_norm_mid_speed() -> None:
    # Quadratic formula: (12/15)² = 0.64, clamped to min_scale=0.70
    ctrl = _pp_controller()
    meta = ctrl.compute_steering(0.0, _ref(), current_speed=15.0, return_metadata=True)
    assert meta["pp_speed_norm_scale"] == pytest.approx(0.70, rel=1e-6)


def test_speed_norm_disabled() -> None:
    ctrl = _pp_controller(pp_speed_norm_enabled=False)
    meta = ctrl.compute_steering(0.0, _ref(), current_speed=20.0, return_metadata=True)
    assert meta["pp_speed_norm_scale"] == pytest.approx(1.0, rel=1e-6)


def test_speed_norm_handles_none_speed() -> None:
    ctrl = _pp_controller()
    meta = ctrl.compute_steering(0.0, _ref(), current_speed=None, return_metadata=True)
    assert meta["pp_speed_norm_scale"] == pytest.approx(1.0, rel=1e-6)


def test_speed_norm_preserves_steering_sign() -> None:
    # Quadratic formula: (12/15)² = 0.64, clamped to min_scale=0.70
    ctrl = _pp_controller()
    meta = ctrl.compute_steering(
        0.0, _ref(x=-2.0), current_speed=15.0, return_metadata=True
    )
    assert meta["steering"] < 0.0
    assert meta["pp_speed_norm_scale"] == pytest.approx(0.70, rel=1e-6)


def test_speed_norm_below_reference_is_unity() -> None:
    ctrl = _pp_controller()
    meta = ctrl.compute_steering(0.0, _ref(), current_speed=11.9, return_metadata=True)
    assert meta["pp_speed_norm_scale"] == pytest.approx(1.0, rel=1e-6)


def test_speed_norm_quadratic_formula_unclamped() -> None:
    # Verify the quadratic formula (v_ref/v)² is used when the result is above min_scale.
    # At v=13.5 with min_scale=0.60: (12/13.5)² = 0.7901 — above floor, clamp does not fire.
    ctrl = _pp_controller(pp_speed_norm_min_scale=0.60)
    meta = ctrl.compute_steering(0.0, _ref(), current_speed=13.5, return_metadata=True)
    expected = (12.0 / 13.5) ** 2
    assert meta["pp_speed_norm_scale"] == pytest.approx(expected, rel=1e-5)
