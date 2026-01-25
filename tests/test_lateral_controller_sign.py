import numpy as np

from control.pid_controller import LateralController


def _make_reference_point(x: float, y: float = 8.0, heading: float = 0.0) -> dict:
    return {"x": x, "y": y, "heading": heading, "velocity": 8.0}


def test_lateral_controller_steers_toward_positive_ref_x() -> None:
    controller = LateralController(kp=1.0, ki=0.0, kd=0.0, max_steering=1.0)
    ref = _make_reference_point(x=1.0)
    steering = controller.compute_steering(0.0, ref)
    assert steering > 0.0


def test_lateral_controller_steers_toward_negative_ref_x() -> None:
    controller = LateralController(kp=1.0, ki=0.0, kd=0.0, max_steering=1.0)
    ref = _make_reference_point(x=-1.0)
    steering = controller.compute_steering(0.0, ref)
    assert steering < 0.0


def test_lateral_controller_heading_error_does_not_flip_on_straight() -> None:
    controller = LateralController(
        kp=1.0,
        ki=0.0,
        kd=0.0,
        max_steering=1.0,
        heading_weight=1.0,
        lateral_weight=0.0,
    )
    ref = _make_reference_point(x=1.0, y=8.0, heading=0.0)
    steering = controller.compute_steering(1.0, ref)
    assert steering > 0.0
