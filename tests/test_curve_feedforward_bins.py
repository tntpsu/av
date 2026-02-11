from __future__ import annotations

from control.pid_controller import LateralController


def test_curve_feedforward_bins_apply_scale() -> None:
    controller = LateralController(
        curve_feedforward_bins=[
            {"min_curvature": 0.0, "max_curvature": 0.02, "gain_scale": 1.0},
            {"min_curvature": 0.02, "max_curvature": 10.0, "gain_scale": 1.1},
        ],
        curve_feedforward_curvature_min=0.001,
        curve_feedforward_curvature_max=0.02,
        curve_feedforward_threshold=0.001,
        max_steering=1.0,
    )
    reference_point = {"x": 0.5, "y": 10.0, "heading": 0.0, "curvature": 0.03}
    result = controller.compute_steering(
        current_heading=0.0,
        reference_point=reference_point,
        current_speed=5.0,
        return_metadata=True,
    )
    assert result["curve_feedforward_scale"] == 1.1
