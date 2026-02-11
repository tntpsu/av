from __future__ import annotations

from trajectory.utils import select_curvature_bin_limits


def test_select_curvature_bin_limits_uses_defaults_when_no_bins() -> None:
    max_lat, min_speed = select_curvature_bin_limits(0.02, None, 1.3, 3.2)
    assert max_lat == 1.3
    assert min_speed == 3.2


def test_select_curvature_bin_limits_applies_matching_bin() -> None:
    bins = [
        {"min_curvature": 0.0, "max_curvature": 0.01, "max_lateral_accel": 1.3, "min_curve_speed": 3.2},
        {"min_curvature": 0.01, "max_curvature": 0.02, "max_lateral_accel": 1.25, "min_curve_speed": 3.1},
        {"min_curvature": 0.02, "max_curvature": 10.0, "max_lateral_accel": 1.2, "min_curve_speed": 3.0},
    ]
    max_lat, min_speed = select_curvature_bin_limits(0.015, bins, 1.3, 3.2)
    assert max_lat == 1.25
    assert min_speed == 3.1


def test_select_curvature_bin_limits_last_bin() -> None:
    bins = [
        {"min_curvature": 0.0, "max_curvature": 0.01, "max_lateral_accel": 1.3, "min_curve_speed": 3.2},
        {"min_curvature": 0.01, "max_curvature": 0.02, "max_lateral_accel": 1.25, "min_curve_speed": 3.1},
        {"min_curvature": 0.02, "max_curvature": 10.0, "max_lateral_accel": 1.2, "min_curve_speed": 3.0},
    ]
    max_lat, min_speed = select_curvature_bin_limits(0.5, bins, 1.3, 3.2)
    assert max_lat == 1.2
    assert min_speed == 3.0
