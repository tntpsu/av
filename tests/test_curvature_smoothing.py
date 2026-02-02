from __future__ import annotations

import math

import pytest

from trajectory.utils import curvature_smoothing_alpha, smooth_curvature_distance


def test_curvature_smoothing_alpha_distance_window() -> None:
    alpha = curvature_smoothing_alpha(distance_m=10.0, window_m=10.0)
    assert alpha == pytest.approx(1.0 - math.exp(-1.0), rel=1e-6)


def test_smooth_curvature_distance_step_response() -> None:
    curvature = [0.0, 1.0]
    speed = [10.0, 10.0]
    timestamps = [0.0, 1.0]
    smoothed = smooth_curvature_distance(curvature, speed, timestamps, window_m=10.0, min_speed=2.0)
    expected_alpha = 1.0 - math.exp(-1.0)
    assert smoothed[0] == pytest.approx(0.0, rel=1e-6)
    assert smoothed[1] == pytest.approx(expected_alpha, rel=1e-6)


def test_smooth_curvature_distance_constant_signal() -> None:
    curvature = [0.2, 0.2, 0.2]
    speed = [5.0, 5.0, 5.0]
    timestamps = [0.0, 0.5, 1.0]
    smoothed = smooth_curvature_distance(curvature, speed, timestamps, window_m=12.0, min_speed=2.0)
    assert smoothed == pytest.approx(curvature, rel=1e-6)


def test_smooth_curvature_distance_window_disabled() -> None:
    curvature = [0.0, 1.0, -1.0]
    speed = [3.0, 3.0, 3.0]
    timestamps = [0.0, 1.0, 2.0]
    smoothed = smooth_curvature_distance(curvature, speed, timestamps, window_m=0.0, min_speed=2.0)
    assert smoothed == pytest.approx(curvature, rel=1e-6)
