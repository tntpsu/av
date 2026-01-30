import importlib.util
from pathlib import Path

import pytest


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_apply_speed_limits_with_curve_limit():
    project_root = Path(__file__).resolve().parents[1]
    av_module = _load_module("av_stack", project_root / "av_stack.py")

    speed = av_module.AVStack._apply_speed_limits(
        base_speed=8.0,
        speed_limit=12.0,
        path_curvature=0.1,  # 1/m
        max_lateral_accel=2.5,
        min_curve_speed=0.0,
    )
    assert speed == pytest.approx(5.0, rel=1e-3)


def test_apply_speed_limits_with_map_limit():
    project_root = Path(__file__).resolve().parents[1]
    av_module = _load_module("av_stack", project_root / "av_stack.py")

    speed = av_module.AVStack._apply_speed_limits(
        base_speed=12.0,
        speed_limit=8.0,
        path_curvature=0.0,
        max_lateral_accel=2.5,
        min_curve_speed=0.0,
    )
    assert speed == pytest.approx(8.0, rel=1e-6)


def test_apply_speed_limits_min_curve_speed():
    project_root = Path(__file__).resolve().parents[1]
    av_module = _load_module("av_stack", project_root / "av_stack.py")

    speed = av_module.AVStack._apply_speed_limits(
        base_speed=8.0,
        speed_limit=0.0,
        path_curvature=10.0,
        max_lateral_accel=2.5,
        min_curve_speed=2.0,
    )
    assert speed == pytest.approx(2.0, rel=1e-6)
