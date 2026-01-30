import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


project_root = Path(__file__).resolve().parents[1]
analyzer_module = _load_module(
    "analyze_drive_overall",
    project_root / "tools/analyze/analyze_drive_overall.py",
)
summary_module = _load_module(
    "summary_analyzer",
    project_root / "tools/debug_visualizer/backend/summary_analyzer.py",
)

DriveAnalyzer = analyzer_module.DriveAnalyzer
analyze_recording_summary = summary_module.analyze_recording_summary


def _write_basic_speed_recording(path: Path) -> None:
    timestamps = np.array([0, 1, 2, 3, 4], dtype=np.float64)
    speed = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    ref_speed = np.array([0, 1, 1, 1, 1], dtype=np.float32)
    steering = np.zeros_like(speed, dtype=np.float32)

    with h5py.File(path, "w") as f:
        f.create_dataset("vehicle_state/timestamp", data=timestamps)
        f.create_dataset("vehicle_state/speed", data=speed)
        f.create_dataset("control/steering", data=steering)
        f.create_dataset("trajectory/reference_point_velocity", data=ref_speed)


def test_drive_analyzer_speed_and_comfort_metrics(tmp_path: Path) -> None:
    recording = tmp_path / "speed_metrics.h5"
    _write_basic_speed_recording(recording)

    analyzer = DriveAnalyzer(recording)
    assert analyzer.load_data()
    metrics = analyzer.calculate_metrics()

    assert metrics.speed_error_rmse == pytest.approx(np.sqrt(2.8), rel=1e-3)
    assert metrics.speed_error_mean == pytest.approx(1.2, rel=1e-3)
    assert metrics.speed_error_max == pytest.approx(3.0, rel=1e-3)
    assert metrics.speed_overspeed_rate == pytest.approx(60.0, rel=1e-3)

    assert metrics.acceleration_mean == pytest.approx(1.0, rel=1e-3)
    assert metrics.acceleration_max == pytest.approx(1.0, rel=1e-3)
    assert metrics.acceleration_p95 == pytest.approx(1.0, rel=1e-3)
    assert metrics.jerk_mean == pytest.approx(0.0, rel=1e-3)
    assert metrics.jerk_max == pytest.approx(0.0, rel=1e-3)
    assert metrics.jerk_p95 == pytest.approx(0.0, rel=1e-3)


def test_summary_analyzer_speed_control_metrics(tmp_path: Path) -> None:
    recording = tmp_path / "speed_summary.h5"
    _write_basic_speed_recording(recording)

    summary = analyze_recording_summary(recording)
    speed_control = summary["speed_control"]

    assert speed_control["speed_error_rmse"] == pytest.approx(np.sqrt(2.8), rel=1e-3)
    assert speed_control["speed_error_mean"] == pytest.approx(1.2, rel=1e-3)
    assert speed_control["speed_error_max"] == pytest.approx(3.0, rel=1e-3)
    assert speed_control["speed_overspeed_rate"] == pytest.approx(60.0, rel=1e-3)

    assert speed_control["acceleration_mean"] == pytest.approx(1.0, rel=1e-3)
    assert speed_control["acceleration_max"] == pytest.approx(1.0, rel=1e-3)
    assert speed_control["acceleration_p95"] == pytest.approx(1.0, rel=1e-3)
    assert speed_control["jerk_mean"] == pytest.approx(0.0, rel=1e-3)
    assert speed_control["jerk_max"] == pytest.approx(0.0, rel=1e-3)
    assert speed_control["jerk_p95"] == pytest.approx(0.0, rel=1e-3)
