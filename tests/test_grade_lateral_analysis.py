"""Tests for grade_lateral_v1 analytics (synthetic HDF5)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_grade_lateral_module():
    mod_path = REPO_ROOT / "tools" / "grade_lateral_analysis.py"
    spec = importlib.util.spec_from_file_location("grade_lateral_analysis", mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["grade_lateral_analysis"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _write_minimal_recording(path: Path, n: int = 120) -> None:
    """Recording with stepped road_grade and simple lateral/steering series."""
    ts = np.linspace(0.0, float(n - 1) / 30.0, n, dtype=np.float64)
    grade = np.zeros(n, dtype=np.float32)
    grade[: n // 3] = -0.05
    grade[n // 3 : 2 * n // 3] = 0.0
    grade[2 * n // 3 :] = 0.06
    lat = np.full(n, 0.1, dtype=np.float32)
    steer = np.zeros(n, dtype=np.float32)
    steer[::10] = 0.2
    steer[1::10] = -0.2

    with h5py.File(path, "w") as f:
        f.create_dataset("vehicle/timestamps", data=ts.astype(np.float64))
        f.create_dataset("vehicle/road_grade", data=grade)
        f.create_dataset("control/timestamps", data=ts.astype(np.float64))
        f.create_dataset("control/steering", data=steer)
        f.create_dataset("control/lateral_error", data=lat)


def test_grade_lateral_v1_bins_and_schema(tmp_path: Path) -> None:
    gla = _load_grade_lateral_module()
    SCHEMA_VERSION = gla.SCHEMA_VERSION
    analyze_grade_lateral = gla.analyze_grade_lateral

    path = tmp_path / "grade_lat_synth.h5"
    _write_minimal_recording(path, n=120)

    out = analyze_grade_lateral(
        path,
        pre_failure_only=False,
        grade_threshold=0.02,
    )
    assert out.get("schema_version") == SCHEMA_VERSION
    assert "error" not in out
    bins = out["bins"]
    assert set(bins.keys()) >= {"down", "flat", "up"}
    assert bins["down"]["frame_count"] == 40
    assert bins["flat"]["frame_count"] == 40
    assert bins["up"]["frame_count"] == 40
    assert "correlation_summary" in out
    assert "lateral_error_abs_mean_m" in bins["down"]


def test_recorder_writes_lateral_grade_fields(tmp_path: Path) -> None:
    from data.formats.data_format import CameraFrame, ControlCommand, RecordingFrame
    from data.recorder import DataRecorder

    recorder = DataRecorder(str(tmp_path), recording_name="grade_lat_telemetry")
    try:
        frame = RecordingFrame(
            timestamp=0.0,
            frame_id=0,
            camera_frame=CameraFrame(
                image=np.zeros((1, 1, 3), dtype=np.uint8),
                timestamp=0.0,
                frame_id=0,
            ),
            vehicle_state=None,
            control_command=ControlCommand(
                timestamp=0.0,
                steering=0.0,
                throttle=0.0,
                brake=0.0,
                lateral_grade_damping=0.12,
                lateral_error_smoothing_alpha_effective=0.35,
            ),
            perception_output=None,
            trajectory_output=None,
            unity_feedback=None,
        )
        recorder._write_control_commands([frame])
        recorder.h5_file.flush()
    finally:
        recorder.close()

    with h5py.File(tmp_path / "grade_lat_telemetry.h5", "r") as h5_file:
        assert float(h5_file["control/lateral_grade_damping"][0]) == pytest.approx(0.12)
        assert float(h5_file["control/lateral_error_smoothing_alpha_effective"][0]) == pytest.approx(
            0.35
        )
