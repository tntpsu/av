"""Tests for locked reference curvature loading fallbacks."""

import importlib.util
from pathlib import Path

import h5py
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "analyze" / "replay_trajectory_locked.py"
SPEC = importlib.util.spec_from_file_location("replay_trajectory_locked_module", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
_load_locked_reference = MODULE._load_locked_reference


def _create_minimal_lock_file(path: Path) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("trajectory/reference_point_x", data=[1.0])
        f.create_dataset("trajectory/reference_point_y", data=[2.0])
        f.create_dataset("trajectory/reference_point_heading", data=[0.1])
        f.create_dataset("trajectory/reference_point_velocity", data=[6.0])


def test_locked_reference_prefers_reference_point_curvature(tmp_path: Path):
    lock_path = tmp_path / "lock.h5"
    _create_minimal_lock_file(lock_path)
    with h5py.File(lock_path, "a") as f:
        f.create_dataset("trajectory/reference_point_curvature", data=[0.021])
        f.create_dataset("ground_truth/path_curvature", data=[0.033])

    with h5py.File(lock_path, "r") as f:
        ref = _load_locked_reference(f, 0)

    assert ref is not None
    assert ref["curvature"] == pytest.approx(0.021)
    assert ref["curvature_source"] == "trajectory_reference_point_curvature"


def test_locked_reference_falls_back_to_ground_truth_curvature(tmp_path: Path):
    lock_path = tmp_path / "lock.h5"
    _create_minimal_lock_file(lock_path)
    with h5py.File(lock_path, "a") as f:
        f.create_dataset("ground_truth/path_curvature", data=[0.028])

    with h5py.File(lock_path, "r") as f:
        ref = _load_locked_reference(f, 0)

    assert ref is not None
    assert ref["curvature"] == pytest.approx(0.028)
    assert ref["curvature_source"] == "ground_truth_path_curvature"


def test_locked_reference_marks_missing_curvature_source(tmp_path: Path):
    lock_path = tmp_path / "lock.h5"
    _create_minimal_lock_file(lock_path)

    with h5py.File(lock_path, "r") as f:
        ref = _load_locked_reference(f, 0)

    assert ref is not None
    assert ref["curvature"] == pytest.approx(0.0)
    assert ref["curvature_source"] == "missing"
