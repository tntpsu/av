"""Tests for flat-bin high-|lat| frame range helper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_flat_focus():
    mod_path = REPO_ROOT / "tools" / "analyze_grade_lateral_flat_focus.py"
    spec = importlib.util.spec_from_file_location("analyze_grade_lateral_flat_focus", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_analyze_flat_focus_ranges(tmp_path: Path) -> None:
    mod = _load_flat_focus()
    n = 60
    lat = np.zeros(n, dtype=np.float32)
    lat[40:50] = 0.5
    grade = np.zeros(n, dtype=np.float32)
    grade[:] = 0.0
    path = tmp_path / "t.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("control/lateral_error", data=lat)
        f.create_dataset("vehicle/road_grade", data=grade)

    out = mod.analyze_flat_focus(path, abs_focus_m=0.1, top_n=5)
    assert "error" not in out
    assert any(r["start_frame"] == 40 and r["end_frame"] == 49 for r in out["flat_high_lateral_ranges"])
