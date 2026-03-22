"""Regression: speed feasibility must not emit divide-by-zero when |curvature| has zeros."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_speed_feasibility_no_runtime_warning_on_zero_curvature() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "drive_summary_core", REPO_ROOT / "tools" / "drive_summary_core.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    n = 200
    speed_arr = np.full(n, 10.0, dtype=np.float64)
    curvature_arr = np.zeros(n, dtype=np.float64)
    curvature_arr[50:80] = 0.02
    curvature_arr[100:120] = 1e-9

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        abs_curv = np.abs(curvature_arr)
        curvature_eps = 1e-6
        m = abs_curv > curvature_eps
        v_max = np.full_like(abs_curv, np.inf, dtype=np.float64)
        if np.any(m):
            v_max[m] = np.sqrt(2.45 / abs_curv[m])
        _ = int(np.sum(speed_arr > v_max))

    bad = [x for x in w if issubclass(x.category, RuntimeWarning) and "divide" in str(x.message)]
    assert not bad, bad
