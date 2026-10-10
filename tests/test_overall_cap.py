"""Overall-score critical-layer cap (T-METRIC-UNCAP, 2026-10-10).

The legacy step cap (79 if Safety/Trajectory yellow, 59 if red) made s_loop's overall flip
79.0 <-> 94.4 on a 0.8-pt Trajectory move with identical driving. The continuous cap keeps
the bands and removes the cliff.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from drive_summary_core import critical_layer_cap  # noqa: E402
import scoring_registry as reg  # noqa: E402


def test_registry_selects_continuous_mode():
    assert reg.OVERALL_CRITICAL_CAP_MODE == "continuous"


def test_step_mode_reproduces_legacy_values():
    assert critical_layer_cap(85.0, "step") == 100.0
    assert critical_layer_cap(79.6, "step") == 79.0      # the s_loop golden
    assert critical_layer_cap(50.2, "step") == 59.0      # the hairpin golden


def test_continuous_is_continuous_at_both_band_edges():
    assert critical_layer_cap(80.0, "continuous") == 100.0
    assert critical_layer_cap(79.999, "continuous") == pytest.approx(100.0, abs=0.01)
    assert critical_layer_cap(60.0, "continuous") == pytest.approx(59.0)
    assert critical_layer_cap(59.999, "continuous") == pytest.approx(59.0, abs=0.01)
    assert critical_layer_cap(0.0, "continuous") == 0.0


def test_continuous_is_monotone_and_ranks_inside_the_bands():
    xs = [0, 10, 30, 59, 60, 61, 70, 79, 79.6, 80, 90, 100]
    ys = [critical_layer_cap(x, "continuous") for x in xs]
    assert ys == sorted(ys)
    assert critical_layer_cap(79.6, "continuous") > critical_layer_cap(70.0, "continuous") > critical_layer_cap(61.0, "continuous")


def test_no_cliff_on_a_sub_point_move():
    """0.8 pt of Trajectory may move the cap by at most ~2 pts, not 15."""
    assert critical_layer_cap(80.4, "continuous") - critical_layer_cap(79.6, "continuous") < 2.0
