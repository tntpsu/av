"""reference_point['curvature'] sign source (2026-10-17, T-LATERAL-DUAL-CURVATURE-SIGN).

Before 2026-10-17 the orchestrator signed reference_point['curvature'] from the planner's own reference curvature first.
That value is UNSIGNED (trajectory/inference.py takes abs), so the sign resolved to +1 on every recording — sweeping_highway
(all left curves) carried +0.0033 on every arc frame — and every consumer (LMPC camera-frame kappa_ref, Stanley's legacy
FF source, the local-curve-reference direction) saw |κ|. Unity's GetPathCurvature was ALSO unsigned (cross product dotted
with the wrong up axis; fixed the same day, +left after the fix). 'gt_map' consults the signed GT/map curvature first.
"""
from __future__ import annotations

import pytest

from av_stack.orchestrator import select_reference_curvature_sign as sel


class TestLegacyPlannerFirst:
    def test_unsigned_planner_value_pins_the_sign_positive_even_on_a_left_curve(self):
        # recorded sweeping_highway frame: planner |κ| 0.0033, GT (fixed Unity, +left) +0.0033 — legacy still says +1
        assert sel(0.0033, -0.02, +0.0033, 'planner') == 1.0
        # and on a right curve the legacy chain is +1 as well (the defect: no information in the sign)
        assert sel(0.0050, +0.03, -0.0050, 'planner') == 1.0

    def test_legacy_falls_back_to_heading_then_gt_then_plus_one(self):
        assert sel(0.0, -0.02, +0.01, 'planner') == -1.0
        assert sel(0.0, 0.0, -0.01, 'planner') == -1.0
        assert sel(0.0, 0.0, 0.0, 'planner') == 1.0


class TestGtMapFirst:
    def test_signed_gt_curvature_decides(self):
        assert sel(0.0033, -0.02, +0.0033, 'gt_map') == 1.0      # left curve → +left
        assert sel(0.0050, +0.03, -0.0050, 'gt_map') == -1.0     # right curve → negative

    def test_falls_back_to_heading_then_planner_when_gt_is_zero_or_missing(self):
        assert sel(0.0033, -0.02, 0.0, 'gt_map') == -1.0
        assert sel(0.0033, 0.0, None, 'gt_map') == 1.0
        assert sel(None, None, None, 'gt_map') == 1.0

    def test_non_numeric_inputs_are_ignored(self):
        assert sel('x', float('nan'), -0.004, 'gt_map') == -1.0
