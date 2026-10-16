"""Trajectory heading-penalty source contract (2026-10-15, sixth metric-frame defect).

The Trajectory layer's heading penalty (up to −20) scored `control/heading_error` — Pure Pursuit's pursuit angle to its
lookahead point. On a tight curve that angle is geometry (~atan(Ld·κ)), not error: a hairpin_15 Stanley run tracking to
0.089 m at-car RMSE read 17.98° (vs a true vehicle heading error of 2.95°) and took the full −20 on every run, pinning
Trajectory at 75.4. Scoring a controller-internal signal also makes the score depend on WHICH controller runs.
`scoring_registry.TRAJECTORY_HEADING_SOURCE = "gt_path_tangent"` scores vehicle/heading_delta_deg instead. The Pure
Pursuit goldens are unchanged: their true heading RMSE (hairpin 3.47°, s_loop 1.34°) is below the 10° floor, as their
pursuit angles were.
"""
from __future__ import annotations

import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
from drive_summary_core import scored_heading_rmse  # noqa: E402
import scoring_registry  # noqa: E402

PURSUIT = np.full(500, np.radians(17.98))      # recorded hairpin arm-C pursuit-angle RMSE
TRUE_DEG = np.full(500, 2.95)                  # recorded true heading RMSE (degrees)


def _penalty(rmse_rad):
    return min(20, max(0.0, np.degrees(rmse_rad) - scoring_registry.HEADING_PENALTY_FLOOR_DEG) * 2.5)


class TestSource:
    def test_legacy_scores_the_pursuit_angle_and_takes_the_full_penalty(self):
        r, src = scored_heading_rmse('controller_heading_error', PURSUIT, TRUE_DEG)
        assert src == 'controller_heading_error' and _penalty(r) == pytest.approx(19.95, abs=0.01)

    def test_gt_path_tangent_scores_the_vehicle_heading_and_takes_none(self):
        r, src = scored_heading_rmse('gt_path_tangent', PURSUIT, TRUE_DEG)
        assert src == 'gt_path_tangent' and r == pytest.approx(np.radians(2.95)) and _penalty(r) == 0.0

    def test_gt_source_converts_degrees_and_ignores_nonfinite(self):
        r, _ = scored_heading_rmse('gt_path_tangent', None, np.array([3.0, np.nan, 3.0]))
        assert r == pytest.approx(np.radians(3.0))

    def test_falls_back_to_legacy_when_the_field_is_absent(self):
        r, src = scored_heading_rmse('gt_path_tangent', PURSUIT, None)
        assert src == 'controller_heading_error' and r == pytest.approx(np.radians(17.98))

    def test_registry_default_is_gt_path_tangent(self):
        assert scoring_registry.TRAJECTORY_HEADING_SOURCE == 'gt_path_tangent'
