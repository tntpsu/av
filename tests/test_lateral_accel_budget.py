"""One lateral-accel budget (T-SPEED-OWNER-CONSOLIDATION P1, 2026-10-11).

Five owners held five budgets (profile 0.05 g, comfort 0.20/2.5 = 0.08 g, preview 0.08 g, curve cap
0.20/0.26 g, bins 0.31 g). P1 makes profile + comfort + preview read `trajectory.lateral_accel_budget_g`
when enabled, leaves the curve cap on its legacy values, and changes nothing when disabled.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from control.speed_governor import G, build_speed_governor, unified_lateral_accel_budget_g  # noqa: E402

GOV = {"comfort_governor_max_lat_accel_g": 0.20, "curvature_calibration_scale": 2.5,
       "comfort_governor_min_speed": 3.0, "curve_cap_enabled": True, "curve_cap_estimator_enabled": True,
       "curve_cap_peak_lat_accel_g": 0.26, "curve_preview_enabled": True}


def _gov(enabled: bool, budget: float = 0.08):
    cfg = {"speed_governor": dict(GOV), "lateral_accel_budget_enabled": enabled, "lateral_accel_budget_g": budget}
    return build_speed_governor(cfg, {})


class TestHelper:
    def test_disabled_or_missing_is_none(self):
        assert unified_lateral_accel_budget_g({}) is None
        assert unified_lateral_accel_budget_g({"lateral_accel_budget_enabled": False, "lateral_accel_budget_g": 0.1}) is None
        assert unified_lateral_accel_budget_g({"lateral_accel_budget_enabled": True, "lateral_accel_budget_g": 0}) is None

    def test_enabled_returns_budget(self):
        assert unified_lateral_accel_budget_g({"lateral_accel_budget_enabled": True, "lateral_accel_budget_g": 0.12}) == 0.12


class TestComfortAndPreview:
    def test_legacy_effective_budget_is_0_08g(self):
        """0.20 g / scale 2.5: v = sqrt(0.08 g / kappa)."""
        g = _gov(False)
        assert g._compute_comfort_speed(0.01) == pytest.approx(math.sqrt(0.08 * G / 0.01), rel=1e-6)

    def test_unified_budget_replaces_scale(self):
        g = _gov(True, 0.08)
        assert g.config.curvature_calibration_scale == 1.0
        assert g._compute_comfort_speed(0.01) == pytest.approx(math.sqrt(0.08 * G / 0.01), rel=1e-6)
        assert _gov(True, 0.16)._compute_comfort_speed(0.01) == pytest.approx(math.sqrt(0.16 * G / 0.01), rel=1e-6)

    def test_first_ladder_step_is_neutral_for_comfort(self):
        """Step 1 (0.08 g) must leave the comfort governor exactly where legacy had it."""
        for k in (0.002, 0.01, 0.025):
            assert _gov(True, 0.08)._compute_comfort_speed(k) == pytest.approx(_gov(False)._compute_comfort_speed(k), rel=1e-9)


class TestCurveCapUnchanged:
    def test_cap_keeps_legacy_comfort_limit_when_budget_enabled(self):
        legacy, unified = _gov(False), _gov(True, 0.16)
        assert unified.config.curve_cap_comfort_lat_accel_g == pytest.approx(0.20)
        assert unified._curve_cap_comfort_g() == legacy._curve_cap_comfort_g() == pytest.approx(0.20)
        kw = dict(current_target=12.0, current_speed=10.0, curvature=0.012, preview_curvature=0.012,
                  curve_intent=0.8, curve_intent_state="COMMIT", curve_rise=0.0, curve_local_state="COMMIT",
                  curve_local_gate_weight=1.0, local_curve_reference_active=True)
        assert unified._compute_curve_cap_speed(**kw)[0] == pytest.approx(legacy._compute_curve_cap_speed(**kw)[0], rel=1e-9)


def test_base_config_carries_the_switch_and_a_ladder_value():
    import yaml
    t = yaml.safe_load((REPO_ROOT / "config" / "av_stack_config.yaml").read_text())["trajectory"]
    assert "lateral_accel_budget_enabled" in t and t["lateral_accel_budget_g"] >= 0.08
