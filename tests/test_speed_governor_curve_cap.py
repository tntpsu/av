"""Curve-cap latch on straight tracks (T-GOVERNOR-CURVE-CAP-LATCH, 2026-10-07).

Reproduces the H8/H2/H4 signature from recording_20261005_040914.h5: κ_now 0, preview κ
0.002 (highway_65's R500, exactly curve_cap_curvature_min), intent STRAIGHT, target 12.0 →
legacy cap 11.6 m/s (target − margin) for the whole run although the curve's feasible speed
is ~36 m/s. With `curve_cap_only_when_binding` the cap stays off until the curve binds.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from control.speed_governor import build_speed_governor  # noqa: E402

BASE_GOV = {
    "enabled": True, "comfort_governor_max_lat_accel_g": 0.2, "comfort_governor_min_speed": 3.0,
    "curvature_calibration_scale": 2.5, "curve_cap_enabled": True, "curve_cap_shadow_mode": False,
    "curve_cap_estimator_enabled": True, "curve_cap_hysteresis_enabled": True,
    "curve_cap_entry_intent_min": 0.35, "curve_cap_commit_intent_min": 0.55, "curve_cap_max_decel_mps2": 1.6,
    "curve_cap_hysteresis_mps": 0.35, "curve_cap_min_speed_mps": 3.0, "curve_cap_margin_mps": 0.4,
    "curve_cap_curvature_min": 0.002, "curve_cap_rise_min": 0.0005, "curve_cap_peak_lat_accel_g": 0.26,
    "curve_cap_use_preview_curvature": True, "curve_preview_lookahead_scale": 1.6,
}


def _gov(**over):
    cfg = dict(BASE_GOV); cfg.update(over)
    return build_speed_governor({"speed_governor": cfg}, {})


def _cap(gov, *, target=12.0, speed=11.3, k_now=0.0, k_prev=0.002, intent=0.285, state="STRAIGHT", rise=0.0):
    return gov._compute_curve_cap_speed(
        current_target=target, current_speed=speed, curvature=k_now, preview_curvature=k_prev,
        curve_intent=intent, curve_intent_state=state, curve_rise=rise, curve_local_state="STRAIGHT",
        curve_local_gate_weight=0.0, local_curve_reference_active=False,
    )


class TestLegacyLatch:
    def test_h8_signature_legacy_caps_target_minus_margin_and_is_labelled_map_preview(self):
        speed, active, reason, margin = _cap(_gov())
        assert active and speed == pytest.approx(11.6) and reason == "map_preview" and margin == pytest.approx(0.4)

    def test_legacy_cap_tracks_target_not_the_curve(self):
        """The cap is target − 0.4 whatever the target: the 11.6 / 14.6 the nightlies saw were
        targets 12.0 / 15.0, not governor noise."""
        assert _cap(_gov(), target=15.0)[0] == pytest.approx(14.6)

    def test_legacy_curve_speed_is_far_above_target(self):
        v_curve = math.sqrt(0.26 * 9.81 / 0.002)   # peak-g feasibility at R500 ≈ 35.7 m/s
        assert v_curve > 30.0
        assert _cap(_gov())[1] is True             # …and the cap is active anyway


class TestOnlyWhenBinding:
    def test_h8_signature_no_cap_when_curve_allows_more_than_target(self):
        speed, active, reason, margin = _cap(_gov(curve_cap_only_when_binding=True))
        assert not active and speed is None and reason == "not_binding" and margin == 0.0

    def test_still_caps_when_the_curve_binds(self):
        """R33 ahead (κ 0.03): feasible ≈ 9.2 m/s, entry speed ≈ 12.1 → cap 11.7 < target 12.0."""
        legacy = _cap(_gov(), k_prev=0.03)
        flagged = _cap(_gov(curve_cap_only_when_binding=True), k_prev=0.03)
        assert legacy[1] and flagged[1]
        assert flagged[0] == pytest.approx(legacy[0]) and flagged[0] < 12.0

    def test_low_curvature_still_inactive(self):
        assert _cap(_gov(curve_cap_only_when_binding=True), k_prev=0.001)[2] == "low_curvature"

    def test_default_is_legacy(self):
        assert _gov().config.curve_cap_only_when_binding is False
