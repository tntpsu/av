"""SpeedCompliance layer (T-SCORE-SPEED-COMPLIANCE, 2026-10-11).

Nothing scored whether the car did the job: hill_highway scored 99 at 60 % of its posted
limit and s_loop at 53 %. The seventh layer scores the report-only utilisation metric; it is
absent (six-layer weights) when a run has too few eligible frames, e.g. ACC following.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from speed_utilization import score_speed_compliance  # noqa: E402
import scoring_registry as reg  # noqa: E402
from conftest import golden_recording_path  # noqa: E402


def _util(median: float, under_pct: float, status: str = "ok") -> dict:
    return {"status": status, "vs_allowed_median": median, "under_utilised_pct": under_pct,
            "speed_median_mps": median * 12.0, "allowed_median_mps": 12.0, "eligible_pct": 80.0,
            "binding_cap_at_under_utilised_pct": {"velocity_profile": 100.0}}


def _find(d, key):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = _find(v, key)
            if r is not None:
                return r
    return None


class TestScoreFunction:
    def test_at_allowed_speed_is_free(self):
        assert score_speed_compliance(_util(0.97, 0.0))["score"] == 100.0
        assert score_speed_compliance(_util(reg.SPEED_COMPLIANCE_FREE_RATIO, 10.0))["score"] == 100.0

    def test_half_speed_is_red(self):
        """s_loop shape: median 0.53 of allowed, 96 % of eligible time under 0.70."""
        sc = score_speed_compliance(_util(0.53, 96.0))
        assert sc["score"] < 50.0
        names = [d["name"] for d in sc["deductions"]]
        assert "Speed vs allowed (median)" in names and "Time under-utilised" in names
        assert "velocity_profile" in sc["binding_cap"]

    def test_monotone_in_shortfall(self):
        scores = [score_speed_compliance(_util(m, 0.0))["score"] for m in (0.9, 0.8, 0.7, 0.6, 0.5)]
        assert scores == sorted(scores, reverse=True) and scores[0] == 100.0

    def test_soft_caps_keep_ranking(self):
        a = score_speed_compliance(_util(0.30, 100.0))["score"]
        b = score_speed_compliance(_util(0.10, 100.0))["score"]
        assert b < a and b > 0.0

    def test_absent_when_not_ok_or_none(self):
        assert score_speed_compliance(None) is None
        assert score_speed_compliance(_util(0.5, 50.0, status="insufficient_eligible_frames")) is None


@pytest.mark.parametrize("track_id,expect_layer_ge_95", [("highway_65", True), ("s_loop", False), ("hill_highway", False)])
def test_golden_recordings_carry_the_layer(track_id, expect_layer_ge_95):
    from drive_summary_core import analyze_recording_summary
    path = golden_recording_path(track_id)
    if path is None:
        pytest.skip("golden not on disk")
    s = analyze_recording_summary(path)
    layers = s["layer_scores"]
    assert "SpeedCompliance" in layers
    weights = _find(s, "layer_weights")
    assert weights["SpeedCompliance"] == pytest.approx(reg.SPEED_COMPLIANCE_WEIGHT)
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
    if expect_layer_ge_95:
        assert layers["SpeedCompliance"] >= 95.0
    else:
        assert layers["SpeedCompliance"] < 60.0, layers["SpeedCompliance"]
    assert _find(s, "speed_utilisation")["status"] == "ok"


def test_kill_switch_removes_the_layer(monkeypatch):
    import drive_summary_core as core
    monkeypatch.setattr(core, "SPEED_COMPLIANCE_LAYER_ENABLED", False)
    path = golden_recording_path("highway_65")
    if path is None:
        pytest.skip("golden not on disk")
    s = core.analyze_recording_summary(path)
    assert "SpeedCompliance" not in s["layer_scores"]
    weights = _find(s, "layer_weights")
    assert "SpeedCompliance" not in weights and sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
