"""Tests for tools/nightly/report_render.py — the nightly email renderer.

Synthetic report texts mirror the structures the PROMPTs contract (GATE line,
table rows, `Changes from Night-N:` block, per-failure `Root cause:` / `Action:`
lines). A smoke test also renders whatever real reports are on disk.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.nightly import report_render as rr  # noqa: E402

ACC = """ACC SCENARIO SWEEP — 2026-09-24 (Night 43)
══════════════════════════════════════════════════════════════════════════════
Generated: 2026-09-24T04:20 EDT
Mode: --quick + 3 fresh /e2e runs (H3, H6, H8)
Scoring frame: bumper-gap (offset 4.43m); lead_collision_detected; post-09-22 scorer
══════════════════════════════════════════════════════════════════════════════

Scenario                              Base          Rec age  Verdict  Score    Sub-scores
─────────────────────────────────────────────────────────────────────────────────────────
A1  (autobahn_a1_steady)              autobahn_30   0.99d     FAIL     n/a      ACC <5% (DETECTION_LOSS 95.9%)
H6  (highway_h6_close_gap)            highway_65    fresh     PASS     99.9     S100 T100 B100
G2  (hill_g2_stop_on_grade)           hill_highway  1.15d     PASS     94.7     S100 T93 B84
─────────────────────────────────────────────────────────────────────────────────────────
GATE: FAIL
pass=2  fail=1  warn=0  skip=0  amb=0  total=3

Changes from Night-42 (2026-09-23):
  H8: PASS→FAIL  Fresh run: detection 86%→61.8%
  H3: SKIP→PASS  7d boundary expired

FAILURES (1)
────────────────────────────────────────────────────────────────────────────
A1 — autobahn_a1_steady [recording_20260923_040935.h5, 0.99d]
  Recording: fresh Sep 23 (post-09-22 build)
  Detection rate: 4% [FAIL ≥95%]
  Root cause: Lead at 20 m/s vs ego 12 m/s → lead exits radar range immediately.
    Ego can never match lead speed → permanent DETECTION_LOSS.
  Action: Human — lower A1 lead speed ≤ 12 m/s in scenario YAML, OR accept
    this is a future scenario pending NMPC/high-speed work.
  11th consecutive FAIL.

2026-09-24 ACC_SWEEP gate=FAIL pass=2 fail=1 warn=0 skip=0 amb=0 total=3
"""

SWEEP = """CROSS-TRACK SWEEP — 2026-09-24 (night 118)
═══════════════════════════════════════════════════════════════
⚠  FROZEN RECORDINGS — newest recording is 2026-08-15 (39 days old).

Track            Baseline    Now     Delta    Traj    Util   Gate
──────────────── ────────── ────── ──────── ─────── ────── ─────────────
s_loop           99.1        99.0    -0.1    96.4    0.49   PASS
hill_highway     97.6        97.6     0.0    91.5    0.55   FAIL (Traj<95, worst)
═══════════════════════════════════════════════════════════════
GATE: FAIL

Regressions (> 2 pts drop): none
New e-stops: none
Flags (> 1 pt drop): none

Standing failures (PP curve-tracking ceiling, unchanged since night 1):
  hill_highway: Trajectory 91.5 (floor worst) — apex cutting C2

Next action: schedule fresh daytime Unity lateral sweep before any diagnosis.
Do NOT tune — FAIL is architectural (PP→LMPC transition).
"""

NIGHTLY = """NIGHTLY TEST REPORT — 2026-09-24
==================================
Total: 2350
Passed: 2345
Fixed: 0
Real breaks (unfixed): 1
Flaky: 1

## REAL BREAKS (need human review)
  [REAL_BREAK] tests/test_x.py::test_y
    — something changed in the scorer

## FLAKY
  [FLAKY] tests/test_mpc_controller.py::test_lateral_error_correction
    — xdist _last_steering state contamination
"""


class TestAccSweep:
    def test_parses_gate_rows_changes_and_actions(self):
        r = rr.parse_acc_sweep(ACC, "2026-09-24")
        assert r.gate == "FAIL"
        assert dict(r.kpis)["FAIL"] == "1" and dict(r.kpis)["Night"] == "43"
        assert [row[0] for row in r.rows] == ["A1 autobahn_a1_steady", "H6 highway_h6_close_gap", "G2 hill_g2_stop_on_grade"]
        assert r.rows[2][r.verdict_col] == "PASS"
        assert r.changes[0].startswith("H8: PASS→FAIL")
        assert len(r.cards) == 1
        assert r.cards[0].root_cause.startswith("Lead at 20 m/s") and "permanent DETECTION_LOSS" in r.cards[0].root_cause
        assert r.next_steps and r.next_steps[0].startswith("A1: Human — lower A1 lead speed")
        assert any(n.startswith("Scoring:") for n in r.notes)

    def test_render_html_has_badge_next_steps_first_and_table(self):
        r = rr.parse_acc_sweep(ACC, "2026-09-24")
        html = rr.render_html(r, "av acc-sweep 2026-09-24: gate=FAIL", "preflight dt_p95: 153ms (threshold ≤250ms)", {"Preflight": "153ms"})
        assert "GATE FAIL" in html
        assert html.index("Next steps") < html.index("Results")
        assert "hill_g2_stop_on_grade" in html and "Raw report" in html and "Log tail" in html
        assert "<script" not in html

    def test_render_text_is_readable(self):
        r = rr.parse_acc_sweep(ACC, "2026-09-24")
        txt = rr.render_text(r, "subject", "")
        assert txt.startswith("subject") and "NEXT STEPS" in txt and "GATE: FAIL" in txt


class TestSweep:
    def test_parses_util_column_and_next_action(self):
        r = rr.parse_sweep(SWEEP, "2026-09-24")
        assert r.gate == "FAIL"
        assert r.columns == ["Track", "Baseline", "Now", "Delta", "Trajectory", "Util", "Gate"]
        assert r.rows[0][:6] == ["s_loop", "99.1", "99.0", "-0.1", "96.4", "0.49"]
        assert r.rows[1][-1].startswith("FAIL")
        assert any(n.startswith("⚠") for n in r.notes)
        assert any(s.startswith("Next action:") for s in r.next_steps)
        assert dict(r.kpis)["Regressions"] == "none"

    def test_table_without_util_column_still_parses(self):
        txt = SWEEP.replace("    Util   Gate", "    Gate").replace("   0.49   PASS", "   PASS").replace("   0.55   FAIL", "   FAIL")
        r = rr.parse_sweep(txt, "2026-09-24")
        assert "Util" not in r.columns and len(r.rows) == 2


class TestNightly:
    def test_real_break_becomes_next_step_and_gate_fail(self):
        r = rr.parse_nightly(NIGHTLY, "step6_done delivery=local_only", "2026-09-24")
        assert r.gate == "FAIL"
        assert dict(r.kpis)["Delivery"] == "local_only"
        assert any(row[0] == "FAIL" and "test_x.py" in row[1] for row in r.rows)
        assert any("REAL_BREAK" in s for s in r.next_steps)

    def test_clean_night_is_pass(self):
        r = rr.parse_nightly(NIGHTLY.replace("Real breaks (unfixed): 1", "Real breaks (unfixed): 0"), "", "d")
        assert r.gate == "PASS"


class TestRobustness:
    def test_missing_report_still_renders(self, tmp_path):
        text, html = rr.render("acc-sweep", "subj", "log tail here", reports_dir=tmp_path)
        assert "report file missing" in html and "log tail here" in html and "subj" in text

    def test_parse_error_is_reported_not_raised(self, monkeypatch, tmp_path):
        (tmp_path / "acc_sweep_report.txt").write_text("GATE: FAIL\n")
        monkeypatch.setattr(rr, "parse_acc_sweep", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("boom")))
        r = rr.load_report("acc-sweep", tmp_path)
        assert "ValueError" in r.parse_error
        html = rr.render_html(r, "s", "", {})
        assert "Report parse failed" in html

    def test_meta_extracted_from_log_tail(self, tmp_path):
        (tmp_path / "sweep_report.txt").write_text(SWEEP)
        text, html = rr.render("sweep", "s", "--- claude -p (model=sonnet-5, budget=$10) ---\npreflight dt_p95: 153ms (threshold ≤250ms)", reports_dir=tmp_path)
        assert "sonnet-5" in html and "153ms" in html

    @pytest.mark.parametrize("job", ["nightly", "sweep", "acc-sweep", "process-health"])
    def test_real_reports_on_disk_render(self, job):
        if not rr.REPORTS.exists():
            pytest.skip("no data/reports")
        text, html = rr.render(job, f"smoke {job}", "")
        assert "<html" in html and len(text) > 20
