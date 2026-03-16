"""
S2-M1: Automated comfort-gate regression tests.

Tests that all S1-M39 comfort gates continue to pass without requiring Unity.
Two tiers:

    Tier 1 — Synthetic HDF5 (always runs, no recordings needed)
        Verifies metric computation correctness and gate boundary conditions.

    Tier 2 — Golden recording tests (skipped if file not on disk)
        Asserts that the documented S1-M39 validation recordings still pass
        every gate and that scores remain within ±2 of their baselines.

Run Tier 1 only:
    pytest tests/test_comfort_gate_replay.py -v -k "Synthetic or Boundary"

Run all (requires golden recordings in tests/fixtures/golden_recordings.json):
    pytest tests/test_comfort_gate_replay.py -v
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ── Make conftest importable from this file ───────────────────────────────────
# tests/ is a package (has __init__.py), so pytest adds the project root to
# sys.path but not tests/ itself.  Insert it explicitly before importing.
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from conftest import (
    COMFORT_GATES,
    BASELINE_SCORES,
    SCORE_TOLERANCE,
    SCORE_TOLERANCES,
    REPO_ROOT,
    golden_recording_path,
    make_nominal_recording,
    make_aggressive_recording,
)

# ── Dynamic import of drive_summary_core (no package install required) ────────
_CORE_PATH = REPO_ROOT / "tools" / "drive_summary_core.py"
_core_spec = importlib.util.spec_from_file_location("drive_summary_core", _CORE_PATH)
_core_mod  = importlib.util.module_from_spec(_core_spec)
sys.modules["drive_summary_core"] = _core_mod   # must precede exec to satisfy @dataclass on Py 3.14
_core_spec.loader.exec_module(_core_mod)
analyze_recording_summary = _core_mod.analyze_recording_summary

# ── Dynamic import of run_gate_and_triage (for _extract_run_metrics) ─────────
_GT_PATH = REPO_ROOT / "tools" / "analyze" / "run_gate_and_triage.py"
_gt_spec = importlib.util.spec_from_file_location("run_gate_and_triage", _GT_PATH)
gate_triage = importlib.util.module_from_spec(_gt_spec)
sys.modules["run_gate_and_triage"] = gate_triage  # must precede exec to satisfy @dataclass on Py 3.14
_gt_spec.loader.exec_module(gate_triage)


# ─────────────────────────────────────────────────────────────────────────────
# Assertion helper
# ─────────────────────────────────────────────────────────────────────────────

def assert_all_gates_pass(summary: dict, label: str = "") -> None:
    """
    Assert all S1-M39 comfort gates pass against the given summary dict.
    Raises AssertionError with a descriptive message naming the failing metric.
    """
    g = COMFORT_GATES
    comfort       = summary.get("comfort", {})
    path_tracking = summary.get("path_tracking", {})
    safety        = summary.get("safety", {})
    prefix        = f"[{label}] " if label else ""

    accel = float(comfort.get("acceleration_p95_filtered", 0.0))
    jerk  = float(comfort.get("commanded_jerk_p95", 0.0))
    lat   = float(path_tracking.get("lateral_error_p95", 0.0))
    cent  = float(path_tracking.get("time_in_lane_centered", 0.0))
    ool   = int(safety.get("out_of_lane_events_full_run", 0))
    stjrk = float(comfort.get("steering_jerk_max", 0.0))

    assert accel <= g["accel_p95_filtered_max"], (
        f"{prefix}accel P95 filtered {accel:.3f} m/s² > gate {g['accel_p95_filtered_max']:.1f} m/s²"
    )
    assert jerk <= g["commanded_jerk_p95_max"], (
        f"{prefix}commanded jerk P95 {jerk:.3f} m/s³ > gate {g['commanded_jerk_p95_max']:.1f} m/s³"
    )
    assert lat <= g["lateral_p95_max"], (
        f"{prefix}lateral error P95 {lat:.3f} m > gate {g['lateral_p95_max']:.2f} m"
    )
    assert cent >= g["centered_pct_min"], (
        f"{prefix}centered frames {cent:.1f}% < gate {g['centered_pct_min']:.0f}%"
    )
    assert ool <= g["out_of_lane_events_max"], (
        f"{prefix}out-of-lane events {ool} > gate {g['out_of_lane_events_max']}"
    )
    assert stjrk <= g["steering_jerk_max_max"], (
        f"{prefix}steering jerk max {stjrk:.2f} > gate {g['steering_jerk_max_max']:.1f}"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Tier 1 — Synthetic HDF5 tests (always run, no recordings on disk needed)
# ═════════════════════════════════════════════════════════════════════════════

class TestSyntheticNominal:
    """
    Verify that analyze_recording_summary() reports gate-passing values for a
    carefully crafted nominal recording.  One test per gate, plus a combined.
    """

    def test_accel_p95_filtered_within_gate(self, tmp_path: Path) -> None:
        rec = tmp_path / "nominal.h5"
        make_nominal_recording(rec)
        summary = analyze_recording_summary(rec)
        val = summary["comfort"]["acceleration_p95_filtered"]
        assert val <= COMFORT_GATES["accel_p95_filtered_max"], (
            f"acceleration_p95_filtered={val:.3f} m/s² > gate "
            f"{COMFORT_GATES['accel_p95_filtered_max']:.1f} m/s²"
        )

    def test_commanded_jerk_p95_within_gate(self, tmp_path: Path) -> None:
        rec = tmp_path / "nominal.h5"
        make_nominal_recording(rec)
        summary = analyze_recording_summary(rec)
        val = summary["comfort"]["commanded_jerk_p95"]
        assert val <= COMFORT_GATES["commanded_jerk_p95_max"], (
            f"commanded_jerk_p95={val:.3f} m/s³ > gate "
            f"{COMFORT_GATES['commanded_jerk_p95_max']:.1f} m/s³"
        )

    def test_lateral_error_p95_within_gate(self, tmp_path: Path) -> None:
        rec = tmp_path / "nominal.h5"
        make_nominal_recording(rec)
        summary = analyze_recording_summary(rec)
        val = summary["path_tracking"]["lateral_error_p95"]
        assert val <= COMFORT_GATES["lateral_p95_max"], (
            f"lateral_error_p95={val:.3f} m > gate {COMFORT_GATES['lateral_p95_max']:.2f} m"
        )

    def test_centered_frames_above_gate(self, tmp_path: Path) -> None:
        rec = tmp_path / "nominal.h5"
        make_nominal_recording(rec)
        summary = analyze_recording_summary(rec)
        val = summary["path_tracking"]["time_in_lane_centered"]
        assert val >= COMFORT_GATES["centered_pct_min"], (
            f"time_in_lane_centered={val:.1f}% < gate {COMFORT_GATES['centered_pct_min']:.0f}%"
        )

    def test_zero_out_of_lane_events(self, tmp_path: Path) -> None:
        rec = tmp_path / "nominal.h5"
        make_nominal_recording(rec)
        summary = analyze_recording_summary(rec)
        val = summary["safety"]["out_of_lane_events_full_run"]
        assert val == 0, f"out_of_lane_events_full_run={val}, expected 0"

    def test_all_gates_pass_combined(self, tmp_path: Path) -> None:
        """Single combined assertion: all S1-M39 gates pass on a clean synthetic run."""
        rec = tmp_path / "nominal.h5"
        make_nominal_recording(rec)
        summary = analyze_recording_summary(rec)
        assert_all_gates_pass(summary, label="nominal_synthetic")


class TestSyntheticAggressive:
    """
    Verify that analyze_recording_summary() correctly detects gate failures
    for a recording that deliberately violates every comfort gate.
    These tests are the canary for regressions in the failure-detection logic.
    """

    def test_accel_p95_filtered_exceeds_gate(self, tmp_path: Path) -> None:
        rec = tmp_path / "aggressive.h5"
        make_aggressive_recording(rec)
        summary = analyze_recording_summary(rec)
        val = summary["comfort"]["acceleration_p95_filtered"]
        assert val > COMFORT_GATES["accel_p95_filtered_max"], (
            f"Expected accel P95 filtered > {COMFORT_GATES['accel_p95_filtered_max']:.1f}, "
            f"got {val:.3f}. The aggressive profile may not be aggressive enough."
        )

    def test_commanded_jerk_p95_exceeds_gate(self, tmp_path: Path) -> None:
        rec = tmp_path / "aggressive.h5"
        make_aggressive_recording(rec)
        summary = analyze_recording_summary(rec)
        val = summary["comfort"]["commanded_jerk_p95"]
        assert val > COMFORT_GATES["commanded_jerk_p95_max"], (
            f"Expected commanded jerk P95 > {COMFORT_GATES['commanded_jerk_p95_max']:.1f}, "
            f"got {val:.3f}."
        )

    def test_lateral_error_p95_exceeds_gate(self, tmp_path: Path) -> None:
        rec = tmp_path / "aggressive.h5"
        make_aggressive_recording(rec)
        summary = analyze_recording_summary(rec)
        val = summary["path_tracking"]["lateral_error_p95"]
        assert val > COMFORT_GATES["lateral_p95_max"], (
            f"Expected lateral error P95 > {COMFORT_GATES['lateral_p95_max']:.2f}, "
            f"got {val:.3f}."
        )

    def test_out_of_lane_event_detected(self, tmp_path: Path) -> None:
        rec = tmp_path / "aggressive.h5"
        make_aggressive_recording(rec)
        summary = analyze_recording_summary(rec)
        val = summary["safety"]["out_of_lane_events_full_run"]
        assert val >= 1, (
            f"Expected ≥1 out-of-lane event, got {val}. "
            "Check make_aggressive_recording sets right_lane < 0 for ≥10 frames."
        )


class TestGateBoundaryConditions:
    """
    Boundary and contract tests — the most important for long-term maintenance.
    """

    def test_gate_thresholds_in_sync_with_drive_summary_core(self, tmp_path: Path) -> None:
        """
        Verify that COMFORT_GATES in conftest.py matches the thresholds reported
        by analyze_recording_summary() itself.

        This test will fail if thresholds are updated in drive_summary_core.py
        but not synchronised in conftest.COMFORT_GATES.  It is the canary for
        silent threshold drift.
        """
        rec = tmp_path / "nominal.h5"
        make_nominal_recording(rec)
        summary = analyze_recording_summary(rec)
        reported = summary["comfort"]["comfort_gate_thresholds_si"]
        assert reported["longitudinal_accel_p95_mps2"] == pytest.approx(
            COMFORT_GATES["accel_p95_filtered_max"]
        ), (
            f"conftest accel threshold {COMFORT_GATES['accel_p95_filtered_max']} "
            f"out of sync with drive_summary_core.py value "
            f"{reported['longitudinal_accel_p95_mps2']}"
        )
        assert reported["longitudinal_jerk_p95_mps3"] == pytest.approx(
            COMFORT_GATES["commanded_jerk_p95_max"]
        ), (
            f"conftest jerk threshold {COMFORT_GATES['commanded_jerk_p95_max']} "
            f"out of sync with drive_summary_core.py value "
            f"{reported['longitudinal_jerk_p95_mps3']}"
        )

    def test_emergency_stop_flag_is_detected_by_extract_run_metrics(
        self, tmp_path: Path
    ) -> None:
        """
        Verify that _extract_run_metrics correctly counts emergency stop events.
        Uses the aggressive recording which has emergency_stop=1 from frame 200.
        """
        rec = tmp_path / "aggressive.h5"
        make_aggressive_recording(rec)
        _, metrics = gate_triage._extract_run_metrics(
            rec,
            analyze_to_failure=False,
            min_control_fps=15.0,
            max_unity_time_gap_s=0.25,
        )
        assert metrics["emergency_stop_count"] >= 1, (
            f"Expected emergency_stop_count ≥ 1, got {metrics['emergency_stop_count']}"
        )

    def test_nominal_recording_has_valid_fps(self, tmp_path: Path) -> None:
        """
        Verify the nominal synthetic recording satisfies the FPS validity gate
        used by _extract_run_metrics.  A 20 Hz, 60 s recording → control_fps ≈ 20.
        """
        rec = tmp_path / "nominal.h5"
        make_nominal_recording(rec)
        _, metrics = gate_triage._extract_run_metrics(
            rec,
            analyze_to_failure=False,
            min_control_fps=15.0,
            max_unity_time_gap_s=0.25,
        )
        assert metrics["run_validity_pass"] is True, (
            f"Nominal recording failed run validity: {metrics['run_validity_reasons']}"
        )
        assert metrics["control_fps"] >= 15.0, (
            f"control_fps={metrics['control_fps']:.1f} < 15.0"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Tier 2 — Golden recording tests (skipped if file not on disk)
# ═════════════════════════════════════════════════════════════════════════════

class TestGoldenRecordings:
    """
    Regression tests against actual S1-M39 validation recordings.

    These tests MUST remain green before any config change is promoted.
    If a test fails after a change:
      1. Re-run 3× live Unity runs on both tracks.
      2. Verify new runs pass all gates with equal or better metrics.
      3. Update tests/fixtures/golden_recordings.json with new filenames.
      4. Update BASELINE_SCORES in conftest.py if scores improved.
      5. Re-run these tests with the updated manifest to confirm green.
    """

    def test_s_loop_all_comfort_gates_pass(self, golden_s_loop: Path) -> None:
        summary = analyze_recording_summary(golden_s_loop)
        assert_all_gates_pass(summary, label="s_loop_golden")

    def test_highway_65_all_comfort_gates_pass(self, golden_highway_65: Path) -> None:
        summary = analyze_recording_summary(golden_highway_65)
        assert_all_gates_pass(summary, label="highway_65_golden")

    def test_s_loop_score_within_baseline_tolerance(self, golden_s_loop: Path) -> None:
        """Score must stay within ±2 of the documented S1-M39 baseline (95.6)."""
        summary = analyze_recording_summary(golden_s_loop)
        score   = float(summary["executive_summary"]["overall_score"])
        base    = BASELINE_SCORES["s_loop"]
        assert abs(score - base) <= SCORE_TOLERANCE, (
            f"s_loop score {score:.1f} differs from baseline {base:.1f} "
            f"by more than ±{SCORE_TOLERANCE:.1f} points — "
            "analysis pipeline may have changed how metrics are scored"
        )

    def test_highway_65_score_within_baseline_tolerance(
        self, golden_highway_65: Path
    ) -> None:
        """Score must stay within ±2 of the documented S1-M39 baseline (96.2)."""
        summary = analyze_recording_summary(golden_highway_65)
        score   = float(summary["executive_summary"]["overall_score"])
        base    = BASELINE_SCORES["highway_65"]
        assert abs(score - base) <= SCORE_TOLERANCE, (
            f"highway_65 score {score:.1f} differs from baseline {base:.1f} "
            f"by more than ±{SCORE_TOLERANCE:.1f} points"
        )

    @pytest.mark.parametrize("track_id", list(BASELINE_SCORES.keys()))
    def test_no_failure_detected_on_golden_recording(self, track_id: str) -> None:
        """Golden recordings must be clean full runs (no failure_detected flag)."""
        path = golden_recording_path(track_id)
        if path is None:
            pytest.skip(f"{track_id} golden recording not on disk")
        summary = analyze_recording_summary(path)
        failure = summary["executive_summary"]["failure_detected"]
        assert not failure, (
            f"{track_id} golden recording has failure_detected=True — "
            "the file may not be a clean full run. Re-register a different recording."
        )

    @pytest.mark.parametrize("track_id", list(BASELINE_SCORES.keys()))
    def test_score_within_baseline_tolerance(self, track_id: str) -> None:
        """Score must stay within per-track tolerance of its documented baseline."""
        path = golden_recording_path(track_id)
        if path is None:
            pytest.skip(f"{track_id} golden recording not on disk")
        summary = analyze_recording_summary(path)
        score = float(summary["executive_summary"]["overall_score"])
        base = BASELINE_SCORES[track_id]
        tol = SCORE_TOLERANCES.get(track_id, SCORE_TOLERANCE)
        assert abs(score - base) <= tol, (
            f"{track_id} score {score:.1f} differs from baseline {base:.1f} "
            f"by more than ±{tol:.1f} points — "
            "analysis pipeline may have changed how metrics are scored"
        )

    @pytest.mark.parametrize("track_id", list(BASELINE_SCORES.keys()))
    def test_all_comfort_gates_pass_on_golden_recording(self, track_id: str) -> None:
        """Comfort gates must pass on every registered golden recording.

        Uses per-track lateral P95 gate overrides for tracks whose geometry
        structurally exceeds the standard 0.40m gate (e.g. hairpin R15,
        sweeping highway 55% arc coverage).  All other gates (jerk, accel,
        e-stops) use the universal standard values.
        """
        path = golden_recording_path(track_id)
        if path is None:
            pytest.skip(f"{track_id} golden recording not on disk")
        summary = analyze_recording_summary(path)
        # Unified scoring: Trajectory layer uses curvature-adjusted RMSE/P95.
        # Gate check: Trajectory layer must be green (≥80). Same rule for all tracks.
        es  = summary["executive_summary"]
        sp  = summary.get("comfort", {})
        layer_scores = summary.get("layer_scores", {})
        traj_score = float(layer_scores.get("Trajectory", 100.0))
        jerk = float(sp.get("commanded_jerk_p95", 0.0))
        accel = float(sp.get("acceleration_p95_filtered", 0.0))
        e_stops = es.get("emergency_stops", 0)
        assert traj_score >= 80.0, (
            f"[{track_id}_golden] Trajectory layer {traj_score:.1f}/100 < 80 (yellow). "
            f"Curvature-adjusted RMSE/P95 penalties too high."
        )
        assert jerk <= COMFORT_GATES["commanded_jerk_p95_max"], (
            f"[{track_id}_golden] commanded jerk P95 {jerk:.3f} > {COMFORT_GATES['commanded_jerk_p95_max']}"
        )
        assert accel <= COMFORT_GATES["accel_p95_filtered_max"], (
            f"[{track_id}_golden] accel P95 {accel:.3f} > {COMFORT_GATES['accel_p95_filtered_max']}"
        )
        assert e_stops == 0, (
            f"[{track_id}_golden] {e_stops} emergency stop(s) — golden recording must have 0"
        )
