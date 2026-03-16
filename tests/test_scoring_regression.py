"""
T-033: Scoring pipeline regression tests.

Re-scores every golden recording against frozen metric baselines to detect
silent scoring drift from changes to drive_summary_core.py, config, or
control/trajectory code.

Different from test_comfort_gate_replay.py:
- Tests the scoring pipeline itself (catches bugs in drive_summary_core.py)
- Compares metric deltas from a frozen baseline (catches silent regressions)
- Emits a regression report artifact (JSON) for PhilViz Gates tab

Run:
    pytest tests/test_scoring_regression.py -v

Skips gracefully if golden recordings are not on disk (CI without LFS).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# ── Make conftest importable ─────────────────────────────────────────────────
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from conftest import (
    BASELINE_SCORES,
    COMFORT_GATES,
    REPO_ROOT,
    SCORE_TOLERANCE,
    SCORE_TOLERANCES,
    golden_recording_path,
)

# ── Dynamic import of drive_summary_core ─────────────────────────────────────
_CORE_PATH = REPO_ROOT / "tools" / "drive_summary_core.py"
_core_spec = importlib.util.spec_from_file_location("drive_summary_core", _CORE_PATH)
_core_mod = importlib.util.module_from_spec(_core_spec)
sys.modules.setdefault("drive_summary_core", _core_mod)
_core_spec.loader.exec_module(_core_mod)
analyze_recording_summary = _core_mod.analyze_recording_summary

# ── Load frozen scoring baselines ────────────────────────────────────────────
_BASELINES_PATH = _TESTS_DIR / "fixtures" / "scoring_baselines.json"


def _load_baselines() -> dict:
    if not _BASELINES_PATH.exists():
        return {}
    return json.loads(_BASELINES_PATH.read_text())


_BASELINES = _load_baselines()
_TRACKS = list(_BASELINES.get("tracks", {}).keys())
_TOLERANCES = _BASELINES.get("tolerances", {})

# ── Regression report accumulator ────────────────────────────────────────────
_regression_results: dict[str, dict] = {}
_REPORT_DIR = REPO_ROOT / "data" / "reports" / "gates"


def _extract_metrics(summary: dict) -> dict:
    """Extract the key metrics from a scoring summary for baseline comparison."""
    pt = summary.get("path_tracking", {})
    comfort = summary.get("comfort", {})
    es = summary.get("executive_summary", {})
    layers = summary.get("layer_scores", {})
    return {
        "overall_score": float(es.get("overall_score", 0)),
        "trajectory_score": float(layers.get("Trajectory", 0)),
        "perception_score": float(layers.get("Perception", 0)),
        "control_score": float(layers.get("Control", 0)),
        "lateral_error_adj_rmse": float(pt.get("lateral_error_adj_rmse", 0)),
        "lateral_error_adj_p95": float(pt.get("lateral_error_adj_p95", 0)),
        "accel_p95_filtered": float(comfort.get("acceleration_p95_filtered", 0)),
        "commanded_jerk_p95": float(comfort.get("commanded_jerk_p95", 0)),
        "emergency_stops": int(es.get("emergency_stops", 0)),
    }


def _compute_deltas(actual: dict, baseline: dict) -> dict:
    """Compute metric deltas (actual - baseline) for all shared keys."""
    deltas = {}
    for key in baseline:
        if key in actual:
            deltas[key] = round(actual[key] - baseline[key], 4)
    return deltas


# ═════════════════════════════════════════════════════════════════════════════
# Parametrized scoring regression tests
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("track_id", _TRACKS)
class TestScoringRegression:
    """Re-score golden recordings and assert metrics match frozen baselines."""

    def test_overall_score_within_tolerance(self, track_id: str) -> None:
        """Overall score must stay within ± tolerance of frozen baseline."""
        path = golden_recording_path(track_id)
        if path is None:
            pytest.skip(f"{track_id} golden recording not on disk")
        summary = analyze_recording_summary(path)
        metrics = _extract_metrics(summary)
        baseline = _BASELINES["tracks"][track_id]
        tol = SCORE_TOLERANCES.get(track_id, SCORE_TOLERANCE)
        actual = metrics["overall_score"]
        expected = baseline["overall_score"]
        # Store for regression report
        _regression_results.setdefault(track_id, {})["overall_score"] = actual
        _regression_results[track_id]["overall_score_delta"] = round(actual - expected, 2)
        assert abs(actual - expected) <= tol, (
            f"{track_id}: overall score {actual:.1f} differs from baseline "
            f"{expected:.1f} by {actual - expected:+.1f} (tolerance ±{tol:.1f})"
        )

    def test_layer_scores_not_red(self, track_id: str) -> None:
        """Every layer (Perception, Trajectory, Control) must score >= 60."""
        path = golden_recording_path(track_id)
        if path is None:
            pytest.skip(f"{track_id} golden recording not on disk")
        summary = analyze_recording_summary(path)
        metrics = _extract_metrics(summary)
        for layer in ("perception_score", "trajectory_score", "control_score"):
            label = layer.replace("_score", "").title()
            val = metrics[layer]
            assert val >= 60.0, (
                f"{track_id}: {label} layer score {val:.1f} < 60 (red zone)"
            )

    def test_trajectory_layer_not_yellow(self, track_id: str) -> None:
        """Trajectory layer must score >= 80 (prevents cap regressions)."""
        path = golden_recording_path(track_id)
        if path is None:
            pytest.skip(f"{track_id} golden recording not on disk")
        summary = analyze_recording_summary(path)
        metrics = _extract_metrics(summary)
        traj = metrics["trajectory_score"]
        _regression_results.setdefault(track_id, {})["trajectory_score"] = traj
        assert traj >= 80.0, (
            f"{track_id}: Trajectory layer {traj:.1f} < 80 (yellow — cap regression)"
        )

    def test_comfort_gates_pass(self, track_id: str) -> None:
        """Accel P95 <= 3.0, jerk P95 <= 6.0, e-stops = 0."""
        path = golden_recording_path(track_id)
        if path is None:
            pytest.skip(f"{track_id} golden recording not on disk")
        summary = analyze_recording_summary(path)
        metrics = _extract_metrics(summary)
        assert metrics["accel_p95_filtered"] <= COMFORT_GATES["accel_p95_filtered_max"], (
            f"{track_id}: accel P95 {metrics['accel_p95_filtered']:.3f} > "
            f"{COMFORT_GATES['accel_p95_filtered_max']}"
        )
        assert metrics["commanded_jerk_p95"] <= COMFORT_GATES["commanded_jerk_p95_max"], (
            f"{track_id}: jerk P95 {metrics['commanded_jerk_p95']:.3f} > "
            f"{COMFORT_GATES['commanded_jerk_p95_max']}"
        )
        assert metrics["emergency_stops"] == 0, (
            f"{track_id}: {metrics['emergency_stops']} emergency stop(s)"
        )

    def test_metric_deltas_within_tolerance(self, track_id: str) -> None:
        """Key metrics must not drift beyond frozen tolerances."""
        path = golden_recording_path(track_id)
        if path is None:
            pytest.skip(f"{track_id} golden recording not on disk")
        summary = analyze_recording_summary(path)
        metrics = _extract_metrics(summary)
        baseline = _BASELINES["tracks"][track_id]
        deltas = _compute_deltas(metrics, baseline)
        # Store full metrics and deltas for regression report
        _regression_results.setdefault(track_id, {}).update(metrics)
        _regression_results[track_id]["deltas"] = deltas
        failures = []
        for key, delta in deltas.items():
            tol = _TOLERANCES.get(key)
            if tol is None:
                continue
            if abs(delta) > tol:
                failures.append(
                    f"  {key}: {metrics[key]:.4f} (baseline {baseline[key]:.4f}, "
                    f"delta {delta:+.4f}, tolerance ±{tol})"
                )
        if failures:
            msg = f"{track_id} metric regression:\n" + "\n".join(failures)
            pytest.fail(msg)


# ═════════════════════════════════════════════════════════════════════════════
# Regression report writer (runs after all tests in this module)
# ═════════════════════════════════════════════════════════════════════════════


def _status_for_delta(key: str, delta: float) -> str:
    """Return green/yellow/red status for a metric delta."""
    tol = _TOLERANCES.get(key)
    if tol is None:
        return "info"
    if abs(delta) <= tol * 0.5:
        return "green"
    if abs(delta) <= tol:
        return "yellow"
    return "red"


@pytest.fixture(scope="session", autouse=True)
def _write_regression_report(request):
    """Write regression report JSON after all tests complete."""
    yield
    if not _regression_results:
        return
    try:
        report = {
            "schema_version": "1",
            "type": "scoring_regression",
            "generated_at_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "baselines_file": str(_BASELINES_PATH),
            "tracks": {},
        }
        for track_id, data in _regression_results.items():
            deltas = data.get("deltas", {})
            statuses = {k: _status_for_delta(k, v) for k, v in deltas.items()}
            worst = "green"
            if "yellow" in statuses.values():
                worst = "yellow"
            if "red" in statuses.values():
                worst = "red"
            report["tracks"][track_id] = {
                "overall_score": data.get("overall_score"),
                "trajectory_score": data.get("trajectory_score"),
                "lateral_error_adj_rmse": data.get("lateral_error_adj_rmse"),
                "lateral_error_adj_p95": data.get("lateral_error_adj_p95"),
                "accel_p95_filtered": data.get("accel_p95_filtered"),
                "commanded_jerk_p95": data.get("commanded_jerk_p95"),
                "deltas": deltas,
                "statuses": statuses,
                "worst_status": worst,
            }
        report_path = _REPORT_DIR / "latest_scoring_regression.json"
        _REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
    except Exception:
        pass  # Never let report writing block pytest output
