"""
Shared pytest fixtures, constants, and HDF5 builders for the AV stack test suite.

This file is automatically imported by pytest before test collection.
To import COMFORT_GATES or helpers from a test file:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from conftest import COMFORT_GATES, make_nominal_recording, ...
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest

# Centralised scoring thresholds — single source of truth
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from scoring_registry import COMFORT_GATES as _REGISTRY_COMFORT_GATES

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parents[1]
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
GOLDEN_MANIFEST_PATH = FIXTURES_DIR / "golden_recordings.json"

# ── Comfort gate thresholds (S1-M39) ─────────────────────────────────────────
# Imported from tools/scoring_registry.py — single source of truth.
# Key name → summary dict path:
#   accel_p95_filtered_max  → summary["comfort"]["acceleration_p95_filtered"]
#   commanded_jerk_p95_max  → summary["comfort"]["commanded_jerk_p95"]
#   lateral_p95_max         → summary["path_tracking"]["lateral_error_p95"]
#   centered_pct_min        → summary["path_tracking"]["time_in_lane_centered"]
#   out_of_lane_events_max  → summary["safety"]["out_of_lane_events_full_run"]
#   steering_jerk_max_max   → summary["comfort"]["steering_jerk_max"]
COMFORT_GATES: dict[str, float] = _REGISTRY_COMFORT_GATES

# Baseline scores — curvature-adjusted scoring (2026-03-15).
# hairpin_15 replaced with Stanley k=3.0 golden (was PP 79.0).
# 2026-04-17: re-baselined after q_lat=1.0 revert (commit 535724d) for 4 tracks.
# 2026-04-18: hairpin_15 re-baselined after PP recovery term landed (replaces orchestrator post-limiter multiplier).
BASELINE_SCORES: dict[str, float] = {
    "s_loop":           99.1,   # 2026-04-17 post q_lat=1.0 revert (was 96.7)
    "highway_65":       99.5,   # 2026-04-17 post q_lat=1.0 revert (was 96.2)
    "hairpin_15":       98.7,   # 2026-04-18 PP recovery term landed (was 91.6 Stanley, 79.0 live PP)
    "sweeping_highway": 91.4,   # NOT re-baselined — no fresh run this cycle.
    "mixed_radius":     98.7,   # 2026-04-17 post q_lat=1.0 revert (was 91.9)
    "hill_highway":     97.6,   # 2026-04-17 first registration (apex cutting residual keeps Trajectory at 91.5)
}

# Per-track score tolerances (default 2.0). Wider for tracks with structural variance.
SCORE_TOLERANCES: dict[str, float] = {
    "s_loop":           2.0,
    "highway_65":       2.0,
    "hairpin_15":       2.0,   # 2026-04-18: PP recovery term is deterministic; tightened from 3.0 (Stanley era)
    "sweeping_highway": 3.0,   # modest variance from bias estimator convergence timing
    "mixed_radius":     3.0,   # PP↔MPC hybrid; MPC-active runs within 0.3pts
    "hill_highway":     3.0,   # new registration — conservative tolerance until multi-session stability data
}
SCORE_TOLERANCE = 2.0  # Default — used when track not in SCORE_TOLERANCES.

# Per-track lateral P95 overrides REMOVED (2026-03-15).
# Replaced by curvature-adjusted scoring in drive_summary_core.py:
#   adjusted_error = max(0, |e_lat| - 3.0*|kappa|)
# Single 0.40m gate now applies universally to all tracks.

# ── Golden recording helpers ──────────────────────────────────────────────────

def load_golden_manifest() -> dict:
    """Return the golden recordings manifest dict, or {} if not present/parseable."""
    if not GOLDEN_MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(GOLDEN_MANIFEST_PATH.read_text())
    except Exception:
        return {}


def golden_recording_path(track_id: str) -> Path | None:
    """
    Return the absolute Path to the golden recording for *track_id*, or None if
    the manifest is missing or the file does not exist on disk.
    """
    manifest = load_golden_manifest()
    rel = manifest.get("tracks", {}).get(track_id)
    if rel is None:
        return None
    p = REPO_ROOT / rel
    return p if p.exists() else None


# ── Synthetic HDF5 builders ───────────────────────────────────────────────────

def make_nominal_recording(path: Path, duration_s: float = 60.0, fps: float = 20.0) -> None:
    """
    Write a minimal synthetic HDF5 that represents a clean, comfortable run.
    All S1-M39 comfort gates should pass with comfortable margin:

        acceleration_p95_filtered ≈ 0.9 m/s²   gate ≤ 3.0
        commanded_jerk_p95        ≈ 1.0 m/s³   gate ≤ 6.0
        lateral_error_p95         ≈ 0.12 m      gate ≤ 0.40
        time_in_lane_centered     ≈ 98 %         gate ≥ 70 %
        out_of_lane_events        = 0            gate = 0
        emergency_stop_count      = 0            gate = 0
    """
    n = int(duration_s * fps)
    t = np.linspace(0.0, duration_s, n, dtype=np.float64)

    # Speed: ramp 0→12 m/s over 10 s then hold
    ramp = int(10.0 * fps)
    speed = np.concatenate([
        np.linspace(0.0, 12.0, ramp, dtype=np.float32),
        np.full(n - ramp, 12.0, dtype=np.float32),
    ])

    # Throttle: smooth ramp then steady cruise (low commanded jerk)
    throttle = np.concatenate([
        np.linspace(0.0, 0.32, ramp, dtype=np.float32),
        np.full(n - ramp, 0.32, dtype=np.float32),
    ])
    brake = np.zeros(n, dtype=np.float32)

    # Steering: gentle sine (simulates mild curve)
    steering = (0.03 * np.sin(2 * np.pi * t / 8.0)).astype(np.float32)

    # Lateral error: small centred value well inside ±0.5m centred threshold
    rng = np.random.default_rng(42)
    lateral_error = (
        0.08 * np.sin(2 * np.pi * t / 6.0) + rng.normal(0.0, 0.02, n)
    ).astype(np.float32)

    # Lane boundaries: standard 7m lane, car always inside
    left_lane  = np.full(n, -3.5, dtype=np.float32)
    right_lane = np.full(n, 3.5,  dtype=np.float32)

    # Path curvature: mild s_loop-style curve
    path_curvature = (0.04 * np.sin(2 * np.pi * t / 15.0)).astype(np.float32)

    emergency_stop = np.zeros(n, dtype=np.int8)
    heading_error  = np.zeros(n, dtype=np.float32)

    with h5py.File(path, "w") as f:
        f.create_dataset("vehicle/timestamps",             data=t)
        f.create_dataset("vehicle/speed",                  data=speed)
        f.create_dataset("vehicle/unity_time",             data=t)
        f.create_dataset("control/timestamps",             data=t)
        f.create_dataset("control/steering",               data=steering)
        f.create_dataset("control/lateral_error",          data=lateral_error)
        f.create_dataset("control/heading_error",          data=heading_error)
        f.create_dataset("control/total_error",            data=lateral_error)
        f.create_dataset("control/pid_integral",           data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/emergency_stop",         data=emergency_stop)
        f.create_dataset("control/throttle",               data=throttle)
        f.create_dataset("control/brake",                  data=brake)
        f.create_dataset("ground_truth/left_lane_line_x",  data=left_lane)
        f.create_dataset("ground_truth/right_lane_line_x", data=right_lane)
        f.create_dataset("ground_truth/path_curvature",    data=path_curvature)


def make_aggressive_recording(path: Path, duration_s: float = 60.0, fps: float = 20.0) -> None:
    """
    Write a synthetic HDF5 that deliberately violates every S1-M39 comfort gate.
    Used to verify that gate failures are correctly detected.

    Gate violations produced:
        acceleration_p95_filtered > 3.0   (±4 m/s oscillation at 2 Hz)
        commanded_jerk_p95        > 6.0   (alternating full throttle/brake at 2 Hz)
        lateral_error_p95         > 0.40  (0.6m amplitude sinusoid)
        out_of_lane_events        ≥ 1     (right_lane < 0 for frames 300–340, 40 frames)
        emergency_stop_count      ≥ 1     (flag set from frame 200 onwards)
    """
    n = int(duration_s * fps)
    t = np.linspace(0.0, duration_s, n, dtype=np.float64)

    # Jerky speed: ±4 m/s at 2 Hz → large filtered accel
    speed = np.clip(
        8.0 + 4.0 * np.sin(2 * np.pi * 2.0 * t),
        0.0, 15.0,
    ).astype(np.float32)

    # Alternating full throttle/brake at 2 Hz → high commanded_jerk_p95
    throttle = np.clip( np.sin(2 * np.pi * 2.0 * t), 0.0, 1.0).astype(np.float32)
    brake    = np.clip(-np.sin(2 * np.pi * 2.0 * t), 0.0, 1.0).astype(np.float32)

    # Large lateral error (drifting far from centre)
    lateral_error = (0.6 * np.sin(2 * np.pi * t / 4.0)).astype(np.float32)

    # Emergency stop: set from frame 200 onwards
    emergency_stop = np.zeros(n, dtype=np.int8)
    emergency_stop[200:] = 1

    # Out-of-lane: right boundary crosses zero for frames 300–340 (40 > 10-frame minimum)
    right_lane = np.full(n, 3.5, dtype=np.float32)
    right_lane[300:340] = -0.2
    left_lane = np.full(n, -3.5, dtype=np.float32)

    heading_error = np.zeros(n, dtype=np.float32)
    steering      = np.zeros(n, dtype=np.float32)

    with h5py.File(path, "w") as f:
        f.create_dataset("vehicle/timestamps",             data=t)
        f.create_dataset("vehicle/speed",                  data=speed)
        f.create_dataset("vehicle/unity_time",             data=t)
        f.create_dataset("control/timestamps",             data=t)
        f.create_dataset("control/steering",               data=steering)
        f.create_dataset("control/lateral_error",          data=lateral_error)
        f.create_dataset("control/heading_error",          data=heading_error)
        f.create_dataset("control/total_error",            data=lateral_error)
        f.create_dataset("control/pid_integral",           data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/emergency_stop",         data=emergency_stop)
        f.create_dataset("control/throttle",               data=throttle)
        f.create_dataset("control/brake",                  data=brake)
        f.create_dataset("ground_truth/left_lane_line_x",  data=left_lane)
        f.create_dataset("ground_truth/right_lane_line_x", data=right_lane)
        f.create_dataset("ground_truth/path_curvature",    data=np.zeros(n, dtype=np.float32))


def make_acc_recording(
    path: Path,
    duration_s: float = 60.0,
    fps: float = 30.0,
    acc_active_start: int = 0,
    gap_m=30.0,
    range_rate_mps: float = 0.0,
    n_near_miss_frames: int = 0,
    n_ttc_warn_frames: int = 0,
    collision: bool = False,
) -> None:
    """
    Write a synthetic HDF5 that includes all ACC/radar fields for replay tests.

    Extends make_nominal_recording with 8 new fields under vehicle/:
      radar_fwd_detected, radar_fwd_distance_m, radar_fwd_range_rate_mps, radar_fwd_snr,
      acc_active, acc_target_gap_m, acc_gap_error_m, acc_ttc_s

    Parameters
    ----------
    acc_active_start   Frame index at which ACC engages (0 = from frame 1)
    gap_m              Float constant or callable(frame_idx) -> gap in metres
    range_rate_mps     Constant approach speed (m/s; + = closing)
    n_near_miss_frames Frames with gap < 2.0m (injected after acc_active_start)
    n_ttc_warn_frames  Frames with TTC in [1.5, 2.5s] window
    collision          If True, inject one frame with gap = -0.1m (collision)

    Sentinel values are written for frames before acc_active_start so that
    drive_summary_core.py sees acc_active=0 and skips ACC scoring for those frames.
    """
    n = int(duration_s * fps)
    rng = np.random.default_rng(42)
    t = np.linspace(0.0, duration_s, n, dtype=np.float64)

    # ── Base nominal signals (same comfortable profile as make_nominal_recording) ─
    ramp = int(5.0 * fps)
    speed = np.concatenate([
        np.linspace(0.0, 20.0, ramp, dtype=np.float32),
        np.full(n - ramp, 20.0, dtype=np.float32),
    ])
    steering = (0.02 * np.sin(2 * np.pi * t / 10.0)).astype(np.float32)
    lateral_error = (
        0.06 * np.sin(2 * np.pi * t / 8.0) + rng.normal(0.0, 0.015, n)
    ).astype(np.float32)
    throttle = np.full(n, 0.30, dtype=np.float32)
    brake = np.zeros(n, dtype=np.float32)
    left_lane = np.full(n, -3.5, dtype=np.float32)
    right_lane = np.full(n, 3.5, dtype=np.float32)

    # ── ACC / radar fields ────────────────────────────────────────────────────
    detected = np.zeros(n, dtype=np.float32)
    distance = np.zeros(n, dtype=np.float32)
    rate = np.full(n, range_rate_mps, dtype=np.float32)
    snr = np.zeros(n, dtype=np.float32)
    acc_active = np.zeros(n, dtype=np.float32)
    target_gap = np.zeros(n, dtype=np.float32)
    gap_error = np.zeros(n, dtype=np.float32)
    ttc = np.full(n, 999.0, dtype=np.float32)

    # Resolve gap callable or constant
    _gap_fn = gap_m if callable(gap_m) else (lambda _i: float(gap_m))

    for i in range(acc_active_start, n):
        g = float(_gap_fn(i))
        detected[i] = 1.0
        distance[i] = max(0.0, g)
        snr[i] = max(0.0, min(1.0, 1.0 / (1.0 + g ** 2 / 900.0)))
        acc_active[i] = 1.0
        tg = 2.0 + speed[i] * 1.5   # s0=2.0, T=1.5 (default params)
        target_gap[i] = tg
        gap_error[i] = g - tg
        if rate[i] > 0.0:
            ttc[i] = g / max(rate[i], 1e-6)

    # Inject near-miss frames (gap < 2.0m) after acc_active_start
    if n_near_miss_frames > 0 and acc_active_start < n:
        nm_start = acc_active_start + 10
        nm_end = min(nm_start + n_near_miss_frames, n)
        distance[nm_start:nm_end] = 1.5
        gap_error[nm_start:nm_end] = 1.5 - target_gap[nm_start:nm_end]

    # Inject TTC warning zone frames (TTC in [1.5, 2.5s])
    if n_ttc_warn_frames > 0 and acc_active_start < n:
        tw_start = acc_active_start + 10 + n_near_miss_frames + 5
        tw_end = min(tw_start + n_ttc_warn_frames, n)
        ttc[tw_start:tw_end] = 2.0    # middle of warning zone
        distance[tw_start:tw_end] = 2.0 * rate[tw_start:tw_end]  # gap = TTC × rate

    # Inject collision event (gap ≤ 0)
    if collision and acc_active_start < n:
        col_frame = acc_active_start + 5
        distance[col_frame] = -0.1
        gap_error[col_frame] = -0.1 - target_gap[col_frame]

    with h5py.File(path, "w") as f:
        f.create_dataset("vehicle/timestamps",                  data=t)
        f.create_dataset("vehicle/speed",                       data=speed)
        f.create_dataset("vehicle/unity_time",                  data=t)
        f.create_dataset("vehicle/radar_fwd_detected",          data=detected)
        f.create_dataset("vehicle/radar_fwd_distance_m",        data=distance)
        f.create_dataset("vehicle/radar_fwd_range_rate_mps",    data=rate)
        f.create_dataset("vehicle/radar_fwd_snr",               data=snr)
        f.create_dataset("vehicle/acc_active",                  data=acc_active)
        f.create_dataset("vehicle/acc_target_gap_m",            data=target_gap)
        f.create_dataset("vehicle/acc_gap_error_m",             data=gap_error)
        f.create_dataset("vehicle/acc_ttc_s",                   data=ttc)
        f.create_dataset("control/timestamps",                  data=t)
        f.create_dataset("control/steering",                    data=steering)
        f.create_dataset("control/lateral_error",               data=lateral_error)
        f.create_dataset("control/heading_error",               data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/total_error",                 data=lateral_error)
        f.create_dataset("control/pid_integral",                data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/emergency_stop",              data=np.zeros(n, dtype=np.int8))
        f.create_dataset("control/throttle",                    data=throttle)
        f.create_dataset("control/brake",                       data=brake)
        f.create_dataset("ground_truth/left_lane_line_x",       data=left_lane)
        f.create_dataset("ground_truth/right_lane_line_x",      data=right_lane)
        f.create_dataset("ground_truth/path_curvature",         data=np.zeros(n, dtype=np.float32))


# ── Pytest fixtures for golden recordings ────────────────────────────────────

@pytest.fixture
def golden_s_loop():
    """Path to the S1-M39 s_loop validation recording, or skip if absent."""
    path = golden_recording_path("s_loop")
    if path is None:
        pytest.skip(
            "s_loop golden recording not registered or not on disk — "
            "see tests/fixtures/golden_recordings.json"
        )
    return path


@pytest.fixture
def golden_highway_65():
    """Path to the S1-M39 highway_65 validation recording, or skip if absent."""
    path = golden_recording_path("highway_65")
    if path is None:
        pytest.skip(
            "highway_65 golden recording not registered or not on disk — "
            "see tests/fixtures/golden_recordings.json"
        )
    return path


@pytest.fixture
def golden_hairpin_15():
    """Path to the hairpin_15 validation recording, or skip if absent."""
    path = golden_recording_path("hairpin_15")
    if path is None:
        pytest.skip(
            "hairpin_15 golden recording not registered or not on disk — "
            "see tests/fixtures/golden_recordings.json"
        )
    return path


@pytest.fixture
def golden_sweeping_highway():
    """Path to the sweeping_highway validation recording, or skip if absent."""
    path = golden_recording_path("sweeping_highway")
    if path is None:
        pytest.skip(
            "sweeping_highway golden recording not registered or not on disk — "
            "see tests/fixtures/golden_recordings.json"
        )
    return path


# ── Auto gate-bundle writer ───────────────────────────────────────────────────
# When the full comfort-gate suite passes (including at least one golden
# recording test), a gate bundle is written automatically to
# data/reports/gates/<timestamp>_pytest_comfort_gates/.
# This populates the Gates tab in PhilViz with zero manual steps.

_GATES_REPORTS_DIR = REPO_ROOT / "data" / "reports" / "gates"
_CONFIG_PATH       = REPO_ROOT / "config" / "av_stack_config.yaml"
_BUNDLE_SCHEMA_V   = "v1"

# Accumulates test outcomes across the session; filtered to comfort-gate file only.
_comfort_gate_results: dict[str, str] = {}


def pytest_runtest_logreport(report) -> None:  # noqa: D103
    """Collect per-test call outcomes for the gate bundle writer."""
    if report.when == "call" and "test_comfort_gate_replay" in report.nodeid:
        _comfort_gate_results[report.nodeid] = report.outcome  # "passed"|"failed"|"skipped"


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: D103
    """Write a gate bundle after a fully green comfort-gate run with golden recordings."""
    if int(exitstatus) != 0 or not _comfort_gate_results:
        return
    # Only write when at least one golden recording test actually ran (not just synthetics).
    golden_ran = any(
        "Golden" in nid and outcome in ("passed", "failed")
        for nid, outcome in _comfort_gate_results.items()
    )
    if not golden_ran:
        return
    try:
        _write_comfort_gate_bundle(_comfort_gate_results)
    except Exception:
        pass  # Never let bundle writes block or corrupt pytest output


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "missing"


def _load_scoring_regression_report() -> dict | None:
    """Load the latest scoring regression report if it exists."""
    report_path = _GATES_REPORTS_DIR / "latest_scoring_regression.json"
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text())
    except Exception:
        return None


def _build_regression_deltas(regression_report: dict) -> dict:
    """Build a regression_deltas section from the scoring regression report."""
    if not regression_report:
        return {}
    deltas = {}
    for track_id, data in regression_report.get("tracks", {}).items():
        deltas[track_id] = {
            "overall_score": data.get("overall_score"),
            "trajectory_score": data.get("trajectory_score"),
            "lateral_error_adj_rmse": data.get("lateral_error_adj_rmse"),
            "accel_p95_filtered": data.get("accel_p95_filtered"),
            "commanded_jerk_p95": data.get("commanded_jerk_p95"),
            "deltas": data.get("deltas", {}),
            "statuses": data.get("statuses", {}),
            "worst_status": data.get("worst_status", "info"),
        }
    return deltas


def _write_comfort_gate_bundle(results: dict[str, str]) -> None:
    """
    Build and write a PhilViz-compatible gate bundle from pytest comfort-gate results.

    Bundle layout:
        data/reports/gates/<ts>_pytest_comfort_gates/
            gate_report.json   — pass/fail verdict + per-test checks
            decision.json      — promote/reject decision with git/config provenance
    """
    manifest = load_golden_manifest()
    tracks   = manifest.get("tracks", {})
    # recording_ids: relative paths as stored in the manifest (e.g. "data/recordings/foo.h5")
    recording_ids = [v for v in tracks.values() if v]

    # Build checks dict: test_method_name (without "test_" prefix) → bool|None
    checks: dict[str, bool | None] = {}
    for nid, outcome in sorted(results.items()):
        # nodeid format: "tests/test_comfort_gate_replay.py::ClassName::test_name[param]"
        parts = nid.split("::")
        raw_name = parts[-1]  # e.g. "test_s_loop_all_comfort_gates_pass" or "[s_loop]" variant
        check_key = raw_name.removeprefix("test_")
        if outcome == "passed":
            checks[check_key] = True
        elif outcome == "failed":
            checks[check_key] = False
        else:
            checks[check_key] = None  # skipped — shown in grey in the Gates tab

    # pass_fail: True only if no check explicitly failed AND at least one golden check passed
    any_failed   = any(v is False for v in checks.values())
    golden_keys  = {"s_loop_all_comfort_gates_pass", "highway_65_all_comfort_gates_pass"}
    golden_pass  = any(checks.get(k) is True for k in golden_keys)
    pass_fail    = (not any_failed) and golden_pass

    ts         = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label      = "pytest_comfort_gates"
    bundle_dir = _GATES_REPORTS_DIR / f"{ts}_{label}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    git_sha_val  = _git_sha()
    config_hash  = _sha256_file(_CONFIG_PATH)
    matrix_hash  = hashlib.sha256(
        json.dumps(checks, sort_keys=True).encode()
    ).hexdigest()
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Load scoring regression report for delta inclusion
    regression_report = _load_scoring_regression_report()
    regression_deltas = _build_regression_deltas(regression_report)

    gate_report = {
        "schema_version": _BUNDLE_SCHEMA_V,
        "generated_at_utc": generated_at,
        "label": label,
        "source": "pytest",
        "git_sha": git_sha_val,
        "config_hash": config_hash,
        "matrix_hash": matrix_hash,
        "recording_ids": recording_ids,
        "pass_fail": pass_fail,
        "regression_budget": {
            "config": {
                "source": "pytest_comfort_gate_replay",
                "milestone": manifest.get("milestone", "unknown"),
            },
            "checks": checks,
            "track_metrics": None,
            "baseline_track_metrics": None,
        },
        "regression_deltas": regression_deltas if regression_deltas else None,
        "run_counts": {track_id: 1 for track_id in tracks},
    }

    decision     = "promote" if pass_fail else "reject"
    next_actions = (
        ["All comfort-gate tests passed — safe to promote this config."]
        if pass_fail
        else ["Fix failing comfort-gate tests before promoting."]
    )
    decision_payload = {
        "schema_version": _BUNDLE_SCHEMA_V,
        "generated_at_utc": generated_at,
        "git_sha": git_sha_val,
        "config_hash": config_hash,
        "matrix_hash": matrix_hash,
        "recording_ids": recording_ids,
        "pass_fail": pass_fail,
        "regression_budget": gate_report["regression_budget"],
        "regression_deltas": regression_deltas if regression_deltas else None,
        "decision": decision,
        "rationale": checks,
        "next_actions": next_actions,
        "triage_packet_count": 0,
        "counterfactual": None,
    }

    (bundle_dir / "gate_report.json").write_text(json.dumps(gate_report, indent=2))
    (bundle_dir / "decision.json").write_text(json.dumps(decision_payload, indent=2))
