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

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parents[1]
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
GOLDEN_MANIFEST_PATH = FIXTURES_DIR / "golden_recordings.json"

# ── Comfort gate thresholds (S1-M39) ─────────────────────────────────────────
# Mirror the values hardcoded in tools/drive_summary_core.py.
# If thresholds change there, update here and document in ROADMAP.md.
#
# Key name → summary dict path:
#   accel_p95_filtered_max  → summary["comfort"]["acceleration_p95_filtered"]
#   commanded_jerk_p95_max  → summary["comfort"]["commanded_jerk_p95"]
#   lateral_p95_max         → summary["path_tracking"]["lateral_error_p95"]
#   centered_pct_min        → summary["path_tracking"]["time_in_lane_centered"]
#   out_of_lane_events_max  → summary["safety"]["out_of_lane_events_full_run"]
#   steering_jerk_max_max   → summary["comfort"]["steering_jerk_max"]
COMFORT_GATES: dict[str, float] = {
    "accel_p95_filtered_max":  3.0,   # m/s²       — EMA-filtered accel P95
    "commanded_jerk_p95_max":  6.0,   # m/s³       — throttle/brake cmd derivative
    "lateral_p95_max":         0.40,  # m          — lateral error P95
    "centered_pct_min":        70.0,  # %          — % frames within ±0.5m of centre
    "out_of_lane_events_max":  0,     # count      — sustained OOL events (≥10 frames)
    "emergency_stops_max":     0,     # count      — rising edges in emergency_stop flag
    "steering_jerk_max_max":   20.0,  # norm/s²    — gap-filtered steering jerk max
}

# S1-M39 documented baseline scores — used in golden recording regression tests
BASELINE_SCORES: dict[str, float] = {
    "s_loop":     95.6,
    "highway_65": 96.2,
    "hairpin_15": 79.0,   # PP ceiling on R15/R20 geometry — gate is ±4pts (see SCORE_TOLERANCES)
}

# Per-track score tolerances (default 2.0). Wider for tracks with structural variance.
SCORE_TOLERANCES: dict[str, float] = {
    "s_loop":     2.0,
    "highway_65": 2.0,
    "hairpin_15": 4.0,   # bimodal variance (59 or 79) from consecutive arcs
}
SCORE_TOLERANCE = 2.0  # Default — used when track not in SCORE_TOLERANCES.

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
        "decision": decision,
        "rationale": checks,
        "next_actions": next_actions,
        "triage_packet_count": 0,
        "counterfactual": None,
    }

    (bundle_dir / "gate_report.json").write_text(json.dumps(gate_report, indent=2))
    (bundle_dir / "decision.json").write_text(json.dumps(decision_payload, indent=2))
