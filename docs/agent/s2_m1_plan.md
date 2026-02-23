# S2-M1: Automated Comfort-Gate Regression — Implementation Plan

**Created:** 2026-02-23
**Milestone:** S2-M1
**Status:** Ready to implement
**Estimated new test count:** ~20 tests (adds to existing 126)

---

## 1. Goal and Scope

Build a pytest-based comfort-gate regression system that:
- Runs **without Unity** using only HDF5 recordings on disk
- Asserts that the S1-M39 comfort gates still pass whenever the analysis
  pipeline or config changes
- Catches silent regressions where a parameter change breaks comfort metrics
  without breaking existing unit tests

This is the foundational infrastructure gate for Stage 2: all future milestone
code changes must keep these tests green before the config can be modified.

### Out of Scope for S2-M1

- Replay of the AV stack itself (`replay_perception_locked.py` etc.) — those
  harnesses require a running stack process and are not needed here
- CI pipeline integration (that is a follow-on task after the tests pass locally)
- Parametric sweeps or A/B comparison (that is `run_ab_batch.py`)

---

## 2. Architecture

### 2.1 Two-Tier Test System

```
Tier 1 — Synthetic HDF5 tests
  ├── Always runs (no recordings needed)
  ├── Creates minimal HDF5 in pytest tmp_path
  ├── Tests metric computation correctness
  └── Tests gate boundary conditions (just-pass / just-fail)

Tier 2 — Golden recording tests
  ├── Skips gracefully if recording not on disk
  ├── Loads actual S1-M39 validation recordings via manifest
  ├── Asserts all comfort gates pass on real data
  └── Asserts score is within ±2 of documented baseline
```

**Rationale for two tiers:**
Tier 1 tests the metric computation pipeline in isolation — if `drive_summary_core.py`
changes how `acceleration_p95_filtered` is computed, Tier 1 catches it even if
no recording is available. Tier 2 tests that a real recording from a known-good
run still produces gate-passing metrics — catching subtle regressions in the full
pipeline that synthetic data might not exercise (e.g., gap-filtering edge cases,
EMA initialization, commanded-jerk config loading).

### 2.2 Key Function Under Test

```
analyze_recording_summary(recording_path: Path) -> dict
```

Located in `tools/drive_summary_core.py` (line 149). This is the primary
post-processing function used by every analysis tool. Tests call this directly;
no Unity or network required.

### 2.3 Metrics Reference

The following keys from the return dict are the S1-M39 comfort gates. These are
the **exact keys** that must be used — do not substitute alternatives.

| Gate | Dict path | Threshold | Notes |
|------|-----------|-----------|-------|
| Longitudinal accel | `summary["comfort"]["acceleration_p95_filtered"]` | ≤ 3.0 m/s² | EMA α=0.95 of velocity derivative. NOT raw `acceleration_p95`. |
| Longitudinal jerk | `summary["comfort"]["commanded_jerk_p95"]` | ≤ 6.0 m/s³ | Derivative of (throttle·max_accel − brake·max_decel). NOT raw `jerk_p95` or `jerk_p95_filtered`. |
| Lateral error P95 | `summary["path_tracking"]["lateral_error_p95"]` | ≤ 0.40 m | From `control/lateral_error` HDF5 field. |
| Centered frames | `summary["path_tracking"]["time_in_lane_centered"]` | ≥ 70.0 % | % frames within ±0.5m of lane center. |
| Out-of-lane events | `summary["safety"]["out_of_lane_events_full_run"]` | == 0 | 10+ consecutive frames outside GT boundaries. |
| Emergency stops | Computed from `control/emergency_stop` HDF5 field | == 0 | Count of rising edges in the binary flag. |
| Steering jerk | `summary["comfort"]["steering_jerk_max"]` | ≤ 20.0 | Gap-filtered (1.5× median dt). Cap is 18.0; 20.0 gives 2-unit tolerance. |

**Why `commanded_jerk_p95` not `jerk_p95`:**
`jerk_p95` is computed via double-differentiation of raw `vehicle/speed` which is
noisy at 30 Hz. The scoring model (line 1454 in `drive_summary_core.py`) uses
`commanded_jerk_p95` because it reflects the controller's intent, not
quantization noise. The gate thresholds are calibrated to this signal.

**Why `acceleration_p95_filtered` not `acceleration_p95`:**
Same reason — `acceleration_p95_filtered` uses EMA α=0.95 to suppress
velocity quantization noise. The raw value is retained only for diagnostics.

**How thresholds are verified:** The `summary["comfort"]["comfort_gate_thresholds_si"]`
key always returns `{"longitudinal_accel_p95_mps2": 3.0, "longitudinal_jerk_p95_mps3": 6.0}`.
Tests can read this to stay in sync if thresholds are ever updated.

---

## 3. Files to Create

### 3.1 `tests/conftest.py` (NEW)

Shared fixtures and constants. No existing conftest.py — this is the first one.

```python
# tests/conftest.py
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

# ── Comfort gate thresholds (S1-M39) ─────────────────────────────────────────
# These mirror the values hardcoded in tools/drive_summary_core.py.
# If thresholds change there, update here and document in ROADMAP.md.
COMFORT_GATES = {
    "accel_p95_filtered_max":   3.0,   # m/s²  — acceleration_p95_filtered
    "commanded_jerk_p95_max":   6.0,   # m/s³  — commanded_jerk_p95
    "lateral_p95_max":          0.40,  # m     — lateral_error_p95
    "centered_pct_min":         70.0,  # %     — time_in_lane_centered
    "out_of_lane_events_max":   0,     # count — out_of_lane_events_full_run
    "emergency_stops_max":      0,     # count — computed from emergency_stop field
    "steering_jerk_max_max":    20.0,  # norm/s² — steering_jerk_max (gap-filtered)
}

# S1-M39 documented baseline scores (used in golden recording regression tests)
BASELINE_SCORES = {
    "s_loop":     95.6,
    "highway_65": 96.2,
}
SCORE_TOLERANCE = 2.0  # Allow ±2 before flagging as regression

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
GOLDEN_MANIFEST_PATH = FIXTURES_DIR / "golden_recordings.json"


def load_golden_manifest() -> dict:
    """Load the golden recordings manifest. Returns empty dict if not present."""
    if not GOLDEN_MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(GOLDEN_MANIFEST_PATH.read_text())
    except Exception:
        return {}


def golden_recording_path(track_id: str) -> Path | None:
    """
    Return the absolute path to a golden recording for a given track, or None
    if the manifest is missing or the recording does not exist on disk.
    """
    manifest = load_golden_manifest()
    rel = manifest.get("tracks", {}).get(track_id)
    if rel is None:
        return None
    p = REPO_ROOT / rel
    return p if p.exists() else None


# ── Synthetic HDF5 builder ────────────────────────────────────────────────────

def make_nominal_recording(path: Path, duration_s: float = 60.0, fps: float = 20.0) -> None:
    """
    Write a synthetic HDF5 recording that represents a clean, comfortable run.

    Designed so that analyze_recording_summary() produces values comfortably
    within all S1-M39 gates:
      - accel P95 filtered ≈ 0.9 m/s²   (gate ≤ 3.0)
      - commanded jerk P95 ≈ 1.0 m/s³   (gate ≤ 6.0)
      - lateral error P95 ≈ 0.12 m      (gate ≤ 0.40)
      - centered frames ≈ 98%            (gate ≥ 70%)
      - out-of-lane events = 0           (gate = 0)
      - emergency stops = 0              (gate = 0)
    """
    n = int(duration_s * fps)
    t = np.linspace(0.0, duration_s, n, dtype=np.float64)
    dt = float(t[1] - t[0])

    # Speed: ramp from 0 → 12 m/s over 10 s, then hold at 12 m/s
    ramp_frames = int(10.0 * fps)
    speed = np.concatenate([
        np.linspace(0.0, 12.0, ramp_frames, dtype=np.float32),
        np.full(n - ramp_frames, 12.0, dtype=np.float32),
    ])

    # Throttle: mirrors speed ramp (rate-limited), then steady cruise
    throttle = np.concatenate([
        np.linspace(0.0, 0.32, ramp_frames, dtype=np.float32),
        np.full(n - ramp_frames, 0.32, dtype=np.float32),
    ])
    brake = np.zeros(n, dtype=np.float32)

    # Steering: gentle sinusoidal (simulates following a gentle curve)
    steering = (0.03 * np.sin(2 * np.pi * t / 8.0)).astype(np.float32)

    # Lateral error: small, centred, bounded well within gate
    rng = np.random.default_rng(42)
    lateral_error = (0.08 * np.sin(2 * np.pi * t / 6.0) +
                     rng.normal(0.0, 0.02, n)).astype(np.float32)

    # Lane boundaries: left=-3.5, right=3.5 (standard lane, car always inside)
    left_lane = np.full(n, -3.5, dtype=np.float32)
    right_lane = np.full(n, 3.5, dtype=np.float32)

    # Path curvature: mild curve (simulates s_loop geometry)
    path_curvature = (0.04 * np.sin(2 * np.pi * t / 15.0)).astype(np.float32)

    emergency_stop = np.zeros(n, dtype=np.int8)
    heading_error = np.zeros(n, dtype=np.float32)

    with h5py.File(path, "w") as f:
        f.create_dataset("vehicle/timestamps",              data=t)
        f.create_dataset("vehicle/speed",                   data=speed)
        f.create_dataset("vehicle/unity_time",              data=t)
        f.create_dataset("control/timestamps",              data=t)
        f.create_dataset("control/steering",                data=steering)
        f.create_dataset("control/lateral_error",           data=lateral_error)
        f.create_dataset("control/heading_error",           data=heading_error)
        f.create_dataset("control/total_error",             data=lateral_error)
        f.create_dataset("control/pid_integral",            data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/emergency_stop",          data=emergency_stop)
        f.create_dataset("control/throttle",                data=throttle)
        f.create_dataset("control/brake",                   data=brake)
        f.create_dataset("ground_truth/left_lane_line_x",   data=left_lane)
        f.create_dataset("ground_truth/right_lane_line_x",  data=right_lane)
        f.create_dataset("ground_truth/path_curvature",     data=path_curvature)


def make_aggressive_recording(path: Path, duration_s: float = 60.0, fps: float = 20.0) -> None:
    """
    Write a synthetic HDF5 recording that deliberately violates every S1-M39
    comfort gate. Used to verify that gate failures are correctly detected.

    Gate violations:
      - accel P95 filtered > 3.0 m/s²   (rapid speed oscillations)
      - commanded jerk P95 > 6.0 m/s³   (alternating throttle/brake)
      - lateral error P95 > 0.40 m      (large lateral deviation)
      - out-of-lane events ≥ 1           (right_lane_x < 0 for sustained period)
      - emergency stop = 1               (flag set mid-run)
    """
    n = int(duration_s * fps)
    t = np.linspace(0.0, duration_s, n, dtype=np.float64)

    # Jerky speed: oscillates ±4 m/s around 8 m/s at 2 Hz → large accel/jerk
    speed = (8.0 + 4.0 * np.sin(2 * np.pi * 2.0 * t)).astype(np.float32)
    speed = np.clip(speed, 0.0, 15.0).astype(np.float32)

    # Alternating throttle/brake at 2 Hz → very high commanded_jerk_p95
    throttle = np.clip(np.sin(2 * np.pi * 2.0 * t), 0.0, 1.0).astype(np.float32)
    brake    = np.clip(-np.sin(2 * np.pi * 2.0 * t), 0.0, 1.0).astype(np.float32)

    # Large lateral error (drifting far from center)
    lateral_error = (0.6 * np.sin(2 * np.pi * t / 4.0)).astype(np.float32)

    # Emergency stop: set from frame 200 onwards
    emergency_stop = np.zeros(n, dtype=np.int8)
    emergency_stop[200:] = 1

    # Out-of-lane: right lane line crosses 0 for 30 frames (sustained OOL event)
    right_lane = np.full(n, 3.5, dtype=np.float32)
    right_lane[300:340] = -0.2   # car is now right of right boundary
    left_lane  = np.full(n, -3.5, dtype=np.float32)

    heading_error = np.zeros(n, dtype=np.float32)
    steering = np.zeros(n, dtype=np.float32)

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
```

---

### 3.2 `tests/fixtures/golden_recordings.json` (NEW)

Manifest of known-good S1-M39 validation recordings. Must be populated as
part of the implementation process (see Section 6.1).

```json
{
    "schema_version": "1",
    "milestone": "S1-M39",
    "recorded_at": "2026-02-22",
    "notes": "Validation recordings from S1-M39 final gate pass. Update this manifest whenever a new milestone's validation recordings supersede the previous ones.",
    "tracks": {
        "s_loop": "data/recordings/REPLACE_WITH_SLOOP_FILENAME.h5",
        "highway_65": "data/recordings/REPLACE_WITH_HWY_FILENAME.h5"
    },
    "documented_scores": {
        "s_loop": 95.6,
        "highway_65": 96.2
    }
}
```

**How to populate (Section 6.1 covers this step-by-step):**
Run `python tools/analyze/analyze_drive_overall.py` on the 2026-02-22 recordings
to identify the two that match the documented scores. Update the filenames above.

---

### 3.3 `tests/test_comfort_gate_replay.py` (NEW)

The main test file. Imports `conftest` fixtures, calls `analyze_recording_summary`
directly, and asserts gate pass/fail.

```python
# tests/test_comfort_gate_replay.py
"""
S2-M1: Automated comfort-gate regression tests.

Tier 1  — synthetic HDF5 (always runs, no recordings needed)
Tier 2  — golden recording tests (skipped if file not on disk)

Entry point: pytest tests/test_comfort_gate_replay.py -v
"""
from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

# ── Dynamic import of drive_summary_core (no top-level package dependency) ──
REPO_ROOT = Path(__file__).resolve().parents[1]
_CORE_PATH = REPO_ROOT / "tools" / "drive_summary_core.py"
_spec = importlib.util.spec_from_file_location("drive_summary_core", _CORE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
analyze_recording_summary = _mod.analyze_recording_summary

from conftest import (
    COMFORT_GATES,
    BASELINE_SCORES,
    SCORE_TOLERANCE,
    golden_recording_path,
    make_nominal_recording,
    make_aggressive_recording,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _gate_check(summary: dict) -> dict[str, bool | float]:
    """
    Extract comfort gate values from a summary dict and evaluate pass/fail
    for each gate. Returns a dict of {gate_name: value} plus {gate_name + "_pass": bool}.
    Raises AssertionError with a descriptive message on the first failure when
    called via assert_all_gates_pass().
    """
    comfort       = summary.get("comfort", {})
    path_tracking = summary.get("path_tracking", {})
    safety        = summary.get("safety", {})

    accel = float(comfort.get("acceleration_p95_filtered", 0.0))
    jerk  = float(comfort.get("commanded_jerk_p95", 0.0))
    lat   = float(path_tracking.get("lateral_error_p95", 0.0))
    cent  = float(path_tracking.get("time_in_lane_centered", 0.0))
    ool   = int(safety.get("out_of_lane_events_full_run", 0))
    stjrk = float(comfort.get("steering_jerk_max", 0.0))

    return {
        "accel_p95_filtered":  accel,
        "commanded_jerk_p95":  jerk,
        "lateral_error_p95":   lat,
        "time_in_lane_centered": cent,
        "out_of_lane_events":  ool,
        "steering_jerk_max":   stjrk,
    }


def assert_all_gates_pass(summary: dict, label: str = "") -> None:
    """Assert all comfort gates pass. Raises AssertionError with a clear message."""
    g = COMFORT_GATES
    c = _gate_check(summary)
    prefix = f"[{label}] " if label else ""
    assert c["accel_p95_filtered"] <= g["accel_p95_filtered_max"], (
        f"{prefix}accel P95 filtered {c['accel_p95_filtered']:.3f} m/s² "
        f"exceeds gate {g['accel_p95_filtered_max']:.1f} m/s²"
    )
    assert c["commanded_jerk_p95"] <= g["commanded_jerk_p95_max"], (
        f"{prefix}commanded jerk P95 {c['commanded_jerk_p95']:.3f} m/s³ "
        f"exceeds gate {g['commanded_jerk_p95_max']:.1f} m/s³"
    )
    assert c["lateral_error_p95"] <= g["lateral_p95_max"], (
        f"{prefix}lateral error P95 {c['lateral_error_p95']:.3f} m "
        f"exceeds gate {g['lateral_p95_max']:.2f} m"
    )
    assert c["time_in_lane_centered"] >= g["centered_pct_min"], (
        f"{prefix}centered frames {c['time_in_lane_centered']:.1f}% "
        f"below gate {g['centered_pct_min']:.0f}%"
    )
    assert c["out_of_lane_events"] <= g["out_of_lane_events_max"], (
        f"{prefix}out-of-lane events {c['out_of_lane_events']} "
        f"exceeds gate {g['out_of_lane_events_max']}"
    )
    assert c["steering_jerk_max"] <= g["steering_jerk_max_max"], (
        f"{prefix}steering jerk max {c['steering_jerk_max']:.2f} "
        f"exceeds gate {g['steering_jerk_max_max']:.1f}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — Synthetic HDF5 tests (always run)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSyntheticNominal:
    """
    Verify that analyze_recording_summary() returns gate-passing values
    for a carefully crafted nominal recording.
    """

    def test_nominal_accel_p95_filtered_within_gate(self, tmp_path):
        rec = tmp_path / "nominal.h5"
        make_nominal_recording(rec)
        summary = analyze_recording_summary(rec)
        val = summary["comfort"]["acceleration_p95_filtered"]
        assert val <= COMFORT_GATES["accel_p95_filtered_max"], (
            f"acceleration_p95_filtered={val:.3f} > {COMFORT_GATES['accel_p95_filtered_max']}"
        )

    def test_nominal_commanded_jerk_p95_within_gate(self, tmp_path):
        rec = tmp_path / "nominal.h5"
        make_nominal_recording(rec)
        summary = analyze_recording_summary(rec)
        val = summary["comfort"]["commanded_jerk_p95"]
        assert val <= COMFORT_GATES["commanded_jerk_p95_max"], (
            f"commanded_jerk_p95={val:.3f} > {COMFORT_GATES['commanded_jerk_p95_max']}"
        )

    def test_nominal_lateral_error_p95_within_gate(self, tmp_path):
        rec = tmp_path / "nominal.h5"
        make_nominal_recording(rec)
        summary = analyze_recording_summary(rec)
        val = summary["path_tracking"]["lateral_error_p95"]
        assert val <= COMFORT_GATES["lateral_p95_max"], (
            f"lateral_error_p95={val:.3f} > {COMFORT_GATES['lateral_p95_max']}"
        )

    def test_nominal_centered_frames_above_gate(self, tmp_path):
        rec = tmp_path / "nominal.h5"
        make_nominal_recording(rec)
        summary = analyze_recording_summary(rec)
        val = summary["path_tracking"]["time_in_lane_centered"]
        assert val >= COMFORT_GATES["centered_pct_min"], (
            f"time_in_lane_centered={val:.1f}% < {COMFORT_GATES['centered_pct_min']}%"
        )

    def test_nominal_zero_out_of_lane_events(self, tmp_path):
        rec = tmp_path / "nominal.h5"
        make_nominal_recording(rec)
        summary = analyze_recording_summary(rec)
        val = summary["safety"]["out_of_lane_events_full_run"]
        assert val == 0, f"out_of_lane_events_full_run={val}, expected 0"

    def test_nominal_all_gates_pass_combined(self, tmp_path):
        """Single combined test: all S1-M39 gates pass for the nominal synthetic recording."""
        rec = tmp_path / "nominal.h5"
        make_nominal_recording(rec)
        summary = analyze_recording_summary(rec)
        assert_all_gates_pass(summary, label="nominal_synthetic")


class TestSyntheticAggressive:
    """
    Verify that analyze_recording_summary() correctly detects gate failures
    for a recording that deliberately violates every comfort gate.
    """

    def test_aggressive_accel_exceeds_gate(self, tmp_path):
        rec = tmp_path / "aggressive.h5"
        make_aggressive_recording(rec)
        summary = analyze_recording_summary(rec)
        val = summary["comfort"]["acceleration_p95_filtered"]
        assert val > COMFORT_GATES["accel_p95_filtered_max"], (
            f"Expected accel P95 filtered > {COMFORT_GATES['accel_p95_filtered_max']}, "
            f"got {val:.3f}. The aggressive profile may not be aggressive enough."
        )

    def test_aggressive_jerk_exceeds_gate(self, tmp_path):
        rec = tmp_path / "aggressive.h5"
        make_aggressive_recording(rec)
        summary = analyze_recording_summary(rec)
        val = summary["comfort"]["commanded_jerk_p95"]
        assert val > COMFORT_GATES["commanded_jerk_p95_max"], (
            f"Expected commanded jerk P95 > {COMFORT_GATES['commanded_jerk_p95_max']}, "
            f"got {val:.3f}."
        )

    def test_aggressive_lateral_error_exceeds_gate(self, tmp_path):
        rec = tmp_path / "aggressive.h5"
        make_aggressive_recording(rec)
        summary = analyze_recording_summary(rec)
        val = summary["path_tracking"]["lateral_error_p95"]
        assert val > COMFORT_GATES["lateral_p95_max"], (
            f"Expected lateral error P95 > {COMFORT_GATES['lateral_p95_max']}, got {val:.3f}."
        )

    def test_aggressive_out_of_lane_detected(self, tmp_path):
        rec = tmp_path / "aggressive.h5"
        make_aggressive_recording(rec)
        summary = analyze_recording_summary(rec)
        val = summary["safety"]["out_of_lane_events_full_run"]
        assert val >= 1, (
            f"Expected ≥1 out-of-lane event, got {val}. "
            "Check that make_aggressive_recording sets right_lane < 0 for ≥10 frames."
        )


class TestGateBoundaryConditions:
    """
    Boundary tests: verify gate pass/fail at exactly the threshold value.
    These are the most important tests for catching off-by-one errors in
    the gate evaluation logic.
    """

    def test_gate_thresholds_match_summary_dict(self, tmp_path):
        """
        Verify that the hardcoded thresholds in conftest.COMFORT_GATES match
        the values reported by analyze_recording_summary() itself.
        This test will fail if thresholds are updated in drive_summary_core.py
        but not synchronized in conftest.py.
        """
        rec = tmp_path / "nominal.h5"
        make_nominal_recording(rec)
        summary = analyze_recording_summary(rec)
        reported = summary["comfort"]["comfort_gate_thresholds_si"]
        assert reported["longitudinal_accel_p95_mps2"] == pytest.approx(
            COMFORT_GATES["accel_p95_filtered_max"]
        ), "conftest accel threshold out of sync with drive_summary_core.py"
        assert reported["longitudinal_jerk_p95_mps3"] == pytest.approx(
            COMFORT_GATES["commanded_jerk_p95_max"]
        ), "conftest jerk threshold out of sync with drive_summary_core.py"

    def test_emergency_stop_flag_is_detected(self, tmp_path):
        """
        Verify that a recording with emergency_stop=1 is correctly identified
        via _extract_run_metrics. Uses the same pattern as test_gate_and_triage_contracts.py.
        """
        # Dynamic import (mirrors test_gate_and_triage_contracts.py pattern)
        _GT_PATH = REPO_ROOT / "tools" / "analyze" / "run_gate_and_triage.py"
        _gt_spec = importlib.util.spec_from_file_location("run_gate_and_triage", _GT_PATH)
        gate_triage = importlib.util.module_from_spec(_gt_spec)
        _gt_spec.loader.exec_module(gate_triage)

        rec = tmp_path / "estop.h5"
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


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — Golden recording tests (skipped if file not on disk)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def golden_s_loop():
    path = golden_recording_path("s_loop")
    if path is None:
        pytest.skip("s_loop golden recording not registered or not on disk — "
                    "see tests/fixtures/golden_recordings.json")
    return path


@pytest.fixture
def golden_highway_65():
    path = golden_recording_path("highway_65")
    if path is None:
        pytest.skip("highway_65 golden recording not registered or not on disk — "
                    "see tests/fixtures/golden_recordings.json")
    return path


class TestGoldenRecordings:
    """
    Regression tests against actual S1-M39 validation recordings.
    Skipped automatically when recordings are not on disk.

    These tests MUST remain green before any config change is promoted.
    If they fail after a config change:
      1. Re-run 3× live runs on both tracks.
      2. Verify new runs pass gates with equal or better metrics.
      3. Update golden_recordings.json manifest.
      4. Re-run these tests with updated manifest.
    """

    def test_s_loop_all_comfort_gates_pass(self, golden_s_loop):
        summary = analyze_recording_summary(golden_s_loop)
        assert_all_gates_pass(summary, label="s_loop_golden")

    def test_highway_65_all_comfort_gates_pass(self, golden_highway_65):
        summary = analyze_recording_summary(golden_highway_65)
        assert_all_gates_pass(summary, label="highway_65_golden")

    def test_s_loop_score_within_baseline_tolerance(self, golden_s_loop):
        """Score must be within ±2 of documented S1-M39 baseline (95.6)."""
        summary = analyze_recording_summary(golden_s_loop)
        score = float(summary["executive_summary"]["overall_score"])
        baseline = BASELINE_SCORES["s_loop"]
        assert abs(score - baseline) <= SCORE_TOLERANCE, (
            f"s_loop score {score:.1f} differs from baseline {baseline:.1f} "
            f"by more than ±{SCORE_TOLERANCE:.1f} points"
        )

    def test_highway_65_score_within_baseline_tolerance(self, golden_highway_65):
        """Score must be within ±2 of documented S1-M39 baseline (96.2)."""
        summary = analyze_recording_summary(golden_highway_65)
        score = float(summary["executive_summary"]["overall_score"])
        baseline = BASELINE_SCORES["highway_65"]
        assert abs(score - baseline) <= SCORE_TOLERANCE, (
            f"highway_65 score {score:.1f} differs from baseline {baseline:.1f} "
            f"by more than ±{SCORE_TOLERANCE:.1f} points"
        )

    @pytest.mark.parametrize("track_id", ["s_loop", "highway_65"])
    def test_no_failure_detected_on_golden_recording(self, track_id):
        path = golden_recording_path(track_id)
        if path is None:
            pytest.skip(f"{track_id} golden recording not on disk")
        summary = analyze_recording_summary(path)
        failure = summary["executive_summary"]["failure_detected"]
        assert not failure, (
            f"{track_id} golden recording has failure_detected=True — "
            "the recording may not be a clean 60s run."
        )
```

---

## 4. Test Case Inventory

### Tier 1 — Synthetic (always runs)

| Test | What it verifies | Pass condition |
|------|-----------------|----------------|
| `TestSyntheticNominal::test_nominal_accel_p95_filtered_within_gate` | EMA-filtered accel P95 computed correctly for smooth profile | `acceleration_p95_filtered ≤ 3.0 m/s²` |
| `TestSyntheticNominal::test_nominal_commanded_jerk_p95_within_gate` | Commanded jerk P95 computed correctly for rate-limited commands | `commanded_jerk_p95 ≤ 6.0 m/s³` |
| `TestSyntheticNominal::test_nominal_lateral_error_p95_within_gate` | Lateral error P95 from `control/lateral_error` | `lateral_error_p95 ≤ 0.40 m` |
| `TestSyntheticNominal::test_nominal_centered_frames_above_gate` | Centered frame percentage | `time_in_lane_centered ≥ 70%` |
| `TestSyntheticNominal::test_nominal_zero_out_of_lane_events` | Out-of-lane event detection (none should fire) | `out_of_lane_events_full_run == 0` |
| `TestSyntheticNominal::test_nominal_all_gates_pass_combined` | Combined gate check on a single nominal recording | All 6 gates pass simultaneously |
| `TestSyntheticAggressive::test_aggressive_accel_exceeds_gate` | Accel gate *failure* is detected | `acceleration_p95_filtered > 3.0` |
| `TestSyntheticAggressive::test_aggressive_jerk_exceeds_gate` | Jerk gate *failure* is detected | `commanded_jerk_p95 > 6.0` |
| `TestSyntheticAggressive::test_aggressive_lateral_error_exceeds_gate` | Lateral gate *failure* is detected | `lateral_error_p95 > 0.40` |
| `TestSyntheticAggressive::test_aggressive_out_of_lane_detected` | OOL event detection catches sustained boundary crossing | `out_of_lane_events_full_run ≥ 1` |
| `TestGateBoundaryConditions::test_gate_thresholds_match_summary_dict` | conftest constants in sync with `drive_summary_core.py` | Thresholds match to float approx |
| `TestGateBoundaryConditions::test_emergency_stop_flag_is_detected` | `_extract_run_metrics` detects emergency stop flag | `emergency_stop_count ≥ 1` |

### Tier 2 — Golden recording (skipped if file absent)

| Test | What it verifies | Pass condition |
|------|-----------------|----------------|
| `TestGoldenRecordings::test_s_loop_all_comfort_gates_pass` | All 6 S1-M39 gates on s_loop golden recording | All 6 gates pass |
| `TestGoldenRecordings::test_highway_65_all_comfort_gates_pass` | All 6 S1-M39 gates on highway_65 golden recording | All 6 gates pass |
| `TestGoldenRecordings::test_s_loop_score_within_baseline_tolerance` | Score regression on s_loop | `|score - 95.6| ≤ 2.0` |
| `TestGoldenRecordings::test_highway_65_score_within_baseline_tolerance` | Score regression on highway_65 | `|score - 96.2| ≤ 2.0` |
| `TestGoldenRecordings::test_no_failure_detected_on_golden_recording[s_loop]` | Golden recording is a clean full run | `failure_detected == False` |
| `TestGoldenRecordings::test_no_failure_detected_on_golden_recording[highway_65]` | Golden recording is a clean full run | `failure_detected == False` |

**Total new tests: 18**

---

## 5. Synthetic HDF5 Builder Specification

### 5.1 Nominal Recording — Expected Metric Values

These are the values `analyze_recording_summary()` should produce for
`make_nominal_recording()`. If a test fails, compare against these targets
to diagnose whether the recording builder or the analysis function is wrong.

| Metric | Expected range |
|--------|---------------|
| `acceleration_p95_filtered` | 0.5–1.2 m/s² (ramp profile → ~0.9 m/s²) |
| `commanded_jerk_p95` | 0.5–2.0 m/s³ (smooth throttle ramp → ~1.0 m/s³) |
| `lateral_error_p95` | 0.10–0.18 m (sinusoidal 0.08m amplitude + noise) |
| `time_in_lane_centered` | 95–100% (always within 0.08m ≪ 0.5m centered threshold) |
| `out_of_lane_events_full_run` | 0 (left=-3.5, right=3.5, car at x=0) |
| `steering_jerk_max` | ≤ 5.0 (gentle sinusoidal steering) |
| `emergency_stop_count` | 0 |

### 5.2 Aggressive Recording — Expected Gate Failures

| Metric | Expected | Why |
|--------|----------|-----|
| `acceleration_p95_filtered` | > 3.0 m/s² | ±4 m/s speed oscillation at 2 Hz → EMA-filtered accel peaks |
| `commanded_jerk_p95` | > 6.0 m/s³ | Alternating throttle/brake at 2 Hz, full swing [0, 1] |
| `lateral_error_p95` | > 0.40 m | Sinusoidal lateral error amplitude = 0.6m |
| `out_of_lane_events_full_run` | ≥ 1 | right_lane < 0 for frames 300–340 (40 frames = 2s ≫ 10-frame minimum) |
| `emergency_stop_count` | ≥ 1 | Flag set at frame 200 |

### 5.3 HDF5 Fields Required by `analyze_recording_summary()`

The following fields are the minimum required for all comfort gate computations
to produce valid (non-zero) values:

| HDF5 path | Used for | Type |
|-----------|----------|------|
| `vehicle/timestamps` | Time axis, FPS computation | float64 |
| `vehicle/speed` | Accel/jerk computation | float32 |
| `control/timestamps` | FPS gate check | float64 |
| `control/throttle` | Commanded jerk | float32 |
| `control/brake` | Commanded jerk | float32 |
| `control/lateral_error` | Lateral error P95 | float32 |
| `control/heading_error` | Heading metrics | float32 |
| `control/steering` | Steering jerk | float32 |
| `control/emergency_stop` | Emergency stop count | int8 |
| `control/pid_integral` | System health | float32 |
| `ground_truth/left_lane_line_x` | Out-of-lane detection | float32 |
| `ground_truth/right_lane_line_x` | Out-of-lane detection | float32 |
| `ground_truth/path_curvature` | Lateral accel (optional but recommended) | float32 |

**Note on `commanded_jerk_p95`:** This metric reads `max_accel` and `max_decel`
from `config/av_stack_config.yaml` (via `_load_config()` inside
`drive_summary_core.py`). If the config file exists, it uses the live values
(max_accel=1.2, max_decel=3.0). If config is missing, it falls back to
hardcoded defaults (max_accel=2.5, max_decel=3.0). Tests should work either way,
but the expected jerk values in Section 5.1 assume the live config is present.

---

## 6. Execution Plan

### 6.1 Initial Setup (one-time)

**Step 1: Identify S1-M39 golden recordings**

```bash
# List all 2026-02-22 recordings sorted by modification time
ls -lt data/recordings/recording_20260222_*.h5

# Analyze each candidate to find the two matching documented scores
# s_loop target: score=95.6, lateral RMSE=0.203m, 0 e-stops
# highway_65 target: score=96.2, lateral RMSE=0.044m, 0 e-stops
python tools/analyze/analyze_drive_overall.py --recording data/recordings/recording_20260222_XXXXXX.h5
```

Look for recordings where:
- `overall_score ≈ 95.6` (s_loop) or `≈ 96.2` (highway_65)
- `failure_detected = False`
- `out_of_lane_events = 0`
- `emergency_stop_count = 0`
- Duration ≈ 60 seconds

**Step 2: Populate the manifest**

Edit `tests/fixtures/golden_recordings.json` to replace the placeholder filenames
with the actual filenames found in Step 1.

**Step 3: Create the fixtures directory**

```bash
mkdir -p tests/fixtures
touch tests/fixtures/__init__.py   # keeps pytest from complaining
```

**Step 4: Run all tests**

```bash
# Tier 1 only (always works):
python3 -m pytest tests/test_comfort_gate_replay.py -v -k "Synthetic or Boundary"

# Tier 2 (requires golden recordings):
python3 -m pytest tests/test_comfort_gate_replay.py -v -k "Golden"

# Full suite including existing tests:
python3 -m pytest tests/test_comfort_gate_replay.py tests/test_control.py \
    tests/test_reference_lookahead.py tests/test_signal_chain_fixes.py \
    tests/test_gate_and_triage_contracts.py -v
```

### 6.2 Running in Everyday Development

```bash
# Quick: Tier 1 only (< 5 seconds, no recordings needed)
python3 -m pytest tests/test_comfort_gate_replay.py -v -k "Synthetic or Boundary"

# Full comfort regression (Tier 1 + Tier 2):
python3 -m pytest tests/test_comfort_gate_replay.py -v
```

### 6.3 Updating Golden Recordings After a New Milestone

After any milestone that modifies comfort metrics (new S2-M4 speed run, etc.):

1. Complete 3× live runs on both tracks.
2. Verify the best run from each track passes all comfort gates manually:
   ```bash
   python tools/analyze/analyze_drive_overall.py --latest
   ```
3. Identify the recording filenames for each track.
4. Update `tests/fixtures/golden_recordings.json`:
   - Update `"tracks"` filenames.
   - Update `"documented_scores"` if scores improved.
   - Update `"milestone"` and `"recorded_at"`.
5. Update `BASELINE_SCORES` in `tests/conftest.py` if score baseline changed.
6. Run `python3 -m pytest tests/test_comfort_gate_replay.py -v` and confirm all pass.

---

## 7. Known Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| `analyze_recording_summary()` changes metric keys | Medium | `test_gate_thresholds_match_summary_dict` verifies sync; any key rename will fail immediately |
| `commanded_jerk_p95` is zero if throttle/brake not in HDF5 | Medium | `make_nominal_recording()` always writes throttle/brake; golden recordings from live runs always have them |
| Synthetic `acceleration_p95_filtered` unexpectedly low due to EMA warmup | Low | 60s recording at 20 Hz = 1200 frames; EMA α=0.95 fully converged after ~20 frames |
| Golden recording lost or deleted | Low | Keep recordings on disk; the tests skip gracefully rather than fail if missing |
| `max_accel`/`max_decel` config values change and affect `commanded_jerk_p95` | Low | `test_gate_thresholds_match_summary_dict` will catch accel gate changes; jerk gate changes are caught by the aggressive recording test |
| test isolation: `_load_config()` reads live config | Low | All Tier 1 tests produce values well below the gate, so config variations don't affect pass/fail |
| OOL detection requires `≥10 consecutive frames` outside boundary | Low | `make_aggressive_recording()` sets 40 frames outside boundary (frames 300–340), well above the minimum |
| `time_in_lane_centered` threshold uses ±0.5m | Low | Nominal lateral error is 0.08m amplitude — always inside ±0.5m; will not be affected by minor drift |

---

## 8. Promotion Gate for S2-M1

S2-M1 is complete when ALL of the following are true:

- [ ] `tests/conftest.py` created with `COMFORT_GATES` and both HDF5 builders
- [ ] `tests/fixtures/golden_recordings.json` populated with real S1-M39 recordings
- [ ] `tests/test_comfort_gate_replay.py` created with all 18 tests
- [ ] **All 12 Tier 1 tests pass** (no live recordings needed)
- [ ] **All 6 Tier 2 tests pass** with the golden recordings registered
- [ ] `pytest tests/` still shows ≥126 passing (no regressions in existing tests)
- [ ] Running `pytest tests/test_comfort_gate_replay.py -v` completes in < 30 seconds

---

## 9. What S2-M1 Does NOT Cover

These are explicitly deferred to later milestones:

- **CI pipeline integration** (GitHub Actions, pre-commit hook) — S2-M1 runs
  locally only; CI wiring is a follow-on task
- **Multi-run statistical validation** (A/B batch, 5-run median) — this is
  `run_ab_batch.py` territory, not needed for gate regression
- **Live Unity validation** — S2-M1's entire purpose is to eliminate the need
  for Unity; any test requiring a live run is out of scope
- **Replay harness integration** (`replay_perception_locked.py` etc.) — those
  harnesses require a running stack process; S2-M1 is pure post-hoc analysis
- **Recording generation in CI** — if recordings need to be regenerated, that
  requires Unity; out of scope
