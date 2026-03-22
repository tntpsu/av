"""
Synthetic HDF5 regressions for trajectory-adjacent health metrics in drive_summary_core.

These tests encode the *definitions* of issues seen on aggressive map tracks (e.g. hill
highway) so future refactors cannot silently break detection without updating tests.

They do not replace golden recording gates; they give fast CI signal on summary math.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from tools.drive_summary_core import analyze_recording_summary


def _write_vlen_str_dataset(grp, name: str, strings: list[str]) -> None:
    dt = h5py.string_dtype(encoding="utf-8")
    dset = grp.create_dataset(name, (len(strings),), dtype=dt)
    dset[:] = [s.encode("utf-8") for s in strings]


def _base_recording(path: Path, n: int, *, dt_s: float = 0.077) -> None:
    """Minimal fields required for analyze_recording_summary to run end-to-end."""
    t = np.arange(n, dtype=np.float64) * dt_s
    with h5py.File(path, "w") as f:
        f.create_dataset("vehicle/timestamps", data=t)
        f.create_dataset("vehicle/speed", data=np.full(n, 8.0, dtype=np.float32))
        f.create_dataset("control/steering", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/lateral_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/heading_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/total_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/pid_integral", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/emergency_stop", data=np.zeros(n, dtype=np.int8))
        f.create_dataset("ground_truth/left_lane_line_x", data=np.full(n, -3.5, dtype=np.float32))
        f.create_dataset("ground_truth/right_lane_line_x", data=np.full(n, 3.5, dtype=np.float32))


def test_signal_integrity_penalty_when_heading_gate_on_during_approach_curvature(
    tmp_path: Path,
) -> None:
    """Heading-zero gate active while |κ|>5e-4 should inflate heading_suppression_rate and score."""
    n = 160
    path = tmp_path / "sigint_heading.h5"
    _base_recording(path, n)
    with h5py.File(path, "a") as f:
        # Approach-phase curvature (summary uses same band for suppression statistic).
        f.create_dataset(
            "trajectory/reference_point_curvature",
            data=np.full(n, 0.002, dtype=np.float32),
        )
        gate = np.zeros(n, dtype=np.float32)
        gate[: int(n * 0.55)] = 1.0
        f.create_dataset("trajectory/diag_heading_zero_gate_active", data=gate)

    summary = analyze_recording_summary(path, analyze_to_failure=False)
    assert "error" not in summary
    layers = summary.get("layer_scores") or {}
    assert float(layers.get("SignalIntegrity", 100.0)) < 100.0
    sig = (summary.get("layer_score_breakdown") or {}).get("SignalIntegrity") or {}
    deds = sig.get("deductions") or []
    heading_row = next((d for d in deds if "Heading" in str(d.get("name", ""))), None)
    assert heading_row is not None
    assert float(heading_row.get("value", 0.0)) > 0.0


def test_curve_intent_arm_early_rate_marks_late_arm(tmp_path: Path) -> None:
    """One GT curve start with arm only 2 frames before → arm_early_enough_rate 0%."""
    n = 120
    path = tmp_path / "curve_intent_late.h5"
    _base_recording(path, n)
    gt_k = np.zeros(n, dtype=np.float32)
    # Curve starts at index 60 (jump across 0.003 threshold).
    gt_k[60:] = 0.006
    ctrl_k = np.copy(gt_k)
    states = ["STRAIGHT"] * n
    # Arm ENTRY at 58 → lead = 60 - 58 = 2 < arm_min_lead_frames (6).
    for i in range(58, n):
        states[i] = "ENTRY"
    with h5py.File(path, "a") as f:
        f.create_dataset("ground_truth/path_curvature", data=gt_k)
        f.create_dataset("control/path_curvature_input", data=ctrl_k)
        _write_vlen_str_dataset(f, "control/curve_intent_state", states)

    summary = analyze_recording_summary(path, analyze_to_failure=False)
    assert "error" not in summary
    diag = summary.get("curve_intent_diagnostics") or {}
    assert diag.get("available") is True
    assert float(diag.get("arm_early_enough_rate", 100.0)) == pytest.approx(0.0, abs=1e-6)
    assert any(
        "curve intent arms too late" in r.lower() for r in (summary.get("recommendations") or [])
    )


def test_curve_local_active_on_straight_exceeds_contract(tmp_path: Path) -> None:
    """ENTRY/COMMIT on is_straight frames produces curve_local_active_straight_rate > 5%."""
    n = 100
    path = tmp_path / "curve_local_straight.h5"
    _base_recording(path, n)
    straight = np.ones(n, dtype=np.float32)
    states = ["STRAIGHT"] * n
    for i in range(0, n, 10):
        states[i] = "ENTRY"
    with h5py.File(path, "a") as f:
        f.create_dataset("control/is_straight", data=straight)
        _write_vlen_str_dataset(f, "control/curve_local_state", states)

    summary = analyze_recording_summary(path, analyze_to_failure=False)
    assert "error" not in summary
    clc = summary.get("curve_local_contract") or {}
    assert clc.get("curve_local_contract_available") is True
    assert float(clc.get("curve_local_active_straight_rate", 0.0)) > 5.0
    rec = " ".join(summary.get("recommendations") or [])
    assert "straights" in rec.lower() or "straight" in rec.lower()
