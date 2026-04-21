"""
Verify ACC emergency-state comfort exemption (acc-idm-accel-plumbing.md).

Frames in EMERGENCY_BRAKE / TTC_ESTOP / COLLAPSED_GAP_STOP authorize IDM to
command up to its -12 m/s² saturation floor. If those frames were still
included in the nominal comfort P95, intended emergency braking would flag
as discomfort and the H5 fix would show a false regression in the comfort
layer even though the safety layer improved.

This test injects a synthetic -12 m/s² deceleration spike labeled as an
emergency ACC state and verifies:
 1. Without exemption (no emergency label) → P95 reflects the spike.
 2. With the exemption (spike frames labeled emergency) → P95 drops to the
    nominal background level.

The 5 frozen regression tracks have no ACC emergency frames, so this mask
is all-False for them and scoring is byte-identical — covered by
test_scoring_regression.py.
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


def _write_recording(
    path: Path,
    *,
    n: int,
    dt_s: float = 0.077,
    speed_profile: np.ndarray,
    acc_state_codes: list[str] | None,
) -> None:
    """Minimal HDF5 recording to exercise analyze_recording_summary end-to-end."""
    t = np.arange(n, dtype=np.float64) * dt_s
    with h5py.File(path, "w") as f:
        f.create_dataset("vehicle/timestamps", data=t)
        f.create_dataset("vehicle/speed", data=speed_profile.astype(np.float32))
        f.create_dataset("control/steering", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/lateral_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/heading_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/total_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/pid_integral", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/emergency_stop", data=np.zeros(n, dtype=np.int8))
        f.create_dataset(
            "ground_truth/left_lane_line_x",
            data=np.full(n, -3.5, dtype=np.float32),
        )
        f.create_dataset(
            "ground_truth/right_lane_line_x",
            data=np.full(n, 3.5, dtype=np.float32),
        )
        if acc_state_codes is not None:
            _write_vlen_str_dataset(f["vehicle"], "acc_state_code", acc_state_codes)


def _build_emergency_spike_profile(
    n: int,
    dt_s: float,
    spike_start: int,
    spike_len: int,
    decel_mps2: float,
) -> np.ndarray:
    """Nominal 8 m/s cruise with a localized decel spike (simulates emergency brake).

    The frames [spike_start, spike_start + spike_len) see a cumulative velocity
    drop of decel_mps2 * dt_s per frame. After the spike, speed recovers
    gently to keep the tail of the profile nominal.
    """
    speed = np.full(n, 8.0, dtype=np.float64)
    for i in range(spike_len):
        speed[spike_start + i] = max(
            0.0, speed[spike_start - 1] + decel_mps2 * dt_s * (i + 1)
        )
    # Recover to 8.0 gently over the next few frames
    recovery_end = min(n, spike_start + spike_len + 5)
    recover_from = speed[spike_start + spike_len - 1]
    for j, k in enumerate(range(spike_start + spike_len, recovery_end)):
        frac = (j + 1) / max(1, recovery_end - (spike_start + spike_len))
        speed[k] = recover_from + frac * (8.0 - recover_from)
    return speed


def test_emergency_spike_inflates_p95_when_no_state_label(tmp_path: Path) -> None:
    """Baseline: without acc_state_code, the decel spike enters the P95."""
    n = 120
    dt_s = 0.077
    speed = _build_emergency_spike_profile(
        n, dt_s, spike_start=40, spike_len=5, decel_mps2=-12.0,
    )
    path = tmp_path / "no_label.h5"
    _write_recording(
        path, n=n, dt_s=dt_s, speed_profile=speed, acc_state_codes=None,
    )
    summary = analyze_recording_summary(path, analyze_to_failure=False)
    # Un-exempted: the -12 m/s² decel should dominate the P95
    assert summary["comfort"]["acceleration_p95"] >= 8.0, (
        f"baseline P95 should include the spike, got {summary['acceleration_p95']}"
    )


def test_emergency_spike_exempted_when_state_labeled(tmp_path: Path) -> None:
    """With emergency state labels on spike frames, P95 drops to nominal level."""
    n = 120
    dt_s = 0.077
    spike_start = 40
    spike_len = 5
    speed = _build_emergency_spike_profile(
        n, dt_s, spike_start=spike_start, spike_len=spike_len, decel_mps2=-12.0,
    )
    # Label the spike frames AND the immediate post-spike recovery frame as
    # EMERGENCY_BRAKE — diff-aligned masks require the adjacent frame to also
    # be exempt so the accel/jerk output cell propagates correctly.
    state_codes = ["ACC_ACTIVE"] * n
    for i in range(spike_start, spike_start + spike_len + 1):
        state_codes[i] = "EMERGENCY_BRAKE"
    path = tmp_path / "labeled.h5"
    _write_recording(
        path, n=n, dt_s=dt_s, speed_profile=speed, acc_state_codes=state_codes,
    )
    summary = analyze_recording_summary(path, analyze_to_failure=False)
    # Exempted: nominal cruise has ~0 accel, so P95 should be tiny.
    assert summary["comfort"]["acceleration_p95"] < 1.0, (
        f"exempted P95 should reflect nominal cruise, got {summary['acceleration_p95']}"
    )


def test_ttc_estop_and_collapsed_gap_stop_also_exempt(tmp_path: Path) -> None:
    """All three exempt state codes should be equivalently masked."""
    n = 120
    dt_s = 0.077
    for exempt_label in ("TTC_ESTOP", "COLLAPSED_GAP_STOP"):
        speed = _build_emergency_spike_profile(
            n, dt_s, spike_start=40, spike_len=5, decel_mps2=-12.0,
        )
        state_codes = ["ACC_ACTIVE"] * n
        for i in range(40, 40 + 5 + 1):
            state_codes[i] = exempt_label
        path = tmp_path / f"{exempt_label}.h5"
        _write_recording(
            path, n=n, dt_s=dt_s, speed_profile=speed, acc_state_codes=state_codes,
        )
        summary = analyze_recording_summary(path, analyze_to_failure=False)
        assert summary["comfort"]["acceleration_p95"] < 1.0, (
            f"{exempt_label} spike should be exempted; P95={summary['acceleration_p95']}"
        )


def test_non_exempt_state_still_counts(tmp_path: Path) -> None:
    """A non-exempt state code (e.g. CUTOUT) must NOT exempt the spike."""
    n = 120
    dt_s = 0.077
    speed = _build_emergency_spike_profile(
        n, dt_s, spike_start=40, spike_len=5, decel_mps2=-12.0,
    )
    state_codes = ["ACC_ACTIVE"] * n
    for i in range(40, 40 + 5 + 1):
        state_codes[i] = "CUTOUT"  # NOT an exempt state
    path = tmp_path / "cutout.h5"
    _write_recording(
        path, n=n, dt_s=dt_s, speed_profile=speed, acc_state_codes=state_codes,
    )
    summary = analyze_recording_summary(path, analyze_to_failure=False)
    assert summary["comfort"]["acceleration_p95"] >= 8.0, (
        f"CUTOUT must not exempt; P95={summary['acceleration_p95']}"
    )


def test_no_acc_state_code_series_is_noop(tmp_path: Path) -> None:
    """Recordings without the acc_state_code dataset must compute normally."""
    n = 120
    dt_s = 0.077
    # Linear ramp → constant accel so P95 is clearly non-zero even after filters.
    speed = np.linspace(6.0, 10.0, n).astype(np.float64)
    path = tmp_path / "no_series.h5"
    _write_recording(
        path, n=n, dt_s=dt_s, speed_profile=speed, acc_state_codes=None,
    )
    summary = analyze_recording_summary(path, analyze_to_failure=False)
    # Simply verify we got a summary back; numeric value is incidental.
    assert summary["comfort"]["acceleration_p95"] > 0.0
