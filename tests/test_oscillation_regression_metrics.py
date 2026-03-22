"""
Regression tests for oscillation / runaway metrics (drive_summary + optional new recording).

- **Synthetic:** constant-amplitude oscillation on a curved track must NOT trip runaway
  (flat RMS envelope), matching the heuristic in ``drive_summary_core``.
- **Growth:** growing envelope still trips runaway (see ``test_summary_analyzer_oscillation_growth``).
- **Optional recording:** set ``AV_HILL_HIGHWAY_OSCILLATION_REGRESSION_H5`` to a **new** HDF5
  captured **after** config tuning; asserts ``oscillation_amplitude_runaway`` is False.
  Old recordings remain runaway-positive until re-captured — do not point at pre-fix files.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from tools.drive_summary_core import analyze_recording_summary
from tools.oscillation_attribution import analyze_oscillation_attribution


def _write_curved_track_minimal(
    path: Path,
    *,
    lateral_error: np.ndarray,
    timestamps: np.ndarray,
) -> None:
    n = len(lateral_error)
    assert len(timestamps) == n
    with h5py.File(path, "w") as f:
        f.create_dataset("vehicle/timestamps", data=timestamps.astype(np.float64))
        f.create_dataset("vehicle/speed", data=np.full(n, 8.0, dtype=np.float32))
        f.create_dataset("control/steering", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/lateral_error", data=lateral_error.astype(np.float32))
        f.create_dataset("control/heading_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/total_error", data=lateral_error.astype(np.float32))
        f.create_dataset("control/pid_integral", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/emergency_stop", data=np.zeros(n, dtype=np.int8))
        f.create_dataset("ground_truth/left_lane_line_x", data=np.full(n, -3.5, dtype=np.float32))
        f.create_dataset("ground_truth/right_lane_line_x", data=np.full(n, 3.5, dtype=np.float32))
        # >30% frames with |κ|>0.002 → curved-track oscillation gate in drive_summary
        f.create_dataset("ground_truth/path_curvature", data=np.full(n, 0.005, dtype=np.float32))


def test_constant_amplitude_on_curved_track_not_runaway(tmp_path: Path) -> None:
    """Flat RMS envelope → growth slope ~0 → no runaway even with high zero-crossing rate."""
    n_frames = 450
    dt = 1.0 / 30.0
    t = np.arange(n_frames, dtype=np.float64) * dt
    amp = 0.12
    lateral_error = (amp * np.sin(2.0 * np.pi * 0.45 * t)).astype(np.float64)

    path = tmp_path / "osc_flat_rms_curved.h5"
    _write_curved_track_minimal(path, lateral_error=lateral_error, timestamps=t)

    summary = analyze_recording_summary(path, analyze_to_failure=False)
    smooth = summary["control_smoothness"]

    assert smooth["oscillation_amplitude_runaway"] is False
    assert float(smooth.get("oscillation_rms_growth_slope_mps", 1.0)) <= 0.002


def test_growing_envelope_still_runaway(tmp_path: Path) -> None:
    """Regression guard: growing envelope must still flag runaway (existing behavior)."""
    n_frames = 300
    dt = 0.1
    t = np.arange(n_frames, dtype=np.float64) * dt
    amp = 0.02 + (0.004 * t)
    lateral_error = (amp * np.sin(2.0 * np.pi * 0.5 * t)).astype(np.float64)

    path = tmp_path / "osc_growth_curved.h5"
    _write_curved_track_minimal(path, lateral_error=lateral_error, timestamps=t)

    summary = analyze_recording_summary(path, analyze_to_failure=False)
    smooth = summary["control_smoothness"]

    assert smooth["oscillation_amplitude_runaway"] is True
    assert float(smooth["oscillation_rms_growth_slope_mps"]) > 0.002


@pytest.mark.skipif(
    not os.environ.get("AV_HILL_HIGHWAY_OSCILLATION_REGRESSION_H5"),
    reason="Set AV_HILL_HIGHWAY_OSCILLATION_REGRESSION_H5 to a post-fix hill_highway recording",
)
def test_post_fix_recording_no_oscillation_runaway() -> None:
    """
    End-to-end check after re-running Unity with updated ``av_stack_config.yaml``.

    Example::
        AV_HILL_HIGHWAY_OSCILLATION_REGRESSION_H5=data/recordings/recording_NEW.h5 pytest ...
    """
    rec = Path(os.environ["AV_HILL_HIGHWAY_OSCILLATION_REGRESSION_H5"]).expanduser()
    if not rec.is_file():
        pytest.skip(f"Recording not found: {rec}")

    summary = analyze_recording_summary(rec, analyze_to_failure=False)
    smooth = summary["control_smoothness"]

    assert smooth.get("oscillation_amplitude_runaway") is False, (
        "Post-fix recording still shows oscillation_amplitude_runaway — "
        "tune further or extend segment length."
    )

    attr = analyze_oscillation_attribution(rec, pre_failure_only=True, hill_highway_override=None)
    assert "error" not in attr
    assert attr.get("schema_version") == "oscillation_attribution_v1"

