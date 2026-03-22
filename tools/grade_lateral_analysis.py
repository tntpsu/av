"""
Grade vs lateral analytics (HDF5 recording → JSON).

Shared by CLI, optional drive_summary, and PhilViz. Pre-failure window uses the same
``failure_frame`` semantics as ``executive_summary`` (via optional injection or
``get_failure_frame_from_summary``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

SCHEMA_VERSION = "grade_lateral_v1"


def _read_1d(f: h5py.File, path: str, n: int) -> Optional[np.ndarray]:
    if path not in f:
        return None
    d = f[path]
    m = min(int(d.shape[0]), n)
    return np.asarray(d[:m], dtype=np.float64)


def _bin_mask(grade: np.ndarray, thr: float, bin_name: str) -> np.ndarray:
    g = np.asarray(grade, dtype=np.float64)
    if bin_name == "down":
        return g < -thr
    if bin_name == "flat":
        return (g >= -thr) & (g <= thr)
    if bin_name == "up":
        return g > thr
    raise ValueError(f"unknown bin: {bin_name}")


def _pair_mask(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if m.size < 2:
        return np.zeros(0, dtype=bool)
    return m[:-1] & m[1:]


def _derivative(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if y.size < 2 or t.size < 2:
        return np.array([])
    dt = np.diff(t)
    dt = np.where(np.abs(dt) < 1e-9, 1.0 / 30.0, dt)
    return np.diff(y) / dt


def _mean_abs_masked_deriv(
    y: np.ndarray, t: np.ndarray, frame_mask: np.ndarray
) -> float:
    dy = _derivative(y, t)
    pm = _pair_mask(frame_mask)
    if dy.size == 0 or pm.size == 0:
        return 0.0
    n = min(len(dy), len(pm))
    if not np.any(pm[:n]):
        return 0.0
    return float(np.nanmean(np.abs(dy[:n][pm[:n]])))


def _zero_crossing_rate_hz_detrended(y: np.ndarray, t: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(t)
    if int(np.sum(m)) < 8:
        return 0.0
    y = y[m] - float(np.nanmean(y[m]))
    t = t[m]
    signs = np.sign(y)
    signs[signs == 0] = 1.0
    zc = int(np.sum(np.abs(np.diff(signs)) > 0))
    duration = float(t[-1] - t[0]) if t.size > 1 else 1.0
    return float(0.5 * zc / max(duration, 1e-6))


def analyze_grade_lateral(
    recording_path: Path | str,
    *,
    pre_failure_only: bool = True,
    grade_threshold: float = 0.02,
    grade_bins: Tuple[str, ...] = ("down", "flat", "up"),
    failure_frame: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Bin frames by ``vehicle/road_grade`` and compute conditional lateral / steering metrics.

    Args:
        recording_path: Path to HDF5.
        pre_failure_only: Clip to ``[0, failure_frame]`` inclusive when failure known.
        grade_threshold: |grade| below this counts as flat (rise/run, same sign as triage).
        grade_bins: Ordered bin names down / flat / up.
        failure_frame: If provided, skips loading failure from drive_summary (avoids extra work).
    """
    path = Path(recording_path)
    if not path.is_file():
        return {"schema_version": SCHEMA_VERSION, "error": f"Recording not found: {path}"}

    warnings: List[str] = []

    if failure_frame is None and pre_failure_only:
        from oscillation_attribution import get_failure_frame_from_summary

        failure_frame = get_failure_frame_from_summary(path)

    with h5py.File(path, "r") as f:
        n_ctrl = 0
        for key in ("control/steering", "control/timestamps", "vehicle/timestamps"):
            if key in f:
                n_ctrl = int(f[key].shape[0])
                break
        if n_ctrl <= 0:
            return {"schema_version": SCHEMA_VERSION, "error": "No control series in recording"}

        steer = _read_1d(f, "control/steering", n_ctrl)
        lat = _read_1d(f, "control/lateral_error", n_ctrl)
        grade = _read_1d(f, "vehicle/road_grade", n_ctrl)
        if grade is None:
            grade = np.zeros(n_ctrl, dtype=np.float64)
            warnings.append("vehicle/road_grade missing; using zeros")

        ts = _read_1d(f, "vehicle/timestamps", n_ctrl)
        if ts is None:
            ts = _read_1d(f, "control/timestamps", n_ctrl)
        if ts is None:
            warnings.append("No timestamps; using uniform 30 Hz")
            ts = np.arange(n_ctrl, dtype=np.float64) / 30.0
        else:
            ts = ts - float(ts[0])

    if steer is None or lat is None:
        return {"schema_version": SCHEMA_VERSION, "error": "Missing steering or lateral_error"}

    n = min(int(steer.shape[0]), int(lat.shape[0]), int(grade.shape[0]), int(ts.shape[0]))
    if n < 2:
        return {"schema_version": SCHEMA_VERSION, "error": "Insufficient frames"}

    steer = steer[:n]
    lat = lat[:n]
    grade = grade[:n]
    ts = ts[:n]

    end = n
    if pre_failure_only and failure_frame is not None:
        end = min(n, int(failure_frame) + 1)
    end = max(end, 2)

    sl = slice(0, end)
    steer_w = steer[sl]
    lat_w = lat[sl]
    grade_w = grade[sl]
    ts_w = ts[sl]

    thr = float(grade_threshold)
    bins_out: Dict[str, Any] = {}
    total_frames = int(end)

    chatter_flat = 0.0
    chatter_down = 0.0

    for bname in grade_bins:
        mask = _bin_mask(grade_w, thr, bname)
        count = int(np.sum(mask))
        pct_time = float(count / max(total_frames, 1))
        lat_sel = lat_w[mask]
        mean_abs_lat = (
            float(np.nanmean(np.abs(lat_sel)))
            if count > 0
            else 0.0
        )
        if count > 1:
            p95_abs_lat = float(np.percentile(np.abs(lat_sel), 95))
        elif count == 1:
            p95_abs_lat = float(np.abs(lat_sel[0]))
        else:
            p95_abs_lat = 0.0
        lat_deriv_mean = _mean_abs_masked_deriv(lat_w, ts_w, mask)
        steer_deriv_mean = _mean_abs_masked_deriv(steer_w, ts_w, mask)
        # Steering oscillation proxy on frames in bin (detrended ZC rate, full segment)
        steer_bin = steer_w[mask]
        ts_bin = ts_w[mask]
        zc_hz = (
            _zero_crossing_rate_hz_detrended(steer_bin, ts_bin)
            if steer_bin.size > 4
            else 0.0
        )

        bins_out[bname] = {
            "frame_count": count,
            "time_fraction": round(pct_time, 5),
            "lateral_error_abs_mean_m": round(mean_abs_lat, 6),
            "lateral_error_abs_p95_m": round(p95_abs_lat, 6),
            "lateral_error_abs_deriv_mean_mps": round(lat_deriv_mean, 6),
            "steering_deriv_abs_mean_per_s": round(steer_deriv_mean, 6),
            "steering_zero_crossing_rate_hz_detrended": round(zc_hz, 6),
        }
        if bname == "flat":
            chatter_flat = zc_hz
        if bname == "down":
            chatter_down = zc_hz

    ratio = None
    if chatter_flat > 1e-9:
        ratio = round(float(chatter_down / chatter_flat), 4)

    return {
        "schema_version": SCHEMA_VERSION,
        "recording": path.name,
        "pre_failure_only": pre_failure_only,
        "failure_frame": int(failure_frame) if failure_frame is not None else None,
        "analysis_end_frame": int(end - 1),
        "grade_threshold_rise_per_run": thr,
        "grade_bins": list(grade_bins),
        "warnings": warnings,
        "bins": bins_out,
        "correlation_summary": {
            "downhill_vs_flat_steering_chatter_ratio": ratio,
            "notes": (
                "Ratio uses detrended steering zero-crossing rate (Hz) in down vs flat bins; "
                "None if flat bin rate ~0."
            ),
        },
    }
