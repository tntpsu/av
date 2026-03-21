#!/usr/bin/env python3
"""Replay recorded MPC inputs through a new config and compare steering outputs.

This tool provides a prediction of whether a config change will improve the
scoring metrics BEFORE running in Unity.  It replays the exact frame-by-frame
inputs that Unity produced in a previous run through the MPC with the new
config, then compares the resulting steering sequence.

Limitations (open-loop replay):
  - The lateral error sequence is fixed (from the recording).  In reality,
    changing steering changes where the car goes, which changes future errors.
  - This means the tool is best at detecting DIRECTION of change (jerk up/down,
    oscillation up/down) rather than predicting exact magnitude.
  - For a paired closed-loop prediction, see test_closedloop_stability.py.

Usage:
    python tools/replay_validate.py \\
        --recording data/recordings/recording_20260320_222115.h5 \\
        --config config/av_stack_config.yaml \\
        [--config config/sloop.yaml ...]   # overlays applied in order

    # Compare two configs side-by-side:
    python tools/replay_validate.py \\
        --recording data/recordings/recording_20260320_222115.h5 \\
        --config config/av_stack_config.yaml \\
        --compare-config config/highway_15mps.yaml
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Make sure the project root is on the path when run directly
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from control.mpc_controller import MPCController  # noqa: E402


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config(paths: list[str]) -> dict:
    """Load and merge YAML config files in order (later files override earlier)."""
    merged: dict = {}
    for p in paths:
        with open(p) as fh:
            data = yaml.safe_load(fh) or {}
        merged = _deep_merge(merged, data)
    return merged


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Signal extraction from HDF5
# ---------------------------------------------------------------------------

def _load_signals(h5path: str) -> dict:
    """Extract MPC input / output signals from a recording."""
    with h5py.File(h5path, "r") as f:
        def _get(path, default=None):
            if path in f:
                return f[path][:]
            return default

        n = len(f["vehicle/speed"][:])
        signals = {
            # MPC inputs (prefer dedicated MPC channels; fall back to generic)
            "e_lat":    _get("control/mpc_e_lat",     _get("control/lateral_error",  np.zeros(n))),
            "e_heading":_get("control/mpc_e_heading", _get("control/heading_error",   np.zeros(n))),
            "kappa":    _get("control/mpc_kappa_ref", _get("control/path_curvature_input", np.zeros(n))),
            "speed":    f["vehicle/speed"][:],
            "dt":       _get("vehicle/unity_delta_time", np.full(n, 0.033)),
            # Recorded outputs
            "steering_original": _get("control/mpc_last_steering_pre_modify",
                                      _get("control/steering", np.zeros(n))),
            # Ground-truth lateral offset for error calibration
            "gt_lateral": _get("vehicle/road_frame_lateral_offset"),
            # Metadata
            "is_straight": _get("control/is_straight", np.zeros(n, dtype=np.int8)),
            "mpc_feasible": _get("control/mpc_feasible", np.ones(n, dtype=bool)),
        }

    # Sanitise: replace NaN with 0
    for k, v in signals.items():
        if v is not None and np.issubdtype(np.asarray(v).dtype, np.floating):
            signals[k] = np.nan_to_num(v, nan=0.0)

    signals["n_frames"] = int(len(signals["speed"]))
    return signals


# ---------------------------------------------------------------------------
# Core replay
# ---------------------------------------------------------------------------

def replay(signals: dict, ctrl: MPCController) -> np.ndarray:
    """Run the MPC against recorded inputs; return steering_normalized array."""
    e_lat      = signals["e_lat"]
    e_heading  = signals["e_heading"]
    kappa      = signals["kappa"]
    speed      = signals["speed"]
    dt_arr     = signals["dt"]
    n          = len(e_lat)

    predicted  = np.empty(n)
    last_delta = 0.0

    for i in range(n):
        v   = float(speed[i])
        dt  = float(dt_arr[i])
        if v < 0.5:                   # skip frames where car is stationary
            predicted[i] = last_delta
            continue

        result = ctrl.compute_steering(
            e_lat=float(e_lat[i]),
            e_heading=float(e_heading[i]),
            current_speed=v,
            last_delta_norm=last_delta,
            kappa_ref=float(kappa[i]),
            v_target=v,
            v_max=v + 2.0,
            dt=dt,
        )
        last_delta  = float(result.get("steering_normalized", 0.0))
        predicted[i] = last_delta

    return predicted


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(steering: np.ndarray, label: str = "") -> dict:
    jerk      = np.abs(np.diff(steering))
    n_frames  = len(steering)

    # Oscillation: zero-crossing rate in steady state (skip first 60 frames)
    tail = steering[60:]
    signs = np.sign(tail)
    signs[signs == 0] = 1
    crossings = int(np.sum(np.diff(signs) != 0))
    zc_rate_hz = crossings / max(1, len(tail)) * 30.0  # assume 30 Hz

    # Oscillation amplitude: peak-to-peak in 5-frame windows over the tail
    if len(tail) >= 5:
        pp_windows = [tail[i:i+5].max() - tail[i:i+5].min()
                      for i in range(len(tail) - 4)]
        pp_median = float(np.median(pp_windows))
        pp_max    = float(np.max(pp_windows))
    else:
        pp_median = pp_max = 0.0

    return {
        "label":      label,
        "n_frames":   n_frames,
        "jerk_mean":  float(jerk.mean()),
        "jerk_p95":   float(np.percentile(jerk, 95)),
        "jerk_max":   float(jerk.max()),
        "zc_rate_hz": zc_rate_hz,
        "pp_median":  pp_median,
        "pp_max":     pp_max,
        "steer_rms":  float(np.sqrt(np.mean(steering ** 2))),
    }


def _print_report(orig: dict, pred: dict, signals: dict) -> None:
    # Perception noise calibration (straight segments at speed)
    mask = (signals["is_straight"] == 1) & (signals["speed"] > 4.0)
    noise_std = float(np.std(signals["e_lat"][mask])) if mask.sum() > 5 else float("nan")

    print()
    print("=" * 64)
    print("  REPLAY VALIDATION REPORT")
    print("=" * 64)
    print(f"  Frames:              {orig['n_frames']}")
    print(f"  Perception noise σ:  {noise_std:.4f} m  (straight segments, n={mask.sum()})")
    print()
    print(f"  {'Metric':<28}  {'Original':>10}  {'Replayed':>10}  {'Δ':>10}")
    print(f"  {'-'*28}  {'-'*10}  {'-'*10}  {'-'*10}")

    rows = [
        ("Steering jerk mean",   "jerk_mean",  "{:.4f}"),
        ("Steering jerk p95",    "jerk_p95",   "{:.4f}"),
        ("Steering jerk max",    "jerk_max",   "{:.4f}"),
        ("Osc. zero-cross (Hz)", "zc_rate_hz", "{:.3f}"),
        ("Osc. pp median",       "pp_median",  "{:.4f}"),
        ("Osc. pp max",          "pp_max",     "{:.4f}"),
        ("Steering RMS",         "steer_rms",  "{:.4f}"),
    ]
    for name, key, fmt in rows:
        o = orig[key]
        p = pred[key]
        delta = p - o
        sign  = "▲" if delta > 1e-5 else ("▼" if delta < -1e-5 else "—")
        better = ""
        if key in ("jerk_mean", "jerk_p95", "jerk_max", "zc_rate_hz", "pp_median", "pp_max"):
            better = "  ✓" if delta < -1e-5 else ("  ✗" if delta > 1e-5 else "")
        print(f"  {name:<28}  {fmt.format(o):>10}  {fmt.format(p):>10}"
              f"  {sign}{fmt.format(abs(delta)):>8}{better}")

    print()
    print("  Interpretation:")
    jerk_pct  = 100 * (pred["jerk_max"] - orig["jerk_max"]) / max(1e-6, orig["jerk_max"])
    pp_pct    = 100 * (pred["pp_max"]   - orig["pp_max"])   / max(1e-6, orig["pp_max"])
    print(f"    Jerk max:          {jerk_pct:+.1f} %  ({'↓ likely score ↑' if jerk_pct < 0 else '↑ likely score ↓'})")
    print(f"    Oscillation pp:    {pp_pct:+.1f} %  ({'↓ likely score ↑' if pp_pct < 0 else '↑ likely score ↓'})")
    print()
    print("  NOTE: This is an open-loop replay.  Lateral errors are fixed from")
    print("  the recording — only the steering *response* to those errors changes.")
    print("  Direction of change is reliable; magnitude prediction is approximate.")
    print("=" * 64)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recording", required=True,
                   help="Path to HDF5 recording file")
    p.add_argument("--config", action="append", default=[],
                   help="Config YAML file(s). Pass multiple times for overlays.")
    p.add_argument("--out", default=None,
                   help="Optional: save replayed steering to .npy file")
    return p.parse_args()


def main():
    args = _parse_args()

    if not args.config:
        args.config = [str(_ROOT / "config" / "av_stack_config.yaml")]
        print(f"[replay_validate] No --config supplied; using {args.config[0]}")

    print(f"[replay_validate] Recording : {args.recording}")
    print(f"[replay_validate] Config(s) : {args.config}")

    config  = _load_config(args.config)
    signals = _load_signals(args.recording)
    ctrl    = MPCController(config)

    print(f"[replay_validate] Replaying {signals['n_frames']} frames …")
    predicted = replay(signals, ctrl)

    orig_metrics = _metrics(signals["steering_original"], "original")
    pred_metrics = _metrics(predicted,                    "replayed")

    _print_report(orig_metrics, pred_metrics, signals)

    if args.out:
        np.save(args.out, predicted)
        print(f"[replay_validate] Replayed steering saved to {args.out}")


if __name__ == "__main__":
    main()
