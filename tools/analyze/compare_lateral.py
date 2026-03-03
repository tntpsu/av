#!/usr/bin/env python3
"""Compare lateral control performance between two HDF5 recordings.

Automatically detects MPC telemetry fields and shows them when present.
Designed for PP-vs-MPC comparison after A/B batch runs.

Usage:
    python tools/analyze/compare_lateral.py rec_a.h5 rec_b.h5 [label_a] [label_b]

Examples:
    # Compare latest two recordings from a batch run
    python tools/analyze/compare_lateral.py \\
        data/recordings/recording_pp.h5 \\
        data/recordings/recording_mpc.h5 \\
        "PP" "MPC"

    # Compare with default labels (A / B)
    python tools/analyze/compare_lateral.py recording_a.h5 recording_b.h5
"""

import sys

import h5py
import numpy as np


def analyze_recording(path: str) -> dict:
    """Extract lateral quality metrics from an HDF5 recording."""
    with h5py.File(path, "r") as f:
        lat_err = np.abs(np.array(f["control/lateral_error"]))
        steering = np.array(f["control/steering"])
        speed = np.array(f["vehicle/speed"])
        n = len(lat_err)

        # MPC fields (absent in PP-only recordings)
        has_mpc = "control/mpc_feasible" in f
        mpc_feasible = np.array(f["control/mpc_feasible"]) if has_mpc else np.zeros(n)
        mpc_solve_ms = np.array(f["control/mpc_solve_time_ms"]) if has_mpc else np.zeros(n)
        regime = (
            np.array(f["control/regime"])
            if "control/regime" in f
            else np.zeros(n)
        )

        steer_rate = np.abs(np.diff(steering))

        # E-stop proxy: speed drop > 3 m/s in a single frame
        e_stops = int((np.diff(speed) < -3.0).sum())

        # Quarter-split for error growth detection (stability indicator)
        q = max(1, n // 4)
        err_q1 = float(np.percentile(lat_err[:q], 95))
        err_q4 = float(np.percentile(lat_err[3 * q :], 95))

        return {
            "frames": n,
            "speed_mean": float(speed.mean()),
            "speed_p95": float(np.percentile(speed, 95)),
            "lat_err_p50": float(np.percentile(lat_err, 50)),
            "lat_err_p95": float(np.percentile(lat_err, 95)),
            "lat_err_max": float(lat_err.max()),
            "lat_rmse": float(np.sqrt(np.mean(lat_err**2))),
            "steer_rate_p95": (
                float(np.percentile(steer_rate, 95)) if len(steer_rate) > 0 else 0.0
            ),
            "steer_sign_changes": int(np.sum(np.diff(np.sign(steering)) != 0)),
            "mpc_active": has_mpc,
            "mpc_feasibility_pct": (
                float(mpc_feasible.mean() * 100) if has_mpc else 0.0
            ),
            "mpc_solve_p95_ms": (
                float(np.percentile(mpc_solve_ms, 95))
                if has_mpc and mpc_solve_ms.any()
                else 0.0
            ),
            "regime_mpc_pct": float((regime > 0).mean() * 100),
            "e_stops": e_stops,
            "err_q1_p95": err_q1,
            "err_q4_p95": err_q4,
            "error_growth_ratio": float(err_q4 / max(0.001, err_q1)),
        }


def compare(
    path_a: str, path_b: str, label_a: str = "A", label_b: str = "B"
) -> tuple:
    """Print a side-by-side comparison table and return (metrics_a, metrics_b)."""
    a = analyze_recording(path_a)
    b = analyze_recording(path_b)

    w = 65
    print(f"\n{'=' * w}")
    print(f"  LATERAL CONTROL COMPARISON")
    print(f"  {label_a}: {path_a.split('/')[-1]}")
    print(f"  {label_b}: {path_b.split('/')[-1]}")
    print(f"{'=' * w}")
    print(f"  {'Metric':<32} {label_a:>14} {label_b:>14}")
    print(f"  {'-' * 60}")

    rows = [
        ("Frames", "frames", "{:.0f}", False),
        ("Speed mean (m/s)", "speed_mean", "{:.1f}", False),
        ("Speed P95 (m/s)", "speed_p95", "{:.1f}", False),
        ("Lat error P50 (m)", "lat_err_p50", "{:.3f}", True),
        ("Lat error P95 (m)", "lat_err_p95", "{:.3f}", True),
        ("Lat error max (m)", "lat_err_max", "{:.3f}", True),
        ("Lat RMSE (m)", "lat_rmse", "{:.3f}", True),
        ("Steer rate P95", "steer_rate_p95", "{:.4f}", True),
        ("Steer sign changes", "steer_sign_changes", "{:.0f}", True),
        ("E-stops", "e_stops", "{:.0f}", True),
        ("Error Q1 P95 (m)", "err_q1_p95", "{:.3f}", True),
        ("Error Q4 P95 (m)", "err_q4_p95", "{:.3f}", True),
        ("Error growth ratio", "error_growth_ratio", "{:.1f}", True),
    ]

    # Show MPC rows when either recording has MPC telemetry
    if a.get("mpc_active") or b.get("mpc_active"):
        rows += [
            ("MPC feasibility %", "mpc_feasibility_pct", "{:.1f}", False),
            ("MPC solve P95 (ms)", "mpc_solve_p95_ms", "{:.2f}", True),
            ("MPC regime %", "regime_mpc_pct", "{:.1f}", False),
        ]

    for name, key, fmt, lower_better in rows:
        av, bv = a[key], b[key]
        a_str = fmt.format(av)
        b_str = fmt.format(bv)
        marker = ""
        if lower_better and bv < av * 0.9:
            marker = " <<<"
        print(f"  {name:<32} {a_str:>14} {b_str:>14}{marker}")

    b_better = b["lat_rmse"] < a["lat_rmse"] and b["e_stops"] <= a["e_stops"]
    verdict = f"{label_b} BETTER" if b_better else f"{label_b} NOT BETTER"
    print(f"\n  VERDICT: {verdict}")
    if b["error_growth_ratio"] > 3.0:
        print(
            f"  WARNING: {label_b} error growing "
            f"(ratio {b['error_growth_ratio']:.1f}x) — unstable"
        )
    print(f"{'=' * w}\n")
    return a, b


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python tools/analyze/compare_lateral.py"
            " <rec_a.h5> <rec_b.h5> [label_a] [label_b]"
        )
        sys.exit(1)
    la = sys.argv[3] if len(sys.argv) > 3 else "A (PP)"
    lb = sys.argv[4] if len(sys.argv) > 4 else "B (MPC)"
    compare(sys.argv[1], sys.argv[2], la, lb)
