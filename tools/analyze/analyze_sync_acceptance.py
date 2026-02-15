#!/usr/bin/env python3
"""
Generate sync acceptance metrics and SLO pass/fail report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def find_latest_recording() -> Path | None:
    candidates = sorted(Path("data/recordings").glob("recording_*.h5"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def stats(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def bool_rate(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.mean(arr > 0.5))


def run_lengths(mask: np.ndarray, target_value: bool) -> np.ndarray:
    if mask.size == 0:
        return np.array([], dtype=np.float64)
    runs: list[int] = []
    cur = bool(mask[0])
    ln = 1
    for v in mask[1:]:
        b = bool(v)
        if b == cur:
            ln += 1
        else:
            if cur == target_value:
                runs.append(ln)
            cur = b
            ln = 1
    if cur == target_value:
        runs.append(ln)
    return np.asarray(runs, dtype=np.float64)


def nearest_dt_ms(
    camera_ts: np.ndarray,
    series_ts: np.ndarray,
    max_diff_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(camera_ts.size)
    dt_ms = np.full(n, np.nan, dtype=np.float64)
    missing = np.ones(n, dtype=bool)
    if n == 0 or series_ts.size == 0:
        return dt_ms, missing
    for i, ts in enumerate(camera_ts):
        if not np.isfinite(ts):
            continue
        diffs = np.abs(series_ts - ts)
        idx = int(np.argmin(diffs))
        best = float(diffs[idx])
        if best <= max_diff_s:
            missing[i] = False
            dt_ms[i] = float(series_ts[idx] - ts) * 1000.0
    return dt_ms, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync acceptance/SLO analysis")
    parser.add_argument("recording", nargs="?", help="Path to recording .h5")
    parser.add_argument("--latest", action="store_true", help="Use latest recording")
    parser.add_argument("--baseline-json", default="", help="Optional baseline metrics JSON for comparison")
    parser.add_argument("--dt-p95-ms-threshold", type=float, default=10.0)
    parser.add_argument("--dt-p99-ms-threshold", type=float, default=20.0)
    parser.add_argument("--burst-p95-threshold", type=float, default=3.0)
    parser.add_argument("--misaligned-rate-threshold", type=float, default=0.02)
    parser.add_argument("--contract-window-ms", type=float, default=20.0)
    parser.add_argument("--output-json", default="", help="Optional output JSON path")
    args = parser.parse_args()

    rec_path: Path | None
    if args.latest:
        rec_path = find_latest_recording()
    elif args.recording:
        rec_path = Path(args.recording)
    else:
        rec_path = find_latest_recording()
    if rec_path is None or not rec_path.exists():
        raise SystemExit("No recording found.")

    with h5py.File(rec_path, "r") as f:
        v = f["vehicle"]
        metadata = {}
        meta_raw = f.attrs.get("metadata")
        if meta_raw is not None:
            try:
                meta_str = (
                    meta_raw.decode("utf-8", "ignore")
                    if isinstance(meta_raw, (bytes, bytearray))
                    else str(meta_raw)
                )
                metadata = json.loads(meta_str)
            except Exception:
                metadata = {}
        sync_policy = str(metadata.get("stream_sync_policy", "")).lower()
        camera_ts = np.asarray(f["camera/timestamps"][:], dtype=np.float64) if "camera/timestamps" in f else np.array([])
        traj_ts = np.asarray(f["trajectory/timestamps"][:], dtype=np.float64) if "trajectory/timestamps" in f else np.array([])
        control_ts = np.asarray(f["control/timestamps"][:], dtype=np.float64) if "control/timestamps" in f else np.array([])
        dt_front = np.asarray(v["stream_front_unity_dt_ms"][:], dtype=np.float64) if "stream_front_unity_dt_ms" in v else np.array([])
        dt_top = np.asarray(v["stream_topdown_unity_dt_ms"][:], dtype=np.float64) if "stream_topdown_unity_dt_ms" in v else np.array([])
        frame_delta = np.asarray(v["stream_front_frame_id_delta"][:], dtype=np.float64) if "stream_front_frame_id_delta" in v else np.array([])
        frame_delta_top = np.asarray(v["stream_topdown_frame_id_delta"][:], dtype=np.float64) if "stream_topdown_frame_id_delta" in v else np.array([])
        n = int(min(len(dt_front), len(frame_delta))) if len(dt_front) and len(frame_delta) else int(len(v["timestamps"]))

        frame_delta_risk_threshold = 3.0 if sync_policy == "latest" else 2.0
        unity_dt_risk_threshold_ms = 20.0
        snap_risk = (
            (np.isfinite(frame_delta[:n]) & (frame_delta[:n] >= frame_delta_risk_threshold))
            | (np.isfinite(dt_front[:n]) & (np.abs(dt_front[:n]) >= unity_dt_risk_threshold_ms))
        ) if n > 0 else np.array([], dtype=bool)
        cadence_risk = snap_risk.copy()

        contract_window_s = float(args.contract_window_ms) / 1000.0
        dt_cam_traj_ms, traj_missing = nearest_dt_ms(camera_ts, traj_ts, contract_window_s)
        dt_cam_control_ms, control_missing = nearest_dt_ms(camera_ts, control_ts, contract_window_s)
        m = int(min(n, dt_cam_traj_ms.size, dt_cam_control_ms.size))
        if m > 0:
            contract_misaligned = (
                traj_missing[:m]
                | control_missing[:m]
                | (np.abs(dt_cam_traj_ms[:m]) > float(args.contract_window_ms))
                | (np.abs(dt_cam_control_ms[:m]) > float(args.contract_window_ms))
            )
        else:
            contract_misaligned = np.array([], dtype=bool)

        front_ts_reused = np.asarray(v["stream_front_timestamp_reused"][:], dtype=np.float64) if "stream_front_timestamp_reused" in v else np.array([])
        front_ts_nonmono = np.asarray(v["stream_front_timestamp_non_monotonic"][:], dtype=np.float64) if "stream_front_timestamp_non_monotonic" in v else np.array([])
        front_id_reused = np.asarray(v["stream_front_frame_id_reused"][:], dtype=np.float64) if "stream_front_frame_id_reused" in v else np.array([])
        front_negative_delta = np.asarray(v["stream_front_negative_frame_delta"][:], dtype=np.float64) if "stream_front_negative_frame_delta" in v else np.array([])
        front_clock_jump = np.asarray(v["stream_front_clock_jump"][:], dtype=np.float64) if "stream_front_clock_jump" in v else np.array([])

    snap_runs = run_lengths(snap_risk, True)
    no_snap_runs = run_lengths(snap_risk, False)
    contract_runs = run_lengths(contract_misaligned, True)
    contract_no_runs = run_lengths(contract_misaligned, False)

    result = {
        "recording": str(rec_path),
        "frames": n,
        "slo_targets": {
            "dt_p95_ms_max": float(args.dt_p95_ms_threshold),
            "dt_p99_ms_max": float(args.dt_p99_ms_threshold),
            "snap_burst_p95_max_frames": float(args.burst_p95_threshold),
            "misaligned_rate_max": float(args.misaligned_rate_threshold),
        },
        "metrics": {
            "sync_policy": sync_policy or None,
            "cadence_thresholds": {
                "frame_delta_risk_threshold": float(frame_delta_risk_threshold),
                "unity_dt_risk_threshold_ms": float(unity_dt_risk_threshold_ms),
            },
            "stream_front_unity_dt_ms": stats(dt_front),
            "stream_topdown_unity_dt_ms": stats(dt_top),
            "stream_front_frame_id_delta": stats(frame_delta),
            "stream_topdown_frame_id_delta": stats(frame_delta_top),
            "dt_cam_traj_ms": stats(dt_cam_traj_ms),
            "dt_cam_control_ms": stats(dt_cam_control_ms),
            "contract_misaligned_rate": float(np.mean(contract_misaligned)) if contract_misaligned.size else None,
            "contract_misaligned_run_lengths": stats(contract_runs),
            "contract_aligned_run_lengths": stats(contract_no_runs),
            "cadence_risk_rate": float(np.mean(cadence_risk)) if cadence_risk.size else None,
            "cadence_risk_run_lengths": stats(snap_runs),
            "no_cadence_risk_run_lengths": stats(no_snap_runs),
            "clock_integrity_rates": {
                "front_timestamp_reused_rate": bool_rate(front_ts_reused),
                "front_timestamp_non_monotonic_rate": bool_rate(front_ts_nonmono),
                "front_frame_id_reused_rate": bool_rate(front_id_reused),
                "front_negative_frame_delta_rate": bool_rate(front_negative_delta),
                "front_clock_jump_rate": bool_rate(front_clock_jump),
            },
        },
    }

    m = result["metrics"]
    dt_p95 = m["dt_cam_traj_ms"]["p95"]
    dt_p99 = m["dt_cam_traj_ms"]["p99"]
    burst_stats = m["contract_misaligned_run_lengths"]
    burst_p95 = burst_stats["p95"]
    if burst_stats.get("count", 0) == 0:
        burst_p95 = 0.0
    misaligned_rate = m["contract_misaligned_rate"]
    result["slo_pass"] = {
        "dt_p95_pass": bool(dt_p95 is not None and abs(dt_p95) <= args.dt_p95_ms_threshold),
        "dt_p99_pass": bool(dt_p99 is not None and abs(dt_p99) <= args.dt_p99_ms_threshold),
        "burst_p95_pass": bool(burst_p95 is not None and burst_p95 <= args.burst_p95_threshold),
        "misaligned_rate_pass": bool(misaligned_rate is not None and misaligned_rate <= args.misaligned_rate_threshold),
    }
    result["slo_pass"]["overall"] = all(result["slo_pass"].values())

    if args.baseline_json:
        baseline_path = Path(args.baseline_json)
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text())
            result["baseline_comparison"] = {
                "baseline_file": str(baseline_path),
                "contract_misaligned_rate_delta": (
                    (misaligned_rate - baseline.get("contract_misaligned_rate"))
                    if (misaligned_rate is not None and baseline.get("contract_misaligned_rate") is not None)
                    else None
                ),
                "burst_p95_delta": (
                    (burst_p95 - baseline.get("contract_misaligned_run_lengths", {}).get("p95"))
                    if (burst_p95 is not None and baseline.get("contract_misaligned_run_lengths", {}).get("p95") is not None)
                    else None
                ),
            }

    if args.output_json:
        out_path = Path(args.output_json)
    else:
        out_path = Path("tmp/analysis") / f"sync_acceptance_{rec_path.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
