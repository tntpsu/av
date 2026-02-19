#!/usr/bin/env python3
"""
Batch A/B runner for AV stack tuning with robust summary stats.

Runs repeated A/B trials on the same track and reports median/p25/p75 for:
- first centerline-cross frame
- time-to-failure
- distance-window feasibility metrics (authority gap, transfer ratio, stale %)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "av_stack_config.yaml"
RECORDINGS_DIR = REPO_ROOT / "data" / "recordings"
START_SCRIPT = REPO_ROOT / "start_av_stack.sh"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _set_nested(cfg: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def _latest_recording(previous: set[Path]) -> Optional[Path]:
    candidates = [p for p in RECORDINGS_DIR.glob("*.h5") if p not in previous]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_centerline_cross(
    signal: np.ndarray,
    curve_start_frame: int,
    min_abs: float = 0.0,
    persist_frames: int = 1,
) -> Optional[int]:
    if len(signal) == 0:
        return None
    init = float(np.median(signal[: min(60, len(signal))]))
    initial_sign = int(np.sign(init)) if abs(init) >= 1e-3 else int(np.sign(signal[0]))
    if initial_sign == 0:
        return None
    threshold = max(1e-3, float(min_abs))
    for i in range(max(0, curve_start_frame), len(signal) - persist_frames + 1):
        w = signal[i : i + persist_frames]
        if np.any(np.abs(w) < threshold):
            continue
        if np.all(np.sign(w) == -initial_sign):
            return i
    return None


def _first_true(mask: np.ndarray) -> Optional[int]:
    idx = np.where(mask)[0]
    return int(idx[0]) if len(idx) else None


def _extract_stale_reasons(
    f: h5py.File,
    start_idx: int,
    end_idx: int,
) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if "control/stale_perception_reason" not in f:
        return out
    raw = f["control/stale_perception_reason"][start_idx:end_idx]
    for x in raw:
        if isinstance(x, bytes):
            s = x.decode("utf-8", "ignore").strip()
        else:
            s = str(x).strip()
        if not s:
            continue
        out[s] = out.get(s, 0) + 1
    return out


@dataclass
class TrialMetrics:
    recording: str
    centerline_cross_frame: Optional[int]
    offroad_frame: Optional[int]
    first_failure_frame: Optional[int]
    time_to_failure_s: Optional[float]
    authority_gap_mean: Optional[float]
    transfer_ratio_mean: Optional[float]
    stale_pct: Optional[float]
    speed_limited_pct: Optional[float]
    feasibility_classification: Optional[str]
    entry_start_frame: Optional[int]
    entry_end_frame: Optional[int]
    entry_start_dist_m: Optional[float]
    entry_end_dist_m: Optional[float]
    stale_reason_counts: Dict[str, int]
    jerk_p95: Optional[float]
    lateral_jerk_p95: Optional[float]
    lateral_accel_p95: Optional[float]


def _analyze_recording(
    recording_path: Path,
    curve_start_frame: int,
    entry_start_distance_m: float,
    entry_window_distance_m: float,
) -> TrialMetrics:
    from tools.debug_visualizer.backend.diagnostics import analyze_trajectory_vs_steering
    from tools.debug_visualizer.backend.summary_analyzer import analyze_recording_summary

    with h5py.File(recording_path, "r") as f:
        ts = np.asarray(f["vehicle/timestamps"][:], dtype=float)
        lce = np.asarray(f["vehicle/road_frame_lane_center_error"][:], dtype=float)
        em = np.asarray(f["control/emergency_stop"][:]).astype(bool)

        cross = _find_centerline_cross(lce, curve_start_frame=curve_start_frame)
        offroad = _first_true(em)
        fail_candidates = [x for x in [cross, offroad] if x is not None]
        fail = min(fail_candidates) if fail_candidates else None
        ttf = float(ts[fail] - ts[0]) if fail is not None and len(ts) else None

    diag = analyze_trajectory_vs_steering(
        recording_path,
        analyze_to_failure=True,
        curve_entry_start_distance_m=entry_start_distance_m,
        curve_entry_window_distance_m=entry_window_distance_m,
    )
    feas = ((diag.get("control_analysis") or {}).get("curve_entry_feasibility") or {})
    auth = feas.get("steering_authority") or {}
    spd = feas.get("speed_feasibility") or {}

    entry_start_frame = feas.get("entry_start_frame")
    entry_end_frame = feas.get("entry_end_frame")
    summary = analyze_recording_summary(str(recording_path))
    comfort = ((summary.get("metrics") or {}).get("comfort") or {})
    jerk_p95 = comfort.get("jerk_p95")
    lateral_jerk_p95 = comfort.get("lateral_jerk_p95")
    lateral_accel_p95 = comfort.get("lateral_accel_p95")
    stale_reasons: Dict[str, int] = {}
    if entry_start_frame is not None and entry_end_frame is not None:
        with h5py.File(recording_path, "r") as f:
            stale_reasons = _extract_stale_reasons(
                f,
                int(entry_start_frame),
                int(entry_end_frame),
            )

    return TrialMetrics(
        recording=recording_path.name,
        centerline_cross_frame=cross,
        offroad_frame=offroad,
        first_failure_frame=fail,
        time_to_failure_s=ttf,
        authority_gap_mean=auth.get("authority_gap_mean"),
        transfer_ratio_mean=auth.get("transfer_ratio_mean"),
        stale_pct=feas.get("stale_perception_pct"),
        speed_limited_pct=spd.get("speed_limited_pct"),
        feasibility_classification=feas.get("primary_classification"),
        entry_start_frame=entry_start_frame,
        entry_end_frame=entry_end_frame,
        entry_start_dist_m=feas.get("entry_start_distance_used_m"),
        entry_end_dist_m=feas.get("entry_end_distance_used_m"),
        stale_reason_counts=stale_reasons,
        jerk_p95=jerk_p95,
        lateral_jerk_p95=lateral_jerk_p95,
        lateral_accel_p95=lateral_accel_p95,
    )


def _q(arr: List[float], q: float) -> float:
    return float(np.quantile(np.asarray(arr, dtype=float), q))


def _summarize(label: str, trials: List[TrialMetrics]) -> None:
    def collect(name: str) -> List[float]:
        vals = []
        for t in trials:
            v = getattr(t, name)
            if v is not None:
                vals.append(float(v))
        return vals

    print(f"\n=== {label} summary ({len(trials)} trials) ===")
    for metric in [
        "centerline_cross_frame",
        "first_failure_frame",
        "time_to_failure_s",
        "authority_gap_mean",
        "transfer_ratio_mean",
        "stale_pct",
        "speed_limited_pct",
        "jerk_p95",
        "lateral_jerk_p95",
        "lateral_accel_p95",
    ]:
        vals = collect(metric)
        if not vals:
            print(f"{metric}: n/a")
            continue
        print(
            f"{metric}: median={_q(vals,0.5):.3f} p25={_q(vals,0.25):.3f} "
            f"p75={_q(vals,0.75):.3f}"
        )

    class_counts: Dict[str, int] = {}
    reason_counts: Dict[str, int] = {}
    for t in trials:
        cls = t.feasibility_classification or "unknown"
        class_counts[cls] = class_counts.get(cls, 0) + 1
        for k, v in t.stale_reason_counts.items():
            reason_counts[k] = reason_counts.get(k, 0) + v
    print(f"feasibility_class_counts: {class_counts}")
    if reason_counts:
        top = sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
        print(f"top_stale_reasons: {top}")


def _run_once(track_yaml: str, duration_s: int, start_t: Optional[float] = None) -> Path:
    before = set(RECORDINGS_DIR.glob("*.h5"))
    cmd = [
        str(START_SCRIPT),
        "--run-unity-player",
        "--duration",
        str(duration_s),
        "--track-yaml",
        str(Path(track_yaml).resolve()),
        "--force",
    ]
    if start_t is not None:
        cmd.extend(["--start-t", f"{float(start_t):.6f}"])
    subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
    )
    time.sleep(1.0)
    rec = _latest_recording(before)
    if rec is None:
        raise RuntimeError("No new recording produced")
    return rec


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch A/B runner with robust stats.")
    parser.add_argument("--param", required=True, help="Dotted config key, e.g. control.lateral.kp")
    parser.add_argument(
        "--a",
        required=True,
        help="Baseline value (YAML scalar, e.g. 0.2, true, 12)",
    )
    parser.add_argument(
        "--b",
        required=True,
        help="Treatment value (YAML scalar, e.g. 0.3, false, 20)",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Number of A/B pairs")
    parser.add_argument("--track-yaml", default=str(REPO_ROOT / "tracks" / "s_loop.yml"))
    parser.add_argument("--duration", type=int, default=40)
    parser.add_argument("--curve-start-frame", type=int, default=174)
    parser.add_argument("--entry-start-distance-m", type=float, default=25.0)
    parser.add_argument("--entry-window-distance-m", type=float, default=8.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=20260217,
        help="Seed for deterministic per-pair start-t sampling",
    )
    parser.add_argument(
        "--fixed-start-t",
        type=float,
        default=None,
        help="If set, use this exact start_t for every pair (overrides seeded sampling)",
    )
    parser.add_argument(
        "--emergency-stop-end-run-after-seconds",
        type=float,
        default=None,
        help=(
            "Override safety.emergency_stop_end_run_after_seconds during batch runs "
            "(e.g. 0.5 for faster fail-fast loops)"
        ),
    )
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    value_a = yaml.safe_load(args.a)
    value_b = yaml.safe_load(args.b)

    original_text = CONFIG_PATH.read_text()
    trials_a: List[TrialMetrics] = []
    trials_b: List[TrialMetrics] = []
    rng = np.random.default_rng(args.seed)

    try:
        for i in range(args.repeats):
            pair_start_t = (
                float(args.fixed_start_t)
                if args.fixed_start_t is not None
                else float(rng.uniform(0.0, 1.0))
            )
            print(f"\n--- Pair {i+1}/{args.repeats}: A then B ---")
            if args.fixed_start_t is None:
                print(f"[pair] start_t={pair_start_t:.6f} (seed={args.seed})")
            else:
                print(f"[pair] start_t={pair_start_t:.6f} (fixed)")
            for label, value, store in [("A", value_a, trials_a), ("B", value_b, trials_b)]:
                cfg = yaml.safe_load(CONFIG_PATH.read_text())
                _set_nested(cfg, args.param, value)
                if args.emergency_stop_end_run_after_seconds is not None:
                    _set_nested(
                        cfg,
                        "safety.emergency_stop_end_run_after_seconds",
                        float(args.emergency_stop_end_run_after_seconds),
                    )
                CONFIG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False))
                print(f"[{label}] {args.param}={value}")
                if args.emergency_stop_end_run_after_seconds is not None:
                    print(
                        f"[{label}] safety.emergency_stop_end_run_after_seconds="
                        f"{float(args.emergency_stop_end_run_after_seconds):.3f}"
                    )
                rec = _run_once(
                    track_yaml=args.track_yaml,
                    duration_s=args.duration,
                    start_t=pair_start_t,
                )
                metrics = _analyze_recording(
                    rec,
                    curve_start_frame=args.curve_start_frame,
                    entry_start_distance_m=args.entry_start_distance_m,
                    entry_window_distance_m=args.entry_window_distance_m,
                )
                store.append(metrics)
                print(
                    f"[{label}] recording={metrics.recording} cross={metrics.centerline_cross_frame} "
                    f"fail={metrics.first_failure_frame} ttf={metrics.time_to_failure_s} "
                    f"class={metrics.feasibility_classification} gap={metrics.authority_gap_mean} "
                    f"transfer={metrics.transfer_ratio_mean} stale={metrics.stale_pct} "
                    f"jerk_p95={metrics.jerk_p95} lat_jerk_p95={metrics.lateral_jerk_p95}"
                )
    finally:
        CONFIG_PATH.write_text(original_text)

    _summarize("A", trials_a)
    _summarize("B", trials_b)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
