#!/usr/bin/env python3
"""
Curve-gated policy sweep harness (canonical Stage-1 settings).

Runs policy candidates on s_loop with:
- fixed start_t=0.0 by default
- fail-fast emergency-stop run termination
- per-run envelope scoring from analyze_phase_to_failure.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "av_stack_config.yaml"
RECORDINGS_DIR = REPO_ROOT / "data" / "recordings"
REPORTS_DIR = REPO_ROOT / "data" / "reports" / "sweeps"
START_SCRIPT = REPO_ROOT / "start_av_stack.sh"
PHASE_ANALYZER = REPO_ROOT / "tools" / "analyze" / "analyze_phase_to_failure.py"
ENVELOPE_CFG = REPO_ROOT / "tools" / "analyze" / "curve_authority_envelope.yaml"

def _set_nested(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
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


def _run_once(track_yaml: str, duration_s: int, start_t: float) -> Path:
    before = set(RECORDINGS_DIR.glob("*.h5"))
    cmd = [
        str(START_SCRIPT),
        "--run-unity-player",
        "--duration",
        str(duration_s),
        "--track-yaml",
        str(Path(track_yaml).resolve()),
        "--start-t",
        f"{float(start_t):.6f}",
        "--force",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    time.sleep(1.0)
    rec = _latest_recording(before)
    if rec is None:
        raise RuntimeError("No new recording produced")
    return rec


def _parse_envelope_passes(recording: Path) -> Dict[str, int]:
    cmd = [
        sys.executable,
        str(PHASE_ANALYZER),
        str(recording),
        "--envelope-config",
        str(ENVELOPE_CFG),
    ]
    out = subprocess.check_output(cmd, cwd=REPO_ROOT, text=True)
    out_lines = [x.strip() for x in out.splitlines() if x.strip()]
    passes: Dict[str, int] = {"curve_entry": 0, "curve_commit": 0, "curve_steady": 0}
    for line in out_lines:
        if not line.startswith("curve_"):
            continue
        parts = line.split(",")
        if len(parts) < 14:
            continue
        phase = parts[0]
        envelope_pass = int(float(parts[12]))
        if phase in passes:
            passes[phase] = envelope_pass
    return passes


def _first_true(mask: np.ndarray) -> Optional[int]:
    idx = np.where(mask)[0]
    return int(idx[0]) if len(idx) else None


def _find_centerline_cross(signal: np.ndarray, curve_start_frame: int, min_abs: float = 0.15, persist_frames: int = 4) -> Optional[int]:
    if len(signal) == 0:
        return None
    init = float(np.median(signal[: min(60, len(signal))]))
    initial_sign = int(np.sign(init)) if abs(init) >= 1e-3 else int(np.sign(signal[0]))
    if initial_sign == 0:
        return None
    for i in range(max(0, curve_start_frame), len(signal) - persist_frames + 1):
        w = signal[i : i + persist_frames]
        if np.any(np.abs(w) < min_abs):
            continue
        if np.all(np.sign(w) == -initial_sign):
            return i
    return None


def _analyze_recording_simple(recording_path: Path, curve_start_frame: int) -> Dict[str, Optional[float]]:
    with h5py.File(recording_path, "r") as f:
        ts = np.asarray(f["vehicle/timestamps"][:], dtype=float)
        lce = np.asarray(f["vehicle/road_frame_lane_center_error"][:], dtype=float)
        em = np.asarray(f["control/emergency_stop"][:]).astype(bool)
        jerk = (
            np.asarray(f["vehicle/jerk"][:], dtype=float)
            if "vehicle/jerk" in f
            else np.asarray([], dtype=float)
        )
    cross = _find_centerline_cross(lce, curve_start_frame=curve_start_frame)
    offroad = _first_true(em)
    fail_candidates = [x for x in [cross, offroad] if x is not None]
    fail = min(fail_candidates) if fail_candidates else None
    ttf = float(ts[fail] - ts[0]) if fail is not None and len(ts) else None
    jerk_p95 = float(np.quantile(np.abs(jerk), 0.95)) if jerk.size else None
    return {
        "centerline_cross_frame": cross,
        "offroad_frame": offroad,
        "first_failure_frame": fail,
        "time_to_failure_s": ttf,
        "jerk_p95": jerk_p95,
    }


@dataclass
class Candidate:
    name: str
    params: Dict[str, Any]


def _default_candidates() -> List[Candidate]:
    return [
        Candidate(
            "baseline_off",
            {
                "lateral_control.curve_entry_schedule_enabled": False,
                "lateral_control.curve_commit_mode_enabled": False,
                "lateral_control.curve_entry_assist_enabled": False,
            },
        ),
        Candidate(
            "entry_schedule_soft",
            {
                "lateral_control.curve_entry_schedule_enabled": True,
                "lateral_control.curve_entry_schedule_frames": 24,
                "lateral_control.curve_entry_schedule_min_rate": 0.22,
                "lateral_control.curve_entry_schedule_min_jerk": 0.16,
                "lateral_control.curve_entry_schedule_min_hold_frames": 8,
                "lateral_control.curve_commit_mode_enabled": False,
                "lateral_control.curve_entry_assist_enabled": False,
            },
        ),
        Candidate(
            "entry_commit_balanced",
            {
                "lateral_control.curve_entry_schedule_enabled": True,
                "lateral_control.curve_entry_schedule_frames": 28,
                "lateral_control.curve_entry_schedule_min_rate": 0.24,
                "lateral_control.curve_entry_schedule_min_jerk": 0.18,
                "lateral_control.curve_entry_schedule_min_hold_frames": 10,
                "lateral_control.curve_commit_mode_enabled": True,
                "lateral_control.curve_commit_mode_max_frames": 36,
                "lateral_control.curve_commit_mode_min_rate": 0.22,
                "lateral_control.curve_commit_mode_min_jerk": 0.16,
            },
        ),
        Candidate(
            "entry_commit_aggressive",
            {
                "lateral_control.curve_entry_schedule_enabled": True,
                "lateral_control.curve_entry_schedule_frames": 32,
                "lateral_control.curve_entry_schedule_min_rate": 0.26,
                "lateral_control.curve_entry_schedule_min_jerk": 0.20,
                "lateral_control.curve_entry_schedule_min_hold_frames": 12,
                "lateral_control.curve_commit_mode_enabled": True,
                "lateral_control.curve_commit_mode_max_frames": 42,
                "lateral_control.curve_commit_mode_min_rate": 0.24,
                "lateral_control.curve_commit_mode_min_jerk": 0.18,
            },
        ),
    ]


def _score_candidate(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    fail_frames = []
    emergency_flags = []
    envelope_pass_rates = []
    jerk_p95 = []
    for s in samples:
        ff = s.get("first_failure_frame")
        fail_frames.append(float(ff if ff is not None else 9999.0))
        emergency_flags.append(1.0 if s.get("offroad_frame") is not None else 0.0)
        envelope_pass_rates.append(float(s.get("envelope_pass_rate", 0.0)))
        jp = s.get("jerk_p95")
        if jp is not None:
            jerk_p95.append(float(jp))
    med_fail = float(np.median(fail_frames)) if fail_frames else 0.0
    emer_rate = float(np.mean(emergency_flags) * 100.0) if emergency_flags else 100.0
    env_rate = float(np.mean(envelope_pass_rates) * 100.0) if envelope_pass_rates else 0.0
    med_jerk = float(np.median(jerk_p95)) if jerk_p95 else 0.0
    score = med_fail + (1.25 * env_rate) - (2.0 * emer_rate) - (15.0 * med_jerk)
    return {
        "median_failure_frame": med_fail,
        "emergency_stop_rate_pct": emer_rate,
        "envelope_pass_rate_pct": env_rate,
        "median_jerk_p95": med_jerk,
        "score": score,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run curve-gated policy candidate sweep.")
    parser.add_argument("--track-yaml", default=str(REPO_ROOT / "tracks" / "s_loop.yml"))
    parser.add_argument("--duration", type=int, default=35)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--fixed-start-t", type=float, default=0.0)
    parser.add_argument("--curve-start-frame", type=int, default=174)
    parser.add_argument("--entry-start-distance-m", type=float, default=25.0)
    parser.add_argument("--entry-window-distance-m", type=float, default=8.0)
    parser.add_argument("--emergency-stop-end-run-after-seconds", type=float, default=0.5)
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPORTS_DIR / f"s_loop_curve_sweep_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    original_text = CONFIG_PATH.read_text()
    results: List[Dict[str, Any]] = []
    candidates = _default_candidates()

    try:
        for c in candidates:
            print(f"\n=== Candidate: {c.name} ===")
            samples: List[Dict[str, Any]] = []
            for i in range(args.repeats):
                cfg = yaml.safe_load(CONFIG_PATH.read_text())
                _set_nested(
                    cfg,
                    "safety.emergency_stop_end_run_after_seconds",
                    float(args.emergency_stop_end_run_after_seconds),
                )
                for k, v in c.params.items():
                    _set_nested(cfg, k, v)
                CONFIG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False))

                rec = _run_once(args.track_yaml, args.duration, args.fixed_start_t)
                metrics = _analyze_recording_simple(rec, curve_start_frame=args.curve_start_frame)
                envelope_passes = _parse_envelope_passes(rec)
                envelope_pass_rate = float(np.mean(list(envelope_passes.values())))
                sample = {
                    "recording": rec.name,
                    "first_failure_frame": metrics.get("first_failure_frame"),
                    "offroad_frame": metrics.get("offroad_frame"),
                    "time_to_failure_s": metrics.get("time_to_failure_s"),
                    "jerk_p95": metrics.get("jerk_p95"),
                    "envelope_passes": envelope_passes,
                    "envelope_pass_rate": envelope_pass_rate,
                }
                samples.append(sample)
                print(
                    f"[{c.name} #{i+1}] rec={rec.name} fail={metrics.get('first_failure_frame')} "
                    f"offroad={metrics.get('offroad_frame')} env_pass={envelope_passes}"
                )

            agg = _score_candidate(samples)
            results.append(
                {
                    "name": c.name,
                    "params": c.params,
                    "samples": samples,
                    "aggregate": agg,
                }
            )
    finally:
        CONFIG_PATH.write_text(original_text)

    ranked = sorted(results, key=lambda x: float(x["aggregate"]["score"]), reverse=True)
    payload = {"ranked": ranked, "config": vars(args)}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))

    lines = ["# Curve-Gated Policy Sweep", "", f"Output dir: `{out_dir}`", "", "## Ranking", ""]
    for i, r in enumerate(ranked, start=1):
        a = r["aggregate"]
        lines.append(
            f"{i}. `{r['name']}` score={a['score']:.2f} "
            f"fail_med={a['median_failure_frame']:.1f} "
            f"emergency_pct={a['emergency_stop_rate_pct']:.1f} "
            f"env_pass_pct={a['envelope_pass_rate_pct']:.1f} "
            f"jerk_p95_med={a['median_jerk_p95']:.3f}"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote sweep results to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
