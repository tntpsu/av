"""
ACC Pipeline Analysis CLI — Step 5.

Mirrors mpc_pipeline_analysis.py. Four analysis cards:
  1. Radar Health    — detection rate, SNR, dropout events, raw vs filtered gap
  2. IDM State       — gap error, range rate, s* vs actual gap
  3. Safety Layer    — TTC distribution, emergency brake, fallback transitions
  4. Worst Frames    — top-N frames by |acc_gap_error_m| with context

Usage:
    python tools/analyze/acc_pipeline_analysis.py --latest
    python tools/analyze/acc_pipeline_analysis.py --file data/recordings/foo.h5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from scoring_registry import (
    ACC_TTC_CRITICAL_S,
    ACC_TTC_WARNING_S,
    ACC_TTC_MIN_GATE_S,
    ACC_NEAR_MISS_GAP_M,
    ACC_GAP_RMSE_GATE_M,
    ACC_DETECTION_RATE_GATE,
    ACC_JERK_P95_GATE_MPS3,
    ACC_MIN_ACTIVE_FRAME_RATE,
)


def _safe_float(v, default=0.0):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _latest_recording() -> Path:
    recordings_dir = REPO_ROOT / "data" / "recordings"
    files = sorted(recordings_dir.glob("*.h5"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No .h5 files in {recordings_dir}")
    return files[-1]


def _load_acc_arrays(path: Path) -> dict | None:
    """Load all ACC-related arrays from HDF5. Returns None if ACC not present/active."""
    with h5py.File(path, "r") as f:
        if "vehicle/acc_active" not in f:
            return None

        n = len(f["vehicle/acc_active"])

        def arr(key, default=None):
            if key not in f:
                return np.full(n, default) if default is not None else None
            return np.array(f[key][:n], dtype=float)

        acc_active = arr("vehicle/acc_active", 0.0)
        acc_active_pct = float(np.mean(acc_active > 0.5))
        if acc_active_pct < ACC_MIN_ACTIVE_FRAME_RATE:
            return None

        return {
            "n": n,
            "acc_active": acc_active,
            "acc_active_pct": acc_active_pct,
            "detected": arr("vehicle/radar_fwd_detected", 0.0),
            "distance": arr("vehicle/radar_fwd_distance_m"),
            "range_rate": arr("vehicle/radar_fwd_range_rate_mps"),
            "snr": arr("vehicle/radar_fwd_snr"),
            "acc_active_flag": acc_active,
            "acc_ttc_s": arr("vehicle/acc_ttc_s"),
            "acc_gap_error": arr("vehicle/acc_gap_error_m"),
            "acc_target_gap": arr("vehicle/acc_target_gap_m"),
            "speed": arr("vehicle/speed"),
        }


def _card1_radar_health(d: dict) -> None:
    print("\n" + "=" * 72)
    print("  CARD 1 — RADAR HEALTH")
    print("=" * 72)

    detected = d["detected"]
    snr = d["snr"]
    distance = d["distance"]
    n = d["n"]

    detection_rate = float(np.mean(detected > 0.5))
    gate = "PASS" if detection_rate >= ACC_DETECTION_RATE_GATE else "FAIL"
    print(f"  Detection Rate:   {detection_rate * 100:.1f}%  [{gate} ≥ {ACC_DETECTION_RATE_GATE * 100:.0f}%]")

    if snr is not None and not np.all(np.isnan(snr)):
        snr_valid = snr[np.isfinite(snr)]
        print(f"  SNR:              P05={np.percentile(snr_valid, 5):.2f}  "
              f"P50={np.percentile(snr_valid, 50):.2f}  "
              f"P95={np.percentile(snr_valid, 95):.2f}  "
              f"max={float(np.max(snr_valid)):.2f}")

    # Dropout events (runs of no-detect ≥ 3 frames)
    no_detect = detected < 0.5
    dropout_events = 0
    run_len = 0
    for nd in no_detect:
        if nd:
            run_len += 1
            if run_len == 3:
                dropout_events += 1
        else:
            run_len = 0
    print(f"  Dropout Events:   {dropout_events}  (≥3 consecutive no-detect frames)")

    if distance is not None:
        dist_valid = distance[np.isfinite(distance) & (distance > 0.0)]
        if dist_valid.size > 0:
            print(f"  Gap range (when detected): "
                  f"min={float(np.min(dist_valid)):.1f}m  "
                  f"P50={np.percentile(dist_valid, 50):.1f}m  "
                  f"max={float(np.max(dist_valid)):.1f}m")


def _card2_idm_state(d: dict) -> None:
    print("\n" + "=" * 72)
    print("  CARD 2 — IDM STATE QUALITY")
    print("=" * 72)

    acc_mask = d["acc_active"] > 0.5
    gap_error = d["acc_gap_error"]
    target_gap = d["acc_target_gap"]
    distance = d["distance"]

    if gap_error is not None:
        ge_acc = gap_error[acc_mask & np.isfinite(gap_error)]
        if ge_acc.size > 0:
            rmse = float(np.sqrt(np.mean(ge_acc ** 2)))
            gate = "PASS" if rmse <= ACC_GAP_RMSE_GATE_M else "FAIL"
            print(f"  Gap Error RMSE:   {rmse:.3f}m  [{gate} ≤ {ACC_GAP_RMSE_GATE_M}m]")
            print(f"  Gap Error P05/P95: {np.percentile(ge_acc, 5):.2f}m / {np.percentile(ge_acc, 95):.2f}m")
            sign_changes = int(np.sum(np.diff(np.sign(ge_acc)) != 0))
            duration_min = max(1.0, len(ge_acc) / 30.0 / 60.0)
            print(f"  IDM Sign Changes:  {sign_changes / duration_min:.1f}/min  "
                  f"(>20/min = hunting pattern)")

    if target_gap is not None and distance is not None:
        both_valid = acc_mask & np.isfinite(target_gap) & np.isfinite(distance) & (distance > 0.0)
        if np.any(both_valid):
            ratio = distance[both_valid] / np.maximum(target_gap[both_valid], 0.1)
            print(f"  Gap/s* ratio:      P05={np.percentile(ratio, 5):.2f}  "
                  f"P50={np.percentile(ratio, 50):.2f}  "
                  f"P95={np.percentile(ratio, 95):.2f}  "
                  f"(1.0 = exactly at desired gap)")

    range_rate = d["range_rate"]
    if range_rate is not None:
        rr_acc = range_rate[acc_mask & np.isfinite(range_rate)]
        if rr_acc.size > 0:
            print(f"  Range Rate:        P05={np.percentile(rr_acc, 5):.2f} m/s  "
                  f"P95={np.percentile(rr_acc, 95):.2f} m/s  "
                  f"(+closing, −pulling away)")


def _card3_safety_timeline(d: dict) -> None:
    print("\n" + "=" * 72)
    print("  CARD 3 — SAFETY LAYER STATUS")
    print("=" * 72)

    acc_mask = d["acc_active"] > 0.5
    ttc = d["acc_ttc_s"]

    if ttc is not None:
        ttc_acc = ttc[acc_mask & np.isfinite(ttc)]
        if ttc_acc.size > 0:
            ttc_min = float(np.min(ttc_acc))
            ttc_p05 = float(np.percentile(ttc_acc, 5))
            gate_min = "PASS" if ttc_min >= ACC_TTC_MIN_GATE_S else "FAIL"
            gate_p05 = "PASS" if ttc_p05 >= ACC_TTC_MIN_GATE_S else "FAIL"
            print(f"  TTC Minimum:       {ttc_min:.2f}s  [{gate_min} ≥ {ACC_TTC_MIN_GATE_S}s]")
            print(f"  TTC P05:           {ttc_p05:.2f}s  [{gate_p05} ≥ {ACC_TTC_MIN_GATE_S}s]")
            ttc_crit_pct = float(np.mean(ttc_acc < ACC_TTC_CRITICAL_S) * 100.0)
            ttc_warn_pct = float(np.mean(
                (ttc_acc >= ACC_TTC_CRITICAL_S) & (ttc_acc < ACC_TTC_WARNING_S)
            ) * 100.0)
            print(f"  TTC < {ACC_TTC_CRITICAL_S}s (e-stop):    {ttc_crit_pct:.1f}% of ACC frames")
            print(f"  TTC {ACC_TTC_CRITICAL_S}–{ACC_TTC_WARNING_S}s (warn):    {ttc_warn_pct:.1f}% of ACC frames")

    # Near-miss events
    distance = d["distance"]
    if distance is not None:
        nm_frames = acc_mask & (distance > 0.0) & (distance < ACC_NEAR_MISS_GAP_M)
        nm_count = 0
        run_len = 0
        for nd in nm_frames:
            if nd:
                run_len += 1
                if run_len == 3:
                    nm_count += 1
            else:
                run_len = 0
        print(f"  Near-Miss Events:  {nm_count}  (gap < {ACC_NEAR_MISS_GAP_M}m, sustained ≥3 frames)")
        collision_count = int(np.sum(distance < 0.0))
        print(f"  Collision Frames:  {collision_count}  (distance < 0)")


def _card4_worst_frames(d: dict, top_n: int = 5) -> None:
    print("\n" + "=" * 72)
    print(f"  CARD 4 — WORST FRAME INSPECTOR (top {top_n} by |gap_error|)")
    print("=" * 72)

    gap_error = d["acc_gap_error"]
    acc_mask = d["acc_active"] > 0.5
    if gap_error is None:
        print("  (no acc_gap_error_m data)")
        return

    abs_ge = np.abs(gap_error)
    abs_ge[~acc_mask] = 0.0
    abs_ge[~np.isfinite(abs_ge)] = 0.0

    top_frames = np.argsort(abs_ge)[-top_n:][::-1]
    print(f"  {'Frame':>6}  {'|gap_err|':>9}  {'gap_m':>7}  {'ttc_s':>6}  "
          f"{'rr_mps':>7}  {'speed':>6}  {'acc_active':>10}")
    print("  " + "-" * 62)
    for fr in top_frames:
        ge = _safe_float(gap_error[fr] if gap_error is not None else None, 0.0)
        gap = _safe_float(d["distance"][fr] if d["distance"] is not None else None, 0.0)
        ttc = _safe_float(d["acc_ttc_s"][fr] if d["acc_ttc_s"] is not None else None, 999.0)
        rr = _safe_float(d["range_rate"][fr] if d["range_rate"] is not None else None, 0.0)
        spd = _safe_float(d["speed"][fr] if d["speed"] is not None else None, 0.0)
        active = _safe_float(d["acc_active"][fr])
        print(f"  {fr:>6d}  {abs(ge):>9.3f}  {gap:>7.2f}  "
              f"{'999+' if ttc > 99 else f'{ttc:>6.2f}'}  "
              f"{rr:>7.2f}  {spd:>6.1f}  {active:>10.0f}")


def run_analysis(path: Path) -> None:
    print(f"\nACC Pipeline Analysis: {path.name}")

    d = _load_acc_arrays(path)
    if d is None:
        print("  ACC not active in this recording (acc_active_pct below threshold).")
        return

    print(f"  ACC Active: {d['acc_active_pct'] * 100:.1f}%  ({d['n']} total frames)")

    _card1_radar_health(d)
    _card2_idm_state(d)
    _card3_safety_timeline(d)
    _card4_worst_frames(d)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="ACC pipeline diagnostic analysis")
    parser.add_argument("--file", type=Path, help="Recording file path")
    parser.add_argument("--latest", action="store_true", help="Analyze latest recording")
    parser.add_argument("--top-n", type=int, default=5, help="Worst frames to show (card 4)")
    args = parser.parse_args()

    if args.latest:
        path = _latest_recording()
    elif args.file:
        path = args.file
    else:
        parser.print_help()
        sys.exit(1)

    if not path.exists():
        print(f"Error: file not found: {path}")
        sys.exit(1)

    run_analysis(path)


if __name__ == "__main__":
    main()
