#!/usr/bin/env python3
"""MPC Pipeline Analysis — Phase 2.8 diagnostic tool.

Loads any HDF5 recording and outputs a per-regime breakdown of Phase 2.8
pipeline health: regime distribution, recovery suppression, steering feedback
error, rate limiter activations, Smith predictor usage, and top-10 worst
MPC frames for rapid triage.

Usage:
    python tools/analyze/mpc_pipeline_analysis.py --latest
    python tools/analyze/mpc_pipeline_analysis.py --file data/recordings/<name>.h5
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# HDF5 loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_field(f, *keys, default=None):
    """Return np.array for the first key found, or `default`."""
    for k in keys:
        if k in f:
            return np.array(f[k], dtype=float)
    return default


def load_recording(path: Path) -> dict:
    data = {}
    with h5py.File(path, "r") as f:
        n = None

        def _get(key, default=None):
            if key in f:
                arr = np.array(f[key], dtype=float)
                nonlocal n
                if n is None:
                    n = len(arr)
                return arr
            return default

        data["n"] = None  # filled below

        # Core control
        data["regime"]                      = _get("control/regime")
        data["regime_blend_weight"]         = _get("control/regime_blend_weight")
        data["lateral_error"]               = _get("control/lateral_error")
        data["steering"]                    = _get("control/steering")
        data["speed"]                       = _get("vehicle/speed")

        # MPC state quality
        data["mpc_e_lat"]                   = _get("control/mpc_e_lat")
        data["mpc_e_heading"]               = _get("control/mpc_e_heading")
        data["mpc_feasible"]                = _get("control/mpc_feasible")
        data["mpc_solve_time_ms"]           = _get("control/mpc_solve_time_ms")
        data["mpc_kappa_ref"]               = _get("control/mpc_kappa_ref")
        data["mpc_fallback_active"]         = _get("control/mpc_fallback_active")
        data["mpc_consecutive_failures"]    = _get("control/mpc_consecutive_failures")

        # 2.8.1 — Recovery mode suppression
        data["mpc_recovery_mode_suppressed"] = _get("control/mpc_recovery_mode_suppressed")

        # 2.8.2 — Steering feedback error
        data["mpc_last_steering_pre_modify"] = _get("control/mpc_last_steering_pre_modify")
        data["mpc_last_steering_actual"]     = _get("control/mpc_last_steering_actual")

        # 2.8.3 — Rate limiter
        data["mpc_rate_limiter_active"]     = _get("control/mpc_rate_limiter_active")

        # 2.8.4 — Smith predictor
        data["mpc_smith_raw_e_lat"]         = _get("control/mpc_smith_raw_e_lat")
        data["mpc_smith_e_lat_predicted"]   = _get("control/mpc_smith_e_lat_predicted")
        data["mpc_smith_e_heading_predicted"] = _get("control/mpc_smith_e_heading_predicted")
        data["mpc_delay_frames_used"]       = _get("control/mpc_delay_frames_used")

        data["n"] = n or 0
    return data


# ──────────────────────────────────────────────────────────────────────────────
# Regime masks
# ──────────────────────────────────────────────────────────────────────────────

def _regime_masks(data):
    """Return (pp_mask, mpc_mask, blend_mask).

    Regime encoding: 0=PURE_PURSUIT, 1=LINEAR_MPC, 2=NONLINEAR_MPC (future).
    blend_weight is a SEPARATE field — there is no integer "BLEND" regime value.
    blend_mask here marks frames where MPC is active but blend_weight < 1.0
    (i.e., within the ramp-up transition window).
    """
    regime = data.get("regime")
    blend_w = data.get("regime_blend_weight")
    n = data["n"]
    if regime is None or n == 0:
        return np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    regime = regime[:n]
    pp_mask  = regime < 0.5          # regime == 0 (PP)
    mpc_mask = regime >= 0.5         # regime == 1 or 2 (any MPC)
    if blend_w is not None:
        bw = blend_w[:n]
        blend_mask = mpc_mask & (bw < 0.99)   # MPC active but still ramping
    else:
        blend_mask = np.zeros(n, bool)
    return pp_mask, mpc_mask, blend_mask


# ──────────────────────────────────────────────────────────────────────────────
# Per-section statistics helper
# ──────────────────────────────────────────────────────────────────────────────

def _pct(arr, mask, p):
    sub = arr[mask]
    if len(sub) == 0:
        return float("nan")
    return float(np.percentile(sub, p))

def _rate(arr, mask):
    """Fraction of masked frames where arr > 0.5."""
    if arr is None or not np.any(mask):
        return float("nan")
    return float(np.mean(arr[:len(mask)][mask] > 0.5))

def _fmt(v, fmt=".3f", na="N/A"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return na
    return format(v, fmt)


# ──────────────────────────────────────────────────────────────────────────────
# Transition detector
# ──────────────────────────────────────────────────────────────────────────────

def _find_transitions(regime, n):
    transitions = []
    if regime is None or n < 2:
        return transitions
    r = regime[:n]
    for i in range(1, n):
        prev = int(round(r[i - 1]))
        curr = int(round(r[i]))
        if prev != curr:
            transitions.append((i, prev, curr))
    return transitions


# ──────────────────────────────────────────────────────────────────────────────
# Main report
# ──────────────────────────────────────────────────────────────────────────────

def _mode_name(v):
    return {0: "PP", 1: "MPC", 2: "BLEND"}.get(int(round(v)), f"?{v}")


def print_report(path: Path, data: dict):
    n = data["n"]
    pp_mask, mpc_mask, blend_mask = _regime_masks(data)

    pp_n    = int(np.sum(pp_mask))
    mpc_n   = int(np.sum(mpc_mask))
    blend_n = int(np.sum(blend_mask))   # MPC frames still within the blend ramp
    settled_n = mpc_n - blend_n

    sep = "=" * 60
    thin = "-" * 60

    print(f"\n{sep}")
    print(f"MPC PIPELINE ANALYSIS — {path.name}")
    print(f"{sep}")
    print(f"Total frames: {n}")

    # ── Regime distribution ──────────────────────────────────────────────────
    print(f"\n{'Regime distribution':}")
    print(f"  PP    frames: {pp_n:4d} / {n}  ({pp_n/n*100:.1f}%)")
    print(f"  MPC   frames: {mpc_n:4d} / {n}  ({mpc_n/n*100:.1f}%)", end="")
    if blend_n > 0:
        print(f"  [of which {blend_n} ramping / {settled_n} settled]")
    else:
        print()

    # Transitions
    transitions = _find_transitions(data.get("regime"), n)
    pp_to_mpc = [(fr, a, b) for fr, a, b in transitions if a == 0 and b >= 1]
    mpc_to_pp = [(fr, a, b) for fr, a, b in transitions if a >= 1 and b == 0]

    steering = data.get("steering")
    max_delta_at_trans = float("nan")
    if steering is not None and transitions:
        deltas = [abs(steering[min(fr, n-1)] - steering[max(fr-1, 0)]) for fr, _, _ in transitions
                  if 0 < fr < n]
        max_delta_at_trans = max(deltas) if deltas else float("nan")

    print(f"\nRegime transitions: {len(pp_to_mpc)} PP→MPC, {len(mpc_to_pp)} MPC→PP")
    delta_str = _fmt(max_delta_at_trans)
    gate = "PASS" if not np.isnan(max_delta_at_trans) and max_delta_at_trans < 0.05 else "FAIL"
    print(f"  Max |Δsteering| at transition: {delta_str}  [{gate} < 0.05]")

    if mpc_n == 0:
        print("\n[No MPC frames — PP-only recording. Phase 2.8 diagnostics skipped.]\n")
        return

    # ── 2.8.1 Recovery suppression ───────────────────────────────────────────
    print(f"\n{thin}")
    print("2.8.1 — Recovery mode suppression")
    rec = data.get("mpc_recovery_mode_suppressed")
    if rec is not None:
        r_mpc = rec[:n]
        suppressed  = int(np.sum(r_mpc[mpc_mask] > 0.5))
        would_fire_pp = int(np.sum(r_mpc[pp_mask] > 0.5)) if pp_n > 0 else 0
        print(f"  Recovery suppressed (MPC frames): {suppressed} / {mpc_n}  ({suppressed/mpc_n*100:.1f}%)")
        print(f"  Recovery NOT suppressed on PP:    {would_fire_pp} / {pp_n}  ({would_fire_pp/pp_n*100:.1f}% — expected)")
    else:
        print("  [Field mpc_recovery_mode_suppressed not in recording]")

    # ── 2.8.2 Steering feedback error ────────────────────────────────────────
    print(f"\n{thin}")
    print("2.8.2 — Steering feedback error  (pre_modify vs actual_sent)")
    pre  = data.get("mpc_last_steering_pre_modify")
    act  = data.get("mpc_last_steering_actual")
    if pre is not None and act is not None:
        err = np.abs(pre[:n] - act[:n])
        e_mpc = err[mpc_mask]
        p50 = _fmt(np.percentile(e_mpc, 50))
        p95 = _fmt(np.percentile(e_mpc, 95))
        mx  = _fmt(np.max(e_mpc))
        gate_val = float(np.percentile(e_mpc, 95))
        gate = "PASS" if gate_val < 0.01 else "FAIL"
        print(f"  Feedback error P50 / P95 / MAX: {p50} / {p95} / {mx}  [{gate} P95<0.01]")
    else:
        print("  [Fields mpc_last_steering_pre_modify / mpc_last_steering_actual not in recording]")

    # ── 2.8.3 Rate limiter ───────────────────────────────────────────────────
    print(f"\n{thin}")
    print("2.8.3 — MPC rate limiter")
    rl = data.get("mpc_rate_limiter_active")
    if rl is not None:
        rl_mpc = int(np.sum(rl[:n][mpc_mask] > 0.5))
        rate = rl_mpc / mpc_n * 100
        gate = "green" if rate < 5 else ("yellow" if rate < 15 else "red — limiter firing too often")
        print(f"  Activations: {rl_mpc} / {mpc_n} MPC frames  ({rate:.1f}%)  [{gate}]")
    else:
        print("  [Field mpc_rate_limiter_active not in recording]")

    # ── 2.8.4 Smith predictor ────────────────────────────────────────────────
    print(f"\n{thin}")
    print("2.8.4 — Smith predictor delay compensation")
    raw_e  = data.get("mpc_smith_raw_e_lat")
    pred_e = data.get("mpc_smith_e_lat_predicted")
    delay  = data.get("mpc_delay_frames_used")
    if raw_e is not None and pred_e is not None:
        smith_delta = np.abs(pred_e[:n] - raw_e[:n])
        sd_mpc = smith_delta[mpc_mask]
        print(f"  Smith correction |pred-raw| P50 / P95: "
              f"{_fmt(np.percentile(sd_mpc,50))}m / {_fmt(np.percentile(sd_mpc,95))}m")
    else:
        print("  [Smith predictor fields not in recording]")
    if delay is not None:
        d_mpc = delay[:n][mpc_mask]
        p50_d = int(np.median(d_mpc)) if len(d_mpc) > 0 else 0
        print(f"  Delay frames used P50: {p50_d}  (expected: see mpc_delay_frames config)")
    else:
        print("  [Field mpc_delay_frames_used not in recording]")

    # ── MPC state quality ────────────────────────────────────────────────────
    print(f"\n{thin}")
    print("MPC State Quality")
    e_lat = data.get("mpc_e_lat")
    e_hdg = data.get("mpc_e_heading")
    solve = data.get("mpc_solve_time_ms")
    if e_lat is not None:
        el = np.abs(e_lat[:n][mpc_mask])
        print(f"  mpc_e_lat    P50 / P95 / MAX: "
              f"{_fmt(np.percentile(el,50))}m / {_fmt(np.percentile(el,95))}m / {_fmt(np.max(el))}m")
    if e_hdg is not None:
        eh = np.abs(e_hdg[:n][mpc_mask])
        print(f"  mpc_e_heading P50 / P95 / MAX: "
              f"{_fmt(np.percentile(eh,50))}r / {_fmt(np.percentile(eh,95))}r / {_fmt(np.max(eh))}r")
    if solve is not None:
        sv = solve[:n][mpc_mask]
        gate = "PASS" if float(np.percentile(sv, 95)) <= 5.0 else "FAIL"
        print(f"  solve_time_ms P50 / P95 / MAX: "
              f"{_fmt(np.percentile(sv,50),'0.2f')}ms / "
              f"{_fmt(np.percentile(sv,95),'0.2f')}ms / "
              f"{_fmt(np.max(sv),'0.2f')}ms  [{gate} P95≤5ms]")

    fallback = data.get("mpc_fallback_active")
    if fallback is not None:
        fb = int(np.sum(fallback[:n][mpc_mask] > 0.5))
        print(f"  Fallback activations: {fb} / {mpc_n}  ({fb/mpc_n*100:.1f}%)")

    # ── Top-10 worst MPC frames ───────────────────────────────────────────────
    print(f"\n{thin}")
    print("Top-10 MPC frames by |mpc_e_lat|")
    if e_lat is not None:
        mpc_idxs = np.where(mpc_mask)[0]
        abs_e = np.abs(e_lat[:n][mpc_mask])
        top_k = min(10, len(abs_e))
        top_order = np.argsort(abs_e)[::-1][:top_k]

        speed_arr = data.get("speed")
        kappa_arr = data.get("mpc_kappa_ref")
        rec_arr   = data.get("mpc_recovery_mode_suppressed")
        rl_arr    = data.get("mpc_rate_limiter_active")
        sd_arr    = (np.abs(pred_e[:n] - raw_e[:n])
                     if (pred_e is not None and raw_e is not None) else None)

        hdr = f"  {'Frame':>6}  {'e_lat':>7}  {'e_hdg':>7}  {'spd':>5}  {'kappa':>7}  {'RecSup':>6}  {'RtLim':>5}  {'SmithΔ':>7}"
        print(hdr)
        for rank_i in top_order:
            fr = int(mpc_idxs[rank_i])
            el_v   = float(e_lat[fr])
            eh_v   = float(e_hdg[fr]) if e_hdg is not None else float("nan")
            spd_v  = float(speed_arr[fr]) if speed_arr is not None else float("nan")
            kap_v  = float(kappa_arr[fr]) if kappa_arr is not None else float("nan")
            rec_v  = int(rec_arr[fr] > 0.5) if rec_arr is not None else -1
            rl_v   = int(rl_arr[fr] > 0.5) if rl_arr is not None else -1
            sd_v   = float(sd_arr[fr]) if sd_arr is not None else float("nan")
            print(
                f"  {fr:>6d}  {el_v:>+7.3f}  {eh_v:>+7.3f}  "
                f"{spd_v:>5.1f}  {kap_v:>7.4f}  "
                f"{'Y' if rec_v==1 else ('N' if rec_v==0 else '?'):>6}  "
                f"{'Y' if rl_v==1 else ('N' if rl_v==0 else '?'):>5}  "
                f"{_fmt(sd_v, '.4f'):>7}"
            )
    else:
        print("  [mpc_e_lat not available]")

    print(f"\n{sep}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_path(args) -> Path:
    if args.file:
        p = Path(args.file)
        if not p.exists():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            sys.exit(1)
        return p
    recordings_dir = Path("data/recordings")
    if not recordings_dir.exists():
        print("ERROR: data/recordings directory not found", file=sys.stderr)
        sys.exit(1)
    recordings = sorted(recordings_dir.glob("*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not recordings:
        print("ERROR: no recordings found in data/recordings/", file=sys.stderr)
        sys.exit(1)
    return recordings[0]


def main():
    parser = argparse.ArgumentParser(description="MPC Pipeline Analysis — Phase 2.8 diagnostic")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--latest", action="store_true", help="Analyze the most recent recording")
    grp.add_argument("--file", metavar="PATH", help="Path to a specific .h5 recording")
    args = parser.parse_args()

    path = _resolve_path(args)
    data = load_recording(path)
    print_report(path, data)


if __name__ == "__main__":
    main()
