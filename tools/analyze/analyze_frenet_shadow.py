"""Frenet-frame MPC reference — Phase 0 offline verification.

Computes the linearized Frenet d signal from an existing recording and compares
it against the current MPC e_lat input (lookahead offset). The pass criterion
is that d_linearized P95 on a straight-road recording drops far below the
observed lookahead-offset P95, which would confirm the decomposition:

    lookahead_offset = d_frenet + Ld·sin(e_heading) + Ld²·κ/2 + O(Ld³)

Usage
-----
    python3 tools/analyze/analyze_frenet_shadow.py <recording.h5> [<recording2.h5> ...]

Default: runs against the two plan-specified recordings if no argv given.

Pass criteria (per plan .claude/plans/greedy-swimming-naur.md):
    - Step 0.1: heading-error signal distribution is not obviously broken
      (finite, reasonable magnitude, no frozen-value pathology).
    - Step 0.2: `d_linearized` P95 on H2 < 0.4 m (down from observed
      lookahead-offset P95 ≈ 1.34 m).
    - Step 0.3: On hairpin, `d_linearized` P95 < 0.8 m.

Decision gate: all three pass → proceed to Phase A. Any fail → re-diagnose.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import h5py
import numpy as np


REQUIRED_FIELDS = (
    "control/mpc_gt_cross_track_lookahead_m",
    "control/mpc_gt_cross_track_at_car_m",
    "control/mpc_gt_heading_error_rad",
    "control/pp_lookahead_distance",
)
OPTIONAL_FIELDS = (
    "control/curvature_primary_abs",
    "control/mpc_e_lat_reference_divergence_m",
    "control/mpc_e_lat_reference_source",
    "ground_truth/desired_heading",
    "ground_truth/path_curvature",
)


def _pct_abs(x: np.ndarray) -> Dict[str, float]:
    """Return abs-mean, P50, P95, max on finite entries only."""
    ax = np.abs(x[np.isfinite(x)])
    if ax.size == 0:
        return {"n": 0, "abs_mean": float("nan"), "p50": float("nan"),
                "p95": float("nan"), "max": float("nan")}
    return {
        "n": int(ax.size),
        "abs_mean": float(np.mean(ax)),
        "p50": float(np.percentile(ax, 50)),
        "p95": float(np.percentile(ax, 95)),
        "max": float(np.max(ax)),
    }


def analyze(path: Path) -> Dict[str, object]:
    with h5py.File(path, "r") as f:
        missing = [k for k in REQUIRED_FIELDS if k not in f]
        if missing:
            return {"path": str(path), "error": f"missing fields: {missing}"}

        lookahead = np.asarray(f["control/mpc_gt_cross_track_lookahead_m"])
        at_car = np.asarray(f["control/mpc_gt_cross_track_at_car_m"])
        e_heading = np.asarray(f["control/mpc_gt_heading_error_rad"])
        Ld = np.asarray(f["control/pp_lookahead_distance"])

        kappa = (np.asarray(f["control/curvature_primary_abs"])
                 if "control/curvature_primary_abs" in f else None)
        existing_div = (np.asarray(f["control/mpc_e_lat_reference_divergence_m"])
                        if "control/mpc_e_lat_reference_divergence_m" in f else None)
        ref_src = (f["control/mpc_e_lat_reference_source"].asstr()[...]
                   if "control/mpc_e_lat_reference_source" in f else None)

        # Only use frames where MPC was active with a finite Ld.
        valid = np.isfinite(lookahead) & np.isfinite(e_heading) & np.isfinite(Ld) & (Ld > 0)
        # Optionally restrict to frames where the MPC reference was actually
        # the lookahead signal (shadow-mode-neutral baseline).
        if ref_src is not None:
            active = np.array([s in ("lookahead_gt", "at_car_gt", "fallback_pp") for s in ref_src])
            valid &= active

    n_total = lookahead.shape[0]
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return {"path": str(path), "error": "no valid frames"}

    lookahead_v = lookahead[valid]
    at_car_v = at_car[valid]
    e_heading_v = e_heading[valid]
    Ld_v = Ld[valid]

    # Four sign-convention candidates for linearized Frenet d. Which one
    # matches at-car (≈ true d_frenet) best will reveal the sign convention.
    heading_leak = Ld_v * np.sin(e_heading_v)
    variants = {
        "A: lookahead - Ld·sin(eh)":   lookahead_v - heading_leak,
        "B: lookahead + Ld·sin(eh)":   lookahead_v + heading_leak,
        "C: -(lookahead - Ld·sin(eh))": -(lookahead_v - heading_leak),
        "D: -(lookahead + Ld·sin(eh))": -(lookahead_v + heading_leak),
    }
    variant_stats = {}
    for label, arr in variants.items():
        rms_err = float(np.sqrt(np.mean((arr - at_car_v) ** 2)))
        corr = float(np.corrcoef(arr, at_car_v)[0, 1]) if n_valid > 1 else float("nan")
        variant_stats[label] = {
            "stats": _pct_abs(arr),
            "rms_err_vs_at_car": rms_err,
            "corr_vs_at_car": corr,
        }

    # Also report: P95 of |d_candidate| for each variant (smaller = more
    # Frenet-like = less heading-leakage contamination).

    kappa_leak = None
    if kappa is not None:
        kappa_v = kappa[valid]
        kappa_leak = 0.5 * (Ld_v ** 2) * kappa_v * np.sign(lookahead_v)

    return {
        "path": str(path),
        "n_total": n_total,
        "n_valid": n_valid,
        "Ld": {"mean": float(np.mean(Ld_v)), "p50": float(np.percentile(Ld_v, 50)),
               "p95": float(np.percentile(Ld_v, 95))},
        "e_heading_rad": _pct_abs(e_heading_v),
        "lookahead_offset_m": _pct_abs(lookahead_v),
        "at_car_m": _pct_abs(at_car_v),
        "heading_leak_m": _pct_abs(heading_leak),
        "kappa_leak_m": _pct_abs(kappa_leak) if kappa_leak is not None else None,
        "existing_divergence_m": _pct_abs(existing_div) if existing_div is not None else None,
        "variants": variant_stats,
    }


def _print_row(label: str, stats: Dict[str, float]) -> None:
    if stats is None:
        print(f"  {label:28s} (unavailable)")
        return
    print(f"  {label:28s} n={stats['n']:5d}  |mean|={stats['abs_mean']:.4f}  "
          f"P50={stats['p50']:.4f}  P95={stats['p95']:.4f}  max={stats['max']:.4f}")


def main(argv: list[str]) -> int:
    if not argv:
        argv = [
            "data/recordings/recording_20260418_195233.h5",  # H2
            "data/recordings/recording_20260418_105316.h5",  # hairpin
        ]

    all_pass = True
    for arg in argv:
        p = Path(arg)
        if not p.exists():
            print(f"[SKIP] {p} — not found")
            all_pass = False
            continue

        print("=" * 80)
        print(f"RECORDING: {p.name}")
        print("=" * 80)

        r = analyze(p)
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            all_pass = False
            continue

        print(f"  frames total = {r['n_total']}   valid (lookahead active) = {r['n_valid']}")
        print(f"  Ld m: mean={r['Ld']['mean']:.2f}  P50={r['Ld']['p50']:.2f}  P95={r['Ld']['p95']:.2f}")
        print()
        _print_row("e_heading (rad)",           r["e_heading_rad"])
        _print_row("lookahead_offset (m)",      r["lookahead_offset_m"])
        _print_row("at_car (m)",                r["at_car_m"])
        _print_row("heading_leak = Ld·sin(eh)", r["heading_leak_m"])
        _print_row("kappa_leak = Ld²κ/2",       r["kappa_leak_m"])
        _print_row("existing divergence",       r["existing_divergence_m"])
        print()
        print("  CANDIDATE FORMULAS (sort by RMS-error vs at_car):")
        ranked = sorted(r["variants"].items(), key=lambda kv: kv[1]["rms_err_vs_at_car"])
        for label, v in ranked:
            s = v["stats"]
            print(f"    {label:32s}  |mean|={s['abs_mean']:.4f}  P95={s['p95']:.4f}"
                  f"  rms_err={v['rms_err_vs_at_car']:.4f}  corr={v['corr_vs_at_car']:+.4f}")

        best_label, best = ranked[0]
        d_p95 = best["stats"]["p95"]
        look_p95 = r["lookahead_offset_m"]["p95"]
        tag = "h2" if "195233" in p.name else ("hairpin" if "105316" in p.name else "other")

        print()
        print("  GATE EVALUATION (against BEST candidate):")
        print(f"    Best formula: {best_label}  (RMS err vs at_car = {best['rms_err_vs_at_car']:.4f} m)")
        if tag == "h2":
            ok = d_p95 < 0.4
            print(f"    [Step 0.2] best_candidate P95 < 0.4 m on H2:    {d_p95:.4f} m   "
                  f"{'PASS' if ok else 'FAIL'}")
            print(f"               (lookahead_offset P95 = {look_p95:.4f} m — the baseline being corrected)")
            if not ok:
                all_pass = False
        elif tag == "hairpin":
            ok = d_p95 < 0.8
            print(f"    [Step 0.3] best_candidate P95 < 0.8 m on hairpin: {d_p95:.4f} m   "
                  f"{'PASS' if ok else 'FAIL'}")
            if not ok:
                all_pass = False
        else:
            print(f"    (untagged — best P95 = {d_p95:.4f} m)")

        corr_ok = best["corr_vs_at_car"] > 0.5
        print(f"    corr(best, at_car) > 0.5:                       "
              f"{best['corr_vs_at_car']:+.4f}   {'PASS' if corr_ok else 'FAIL'}")
        if not corr_ok:
            all_pass = False
        print()

    print("=" * 80)
    print(f"OVERALL: {'PASS — proceed to Phase A' if all_pass else 'FAIL — re-diagnose before Phase A'}")
    print("=" * 80)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
