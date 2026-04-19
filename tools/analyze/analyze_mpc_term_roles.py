#!/usr/bin/env python3
"""
analyze_mpc_term_roles.py — Smell 7 gate for decomposed-MPC-reference plan.

Empirically characterizes the closed-loop role of the three terms that
`gt_cross_track_lookahead_m` bundled into MPC's e_lat:

    lookahead = d_at_car + Ld·sin(e_h) + Ld²·κ/2 + O(Ld³)

For each recording, reports correlation of:
  • Ld·sin(e_h)      vs  d/dt(heading_error)   → is it PD damping?
  • Ld·sin(e_h)      vs  heading_error_lead(Δ) → is it preview?
  • Ld²·κ/2          vs  mean(κ_ref_horizon)   → does horizon cover it?

Usage:
    python3 tools/analyze/analyze_mpc_term_roles.py <recording.h5> [<recording.h5> ...]
    python3 tools/analyze/analyze_mpc_term_roles.py --compare  # runs all 4 plan-cited recordings

Written for plan: /Users/philtullai/.claude/plans/decomposed-mpc-reference.md
Smell 7 precedent: project_frenet_mpc_reference.md (G2 activation failed
because removed term's closed-loop role wasn't measured before substitution).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

# Plan-cited recordings (Phase 0.5 reference set).
COMPARE_RECORDINGS = [
    ("H2 baseline (7e3caf0 active, score ~79)",
     "data/recordings/recording_20260418_195233.h5"),
    ("H2 Frenet G2 FAILURE (score 59)",
     "data/recordings/recording_20260419_140403.h5"),
    ("H2 probe at_car_gt_legacy (score 98.8)",
     "data/recordings/recording_20260419_165036.h5"),
    ("mixed_radius probe at_car_gt_legacy (score 96.6, Traj 87.7)",
     "data/recordings/recording_20260419_165335.h5"),
]

# Derivative step targets. ~13 FPS cadence → dt ≈ 0.077s.
# These are PHYSICAL seconds; converted to frame offsets below.
LEAD_TIMES_S = [0.1, 0.3, 0.5]
CADENCE_HZ_DEFAULT = 13.0


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation on finite-mask-intersected arrays."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return float("nan")
    xs, ys = x[m], y[m]
    if xs.std() < 1e-9 or ys.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def _percentile(x: np.ndarray, p: float) -> float:
    m = np.isfinite(x)
    if m.sum() == 0:
        return float("nan")
    return float(np.percentile(x[m], p))


def _estimate_dt_s(f: h5py.File) -> float:
    """Estimate dt from frame count / approximate cadence. No timestamp
    field exists on these recordings; memory notes ~13 FPS cadence.
    """
    # Use the first control dataset to find frame count.
    n = f["control/heading_error"].shape[0]
    # Prefer runtime_config or cadence hint if present; else default.
    # (No meta/duration field on these recordings.)
    return 1.0 / CADENCE_HZ_DEFAULT


def _load_signals(f: h5py.File) -> dict:
    """Pull the signals needed for term-role analysis. Falls back gracefully
    if a field is missing on older recordings."""
    c = f["control"]

    def get(path: str, required: bool = True) -> Optional[np.ndarray]:
        if path in f:
            return f[path][:].astype(np.float64)
        if required:
            raise KeyError(f"Missing required field: {path}")
        return None

    sig = {}
    # Error signals (MPC inputs).
    sig["e_h"] = get("control/mpc_gt_heading_error_rad")
    sig["d_at_car"] = get("control/mpc_gt_cross_track_at_car_m")
    sig["lookahead"] = get("control/mpc_gt_cross_track_lookahead_m")
    # Curvature reference (horizon-mean when MPC samples it).
    sig["kappa_ref"] = get("control/mpc_kappa_ref", required=False)
    if sig["kappa_ref"] is None:
        sig["kappa_ref"] = get("control/curvature_preview_abs", required=False)
    if sig["kappa_ref"] is None:
        sig["kappa_ref"] = get("control/curvature_primary_abs", required=False)
    # Shadow telemetry (populated on recordings after Frenet plan Phase C).
    sig["shadow_delta"] = get("control/mpc_e_lat_shadow_delta_m", required=False)
    sig["frenet_linearized"] = get(
        "control/mpc_e_lat_frenet_linearized_m", required=False
    )
    # Ld proxy (no MPC-specific field; PP value is close).
    sig["Ld"] = get("control/pp_lookahead_distance", required=False)
    if sig["Ld"] is None:
        # Fallback: back out Ld from shadow_delta where sin(e_h) is large enough.
        if sig["shadow_delta"] is not None:
            sin_eh = np.sin(sig["e_h"])
            ok = np.abs(sin_eh) > 0.01
            ld_est = np.where(ok, sig["shadow_delta"] / np.where(ok, sin_eh, 1.0), np.nan)
            median_ld = float(np.nanmedian(ld_est)) if np.isfinite(ld_est).any() else 8.0
            sig["Ld"] = np.full_like(sig["e_h"], median_ld)
        else:
            sig["Ld"] = np.full_like(sig["e_h"], 8.0)  # final fallback — plan memo says Ld=8 on H2
    # Speed (for dt-based derivative context).
    sig["speed"] = get("vehicle/speed", required=False)
    if sig["speed"] is None:
        sig["speed"] = np.full_like(sig["e_h"], float("nan"))
    # Reference-source mask (separates active from fallback frames).
    if "control/mpc_e_lat_reference_source" in f:
        sig["ref_source"] = f["control/mpc_e_lat_reference_source"][:]
    else:
        sig["ref_source"] = np.array([b"unknown"] * len(sig["e_h"]))
    return sig


def _compute_terms(sig: dict) -> dict:
    """Compute Ld·sin(e_h), Ld²κ/2, d/dt(e_h), e_h_lead(Δ)."""
    e_h = sig["e_h"]
    Ld = sig["Ld"]
    kappa = sig["kappa_ref"] if sig["kappa_ref"] is not None else np.zeros_like(e_h)

    ld_sin_eh = Ld * np.sin(e_h)
    ld2_k_half = Ld * Ld * kappa / 2.0
    # Time-derivative of heading error.
    d_e_h_dt = np.concatenate([[0.0], np.diff(e_h)])  # units: rad/frame
    # Time-derivative of lateral error (for completeness).
    d_d_dt = np.concatenate([[0.0], np.diff(sig["d_at_car"])])

    return {
        "ld_sin_eh": ld_sin_eh,
        "ld2_k_half": ld2_k_half,
        "d_e_h_dt": d_e_h_dt,
        "d_d_dt": d_d_dt,
        "kappa": kappa,
    }


def _lead(x: np.ndarray, frames: int) -> np.ndarray:
    """Return x shifted so y[k] = x[k+frames], padded with NaN."""
    if frames <= 0:
        return x.copy()
    out = np.full_like(x, np.nan)
    out[:-frames] = x[frames:]
    return out


def _report(name: str, path: str, sig: dict, terms: dict, dt_s: float) -> None:
    n = len(sig["e_h"])
    # Active-frame mask — exclude fallback/inactive frames.
    src = sig["ref_source"]
    try:
        inactive_mask = np.array([s in (b"inactive", b"fallback_pp") for s in src])
    except Exception:
        inactive_mask = np.zeros(n, dtype=bool)
    active = ~inactive_mask & np.isfinite(sig["e_h"])
    act_count = int(active.sum())

    print("=" * 78)
    print(f"  {name}")
    print(f"  {path}")
    print("=" * 78)
    print(f"  frames total: {n}   active: {act_count}   dt: {dt_s:.4f}s (assumed)")

    # Segment split: straights (|κ| < 0.003) vs curves (|κ| >= 0.003).
    kappa_abs = np.abs(terms["kappa"])
    straight = active & (kappa_abs < 0.003)
    curve = active & (kappa_abs >= 0.003)
    print(f"  straight frames: {int(straight.sum())}   curve frames: {int(curve.sum())}")

    # Amplitude check — sanity for downstream correlations.
    print()
    print(f"  |Ld·sin(e_h)|   P50={_percentile(np.abs(terms['ld_sin_eh']), 50):.4f}  "
          f"P95={_percentile(np.abs(terms['ld_sin_eh']), 95):.4f}  "
          f"max={_percentile(np.abs(terms['ld_sin_eh']), 100):.4f}")
    print(f"  |Ld²·κ/2|       P50={_percentile(np.abs(terms['ld2_k_half']), 50):.4f}  "
          f"P95={_percentile(np.abs(terms['ld2_k_half']), 95):.4f}  "
          f"max={_percentile(np.abs(terms['ld2_k_half']), 100):.4f}")
    print(f"  |d_at_car|      P50={_percentile(np.abs(sig['d_at_car']), 50):.4f}  "
          f"P95={_percentile(np.abs(sig['d_at_car']), 95):.4f}")
    print(f"  |e_h (rad)|     P50={_percentile(np.abs(sig['e_h']), 50):.4f}  "
          f"P95={_percentile(np.abs(sig['e_h']), 95):.4f}")
    print(f"  |κ_ref|         P50={_percentile(kappa_abs, 50):.5f}  "
          f"P95={_percentile(kappa_abs, 95):.5f}")

    # Core correlations — the Smell 7 gate.
    print()
    print("  DAMPING ROLE — corr(Ld·sin(e_h), d/dt(e_h))")
    print("  (if |corr| > 0.5 on stable frames → term is PD damping)")
    print(f"     all active frames : {_safe_corr(terms['ld_sin_eh'][active], terms['d_e_h_dt'][active]):+.3f}")
    print(f"     straights only    : {_safe_corr(terms['ld_sin_eh'][straight], terms['d_e_h_dt'][straight]):+.3f}")
    print(f"     curves only       : {_safe_corr(terms['ld_sin_eh'][curve], terms['d_e_h_dt'][curve]):+.3f}")

    print()
    print("  PREVIEW ROLE — corr(Ld·sin(e_h), e_h_lead(Δ))")
    print("  (if |corr| > 0.5 → term is acting as a preview signal)")
    for lead_s in LEAD_TIMES_S:
        frames = max(1, int(round(lead_s / dt_s)))
        lead = _lead(sig["e_h"], frames)
        r_all = _safe_corr(terms["ld_sin_eh"][active], lead[active])
        r_str = _safe_corr(terms["ld_sin_eh"][straight], lead[straight])
        r_crv = _safe_corr(terms["ld_sin_eh"][curve], lead[curve])
        print(f"     Δ={lead_s:.1f}s ({frames} frames)   all: {r_all:+.3f}   "
              f"straights: {r_str:+.3f}   curves: {r_crv:+.3f}")

    print()
    print("  CURVATURE-FF COVERAGE — corr(Ld²·κ/2, κ_ref)")
    print("  (if |corr| > 0.9 → kappa_ref_horizon fully covers the bundled FF;")
    print("   if < 0.9 on curves → need explicit curvature FF term)")
    print(f"     all active        : {_safe_corr(terms['ld2_k_half'][active], kappa_abs[active]):+.3f}")
    print(f"     curves only       : {_safe_corr(terms['ld2_k_half'][curve], kappa_abs[curve]):+.3f}")

    # Amplitude-weighted check: when Ld²κ/2 is large, does κ_ref see it?
    # Frames where Ld²κ/2 contributes >0.02m to lookahead.
    significant = active & (np.abs(terms["ld2_k_half"]) > 0.02)
    print(f"     frames w/ |Ld²κ/2| > 0.02m: {int(significant.sum())}")
    if significant.sum() > 10:
        print(f"     on those frames, corr: "
              f"{_safe_corr(terms['ld2_k_half'][significant], kappa_abs[significant]):+.3f}")

    # Heading-error FFT — detect oscillation signature.
    print()
    e_h_active = sig["e_h"][active]
    if len(e_h_active) > 64:
        e_h_centered = e_h_active - np.nanmean(e_h_active)
        fft = np.fft.rfft(e_h_centered)
        power = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(e_h_centered), d=dt_s)
        # Exclude DC; find peak.
        if len(freqs) > 1:
            peak_idx = int(np.argmax(power[1:])) + 1
            peak_hz = freqs[peak_idx]
            total_power = float(power[1:].sum())
            peak_power_frac = float(power[peak_idx]) / max(total_power, 1e-12)
            print(f"  HEADING-ERROR FFT peak: {peak_hz:.2f} Hz  "
                  f"(peak captures {peak_power_frac * 100:.1f}% of AC power)")

    # Reference-source breakdown — verifies which signal was actually used.
    print()
    unique, counts = np.unique(src, return_counts=True)
    print("  Reference-source breakdown (ground truth for what MPC saw):")
    for u, c in zip(unique, counts):
        u_str = u.decode() if isinstance(u, bytes) else str(u)
        print(f"     {u_str:40s} {c} frames ({100 * c / n:.1f}%)")
    print()


def analyze(path: str, label: Optional[str] = None) -> None:
    p = Path(path)
    if not p.exists():
        print(f"SKIP: {path} does not exist", file=sys.stderr)
        return
    with h5py.File(path, "r") as f:
        sig = _load_signals(f)
        terms = _compute_terms(sig)
        dt_s = _estimate_dt_s(f)
        _report(label or p.name, str(p), sig, terms, dt_s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="*", help="HDF5 recordings")
    ap.add_argument("--compare", action="store_true",
                    help="Run on the 4 plan-cited recordings")
    args = ap.parse_args()

    if args.compare:
        for label, path in COMPARE_RECORDINGS:
            analyze(path, label)
        # Synthesis table.
        print("=" * 78)
        print("  SYNTHESIS — Smell 7 decision gate")
        print("=" * 78)
        print("  Expected signatures:")
        print("   - H2 7e3caf0 (score 79): HIGH  |corr(Ld·sin(e_h), d/dt(e_h))|  on straights")
        print("                            (term was coupled into loop as it should; bug is")
        print("                            signal-routing, not dynamic role absence)")
        print("   - H2 Frenet FAIL (59):   LOW   damping-role corr (term removed from e_lat;")
        print("                            oscillation FFT should peak ~1–3 Hz)")
        print("   - H2 at_car (98.8):      LOW   damping corr (term absent, but no excitation;")
        print("                            curves absent so no info on FF)")
        print("   - mixed_radius at_car:   Ld²·κ/2 amplitudes should be non-trivial on curves;")
        print("                            corr(Ld²·κ/2, κ_ref) ≈ +1.0 expected by construction")
        print()
        print("  DECISION CRITERIA for proceeding with plan Phase A:")
        print("   (1) H2 7e3caf0 straights show |corr| > 0.5 for Ld·sin(e_h) vs d/dt(e_h)  →")
        print("       CONFIRMS: term was doing damping work; need substitute in cost-side")
        print("       activation (mpc_lookahead_cost_enabled=true)")
        print("   (2) H2 Frenet FAIL FFT peak matches oscillation frequency  →")
        print("       CONFIRMS: removal of term shifted closed-loop poles")
        print("   (3) mixed_radius Ld²·κ/2 P95 > 0.10m on curves  →")
        print("       CONFIRMS: bundle's curvature-FF contribution is material; kappa_ref")
        print("       horizon MUST carry it post-activation (check via E2E after flip)")
        print()
        print("  If (1) and (2) hold → Phase A (config flip) proceeds. Phase B.1 activates")
        print("  cost-side mechanism. Re-run this tool on post-activation recording to confirm")
        print("  the damping role transfers to the cost cross-term.")
        return 0

    if not args.recordings:
        ap.print_help()
        return 1
    for path in args.recordings:
        analyze(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
