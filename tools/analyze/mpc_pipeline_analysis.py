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

# Scoring gate import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scoring_registry import TIRE_EKF_INNOVATION_P95_GATE


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

        # Lateral accel budget (regime selector)
        data["regime_lateral_accel"]        = _get("control/regime_lateral_accel_mps2")
        data["regime_lateral_accel_thr"]    = _get("control/regime_lateral_accel_threshold_mps2")

        # MPC reference alignment
        data["mpc_e_lat_ref_source"]       = None
        data["mpc_e_lat_ref_divergence"]   = _get("control/mpc_e_lat_reference_divergence_m")
        if "control/mpc_e_lat_reference_source" in f:
            raw = f["control/mpc_e_lat_reference_source"][:]
            data["mpc_e_lat_ref_source"] = np.array(
                [s.decode() if isinstance(s, bytes) else str(s) for s in raw]
            )

        # 2.8.5 — r_steer_rate scheduling + 2.8.6 — e_lat attenuation
        data["mpc_r_steer_rate_effective"]  = _get("control/mpc_r_steer_rate_effective")
        data["mpc_e_lat"]                   = _get("control/mpc_e_lat")

        # Step 5 — NMPC telemetry
        data["nmpc_used"]                   = _get("control/nmpc_used")
        data["nmpc_feasible"]               = _get("control/nmpc_feasible")
        data["nmpc_solve_time_ms"]          = _get("control/nmpc_solve_time_ms")
        data["nmpc_cost"]                   = _get("control/nmpc_cost")
        data["nmpc_iterations"]             = _get("control/nmpc_iterations")
        data["nmpc_fallback_active"]        = _get("control/nmpc_fallback_active")
        data["nmpc_consecutive_failures"]   = _get("control/nmpc_consecutive_failures")

        # L_eff estimation
        data["mpc_leff_value"]                 = _get("control/mpc_leff_value")
        data["mpc_leff_theta"]                 = _get("control/mpc_leff_theta")
        data["mpc_leff_innovation"]            = _get("control/mpc_leff_innovation")
        data["mpc_leff_update_count"]          = _get("control/mpc_leff_update_count")

        # Tire estimation (dynamic bicycle)
        data["mpc_tire_cf"]                    = _get("control/mpc_tire_cf")
        data["mpc_tire_cr"]                    = _get("control/mpc_tire_cr")
        data["mpc_tire_ekf_innovation"]        = _get("control/mpc_tire_ekf_innovation")
        data["mpc_tire_ekf_P_trace"]           = _get("control/mpc_tire_ekf_P_trace")
        data["mpc_tire_slip_angle_front"]      = _get("control/mpc_tire_slip_angle_front")
        data["mpc_tire_slip_angle_rear"]       = _get("control/mpc_tire_slip_angle_rear")
        data["mpc_tire_understeer_gradient"]   = _get("control/mpc_tire_understeer_gradient")
        data["mpc_dynamic_model_active"]       = _get("control/mpc_dynamic_model_active")
        data["mpc_tire_ekf_update_count"]      = _get("control/mpc_tire_ekf_update_count")
        data["mpc_v_y_estimate"]               = _get("control/mpc_v_y_estimate")
        data["mpc_yaw_rate_estimate"]          = _get("control/mpc_yaw_rate_estimate")
        data["mpc_yaw_rate_measurement"]       = _get("control/mpc_yaw_rate_measurement")
        data["mpc_imu_yaw_rate_raw"]           = _get("control/mpc_imu_yaw_rate_raw")

        # IMU raw signal (3-axis angular velocity from Unity)
        data["vehicle_angular_velocity"]       = _get("vehicle/angular_velocity")

        # Unity geometry override
        data["mpc_unity_geometry_lf"]          = _get("control/mpc_unity_geometry_lf")
        data["mpc_unity_geometry_lr"]          = _get("control/mpc_unity_geometry_lr")
        data["mpc_unity_geometry_mass"]        = _get("control/mpc_unity_geometry_mass")
        data["mpc_unity_geometry_iz"]          = _get("control/mpc_unity_geometry_iz")
        data["mpc_unity_geometry_active"]      = _get("control/mpc_unity_geometry_active")

        # Inter-frame extrapolation
        data["interframe_active"]              = _get("control/interframe_active")
        data["interframe_updates_this_cycle"]  = _get("control/interframe_updates_this_cycle")
        data["interframe_total_count"]         = _get("control/interframe_total_count")
        data["interframe_last_e_lat"]          = _get("control/interframe_last_e_lat")
        data["interframe_last_e_heading"]      = _get("control/interframe_last_e_heading")
        data["interframe_dt_actual"]           = _get("control/interframe_dt_actual")

        data["n"] = n or 0
    return data


# ──────────────────────────────────────────────────────────────────────────────
# Regime masks
# ──────────────────────────────────────────────────────────────────────────────

def _regime_masks(data):
    """Return (pp_mask, lmpc_mask, nmpc_mask, any_mpc_mask, blend_mask).

    Regime encoding: 0=PURE_PURSUIT, 1=LINEAR_MPC, 2=NONLINEAR_MPC.
    blend_weight is a SEPARATE field — there is no integer "BLEND" regime value.
    blend_mask here marks frames where MPC is active but blend_weight < 1.0.
    """
    regime = data.get("regime")
    blend_w = data.get("regime_blend_weight")
    n = data["n"]
    zeros = np.zeros(n, bool)
    if regime is None or n == 0:
        return zeros, zeros, zeros, zeros, zeros
    regime = regime[:n]
    pp_mask      = regime < 0.5                 # regime == 0 (PP)
    lmpc_mask    = (regime >= 0.5) & (regime < 1.5)   # regime == 1 (LMPC)
    nmpc_mask    = regime >= 1.5                # regime == 2 (NMPC)
    any_mpc_mask = regime >= 0.5               # any MPC (LMPC or NMPC)
    if blend_w is not None:
        bw = blend_w[:n]
        blend_mask = any_mpc_mask & (bw < 0.99)
    else:
        blend_mask = zeros
    return pp_mask, lmpc_mask, nmpc_mask, any_mpc_mask, blend_mask


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
    return {0: "PP", 1: "LMPC", 2: "NMPC"}.get(int(round(v)), f"?{v}")


def print_report(path: Path, data: dict):
    n = data["n"]
    pp_mask, lmpc_mask, nmpc_mask, mpc_mask, blend_mask = _regime_masks(data)

    pp_n      = int(np.sum(pp_mask))
    lmpc_n    = int(np.sum(lmpc_mask))
    nmpc_n    = int(np.sum(nmpc_mask))
    mpc_n     = int(np.sum(mpc_mask))   # LMPC + NMPC
    blend_n   = int(np.sum(blend_mask))
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
    print(f"  LMPC  frames: {lmpc_n:4d} / {n}  ({lmpc_n/n*100:.1f}%)")
    if nmpc_n > 0:
        print(f"  NMPC  frames: {nmpc_n:4d} / {n}  ({nmpc_n/n*100:.1f}%)")
    print(f"  MPC (total):  {mpc_n:4d} / {n}  ({mpc_n/n*100:.1f}%)", end="")
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

    # ── 2.8.5 r_steer_rate scheduling ──────────────────────────────────────
    print(f"\n{thin}")
    print("2.8.5 — r_steer_rate scheduling")
    rsr_eff = data.get("mpc_r_steer_rate_effective")
    if rsr_eff is not None:
        rsr_mpc = rsr_eff[:n][mpc_mask]
        if len(rsr_mpc) > 0:
            print(f"  Effective r_steer_rate: Mean={np.mean(rsr_mpc):.3f}  "
                  f"Min={np.min(rsr_mpc):.3f}  Max={np.max(rsr_mpc):.3f}")
            transitions = int(np.sum(np.abs(np.diff(rsr_mpc)) > 0.1))
            print(f"  Band transitions: {transitions}")
            if np.max(rsr_mpc) - np.min(rsr_mpc) < 0.01:
                print(f"  NOTE: Scheduling appears INACTIVE (constant value)")
        else:
            print("  [No MPC frames for scheduling analysis]")
    else:
        print("  [Field mpc_r_steer_rate_effective not in recording]")

    # ── 2.8.6 e_lat speed attenuation ──────────────────────────────────────
    print(f"\n{thin}")
    print("2.8.6 — e_lat speed attenuation")
    mpc_elat = data.get("mpc_e_lat")
    if raw_e is not None and mpc_elat is not None:
        raw_mpc = np.abs(raw_e[:n][mpc_mask])
        inp_mpc = np.abs(mpc_elat[:n][mpc_mask])
        valid = raw_mpc > 0.01
        if valid.sum() > 10:
            ratio = inp_mpc[valid] / raw_mpc[valid]
            r_p50 = float(np.median(ratio))
            print(f"  MPC/raw |e_lat| ratio P50: {r_p50:.3f}  "
                  f"({'ACTIVE' if r_p50 < 0.95 else 'inactive/minimal'})")
        else:
            print("  [Insufficient valid frames]")
    else:
        print("  [Smith raw or mpc_e_lat fields not in recording]")

    # ── 2.8.7 Lateral accel budget ──────────────────────────────────────────
    print(f"\n{thin}")
    print("2.8.7 — Lateral acceleration budget (regime selector)")
    a_lat = data.get("regime_lateral_accel")
    a_thr = data.get("regime_lateral_accel_thr")
    if a_lat is not None and mpc_n > 0:
        a_lat_mpc = a_lat[:n][mpc_mask]
        p50 = float(np.percentile(a_lat_mpc, 50))
        p95 = float(np.percentile(a_lat_mpc, 95))
        mx = float(np.max(a_lat_mpc))
        print(f"  a_lat (κ×v²) during MPC — P50: {p50:.3f}  P95: {p95:.3f}  Max: {mx:.3f} m/s²")
        if a_thr is not None:
            exceeded = a_lat_mpc > a_thr[:n][mpc_mask]
            exc_n = int(np.sum(exceeded))
            exc_pct = exc_n / mpc_n * 100
            gate = "green" if exc_pct < 1.0 else ("yellow" if exc_pct < 5.0 else "red")
            print(f"  Budget exceeded: {exc_n}/{mpc_n} MPC frames ({exc_pct:.2f}%)  [{gate}]")
            headroom = (a_thr[:n][mpc_mask] - a_lat_mpc) / np.maximum(a_thr[:n][mpc_mask], 0.01)
            print(f"  Budget headroom — P50: {float(np.percentile(headroom, 50)):.2f}  P5: {float(np.percentile(headroom, 5)):.2f}")
        else:
            print("  [Threshold field not in recording]")
    else:
        print("  [Field regime_lateral_accel_mps2 not in recording]")

    # ── L_eff Estimation ──────────────────────────────────────────────────────
    leff_val = data.get("mpc_leff_value")
    if leff_val is not None and np.any(leff_val[:n] != 0):
        print(f"\n{thin}")
        print("L_eff Estimation")
        lv = leff_val[:n]
        leff_innov = data.get("mpc_leff_innovation")
        leff_upd   = data.get("mpc_leff_update_count")

        mpc_leff = lv[mpc_mask]
        print(f"  L_eff final:     {mpc_leff[-1]:.3f} m  (nominal: 2.500 m)")
        print(f"  L_eff range:     [{np.min(mpc_leff):.3f}, {np.max(mpc_leff):.3f}] m")
        print(f"  L_eff mean:      {np.mean(mpc_leff):.3f} m")

        if leff_innov is not None:
            innov_mpc = np.abs(leff_innov[:n][mpc_mask])
            print(f"  Innovation P95:  {np.percentile(innov_mpc, 95):.4f} rad")

        if leff_upd is not None:
            upd_mpc = leff_upd[:n][mpc_mask]
            total_updates = int(np.max(upd_mpc)) if len(upd_mpc) > 0 else 0
            print(f"  Updates:         {total_updates} frames")

        if len(mpc_leff) > 100:
            tail_std = float(np.std(mpc_leff[-100:]))
            status = "\u2713 converged" if tail_std < 0.01 else "\u26a0 still adapting"
            print(f"  Tail stability:  \u03c3={tail_std:.4f} m {status}")

    # ── Tire Estimation (Dynamic Bicycle) ──────────────────────────────────
    tire_cf = data.get("mpc_tire_cf")
    if tire_cf is not None and np.any(tire_cf[:n] != 0):
        print(f"\n{thin}")
        print("Tire Estimation (Dynamic Bicycle)")
        cf = tire_cf[:n]
        cr = data.get("mpc_tire_cr")
        dyn_active = data.get("mpc_dynamic_model_active")
        tire_innov = data.get("mpc_tire_ekf_innovation")
        slip_f = data.get("mpc_tire_slip_angle_front")
        slip_r = data.get("mpc_tire_slip_angle_rear")
        k_us = data.get("mpc_tire_understeer_gradient")
        tire_upd = data.get("mpc_tire_ekf_update_count")

        # Dynamic model active rate (MPC frames only)
        if dyn_active is not None:
            da_mpc = dyn_active[:n][mpc_mask]
            da_pct = float(np.mean(da_mpc > 0.5) * 100.0)
            da_yes = "YES" if da_pct > 50 else "NO"
            print(f"  Dynamic model active:    {da_yes} ({da_pct:.1f}% of MPC frames)")
        else:
            print("  Dynamic model active:    [field not in recording]")

        # C_f / C_r final
        cf_mpc = cf[mpc_mask]
        if len(cf_mpc) > 0:
            print(f"  C_f final:               {cf_mpc[-1]:,.0f} N/rad (nominal: 40,000)")
        if cr is not None:
            cr_mpc = cr[:n][mpc_mask]
            if len(cr_mpc) > 0:
                print(f"  C_r final:               {cr_mpc[-1]:,.0f} N/rad (nominal: 40,000)")

        # Understeer gradient
        if k_us is not None:
            kus_mpc = k_us[:n][mpc_mask]
            if len(kus_mpc) > 0:
                print(f"  Understeer gradient K_us: {kus_mpc[-1]:.4f} rad/(m/s\u00b2)")

        # EKF innovation P95 with gate
        if tire_innov is not None:
            innov_mpc = np.abs(tire_innov[:n][mpc_mask])
            if len(innov_mpc) > 0:
                innov_p95 = float(np.percentile(innov_mpc, 95))
                gate = "PASS" if innov_p95 < TIRE_EKF_INNOVATION_P95_GATE else "FAIL"
                print(f"  EKF innovation P95:      {innov_p95:.4f} rad/s  "
                      f"[{gate} < {TIRE_EKF_INNOVATION_P95_GATE}]")

        # Slip angles
        if slip_f is not None:
            sf_mpc = np.abs(slip_f[:n][mpc_mask])
            if len(sf_mpc) > 0:
                sf_p95 = float(np.percentile(sf_mpc, 95))
                print(f"  Slip angle front P95:    {sf_p95:.3f} rad ({np.degrees(sf_p95):.1f} deg)")
        if slip_r is not None:
            sr_mpc = np.abs(slip_r[:n][mpc_mask])
            if len(sr_mpc) > 0:
                sr_p95 = float(np.percentile(sr_mpc, 95))
                print(f"  Slip angle rear P95:     {sr_p95:.3f} rad ({np.degrees(sr_p95):.1f} deg)")

        # EKF updates
        if tire_upd is not None:
            upd_mpc = tire_upd[:n][mpc_mask]
            total_updates = int(np.max(upd_mpc)) if len(upd_mpc) > 0 else 0
            print(f"  EKF updates:             {total_updates} frames")

    # ── Unity Geometry Override ────────────────────────────────────────────
    geo_lf = data.get("mpc_unity_geometry_lf")
    geo_active = data.get("mpc_unity_geometry_active")
    if geo_lf is not None and np.any(geo_lf[:n] > 0):
        lf_val = float(geo_lf[n - 1])
        lr_val = float(data.get("mpc_unity_geometry_lr", np.zeros(1))[n - 1])
        mass_val = float(data.get("mpc_unity_geometry_mass", np.zeros(1))[n - 1])
        iz_val = float(data.get("mpc_unity_geometry_iz", np.zeros(1))[n - 1])
        active = bool(geo_active[n - 1]) if geo_active is not None else False
        print(f"\n{thin}")
        print("Unity Geometry Override")
        print(f"  Status:          {'ACTIVE' if active else 'INACTIVE'}")
        print(f"  l_f:             {lf_val:.3f} m")
        print(f"  l_r:             {lr_val:.3f} m")
        print(f"  Wheelbase:       {lf_val + lr_val:.3f} m")
        print(f"  Asymmetry ratio: {lf_val / (lf_val + lr_val):.3f} (0.5 = symmetric)")
        print(f"  Mass:            {mass_val:.1f} kg")
        print(f"  Iz:              {iz_val:.1f} kg*m²")
        _cr = data.get("mpc_tire_cr")
        if tire_cf is not None and _cr is not None:
            cf_final = float(tire_cf[n - 1]) if tire_cf[n - 1] > 0 else 40000.0
            cr_final = float(_cr[n - 1]) if _cr[n - 1] > 0 else 40000.0
            L = lf_val + lr_val
            if L > 0 and cf_final > 0 and cr_final > 0:
                K_us = mass_val * lr_val / (2 * L * cf_final) - mass_val * lf_val / (2 * L * cr_final)
                print(f"  K_us (understeer): {K_us:.4f} rad/(m/s²) "
                      f"[{'physical' if K_us > 0.0005 else 'near-neutral'}]")

    # ── IMU Yaw Rate Quality ───────────────────────────────────────────────
    imu_raw = data.get("mpc_imu_yaw_rate_raw")
    yr_meas = data.get("mpc_yaw_rate_measurement")
    if imu_raw is not None and mpc_mask is not None and np.sum(mpc_mask) > 0:
        imu_mpc = np.abs(imu_raw[:n][mpc_mask])
        nonzero_frac = float(np.mean(imu_mpc > 1e-6))
        print(f"\n{thin}")
        print("IMU Yaw Rate Quality")

        # Determine source from yaw_rate_measurement vs derived comparison
        if yr_meas is not None:
            yr_mpc = np.abs(yr_meas[:n][mpc_mask])
            print(f"  |yaw_rate_meas| P50 / P95 / MAX: "
                  f"{_fmt(np.percentile(yr_mpc,50))} / {_fmt(np.percentile(yr_mpc,95))} / "
                  f"{_fmt(np.max(yr_mpc))} rad/s")
        print(f"  |IMU raw|        P50 / P95 / MAX: "
              f"{_fmt(np.percentile(imu_mpc,50))} / {_fmt(np.percentile(imu_mpc,95))} / "
              f"{_fmt(np.max(imu_mpc))} rad/s")
        print(f"  Signal present:  {nonzero_frac*100:.0f}% "
              f"[{'PASS' if nonzero_frac >= 0.95 else 'FAIL'} >= 95%]")

        # Compare IMU vs derived jitter if we have heading data
        if yr_meas is not None and len(yr_mpc) > 2:
            imu_jitter = float(np.std(np.diff(imu_mpc)))
            meas_jitter = float(np.std(np.diff(yr_mpc)))
            if meas_jitter > 1e-9:
                ratio = imu_jitter / meas_jitter
                print(f"  IMU jitter σ(Δ): {imu_jitter:.4f} rad/s²")

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
    if solve is not None and lmpc_n > 0:
        sv = solve[:n][lmpc_mask]
        gate = "PASS" if float(np.percentile(sv, 95)) <= 5.0 else "FAIL"
        print(f"  LMPC solve_time_ms P50 / P95 / MAX: "
              f"{_fmt(np.percentile(sv,50),'0.2f')}ms / "
              f"{_fmt(np.percentile(sv,95),'0.2f')}ms / "
              f"{_fmt(np.max(sv),'0.2f')}ms  [{gate} P95≤5ms]")

    fallback = data.get("mpc_fallback_active")
    if fallback is not None:
        fb = int(np.sum(fallback[:n][mpc_mask] > 0.5))
        print(f"  Fallback activations: {fb} / {mpc_n}  ({fb/mpc_n*100:.1f}%)")

    # ── MPC Reference Alignment ─────────────────────────────────────────────
    ref_source = data.get("mpc_e_lat_ref_source")
    ref_div = data.get("mpc_e_lat_ref_divergence")
    if ref_source is not None or ref_div is not None:
        print(f"\n{thin}")
        print("MPC Reference Alignment")
        if ref_source is not None and mpc_n > 0:
            mpc_sources = ref_source[:n][mpc_mask]
            unique, counts = np.unique(mpc_sources, return_counts=True)
            for src, cnt in zip(unique, counts):
                print(f"  Reference source: {src} ({cnt/mpc_n*100:.1f}% of MPC frames)")
        if ref_div is not None and mpc_n > 0:
            div = ref_div[:n][mpc_mask]
            abs_div = np.abs(div)
            print(f"  Reference divergence (lookahead - at_car):")
            print(f"    P50: {np.percentile(abs_div,50):.3f}m  "
                  f"P95: {np.percentile(abs_div,95):.3f}m  "
                  f"Max: {np.max(abs_div):.3f}m")
        # Alignment check: correlation between mpc_e_lat and -lateral_error
        lat_err = data.get("lateral_error")
        e_lat = data.get("mpc_e_lat")
        if lat_err is not None and e_lat is not None and mpc_n > 10:
            le = lat_err[:n][mpc_mask]
            el = e_lat[:n][mpc_mask]
            corr = float(np.corrcoef(el, -le)[0, 1]) if np.std(el) > 1e-9 and np.std(le) > 1e-9 else 0.0
            gate = "PASS" if corr > 0.95 else ("WARN" if corr > 0.80 else "FAIL")
            print(f"  Correlation(mpc_e_lat, -lateral_error): {corr:.3f}  [{gate} > 0.95]")

    # ── Step 5 — NMPC section ────────────────────────────────────────────────
    if nmpc_n > 0:
        print(f"\n{thin}")
        print("Step 5 — NMPC solver health")
        nmpc_solve = data.get("nmpc_solve_time_ms")
        nmpc_cost  = data.get("nmpc_cost")
        nmpc_iters = data.get("nmpc_iterations")
        nmpc_feas  = data.get("nmpc_feasible")
        nmpc_used  = data.get("nmpc_used")
        nmpc_fb    = data.get("nmpc_fallback_active")

        if nmpc_solve is not None:
            sv_n = nmpc_solve[:n][nmpc_mask]
            gate = "PASS" if float(np.percentile(sv_n, 95)) <= 20.0 else "FAIL"
            print(f"  NMPC solve_time_ms P50 / P95 / MAX: "
                  f"{_fmt(np.percentile(sv_n,50),'0.2f')}ms / "
                  f"{_fmt(np.percentile(sv_n,95),'0.2f')}ms / "
                  f"{_fmt(np.max(sv_n),'0.2f')}ms  [{gate} P95≤20ms]")

        # NMPC vs LMPC-fallback breakdown within NONLINEAR_MPC regime frames
        if nmpc_used is not None:
            nu = nmpc_used[:n][nmpc_mask]
            nmpc_ran = int(np.sum(nu > 0.5))
            lmpc_ran = nmpc_n - nmpc_ran
            print(f"  NMPC ran: {nmpc_ran}/{nmpc_n}  ({nmpc_ran/nmpc_n*100:.1f}%)  "
                  f"LMPC-fallback: {lmpc_ran}/{nmpc_n}  ({lmpc_ran/nmpc_n*100:.1f}%)")

        if nmpc_feas is not None:
            infeasible = int(np.sum(nmpc_feas[:n][nmpc_mask] < 0.5))
            print(f"  Infeasible frames: {infeasible}/{nmpc_n}  ({infeasible/nmpc_n*100:.2f}%)")

        if nmpc_fb is not None:
            fb_n = int(np.sum(nmpc_fb[:n][nmpc_mask] > 0.5))
            print(f"  Fallback-active frames: {fb_n}/{nmpc_n}  ({fb_n/nmpc_n*100:.2f}%)")

        if nmpc_iters is not None:
            it = nmpc_iters[:n][nmpc_mask]
            print(f"  SLSQP iterations P50 / P95 / MAX: "
                  f"{_fmt(np.percentile(it,50),'0.0f')} / "
                  f"{_fmt(np.percentile(it,95),'0.0f')} / "
                  f"{_fmt(np.max(it),'0.0f')}")

        if nmpc_cost is not None:
            cs = nmpc_cost[:n][nmpc_mask]
            # Cost drift check: large spread = solver not warm-starting well
            cost_spread = float(np.std(cs)) if len(cs) > 1 else 0.0
            print(f"  NLP cost median: {_fmt(np.median(cs),'.1f')}  "
                  f"std: {_fmt(cost_spread,'.1f')}  "
                  f"(high std → warm-start diverging)")

    # ── Inter-frame control extrapolation ────────────────────────────────────
    if_active = data.get("interframe_active")
    if if_active is not None and np.any(if_active[:n] > 0.5):
        print(f"\n{thin}")
        print("Inter-frame control extrapolation")
        if_arr = if_active[:n]
        if_mask = if_arr > 0.5
        if_count = int(np.sum(if_mask))
        if_total_arr = data.get("interframe_total_count")
        if_updates = data.get("interframe_updates_this_cycle")
        if_dt = data.get("interframe_dt_actual")
        if_e_lat_arr = data.get("interframe_last_e_lat")

        total_updates = int(if_total_arr[n - 1]) if if_total_arr is not None else if_count
        speed_arr = data.get("speed")
        duration_s = n / 30.0  # approximate
        cam_hz = n / max(1.0, duration_s)
        eff_hz = (n + total_updates) / max(1.0, duration_s)
        print(f"  Camera rate:         ~{cam_hz:.1f} Hz")
        print(f"  Effective rate:      ~{eff_hz:.1f} Hz (camera + {total_updates} inter-frame)")
        print(f"  Inter-frame active:  {if_count}/{n} frames ({if_count/max(1,n)*100:.0f}%)")

        if if_updates is not None:
            up = if_updates[:n][if_mask].astype(float)
            if len(up) > 0:
                at_max_pct = float(np.mean(up >= 3.0) * 100.0)
                print(f"  Updates/cycle:       mean={np.mean(up):.1f}  "
                      f"P50={np.median(up):.0f}  at_max={at_max_pct:.0f}%")

        if if_dt is not None:
            dt_vals = if_dt[:n][if_mask].astype(float)
            dt_vals = dt_vals[dt_vals > 0]
            if len(dt_vals) > 0:
                print(f"  Inter-frame dt:      P50={np.median(dt_vals)*1000:.1f}ms  "
                      f"P95={np.percentile(dt_vals, 95)*1000:.1f}ms  "
                      f"max={np.max(dt_vals)*1000:.1f}ms")

        if if_e_lat_arr is not None and e_lat is not None:
            div = np.abs(if_e_lat_arr[:n-1] - e_lat[1:n])[if_mask[:n-1]]
            if len(div) > 0:
                print(f"  GT divergence:       mean={np.mean(div):.4f}m  "
                      f"P95={np.percentile(div, 95):.4f}m  "
                      f"max={np.max(div):.4f}m")
                over_02 = float(np.mean(div > 0.2) * 100.0)
                if over_02 > 0:
                    print(f"  Divergence > 0.2m:   {over_02:.1f}% of inter-frame cycles")

        # Steering delta at inter-frame boundaries
        if if_e_lat_arr is not None:
            steer_arr = data.get("steering")
            if steer_arr is not None:
                steer_delta = np.abs(np.diff(steer_arr[:n]))
                if_steer_delta = steer_delta[if_mask[:n-1]]
                noif_steer_delta = steer_delta[~if_mask[:n-1]]
                if len(if_steer_delta) > 0 and len(noif_steer_delta) > 0:
                    print(f"  Steering delta:      if={np.mean(if_steer_delta):.4f}  "
                          f"no-if={np.mean(noif_steer_delta):.4f}")

    # ── Curve-Phase Decomposition ──────────────────────────────────────────────
    kappa_arr = data.get("mpc_kappa_ref")
    if kappa_arr is not None and e_lat is not None:
        print(f"\n{thin}")
        print("Curve-Phase Decomposition  (entry 20% / steady 60% / exit 20%)")
        kappa = kappa_arr[:n]
        mpc_elat_full = e_lat[:n]
        speed_arr = data.get("speed")

        # Detect contiguous curve runs where |κ| > 0.005 (R < 200m)
        curve_kappa_threshold = 0.005
        curve_bit = np.abs(kappa) > curve_kappa_threshold
        diff_c = np.diff(curve_bit.astype(int))
        c_starts = np.where(diff_c == 1)[0] + 1
        c_ends = np.where(diff_c == -1)[0] + 1
        if curve_bit[0]:
            c_starts = np.concatenate([[0], c_starts])
        if curve_bit[-1]:
            c_ends = np.concatenate([c_ends, [n]])

        # Filter to substantial curves (> 30 frames ≈ 1 sec)
        curves = [(int(s), int(e)) for s, e in zip(c_starts, c_ends) if e - s > 30]

        if not curves:
            print("  [No substantial curves detected (|κ| > 0.005 for > 30 frames)]")
        else:
            all_entry, all_steady, all_exit = [], [], []
            for ci, (cs, ce) in enumerate(curves):
                nf = ce - cs
                entry_end = cs + nf // 5
                exit_start = ce - nf // 5
                if entry_end >= exit_start:
                    continue  # curve too short for meaningful split

                entry_e = np.abs(mpc_elat_full[cs:entry_end])
                steady_e = np.abs(mpc_elat_full[entry_end:exit_start])
                exit_e = np.abs(mpc_elat_full[exit_start:ce])
                all_entry.extend(entry_e)
                all_steady.extend(steady_e)
                all_exit.extend(exit_e)

                spd_str = f"  v={np.mean(speed_arr[cs:ce]):.1f}" if speed_arr is not None else ""
                kap_mean = float(np.mean(np.abs(kappa[cs:ce])))
                print(f"\n  C{ci+1} [{cs}..{ce}] ({nf}fr){spd_str}m/s  κ={kap_mean:.4f}")
                print(f"    {'Phase':<8} {'frames':>6}  {'P50':>6}  {'P95':>6}  {'MAX':>6}")
                for label, arr_phase in [("ENTRY", entry_e), ("STEADY", steady_e), ("EXIT", exit_e)]:
                    print(f"    {label:<8} {len(arr_phase):>6d}  "
                          f"{np.percentile(arr_phase, 50):>6.3f}  "
                          f"{np.percentile(arr_phase, 95):>6.3f}  "
                          f"{np.max(arr_phase):>6.3f}")

            if all_entry:
                ae, ase, axe = np.array(all_entry), np.array(all_steady), np.array(all_exit)
                print(f"\n  ── AGGREGATE across {len(curves)} curves ──")
                print(f"    {'Phase':<8} {'frames':>6}  {'P50':>6}  {'P95':>6}  {'MAX':>6}")
                for label, ap in [("ENTRY", ae), ("STEADY", ase), ("EXIT", axe)]:
                    print(f"    {label:<8} {len(ap):>6d}  "
                          f"{np.percentile(ap, 50):>6.3f}  "
                          f"{np.percentile(ap, 95):>6.3f}  "
                          f"{np.max(ap):>6.3f}")

                # Diagnosis: is the bottleneck entry transients or steady-state tracking?
                entry_p50 = float(np.percentile(ae, 50))
                steady_p50 = float(np.percentile(ase, 50))
                if steady_p50 > entry_p50 * 1.1:
                    diagnosis = "STEADY-STATE dominated — model bias, not turn-in latency"
                elif entry_p50 > steady_p50 * 1.3:
                    diagnosis = "ENTRY dominated — late turn-in or authority gap"
                else:
                    diagnosis = "MIXED — both entry and steady-state contribute"
                print(f"\n  Diagnosis: {diagnosis}")
                print(f"    Entry P50={entry_p50:.3f}m  Steady P50={steady_p50:.3f}m  "
                      f"ratio={entry_p50/max(steady_p50,0.001):.2f}")

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
