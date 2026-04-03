"""MPC Pipeline diagnostic backend — Phase 2.8.

Exposes seven analysis cards for the PhilViz "MPC Pipeline" tab:
  Card 1 — Regime Health (distribution, transitions, blend continuity)
  Card 2 — Pipeline Fix Status (2.8.1 recovery, 2.8.2 feedback, 2.8.3 rate, 2.8.4 Smith)
  Card 3 — MPC State Quality (e_lat, e_heading, solve time)
  Card 4 — Severe Frame Inspector (top-N frames by |mpc_e_lat|)
  Card 5 — NMPC Solver Health (Step 5)
  Card 6 — L_eff Estimation (wheelbase adaptation convergence)
  Card 7 — Tire Estimation (dynamic bicycle EKF cornering stiffness)

Returns None for PP-only recordings (no MPC fields).
"""

import math
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np


def _safe_float(v, default=0.0):
    if v is None:
        return default
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def _pct(arr, p):
    if len(arr) == 0:
        return None
    return float(np.percentile(arr, p))


def _rate(arr):
    """Fraction of elements > 0.5."""
    if len(arr) == 0:
        return None
    return float(np.mean(arr > 0.5))


def _load(f: h5py.File, key: str, n: int) -> Optional[np.ndarray]:
    if key not in f:
        return None
    return np.asarray(f[key][:n], dtype=float)


def analyze_mpc_pipeline(filepath: Path) -> Optional[Dict]:
    """Return full 4-card MPC pipeline report, or None for PP-only recordings."""
    with h5py.File(filepath, "r") as f:
        if "control/regime" not in f:
            return None

        # ── Determine n_frames ───────────────────────────────────────────────
        regime_raw = np.asarray(f["control/regime"][:], dtype=float)
        n = len(regime_raw)
        if n == 0:
            return None

        # ── Regime masks ────────────────────────────────────────────────────
        pp_mask   = regime_raw < 0.5
        lmpc_mask = (regime_raw >= 0.5) & (regime_raw < 1.5)
        nmpc_mask = regime_raw >= 1.5
        mpc_mask  = regime_raw >= 0.5  # any MPC (LMPC + NMPC)
        pp_n   = int(np.sum(pp_mask))
        lmpc_n = int(np.sum(lmpc_mask))
        nmpc_n = int(np.sum(nmpc_mask))
        mpc_n  = int(np.sum(mpc_mask))

        blend_weight = _load(f, "control/regime_blend_weight", n)
        blend_n = 0
        if blend_weight is not None:
            blend_n = int(np.sum(mpc_mask & (blend_weight < 0.99)))

        # ── Card 1: Regime Health ────────────────────────────────────────────
        # Transition detection
        transitions: List[Dict] = []
        pp_to_mpc: List[int] = []
        mpc_to_pp: List[int] = []
        steering_arr = _load(f, "control/steering", n)

        for i in range(1, n):
            prev = int(round(regime_raw[i - 1]))
            curr = int(round(regime_raw[i]))
            if prev == curr:
                continue
            direction = "PP→MPC" if (prev == 0 and curr >= 1) else ("MPC→PP" if (prev >= 1 and curr == 0) else "OTHER")
            delta_steering = (
                abs(float(steering_arr[i]) - float(steering_arr[i - 1]))
                if steering_arr is not None else None
            )
            bw_at_trans = (
                float(blend_weight[i]) if blend_weight is not None else None
            )
            transitions.append({
                "frame": i,
                "direction": direction,
                "delta_steering": _safe_float(delta_steering),
                "blend_weight": _safe_float(bw_at_trans, 1.0),
            })
            if direction == "PP→MPC":
                pp_to_mpc.append(i)
            elif direction == "MPC→PP":
                mpc_to_pp.append(i)

        max_delta = (
            max(t["delta_steering"] for t in transitions)
            if transitions else 0.0
        )
        speed_arr = _load(f, "vehicle/speed", n)
        mpc_speed = speed_arr[mpc_mask] if speed_arr is not None else np.array([])

        card1 = {
            "total_frames": n,
            "pp_frames": pp_n,
            "lmpc_frames": lmpc_n,
            "nmpc_frames": nmpc_n,
            "mpc_frames": mpc_n,
            "blend_frames": blend_n,
            "pp_rate": _safe_float(pp_n / n),
            "lmpc_rate": _safe_float(lmpc_n / n),
            "nmpc_rate": _safe_float(nmpc_n / n),
            "mpc_rate": _safe_float(mpc_n / n),
            "pp_to_mpc_count": len(pp_to_mpc),
            "mpc_to_pp_count": len(mpc_to_pp),
            "max_steering_delta_at_transition": _safe_float(max_delta),
            "transition_delta_gate_pass": bool(max_delta < 0.05),
            "transitions": transitions[:20],  # cap to avoid huge payloads
            "mpc_activation_speed_median": _safe_float(_pct(mpc_speed, 50)),
            "mpc_activation_speed_p95": _safe_float(_pct(mpc_speed, 95)),
        }

        if mpc_n == 0:
            return {
                "has_mpc": False,
                "card1": card1,
                "card2": None,
                "card3": None,
                "card4": None,
                "card5": None,
                "card6": None,
                "card7": None,
            }

        # ── Card 2: Pipeline Fix Status ──────────────────────────────────────
        # 2.8.1 — Recovery suppression
        rec_sup = _load(f, "control/mpc_recovery_mode_suppressed", n)
        rec_suppressed_n = int(np.sum(rec_sup[mpc_mask] > 0.5)) if rec_sup is not None else None
        rec_rate = _safe_float(rec_suppressed_n / mpc_n) if rec_suppressed_n is not None else None

        # 2.8.2 — Steering feedback error
        pre = _load(f, "control/mpc_last_steering_pre_modify", n)
        act = _load(f, "control/mpc_last_steering_actual", n)
        fb_err_p50 = fb_err_p95 = fb_err_max = None
        fb_gate = None
        if pre is not None and act is not None:
            err = np.abs(pre - act)[mpc_mask]
            fb_err_p50 = _safe_float(_pct(err, 50))
            fb_err_p95 = _safe_float(_pct(err, 95))
            fb_err_max = _safe_float(float(np.max(err)))
            fb_gate = bool(fb_err_p95 < 0.01) if fb_err_p95 is not None else None

        # 2.8.3 — Rate limiter
        rl = _load(f, "control/mpc_rate_limiter_active", n)
        rl_n = int(np.sum(rl[mpc_mask] > 0.5)) if rl is not None else None
        rl_rate = _safe_float(rl_n / mpc_n) if rl_n is not None else None

        # 2.8.5 — r_steer_rate scheduling
        rsr_eff = _load(f, "control/mpc_r_steer_rate_effective", n)
        rsr_mean = rsr_min = rsr_max = rsr_transitions = None
        if rsr_eff is not None:
            rsr_mpc = rsr_eff[mpc_mask]
            if len(rsr_mpc) > 0:
                rsr_mean = _safe_float(float(np.mean(rsr_mpc)))
                rsr_min = _safe_float(float(np.min(rsr_mpc)))
                rsr_max = _safe_float(float(np.max(rsr_mpc)))
                rsr_transitions = int(np.sum(np.abs(np.diff(rsr_mpc)) > 0.1))

        # 2.8.4 — Smith predictor
        raw_e  = _load(f, "control/mpc_smith_raw_e_lat", n)
        pred_e = _load(f, "control/mpc_smith_e_lat_predicted", n)
        delay  = _load(f, "control/mpc_delay_frames_used", n)
        smith_delta_p50 = smith_delta_p95 = None
        if raw_e is not None and pred_e is not None:
            sd = np.abs(pred_e - raw_e)[mpc_mask]
            smith_delta_p50 = _safe_float(_pct(sd, 50))
            smith_delta_p95 = _safe_float(_pct(sd, 95))
        delay_p50 = int(np.median(delay[mpc_mask])) if delay is not None else None

        # 2.8.6 — e_lat speed attenuation (loop gain reduction)
        mpc_e_lat = _load(f, "control/mpc_e_lat", n)
        elat_atten_ratio = None
        if raw_e is not None and mpc_e_lat is not None:
            # ratio of |MPC input| / |raw| — < 1.0 means attenuation active
            _raw_mpc = np.abs(raw_e[mpc_mask])
            _inp_mpc = np.abs(mpc_e_lat[mpc_mask])
            _valid = _raw_mpc > 0.01  # avoid div-by-zero on near-zero errors
            if np.sum(_valid) > 10:
                elat_atten_ratio = _safe_float(float(np.median(_inp_mpc[_valid] / _raw_mpc[_valid])))

        card2 = {
            # 2.8.1
            "recovery_suppressed_n": rec_suppressed_n,
            "recovery_suppressed_rate": rec_rate,
            # 2.8.2
            "feedback_error_p50": fb_err_p50,
            "feedback_error_p95": fb_err_p95,
            "feedback_error_max": fb_err_max,
            "feedback_error_gate_pass": fb_gate,
            # 2.8.3
            "rate_limiter_activations": rl_n,
            "rate_limiter_rate": rl_rate,
            "rate_limiter_gate": (
                "green" if rl_rate is not None and rl_rate < 0.05
                else "yellow" if rl_rate is not None and rl_rate < 0.15
                else "red"
            ),
            # 2.8.4
            "smith_delta_p50": smith_delta_p50,
            "smith_delta_p95": smith_delta_p95,
            "smith_delay_frames_p50": delay_p50,
            # 2.8.5
            "r_steer_rate_eff_mean": rsr_mean,
            "r_steer_rate_eff_min": rsr_min,
            "r_steer_rate_eff_max": rsr_max,
            "r_steer_rate_band_transitions": rsr_transitions,
            # 2.8.6
            "elat_atten_ratio": elat_atten_ratio,
        }

        # ── Card 3: MPC State Quality ────────────────────────────────────────
        e_lat = _load(f, "control/mpc_e_lat", n)
        e_hdg = _load(f, "control/mpc_e_heading", n)
        solve = _load(f, "control/mpc_solve_time_ms", n)
        fallback = _load(f, "control/mpc_fallback_active", n)
        kappa = _load(f, "control/mpc_kappa_ref", n)

        el_mpc = np.abs(e_lat[mpc_mask]) if e_lat is not None else np.array([])
        eh_mpc = np.abs(e_hdg[mpc_mask]) if e_hdg is not None else np.array([])
        sv_mpc = solve[mpc_mask] if solve is not None else np.array([])

        solve_p95 = _pct(sv_mpc, 95)
        card3 = {
            "e_lat_p50": _safe_float(_pct(el_mpc, 50)),
            "e_lat_p95": _safe_float(_pct(el_mpc, 95)),
            "e_lat_max": _safe_float(float(np.max(el_mpc)) if len(el_mpc) > 0 else 0.0),
            "e_lat_rmse": _safe_float(float(np.sqrt(np.mean(el_mpc ** 2))) if len(el_mpc) > 0 else 0.0),
            "e_heading_p50": _safe_float(_pct(eh_mpc, 50)),
            "e_heading_p95": _safe_float(_pct(eh_mpc, 95)),
            "e_heading_max": _safe_float(float(np.max(eh_mpc)) if len(eh_mpc) > 0 else 0.0),
            "solve_time_p50": _safe_float(_pct(sv_mpc, 50)),
            "solve_time_p95": _safe_float(solve_p95),
            "solve_time_max": _safe_float(float(np.max(sv_mpc)) if len(sv_mpc) > 0 else 0.0),
            "solve_time_gate_pass": bool(solve_p95 <= 5.0) if solve_p95 is not None else None,  # LMPC gate
            "fallback_rate": _safe_float(_rate(fallback[mpc_mask]) if fallback is not None else None),
            # Time series for chart overlay
            "e_lat_series": (e_lat.tolist() if e_lat is not None else []),
            "e_heading_series": (e_hdg.tolist() if e_hdg is not None else []),
            "regime_series": regime_raw.tolist(),
            "blend_weight_series": (blend_weight.tolist() if blend_weight is not None else []),
        }

        # ── Card 4: Severe Frame Inspector ──────────────────────────────────
        severe_frames: List[Dict] = []
        if e_lat is not None and mpc_n > 0:
            mpc_idxs = np.where(mpc_mask)[0]
            abs_e = np.abs(e_lat[mpc_mask])
            top_k = min(10, len(abs_e))
            top_order = np.argsort(abs_e)[::-1][:top_k]
            for ri in top_order:
                fr = int(mpc_idxs[ri])
                rec_v = (int(rec_sup[fr] > 0.5) if rec_sup is not None else None)
                rl_v  = (int(rl[fr] > 0.5) if rl is not None else None)
                sd_v  = (
                    abs(float(pred_e[fr]) - float(raw_e[fr]))
                    if (pred_e is not None and raw_e is not None)
                    else None
                )
                severe_frames.append({
                    "frame": fr,
                    "e_lat": _safe_float(float(e_lat[fr])),
                    "e_heading": _safe_float(float(e_hdg[fr]) if e_hdg is not None else 0.0),
                    "speed": _safe_float(float(speed_arr[fr]) if speed_arr is not None else 0.0),
                    "kappa": _safe_float(float(kappa[fr]) if kappa is not None else 0.0),
                    "recovery_suppressed": rec_v,
                    "rate_limiter_active": rl_v,
                    "smith_delta": _safe_float(sd_v),
                })

        card4 = {"severe_frames": severe_frames}

        # ── Card 5: NMPC Solver Health (Step 5) ─────────────────────────────
        card5 = None
        if nmpc_n > 0:
            nmpc_solve    = _load(f, "control/nmpc_solve_time_ms", n)
            nmpc_feasible = _load(f, "control/nmpc_feasible", n)
            nmpc_fallback = _load(f, "control/nmpc_fallback_active", n)
            nmpc_iters    = _load(f, "control/nmpc_iterations", n)
            nmpc_cost     = _load(f, "control/nmpc_cost", n)
            nmpc_consec   = _load(f, "control/nmpc_consecutive_failures", n)

            sv_nmpc = nmpc_solve[nmpc_mask] if nmpc_solve is not None else np.array([])
            nmpc_solve_p95 = _pct(sv_nmpc, 95)

            card5 = {
                "nmpc_frames": nmpc_n,
                "nmpc_rate": _safe_float(nmpc_n / n),
                "solve_time_p50": _safe_float(_pct(sv_nmpc, 50)),
                "solve_time_p95": _safe_float(nmpc_solve_p95),
                "solve_time_max": _safe_float(float(np.max(sv_nmpc)) if len(sv_nmpc) > 0 else 0.0),
                "solve_time_gate_pass": bool(nmpc_solve_p95 <= 20.0) if nmpc_solve_p95 is not None else None,
                "infeasible_rate": _safe_float(
                    1.0 - float(np.mean(nmpc_feasible[nmpc_mask] > 0.5))
                    if nmpc_feasible is not None else None
                ),
                "fallback_rate": _safe_float(
                    _rate(nmpc_fallback[nmpc_mask]) if nmpc_fallback is not None else None
                ),
                "iterations_p50": _safe_float(_pct(nmpc_iters[nmpc_mask], 50) if nmpc_iters is not None else None),
                "iterations_p95": _safe_float(_pct(nmpc_iters[nmpc_mask], 95) if nmpc_iters is not None else None),
                "cost_p50": _safe_float(_pct(nmpc_cost[nmpc_mask], 50) if nmpc_cost is not None else None),
                "cost_p95": _safe_float(_pct(nmpc_cost[nmpc_mask], 95) if nmpc_cost is not None else None),
                "max_consecutive_failures": int(np.max(nmpc_consec)) if nmpc_consec is not None else 0,
            }

        # ── Card 6: L_eff Estimation ─────────────────────────────────────
        card6 = None
        mpc_leff_raw = _load(f, "control/mpc_leff_value", n)
        leff_innov_raw = _load(f, "control/mpc_leff_innovation", n)
        if (mpc_leff_raw is not None and leff_innov_raw is not None
                and np.any(mpc_leff_raw != 0)):
            mpc_leff = mpc_leff_raw[mpc_mask]
            leff_innov_mpc = leff_innov_raw[mpc_mask]
            if len(mpc_leff) > 0:
                converged = (
                    "yes" if len(mpc_leff) > 100 and np.std(mpc_leff[-100:]) < 0.01
                    else "no"
                )
                card6 = {
                    "title": "L_eff Estimation",
                    "items": [
                        f"L_eff final: {mpc_leff[-1]:.3f} m (nominal: 2.500 m)",
                        f"L_eff range: [{np.min(mpc_leff):.3f}, {np.max(mpc_leff):.3f}] m",
                        f"Innovation P95: {_safe_float(_pct(np.abs(leff_innov_mpc), 95)):.4f} rad",
                        f"Converged: {converged}",
                    ],
                }

        # ── Card 7: Tire Estimation (Dynamic Bicycle) ─────────────────────
        card7 = None
        tire_cf_raw = _load(f, "control/mpc_tire_cf", n)
        tire_cr_raw = _load(f, "control/mpc_tire_cr", n)
        tire_innov_raw = _load(f, "control/mpc_tire_ekf_innovation", n)
        tire_kus_raw = _load(f, "control/mpc_tire_understeer_gradient", n)
        if (tire_cf_raw is not None and np.any(tire_cf_raw != 0)):
            cf_mpc = tire_cf_raw[mpc_mask]
            cr_mpc = tire_cr_raw[mpc_mask] if tire_cr_raw is not None else np.array([])
            if len(cf_mpc) > 0:
                items = [
                    f"C_f final: {cf_mpc[-1]:,.0f} N/rad (nominal: 40,000)",
                ]
                if len(cr_mpc) > 0:
                    items.append(f"C_r final: {cr_mpc[-1]:,.0f} N/rad (nominal: 40,000)")
                if tire_kus_raw is not None:
                    kus_mpc = tire_kus_raw[mpc_mask]
                    if len(kus_mpc) > 0:
                        items.append(f"Understeer gradient K_us: {kus_mpc[-1]:.4f} rad/(m/s\u00b2)")
                if tire_innov_raw is not None:
                    innov_mpc = np.abs(tire_innov_raw[mpc_mask])
                    if len(innov_mpc) > 0:
                        innov_p95 = _safe_float(_pct(innov_mpc, 95))
                        gate = "PASS" if innov_p95 < 0.05 else "FAIL"
                        items.append(f"Innovation P95: {innov_p95:.4f} rad/s [{gate} < 0.05]")
                # Convergence: std of last 100 C_f values < 100 → converged
                converged = (
                    "yes" if len(cf_mpc) > 100 and np.std(cf_mpc[-100:]) < 100
                    else "no"
                )
                items.append(f"Converged: {converged}")
                card7 = {
                    "title": "Tire Estimation",
                    "items": items,
                }

        return {
            "has_mpc": True,
            "recording": filepath.name,
            "card1": card1,
            "card2": card2,
            "card3": card3,
            "card4": card4,
            "card5": card5,
            "card6": card6,
            "card7": card7,
        }
