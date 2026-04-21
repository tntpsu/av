#!/usr/bin/env python3
"""
Phase 0.5 empirical instrument for the ACC IDM accel plumbing plan.

Verifies the three gates before B2 activation:
  0.5.1 — IDM coherence on safe frames: corr(acc_idm_accel_mps2, d(speed)/dt)
          on NON-emergency frames. If > 0.5, IDM's accel is a sensible
          physics intent that is safe to plumb. If < 0.3, IDM is noisy and
          B2 would inject noise into the controller.

  0.5.2 — IDM leads speed-target signal: cross-correlation of IDM accel vs
          speed-error at Δ ∈ {-0.3s, 0, +0.3s}. IDM should anticipate speed
          error; this proves today's controller sees a lagged signal.

  0.5.3 — Save-the-collision counterfactual: simulate what would happen if
          reference_accel = acc_idm_accel_mps2 were plumbed through the
          longitudinal controller's jerk limiter and decel clip, starting
          from the 14-frame warning window before the first collision.
          Two variants:
            - no_relaxation: clip at -3 m/s², jerk floor 1.5
            - emergency_relaxation: clip at -12 m/s², jerk floor 8.0 in
              EMERGENCY_BRAKE states

Usage:
    python3 tools/analyze/analyze_acc_command_chain.py <recording.h5>
    python3 tools/analyze/analyze_acc_command_chain.py --all
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np


# --- Controller constants (mirror pid_controller.py defaults) ---
MAX_DECEL = 3.0
MAX_ACCEL = 2.5
MAX_JERK_MIN = 1.5
MAX_JERK_MAX = 6.0
JERK_ERROR_MIN = 0.5
JERK_ERROR_MAX = 3.0
ACCEL_SMOOTHING_ALPHA = 0.6
EMERGENCY_JERK_MAX = 10.0
EMERGENCY_DECEL_FLOOR = -12.0


def _decode(raw):
    if isinstance(raw, bytes):
        return raw.decode()
    return str(raw)


def _load_recording(path: Path) -> dict:
    """Pull all signals the analyzer needs into a dict of numpy arrays."""
    with h5py.File(path, "r") as f:
        d = {
            "idm_accel":   f["vehicle/acc_idm_accel_mps2"][:].astype(float),
            "state":       np.array([_decode(s) for s in f["vehicle/acc_state_code"][:]]),
            "estop_req":   f["vehicle/acc_request_estop"][:].astype(int),
            "ego_speed":   f["vehicle/speed"][:].astype(float),
            "target_speed": f["vehicle/acc_target_speed_mps"][:].astype(float),
            "accel_raw":   f["control/longitudinal_accel_cmd_raw"][:].astype(float),
            "brake_raw":   f["control/brake_before_limits"][:].astype(float),
            "brake":       f["control/brake"][:].astype(float),
            "dt":          f["control/interframe_dt_actual"][:].astype(float),
            "grade":       f["vehicle/road_grade"][:].astype(float),
            "gap_err":     f["vehicle/acc_gap_error_m"][:].astype(float),
            "target_gap":  f["vehicle/acc_target_gap_m"][:].astype(float),
            "lead_speed":  f["vehicle/acc_lead_speed_estimate_mps"][:].astype(float),
            "acc_active":  f["vehicle/acc_active"][:].astype(float),
        }
    d["actual_gap"] = d["target_gap"] + d["gap_err"]
    return d


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 10 or len(y) < 10:
        return float("nan")
    sx = np.std(x); sy = np.std(y)
    if sx < 1e-9 or sy < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ---------- Gate 0.5.1: IDM coherence on safe frames ----------
def gate_0_5_1_idm_coherence(d: dict) -> dict:
    """
    On NON-failure frames (ACC_ACTIVE or CUTOUT), does IDM's commanded accel
    correlate with what the ego actually does?

    "Safe" excludes COLLAPSED_GAP_STOP (failure state), EMERGENCY_BRAKE, and
    TTC_ESTOP. This is where the controller is in normal following regime.
    """
    safe_states = np.isin(d["state"], ["ACC_ACTIVE", "CUTOUT"])
    safe = (d["acc_active"] > 0.5) & safe_states

    # Measured accel: finite-difference on ego_speed over recorded dt.
    # Pad with 0 at index 0.
    dv = np.zeros_like(d["ego_speed"])
    dt = np.where(d["dt"] > 1e-3, d["dt"], 1.0 / 13.0)  # guard against zeros
    dv[1:] = (d["ego_speed"][1:] - d["ego_speed"][:-1]) / dt[1:]

    idm = d["idm_accel"][safe]
    meas = dv[safe]

    corr = _safe_corr(idm, meas)
    indeterminate = int(safe.sum()) < 20  # too few safe frames to test
    return {
        "frames_used": int(safe.sum()),
        "corr_idm_vs_measured_accel": corr,
        "idm_p50": float(np.median(idm)) if len(idm) else float("nan"),
        "idm_p95_abs": float(np.percentile(np.abs(idm), 95)) if len(idm) else float("nan"),
        "measured_p50": float(np.median(meas)) if len(meas) else float("nan"),
        "indeterminate": indeterminate,
        "pass": bool(not indeterminate and corr > 0.5),
        "soft_pass": bool(not indeterminate and corr > 0.3 and corr <= 0.5),
        "interpretation": (
            "INDETERMINATE: too few safe frames in recording. Scenario does not "
            "exercise normal ACC following — use a different recording to verify "
            "IDM coherence (e.g., a highway ACC scenario without immediate "
            "emergency). This is NOT a failure signal; it's a testability gap."
            if indeterminate else
            "IDM accel coherent with ego accel on safe frames — safe to plumb."
            if corr > 0.5 else
            "IDM accel weakly coherent — B2 may need low-pass filter."
            if corr > 0.3 else
            "IDM accel NOT coherent — B2 would inject noise. STOP."
        ),
    }


# ---------- Gate 0.5.2: IDM leads speed target ----------
def gate_0_5_2_idm_leads_speed_error(d: dict) -> dict:
    """
    Cross-correlation of IDM accel vs speed_error at different lags.
    Speed-error at future time should correlate with IDM-now if IDM anticipates.
    Lag Δ in seconds → frames ≈ Δ / median_dt.
    """
    active = d["acc_active"] > 0.5
    speed_error = d["target_speed"] - d["ego_speed"]
    idm = d["idm_accel"]

    median_dt = float(np.median(d["dt"][d["dt"] > 1e-3])) if np.any(d["dt"] > 1e-3) else 1.0 / 13.0
    out = {"median_dt_s": median_dt, "frames_active": int(active.sum())}
    for delta_s in [-0.3, 0.0, 0.3, 0.6]:
        lag = int(round(delta_s / median_dt))
        n = len(idm)
        if lag >= 0:
            x = idm[:n - lag] if lag > 0 else idm
            y = speed_error[lag:] if lag > 0 else speed_error
        else:
            x = idm[-lag:]
            y = speed_error[:lag]
        mask = active[:len(x)] if lag >= 0 else active[-lag:len(x) - lag]
        mask = mask[:min(len(mask), len(x), len(y))]
        x = x[:len(mask)][mask]; y = y[:len(mask)][mask]
        out[f"corr_lag_{delta_s:+.1f}s"] = _safe_corr(x, y)

    # Pass criterion: corr at Δ=+0.3s (future speed_error given IDM now) is
    # meaningfully higher than Δ=0.  If it is, IDM predicts future error.
    lead_gain = out["corr_lag_+0.3s"] - out["corr_lag_+0.0s"]
    out["lead_gain_vs_zero"] = lead_gain
    out["pass"] = bool(abs(lead_gain) > 0.05)
    out["interpretation"] = (
        f"IDM leads future speed-error by {lead_gain:+.3f} — anticipation confirmed."
        if abs(lead_gain) > 0.05 else
        "IDM and speed-error are contemporaneous — no lead — B2 benefit less certain."
    )
    return out


# ---------- Gate 0.5.3: Save-the-collision counterfactual ----------
def _find_collision_window(d: dict, pre_frames: int = 30, post_frames: int = 20) -> dict | None:
    """
    Find the first frame where the state machine entered COLLAPSED_GAP_STOP —
    the "pinned against lead" failure state. This is the scoring-relevant
    collision, not an actual_gap threshold (Unity clips gap at ~0.10m floor,
    so gap never goes strictly negative).

    Return the window [collision - pre_frames, collision + post_frames].
    """
    collapsed = d["state"] == "COLLAPSED_GAP_STOP"
    if not collapsed.any():
        return None
    collision_f = int(np.argmax(collapsed))  # first True
    # Ensure there's enough history
    if collision_f < 3:
        return None
    start = max(0, collision_f - pre_frames)
    end = min(len(d["state"]), collision_f + post_frames)
    return {"collision_frame": collision_f, "start": start, "end": end}


def _simulate_controller(
    ref_accel: np.ndarray,
    ref_speed: np.ndarray,
    ego_speed_init: float,
    lead_speed: np.ndarray,
    dt: np.ndarray,
    state: np.ndarray,
    estop_req: np.ndarray,
    grade: np.ndarray,
    relaxation_enabled: bool,
    start_accel: float = 0.0,
) -> dict:
    """
    Simplified simulation of the longitudinal controller's accel/jerk chain,
    then forward-integrate ego speed and gap closure.

    This is NOT a full replay — it ignores throttle-PID, grade-FF throttle
    dynamics, and the many secondary clips. It captures the two terms that
    actually matter for decel response: the accel clip and the jerk limiter.
    """
    n = len(ref_accel)
    accel_applied = np.zeros(n)
    speed_sim = np.zeros(n)
    speed_sim[0] = ego_speed_init
    smoothed_accel = start_accel
    last_applied = start_accel

    for i in range(n):
        # Emergency state check
        s = state[i]
        in_emergency = bool(estop_req[i]) or s in ("EMERGENCY_BRAKE", "TTC_ESTOP", "COLLAPSED_GAP_STOP")

        # Accel clip
        g_accel = 9.81 * math.sin(grade[i])
        eff_max_accel = MAX_ACCEL + abs(g_accel)
        eff_max_decel = MAX_DECEL + abs(g_accel)
        if in_emergency and relaxation_enabled:
            eff_max_decel = max(eff_max_decel, abs(EMERGENCY_DECEL_FLOOR) + abs(g_accel))
        desired = float(np.clip(ref_accel[i], -eff_max_decel, eff_max_accel))

        # Smoothing (alpha=0.6, same as controller)
        smoothed_accel = ACCEL_SMOOTHING_ALPHA * smoothed_accel + (1 - ACCEL_SMOOTHING_ALPHA) * desired

        # Dynamic jerk
        speed_err_mag = abs(ref_speed[i] - speed_sim[max(i - 1, 0)])
        if JERK_ERROR_MAX > JERK_ERROR_MIN:
            ratio = float(np.clip(
                (speed_err_mag - JERK_ERROR_MIN) / (JERK_ERROR_MAX - JERK_ERROR_MIN),
                0.0, 1.0,
            ))
        else:
            ratio = 1.0
        dyn_jerk = MAX_JERK_MIN + ratio * (MAX_JERK_MAX - MAX_JERK_MIN)
        if in_emergency and relaxation_enabled:
            dyn_jerk = max(dyn_jerk, EMERGENCY_JERK_MAX)

        # Jerk-limit the change
        dt_i = max(float(dt[i]), 1e-3)
        max_delta = dyn_jerk * dt_i
        delta = smoothed_accel - last_applied
        delta = float(np.clip(delta, -max_delta, max_delta))
        applied = last_applied + delta
        last_applied = applied
        accel_applied[i] = applied

        # Integrate speed (+ gravity)
        if i < n - 1:
            speed_sim[i + 1] = max(0.0, speed_sim[i] + (applied + g_accel) * dt_i)

    return {"accel_applied": accel_applied, "speed_sim": speed_sim}


def gate_0_5_3_save_counterfactual(d: dict) -> dict:
    """
    Simulate the 14-frame warning window with and without B2, and see
    whether gap stays positive.
    """
    win = _find_collision_window(d)
    if win is None:
        return {"pass": True, "no_collision_in_recording": True,
                "interpretation": "No frames with gap < 0 found — no collision to counterfactual."}

    s, e = win["start"], win["end"]
    idx = slice(s, e)
    n_win = e - s
    # Slice window
    idm = d["idm_accel"][idx]
    ref_v = d["target_speed"][idx]
    dt = d["dt"][idx]
    state = d["state"][idx]
    estop_req = d["estop_req"][idx]
    grade = d["grade"][idx]
    lead_speed = d["lead_speed"][idx]
    actual_gap_orig = d["actual_gap"][idx]
    ego_init = float(d["ego_speed"][s])

    # Baseline actual (from recording): we already know it collided
    baseline_min_gap = float(actual_gap_orig.min())

    # Counterfactual 1: B2 no-relaxation (clip at -3)
    sim1 = _simulate_controller(idm, ref_v, ego_init, lead_speed, dt, state, estop_req, grade,
                                relaxation_enabled=False, start_accel=float(d["accel_raw"][s]))
    # Integrate gap: gap(k+1) = gap(k) + (v_lead(k) - v_ego(k)) * dt
    gap1 = np.zeros(n_win)
    gap1[0] = actual_gap_orig[0]
    for k in range(n_win - 1):
        closing = sim1["speed_sim"][k] - lead_speed[k]
        gap1[k + 1] = gap1[k] - closing * max(float(dt[k]), 1e-3)

    # Counterfactual 2: B2 with emergency relaxation
    sim2 = _simulate_controller(idm, ref_v, ego_init, lead_speed, dt, state, estop_req, grade,
                                relaxation_enabled=True, start_accel=float(d["accel_raw"][s]))
    gap2 = np.zeros(n_win)
    gap2[0] = actual_gap_orig[0]
    for k in range(n_win - 1):
        closing = sim2["speed_sim"][k] - lead_speed[k]
        gap2[k + 1] = gap2[k] - closing * max(float(dt[k]), 1e-3)

    return {
        "collision_frame": win["collision_frame"],
        "window": [s, e],
        "baseline_min_gap_m": baseline_min_gap,
        "counterfactual_1_no_relaxation": {
            "min_gap_m": float(gap1.min()),
            "min_ego_speed": float(sim1["speed_sim"].min()),
            "max_applied_decel": float(sim1["accel_applied"].min()),
            "avoids_collision": bool(gap1.min() >= 1.0),
        },
        "counterfactual_2_with_relaxation": {
            "min_gap_m": float(gap2.min()),
            "min_ego_speed": float(sim2["speed_sim"].min()),
            "max_applied_decel": float(sim2["accel_applied"].min()),
            "avoids_collision": bool(gap2.min() >= 1.0),
        },
        "pass": bool(gap2.min() >= 1.0),
        "interpretation": (
            f"B2 w/ relaxation holds min_gap = {gap2.min():.2f} m (baseline was {baseline_min_gap:.2f}). "
            f"B2 alone achieves {gap1.min():.2f} m. "
            + ("Plan is viable." if gap2.min() >= 1.0 else
               "B2 + relaxation insufficient — B1 strictly required.")
        ),
    }


def run_analysis(path: Path) -> dict:
    d = _load_recording(path)
    out = {
        "recording": str(path),
        "frames_total": len(d["idm_accel"]),
        "gate_0_5_1": gate_0_5_1_idm_coherence(d),
        "gate_0_5_2": gate_0_5_2_idm_leads_speed_error(d),
        "gate_0_5_3": gate_0_5_3_save_counterfactual(d),
    }
    g1 = out["gate_0_5_1"]
    g1_ok_or_indet = g1["pass"] or g1.get("soft_pass", False) or g1.get("indeterminate", False)
    out["verdict"] = {
        "phase_0_5_1_pass_or_indeterminate": bool(g1_ok_or_indet),
        "phase_0_5_3_pass": out["gate_0_5_3"]["pass"],
        "b2_viable": bool(g1_ok_or_indet and out["gate_0_5_3"]["pass"]),
    }
    return out


def _fmt(r: dict) -> str:
    lines = []
    lines.append(f"Recording: {r['recording']}")
    lines.append(f"Frames total: {r['frames_total']}")
    lines.append("")
    lines.append("--- Gate 0.5.1: IDM coherence on safe frames ---")
    g = r["gate_0_5_1"]
    lines.append(f"  frames_used: {g['frames_used']}")
    lines.append(f"  corr(idm, measured_accel): {g['corr_idm_vs_measured_accel']:+.4f}")
    lines.append(f"  idm p50 / p95(abs): {g['idm_p50']:+.2f} / {g['idm_p95_abs']:.2f}")
    if g.get("indeterminate"):
        result = "INDETERMINATE"
    elif g["pass"]:
        result = "PASS"
    elif g.get("soft_pass"):
        result = "SOFT-PASS"
    else:
        result = "FAIL"
    lines.append(f"  result: {result}")
    lines.append(f"  {g['interpretation']}")
    lines.append("")
    lines.append("--- Gate 0.5.2: IDM leads speed-error ---")
    g = r["gate_0_5_2"]
    for k, v in g.items():
        if k.startswith("corr_lag_"):
            lines.append(f"  {k}: {v:+.4f}")
    lines.append(f"  lead_gain_vs_zero: {g['lead_gain_vs_zero']:+.4f}")
    lines.append(f"  result: {'PASS' if g['pass'] else 'INFO'}")
    lines.append(f"  {g['interpretation']}")
    lines.append("")
    lines.append("--- Gate 0.5.3: Save-the-collision counterfactual ---")
    g = r["gate_0_5_3"]
    if g.get("no_collision_in_recording"):
        lines.append(f"  {g['interpretation']}")
    else:
        lines.append(f"  collision_frame: {g['collision_frame']}")
        lines.append(f"  baseline_min_gap: {g['baseline_min_gap_m']:+.3f} m")
        c1 = g["counterfactual_1_no_relaxation"]
        lines.append(f"  B2-only (no relaxation):")
        lines.append(f"    min_gap: {c1['min_gap_m']:+.3f} m   max_decel_applied: {c1['max_applied_decel']:+.2f}   min_speed: {c1['min_ego_speed']:.2f}")
        lines.append(f"    avoids_collision: {c1['avoids_collision']}")
        c2 = g["counterfactual_2_with_relaxation"]
        lines.append(f"  B2 + emergency relaxation:")
        lines.append(f"    min_gap: {c2['min_gap_m']:+.3f} m   max_decel_applied: {c2['max_applied_decel']:+.2f}   min_speed: {c2['min_ego_speed']:.2f}")
        lines.append(f"    avoids_collision: {c2['avoids_collision']}")
        lines.append(f"  result: {'PASS' if g['pass'] else 'FAIL'}")
        lines.append(f"  {g['interpretation']}")
    lines.append("")
    lines.append("--- Verdict ---")
    v = r["verdict"]
    lines.append(f"  0.5.1 pass or indeterminate: {v['phase_0_5_1_pass_or_indeterminate']}")
    lines.append(f"  0.5.3 pass: {v['phase_0_5_3_pass']}")
    lines.append(f"  B2 viable for this recording: {v['b2_viable']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", nargs="?", default=None, help="Path to recording .h5")
    ap.add_argument("--all", action="store_true",
                    help="Run on the two plan-cited recordings (H5 and G2)")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable")
    args = ap.parse_args()

    targets = []
    if args.all:
        targets = [
            Path("data/recordings/recording_20260419_234404.h5"),
            Path("data/recordings/recording_20260420_000022.h5"),
        ]
    elif args.recording:
        targets = [Path(args.recording)]
    else:
        ap.error("Provide a recording path or --all")

    all_out = []
    for p in targets:
        if not p.exists():
            print(f"ERROR: recording not found: {p}", file=sys.stderr)
            continue
        r = run_analysis(p)
        all_out.append(r)
        if not args.json:
            print(_fmt(r))
            print("=" * 60)

    if args.json:
        print(json.dumps(all_out, indent=2))

    # Write machine-readable report for regression tracking
    reports_dir = Path("data/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "acc_idm_plumbing_phase_0_5.json"
    out_path.write_text(json.dumps(all_out, indent=2))
    if not args.json:
        print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()
