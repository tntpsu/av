#!/usr/bin/env python3
"""
MPC Prediction vs Reality Diagnostic

Computes one-step prediction error: what the MPC's internal model predicted
would happen at frame k+1 vs what actually happened. Decomposes the error
into component terms (heading, curvature, v_y, tire forces) to identify
which part of the model is wrong.

Usage:
    python tools/analyze/diagnose_mpc_prediction.py --latest
    python tools/analyze/diagnose_mpc_prediction.py --recording path/to/recording.h5
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Recording loader
# ---------------------------------------------------------------------------

def find_latest_recording() -> str:
    rec_dir = Path("data/recordings")
    recordings = sorted(rec_dir.glob("recording_*.h5"), key=os.path.getmtime)
    if not recordings:
        sys.exit("No recordings found in data/recordings/")
    return str(recordings[-1])


def load_recording(path: str) -> dict:
    """Load relevant fields from HDF5 recording."""
    f = h5py.File(path, "r")
    ctrl = f["control"]
    veh = f["vehicle"]
    gt = f["ground_truth"]

    def get(group, key, default=None):
        if key in group:
            return group[key][:]
        return default

    data = {
        # MPC state inputs
        "mpc_e_lat": get(ctrl, "mpc_e_lat"),
        "mpc_e_heading": get(ctrl, "mpc_e_heading"),
        "mpc_v_y": get(ctrl, "mpc_v_y_estimate"),
        "mpc_yaw_rate": get(ctrl, "mpc_yaw_rate_estimate"),
        "mpc_kappa_ref": get(ctrl, "mpc_kappa_ref"),
        "mpc_tire_cf": get(ctrl, "mpc_tire_cf"),
        "mpc_tire_cr": get(ctrl, "mpc_tire_cr"),
        "mpc_feasible": get(ctrl, "mpc_feasible"),
        "mpc_dynamic_model_active": get(ctrl, "mpc_dynamic_model_active"),
        # Control outputs
        "steering_deg": get(ctrl, "calculated_steering_angle_deg"),
        "steering_sent": get(ctrl, "steering"),
        # Vehicle state
        "speed": get(veh, "speed"),
        "steering_actual_deg": get(veh, "steering_angle_actual"),
        # Regime
        "regime": get(ctrl, "regime"),
        # Ground truth
        "gt_cross_track": get(gt, "ego_lane_cross_track_road_frame_at_car"),
        "gt_heading": get(gt, "desired_heading"),
        "gt_curvature": get(gt, "path_curvature"),
        # Smith predictor fields
        "smith_e_lat": get(ctrl, "mpc_smith_e_lat_predicted"),
        "smith_e_heading": get(ctrl, "mpc_smith_e_heading_predicted"),
    }
    f.close()
    return data


# ---------------------------------------------------------------------------
# Vehicle model parameters (match mpc_controller.py defaults + config)
# ---------------------------------------------------------------------------

MASS = 1500.0       # kg
IZ = 2428.0         # kg*m^2
LF = 1.125          # m
LR = 1.125          # m
L = LF + LR         # wheelbase
MAX_STEER_LOW = 0.5236   # rad (30 deg)
MAX_STEER_HIGH = 0.2793  # rad (16 deg)
SPEED_SAT = 12.0         # m/s


def max_steer_at_speed(v: float) -> float:
    """Speed-dependent max steering angle (matches mpc_controller.py)."""
    if v <= 0:
        return MAX_STEER_LOW
    t = min(v / SPEED_SAT, 1.0)
    return MAX_STEER_LOW + t * (MAX_STEER_HIGH - MAX_STEER_LOW)


# ---------------------------------------------------------------------------
# One-step prediction
# ---------------------------------------------------------------------------

def predict_next_state_dynamic(
    e_lat, e_heading, v_y, r, vx, delta_norm, kappa, cf, cr, dt
):
    """One-step dynamic bicycle model prediction (5-state).
    Returns (e_lat_next, e_heading_next, v_y_next, r_next, decomposition).
    """
    vx_safe = max(vx, 3.0)
    ms = max_steer_at_speed(vx_safe)

    # Lateral position: e_lat += vx * e_heading * dt + v_y * dt
    heading_term = vx_safe * e_heading * dt
    vy_lat_term = v_y * dt
    e_lat_next = e_lat + heading_term + vy_lat_term

    # Heading: e_heading += r * dt - kappa * vx * dt
    yaw_term = r * dt
    curvature_term = -kappa * vx_safe * dt
    e_heading_next = e_heading + yaw_term + curvature_term

    # Lateral velocity (v_y)
    a_vy_vy = 1.0 - (cf + cr) / (MASS * vx_safe) * dt
    a_vy_r = (-vx_safe - (cf * LF - cr * LR) / (MASS * vx_safe)) * dt
    b_vy = cf * ms / MASS * dt
    v_y_next = v_y * a_vy_vy + r * a_vy_r + delta_norm * b_vy

    # Yaw rate (r)
    a_r_vy = (-(cf * LF - cr * LR) / (IZ * vx_safe)) * dt
    a_r_r = 1.0 - (cf * LF**2 + cr * LR**2) / (IZ * vx_safe) * dt
    b_r = cf * LF * ms / IZ * dt
    r_next = v_y * a_r_vy + r * a_r_r + delta_norm * b_r

    decomp = {
        "heading_term": heading_term,
        "vy_lat_term": vy_lat_term,
        "yaw_term": yaw_term,
        "curvature_term": curvature_term,
        "a_vy_vy": a_vy_vy,
        "b_vy": b_vy,
        "b_r": b_r,
    }
    return e_lat_next, e_heading_next, v_y_next, r_next, decomp


def predict_next_state_kinematic(e_lat, e_heading, vx, delta_norm, kappa, dt):
    """One-step kinematic model prediction (3-state)."""
    vx_safe = max(vx, 3.0)
    ms = max_steer_at_speed(vx_safe)
    steer_gain = (vx_safe / L) * ms * dt

    heading_term = vx_safe * e_heading * dt
    e_lat_next = e_lat + heading_term

    steer_term = delta_norm * steer_gain
    curvature_term = -kappa * vx_safe * dt
    e_heading_next = e_heading + steer_term + curvature_term

    return e_lat_next, e_heading_next


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(recording_path: str):
    data = load_recording(recording_path)
    N = len(data["speed"])
    print(f"Recording: {os.path.basename(recording_path)}")
    print(f"Total frames: {N}")

    # Identify MPC-active frames (regime=1 for LMPC)
    regime = data["regime"]
    feasible = data["mpc_feasible"]
    is_dynamic = data["mpc_dynamic_model_active"]

    # Need consecutive feasible MPC frames for prediction comparison
    mpc_mask = (regime == 1) & (feasible == 1)
    # Pairs: frame k and k+1 both MPC-active and feasible
    pair_mask = mpc_mask[:-1] & mpc_mask[1:]
    pair_idx = np.where(pair_mask)[0]

    if len(pair_idx) == 0:
        print("No consecutive feasible MPC frame pairs found.")
        return

    print(f"Consecutive feasible MPC pairs: {len(pair_idx)}")

    # Compute dt from speed (assume ~13 FPS = 77ms)
    # Use frame-to-frame dt if timestamps available, otherwise fixed
    dt = 0.077  # nominal dt

    # Compute steering in normalized [-1, 1] from degrees
    speed = data["speed"]
    steer_deg = data["steering_sent"]  # actual sent steering in degrees

    # Classify frames as straight/curve using ground truth curvature
    gt_curv = np.abs(data["gt_curvature"]) if data["gt_curvature"] is not None else np.zeros(N)
    CURVE_THRESHOLD = 0.005  # 1/m → R=200m

    # Storage for prediction errors
    pred_errors_elat = []
    pred_errors_ehead = []
    pred_errors_vy = []
    pred_errors_r = []
    decomp_heading = []
    decomp_vy_lat = []
    decomp_yaw = []
    decomp_curv = []
    actual_delta_elat = []
    actual_delta_ehead = []
    frame_speeds = []
    frame_curvatures = []
    frame_indices = []

    for k in pair_idx:
        # State at frame k (what MPC sees)
        e_lat_k = data["mpc_e_lat"][k]
        e_head_k = data["mpc_e_heading"][k]
        v_y_k = data["mpc_v_y"][k] if data["mpc_v_y"] is not None else 0.0
        r_k = data["mpc_yaw_rate"][k] if data["mpc_yaw_rate"] is not None else 0.0
        vx_k = speed[k]
        kappa_k = data["mpc_kappa_ref"][k]
        cf_k = data["mpc_tire_cf"][k]
        cr_k = data["mpc_tire_cr"][k]

        # Steering at frame k (normalized)
        ms_k = max_steer_at_speed(vx_k)
        steer_rad = math.radians(steer_deg[k]) if steer_deg is not None else 0.0
        delta_norm_k = steer_rad / max(ms_k, 1e-6)
        delta_norm_k = max(-1.0, min(1.0, delta_norm_k))

        # Actual state at frame k+1
        e_lat_k1 = data["mpc_e_lat"][k + 1]
        e_head_k1 = data["mpc_e_heading"][k + 1]
        v_y_k1 = data["mpc_v_y"][k + 1] if data["mpc_v_y"] is not None else 0.0
        r_k1 = data["mpc_yaw_rate"][k + 1] if data["mpc_yaw_rate"] is not None else 0.0

        # Predict next state using the MPC's own model
        if is_dynamic is not None and is_dynamic[k]:
            pred_elat, pred_ehead, pred_vy, pred_r, dec = predict_next_state_dynamic(
                e_lat_k, e_head_k, v_y_k, r_k, vx_k, delta_norm_k, kappa_k, cf_k, cr_k, dt
            )
            pred_errors_vy.append(pred_vy - v_y_k1)
            pred_errors_r.append(pred_r - r_k1)
            decomp_heading.append(dec["heading_term"])
            decomp_vy_lat.append(dec["vy_lat_term"])
            decomp_yaw.append(dec["yaw_term"])
            decomp_curv.append(dec["curvature_term"])
        else:
            pred_elat, pred_ehead = predict_next_state_kinematic(
                e_lat_k, e_head_k, vx_k, delta_norm_k, kappa_k, dt
            )
            decomp_heading.append(vx_k * e_head_k * dt)
            decomp_vy_lat.append(0.0)
            decomp_yaw.append(0.0)
            decomp_curv.append(-kappa_k * vx_k * dt)

        pred_errors_elat.append(pred_elat - e_lat_k1)
        pred_errors_ehead.append(pred_ehead - e_head_k1)
        actual_delta_elat.append(e_lat_k1 - e_lat_k)
        actual_delta_ehead.append(e_head_k1 - e_head_k)
        frame_speeds.append(vx_k)
        frame_curvatures.append(abs(kappa_k))
        frame_indices.append(k)

    pred_errors_elat = np.array(pred_errors_elat)
    pred_errors_ehead = np.array(pred_errors_ehead)
    actual_delta_elat = np.array(actual_delta_elat)
    actual_delta_ehead = np.array(actual_delta_ehead)
    frame_curvatures = np.array(frame_curvatures)
    frame_speeds = np.array(frame_speeds)
    decomp_heading = np.array(decomp_heading)
    decomp_vy_lat = np.array(decomp_vy_lat)
    decomp_yaw = np.array(decomp_yaw)
    decomp_curv = np.array(decomp_curv)

    # Segment into straight and curve
    is_curve = frame_curvatures > CURVE_THRESHOLD
    is_straight = ~is_curve

    print("\n" + "=" * 72)
    print("MPC ONE-STEP PREDICTION ERROR")
    print("=" * 72)
    print(f"Model: {'dynamic 5-state' if is_dynamic is not None and np.mean(is_dynamic[pair_idx]) > 0.5 else 'kinematic 3-state'}")
    print(f"dt (assumed): {dt*1000:.0f} ms")
    print(f"Curve threshold: |kappa| > {CURVE_THRESHOLD} (R < {1/CURVE_THRESHOLD:.0f}m)")

    def report_segment(name, mask):
        n = np.sum(mask)
        if n == 0:
            print(f"\n--- {name}: no frames ---")
            return
        err_el = pred_errors_elat[mask]
        err_eh = pred_errors_ehead[mask]
        del_el = actual_delta_elat[mask]
        del_eh = actual_delta_ehead[mask]
        d_head = decomp_heading[mask]
        d_vy = decomp_vy_lat[mask]
        d_yaw = decomp_yaw[mask]
        d_curv = decomp_curv[mask]

        print(f"\n--- {name} ({n} frames) ---")
        print(f"  e_lat prediction error:   mean={np.mean(err_el)*1000:+.2f} mm   "
              f"RMS={np.sqrt(np.mean(err_el**2))*1000:.2f} mm   "
              f"P95={np.percentile(np.abs(err_el), 95)*1000:.2f} mm")
        print(f"  e_heading prediction err: mean={np.mean(err_eh)*1000:+.2f} mrad  "
              f"RMS={np.sqrt(np.mean(err_eh**2))*1000:.2f} mrad  "
              f"P95={np.percentile(np.abs(err_eh), 95)*1000:.2f} mrad")
        print(f"  actual Δe_lat per frame:  mean={np.mean(del_el)*1000:+.2f} mm   "
              f"RMS={np.sqrt(np.mean(del_el**2))*1000:.2f} mm")
        print(f"  actual Δe_heading/frame:  mean={np.mean(del_eh)*1000:+.2f} mrad  "
              f"RMS={np.sqrt(np.mean(del_eh**2))*1000:.2f} mrad")

        # Decompose e_lat prediction into terms
        print(f"\n  e_lat prediction decomposition (what the model thinks drives lateral motion):")
        print(f"    heading term (vx*e_heading*dt):  mean={np.mean(d_head)*1000:+.3f} mm  "
              f"RMS={np.sqrt(np.mean(d_head**2))*1000:.3f} mm")
        print(f"    v_y term (v_y*dt):               mean={np.mean(d_vy)*1000:+.3f} mm  "
              f"RMS={np.sqrt(np.mean(d_vy**2))*1000:.3f} mm")
        total_pred = d_head + d_vy
        actual = del_el
        residual = actual - total_pred
        print(f"    TOTAL predicted Δe_lat:          mean={np.mean(total_pred)*1000:+.3f} mm")
        print(f"    ACTUAL Δe_lat:                   mean={np.mean(actual)*1000:+.3f} mm")
        print(f"    RESIDUAL (actual - predicted):   mean={np.mean(residual)*1000:+.3f} mm  "
              f"RMS={np.sqrt(np.mean(residual**2))*1000:.3f} mm")

        # Decompose e_heading prediction
        print(f"\n  e_heading prediction decomposition:")
        print(f"    yaw rate term (r*dt):            mean={np.mean(d_yaw)*1000:+.3f} mrad  "
              f"RMS={np.sqrt(np.mean(d_yaw**2))*1000:.3f} mrad")
        print(f"    curvature term (-kappa*vx*dt):    mean={np.mean(d_curv)*1000:+.3f} mrad  "
              f"RMS={np.sqrt(np.mean(d_curv**2))*1000:.3f} mrad")

        if len(pred_errors_vy) > 0:
            err_vy = np.array(pred_errors_vy)[mask[:len(pred_errors_vy)]] if len(pred_errors_vy) == len(mask) else None
            err_r = np.array(pred_errors_r)[mask[:len(pred_errors_r)]] if len(pred_errors_r) == len(mask) else None
            if err_vy is not None and len(err_vy) > 0:
                print(f"\n  v_y prediction error:     mean={np.mean(err_vy)*1000:+.2f} mm/s  "
                      f"RMS={np.sqrt(np.mean(err_vy**2))*1000:.2f} mm/s")
            if err_r is not None and len(err_r) > 0:
                print(f"  yaw_rate prediction err:  mean={np.mean(err_r)*1000:+.2f} mrad/s  "
                      f"RMS={np.sqrt(np.mean(err_r**2))*1000:.2f} mrad/s")

        # Bias analysis
        if n > 20:
            bias_ratio = abs(np.mean(err_el)) / max(np.sqrt(np.mean(err_el**2)), 1e-12)
            if bias_ratio > 0.5:
                sign = "positive" if np.mean(err_el) > 0 else "negative"
                print(f"\n  ** SYSTEMATIC BIAS DETECTED: {sign} "
                      f"(|mean|/RMS = {bias_ratio:.2f}, > 0.5)")
                print(f"     Model consistently {'over' if np.mean(err_el) > 0 else 'under'}-predicts e_lat")

    report_segment("ALL FRAMES", np.ones(len(pred_errors_elat), dtype=bool))
    report_segment("STRAIGHT (|kappa| < 0.005)", is_straight)
    report_segment("CURVE (|kappa| >= 0.005)", is_curve)

    # Worst frames analysis
    print("\n" + "=" * 72)
    print("WORST 10 PREDICTION ERRORS (by |e_lat error|)")
    print("=" * 72)
    worst_idx = np.argsort(np.abs(pred_errors_elat))[-10:][::-1]
    print(f"  {'Frame':>6}  {'|err| mm':>9}  {'e_lat mm':>9}  {'e_head mrad':>11}  "
          f"{'speed':>6}  {'|kappa|':>8}  {'segment':>8}")
    for wi in worst_idx:
        k = frame_indices[wi]
        seg = "CURVE" if is_curve[wi] else "STRAIGHT"
        print(f"  {k:>6}  {abs(pred_errors_elat[wi])*1000:>9.2f}  "
              f"{data['mpc_e_lat'][k]*1000:>9.2f}  "
              f"{data['mpc_e_heading'][k]*1000:>11.2f}  "
              f"{frame_speeds[wi]:>6.1f}  {frame_curvatures[wi]:>8.4f}  {seg:>8}")

    # Kappa analysis: compare MPC's kappa_ref vs ground truth curvature
    if data["gt_curvature"] is not None and data["mpc_kappa_ref"] is not None:
        print("\n" + "=" * 72)
        print("CURVATURE REFERENCE vs GROUND TRUTH")
        print("=" * 72)
        mpc_kappa = data["mpc_kappa_ref"][pair_idx]
        gt_kappa = data["gt_curvature"][pair_idx]
        kappa_err = mpc_kappa - gt_kappa

        print(f"  MPC kappa_ref:    mean={np.mean(np.abs(mpc_kappa)):.5f}  max={np.max(np.abs(mpc_kappa)):.5f}")
        print(f"  GT curvature:     mean={np.mean(np.abs(gt_kappa)):.5f}  max={np.max(np.abs(gt_kappa)):.5f}")
        print(f"  Kappa error:      mean={np.mean(kappa_err):+.6f}  RMS={np.sqrt(np.mean(kappa_err**2)):.6f}")
        print(f"  Kappa error abs:  P50={np.percentile(np.abs(kappa_err), 50):.6f}  "
              f"P95={np.percentile(np.abs(kappa_err), 95):.6f}")

        curve_mask_full = np.abs(gt_kappa) > CURVE_THRESHOLD
        if np.sum(curve_mask_full) > 0:
            kappa_err_curve = kappa_err[curve_mask_full]
            mpc_curve = mpc_kappa[curve_mask_full]
            gt_curve = gt_kappa[curve_mask_full]
            ratio = mpc_curve / np.where(np.abs(gt_curve) > 1e-6, gt_curve, 1e-6)
            print(f"\n  IN CURVES ONLY ({np.sum(curve_mask_full)} frames):")
            print(f"    MPC/GT ratio:   P50={np.percentile(ratio, 50):.3f}  "
                  f"mean={np.mean(ratio):.3f}  "
                  f"P5-P95=[{np.percentile(ratio, 5):.3f}, {np.percentile(ratio, 95):.3f}]")
            print(f"    Kappa error:    mean={np.mean(kappa_err_curve):+.6f}  "
                  f"RMS={np.sqrt(np.mean(kappa_err_curve**2)):.6f}")
            sign_agree = np.mean(np.sign(mpc_curve) == np.sign(gt_curve))
            print(f"    Sign agreement: {sign_agree*100:.1f}%")

    # Cross-track comparison: MPC e_lat vs GT cross-track
    if data["gt_cross_track"] is not None:
        print("\n" + "=" * 72)
        print("MPC e_lat vs GROUND TRUTH CROSS-TRACK")
        print("=" * 72)
        mpc_elat = data["mpc_e_lat"][pair_idx]
        gt_ct = data["gt_cross_track"][pair_idx]
        ct_diff = mpc_elat - gt_ct

        print(f"  MPC e_lat:        mean={np.mean(mpc_elat)*1000:+.2f} mm  "
              f"RMS={np.sqrt(np.mean(mpc_elat**2))*1000:.2f} mm")
        print(f"  GT cross-track:   mean={np.mean(gt_ct)*1000:+.2f} mm  "
              f"RMS={np.sqrt(np.mean(gt_ct**2))*1000:.2f} mm")
        print(f"  Difference:       mean={np.mean(ct_diff)*1000:+.2f} mm  "
              f"RMS={np.sqrt(np.mean(ct_diff**2))*1000:.2f} mm")

        # Correlation
        if np.std(mpc_elat) > 1e-6 and np.std(gt_ct) > 1e-6:
            corr = np.corrcoef(mpc_elat, gt_ct)[0, 1]
            print(f"  Correlation:      {corr:.4f}")
        else:
            print(f"  Correlation:      N/A (constant signal)")

    # Summary diagnosis
    print("\n" + "=" * 72)
    print("DIAGNOSIS SUMMARY")
    print("=" * 72)

    elat_rms_all = np.sqrt(np.mean(pred_errors_elat**2)) * 1000
    elat_rms_straight = np.sqrt(np.mean(pred_errors_elat[is_straight]**2)) * 1000 if np.any(is_straight) else 0
    elat_rms_curve = np.sqrt(np.mean(pred_errors_elat[is_curve]**2)) * 1000 if np.any(is_curve) else 0

    ehead_rms_all = np.sqrt(np.mean(pred_errors_ehead**2)) * 1000
    ehead_rms_straight = np.sqrt(np.mean(pred_errors_ehead[is_straight]**2)) * 1000 if np.any(is_straight) else 0
    ehead_rms_curve = np.sqrt(np.mean(pred_errors_ehead[is_curve]**2)) * 1000 if np.any(is_curve) else 0

    print(f"  1-step e_lat prediction RMS:     {elat_rms_all:.2f} mm  "
          f"(straight: {elat_rms_straight:.2f}, curve: {elat_rms_curve:.2f})")
    print(f"  1-step e_heading prediction RMS: {ehead_rms_all:.2f} mrad  "
          f"(straight: {ehead_rms_straight:.2f}, curve: {ehead_rms_curve:.2f})")

    # Identify dominant error source
    if elat_rms_curve > 2 * elat_rms_straight and np.any(is_curve):
        print(f"\n  >> Curve prediction error is {elat_rms_curve/max(elat_rms_straight,0.01):.1f}x "
              f"worse than straight — model-plant mismatch is curve-specific")

        curve_residual = pred_errors_elat[is_curve]
        curve_bias = np.mean(curve_residual)
        if abs(curve_bias) > 0.5 * np.sqrt(np.mean(curve_residual**2)):
            print(f"  >> Curve e_lat error has SYSTEMATIC BIAS: {curve_bias*1000:+.2f} mm/frame")
            print(f"     Over 20 frames (1 curve): cumulative bias ≈ {curve_bias*20*1000:+.1f} mm")
            if curve_bias > 0:
                print(f"     Model OVER-predicts e_lat → optimizer thinks car drifts more than it does")
                print(f"     → Possible cause: curvature feedforward too strong, or tire model too stiff")
            else:
                print(f"     Model UNDER-predicts e_lat → optimizer thinks car tracks better than it does")
                print(f"     → Possible cause: curvature feedforward too weak, or tire model too soft")

    if ehead_rms_curve > 2 * ehead_rms_straight and np.any(is_curve):
        curve_ehead_bias = np.mean(pred_errors_ehead[is_curve])
        print(f"\n  >> Curve heading error is {ehead_rms_curve/max(ehead_rms_straight,0.01):.1f}x "
              f"worse than straight")
        if abs(curve_ehead_bias) > 0.5 * np.sqrt(np.mean(pred_errors_ehead[is_curve]**2)):
            print(f"  >> Heading error has SYSTEMATIC BIAS: {curve_ehead_bias*1000:+.2f} mrad/frame")
            print(f"     → Likely cause: kappa_ref doesn't match actual road curvature")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPC prediction vs reality diagnostic")
    parser.add_argument("--latest", action="store_true", help="Use latest recording")
    parser.add_argument("--recording", type=str, help="Path to specific recording")
    args = parser.parse_args()

    if args.recording:
        path = args.recording
    elif args.latest:
        path = find_latest_recording()
    else:
        path = find_latest_recording()

    analyze(path)
