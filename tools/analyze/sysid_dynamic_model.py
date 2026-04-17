"""Offline system identification for Unity vehicle dynamic bicycle model.

Fits tire cornering stiffness (C_f, C_r) from HDF5 recording data by
least-squares regression on the yaw rate dynamics equation:

    Iz * dr/dt = l_f * C_f * alpha_f - l_r * C_r * alpha_r

Uses physical steering angle (steeringAngleActual) and IMU yaw rate
directly from Unity telemetry, bypassing any MPC gain assumptions.

Usage:
    python tools/analyze/sysid_dynamic_model.py --latest
    python tools/analyze/sysid_dynamic_model.py data/recordings/recording_XXX.h5
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np


def _find_latest_recording(recordings_dir: str = "data/recordings") -> str:
    files = sorted(glob.glob(os.path.join(recordings_dir, "*.h5")),
                   key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(f"No recordings found in {recordings_dir}")
    return files[-1]


def _load_geometry(f: h5py.File) -> dict:
    """Load vehicle geometry from Unity override fields or defaults."""
    geo = {}

    # Try Unity geometry override fields first
    for field, key, default in [
        ("control/mpc_unity_geometry_lf", "l_f", 1.125),
        ("control/mpc_unity_geometry_lr", "l_r", 1.125),
        ("control/mpc_unity_geometry_mass", "mass", 1500.0),
        ("control/mpc_unity_geometry_iz", "Iz", 1250.0),
    ]:
        if field in f:
            vals = f[field][:]
            nonzero = vals[vals > 0]
            geo[key] = float(np.median(nonzero)) if len(nonzero) > 0 else default
        else:
            geo[key] = default

    geo["L"] = geo["l_f"] + geo["l_r"]
    return geo


def run_sysid(recording_path: str, verbose: bool = True) -> dict:
    """Run system identification on a single recording.

    Returns dict with fitted C_f, C_r, K_us, and validation metrics.
    """
    f = h5py.File(recording_path, "r")

    # ── Load signals ──────────────────────────────────────────────────────
    speed = f["vehicle/speed"][:]
    n = len(speed)

    # IMU yaw rate: angular_velocity[:,1] (Unity Y axis), negated for control convention
    ang_vel = f["vehicle/angular_velocity"][:]
    yaw_rate = -ang_vel[:, 1].astype(float)

    # Physical steering angle — from Unity WheelCollider actual (degrees → radians)
    steer_actual = np.deg2rad(f["vehicle/steering_angle_actual"][:].astype(float))

    # Rear wheel sideways slip for v_y estimation
    wheel_slip = f["vehicle/wheel_sideways_slip"][:].astype(float)  # shape (N, 4)
    # Rear wheels are indices 2, 3 (RL, RR)
    rear_slip_avg = np.mean(wheel_slip[:, 2:4], axis=1)

    # Vehicle geometry
    geo = _load_geometry(f)
    l_f = geo["l_f"]
    l_r = geo["l_r"]
    L = geo["L"]
    mass = geo["mass"]
    Iz = geo["Iz"]

    # Load timestamps for accurate dt before closing file
    if "camera/timestamps" in f:
        timestamps = f["camera/timestamps"][:n].astype(float)
        dt_arr = np.diff(timestamps)
        dt_arr = np.clip(dt_arr, 0.01, 0.5)  # guard against bad timestamps
        dt_median = float(np.median(dt_arr))
    else:
        dt_median = 1.0 / 13.0  # fallback: ~13 FPS typical

    f.close()

    # ── Compute slip angles (kinematic approximation: v_y ≈ 0) ──────────
    # Full: alpha_f = delta - (v_y + l_f*r)/v_x,  alpha_r = -(v_y - l_r*r)/v_x
    # Kinematic (v_y≈0): alpha_f = delta - l_f*r/v_x,  alpha_r = l_r*r/v_x
    # This avoids unreliable v_y from wheel_sideways_slip.
    vx_safe = np.maximum(speed, 3.0)
    alpha_f = steer_actual - l_f * yaw_rate / vx_safe
    alpha_r = l_r * yaw_rate / vx_safe

    # ── Low-pass filter signals before differentiation ──────────────────
    # At ~13 FPS, finite differences amplify noise; filter at 2 Hz cutoff
    from scipy.signal import butter, filtfilt
    fs = 1.0 / dt_median
    cutoff_hz = min(2.0, fs / 4.0)  # Nyquist-safe
    b_filt, a_filt = butter(2, cutoff_hz / (fs / 2.0), btype='low')
    yaw_rate_filt = filtfilt(b_filt, a_filt, yaw_rate) if n > 12 else yaw_rate

    # ── Compute yaw rate derivative (central differences on filtered signal) ─
    dr_dt = np.zeros(n)
    if verbose:
        print(f"Frame rate: ~{1.0/dt_median:.1f} FPS (dt={dt_median*1000:.1f}ms)")
    if n > 1:
        dr_dt[1:-1] = (yaw_rate_filt[2:] - yaw_rate_filt[:-2]) / (2 * dt_median)
        dr_dt[0] = (yaw_rate_filt[1] - yaw_rate_filt[0]) / dt_median
        dr_dt[-1] = (yaw_rate_filt[-1] - yaw_rate_filt[-2]) / dt_median

    # ── Gate frames for valid identification ──────────────────────────────
    gate = (
        (speed >= 5.0)                    # sufficient speed
        & (np.abs(alpha_f) < 0.15)        # linear tire region
        & (np.abs(alpha_r) < 0.15)        # linear tire region
        & (np.abs(yaw_rate) > 0.005)      # sufficient excitation
        & (np.abs(steer_actual) > 0.005)  # actually steering
    )

    n_valid = np.sum(gate)
    if verbose:
        print(f"Recording: {recording_path}")
        print(f"Total frames: {n}")
        print(f"Valid frames for sysid: {n_valid} ({n_valid/n*100:.1f}%)")
        print(f"Geometry: l_f={l_f:.3f}m, l_r={l_r:.3f}m, L={L:.3f}m, "
              f"mass={mass:.0f}kg, Iz={Iz:.0f}kg·m²")

    if n_valid < 50:
        print("ERROR: Insufficient valid frames for identification (need ≥50)")
        return {"error": "insufficient_data", "n_valid": n_valid}

    # ── Least-squares regression (integrated form) ─────────────────────────
    # Instead of fitting noisy dr/dt, use integrated form over windows:
    #   Iz * (r[k+W] - r[k]) = dt * sum_{i=k..k+W-1} (l_f*C_f*α_f[i] - l_r*C_r*α_r[i])
    # This is linear in [C_f, C_r] and avoids differentiation entirely.
    from scipy.optimize import nnls

    W = 5  # integration window (5 frames ≈ 0.38s at 13 FPS)
    gate_idx = np.where(gate)[0]

    # Build integrated regression: for each window of W consecutive gated frames
    rows_A = []
    rows_b = []
    for i in range(len(gate_idx) - W):
        idx_start = gate_idx[i]
        idx_end = gate_idx[i + W]
        # Only use if indices are consecutive (no gaps in gating)
        if idx_end - idx_start != W:
            continue
        slc = slice(idx_start, idx_end)
        delta_r = yaw_rate_filt[idx_end] - yaw_rate_filt[idx_start]
        sum_af = np.sum(alpha_f[slc])
        sum_ar = np.sum(alpha_r[slc])
        rows_b.append(Iz * delta_r / dt_median)
        rows_A.append([l_f * sum_af, -l_r * sum_ar])

    A = np.array(rows_A)
    b = np.array(rows_b)
    n_windows = len(b)

    if verbose:
        print(f"Integration windows: {n_windows} (W={W} frames)")

    if n_windows < 20:
        print("ERROR: Insufficient windows for identification (need ≥20)")
        return {"error": "insufficient_windows", "n_windows": n_windows}

    # Solve with non-negative constraint (cornering stiffness must be positive)
    x, residual = nnls(A, b)
    C_f_fit, C_r_fit = x

    # Also try unconstrained for comparison
    x_unc, res_unc, _, _ = np.linalg.lstsq(A, b, rcond=None)
    C_f_unc, C_r_unc = x_unc

    # ── Validation: predicted vs measured integrated yaw rate ─────────────
    b_pred = A @ x
    rmse = np.sqrt(np.mean((b - b_pred) ** 2))
    ss_res = np.sum((b - b_pred) ** 2)
    ss_tot = np.sum((b - np.mean(b)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # ── Compute understeer gradient ───────────────────────────────────────
    if C_f_fit > 0 and C_r_fit > 0:
        K_us = (mass / L) * (l_r / C_f_fit - l_f / C_r_fit)
    else:
        K_us = float("nan")

    # ── Effective wheelbase at typical speeds ─────────────────────────────
    # L_eff(v) = L + K_us * v^2 (from steady-state bicycle model)
    L_eff_11 = L + K_us * 11.0 ** 2 if not np.isnan(K_us) else float("nan")

    # ── Report ────────────────────────────────────────────────────────────
    result = {
        "C_f": float(C_f_fit),
        "C_r": float(C_r_fit),
        "C_f_unconstrained": float(C_f_unc),
        "C_r_unconstrained": float(C_r_unc),
        "K_us": float(K_us),
        "L_eff_at_11mps": float(L_eff_11),
        "rmse_dr_dt": float(rmse),
        "r_squared": float(r_squared),
        "n_valid": int(n_valid),
        "geometry": geo,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"SYSTEM IDENTIFICATION RESULTS")
        print(f"{'='*60}")
        print(f"Tire Cornering Stiffness (NNLS):")
        print(f"  C_f = {C_f_fit:,.0f} N/rad  (default: 40,000)")
        print(f"  C_r = {C_r_fit:,.0f} N/rad  (default: 40,000)")
        print(f"\nTire Cornering Stiffness (unconstrained):")
        print(f"  C_f = {C_f_unc:,.0f} N/rad")
        print(f"  C_r = {C_r_unc:,.0f} N/rad")
        print(f"\nUndersteer Gradient:")
        print(f"  K_us = {K_us:.6f} rad/(m/s²)")
        print(f"  {'understeer' if K_us > 0 else 'OVERSTEER' if K_us < 0 else 'neutral'}")
        print(f"\nEffective Wheelbase:")
        print(f"  L_eff(0 m/s)  = {L:.3f} m  (geometric)")
        print(f"  L_eff(11 m/s) = {L_eff_11:.3f} m  (at typical curve speed)")
        print(f"\nFit Quality:")
        print(f"  dr/dt RMSE: {rmse:.4f} rad/s²")
        print(f"  R²: {r_squared:.4f}")
        print(f"  Valid frames: {n_valid}")

        # Speed-binned analysis
        print(f"\n{'='*60}")
        print(f"SPEED-BINNED ANALYSIS")
        print(f"{'='*60}")
        for s_lo, s_hi in [(5, 7), (7, 9), (9, 11), (11, 13)]:
            band = gate & (speed >= s_lo) & (speed < s_hi)
            band_idx = np.where(band)[0]
            nb = len(band_idx)
            if nb < 20:
                continue
            # Integrated form for speed band
            rows_Ab, rows_bb = [], []
            for ii in range(len(band_idx) - W):
                i_s = band_idx[ii]
                i_e = band_idx[ii + W]
                if i_e - i_s != W:
                    continue
                slc = slice(i_s, i_e)
                dr = yaw_rate_filt[i_e] - yaw_rate_filt[i_s]
                rows_bb.append(Iz * dr / dt_median)
                rows_Ab.append([l_f * np.sum(alpha_f[slc]), -l_r * np.sum(alpha_r[slc])])
            if len(rows_bb) < 10:
                print(f"  {s_lo}-{s_hi} m/s ({nb} frames): too few windows")
                continue
            try:
                x_b, _ = nnls(np.array(rows_Ab), np.array(rows_bb))
                print(f"  {s_lo}-{s_hi} m/s ({nb} frames, {len(rows_bb)} windows): "
                      f"C_f={x_b[0]:,.0f}, C_r={x_b[1]:,.0f} N/rad")
            except Exception:
                print(f"  {s_lo}-{s_hi} m/s ({nb} frames): fit failed")

        # Config recommendations
        print(f"\n{'='*60}")
        print(f"RECOMMENDED CONFIG")
        print(f"{'='*60}")
        print(f"# config/av_stack_config.yaml")
        print(f"mpc_tire_cf_nominal: {C_f_fit:.0f}")
        print(f"mpc_tire_cr_nominal: {C_r_fit:.0f}")
        print(f"# EKF bounds (±50% of fitted values)")
        print(f"mpc_tire_ekf_cf_min: {max(C_f_fit * 0.5, 5000):.0f}")
        print(f"mpc_tire_ekf_cf_max: {C_f_fit * 2.0:.0f}")
        print(f"mpc_tire_ekf_cr_min: {max(C_r_fit * 0.5, 5000):.0f}")
        print(f"mpc_tire_ekf_cr_max: {C_r_fit * 2.0:.0f}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Offline system identification for Unity vehicle dynamic model")
    parser.add_argument("recording", nargs="?", default=None,
                        help="Path to HDF5 recording (default: latest)")
    parser.add_argument("--latest", action="store_true",
                        help="Use most recent recording")
    parser.add_argument("--multi", nargs="*",
                        help="Run on multiple recordings for consistency check")
    args = parser.parse_args()

    if args.multi is not None:
        # Multi-recording consistency check
        recordings = args.multi if args.multi else []
        if not recordings:
            # Use last 3 recordings
            all_recs = sorted(glob.glob("data/recordings/*.h5"),
                              key=os.path.getmtime)
            recordings = all_recs[-3:] if len(all_recs) >= 3 else all_recs
        results = []
        for rec in recordings:
            print(f"\n{'#'*60}")
            r = run_sysid(rec, verbose=True)
            if "error" not in r:
                results.append(r)
        if len(results) >= 2:
            cfs = [r["C_f"] for r in results]
            crs = [r["C_r"] for r in results]
            print(f"\n{'='*60}")
            print(f"CONSISTENCY CHECK ({len(results)} recordings)")
            print(f"{'='*60}")
            print(f"  C_f: mean={np.mean(cfs):,.0f}, std={np.std(cfs):,.0f}, "
                  f"cv={np.std(cfs)/np.mean(cfs)*100:.1f}%")
            print(f"  C_r: mean={np.mean(crs):,.0f}, std={np.std(crs):,.0f}, "
                  f"cv={np.std(crs)/np.mean(crs)*100:.1f}%")
        return

    if args.recording:
        recording = args.recording
    else:
        recording = _find_latest_recording()

    run_sysid(recording, verbose=True)


if __name__ == "__main__":
    main()
