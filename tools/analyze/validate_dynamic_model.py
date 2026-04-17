"""Offline multi-step prediction validation for the dynamic bicycle model.

Replays recorded data through the 5-state dynamic bicycle A/B matrices
at multiple horizon lengths (1, 5, 10, 20 steps) and compares predicted
states against measured ground truth.  Reports WHERE the model breaks
down — which state diverges first and at which horizon step.

This is the standard industry validation step between system identification
and closed-loop deployment.  Do NOT enable `mpc_dynamic_model_enabled`
until this tool reports acceptable prediction accuracy.

Usage:
    python tools/analyze/validate_dynamic_model.py --latest
    python tools/analyze/validate_dynamic_model.py data/recordings/recording_XXX.h5
"""

import argparse
import glob
import math
import os
import sys

import h5py
import numpy as np
from scipy.linalg import expm


# ── Defaults (overridden by config/recording if available) ──────────────

DEFAULT_CF = 36662.0
DEFAULT_CR = 71799.0
DEFAULT_MASS = 1500.0
DEFAULT_IZ = 2428.0
DEFAULT_LF = 1.125
DEFAULT_LR = 1.125
DEFAULT_MAX_STEER_LOW = math.radians(30.0)  # 0.5236 rad
DEFAULT_MAX_STEER_HIGH = math.radians(16.0)  # 0.2793 rad
DEFAULT_MAX_STEER_SPEED_SAT = 12.0  # m/s


def _find_latest_recording(recordings_dir: str = "data/recordings") -> str:
    files = sorted(glob.glob(os.path.join(recordings_dir, "*.h5")),
                   key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(f"No recordings found in {recordings_dir}")
    return files[-1]


def _load_geometry(f: h5py.File) -> dict:
    """Load vehicle geometry from Unity override fields or defaults."""
    geo = {}
    for field, key, default in [
        ("control/mpc_unity_geometry_lf", "l_f", DEFAULT_LF),
        ("control/mpc_unity_geometry_lr", "l_r", DEFAULT_LR),
        ("control/mpc_unity_geometry_mass", "mass", DEFAULT_MASS),
        ("control/mpc_unity_geometry_iz", "Iz", DEFAULT_IZ),
    ]:
        if field in f:
            vals = f[field][:]
            nonzero = vals[vals > 0]
            geo[key] = float(np.median(nonzero)) if len(nonzero) > 0 else default
        else:
            geo[key] = default
    geo["L"] = geo["l_f"] + geo["l_r"]
    return geo


def _max_steer_at_speed(v: float) -> float:
    """Replicate the MPC's speed-dependent max steering angle."""
    t = min(max(v, 0.0) / DEFAULT_MAX_STEER_SPEED_SAT, 1.0)
    return DEFAULT_MAX_STEER_LOW * (1.0 - t) + DEFAULT_MAX_STEER_HIGH * t


def _propagate_dynamic(state: np.ndarray, u_steer_phys: float,
                        kappa: float, dt: float,
                        Cf: float, Cr: float, m: float, Iz: float,
                        lf: float, lr: float) -> np.ndarray:
    """One-step exact discretization (matrix exponential) of the 5-state
    dynamic bicycle model.

    State: [e_lat, e_heading, v_y, r, v_x]
    Input: physical steering angle (rad), curvature (1/m)

    The v_y/r 2×2 subblock uses expm for unconditional stability.
    Forward Euler oscillates at low speed (discrete eigenvalue < 0).
    """
    e_lat, e_head, vy, r, vx = state
    vx_safe = max(vx, 3.0)

    # Continuous-time 2×2 subblock for v_y/r dynamics
    a22 = -(Cf + Cr) / (m * vx_safe)
    a23 = -vx_safe - (Cf * lf - Cr * lr) / (m * vx_safe)
    a32 = -(Cf * lf - Cr * lr) / (Iz * vx_safe)
    a33 = -(Cf * lf**2 + Cr * lr**2) / (Iz * vx_safe)
    Ac = np.array([[a22, a23], [a32, a33]])

    # Exact discretization
    eM = expm(Ac * dt)

    # ZOH input: B_d = A_c^{-1}(A_d - I)·B_c
    Bc = np.array([Cf / m * u_steer_phys, Cf * lf / Iz * u_steer_phys])
    det_Ac = a22 * a33 - a23 * a32
    if abs(det_Ac) > 1e-10:
        Bd = np.linalg.solve(Ac, (eM - np.eye(2)) @ Bc)
    else:
        Bd = Bc * dt

    # Propagate v_y/r with exact discretization
    vyr_new = eM @ np.array([vy, r]) + Bd

    # Kinematic rows (exact for linear terms)
    e_lat_new = e_lat + vx_safe * e_head * dt + vy * dt
    e_head_new = e_head + r * dt - kappa * vx_safe * dt
    vx_new = vx  # no acceleration in this validation (use measured speed)

    return np.array([e_lat_new, e_head_new, vyr_new[0], vyr_new[1], vx_new])


def run_validation(recording_path: str, verbose: bool = True) -> dict:
    """Run multi-step prediction validation on a recording.

    Returns dict with per-horizon-step prediction errors for each state.
    """
    f = h5py.File(recording_path, "r")

    # ── Load signals ──────────────────────────────────────────────────
    speed = f["vehicle/speed"][:].astype(float)
    n = len(speed)

    # IMU yaw rate (Unity Y axis, negated for control convention)
    ang_vel = f["vehicle/angular_velocity"][:]
    yaw_rate_meas = -ang_vel[:, 1].astype(float)

    # Physical steering angle from Unity WheelCollider
    steer_actual_rad = np.deg2rad(f["vehicle/steering_angle_actual"][:].astype(float))

    # Lateral error and heading error (measured by ground truth reference)
    e_lat_meas = f["control/lateral_error"][:n].astype(float)
    e_head_meas = f["control/heading_error"][:n].astype(float)

    # Curvature
    if "control/mpc_kappa_ref" in f:
        kappa = f["control/mpc_kappa_ref"][:n].astype(float)
    elif "trajectory/reference_point_curvature" in f:
        kappa = f["trajectory/reference_point_curvature"][:n].astype(float)
    else:
        kappa = np.zeros(n)

    # Timestamps for dt
    if "camera/timestamps" in f:
        timestamps = f["camera/timestamps"][:n].astype(float)
        dt_arr = np.diff(timestamps)
        dt_arr = np.clip(dt_arr, 0.01, 0.5)
        dt_median = float(np.median(dt_arr))
    else:
        dt_median = 1.0 / 13.0

    # Vehicle geometry
    geo = _load_geometry(f)
    lf = geo["l_f"]
    lr = geo["l_r"]
    mass = geo["mass"]
    Iz = geo["Iz"]

    # Wheel sideways slip (for observer-based v_y init comparison)
    if "vehicle/wheel_sideways_slip" in f:
        _wheel_slip_raw = f["vehicle/wheel_sideways_slip"][:n].astype(float)
        _has_wheelslip = True
    else:
        _wheel_slip_raw = None
        _has_wheelslip = False

    f.close()

    Cf = DEFAULT_CF
    Cr = DEFAULT_CR

    if verbose:
        print(f"{'='*70}")
        print(f"DYNAMIC BICYCLE MODEL VALIDATION")
        print(f"{'='*70}")
        print(f"Recording: {os.path.basename(recording_path)}")
        print(f"Frames: {n}, dt={dt_median*1000:.1f}ms (~{1/dt_median:.0f} FPS)")
        print(f"Geometry: l_f={lf:.3f}m, l_r={lr:.3f}m, L={lf+lr:.3f}m")
        print(f"          mass={mass:.0f}kg, Iz={Iz:.0f}kg·m²")
        print(f"Tire:     C_f={Cf:.0f}, C_r={Cr:.0f} N/rad")
        print(f"          K_us={(mass*lr/((lf+lr)*Cf) - mass*lf/((lf+lr)*Cr)):.4f} rad/(m/s²)")
        cfl_coupled = (Cf + Cr) * dt_median / (2.0 * mass)
        print(f"CFL floor (coupled): {cfl_coupled:.2f} m/s  "
              f"(with 15% margin: {cfl_coupled*1.15:.2f} m/s)")
        print()

        # ── Oracle: eigenvalue stability check ────────────────────────
        print(f"{'─'*70}")
        print(f"ORACLE — EIGENVALUE STABILITY CHECK")
        print(f"{'─'*70}")
        print(f"{'Speed':>8}  {'λ_cont':>18}  {'Euler_disc':>18}  "
              f"{'Expm_disc':>18}  {'Status':>8}")
        oracle_pass = True
        for v_test in [3.0, 3.5, 4.0, 5.0, 7.0, 10.0]:
            _a22 = -(Cf + Cr) / (mass * v_test)
            _a23 = -v_test - (Cf * lf - Cr * lr) / (mass * v_test)
            _a32 = -(Cf * lf - Cr * lr) / (Iz * v_test)
            _a33 = -(Cf * lf**2 + Cr * lr**2) / (Iz * v_test)
            _Ac = np.array([[_a22, _a23], [_a32, _a33]])
            eigs_cont = np.linalg.eigvals(_Ac)
            eigs_euler = np.linalg.eigvals(np.eye(2) + _Ac * dt_median)
            _eM = expm(_Ac * dt_median)
            eigs_expm = np.linalg.eigvals(_eM)
            rho_euler = max(abs(eigs_euler))
            rho_expm = max(abs(eigs_expm))
            euler_ok = rho_euler < 1.0
            expm_ok = rho_expm < 1.0
            status = "OK" if expm_ok else "FAIL"
            if not expm_ok:
                oracle_pass = False
            # Show sign of real part for Euler (negative = oscillatory)
            euler_signs = [f"{e.real:+.3f}" for e in eigs_euler]
            expm_reals = [f"{e.real:+.3f}" for e in eigs_expm]
            print(f"{v_test:>6.1f}  "
                  f"  {eigs_cont[0].real:+.1f},{eigs_cont[1].real:+.1f}  "
                  f"  ρ={rho_euler:.3f} {'✓' if euler_ok else '✗'}      "
                  f"  ρ={rho_expm:.3f} {'✓' if expm_ok else '✗'}        "
                  f"  {status}")
        print()
        if oracle_pass:
            print("  [PASS] Eigenvalue stability: expm is unconditionally stable")
        else:
            print("  [FAIL] Eigenvalue stability: expm has ρ ≥ 1 — investigate!")
        print()

    # ── Estimate v_y: two methods for comparison ──────────────────────
    # Method A (baseline): v_y ≈ -l_r × r (steady-state kinematic approx)
    vy_est_kinematic = -lr * yaw_rate_meas

    # Method B (observer): v_y from rear wheel sideways slip measurement
    # v_y_rear = -vx × α_r_meas, then v_y_cg = v_y_rear + l_r × r
    if _has_wheelslip and _wheel_slip_raw is not None:
        rear_slip_avg = np.mean(_wheel_slip_raw[:, 2:4], axis=1)
        # Unity WheelCollider slip sign is opposite to SAE convention
        vy_est_wheelslip = speed * rear_slip_avg + lr * yaw_rate_meas
        has_wheelslip = True
    else:
        vy_est_wheelslip = vy_est_kinematic.copy()
        has_wheelslip = False

    # Default: use kinematic for the primary pass (backward compat)
    vy_est = vy_est_kinematic

    # ── Gate frames: sufficient speed and not at track boundaries ──────
    gate = (speed >= 4.0) & (np.arange(n) > 20) & (np.arange(n) < n - 25)

    # ── Multi-step prediction validation ──────────────────────────────
    horizons = [1, 3, 5, 10, 15, 20]
    max_H = max(horizons)

    # For each starting frame, propagate the model H steps and compare
    results = {h: {"e_lat_err": [], "e_head_err": [], "vy_err": [],
                    "r_err": [], "vx_err": []} for h in horizons}

    # Also track per-step cumulative errors for the full 20-step horizon
    step_errors = {s: {"e_lat": [], "e_head": [], "vy": [], "r": []}
                   for s in range(1, max_H + 1)}

    n_starts = 0
    stride = 5  # evaluate every 5th frame to reduce computation

    for i in range(0, n - max_H, stride):
        if not gate[i]:
            continue
        # Check all frames in the horizon are gated
        if not np.all(gate[i:i + max_H]):
            continue

        n_starts += 1

        # Initialize state from measurements at frame i
        state = np.array([
            e_lat_meas[i],
            e_head_meas[i],
            vy_est[i],
            yaw_rate_meas[i],
            speed[i],
        ])

        # Propagate step by step
        for step in range(1, max_H + 1):
            j = i + step
            # Use ACTUAL steering and curvature from the recording
            # (this is open-loop prediction — what would the model predict
            #  given the actual inputs the driver/controller applied?)
            u_steer = steer_actual_rad[j - 1]
            kk = kappa[j - 1]

            state = _propagate_dynamic(
                state, u_steer, kk, dt_median,
                Cf, Cr, mass, Iz, lf, lr
            )
            # Override v_x with measured (we're not validating longitudinal)
            state[4] = speed[j]

            # Record per-step errors
            step_errors[step]["e_lat"].append(state[0] - e_lat_meas[j])
            step_errors[step]["e_head"].append(state[1] - e_head_meas[j])
            step_errors[step]["vy"].append(state[2] - vy_est[j])
            step_errors[step]["r"].append(state[3] - yaw_rate_meas[j])

            # Record at checkpoint horizons
            if step in results:
                results[step]["e_lat_err"].append(state[0] - e_lat_meas[j])
                results[step]["e_head_err"].append(state[1] - e_head_meas[j])
                results[step]["vy_err"].append(state[2] - vy_est[j])
                results[step]["r_err"].append(state[3] - yaw_rate_meas[j])

    if verbose:
        print(f"Prediction windows evaluated: {n_starts} "
              f"(stride={stride}, gate: speed≥4 m/s)")
        print()

    if n_starts < 20:
        print("ERROR: Insufficient windows for validation (need ≥20)")
        return {"error": "insufficient_data", "n_starts": n_starts}

    # ── Report: per-horizon RMSE ──────────────────────────────────────
    if verbose:
        print(f"{'─'*70}")
        print(f"PREDICTION RMSE BY HORIZON STEP")
        print(f"{'─'*70}")
        print(f"{'Step':>5}  {'e_lat(m)':>10}  {'e_head(rad)':>12}  "
              f"{'v_y(m/s)':>10}  {'r(rad/s)':>10}")
        print(f"{'─'*5}  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*10}")

        # Reference: measured signal magnitudes
        e_lat_rms = np.sqrt(np.mean(e_lat_meas[gate]**2))
        e_head_rms = np.sqrt(np.mean(e_head_meas[gate]**2))
        vy_rms = np.sqrt(np.mean(vy_est[gate]**2))
        r_rms = np.sqrt(np.mean(yaw_rate_meas[gate]**2))

        print(f"{'ref':>5}  {e_lat_rms:>10.4f}  {e_head_rms:>12.4f}  "
              f"{vy_rms:>10.4f}  {r_rms:>10.4f}  ← signal RMS")
        print()

        for h in horizons:
            if len(results[h]["e_lat_err"]) == 0:
                continue
            rmse_elat = np.sqrt(np.mean(np.array(results[h]["e_lat_err"])**2))
            rmse_ehead = np.sqrt(np.mean(np.array(results[h]["e_head_err"])**2))
            rmse_vy = np.sqrt(np.mean(np.array(results[h]["vy_err"])**2))
            rmse_r = np.sqrt(np.mean(np.array(results[h]["r_err"])**2))

            # Flag if error exceeds signal magnitude
            flags = []
            if rmse_elat > e_lat_rms * 0.5:
                flags.append("e_lat")
            if rmse_ehead > e_head_rms * 0.5:
                flags.append("e_head")
            if rmse_vy > vy_rms * 0.5:
                flags.append("v_y")
            if rmse_r > r_rms * 0.5:
                flags.append("r")

            flag_str = f"  ⚠ {', '.join(flags)} > 50% signal" if flags else ""
            t_ms = h * dt_median * 1000
            print(f"{h:>5}  {rmse_elat:>10.4f}  {rmse_ehead:>12.4f}  "
                  f"{rmse_vy:>10.4f}  {rmse_r:>10.4f}  ({t_ms:.0f}ms){flag_str}")

        # ── Report: first divergent state ──────────────────────────────
        print()
        print(f"{'─'*70}")
        print(f"DIVERGENCE ANALYSIS — which state breaks first?")
        print(f"{'─'*70}")

        # Find the first step where each state's RMSE exceeds 50% of signal
        first_diverge = {}
        for state_name, sig_rms in [("e_lat", e_lat_rms), ("e_head", e_head_rms),
                                      ("vy", vy_rms), ("r", r_rms)]:
            for step in range(1, max_H + 1):
                errs = step_errors[step][state_name]
                if len(errs) == 0:
                    continue
                rmse = np.sqrt(np.mean(np.array(errs)**2))
                if rmse > sig_rms * 0.5:
                    first_diverge[state_name] = (step, rmse, sig_rms)
                    break
            else:
                first_diverge[state_name] = None

        for state_name, label in [("r", "Yaw rate (r)"),
                                   ("vy", "Lateral vel (v_y)"),
                                   ("e_head", "Heading error"),
                                   ("e_lat", "Lateral error")]:
            info = first_diverge.get(state_name)
            if info is None:
                print(f"  {label:20s}: OK — stays within 50% of signal through step {max_H}")
            else:
                step, rmse, sig = info
                t_ms = step * dt_median * 1000
                print(f"  {label:20s}: DIVERGES at step {step} ({t_ms:.0f}ms) — "
                      f"RMSE {rmse:.4f} > {sig*0.5:.4f} (50% of signal {sig:.4f})")

        # ── Report: bias analysis (systematic vs random error) ────────
        print()
        print(f"{'─'*70}")
        print(f"BIAS ANALYSIS — systematic vs random error at 10-step horizon")
        print(f"{'─'*70}")

        if len(results[10]["e_lat_err"]) > 0:
            for state_name, label in [("e_lat_err", "e_lat"),
                                       ("e_head_err", "e_heading"),
                                       ("vy_err", "v_y"),
                                       ("r_err", "yaw rate")]:
                errs = np.array(results[10][state_name])
                bias = np.mean(errs)
                std = np.std(errs)
                rmse = np.sqrt(np.mean(errs**2))
                bias_frac = abs(bias) / (rmse + 1e-12) * 100
                print(f"  {label:12s}: bias={bias:+.4f}  std={std:.4f}  "
                      f"RMSE={rmse:.4f}  bias%={bias_frac:.0f}%"
                      f"  {'← systematic' if bias_frac > 60 else '← random'}")

        # ── Report: speed-band breakdown ──────────────────────────────
        print()
        print(f"{'─'*70}")
        print(f"SPEED-BAND BREAKDOWN — 10-step prediction RMSE")
        print(f"{'─'*70}")
        print(f"{'Speed':>10}  {'n':>5}  {'e_lat':>8}  {'e_head':>8}  "
              f"{'v_y':>8}  {'r':>8}")

        # Re-run for speed bands
        speed_bands = [(3, 5), (5, 7), (7, 9), (9, 12)]
        H_check = 10
        for v_lo, v_hi in speed_bands:
            band_errs = {"e_lat": [], "e_head": [], "vy": [], "r": []}
            for i in range(0, n - H_check, stride):
                if not gate[i] or speed[i] < v_lo or speed[i] >= v_hi:
                    continue
                if not np.all(gate[i:i + H_check]):
                    continue

                state = np.array([
                    e_lat_meas[i], e_head_meas[i], vy_est[i],
                    yaw_rate_meas[i], speed[i]
                ])
                for step in range(1, H_check + 1):
                    j = i + step
                    state = _propagate_dynamic(
                        state, steer_actual_rad[j - 1], kappa[j - 1],
                        dt_median, Cf, Cr, mass, Iz, lf, lr
                    )
                    state[4] = speed[j]

                band_errs["e_lat"].append(state[0] - e_lat_meas[i + H_check])
                band_errs["e_head"].append(state[1] - e_head_meas[i + H_check])
                band_errs["vy"].append(state[2] - vy_est[i + H_check])
                band_errs["r"].append(state[3] - yaw_rate_meas[i + H_check])

            nb = len(band_errs["e_lat"])
            if nb > 5:
                rmse = {k: np.sqrt(np.mean(np.array(v)**2))
                        for k, v in band_errs.items()}
                print(f"{v_lo:>4}-{v_hi:<4}  {nb:>5}  {rmse['e_lat']:>8.4f}  "
                      f"{rmse['e_head']:>8.4f}  {rmse['vy']:>8.4f}  "
                      f"{rmse['r']:>8.4f}")
            else:
                print(f"{v_lo:>4}-{v_hi:<4}  {nb:>5}  (insufficient data)")

        # ── Report: curvature-band breakdown ──────────────────────────
        print()
        print(f"{'─'*70}")
        print(f"CURVATURE-BAND BREAKDOWN — 10-step prediction RMSE")
        print(f"{'─'*70}")
        print(f"{'|κ|':>12}  {'n':>5}  {'e_lat':>8}  {'e_head':>8}  "
              f"{'v_y':>8}  {'r':>8}")

        kappa_bands = [(0, 0.005), (0.005, 0.015), (0.015, 0.030), (0.030, 0.100)]
        for k_lo, k_hi in kappa_bands:
            band_errs = {"e_lat": [], "e_head": [], "vy": [], "r": []}
            for i in range(0, n - H_check, stride):
                if not gate[i]:
                    continue
                if abs(kappa[i]) < k_lo or abs(kappa[i]) >= k_hi:
                    continue
                if not np.all(gate[i:i + H_check]):
                    continue

                state = np.array([
                    e_lat_meas[i], e_head_meas[i], vy_est[i],
                    yaw_rate_meas[i], speed[i]
                ])
                for step in range(1, H_check + 1):
                    j = i + step
                    state = _propagate_dynamic(
                        state, steer_actual_rad[j - 1], kappa[j - 1],
                        dt_median, Cf, Cr, mass, Iz, lf, lr
                    )
                    state[4] = speed[j]

                band_errs["e_lat"].append(state[0] - e_lat_meas[i + H_check])
                band_errs["e_head"].append(state[1] - e_head_meas[i + H_check])
                band_errs["vy"].append(state[2] - vy_est[i + H_check])
                band_errs["r"].append(state[3] - yaw_rate_meas[i + H_check])

            nb = len(band_errs["e_lat"])
            label = f"{k_lo:.3f}-{k_hi:.3f}"
            if nb > 5:
                rmse = {k: np.sqrt(np.mean(np.array(v)**2))
                        for k, v in band_errs.items()}
                print(f"{label:>12}  {nb:>5}  {rmse['e_lat']:>8.4f}  "
                      f"{rmse['e_head']:>8.4f}  {rmse['vy']:>8.4f}  "
                      f"{rmse['r']:>8.4f}")
            else:
                print(f"{label:>12}  {nb:>5}  (insufficient data)")

        # ── Report: wheel-slip v_y init comparison ────────────────────
        if has_wheelslip:
            print()
            print(f"{'─'*70}")
            print(f"v_y INIT COMPARISON — kinematic (-l_r×r) vs wheel-slip measurement")
            print(f"{'─'*70}")

            # Re-run 10-step prediction with wheel-slip v_y initialization
            ws_errs = {"e_lat": [], "e_head": [], "vy": [], "r": []}
            kin_errs = {"e_lat": [], "e_head": [], "vy": [], "r": []}
            for i in range(0, n - H_check, stride):
                if not gate[i] or not np.all(gate[i:i + H_check]):
                    continue

                # Wheel-slip init
                state_ws = np.array([
                    e_lat_meas[i], e_head_meas[i], vy_est_wheelslip[i],
                    yaw_rate_meas[i], speed[i]
                ])
                # Kinematic init
                state_kin = np.array([
                    e_lat_meas[i], e_head_meas[i], vy_est_kinematic[i],
                    yaw_rate_meas[i], speed[i]
                ])
                for step in range(1, H_check + 1):
                    j = i + step
                    state_ws = _propagate_dynamic(
                        state_ws, steer_actual_rad[j - 1], kappa[j - 1],
                        dt_median, Cf, Cr, mass, Iz, lf, lr
                    )
                    state_ws[4] = speed[j]
                    state_kin = _propagate_dynamic(
                        state_kin, steer_actual_rad[j - 1], kappa[j - 1],
                        dt_median, Cf, Cr, mass, Iz, lf, lr
                    )
                    state_kin[4] = speed[j]

                ws_errs["e_lat"].append(state_ws[0] - e_lat_meas[i + H_check])
                ws_errs["e_head"].append(state_ws[1] - e_head_meas[i + H_check])
                ws_errs["vy"].append(state_ws[2] - vy_est_wheelslip[i + H_check])
                ws_errs["r"].append(state_ws[3] - yaw_rate_meas[i + H_check])
                kin_errs["e_lat"].append(state_kin[0] - e_lat_meas[i + H_check])
                kin_errs["e_head"].append(state_kin[1] - e_head_meas[i + H_check])
                kin_errs["vy"].append(state_kin[2] - vy_est_kinematic[i + H_check])
                kin_errs["r"].append(state_kin[3] - yaw_rate_meas[i + H_check])

            if len(ws_errs["e_lat"]) > 5:
                print(f"{'State':>12}  {'Kinematic':>10}  {'WheelSlip':>10}  {'Improvement':>12}")
                for key, label in [("e_lat", "e_lat(m)"), ("e_head", "e_head(rad)"),
                                   ("vy", "v_y(m/s)"), ("r", "r(rad/s)")]:
                    rmse_kin = np.sqrt(np.mean(np.array(kin_errs[key])**2))
                    rmse_ws = np.sqrt(np.mean(np.array(ws_errs[key])**2))
                    improve = (rmse_kin - rmse_ws) / rmse_kin * 100 if rmse_kin > 1e-9 else 0
                    arrow = "↑" if improve > 0 else "↓"
                    print(f"{label:>12}  {rmse_kin:>10.4f}  {rmse_ws:>10.4f}  "
                          f"{arrow} {abs(improve):>5.1f}%")

                # v_y correlation with ground truth reference
                corr_kin = np.corrcoef(vy_est_kinematic[gate], vy_est_wheelslip[gate])[0, 1]
                vy_kin_rms = np.sqrt(np.mean(vy_est_kinematic[gate]**2))
                vy_ws_rms = np.sqrt(np.mean(vy_est_wheelslip[gate]**2))
                print(f"\n  v_y signal RMS:  kinematic={vy_kin_rms:.4f}  "
                      f"wheelslip={vy_ws_rms:.4f}")
                print(f"  Correlation(kinematic, wheelslip): {corr_kin:.3f}")

        # ── Oracle: sign convention correlation check ─────────────────
        print()
        print(f"{'─'*70}")
        print(f"ORACLE — SIGN CONVENTION & PREDICTION CORRELATION")
        print(f"{'─'*70}")

        # Run 1-step predictions and check correlation with ground truth
        vy_preds_1 = []
        r_preds_1 = []
        vy_gts_1 = []
        r_gts_1 = []
        for i in range(0, n - 1, stride):
            if not gate[i]:
                continue
            state_1 = np.array([
                e_lat_meas[i], e_head_meas[i], vy_est[i],
                yaw_rate_meas[i], speed[i]
            ])
            state_1 = _propagate_dynamic(
                state_1, steer_actual_rad[i], kappa[i],
                dt_median, Cf, Cr, mass, Iz, lf, lr
            )
            state_1[4] = speed[i + 1]
            vy_preds_1.append(state_1[2])
            r_preds_1.append(state_1[3])
            vy_gts_1.append(vy_est[i + 1])
            r_gts_1.append(yaw_rate_meas[i + 1])

        oracle_corr_pass = True
        if len(vy_preds_1) > 20:
            corr_vy = np.corrcoef(vy_preds_1, vy_gts_1)[0, 1]
            corr_r = np.corrcoef(r_preds_1, r_gts_1)[0, 1]
            vy_1step_rmse = np.sqrt(np.mean((np.array(vy_preds_1) - np.array(vy_gts_1))**2))
            r_1step_rmse = np.sqrt(np.mean((np.array(r_preds_1) - np.array(r_gts_1))**2))

            vy_status = "PASS" if corr_vy > 0.5 else "FAIL"
            r_status = "PASS" if corr_r > 0.5 else "FAIL"
            if corr_vy <= 0.5 or corr_r <= 0.5:
                oracle_corr_pass = False

            print(f"  v_y 1-step: corr={corr_vy:.3f} RMSE={vy_1step_rmse:.4f} m/s  "
                  f"[{vy_status}] (gate: corr>0.5)")
            print(f"  r   1-step: corr={corr_r:.3f} RMSE={r_1step_rmse:.4f} rad/s  "
                  f"[{r_status}] (gate: corr>0.5)")
        else:
            print("  (insufficient data for correlation check)")

        # ── Verdict ───────────────────────────────────────────────────
        print()
        print(f"{'='*70}")

        # Check 10-step horizon (typical MPC prediction depth)
        if len(results[10]["r_err"]) > 0:
            r10_rmse = np.sqrt(np.mean(np.array(results[10]["r_err"])**2))
            elat10_rmse = np.sqrt(np.mean(np.array(results[10]["e_lat_err"])**2))

            if r10_rmse < r_rms * 0.3 and elat10_rmse < e_lat_rms * 0.3:
                print("VERDICT: PASS — model predictions accurate through 10-step horizon")
                print("         Safe to enable mpc_dynamic_model_enabled=true")
            elif r10_rmse < r_rms * 0.5 and elat10_rmse < e_lat_rms * 0.5:
                print("VERDICT: MARGINAL — model usable but prediction drift is notable")
                print("         Consider enabling with extra q_vy/q_yawrate regularization")
            else:
                print("VERDICT: FAIL — model predictions diverge before 10-step horizon")
                print("         Do NOT enable dynamic model until fixes applied:")
                if first_diverge.get("r") and first_diverge["r"][0] <= 5:
                    print("         → Yaw rate diverges early — check tire params or steering gain")
                if first_diverge.get("vy") and first_diverge["vy"][0] <= 5:
                    print("         → v_y diverges early — add v_y observation/correction")
                if first_diverge.get("e_head") and first_diverge["e_head"][0] <= 5:
                    print("         → Heading diverges early — check curvature input")
                if first_diverge.get("e_lat") and first_diverge["e_lat"][0] <= 10:
                    print("         → Lateral error drifts — cascading from v_y/heading errors")

        print(f"{'='*70}")

    return {
        "n_starts": n_starts,
        "horizons": {h: {k: np.sqrt(np.mean(np.array(v)**2))
                         for k, v in results[h].items() if len(v) > 0}
                     for h in horizons},
        "first_diverge": first_diverge if verbose else {},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate dynamic bicycle model against recorded data")
    parser.add_argument("recording", nargs="?", default=None,
                        help="Path to .h5 recording (default: latest)")
    parser.add_argument("--latest", action="store_true",
                        help="Use latest recording")
    args = parser.parse_args()

    if args.recording:
        path = args.recording
    else:
        path = _find_latest_recording()

    run_validation(path)


if __name__ == "__main__":
    main()
