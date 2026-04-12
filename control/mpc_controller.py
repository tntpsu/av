"""
Linear MPC (Model Predictive Control) lateral controller.

Uses a kinematic bicycle model in error form with OSQP as the QP solver.
Designed for mid-speed (10–20 m/s) lane-keeping where Pure Pursuit's v²
gain causes underdamped oscillation.

State:  x = [e_lat, e_heading, v]
Input:  u = [δ_norm, a]

Dynamics (discrete, Frenet error model, small-angle linearization):
  e_lat[k+1]     = e_lat[k] + v[k]·e_heading[k]·dt
  e_heading[k+1] = e_heading[k] + (v[k]/L)·δ_norm[k]·δ_max·dt − κ_ref[k]·v[k]·dt
  v[k+1]         = v[k] + a[k]·dt
"""

import logging
import math
import time
from dataclasses import dataclass, fields, replace as _dc_replace
from typing import Dict, Optional

import numpy as np
import osqp
from scipy import sparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MPCParams — tuning parameters from YAML
# ---------------------------------------------------------------------------

@dataclass
class MPCParams:
    """MPC tuning parameters. Loaded from config trajectory.mpc section."""
    horizon: int = 20
    dt: float = 0.033
    wheelbase_m: float = 2.5
    max_steer_rad: float = 0.5236           # 30° (fallback when speed_dependent_steer disabled)

    # Speed-dependent max steering angle (matches Unity CarController.cs lines 1398-1403)
    # Unity lerps max steer from low_speed_rad at 0 m/s to high_speed_rad at saturation_mps.
    speed_dependent_steer_enabled: bool = False
    max_steer_low_speed_rad: float = 0.5236     # 30° (Unity maxSteerAngleLowSpeed)
    max_steer_high_speed_rad: float = 0.2793    # 16° (Unity maxSteerAngleHighSpeed)
    max_steer_speed_saturation_mps: float = 12.0  # Unity maxSteerAngleFullSpeed

    feedforward_wheelbase_m: float = 0.0    # 0 = use wheelbase_m; >0 = separate FF wheelbase (decouples FF from dynamics)

    # State cost weights
    q_lat: float = 10.0                     # lateral tracking
    q_heading: float = 5.0                  # heading tracking
    q_speed: float = 1.0                    # speed tracking

    # Terminal cost multipliers
    q_lat_terminal_scale: float = 3.0
    q_heading_terminal_scale: float = 3.0

    # Lateral velocity damping: penalises (e_lat[k+1] - e_lat[k])² in the QP cost.
    # Since ė_lat = v·e_heading, this acts as speed-scaled heading damping.
    # Provides inherent oscillation suppression without per-track tuning.
    q_lat_rate: float = 2.0                # lateral velocity damping — OSCILLATION DAMPING KNOB

    # Input cost weights
    # r_steer is kept near-zero intentionally. A non-trivial value biases the QP toward
    # δ=0 on every step, causing the solver to systematically understeer on curves.
    # Steering direction is governed by q_lat (lateral error) and r_steer_rate (rate
    # limiting). 1e-4 retains a tiny diagonal entry for QP numerical stability.
    r_steer: float = 1e-4                   # steering magnitude (near-zero — see above)
    r_accel: float = 0.05                   # accel magnitude
    r_steer_rate: float = 1.0              # steering rate — PRIMARY COMFORT KNOB

    # Speed-dependent r_steer_rate scheduling (gain-scheduled MPC)
    # At higher speeds, bicycle model gain v/L increases → loop gain crosses stability
    # boundary → lateral oscillation. Scheduling r_steer_rate with speed damps this.
    # Uses solver re-init on speed band crossings (OSQP P matrix is immutable).
    r_steer_rate_scheduling_enabled: bool = False
    r_steer_rate_speed_onset: float = 12.0       # m/s where scaling begins
    r_steer_rate_speed_gain: float = 0.15        # multiplier per m/s above onset
    r_steer_rate_max_scale: float = 3.0          # cap: r_eff ≤ r_base × max_scale
    r_steer_rate_band_hysteresis: float = 1.0    # m/s hysteresis to prevent chattering
    r_steer_rate_kappa_off: float = 0.005        # curvature where speed scheduling deactivates (legacy)
    r_steer_rate_kappa_gain: float = 0.0         # r_steer_rate increase per unit κ (proportional curve damping)
    r_steer_rate_kappa_saturate: float = 0.015   # κ ceiling for curvature scaling (R67)

    # Curvature-scheduled q_lat: increase lateral tracking weight on curves
    # so MPC pre-steers into curves instead of accepting entry error.
    q_lat_kappa_gain: float = 0.0               # q_lat increase per unit κ
    q_lat_kappa_saturate: float = 0.015          # κ ceiling for q_lat scaling

    # Curvature-adaptive r_steer_rate reduction (straight-line oscillation fix)
    # On low-κ sections, MPC needs responsive corrections to prevent drift accumulation.
    # r_eff *= max(straight_min_scale, κ / straight_kappa_on) when κ < straight_kappa_on.
    # Active independently of speed scheduling. Does NOT require rebuild (uses same band).
    r_steer_rate_straight_damping_enabled: bool = True
    r_steer_rate_straight_kappa_on: float = 0.002   # curvature below which reduction activates
    r_steer_rate_straight_min_scale: float = 0.3     # minimum: r_eff ≥ 30% of base (floor)

    # Constraints
    delta_rate_max: float = 0.5             # max Δδ_norm per step
    max_accel: float = 1.2                  # m/s²
    max_decel: float = 2.4                  # m/s²
    v_min: float = 1.0                      # m/s
    v_max: float = 15.0                     # m/s

    # Arc-entry kappa-transition warmup
    # When kappa_ref jumps by > threshold in one frame (straight→curve boundary),
    # reset the warmup counter so the rate constraint is relaxed for that burst.
    # This lets the feedforward step apply immediately instead of ramping over 6 frames.
    kappa_transition_warmup_enabled: bool = True
    kappa_transition_threshold: float = 0.005  # ~half of R100 kappa (0.010)

    # Speed-adaptive horizon
    speed_adaptive_horizon: bool = True
    speed_adaptive_threshold_mps: float = 15.0
    speed_adaptive_n_high: int = 30

    # Solve budget
    max_solve_time_ms: float = 8.0
    max_consecutive_failures: int = 3

    # Startup warmup: relax constraints for first N frames after activation
    warmup_frames: int = 5
    warmup_max_iter: int = 500
    warmup_rate_relax: float = 1.0    # delta_rate_max during warmup (full range)

    # Lateral bias estimator (integral-like correction for steady-state curve offset)
    bias_enabled: bool = True
    bias_alpha: float = 0.005           # EMA smoothing (τ ≈ 1/α ≈ 200 frames ≈ 6.7s)
    bias_kappa_gain: float = 0.015      # curvature correction per meter of bias
    bias_max_correction: float = 0.002  # max curvature adjustment (rad/m) — never exceed 4x road κ
    bias_min_speed: float = 5.0         # only active above this speed (m/s)
    bias_relative_guard_enabled: bool = True
    bias_relative_guard_ratio: float = 0.50
    bias_relative_guard_min_kappa: float = 0.0005
    bias_ema_abs_max: float = 0.5     # absolute clamp on _bias_ema accumulator (meters)
    bias_active_curve_preserve_enabled: bool = True
    bias_active_curve_preserve_ratio: float = 0.75
    bias_active_curve_gate_min: float = 0.40
    bias_active_curve_preserve_dynamic_enabled: bool = True
    bias_active_curve_preserve_speed_on: float = 13.0
    bias_active_curve_preserve_speed_full: float = 15.0
    bias_active_curve_preserve_curvature_on: float = 0.0015
    bias_active_curve_preserve_curvature_full: float = 0.0020
    bias_active_curve_preserve_full_ratio: float = 1.0
    bias_active_curve_same_sign_guard_ratio: float = 0.25
    active_mild_curve_authority_enabled: bool = True
    active_mild_curve_authority_gate_min: float = 0.35
    active_mild_curve_authority_speed_on: float = 11.5
    active_mild_curve_authority_speed_full: float = 13.0
    active_mild_curve_authority_curvature_on: float = 0.0015
    active_mild_curve_authority_curvature_full: float = 0.0020
    active_mild_curve_authority_ratio: float = 0.88
    active_mild_curve_authority_full_ratio: float = 1.0

    # Online effective wheelbase estimation (RLS on heading prediction error)
    leff_estimation_enabled: bool = False       # off by default — safe for all configs
    leff_forgetting_factor: float = 0.995       # λ — RLS memory (~200-frame window)
    leff_initial_P: float = 100.0               # initial RLS covariance (high = fast initial learning)
    leff_min_speed_mps: float = 3.0             # only update when v > this (low-speed kinematics unreliable)
    leff_min_excitation: float = 0.01            # min |regressor| to update (avoid division noise)
    leff_bounds_min: float = 1.5                # hard clamp — never go below 1.5m
    leff_bounds_max: float = 8.0                # hard clamp — never go above 8.0m

    # Dynamic bicycle model (disabled by default — kinematic model remains active)
    dynamic_model_enabled: bool = False
    vehicle_mass_kg: float = 1500.0             # kg — Unity Rigidbody mass
    vehicle_iz_kgm2: float = 1250.0             # kg·m² — yaw inertia (from collider geometry)
    vehicle_lf_m: float = 1.125                 # m — CoG to front axle
    vehicle_lr_m: float = 1.125                 # m — CoG to rear axle
    tire_cf_nominal: float = 40000.0            # N/rad — front cornering stiffness initial
    tire_cr_nominal: float = 40000.0            # N/rad — rear cornering stiffness initial
    q_vy: float = 0.5                           # lateral velocity state cost
    q_yawrate: float = 1.0                      # yaw rate state cost

    # EKF tire cornering stiffness estimation
    tire_ekf_enabled: bool = False
    tire_ekf_process_noise: float = 100.0       # Q diagonal — C_f/C_r drift per step
    tire_ekf_measurement_noise: float = 0.01    # R — yaw rate measurement noise (rad/s)²
    tire_ekf_min_speed_mps: float = 5.0         # no update below this (v_y/v_x singularity)
    tire_ekf_cf_min: float = 15000.0            # N/rad — hard lower clamp
    tire_ekf_cf_max: float = 80000.0            # N/rad — hard upper clamp
    tire_ekf_cr_min: float = 15000.0            # N/rad — hard lower clamp
    tire_ekf_cr_max: float = 80000.0            # N/rad — hard upper clamp
    tire_ekf_innovation_divergence_threshold: float = 0.15  # rad/s
    tire_ekf_slip_saturation_rad: float = 0.15  # pause EKF when |α| exceeds this
    tire_ekf_use_imu_yaw_rate: bool = True       # Use direct IMU gyro (industry standard)
    tire_ekf_imu_yaw_rate_r: float = 0.001       # (rad/s)² — measurement noise for IMU gyro

    # Curvature preview horizon (feedforward from trajectory planner)
    curvature_preview_enabled: bool = False   # Default OFF — safe for existing configs
    curvature_preview_gain: float = 1.0       # Scale factor for preview curvature

    # Grade compensation (Step 3)
    grade_clamp_rad: float = 0.15             # Max grade input (~15%, 8.5°)
    grade_ff_gain: float = 1.0                # Feedforward gain (1.8 for Unity physics)

    # 2DOF feedforward alignment: penalize Δ(δ − δ_ff)² instead of Δδ²
    # This eliminates the rate-penalty bias against holding a sustained curve steer.
    # On straight roads (κ=0) the correction is exactly zero — no behavior change.
    ff_alignment_enabled: bool = True

    # First-step quadratic rate cost: adds r_steer_rate*(delta[0] - last_delta_norm)^2
    # to the QP.  The within-horizon rate loop only penalises pairs (k, k+1) for
    # k=0..N-2; the jump from the *previously applied* steering to delta[0] is only
    # enforced as a hard box constraint [last±delta_rate_max], giving zero quadratic
    # cost.  This means r_steer_rate has no effect on first-step sign flips, which
    # is the structural root of noise-driven steering jerk.
    # Enabling this makes r_steer_rate symmetric: the same cost applies to each step.
    first_step_rate_enabled: bool = True

    # Lateral-error EMA pre-filter: smooth the e_lat signal before passing to the QP.
    # Perception noise (σ≈0.05 m) creates alternating-sign e_lat measurements that
    # force the MPC to flip steering every frame — the root of steering jerk.
    # An EMA with alpha < 1 attenuates high-frequency noise; the deadband suppresses
    # the residual oscillation when the filtered error is small.
    # alpha=1.0 and deadband=0.0 reproduce the unfiltered behaviour exactly.
    elat_ema_alpha: float = 1.0      # blend ratio new-to-old (1.0 = no smoothing)
    elat_deadband_m: float = 0.0     # suppress steering when |filtered e_lat| < this (m)

    # Dropout-recovery rate limiter: caps the per-frame step in the e_lat signal
    # fed to the QP.  Specifically defends against "silent dropouts" — periods where
    # perception returns e_lat≈0 (no stale flag) while the car drifts off-centre,
    # followed by a sudden large step when perception recovers (e.g. 0 → 0.25 m in
    # one frame → 131 deg/s steering jerk).  0.0 = disabled (no clamping).
    elat_ramp_rate_m_per_frame: float = 0.0

    @classmethod
    def from_config(cls, cfg: dict) -> "MPCParams":
        """Load from nested config dict: cfg['trajectory']['mpc']['mpc_<field>']."""
        mpc_cfg = cfg.get("trajectory", {}).get("mpc", {})
        kwargs = {}
        for f in fields(cls):
            yaml_key = f"mpc_{f.name}"
            if yaml_key in mpc_cfg:
                kwargs[f.name] = type(f.default)(mpc_cfg[yaml_key])
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# MPCSolver — core QP builder and frame-by-frame solver
# ---------------------------------------------------------------------------

class MPCSolver:
    """
    OSQP-based QP solver for the kinematic bicycle MPC.

    Decision variable layout:
        z = [x₀, u₀, x₁, u₁, ..., x_{N-1}, u_{N-1}, x_N]
        dims: (nx+nu)*N + nx = 5N + 3

    QP form:
        min  0.5·z'·P·z + q'·z
        s.t. l ≤ A·z ≤ u
    """

    def __init__(self, params: MPCParams):
        self.p = params
        self._N = params.horizon
        if params.dynamic_model_enabled:
            self._nx = 5   # [e_lat, e_heading, v_y, r, v_x]
        else:
            self._nx = 3   # [e_lat, e_heading, v]
        self._nu = 2   # [δ_norm, a]
        self._prob: Optional[osqp.OSQP] = None
        self._warm_x = None
        self._warm_y = None
        self._nz = 0
        self._nc = 0
        self._current_r_steer_rate = params.r_steer_rate
        self._current_q_lat = params.q_lat
        self._speed_band_center: float = 0.0
        self._kappa_band_center: float = 0.0
        # Feedforward wheelbase: decouples FF (curve-entry ramp) from dynamics model
        self._ff_wheelbase_m = params.feedforward_wheelbase_m if params.feedforward_wheelbase_m > 0 else params.wheelbase_m
        self._build_qp()

    def _max_steer_at_speed(self, speed: float) -> float:
        """Return max steering angle (rad) at the given speed.

        Matches Unity CarController.cs Lerp(lowSpeed, highSpeed, InverseLerp(0, fullSpeed, speed)).
        """
        if not self.p.speed_dependent_steer_enabled:
            return self.p.max_steer_rad
        t = min(max(speed, 0.0) / self.p.max_steer_speed_saturation_mps, 1.0)
        return self.p.max_steer_low_speed_rad * (1.0 - t) + self.p.max_steer_high_speed_rad * t

    def _get_horizon(self, speed: float) -> int:
        if self.p.speed_adaptive_horizon and speed >= self.p.speed_adaptive_threshold_mps:
            return self.p.speed_adaptive_n_high
        return self.p.horizon

    def _get_effective_r_steer_rate(self, speed: float, kappa: float = 0.0) -> float:
        """Compute speed-scheduled r_steer_rate with curvature gating.

        At speeds above onset, r_steer_rate scales linearly to counteract
        the bicycle model's v/L gain increase. On curves (high kappa),
        scheduling deactivates so MPC retains full tracking authority.
        """
        r_base = self.p.r_steer_rate

        # Curvature-adaptive reduction: lower r_steer_rate on straights where
        # MPC needs faster corrections to prevent oscillation buildup.
        if self.p.r_steer_rate_straight_damping_enabled:
            kappa_abs = abs(kappa)
            kappa_on = max(1e-6, self.p.r_steer_rate_straight_kappa_on)
            straight_scale = max(
                self.p.r_steer_rate_straight_min_scale,
                min(1.0, kappa_abs / kappa_on),
            )
            r_base = r_base * straight_scale

        if not self.p.r_steer_rate_scheduling_enabled:
            return r_base

        excess = max(0.0, speed - self.p.r_steer_rate_speed_onset)
        scale = min(1.0 + self.p.r_steer_rate_speed_gain * excess,
                    self.p.r_steer_rate_max_scale)
        # Speed scheduling with curvature gate (existing)
        kappa_abs_v = abs(kappa)
        kappa_off = max(1e-6, self.p.r_steer_rate_kappa_off)
        kappa_gate = max(0.0, min(1.0, 1.0 - kappa_abs_v / kappa_off))
        speed_scale = 1.0 + (scale - 1.0) * kappa_gate

        # Curvature damping: proportional floor on curves
        # Higher κ → higher r_steer_rate → less jerk on tight curves
        kappa_curve_scale = 1.0 + self.p.r_steer_rate_kappa_gain * min(
            kappa_abs_v, self.p.r_steer_rate_kappa_saturate
        )

        # Max of speed scheduling and curvature damping
        effective_scale = max(speed_scale, kappa_curve_scale)
        return r_base * effective_scale

    def _get_effective_q_lat(self, kappa: float = 0.0) -> float:
        """Compute curvature-scheduled q_lat.

        Higher curvature → higher q_lat → MPC values lateral tracking more
        on curves, making pre-steering cheaper than accepting entry error.
        """
        q_base = self.p.q_lat
        if self.p.q_lat_kappa_gain <= 0:
            return q_base
        kappa_abs = abs(kappa)
        scale = 1.0 + self.p.q_lat_kappa_gain * min(
            kappa_abs, self.p.q_lat_kappa_saturate
        )
        return q_base * scale

    @staticmethod
    def _feedforward_delta_norm(kappa: float, wheelbase_m: float, max_steer_rad: float) -> float:
        """Bicycle-model kinematic feedforward steering for curvature κ, normalized to [-1, 1].

        Derived from the steady-state bicycle equation δ = arctan(L · κ), then divided
        by δ_max to bring it into the same normalized space as the QP decision variable.
        Returns 0.0 for κ=0 (straight road), ensuring zero q-correction on straights.
        """
        return math.atan(wheelbase_m * kappa) / max_steer_rad

    @staticmethod
    def _feedforward_delta_norm_dynamic(kappa: float, v_x: float,
                                         C_f: float, C_r: float,
                                         l_f: float, l_r: float,
                                         mass: float, max_steer_rad: float) -> float:
        """Dynamic bicycle model steady-state feedforward with understeer gradient.

        δ_ss = L·κ + K_us·v²·κ  where K_us = (m/L)·(l_r/C_f − l_f/C_r).
        At low speed or zero curvature, reduces to kinematic feedforward.
        """
        L = l_f + l_r
        if abs(C_f) < 1.0 or abs(C_r) < 1.0:
            return math.atan(L * kappa) / max_steer_rad
        K_us = (mass / L) * (l_r / C_f - l_f / C_r)
        delta_rad = L * kappa + K_us * v_x * v_x * kappa
        return delta_rad / max_steer_rad

    # ----- QP construction -----

    def _build_qp(self):
        """Build OSQP problem matrices. Called once at init and on horizon change."""
        N = self._N
        nx, nu = self._nx, self._nu
        nz = (nx + nu) * N + nx
        self._nz = nz

        # --- P (Hessian): block diagonal ---
        P_diag = np.zeros(nz)
        _dyn = self.p.dynamic_model_enabled
        for k in range(N):
            base = k * (nx + nu)
            # State cost at step k (q_lat uses scheduled value)
            P_diag[base + 0] = self._current_q_lat
            P_diag[base + 1] = self.p.q_heading
            if _dyn:
                P_diag[base + 2] = self.p.q_vy         # v_y cost
                P_diag[base + 3] = self.p.q_yawrate     # yaw rate cost
                P_diag[base + 4] = self.p.q_speed        # v_x cost
            else:
                P_diag[base + 2] = self.p.q_speed
            # Input cost at step k
            P_diag[base + nx + 0] = self.p.r_steer
            P_diag[base + nx + 1] = self.p.r_accel

        # Terminal state cost
        term_base = N * (nx + nu)
        P_diag[term_base + 0] = self._current_q_lat * self.p.q_lat_terminal_scale
        P_diag[term_base + 1] = self.p.q_heading * self.p.q_heading_terminal_scale
        if _dyn:
            P_diag[term_base + 2] = self.p.q_vy
            P_diag[term_base + 3] = self.p.q_yawrate
            P_diag[term_base + 4] = self.p.q_speed
        else:
            P_diag[term_base + 2] = self.p.q_speed

        # Off-diagonal terms via COO triplets (OSQP requires upper-triangular P)
        rows, cols, vals = [], [], []
        for i in range(nz):
            if P_diag[i] != 0.0:
                rows.append(i)
                cols.append(i)
                vals.append(P_diag[i])

        # Lateral velocity damping: q_lat_rate * (e_lat[k+1] - e_lat[k])²
        # Since ė_lat ≈ v·e_heading·dt, this penalises lateral drift rate,
        # providing inherent oscillation damping that scales with speed through
        # the dynamics model.  Same off-diagonal pattern as r_steer_rate below.
        q_lr = self.p.q_lat_rate
        if q_lr > 0.0:
            for k in range(N - 1):
                idx_k = k * (nx + nu)            # e_lat at step k
                idx_k1 = (k + 1) * (nx + nu)     # e_lat at step k+1
                rows.append(idx_k)
                cols.append(idx_k)
                vals.append(q_lr)
                rows.append(idx_k1)
                cols.append(idx_k1)
                vals.append(q_lr)
                i_lo, i_hi = min(idx_k, idx_k1), max(idx_k, idx_k1)
                rows.append(i_lo)
                cols.append(i_hi)
                vals.append(-q_lr)

        # Steering rate penalty: r_steer_rate * (δ[k+1] - δ[k])²
        for k in range(N - 1):
            idx_k = k * (nx + nu) + nx       # δ_norm at step k
            idx_k1 = (k + 1) * (nx + nu) + nx  # δ_norm at step k+1
            r_sr = self._current_r_steer_rate

            # (δ[k+1] - δ[k])² = δ[k+1]² - 2·δ[k]·δ[k+1] + δ[k]²
            rows.append(idx_k)
            cols.append(idx_k)
            vals.append(r_sr)

            rows.append(idx_k1)
            cols.append(idx_k1)
            vals.append(r_sr)

            # Off-diagonal (upper triangle only for OSQP)
            i_lo, i_hi = min(idx_k, idx_k1), max(idx_k, idx_k1)
            rows.append(i_lo)
            cols.append(i_hi)
            vals.append(-r_sr)

        # First-step rate cost: 0.5 * r_sr * (delta[0] - last_delta_norm)^2
        # The within-horizon loop already puts r_sr on P[d0, d0] from the k=0 pair,
        # giving 0.5*r_sr*d0^2.  Adding another r_sr here raises the diagonal to
        # 2*r_sr, so the OSQP cost is 0.5*2*r_sr*d0^2 = r_sr*d0^2.  Combined with
        # the linear term q[d0] += -r_sr*last added each solve call, the net cost is:
        #   r_sr*d0^2 - r_sr*last*d0 = 0.5*r_sr*(d0 - last)^2 + 0.5*r_sr*(d0^2-last^2)
        # which equals 0.5*r_sr*(d0-last)^2 up to a constant the solver ignores.
        # This makes r_steer_rate penalise the first-step jump identically to every
        # within-horizon step, closing the structural gap that caused sign-flip jitter.
        if self.p.first_step_rate_enabled:
            idx_d0 = nx  # delta[0] sits at position nx in the decision vector
            rows.append(idx_d0)
            cols.append(idx_d0)
            vals.append(self._current_r_steer_rate)

        P = sparse.csc_matrix((vals, (rows, cols)), shape=(nz, nz))
        # OSQP requires upper-triangular P
        P = sparse.triu(P, format='csc')

        # --- Constraint matrix A ---
        # Constraints:
        # 1. Initial state equality: x₀ = x_init  (nx rows)
        # 2. Dynamics: x[k+1] = A_k·x[k] + B_k·u[k] + c_k  (N*nx rows)
        #    Encoded as: -x[k+1] + A_k·x[k] + B_k·u[k] = -c_k
        # 3. Input bounds: -1 ≤ δ_norm ≤ 1, -max_decel ≤ a ≤ max_accel  (N*nu rows)
        # 4. State bounds: v_min ≤ v[k] ≤ v_max  (N+1 rows, only speed)
        # 5. Steering rate: -rate_max ≤ δ[k+1] - δ[k] ≤ rate_max  (N-1 rows)

        A_rows = []
        l_parts = []
        u_parts = []

        # ---- 1. Initial state: x₀ ----
        # I·x₀ = x_init → row selects x₀ from z
        A_init = sparse.csc_matrix(
            sparse.hstack([
                sparse.eye(nx, format='csc'),
                sparse.csc_matrix((nx, nz - nx))
            ])
        )
        A_rows.append(A_init)
        # l and u will be set to x_init each solve
        l_parts.append(np.zeros(nx))
        u_parts.append(np.zeros(nx))

        # ---- 2. Dynamics: for k = 0..N-1 ----
        # -I·x[k+1] + A_k·x[k] + B_k·u[k] = -c_k
        # A_k and B_k depend on v[k] and κ[k], updated each solve.
        # Use non-zero placeholders (1e-10) to establish the sparse structure
        # so OSQP's update(Ax=...) doesn't see a structural change.
        _EPS = 1e-10
        self._dyn_row_start = nx  # first dynamics row index in A
        for k in range(N):
            row = sparse.lil_matrix((nx, nz))
            x_col = k * (nx + nu)
            u_col = k * (nx + nu) + nx
            x_next_col = (k + 1) * (nx + nu) if k < N - 1 else N * (nx + nu)

            if _dyn:
                # Dynamic bicycle: 5 states [e_lat, e_heading, v_y, r, v_x]
                # A_k (5×5) — all potentially-nonzero entries need placeholders
                row[0, x_col + 0] = 1.0    # e_lat → e_lat
                row[0, x_col + 1] = _EPS   # v_x*dt (heading coupling)
                row[0, x_col + 2] = _EPS   # dt (v_y coupling)
                row[1, x_col + 1] = 1.0    # e_heading → e_heading
                row[1, x_col + 3] = _EPS   # dt (yaw rate coupling)
                row[2, x_col + 2] = _EPS   # v_y self-coupling (tire damping)
                row[2, x_col + 3] = _EPS   # v_y ← r coupling
                row[3, x_col + 2] = _EPS   # r ← v_y coupling
                row[3, x_col + 3] = _EPS   # r self-coupling (tire damping)
                row[4, x_col + 4] = 1.0    # v_x → v_x
                # B_k (5×2) — steering affects v_y and r; accel affects v_x
                row[2, u_col + 0] = _EPS   # C_f*δ_max/m*dt
                row[3, u_col + 0] = _EPS   # C_f*l_f*δ_max/Iz*dt
                row[4, u_col + 1] = _EPS   # dt
                # -I for x[k+1]
                for s in range(nx):
                    row[s, x_next_col + s] = -1.0
            else:
                # Kinematic bicycle: 3 states [e_lat, e_heading, v]
                row[0, x_col + 0] = 1.0    # e_lat → e_lat
                row[0, x_col + 1] = _EPS   # v*dt (updated per-solve)
                row[1, x_col + 1] = 1.0    # e_heading → e_heading
                row[2, x_col + 2] = 1.0    # v → v
                # B_k
                row[1, u_col + 0] = _EPS   # (v/L)*δ_max*dt
                row[2, u_col + 1] = _EPS   # dt
                # -I for x[k+1]
                row[0, x_next_col + 0] = -1.0
                row[1, x_next_col + 1] = -1.0
                row[2, x_next_col + 2] = -1.0

            A_rows.append(row.tocsc())
            l_parts.append(np.zeros(nx))
            u_parts.append(np.zeros(nx))

        # ---- 3. Input bounds: for k = 0..N-1 ----
        self._input_row_start = nx + N * nx
        for k in range(N):
            row = sparse.lil_matrix((nu, nz))
            u_col = k * (nx + nu) + nx
            row[0, u_col + 0] = 1.0   # δ_norm
            row[1, u_col + 1] = 1.0   # a
            A_rows.append(row.tocsc())
            l_parts.append(np.array([-1.0, -self.p.max_decel]))
            u_parts.append(np.array([1.0, self.p.max_accel]))

        # ---- 4. State bounds ----
        # Kinematic: speed only (1 row per step).
        # Dynamic: v_y, r, and v_x (3 rows per step).
        self._state_row_start = nx + N * nx + N * nu
        if _dyn:
            # Bound v_y (index 2), r (index 3), v_x (index 4) per step
            self._state_rows_per_step = 3
            for k in range(N + 1):
                x_base = k * (nx + nu) if k < N else N * (nx + nu)
                # v_y bound
                row_vy = sparse.lil_matrix((1, nz))
                row_vy[0, x_base + 2] = 1.0
                A_rows.append(row_vy.tocsc())
                l_parts.append(np.array([-2.0]))     # |v_y| ≤ 2.0 m/s
                u_parts.append(np.array([2.0]))
                # r bound
                row_r = sparse.lil_matrix((1, nz))
                row_r[0, x_base + 3] = 1.0
                A_rows.append(row_r.tocsc())
                l_parts.append(np.array([-1.0]))      # |r| ≤ 1.0 rad/s
                u_parts.append(np.array([1.0]))
                # v_x bound
                row_vx = sparse.lil_matrix((1, nz))
                row_vx[0, x_base + 4] = 1.0
                A_rows.append(row_vx.tocsc())
                l_parts.append(np.array([self.p.v_min]))
                u_parts.append(np.array([self.p.v_max]))
        else:
            self._state_rows_per_step = 1
            for k in range(N + 1):
                row = sparse.lil_matrix((1, nz))
                if k < N:
                    x_col = k * (nx + nu) + 2   # v at step k
                else:
                    x_col = N * (nx + nu) + 2   # v at terminal
                row[0, x_col] = 1.0
                A_rows.append(row.tocsc())
                l_parts.append(np.array([self.p.v_min]))
                u_parts.append(np.array([self.p.v_max]))

        # ---- 5. Steering rate: for k = 0..N-2 ----
        self._rate_row_start = nx + N * nx + N * nu + (N + 1) * self._state_rows_per_step
        for k in range(N - 1):
            row = sparse.lil_matrix((1, nz))
            idx_k = k * (nx + nu) + nx       # δ_norm at step k
            idx_k1 = (k + 1) * (nx + nu) + nx  # δ_norm at step k+1
            row[0, idx_k] = -1.0
            row[0, idx_k1] = 1.0
            A_rows.append(row.tocsc())
            l_parts.append(np.array([-self.p.delta_rate_max]))
            u_parts.append(np.array([self.p.delta_rate_max]))

        A = sparse.vstack(A_rows, format='csc')
        l = np.concatenate(l_parts)
        u = np.concatenate(u_parts)
        self._nc = len(l)

        # Build (row, col) → CSC data index map for fast in-place updates.
        # This avoids the tolil()→tocsc() roundtrip in solve() that changes
        # the sparse structure and forces expensive OSQP re-setup.
        A.sort_indices()
        self._A_idx: dict[tuple[int, int], int] = {}
        indptr, indices = A.indptr, A.indices
        for c in range(A.shape[1]):
            for pos in range(indptr[c], indptr[c + 1]):
                self._A_idx[(int(indices[pos]), c)] = int(pos)

        # Linear cost vector (zeros — updated each solve)
        q = np.zeros(nz)

        # Store templates
        self._P = P
        self._A = A
        self._q = q
        self._l = l.copy()
        self._u = u.copy()

        # Setup OSQP
        self._prob = osqp.OSQP()
        self._prob.setup(
            P, q, A, l, u,
            warm_start=True,
            verbose=False,
            max_iter=200,
            eps_abs=1e-4,
            eps_rel=1e-4,
            polish=True,
            adaptive_rho=True,
        )

    # ----- Per-frame solve -----

    def solve(self, e_lat: float, e_heading: float, v: float,
              last_delta_norm: float, kappa_ref_horizon: np.ndarray,
              v_target: float, v_max: float, dt: float,
              grade_rad: float = 0.0,
              tire_cf: float = 40000.0, tire_cr: float = 40000.0,
              v_y_init: float = 0.0, r_init: float = 0.0) -> dict:
        """
        Solve QP for one frame.

        Args:
            e_lat: current lateral error (m)
            e_heading: current heading error (rad)
            v: current speed (m/s)
            last_delta_norm: previous normalized steering [-1, 1]
            kappa_ref_horizon: array of N reference curvatures (from map)
            v_target: desired speed (m/s)
            v_max: speed limit (m/s)
            dt: timestep (s)
            grade_rad: road grade in radians (positive = uphill), default 0.0
            tire_cf: front cornering stiffness (N/rad), dynamic model only
            tire_cr: rear cornering stiffness (N/rad), dynamic model only
            v_y_init: lateral velocity (m/s), dynamic model only
            r_init: yaw rate (rad/s), dynamic model only

        Returns:
            dict with: steering_normalized, accel, feasible, solve_time_ms,
                        predicted_trajectory (N+1, nx)
        """
        t0 = time.perf_counter()
        N = self._N
        nx, nu = self._nx, self._nu
        nz = self._nz

        # Check for horizon change or r_steer_rate scheduling band crossing
        new_N = self._get_horizon(v)
        kappa_now = float(kappa_ref_horizon[0]) if kappa_ref_horizon is not None and len(kappa_ref_horizon) > 0 else 0.0
        new_r_sr = self._get_effective_r_steer_rate(v, kappa_now)
        need_rebuild = (new_N != N)

        # Hysteresis: only rebuild for r_steer_rate if change is meaningful
        if abs(new_r_sr - self._current_r_steer_rate) > 0.1:
            speed_crossed = abs(v - self._speed_band_center) > self.p.r_steer_rate_band_hysteresis
            kappa_crossed = abs(kappa_now - self._kappa_band_center) > 0.003
            if speed_crossed or kappa_crossed:
                self._current_r_steer_rate = new_r_sr
                self._speed_band_center = v
                self._kappa_band_center = kappa_now
                need_rebuild = True

        # q_lat curvature scheduling: rebuild when q_lat changes meaningfully
        new_q_lat = self._get_effective_q_lat(kappa_now)
        if abs(new_q_lat - self._current_q_lat) > 0.2:
            kappa_crossed = abs(kappa_now - self._kappa_band_center) > 0.003
            if kappa_crossed or need_rebuild:
                self._current_q_lat = new_q_lat
                self._kappa_band_center = kappa_now
                need_rebuild = True

        if need_rebuild:
            self._N = new_N
            self._build_qp()
            self._warm_x = None
            self._warm_y = None
            N = self._N
            nz = self._nz

        # Ensure kappa_ref has correct length
        kappa = np.zeros(N)
        if kappa_ref_horizon is not None:
            n_copy = min(len(kappa_ref_horizon), N)
            kappa[:n_copy] = kappa_ref_horizon[:n_copy]
            if n_copy < N:
                kappa[n_copy:] = kappa[n_copy - 1] if n_copy > 0 else 0.0

        # Clamp speed for numerical stability
        v_safe = max(v, 0.5)

        # Grade compensation: clamp and compute gravity offset (Step 3)
        grade_clamped = max(-self.p.grade_clamp_rad,
                            min(self.p.grade_clamp_rad, grade_rad))
        gravity_accel_offset = -9.81 * math.sin(grade_clamped) * self.p.grade_ff_gain  # m/s²

        # --- Update dynamics in A matrix and c vector ---
        # For each step k, linearize around operating point.
        # Update entries directly in the CSC data array via the pre-built
        # index map (self._A_idx).  This preserves the sparse structure so
        # OSQP's update(Ax=...) succeeds without a full re-setup.
        A_data = self._A.data
        _ai = self._A_idx   # (row, col) → data index
        l_new = self._l.copy()
        u_new = self._u.copy()
        _dyn = self.p.dynamic_model_enabled

        c_vec = []  # affine terms for dynamics
        for k in range(N):
            dyn_row = nx + k * nx  # row index in A for dynamics block k
            x_col = k * (nx + nu)
            u_col = k * (nx + nu) + nx
            x_next_col = (k + 1) * (nx + nu) if k < N - 1 else N * (nx + nu)
            kk = kappa[k]

            if _dyn:
                # ── Dynamic bicycle model (5 states) ──────────────────────
                C_f = tire_cf
                C_r = tire_cr
                m = self.p.vehicle_mass_kg
                Iz = self.p.vehicle_iz_kgm2
                l_f = self.p.vehicle_lf_m
                l_r = self.p.vehicle_lr_m
                vx = max(v_safe, 3.0)  # conservative clamp for tire force / v_x

                # A_k (5×5) — see plan for full derivation
                # Row 0: e_lat[k+1] = e_lat + vx·e_heading·dt + v_y·dt
                A_data[_ai[(dyn_row + 0, x_col + 0)]] = 1.0
                A_data[_ai[(dyn_row + 0, x_col + 1)]] = vx * dt
                A_data[_ai[(dyn_row + 0, x_col + 2)]] = dt         # v_y coupling
                # Row 1: e_heading[k+1] = e_heading + r·dt (−κ·vx·dt in c_k)
                A_data[_ai[(dyn_row + 1, x_col + 1)]] = 1.0
                A_data[_ai[(dyn_row + 1, x_col + 3)]] = dt         # yaw rate coupling
                # Row 2: v_y dynamics (tire lateral forces)
                A_data[_ai[(dyn_row + 2, x_col + 2)]] = 1.0 - (C_f + C_r) / (m * vx) * dt
                A_data[_ai[(dyn_row + 2, x_col + 3)]] = (-vx - (C_f * l_f - C_r * l_r) / (m * vx)) * dt
                # Row 3: yaw rate dynamics (tire yaw moments)
                A_data[_ai[(dyn_row + 3, x_col + 2)]] = -(C_f * l_f - C_r * l_r) / (Iz * vx) * dt
                A_data[_ai[(dyn_row + 3, x_col + 3)]] = 1.0 - (C_f * l_f**2 + C_r * l_r**2) / (Iz * vx) * dt
                # Row 4: v_x[k+1] = v_x + a·dt
                A_data[_ai[(dyn_row + 4, x_col + 4)]] = 1.0

                # B_k (5×2)
                _ms = self._max_steer_at_speed(vx)
                A_data[_ai[(dyn_row + 2, u_col + 0)]] = C_f * _ms / m * dt
                A_data[_ai[(dyn_row + 3, u_col + 0)]] = C_f * l_f * _ms / Iz * dt
                A_data[_ai[(dyn_row + 4, u_col + 1)]] = dt

                # -I for x[k+1] — these are constant but included for completeness
                for s in range(nx):
                    A_data[_ai[(dyn_row + s, x_next_col + s)]] = -1.0

                # Affine term c_k
                c_vec.append(np.array([
                    0.0,                           # c_lat
                    -kk * vx * dt,                 # c_heading (curvature)
                    0.0,                           # c_vy
                    0.0,                           # c_r
                    gravity_accel_offset * dt,     # c_vx (grade)
                ]))
            else:
                # ── Kinematic bicycle model (3 states) ────────────────────
                # A_k matrix (state transition) — only update varying entries
                A_data[_ai[(dyn_row + 0, x_col + 1)]] = v_safe * dt

                # B_k matrix (input)
                steer_gain = (v_safe / self.p.wheelbase_m) * self._max_steer_at_speed(v_safe) * dt
                A_data[_ai[(dyn_row + 1, u_col + 0)]] = steer_gain
                A_data[_ai[(dyn_row + 2, u_col + 1)]] = dt

                # Affine term c_k
                c_lat = 0.0
                c_head = -kk * v_safe * dt
                c_v = gravity_accel_offset * dt
                c_vec.append(np.array([c_lat, c_head, c_v]))

        # --- Update constraint bounds ---
        # 1. Initial state
        l_new[0] = e_lat
        l_new[1] = e_heading
        u_new[0] = e_lat
        u_new[1] = e_heading
        if _dyn:
            l_new[2] = v_y_init
            l_new[3] = r_init
            l_new[4] = v_safe
            u_new[2] = v_y_init
            u_new[3] = r_init
            u_new[4] = v_safe
        else:
            l_new[2] = v_safe
            u_new[2] = v_safe

        # 2. Dynamics: l = u = -c_k (equality constraints)
        for k in range(N):
            row_start = nx + k * nx
            c_k = c_vec[k]
            l_new[row_start:row_start + nx] = -c_k
            u_new[row_start:row_start + nx] = -c_k

        # 3. Input bounds: grade-aware accel authority expansion.
        # Grade-zero proof: gravity_accel_offset=0 → bounds unchanged.
        abs_gravity = abs(gravity_accel_offset)
        input_start = self._input_row_start
        nu = 2  # (delta, accel)
        for k in range(N):
            idx = input_start + k * nu
            l_new[idx + 1] = -(self.p.max_decel + abs_gravity)   # accel lower bound
            u_new[idx + 1] = self.p.max_accel + abs_gravity       # accel upper bound

        # 4. State bounds: update v_max (and v_y/r bounds for dynamic model)
        state_start = self._state_row_start
        sps = self._state_rows_per_step
        for k in range(N + 1):
            if _dyn:
                base = state_start + k * sps
                l_new[base + 0] = -2.0          # v_y lower
                u_new[base + 0] = 2.0            # v_y upper
                l_new[base + 1] = -1.0           # r lower
                u_new[base + 1] = 1.0            # r upper
                l_new[base + 2] = self.p.v_min   # v_x lower
                u_new[base + 2] = v_max           # v_x upper
            else:
                l_new[state_start + k] = self.p.v_min
                u_new[state_start + k] = v_max

        # 5. Steering rate bounds — add constraint for first step vs last_delta_norm
        # We handle this by augmenting: the first steering rate isn't in the N-1
        # rate constraints. Instead, we adjust the first δ bounds to enforce rate.
        # Restrict first input δ to [last - rate_max, last + rate_max]
        input_start = self._input_row_start
        delta_lo = max(-1.0, last_delta_norm - self.p.delta_rate_max)
        delta_hi = min(1.0, last_delta_norm + self.p.delta_rate_max)
        l_new[input_start + 0] = delta_lo  # first δ lower
        u_new[input_start + 0] = delta_hi  # first δ upper

        # --- Update linear cost q ---
        q_new = np.zeros(nz)
        _v_idx = 4 if _dyn else 2  # speed state index
        for k in range(N):
            base = k * (nx + nu)
            q_new[base + _v_idx] = -self.p.q_speed * v_target

        # Terminal speed tracking
        term_base = N * (nx + nu)
        q_new[term_base + _v_idx] = -self.p.q_speed * v_target

        # --- 2DOF feedforward alignment (rate bias only) ---
        # Recenters the steering-RATE cost at Δδ_ff rather than 0, so MPC does not
        # pay a penalty for matching the kinematic feedforward rate at curve entries.
        #
        # Math: cost changes from 0.5·r_sr·(Δδ)² to 0.5·r_sr·(Δδ − Δff)².
        # Expanding: quadratic term (P) is unchanged; the linear correction to q is:
        #   q[idx_k]  += r_sr · Δff_k     (δ[k]   position)
        #   q[idx_k1] -= r_sr · Δff_k     (δ[k+1] position)
        # where Δff_k = δ_ff[k+1] − δ_ff[k].
        #
        # This only fires when κ is CHANGING (curve entry/exit). On constant-κ arcs
        # and on straights, Δff_k = 0 everywhere → correction is zero.
        # It does NOT bias toward any absolute δ value, so it never fights lateral
        # error corrections needed to re-center the car after a disturbance.
        #
        # NOTE: Part A (magnitude bias toward δ_ff) was intentionally removed.
        # With r_steer reduced to near-zero, the original δ=0 bias that Part A
        # compensated for is gone. Part A's overcorrection when the car has lateral
        # offset was causing persistent centeredness regression on curves.
        if self.p.ff_alignment_enabled:
            _ms_ff = self._max_steer_at_speed(v_safe)
            if _dyn:
                delta_ff = np.array([
                    self._feedforward_delta_norm_dynamic(
                        kappa[k], v_safe, tire_cf, tire_cr,
                        self.p.vehicle_lf_m, self.p.vehicle_lr_m,
                        self.p.vehicle_mass_kg, _ms_ff)
                    for k in range(N)
                ], dtype=float)
            else:
                delta_ff = np.array([
                    self._feedforward_delta_norm(kappa[k], self._ff_wheelbase_m, _ms_ff)
                    for k in range(N)
                ], dtype=float)

            r_sr = self._current_r_steer_rate
            for k in range(N - 1):
                dff = delta_ff[k + 1] - delta_ff[k]
                if dff == 0.0:
                    continue
                idx_k = k * (nx + nu) + nx
                idx_k1 = (k + 1) * (nx + nu) + nx
                q_new[idx_k] += r_sr * dff
                q_new[idx_k1] -= r_sr * dff

        # First-step rate linear correction: -r_sr * last_delta_norm added to q[d0].
        # Combined with the extra r_sr diagonal in P (added in _build_qp), this
        # implements the full 0.5*r_sr*(delta[0] - last_delta_norm)^2 cost.
        if self.p.first_step_rate_enabled:
            q_new[nx] += -self._current_r_steer_rate * last_delta_norm

        # --- Solve ---
        try:
            self._prob.update(Ax=self._A.data, q=q_new, l=l_new, u=u_new)
        except Exception:
            # If structure changed (e.g. after horizon rebuild), need full setup
            self._prob = osqp.OSQP()
            self._prob.setup(
                self._P, q_new, self._A, l_new, u_new,
                warm_start=True, verbose=False,
                max_iter=200, eps_abs=1e-4, eps_rel=1e-4,
                polish=True, adaptive_rho=True,
            )

        # Apply warm-start
        if self._warm_x is not None and len(self._warm_x) == nz:
            try:
                self._prob.warm_start(x=self._warm_x, y=self._warm_y)
            except Exception:
                pass

        res = self._prob.solve()
        solve_ms = (time.perf_counter() - t0) * 1000.0

        # Check feasibility
        if res.info.status not in ('solved', 'solved_inaccurate'):
            return {
                'steering_normalized': 0.0,
                'accel': 0.0,
                'feasible': False,
                'solve_time_ms': solve_ms,
                'predicted_trajectory': np.zeros((N + 1, nx)),
                'osqp_status': res.info.status,
            }

        z = res.x

        # Extract first control input: u[0] = z[nx : nx+nu]
        delta_norm_0 = float(np.clip(z[nx + 0], -1.0, 1.0))
        accel_0 = float(np.clip(z[nx + 1], -self.p.max_decel, self.p.max_accel))

        # Store warm-start: shift solution by one step
        warm_x = np.zeros(nz)
        for k in range(N - 1):
            src = (k + 1) * (nx + nu)
            dst = k * (nx + nu)
            warm_x[dst:dst + nx + nu] = z[src:src + nx + nu]
        # Last step: repeat final
        warm_x[(N - 1) * (nx + nu):] = z[(N - 1) * (nx + nu):]
        self._warm_x = warm_x
        self._warm_y = res.y.copy() if res.y is not None else None

        # Extract predicted trajectory for diagnostics
        predicted = np.zeros((N + 1, nx))
        for k in range(N):
            base = k * (nx + nu)
            predicted[k] = z[base:base + nx]
        predicted[N] = z[N * (nx + nu):N * (nx + nu) + nx]

        return {
            'steering_normalized': delta_norm_0,
            'accel': accel_0,
            'feasible': True,
            'solve_time_ms': solve_ms,
            'predicted_trajectory': predicted,
            'osqp_status': res.info.status,
            'r_steer_rate_effective': self._current_r_steer_rate,
            'q_lat_effective': self._current_q_lat,
        }


# ---------------------------------------------------------------------------
# MPCController — public wrapper with fallback logic
# ---------------------------------------------------------------------------

class MPCController:
    """
    MPC lateral controller with fallback tracking.

    Called by VehicleController each frame when regime = LINEAR_MPC.
    If the solver fails for max_consecutive_failures frames, activates
    fallback (hold last good steering).
    """

    def __init__(self, config: dict):
        self.params = MPCParams.from_config(config)
        self.solver = MPCSolver(self.params)
        self._consecutive_failures = 0
        self._fallback_active = False
        self._last_steering = 0.0
        self._frames_since_reset = 0
        # Lateral bias estimator: slow EMA of e_lat → curvature correction
        self._bias_ema = 0.0
        self._bias_correction = 0.0
        # Arc-entry kappa-transition warmup
        self._last_kappa_ref: float = 0.0
        # Lateral-error EMA pre-filter state
        self._elat_ema: float = 0.0
        # Dropout-recovery rate-limiter state: tracks the previous QP e_lat input
        # so we can cap frame-to-frame steps when perception snaps back from a
        # silent dropout (e_lat was ≈0, now suddenly large).
        self._prev_elat_ramp: float = 0.0
        # Delegate speed-dependent steering to solver for use in EKF/RLS
        self._max_steer_at_speed = self.solver._max_steer_at_speed
        # Online effective wheelbase (RLS on heading prediction error)
        self._leff_theta: float = 1.0 / self.params.wheelbase_m  # θ = 1/L_eff
        self._leff_P: float = self.params.leff_initial_P          # RLS covariance (scalar)
        self._leff_heading_prev: float = 0.0                       # e_heading at previous frame
        self._leff_steering_prev: float = 0.0                      # δ_norm at previous frame
        self._leff_speed_prev: float = 0.0                         # speed at previous frame
        self._leff_update_count: int = 0                           # frames where RLS updated
        # Online tire cornering stiffness estimation (EKF)
        self._tire_theta = np.array([self.params.tire_cf_nominal,
                                      self.params.tire_cr_nominal], dtype=float)
        self._tire_P = np.eye(2) * 1e6                           # high initial uncertainty
        self._tire_ekf_update_count: int = 0
        self._tire_ekf_divergent_count: int = 0                  # consecutive divergent frames
        self._v_y_est: float = 0.0                               # lateral velocity estimate
        self._yaw_rate_est: float = 0.0                          # yaw rate estimate
        self._e_heading_prev_tire: float = 0.0                   # heading at previous frame (for EKF)

    def compute_steering(self, e_lat: float, e_heading: float,
                         current_speed: float, last_delta_norm: float,
                         kappa_ref: float, v_target: float,
                         v_max: float, dt: float,
                         kappa_horizon=None,
                         grade_rad: float = 0.0,
                         curve_local_state: Optional[str] = None,
                         curve_gate_weight: float = 0.0,
                         local_curve_reference_active: bool = False,
                         imu_yaw_rate: Optional[float] = None) -> dict:
        """
        Compute MPC steering. Called by VehicleController each frame.

        Args:
            e_lat: lateral error from reference path (m)
            e_heading: heading error from reference heading (rad)
            current_speed: vehicle speed (m/s)
            last_delta_norm: previous normalized steering [-1, 1]
            kappa_ref: reference curvature at current position (1/m)
            v_target: desired speed (m/s)
            v_max: speed limit (m/s)
            dt: timestep (s)
            kappa_horizon: optional N-element array of signed curvature at
                each MPC prediction step (1/m). When provided and
                curvature_preview_enabled=True, replaces the constant-fill
                κ_ref with a spatially-varying feedforward horizon.
            grade_rad: road grade in radians (positive = uphill), default 0.0
            curve_local_state: scheduler-owned curve state (STRAIGHT/ENTRY/COMMIT/REARM)
            curve_gate_weight: local curve relevance weight [0, 1]
            local_curve_reference_active: whether the local curve reference is valid/active

        Returns:
            dict with: steering_normalized, accel, mpc_feasible, solve_time_ms,
                       mpc_fallback_active, mpc_consecutive_failures,
                       e_lat_input, e_heading_input, kappa_ref_used
        """
        if abs(e_heading) > 0.3:
            logger.warning(
                "MPC: heading error %.3f rad exceeds small-angle limit (0.3 rad)",
                e_heading
            )

        # --- Lateral bias estimator ---
        # Slowly accumulate persistent lateral error into a curvature correction.
        # This acts like integral control: it eliminates steady-state offset in
        # sustained curves without increasing the proportional gain (q_lat) that
        # causes oscillation at highway speeds.
        #
        # Sign logic: the MPC dynamics predict e_lat[k+1] = ... − κ·v²·dt.
        # Increasing κ makes the model think the curve will self-correct the
        # error, so the MPC steers LESS.  We need the opposite: when e_lat > 0
        # persistently (car off-center), DECREASE κ so the MPC thinks the curve
        # is weaker and steers MORE aggressively.  Hence: Δκ = −gain × bias.
        self._bias_correction = 0.0
        bias_guard_active = False
        bias_guard_limit = 0.0
        active_curve_preserve_ratio = float(self.params.bias_active_curve_preserve_ratio)
        active_curve_preserve_active = False
        active_curve_preserve_weight = 0.0
        active_mild_curve_authority_active = False
        active_mild_curve_authority_weight = 0.0
        active_mild_curve_authority_ratio = 0.0
        active_mild_curve_authority_reason = "disabled"
        active_mild_curve_authority_speed_weight = 0.0
        active_mild_curve_authority_curvature_weight = 0.0
        active_mild_curve_authority_gate_weight = 0.0

        def _norm_weight(value: float, value_on: float, value_full: float) -> float:
            if not np.isfinite(value):
                return 0.0
            if value_full <= value_on:
                return 1.0 if value >= value_on else 0.0
            if value <= value_on:
                return 0.0
            if value >= value_full:
                return 1.0
            return float((value - value_on) / max(1e-6, value_full - value_on))

        if self.params.bias_enabled and current_speed >= self.params.bias_min_speed:
            alpha = self.params.bias_alpha
            self._bias_ema = (1.0 - alpha) * self._bias_ema + alpha * e_lat
            if self.params.bias_ema_abs_max > 0:
                self._bias_ema = float(np.clip(
                    self._bias_ema,
                    -self.params.bias_ema_abs_max,
                    self.params.bias_ema_abs_max,
                ))
            raw_correction = -self.params.bias_kappa_gain * self._bias_ema
            self._bias_correction = float(np.clip(
                raw_correction,
                -self.params.bias_max_correction,
                self.params.bias_max_correction,
            ))
            active_curve_state = str(curve_local_state or "").strip().upper()
            active_curve_gate_weight = max(0.0, min(1.0, float(curve_gate_weight or 0.0)))
            guard_limits: list[float] = []
            if (
                self.params.bias_relative_guard_enabled
                and abs(kappa_ref) >= self.params.bias_relative_guard_min_kappa
            ):
                guard_limits.append(self.params.bias_relative_guard_ratio * abs(kappa_ref))
            if (
                self.params.bias_active_curve_preserve_enabled
                and active_curve_state in {"ENTRY", "COMMIT"}
                and active_curve_gate_weight >= self.params.bias_active_curve_gate_min
                and abs(kappa_ref) >= self.params.bias_relative_guard_min_kappa
            ):
                if self.params.bias_active_curve_preserve_dynamic_enabled:
                    speed_on = float(self.params.bias_active_curve_preserve_speed_on)
                    speed_full = float(self.params.bias_active_curve_preserve_speed_full)
                    if speed_full <= speed_on:
                        speed_weight = 1.0 if current_speed >= speed_on else 0.0
                    elif current_speed <= speed_on:
                        speed_weight = 0.0
                    elif current_speed >= speed_full:
                        speed_weight = 1.0
                    else:
                        speed_weight = (current_speed - speed_on) / max(1e-6, speed_full - speed_on)
                    curv_on = float(self.params.bias_active_curve_preserve_curvature_on)
                    curv_full = float(self.params.bias_active_curve_preserve_curvature_full)
                    abs_kappa_ref = abs(kappa_ref)
                    if curv_full <= curv_on:
                        curvature_weight = 1.0 if abs_kappa_ref >= curv_on else 0.0
                    elif abs_kappa_ref <= curv_on:
                        curvature_weight = 0.0
                    elif abs_kappa_ref >= curv_full:
                        curvature_weight = 1.0
                    else:
                        curvature_weight = (abs_kappa_ref - curv_on) / max(1e-6, curv_full - curv_on)
                    gate_weight = (
                        (active_curve_gate_weight - self.params.bias_active_curve_gate_min)
                        / max(1e-6, 1.0 - self.params.bias_active_curve_gate_min)
                    )
                    gate_weight = max(0.0, min(1.0, gate_weight))
                    active_curve_preserve_weight = max(
                        0.0,
                        min(1.0, speed_weight * curvature_weight * gate_weight),
                    )
                    active_curve_preserve_ratio = float(
                        max(
                            self.params.bias_active_curve_preserve_ratio,
                            self.params.bias_active_curve_preserve_ratio
                            + (
                                self.params.bias_active_curve_preserve_full_ratio
                                - self.params.bias_active_curve_preserve_ratio
                            )
                            * active_curve_preserve_weight,
                        )
                    )
                active_curve_preserve_active = active_curve_preserve_weight > 0.0
                if self._bias_correction * kappa_ref > 0.0 and active_curve_preserve_active:
                    same_sign_guard_ratio = float(self.params.bias_relative_guard_ratio)
                    same_sign_guard_ratio = max(
                        0.0,
                        min(
                            same_sign_guard_ratio,
                            same_sign_guard_ratio
                            - (
                                same_sign_guard_ratio
                                - float(self.params.bias_active_curve_same_sign_guard_ratio)
                            )
                            * active_curve_preserve_weight,
                        ),
                    )
                    guard_limits.append(same_sign_guard_ratio * abs(kappa_ref))
                elif self._bias_correction * kappa_ref < 0.0:
                    guard_limits.append(
                        max(0.0, (1.0 - active_curve_preserve_ratio) * abs(kappa_ref))
                    )
            if guard_limits:
                bias_guard_limit = min(self.params.bias_max_correction, *guard_limits)
                if abs(self._bias_correction) > bias_guard_limit:
                    self._bias_correction = math.copysign(bias_guard_limit, self._bias_correction)
                    bias_guard_active = True

        active_curve_state = str(curve_local_state or "").strip().upper()
        active_curve_gate_weight = max(0.0, min(1.0, float(curve_gate_weight or 0.0)))
        abs_kappa_ref = abs(float(kappa_ref))
        if not self.params.active_mild_curve_authority_enabled:
            active_mild_curve_authority_reason = "disabled"
        elif active_curve_state not in {"ENTRY", "COMMIT"}:
            active_mild_curve_authority_reason = "state_inactive"
        elif not bool(local_curve_reference_active):
            active_mild_curve_authority_reason = "local_reference_inactive"
        else:
            active_mild_curve_authority_speed_weight = _norm_weight(
                float(current_speed),
                float(self.params.active_mild_curve_authority_speed_on),
                float(self.params.active_mild_curve_authority_speed_full),
            )
            active_mild_curve_authority_curvature_weight = _norm_weight(
                abs_kappa_ref,
                float(self.params.active_mild_curve_authority_curvature_on),
                float(self.params.active_mild_curve_authority_curvature_full),
            )
            active_mild_curve_authority_gate_weight = _norm_weight(
                active_curve_gate_weight,
                float(self.params.active_mild_curve_authority_gate_min),
                1.0,
            )
            active_mild_curve_authority_weight = max(
                0.0,
                min(
                    1.0,
                    active_mild_curve_authority_speed_weight
                    * active_mild_curve_authority_curvature_weight
                    * active_mild_curve_authority_gate_weight,
                ),
            )
            active_mild_curve_authority_active = active_mild_curve_authority_weight > 0.0
            active_mild_curve_authority_ratio = float(
                max(
                    self.params.active_mild_curve_authority_ratio,
                    self.params.active_mild_curve_authority_ratio
                    + (
                        self.params.active_mild_curve_authority_full_ratio
                        - self.params.active_mild_curve_authority_ratio
                    )
                    * active_mild_curve_authority_weight,
                )
            )
            if active_mild_curve_authority_active:
                active_mild_curve_authority_reason = "active"
            elif active_mild_curve_authority_speed_weight <= 0.0:
                active_mild_curve_authority_reason = "speed_below_on"
            elif active_mild_curve_authority_curvature_weight <= 0.0:
                active_mild_curve_authority_reason = "curvature_below_on"
            elif active_mild_curve_authority_gate_weight <= 0.0:
                active_mild_curve_authority_reason = "gate_below_min"
            else:
                active_mild_curve_authority_reason = "inactive"

        # Build κ_ref horizon with bias correction applied
        _preview_used = (
            self.params.curvature_preview_enabled
            and kappa_horizon is not None
            and len(kappa_horizon) > 0
        )
        if _preview_used:
            kappa_base = np.array(kappa_horizon, dtype=float) * self.params.curvature_preview_gain
            kappa_corrected_horizon = kappa_base + self._bias_correction
        else:
            # Fallback: constant curvature (current behavior)
            kappa_corrected = kappa_ref + self._bias_correction
            kappa_corrected_horizon = np.full(self.solver._N, kappa_corrected)
        if active_mild_curve_authority_active and abs_kappa_ref >= 1e-6:
            if _preview_used:
                base_sign = np.sign(np.where(np.abs(kappa_base) > 1e-6, kappa_base, kappa_ref))
                floor_abs = active_mild_curve_authority_ratio * np.maximum(np.abs(kappa_base), abs_kappa_ref)
                kappa_corrected_horizon = base_sign * np.maximum(
                    np.abs(kappa_corrected_horizon),
                    floor_abs,
                )
            else:
                floor_abs = active_mild_curve_authority_ratio * abs_kappa_ref
                kappa_corrected_horizon = np.full(
                    self.solver._N,
                    math.copysign(max(abs(kappa_corrected_horizon[0]), floor_abs), kappa_ref),
                )
        kappa_horizon = kappa_corrected_horizon

        # --- Arc-entry kappa-transition warmup ---
        # When kappa_ref jumps by > threshold in one frame (car crosses from
        # straight into arc), reset the warmup counter.  The subsequent
        # warmup burst relaxes delta_rate_max to 1.0, letting the solver apply
        # the curvature feedforward in one step rather than ramping over ~6 frames
        # (0.2 s) while the car drifts outside.  Triggers on kappa INCREASE only
        # (straight→curve); curve exit is gradual and doesn't need this.
        if self.params.kappa_transition_warmup_enabled:
            if kappa_ref - self._last_kappa_ref > self.params.kappa_transition_threshold:
                self._frames_since_reset = 0   # trigger warmup burst
        self._last_kappa_ref = kappa_ref

        # --- Startup warmup ---
        # During the first few frames after activation (no warm-start), relax
        # the rate constraint and give OSQP more iterations.  This prevents
        # infeasibility when the solver cold-starts with an unfavourable initial
        # state inherited from Pure Pursuit.
        _in_warmup = self._frames_since_reset < self.params.warmup_frames
        if _in_warmup:
            # Temporarily increase OSQP max_iter for cold-start convergence
            try:
                self.solver._prob.update_settings(max_iter=self.params.warmup_max_iter)
            except Exception:
                pass
            # Relax first-step rate constraint: don't restrict δ based on PP's
            # last steering — use the full range so the solver can find a feasible
            # starting point.
            last_delta_norm = 0.0

        # --- Lateral-error EMA pre-filter ---
        # Attenuates high-frequency perception noise before it reaches the QP.
        # Without this, σ≈0.05 m noise creates alternating-sign errors that force
        # the MPC to flip steering every frame — the primary driver of steering jerk.
        # alpha=1.0 / deadband=0.0 reproduce the unfiltered behaviour exactly.
        alpha_elat = float(self.params.elat_ema_alpha)
        deadband_m = float(self.params.elat_deadband_m)
        if alpha_elat < 1.0 or deadband_m > 0.0:
            self._elat_ema = alpha_elat * e_lat + (1.0 - alpha_elat) * self._elat_ema
            e_lat_qp = self._elat_ema if abs(self._elat_ema) > deadband_m else 0.0
        else:
            e_lat_qp = e_lat

        # --- Dropout-recovery rate limiter ---
        # After a silent perception dropout (e_lat≈0 for N frames), perception
        # can snap back with a large step (e.g. 0 → 0.25 m) that the QP sees as
        # an instantaneous error and responds to with a large steering jerk.
        # The rate limiter caps how much e_lat_qp can change per frame.
        # Only active when elat_ramp_rate_m_per_frame > 0 (disabled by default).
        elat_ramp_active = False
        ramp_rate = float(self.params.elat_ramp_rate_m_per_frame)
        if ramp_rate > 0.0:
            delta = e_lat_qp - self._prev_elat_ramp
            if abs(delta) > ramp_rate:
                e_lat_qp = self._prev_elat_ramp + float(np.sign(delta)) * ramp_rate
                elat_ramp_active = True
        self._prev_elat_ramp = e_lat_qp

        # --- Online effective wheelbase estimation (RLS) ---
        # Update θ = 1/L_eff from heading prediction error BEFORE solving,
        # then inject L_eff into the solver's dynamics model.
        # NOTE: Use actual frame dt for RLS (time between heading measurements),
        # NOT self.params.dt (MPC planning step). The RLS models what happened
        # between real measurements, not the MPC's internal prediction horizon.
        leff_result = self._update_leff(e_heading, current_speed, last_delta_norm, kappa_ref, dt)
        if self.params.leff_estimation_enabled:
            self.solver.p = _dc_replace(self.solver.p, wheelbase_m=self.leff)

        # EKF tire estimation — update before solve so solver gets latest C_f/C_r
        tire_result = self._update_tire_ekf(
            e_heading, current_speed, last_delta_norm, kappa_ref, dt,
            imu_yaw_rate=imu_yaw_rate,
        )

        # Use the configured MPC prediction step (params.dt), NOT the frame dt.
        # The solver plans N steps of params.dt each (e.g. 20×0.1=2.0 s horizon).
        # Passing the frame dt (0.033 s) shrinks the horizon to 0.66 s, making
        # the terminal cost dominate and causing aggressive overcorrection.
        result = self.solver.solve(
            e_lat_qp, e_heading, current_speed, last_delta_norm,
            kappa_horizon, v_target, v_max, self.params.dt,
            grade_rad=grade_rad,
            tire_cf=tire_result['tire_cf'],
            tire_cr=tire_result['tire_cr'],
            v_y_init=tire_result.get('v_y_estimate', 0.0),
            r_init=tire_result.get('yaw_rate_estimate', 0.0),
        )

        # Restore default max_iter after warmup solve
        if _in_warmup:
            try:
                self.solver._prob.update_settings(max_iter=200)
            except Exception:
                pass

        self._frames_since_reset += 1

        if not result['feasible'] or result['solve_time_ms'] > self.params.max_solve_time_ms:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.params.max_consecutive_failures:
                self._fallback_active = True
                logger.warning(
                    "MPC: fallback activated after %d consecutive failures",
                    self._consecutive_failures
                )
            result['steering_normalized'] = self._last_steering
        else:
            self._consecutive_failures = 0
            self._fallback_active = False
            self._last_steering = result['steering_normalized']

        return {
            'steering_normalized': result['steering_normalized'],
            'accel': result.get('accel', 0.0),
            'mpc_feasible': result['feasible'],
            'solve_time_ms': result.get('solve_time_ms', 0.0),
            'mpc_fallback_active': self._fallback_active,
            'mpc_consecutive_failures': self._consecutive_failures,
            'e_lat_raw': e_lat,            # raw (unfiltered) lateral error
            'e_lat_input': e_lat_qp,       # filtered value passed to QP
            'e_lat_ema': self._elat_ema,   # EMA state (for diagnostics)
            'elat_ramp_active': elat_ramp_active,  # True when rate limiter clamped step
            'e_heading_input': e_heading,
            'kappa_ref_used': float(kappa_corrected_horizon[0]) if kappa_corrected_horizon.size > 0 else (kappa_ref + self._bias_correction),
            'kappa_bias_correction': self._bias_correction,
            'kappa_bias_ema': self._bias_ema,
            'kappa_bias_guard_active': bias_guard_active,
            'kappa_bias_guard_limit': bias_guard_limit,
            'kappa_active_curve_preserve_ratio': active_curve_preserve_ratio,
            'kappa_active_curve_preserve_active': active_curve_preserve_active,
            'kappa_active_curve_preserve_weight': active_curve_preserve_weight,
            'kappa_active_mild_curve_authority_active': active_mild_curve_authority_active,
            'kappa_active_mild_curve_authority_weight': active_mild_curve_authority_weight,
            'kappa_active_mild_curve_authority_ratio': active_mild_curve_authority_ratio,
            'kappa_active_mild_curve_authority_reason': active_mild_curve_authority_reason,
            'kappa_active_mild_curve_authority_speed_weight': active_mild_curve_authority_speed_weight,
            'kappa_active_mild_curve_authority_curvature_weight': active_mild_curve_authority_curvature_weight,
            'kappa_active_mild_curve_authority_gate_weight': active_mild_curve_authority_gate_weight,
            'kappa_preview_used': bool(_preview_used),
            'kappa_preview_range': float(np.ptp(kappa_corrected_horizon)) if _preview_used else 0.0,
            'r_steer_rate_effective': result.get('r_steer_rate_effective', self.params.r_steer_rate),
            'leff_value': leff_result['leff_value'],
            'leff_theta': leff_result['leff_theta'],
            'leff_P': leff_result['leff_P'],
            'leff_updated': leff_result['leff_updated'],
            'leff_innovation': leff_result['leff_innovation'],
            'leff_update_count': leff_result['leff_update_count'],
            # Tire estimation (dynamic bicycle)
            'tire_cf': tire_result['tire_cf'],
            'tire_cr': tire_result['tire_cr'],
            'tire_ekf_innovation': tire_result['tire_ekf_innovation'],
            'tire_ekf_P_trace': tire_result['tire_ekf_P_trace'],
            'tire_slip_angle_front': tire_result['tire_slip_angle_front'],
            'tire_slip_angle_rear': tire_result['tire_slip_angle_rear'],
            'tire_understeer_gradient': tire_result['tire_understeer_gradient'],
            'dynamic_model_active': tire_result['dynamic_model_active'],
            'tire_ekf_update_count': tire_result['tire_ekf_update_count'],
            'v_y_estimate': tire_result['v_y_estimate'],
            'yaw_rate_estimate': tire_result['yaw_rate_estimate'],
            'yaw_rate_measurement': tire_result.get('yaw_rate_measurement', 0.0),
            'yaw_rate_source': tire_result.get('yaw_rate_source', 'derived'),
        }

    def reset(self):
        """Reset solver state (e.g., on regime switch or after e-stop)."""
        self.solver._warm_x = None
        self.solver._warm_y = None
        self._consecutive_failures = 0
        self._fallback_active = False
        self._last_steering = 0.0
        self._frames_since_reset = 0
        self._bias_ema = 0.0
        self._bias_correction = 0.0
        self._last_kappa_ref = 0.0
        self._elat_ema = 0.0
        self._prev_elat_ramp = 0.0
        # L_eff: reset to nominal (don't reset update_count — it's cumulative diagnostic)
        self._leff_theta = 1.0 / self.params.wheelbase_m
        self._leff_P = self.params.leff_initial_P
        self._leff_heading_prev = 0.0
        self._leff_steering_prev = 0.0
        self._leff_speed_prev = 0.0
        self._leff_update_count = 0
        # Tire EKF: reset to nominal
        self._tire_theta = np.array([self.params.tire_cf_nominal,
                                      self.params.tire_cr_nominal], dtype=float)
        self._tire_P = np.eye(2) * 1e6
        self._tire_ekf_update_count = 0
        self._tire_ekf_divergent_count = 0
        self._v_y_est = 0.0
        self._yaw_rate_est = 0.0
        self._e_heading_prev_tire = 0.0

    @property
    def leff(self) -> float:
        """Current effective wheelbase estimate (meters)."""
        if not self.params.leff_estimation_enabled:
            return self.params.wheelbase_m
        return 1.0 / self._leff_theta

    def _update_leff(self, e_heading: float, current_speed: float,
                     last_delta_norm: float, kappa_ref: float, dt: float) -> dict:
        """
        RLS update for effective wheelbase from heading prediction error.

        Model: e_heading[k] = e_heading[k-1] + θ·h[k-1] − κ·v·dt
        where θ = 1/L_eff and h = v·δ·δ_max·dt (regressor).

        Innovation: ε = e_heading[k] − predicted_e_heading[k]
        RLS updates θ to minimize ε² with exponential forgetting.
        """
        leff_updated = False
        leff_innovation = 0.0

        if not self.params.leff_estimation_enabled:
            return {
                'leff_value': self.params.wheelbase_m,
                'leff_theta': self._leff_theta,
                'leff_P': self._leff_P,
                'leff_updated': False,
                'leff_innovation': 0.0,
                'leff_update_count': self._leff_update_count,
            }

        v_prev = self._leff_speed_prev
        delta_prev = self._leff_steering_prev

        # Regressor: h = v · δ_norm · δ_max · dt
        h = v_prev * delta_prev * self._max_steer_at_speed(v_prev) * dt

        # Predicted heading (using current θ estimate)
        predicted_heading = (
            self._leff_heading_prev
            + self._leff_theta * h
            - kappa_ref * v_prev * dt
        )

        # Innovation (prediction error)
        leff_innovation = e_heading - predicted_heading

        # Gate: only update if excitation is sufficient and speed is adequate
        if (abs(h) >= self.params.leff_min_excitation
                and v_prev >= self.params.leff_min_speed_mps):

            lam = self.params.leff_forgetting_factor

            # Scalar RLS update: K = P·h / (λ + h·P·h)
            Ph = self._leff_P * h
            denom = lam + h * Ph
            if abs(denom) > 1e-12:
                K = Ph / denom
                self._leff_theta += K * leff_innovation
                self._leff_P = (self._leff_P - K * h * self._leff_P) / lam

                # Clamp θ to safety bounds (θ = 1/L, so bounds are inverted)
                theta_min = 1.0 / self.params.leff_bounds_max  # smallest θ = longest L
                theta_max = 1.0 / self.params.leff_bounds_min  # largest θ = shortest L
                self._leff_theta = max(theta_min, min(theta_max, self._leff_theta))

                # Clamp P to prevent numerical blowup
                self._leff_P = max(0.01, min(1e6, self._leff_P))

                leff_updated = True
                self._leff_update_count += 1

        # Store current frame for next iteration
        self._leff_heading_prev = e_heading
        self._leff_steering_prev = last_delta_norm
        self._leff_speed_prev = current_speed

        return {
            'leff_value': self.leff,
            'leff_theta': self._leff_theta,
            'leff_P': self._leff_P,
            'leff_updated': leff_updated,
            'leff_innovation': leff_innovation,
            'leff_update_count': self._leff_update_count,
        }

    def _compute_understeer_gradient(self, C_f: float, C_r: float) -> float:
        """Compute understeer gradient K_us = (m/L)·(l_r/C_f − l_f/C_r)."""
        L = self.params.vehicle_lf_m + self.params.vehicle_lr_m
        if abs(C_f) < 1.0 or abs(C_r) < 1.0 or L < 0.1:
            return 0.0
        return (self.params.vehicle_mass_kg / L) * (
            self.params.vehicle_lr_m / C_f - self.params.vehicle_lf_m / C_r
        )

    def _update_tire_ekf(self, e_heading: float, current_speed: float,
                          last_delta_norm: float, kappa_ref: float,
                          dt: float,
                          imu_yaw_rate: Optional[float] = None) -> dict:
        """
        EKF update for tire cornering stiffness [C_f, C_r].

        Measurement: IMU gyroscope yaw rate (preferred) or derived from heading rate.
        Model: dynamic bicycle yaw dynamics.
        Only updates when dynamic_model_enabled AND tire_ekf_enabled.
        """
        C_f, C_r = self._tire_theta
        _nominal = {
            'tire_cf': float(C_f),
            'tire_cr': float(C_r),
            'tire_ekf_innovation': 0.0,
            'tire_ekf_P_trace': float(np.trace(self._tire_P)),
            'tire_slip_angle_front': 0.0,
            'tire_slip_angle_rear': 0.0,
            'tire_understeer_gradient': self._compute_understeer_gradient(C_f, C_r),
            'dynamic_model_active': self.params.dynamic_model_enabled,
            'tire_ekf_update_count': self._tire_ekf_update_count,
            'v_y_estimate': self._v_y_est,
            'yaw_rate_estimate': self._yaw_rate_est,
            'yaw_rate_measurement': 0.0,
            'yaw_rate_source': 'imu' if (self.params.tire_ekf_use_imu_yaw_rate and imu_yaw_rate is not None) else 'derived',
        }

        if not self.params.dynamic_model_enabled:
            self._e_heading_prev_tire = e_heading
            return _nominal

        if dt <= 0.0:
            self._e_heading_prev_tire = e_heading
            return _nominal

        v_x = max(current_speed, 3.0)  # clamp to avoid singularity
        l_f = self.params.vehicle_lf_m
        l_r = self.params.vehicle_lr_m
        m = self.params.vehicle_mass_kg
        Iz = self.params.vehicle_iz_kgm2

        # Measurement model: prefer IMU gyroscope when available (industry standard)
        if self.params.tire_ekf_use_imu_yaw_rate and imu_yaw_rate is not None:
            r_meas = imu_yaw_rate
            R_meas = self.params.tire_ekf_imu_yaw_rate_r
        else:
            # Fallback: derive from heading error finite differences
            r_meas = (e_heading - self._e_heading_prev_tire) / dt + kappa_ref * v_x
            R_meas = self.params.tire_ekf_measurement_noise

        # Compute slip angles using current estimates
        delta_rad = last_delta_norm * self._max_steer_at_speed(v_x)
        alpha_f = delta_rad - (self._v_y_est + l_f * self._yaw_rate_est) / v_x
        alpha_r = -(self._v_y_est - l_r * self._yaw_rate_est) / v_x

        # Predict yaw rate from current theta
        r_pred = self._yaw_rate_est + (
            l_f * C_f * alpha_f - l_r * C_r * alpha_r
        ) / Iz * dt

        # Innovation
        innovation = r_meas - r_pred

        # --- C_f/C_r EKF update (only when tire_ekf_enabled) ---
        ekf_updated = False
        if self.params.tire_ekf_enabled:
            if (current_speed >= self.params.tire_ekf_min_speed_mps
                    and abs(alpha_f) < self.params.tire_ekf_slip_saturation_rad
                    and abs(alpha_r) < self.params.tire_ekf_slip_saturation_rad
                    and self._tire_ekf_divergent_count < 10):

                # Process noise
                Q = np.eye(2) * self.params.tire_ekf_process_noise * dt
                R = R_meas

                # Predict covariance
                P_pred = self._tire_P + Q

                # Jacobian H = dr_pred/d[C_f, C_r]
                H = np.array([
                    l_f * alpha_f * dt / Iz,
                    -l_r * alpha_r * dt / Iz,
                ])

                # Innovation covariance
                S = float(H @ P_pred @ H) + R
                if abs(S) > 1e-12:
                    K = P_pred @ H / S  # Kalman gain (2,)

                    # Update
                    self._tire_theta = self._tire_theta + K * innovation
                    self._tire_P = (np.eye(2) - np.outer(K, H)) @ P_pred

                    # Clamp to safety bounds
                    self._tire_theta[0] = np.clip(
                        self._tire_theta[0],
                        self.params.tire_ekf_cf_min,
                        self.params.tire_ekf_cf_max
                    )
                    self._tire_theta[1] = np.clip(
                        self._tire_theta[1],
                        self.params.tire_ekf_cr_min,
                        self.params.tire_ekf_cr_max
                    )

                    # Clamp P to prevent blowup
                    self._tire_P = np.clip(self._tire_P, -1e8, 1e8)
                    np.fill_diagonal(self._tire_P,
                                     np.clip(np.diag(self._tire_P), 0.01, 1e8))

                    ekf_updated = True
                    self._tire_ekf_update_count += 1

            # Divergence tracking
            if abs(innovation) > self.params.tire_ekf_innovation_divergence_threshold:
                self._tire_ekf_divergent_count += 1
            else:
                self._tire_ekf_divergent_count = 0

        # --- v_y and yaw rate estimation (ALWAYS runs when dynamic model active) ---
        # These are needed for correct initial state in the QP, regardless of
        # whether C_f/C_r are being estimated online.
        C_f_new, C_r_new = self._tire_theta
        F_yf = C_f_new * alpha_f
        F_yr = C_r_new * alpha_r
        v_y_dot = (F_yf + F_yr) / m - v_x * self._yaw_rate_est
        self._v_y_est = 0.95 * (self._v_y_est + v_y_dot * dt) + 0.05 * 0.0

        # Update yaw rate estimate (blend measurement and model)
        self._yaw_rate_est = 0.7 * r_meas + 0.3 * r_pred

        # Store heading for next frame
        self._e_heading_prev_tire = e_heading

        return {
            'tire_cf': float(self._tire_theta[0]),
            'tire_cr': float(self._tire_theta[1]),
            'tire_ekf_innovation': float(innovation),
            'tire_ekf_P_trace': float(np.trace(self._tire_P)),
            'tire_slip_angle_front': float(alpha_f),
            'tire_slip_angle_rear': float(alpha_r),
            'tire_understeer_gradient': self._compute_understeer_gradient(
                self._tire_theta[0], self._tire_theta[1]),
            'dynamic_model_active': True,
            'tire_ekf_update_count': self._tire_ekf_update_count,
            'v_y_estimate': float(self._v_y_est),
            'yaw_rate_estimate': float(self._yaw_rate_est),
            'yaw_rate_measurement': float(r_meas),
            'yaw_rate_source': 'imu' if (self.params.tire_ekf_use_imu_yaw_rate and imu_yaw_rate is not None) else 'derived',
        }
