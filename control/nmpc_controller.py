"""
Nonlinear MPC (NMPC) lateral controller for high-speed lane-keeping.

Uses the exact nonlinear bicycle kinematic model in Frenet error coordinates,
solved via scipy SLSQP (sequential least-squares programming, single-shooting NLP).

Activated by RegimeSelector when speed > lmpc_max_speed_mps (20 m/s) or heading
error exceeds lmpc_max_heading_error_rad (0.25 rad).  On solver failure, sets
nmpc_fallback_active=True so the caller can dispatch to LMPC for that frame.

Vs LMPC (mpc_controller.py):
  LMPC  — sin(e)≈e, tan(δ)≈δ  — valid below 20 m/s and |e_heading| < 0.25 rad
  NMPC  — exact sin/tan         — valid at any speed and large heading errors
  Solver: LMPC uses OSQP (QP, ~1 ms), NMPC uses scipy SLSQP (NLP, ≤20 ms budget)

Nonlinear bicycle dynamics (Frenet error frame, discrete, exact):
  L_eff           = L + K_us · v²   (effective wheelbase with understeer gradient)
  e_lat[k+1]     = e_lat[k] + v[k] · sin(e_heading[k]) · dt
  e_heading[k+1] = e_heading[k] + (v[k]/L_eff) · tan(δ_norm[k] · δ_max) · dt − κ[k] · v[k] · dt
  v[k+1]         = v[k] + (a[k] + gravity_offset) · dt

State:  x = [e_lat, e_heading, v]
Input:  u = [δ_norm, a]   (δ_norm ∈ [-1,1], a ∈ [-max_decel, max_accel])

Decision variables (single-shooting, N steps):
  U = [δ₀, a₀, δ₁, a₁, ..., δ_{N-1}, a_{N-1}]   shape (2N,)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, fields
from typing import Optional

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NMPCParams — tuning parameters from YAML
# ---------------------------------------------------------------------------

@dataclass
class NMPCParams:
    """NMPC tuning parameters.  Loaded from config trajectory.nmpc section."""
    horizon: int = 20
    dt: float = 0.1                     # MPC prediction timestep (s) — NOT the frame dt
    wheelbase_m: float = 2.5
    max_steer_rad: float = 0.5236       # 30°

    # State cost weights
    q_lat: float = 2.0                  # lateral tracking
    q_heading: float = 5.0             # heading tracking
    q_speed: float = 1.0               # speed tracking

    # Terminal cost multipliers
    q_lat_terminal_scale: float = 5.0
    q_heading_terminal_scale: float = 3.0

    # Input cost weights
    # r_steer near-zero for same reason as LMPC (biasing QP toward δ=0 causes understeering).
    r_steer: float = 1e-4               # steering magnitude (numerical stability only)
    r_accel: float = 0.05              # accel magnitude
    r_steer_rate: float = 1.5          # steering rate — PRIMARY COMFORT KNOB
    r_steer_ff_mag: float = 0.0        # feedforward magnitude penalty (0=disabled)

    # Constraints
    delta_rate_max: float = 0.5         # max Δδ_norm per step
    max_accel: float = 1.2             # m/s²
    max_decel: float = 2.4             # m/s²
    v_min: float = 1.0                 # m/s
    v_max: float = 30.0                # m/s

    # Grade compensation (same semantics as LMPC)
    grade_clamp_rad: float = 0.15      # max grade input (~8.5°)
    grade_ff_gain: float = 1.8         # calibrated for Unity WheelCollider physics

    # 2DOF feedforward alignment: penalize Δ(δ − δ_ff)² instead of Δδ²
    # Eliminates rate-penalty bias against holding sustained curve steer angle.
    # On straights (κ=0): Δδ_ff=0 everywhere → correction is zero.
    ff_alignment_enabled: bool = True

    # Solve budget and fallback
    max_solve_ms: float = 20.0          # wall-clock budget; exceeded → failure
    max_iter: int = 50                  # SLSQP iteration limit
    max_consecutive_failures: int = 3

    # Speed-adaptive dt: scale prediction dt so the horizon covers a fixed distance.
    # dt_eff = clip(horizon_distance_m / (horizon * v), dt_min, dt_max)
    # At 4 m/s: dt=0.375 → 30m horizon. At 15 m/s: dt=0.1 → 30m horizon.
    horizon_distance_m: float = 50.0    # target lookahead distance (m)
    dt_min: float = 0.1                 # min prediction dt (high speed)
    dt_max: float = 0.75                # max prediction dt (low speed)

    # Understeer gradient: L_eff(v) = wheelbase_m + understeer_gradient * v²
    # Accounts for tire slip at speed; from sysid (tools/analyze/sysid_dynamic_model.py).
    # 0.0 = use geometric wheelbase (kinematic model).
    understeer_gradient: float = 0.025

    # Startup warmup: first warmup_frames after activation use last_delta=0
    # to allow SLSQP to find a feasible point when cold-starting from PP.
    warmup_frames: int = 5

    # On failure: fallback to LMPC (True) vs hold last steering (False)
    fallback_lmpc: bool = True

    # Cold-start sign-preserving seed: when no warm-start is available, the
    # default x0=zeros + curvature-feedforward puts SLSQP at the saddle point
    # of a near-symmetric cost surface for non-zero e_lat on straight roads.
    # BLAS arithmetic noise then determines which basin SLSQP descends into
    # (Mac Accelerate vs Linux OpenBLAS divergence). A small proportional
    # bias on x0[0] toward -sign(e_lat) breaks the symmetry deterministically.
    # Set gain=0.0 to disable. max_frac caps the seed to a fraction of
    # max_steer_rad so it doesn't dominate SLSQP optimization.
    cold_start_e_lat_seed_gain: float = 0.5
    cold_start_e_lat_seed_max_frac: float = 0.10

    @classmethod
    def from_config(cls, cfg: dict) -> "NMPCParams":
        """Load from config dict: cfg['trajectory']['nmpc']['nmpc_<field>']."""
        nmpc_cfg = cfg.get("trajectory", {}).get("nmpc", {})
        kwargs: dict = {}
        for f in fields(cls):
            yaml_key = f"nmpc_{f.name}"
            if yaml_key in nmpc_cfg:
                kwargs[f.name] = type(f.default)(nmpc_cfg[yaml_key])
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# NMPCSolver — core NLP builder and frame-by-frame SLSQP solver
# ---------------------------------------------------------------------------

class SLSQPNMPCSolver:
    """
    scipy SLSQP solver for the nonlinear bicycle MPC (legacy backend).

    Single-shooting NLP: decision variables are controls U = [δ₀,a₀,...,δ_{N-1},a_{N-1}].
    State trajectory is computed as a forward rollout inside the cost function;
    rate constraints between consecutive δ values are explicit LinearConstraints.

    Warm start: previous solution shifted by one step (receding horizon).
    Cold start: feedforward Ackermann angles as initial guess.

    Known limitation: SLSQP hits 50-iteration limit on 64% of tight curve frames
    (R40-R50 at 4 m/s) due to BFGS Hessian approximation. See CasADiNMPCSolver.
    """

    def __init__(self, params: NMPCParams):
        self.p = params
        self._warm_u: Optional[np.ndarray] = None

    @staticmethod
    def _feedforward_delta_norm(kappa: float, wheelbase_m: float,
                                 max_steer_rad: float) -> float:
        """Bicycle-model kinematic feedforward steering, normalized to [-1, 1].

        δ = arctan(L · κ) / δ_max.  Returns 0 for κ=0 (straight road).
        """
        return math.atan(wheelbase_m * kappa) / max_steer_rad

    def _rollout(self, u_flat: np.ndarray,
                 e_lat0: float, e_heading0: float, v0: float,
                 kappa: np.ndarray,
                 gravity_offset: float, v_max_eff: float) -> np.ndarray:
        """Forward-simulate nonlinear bicycle dynamics for N steps.

        Returns state array of shape (N+1, 3): columns [e_lat, e_heading, v].
        Uses exact sin/tan — valid at high speed and large heading errors.
        """
        p = self.p
        N = p.horizon
        dt = p.dt
        L = p.wheelbase_m
        dmax = p.max_steer_rad

        states = np.empty((N + 1, 3))
        states[0, 0] = e_lat0
        states[0, 1] = e_heading0
        states[0, 2] = v0

        K_us = p.understeer_gradient

        for k in range(N):
            e_l = states[k, 0]
            e_h = states[k, 1]
            v = states[k, 2]
            delta_norm = u_flat[2 * k]
            a = u_flat[2 * k + 1]
            kk = kappa[k]

            delta = delta_norm * dmax
            L_eff = L + K_us * v * v

            states[k + 1, 0] = e_l + v * math.sin(e_h) * dt
            states[k + 1, 1] = (
                e_h + (v / L_eff) * math.tan(delta) * dt - kk * v * dt
            )
            v_next = v + (a + gravity_offset) * dt
            states[k + 1, 2] = max(p.v_min, min(v_max_eff, v_next))

        return states

    def _cost_and_jac(self, u_flat: np.ndarray,
                       e_lat0: float, e_heading0: float, v0: float,
                       kappa: np.ndarray, v_target: float,
                       last_delta_norm: float,
                       gravity_offset: float, v_max_eff: float,
                       delta_ff: Optional[np.ndarray]):
        """Compute total cost and analytical gradient via adjoint backpropagation.

        Returns (cost: float, grad: np.ndarray shape (2N,)).

        Adjoint method: one forward pass builds the state trajectory; one backward
        pass accumulates ∂C/∂u using the Jacobians of the nonlinear dynamics.

        Forward dynamics Jacobians at step k:
          A_k = ∂x_{k+1}/∂x_k  (3×3):
            [1,  v·cos(eₕ)·dt,  sin(eₕ)·dt               ]
            [0,  1,             (1/L)·tan(δ)·dt − κ·dt    ]
            [0,  0,             1                           ]

          B_k = ∂x_{k+1}/∂u_k  (3×2):
            [0,   0                                            ]
            [(v/L)·(δ_max/cos²(δ))·dt,  0                    ]
            [0,   dt                                           ]

        Note: A_k[1, 2] = ∂e_heading_{k+1}/∂v_k = (1/L)·tan(δ)·dt − κ·dt  (NOT v/L).

        Backward adjoint update:
          λ_N    = ∂c_T/∂x_N
          λ_k    = A_k^T · λ_{k+1} + ∂c_k/∂x_k
          ∂C/∂u_k += B_k^T · λ_{k+1} + ∂c_k/∂u_k  (direct + dynamics)
        """
        p = self.p
        N = p.horizon
        dt = p.dt
        L = p.wheelbase_m
        dmax = p.max_steer_rad

        K_us = p.understeer_gradient

        # ── Forward pass: simulate dynamics and cache per-step quantities ──
        states = np.empty((N + 1, 3))
        states[0] = [e_lat0, e_heading0, v0]

        sin_eh = np.empty(N)       # sin(e_heading[k])
        cos_eh = np.empty(N)       # cos(e_heading[k])
        tan_d = np.empty(N)        # tan(delta[k])
        cos2_d = np.empty(N)       # cos²(delta[k])
        L_eff_arr = np.empty(N)    # effective wheelbase at each step
        v_clamped = np.zeros(N)    # True when v_{k+1} was clamped

        for k in range(N):
            e_l, e_h, v = states[k]
            delta_norm = u_flat[2 * k]
            a = u_flat[2 * k + 1]
            kk = kappa[k]
            delta = delta_norm * dmax
            cd = math.cos(delta)

            sin_eh[k] = math.sin(e_h)
            cos_eh[k] = math.cos(e_h)
            tan_d[k] = math.tan(delta)
            cos2_d[k] = cd * cd
            L_eff = L + K_us * v * v
            L_eff_arr[k] = L_eff

            states[k + 1, 0] = e_l + v * sin_eh[k] * dt
            states[k + 1, 1] = e_h + (v / L_eff) * tan_d[k] * dt - kk * v * dt
            v_next = v + (a + gravity_offset) * dt
            if v_next <= p.v_min or v_next >= v_max_eff:
                v_clamped[k] = 1.0   # gradient through clamp = 0
            states[k + 1, 2] = max(p.v_min, min(v_max_eff, v_next))

        # ── Cost accumulation and direct ∂c/∂u, ∂c/∂x ──────────────────────
        cost = 0.0

        # Stage costs: [q_lat·e_l², q_heading·e_h², q_speed·(v-vt)²]
        dc_dx = np.zeros((N + 1, 3))
        # dc_du[k] = direct ∂(c_k)/∂u_k (magnitude and rate terms)
        dc_du = np.zeros(2 * N)

        # Rate residuals: r_k = (δ[k] - δ[k-1]) − Δff_k
        # We need r_k and its gradient w.r.t. δ[k] and δ[k-1].
        r_rate = np.empty(N)

        for k in range(N):
            e_l, e_h, v = states[k]
            cost += p.q_lat * e_l * e_l + p.q_heading * e_h * e_h
            cost += p.q_speed * (v - v_target) ** 2
            dc_dx[k, 0] += 2.0 * p.q_lat * e_l
            dc_dx[k, 1] += 2.0 * p.q_heading * e_h
            dc_dx[k, 2] += 2.0 * p.q_speed * (v - v_target)

            cur_d = u_flat[2 * k]
            a_k = u_flat[2 * k + 1]
            cost += p.r_steer * cur_d * cur_d + p.r_accel * a_k * a_k
            dc_du[2 * k] += 2.0 * p.r_steer * cur_d
            dc_du[2 * k + 1] += 2.0 * p.r_accel * a_k

            prev_d = last_delta_norm if k == 0 else u_flat[2 * (k - 1)]
            d_rate = cur_d - prev_d
            if p.ff_alignment_enabled and delta_ff is not None:
                ff_cur = delta_ff[k]
                ff_prev = delta_ff[0] if k == 0 else delta_ff[k - 1]
                d_rate -= (ff_cur - ff_prev)
            r_rate[k] = d_rate
            cost += p.r_steer_rate * d_rate * d_rate

            # Feedforward magnitude penalty: anchor δ to kinematic feedforward
            if p.r_steer_ff_mag > 0.0 and delta_ff is not None:
                ff_err = cur_d - delta_ff[k]
                cost += p.r_steer_ff_mag * ff_err * ff_err
                dc_du[2 * k] += 2.0 * p.r_steer_ff_mag * ff_err

        # Terminal cost
        e_l_T, e_h_T, v_T = states[N]
        cost += (p.q_lat * p.q_lat_terminal_scale * e_l_T * e_l_T
                 + p.q_heading * p.q_heading_terminal_scale * e_h_T * e_h_T
                 + p.q_speed * (v_T - v_target) ** 2)
        dc_dx[N, 0] = 2.0 * p.q_lat * p.q_lat_terminal_scale * e_l_T
        dc_dx[N, 1] = 2.0 * p.q_heading * p.q_heading_terminal_scale * e_h_T
        dc_dx[N, 2] = 2.0 * p.q_speed * (v_T - v_target)

        # Rate gradient contributions (coupling δ[k] and δ[k-1])
        for k in range(N):
            rr = r_rate[k]
            dc_du[2 * k] += 2.0 * p.r_steer_rate * rr       # ∂(r_k²)/∂δ[k]
            if k > 0:
                dc_du[2 * (k - 1)] -= 2.0 * p.r_steer_rate * rr  # ∂(r_k²)/∂δ[k-1]

        # ── Backward pass: adjoint method ────────────────────────────────────
        # λ_k = A_k^T · λ_{k+1} + ∂c_k/∂x_k
        # grad[u_k] += B_k^T · λ_{k+1}  (dynamics contribution to control gradient)
        grad = dc_du.copy()
        lam = dc_dx[N].copy()  # adjoint at terminal step

        for k in range(N - 1, -1, -1):
            e_l, e_h, v = states[k]
            kk = kappa[k]
            Lk = L_eff_arr[k]

            # B_k^T · λ: control gradient from dynamics
            # B_k[1,0] = (v/L_eff)*(dmax/cos²δ)*dt
            steer_gain = (v / Lk) * (dmax / cos2_d[k]) * dt
            grad[2 * k] += steer_gain * lam[1]
            if not v_clamped[k]:
                grad[2 * k + 1] += dt * lam[2]

            # A_k^T · λ + ∂c_k/∂x_k  (backward adjoint step)
            # A_k[1, 2] = ∂e_heading_{k+1}/∂v_k
            #   = [(L - K_us·v²) / L_eff²] · tan(δ) · dt − κ · dt
            a12 = ((L - K_us * v * v) / (Lk * Lk)) * tan_d[k] * dt - kk * dt
            new_lam_0 = lam[0]
            new_lam_1 = v * cos_eh[k] * dt * lam[0] + lam[1]
            new_lam_2 = (sin_eh[k] * dt * lam[0]
                         + a12 * lam[1]
                         + (0.0 if v_clamped[k] else lam[2]))
            lam = np.array([
                new_lam_0 + dc_dx[k, 0],
                new_lam_1 + dc_dx[k, 1],
                new_lam_2 + dc_dx[k, 2],
            ])

        return cost, grad

    def solve(self, e_lat: float, e_heading: float, v: float,
               last_delta_norm: float, kappa_ref_horizon: np.ndarray,
               v_target: float, v_max: float, dt: float,
               grade_rad: float = 0.0,
               prediction_dt: Optional[float] = None) -> dict:
        """
        Solve NMPC NLP for one frame.

        Args:
            e_lat: lateral error from reference path (m)
            e_heading: heading error from reference heading (rad)
            v: current speed (m/s)
            last_delta_norm: previous normalized steering [-1, 1]
            kappa_ref_horizon: array of N reference curvatures (1/m) from map
            v_target: desired speed (m/s)
            v_max: track speed limit (m/s) — combined with p.v_max
            dt: frame timestep (s) — NOT used for prediction
            grade_rad: road grade in radians (positive = uphill)
            prediction_dt: override for prediction step size (s). When set,
                overrides p.dt for this solve — used by speed-adaptive horizon.

        Returns:
            dict: steering_normalized, accel, feasible, solve_time_ms,
                  predicted_trajectory (N+1, 3), nmpc_cost, nmpc_iterations,
                  slsqp_status, prediction_dt_used
        """
        t0 = time.perf_counter()
        p = self.p
        N = p.horizon

        # Speed-adaptive dt: temporarily override p.dt for this solve
        original_dt = p.dt
        if prediction_dt is not None:
            p.dt = prediction_dt

        # Curvature horizon: extend or truncate to exactly N steps
        kappa = np.zeros(N)
        if kappa_ref_horizon is not None and len(kappa_ref_horizon) > 0:
            n_copy = min(len(kappa_ref_horizon), N)
            kappa[:n_copy] = kappa_ref_horizon[:n_copy]
            kappa[n_copy:] = kappa[n_copy - 1]

        v_safe = max(v, 0.5)
        v_max_eff = min(v_max, p.v_max)
        grade_clamped = max(-p.grade_clamp_rad, min(p.grade_clamp_rad, grade_rad))
        gravity_offset = -9.81 * math.sin(grade_clamped) * p.grade_ff_gain

        # Precompute feedforward angles for ff_alignment (or None if disabled)
        delta_ff: Optional[np.ndarray]
        if p.ff_alignment_enabled:
            delta_ff = np.array([
                self._feedforward_delta_norm(kappa[k], p.wheelbase_m, p.max_steer_rad)
                for k in range(N)
            ])
        else:
            delta_ff = None

        # ── Warm start ───────────────────────────────────────────────────────
        # Receding-horizon shift: drop first step, repeat last.
        if self._warm_u is not None and len(self._warm_u) == 2 * N:
            x0 = np.empty(2 * N)
            x0[:2 * (N - 1)] = self._warm_u[2:]
            x0[2 * (N - 1):] = self._warm_u[2 * (N - 1):]
        else:
            # Cold start: use kinematic feedforward angles, zero accel
            x0 = np.zeros(2 * N)
            for k in range(N):
                x0[2 * k] = self._feedforward_delta_norm(
                    kappa[k], p.wheelbase_m, p.max_steer_rad
                )
            # Sign-preserving e_lat seed: bias x0[0] toward -sign(e_lat) so
            # SLSQP doesn't start at a near-saddle on straight roads with
            # non-zero e_lat. Without this, BLAS arithmetic noise tips the
            # basin choice (CI Linux OpenBLAS vs local Mac Accelerate).
            if p.cold_start_e_lat_seed_gain > 0.0:
                seed_max = p.cold_start_e_lat_seed_max_frac * p.max_steer_rad
                x0[0] += float(np.clip(
                    -e_lat * p.cold_start_e_lat_seed_gain,
                    -seed_max, seed_max,
                ))

        # ── Bounds ───────────────────────────────────────────────────────────
        # First δ is also rate-limited from last_delta_norm (hard box).
        # Remaining δ values are rate-constrained via LinearConstraint below.
        # Accel bounds are expanded by |gravity_offset| for grade compensation,
        # matching the LMPC convention so grade-limited frames don't saturate.
        abs_g = abs(gravity_offset)
        lb = np.empty(2 * N)
        ub = np.empty(2 * N)
        for k in range(N):
            if k == 0:
                lb[0] = max(-1.0, last_delta_norm - p.delta_rate_max)
                ub[0] = min(1.0, last_delta_norm + p.delta_rate_max)
            else:
                lb[2 * k] = -1.0
                ub[2 * k] = 1.0
            lb[2 * k + 1] = -(p.max_decel + abs_g)
            ub[2 * k + 1] = p.max_accel + abs_g
        bounds = Bounds(lb, ub)

        # ── Rate constraints for k = 1..N-1 ─────────────────────────────────
        # -rate_max ≤ δ[k] − δ[k-1] ≤ rate_max  (linear in U)
        # Build constraint matrix A_rate of shape (N-1, 2N).
        constraints: list = []
        if N > 1:
            A_rate = np.zeros((N - 1, 2 * N))
            for i in range(N - 1):
                k = i + 1
                A_rate[i, 2 * k] = 1.0
                A_rate[i, 2 * (k - 1)] = -1.0
            constraints.append(LinearConstraint(
                A_rate,
                lb=-p.delta_rate_max * np.ones(N - 1),
                ub=p.delta_rate_max * np.ones(N - 1),
            ))

        # ── Solve ─────────────────────────────────────────────────────────────
        # jac=True: _cost_and_jac returns (cost, gradient) together.
        # This eliminates scipy's finite-difference gradient estimation (2N = 40
        # extra function calls per iteration), giving ~40× speedup vs jac=None.
        try:
            opt = minimize(
                self._cost_and_jac,
                x0,
                args=(e_lat, e_heading, v_safe, kappa, v_target,
                      last_delta_norm, gravity_offset, v_max_eff, delta_ff),
                method='SLSQP',
                jac=True,
                bounds=bounds,
                constraints=constraints,
                options={
                    'maxiter': p.max_iter,
                    # Relax ftol at low speed where the cost landscape is
                    # nearly flat (steering barely changes e_lat when v is
                    # small).  At v>=8 m/s: 1e-6 (no change).  At v=3 m/s:
                    # ~6e-5.  Prevents hitting iteration limit on straights.
                    'ftol': max(1e-6, 1e-4 * (1.0 - min(v_safe / 8.0, 1.0))),
                    'disp': False,
                },
            )
            solve_ms = (time.perf_counter() - t0) * 1000.0

            # status=0: success; status=9: iteration limit but usable solution
            feasible = opt.success or opt.status == 9

            if feasible:
                u_opt = opt.x
                delta_norm_0 = float(np.clip(u_opt[0], -1.0, 1.0))
                accel_0 = float(np.clip(u_opt[1], -p.max_decel, p.max_accel))
                self._warm_u = u_opt.copy()
                predicted = self._rollout(u_opt, e_lat, e_heading, v_safe,
                                          kappa, gravity_offset, v_max_eff)
                result = {
                    'steering_normalized': delta_norm_0,
                    'accel': accel_0,
                    'feasible': True,
                    'solve_time_ms': solve_ms,
                    'predicted_trajectory': predicted,
                    'nmpc_cost': float(opt.fun),
                    'nmpc_iterations': int(opt.nit),
                    'slsqp_status': int(opt.status),
                    'prediction_dt_used': p.dt,
                }
                p.dt = original_dt
                return result

            # Infeasible / failed
            p.dt = original_dt
            return {
                'steering_normalized': 0.0,
                'accel': 0.0,
                'feasible': False,
                'solve_time_ms': solve_ms,
                'predicted_trajectory': np.zeros((N + 1, 3)),
                'nmpc_cost': float('nan'),
                'nmpc_iterations': int(opt.nit),
                'slsqp_status': int(opt.status),
                'prediction_dt_used': original_dt if prediction_dt is None else prediction_dt,
            }

        except Exception as exc:
            p.dt = original_dt
            solve_ms = (time.perf_counter() - t0) * 1000.0
            logger.error("NMPCSolver: unexpected exception: %s", exc)
            return {
                'steering_normalized': 0.0,
                'accel': 0.0,
                'feasible': False,
                'solve_time_ms': solve_ms,
                'predicted_trajectory': np.zeros((N + 1, 3)),
                'nmpc_cost': float('nan'),
                'nmpc_iterations': 0,
                'slsqp_status': -1,
                'prediction_dt_used': original_dt if prediction_dt is None else prediction_dt,
            }


# ---------------------------------------------------------------------------
# CasADiNMPCSolver — IPOPT backend with exact Hessians (industry standard)
# ---------------------------------------------------------------------------

try:
    import casadi
    _CASADI_AVAILABLE = True
except ImportError:
    _CASADI_AVAILABLE = False


class CasADiNMPCSolver:
    """CasADi + IPOPT solver for the nonlinear bicycle MPC.

    Same dynamics model and cost function as SLSQPNMPCSolver, but uses IPOPT
    (interior-point method with exact second-order derivatives via CasADi AD).

    IPOPT exploits the banded KKT structure of MPC and converges in 5-15
    iterations where SLSQP hits the 50-iteration limit on tight curves.
    """

    def __init__(self, params: NMPCParams):
        self.p = params
        self._warm_u: Optional[np.ndarray] = None
        self._cached_dt: Optional[float] = None
        self._solver = None
        self._rebuild_solver(params.dt)

    @staticmethod
    def _feedforward_delta_norm(kappa: float, wheelbase_m: float,
                                 max_steer_rad: float) -> float:
        """Bicycle-model kinematic feedforward steering, normalized to [-1, 1]."""
        return math.atan(wheelbase_m * kappa) / max_steer_rad

    def _rebuild_solver(self, dt: float) -> None:
        """Build (or rebuild) the CasADi NLP for a given prediction dt."""
        if not _CASADI_AVAILABLE:
            raise ImportError("CasADi not installed")

        p = self.p
        N = p.horizon

        # Decision variables: [delta_norm_0, ..., delta_norm_{N-1}, accel_0, ..., accel_{N-1}]
        # Using separate vectors (not interleaved) for cleaner CasADi formulation
        delta_sym = casadi.SX.sym('delta', N)
        accel_sym = casadi.SX.sym('accel', N)

        # Parameters (set each frame)
        e_lat0 = casadi.SX.sym('e_lat0')
        e_head0 = casadi.SX.sym('e_head0')
        v0 = casadi.SX.sym('v0')
        kappa_sym = casadi.SX.sym('kappa', N)
        vt = casadi.SX.sym('v_target')
        last_d = casadi.SX.sym('last_delta')
        g_off = casadi.SX.sym('gravity_offset')
        v_mx = casadi.SX.sym('v_max_eff')
        ff_sym = casadi.SX.sym('delta_ff', N)

        # Forward dynamics (symbolic)
        e_l = e_lat0
        e_h = e_head0
        v = v0
        cost = casadi.SX(0)

        for k in range(N):
            # Stage cost
            cost += p.q_lat * e_l**2 + p.q_heading * e_h**2
            cost += p.q_speed * (v - vt)**2
            cost += p.r_steer * delta_sym[k]**2
            cost += p.r_accel * accel_sym[k]**2

            # Rate cost with ff_alignment
            prev_d = casadi.if_else(k == 0, last_d, delta_sym[k - 1]) if k > 0 else last_d
            if p.ff_alignment_enabled:
                ff_prev = ff_sym[0] if k == 0 else ff_sym[k - 1]
                d_rate = (delta_sym[k] - prev_d) - (ff_sym[k] - ff_prev)
            else:
                d_rate = delta_sym[k] - prev_d
            cost += p.r_steer_rate * d_rate**2

            # FF magnitude penalty (if enabled)
            if p.r_steer_ff_mag > 0:
                cost += p.r_steer_ff_mag * (delta_sym[k] - ff_sym[k])**2

            # Dynamics
            delta_rad = delta_sym[k] * p.max_steer_rad
            L_eff = p.wheelbase_m + p.understeer_gradient * v**2
            e_l = e_l + v * casadi.sin(e_h) * dt
            e_h = e_h + (v / L_eff) * casadi.tan(delta_rad) * dt - kappa_sym[k] * v * dt
            v = casadi.fmin(v_mx, casadi.fmax(p.v_min, v + (accel_sym[k] + g_off) * dt))

        # Terminal cost
        cost += p.q_lat * p.q_lat_terminal_scale * e_l**2
        cost += p.q_heading * p.q_heading_terminal_scale * e_h**2
        cost += p.q_speed * (v - vt)**2

        # Pack decision variables
        x = casadi.vertcat(delta_sym, accel_sym)
        par = casadi.vertcat(e_lat0, e_head0, v0, kappa_sym, vt, last_d, g_off, v_mx, ff_sym)

        # Constraints: rate limits δ[k] - δ[k-1] ∈ [-rate_max, rate_max] for k=1..N-1
        g = []
        for k in range(1, N):
            g.append(delta_sym[k] - delta_sym[k - 1])
        g = casadi.vertcat(*g) if g else casadi.SX(0, 1)

        # Bounds
        lbx = np.concatenate([np.full(N, -1.0), np.full(N, -(p.max_decel + 1.0))])
        ubx = np.concatenate([np.full(N, 1.0), np.full(N, p.max_accel + 1.0)])
        lbg = np.full(N - 1, -p.delta_rate_max) if N > 1 else np.array([])
        ubg = np.full(N - 1, p.delta_rate_max) if N > 1 else np.array([])

        nlp = {'x': x, 'f': cost, 'g': g, 'p': par}

        opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': 200,
            'ipopt.tol': 1e-6,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.mu_init': 1e-3,
            'ipopt.sb': 'yes',  # suppress banner
            'print_time': 0,
        }
        self._solver = casadi.nlpsol('nmpc', 'ipopt', nlp, opts)
        self._lbx_base = lbx
        self._ubx_base = ubx
        self._lbg = lbg
        self._ubg = ubg
        self._N = N
        self._cached_dt = dt
        self._n_params = par.shape[0]

    def solve(self, e_lat: float, e_heading: float, v: float,
              last_delta_norm: float, kappa_ref_horizon: np.ndarray,
              v_target: float, v_max: float, dt: float,
              grade_rad: float = 0.0,
              prediction_dt: Optional[float] = None) -> dict:
        """Solve NMPC NLP for one frame. Same interface as SLSQPNMPCSolver."""
        t0 = time.perf_counter()
        p = self.p
        N = self._N

        effective_dt = prediction_dt if prediction_dt is not None else p.dt
        if self._cached_dt != effective_dt:
            self._rebuild_solver(effective_dt)

        # Curvature horizon
        kappa = np.zeros(N)
        if kappa_ref_horizon is not None and len(kappa_ref_horizon) > 0:
            n_copy = min(len(kappa_ref_horizon), N)
            kappa[:n_copy] = kappa_ref_horizon[:n_copy]
            kappa[n_copy:] = kappa[n_copy - 1]

        v_safe = max(v, 0.5)
        v_max_eff = min(v_max, p.v_max)
        grade_clamped = max(-p.grade_clamp_rad, min(p.grade_clamp_rad, grade_rad))
        gravity_offset = -9.81 * math.sin(grade_clamped) * p.grade_ff_gain

        # Feedforward
        delta_ff = np.array([
            self._feedforward_delta_norm(kappa[k], p.wheelbase_m, p.max_steer_rad)
            for k in range(N)
        ])

        # Parameters: [e_lat0, e_head0, v0, kappa(N), v_target, last_delta, gravity_offset, v_max_eff, delta_ff(N)]
        par_val = np.concatenate([
            [e_lat, e_heading, v_safe],
            kappa,
            [v_target, last_delta_norm, gravity_offset, v_max_eff],
            delta_ff,
        ])

        # Bounds: tighten first delta for rate constraint from last_delta
        lbx = self._lbx_base.copy()
        ubx = self._ubx_base.copy()
        lbx[0] = max(-1.0, last_delta_norm - p.delta_rate_max)
        ubx[0] = min(1.0, last_delta_norm + p.delta_rate_max)
        # Expand accel bounds for grade
        abs_g = abs(gravity_offset)
        lbx[N:] = -(p.max_decel + abs_g)
        ubx[N:] = p.max_accel + abs_g

        # Warm start: convert from interleaved [δ₀,a₀,δ₁,a₁,...] to separate [δ...,a...]
        if self._warm_u is not None and len(self._warm_u) == 2 * N:
            # Shift by one step (receding horizon)
            old_delta = self._warm_u[::2]  # interleaved → delta only
            old_accel = self._warm_u[1::2]  # interleaved → accel only
            x0_delta = np.empty(N)
            x0_accel = np.empty(N)
            x0_delta[:N - 1] = old_delta[1:]
            x0_delta[N - 1] = old_delta[N - 1]
            x0_accel[:N - 1] = old_accel[1:]
            x0_accel[N - 1] = old_accel[N - 1]
            x0 = np.concatenate([x0_delta, x0_accel])
        else:
            x0 = np.concatenate([delta_ff, np.zeros(N)])

        try:
            sol = self._solver(x0=x0, p=par_val,
                               lbx=lbx, ubx=ubx,
                               lbg=self._lbg, ubg=self._ubg)
            solve_ms = (time.perf_counter() - t0) * 1000.0

            stats = self._solver.stats()
            ipopt_status = stats.get('return_status', 'unknown')
            iterations = int(stats.get('iter_count', 0))
            feasible = ipopt_status in ('Solve_Succeeded', 'Solved_To_Acceptable_Level')

            if feasible:
                x_opt = np.array(sol['x']).flatten()
                delta_opt = x_opt[:N]
                accel_opt = x_opt[N:]
                delta_norm_0 = float(np.clip(delta_opt[0], -1.0, 1.0))
                accel_0 = float(np.clip(accel_opt[0], -p.max_decel, p.max_accel))

                # Cache warm start in interleaved format (for compatibility)
                warm = np.empty(2 * N)
                warm[::2] = delta_opt
                warm[1::2] = accel_opt
                self._warm_u = warm

                # Rollout predicted trajectory for diagnostics
                u_interleaved = warm.copy()
                # Use NumPy rollout (same as SLSQP solver)
                predicted = self._rollout_np(u_interleaved, e_lat, e_heading, v_safe,
                                              kappa, gravity_offset, v_max_eff, effective_dt)
                return {
                    'steering_normalized': delta_norm_0,
                    'accel': accel_0,
                    'feasible': True,
                    'solve_time_ms': solve_ms,
                    'predicted_trajectory': predicted,
                    'nmpc_cost': float(sol['f']),
                    'nmpc_iterations': iterations,
                    'slsqp_status': 0,  # compat: 0 = success
                    'prediction_dt_used': effective_dt,
                }

            # Infeasible
            return {
                'steering_normalized': 0.0,
                'accel': 0.0,
                'feasible': False,
                'solve_time_ms': solve_ms,
                'predicted_trajectory': np.zeros((N + 1, 3)),
                'nmpc_cost': float('nan'),
                'nmpc_iterations': iterations,
                'slsqp_status': -1,
                'prediction_dt_used': effective_dt,
            }

        except Exception as exc:
            solve_ms = (time.perf_counter() - t0) * 1000.0
            logger.error("CasADiNMPCSolver: %s", exc)
            return {
                'steering_normalized': 0.0,
                'accel': 0.0,
                'feasible': False,
                'solve_time_ms': solve_ms,
                'predicted_trajectory': np.zeros((N + 1, 3)),
                'nmpc_cost': float('nan'),
                'nmpc_iterations': 0,
                'slsqp_status': -1,
                'prediction_dt_used': effective_dt,
            }

    def _rollout_np(self, u_interleaved: np.ndarray,
                     e_lat0: float, e_heading0: float, v0: float,
                     kappa: np.ndarray, gravity_offset: float,
                     v_max_eff: float, dt: float) -> np.ndarray:
        """NumPy forward rollout for predicted trajectory (same as SLSQPNMPCSolver)."""
        p = self.p
        N = p.horizon
        states = np.empty((N + 1, 3))
        states[0] = [e_lat0, e_heading0, v0]
        for k in range(N):
            e_l, e_h, v = states[k]
            delta = u_interleaved[2 * k] * p.max_steer_rad
            a = u_interleaved[2 * k + 1]
            L_eff = p.wheelbase_m + p.understeer_gradient * v * v
            states[k + 1, 0] = e_l + v * math.sin(e_h) * dt
            states[k + 1, 1] = e_h + (v / L_eff) * math.tan(delta) * dt - kappa[k] * v * dt
            states[k + 1, 2] = max(p.v_min, min(v_max_eff, v + (a + gravity_offset) * dt))
        return states


# Alias for backward compatibility
NMPCSolver = SLSQPNMPCSolver


# ---------------------------------------------------------------------------
# NMPCController — public wrapper with fallback logic
# ---------------------------------------------------------------------------

class NMPCController:
    """
    NMPC lateral controller with LMPC fallback.

    Called by VehicleController each frame when regime = NONLINEAR_MPC.
    If the solver fails for max_consecutive_failures frames, sets
    nmpc_fallback_active=True — caller should dispatch to LMPC for that frame.

    Public interface mirrors MPCController.compute_steering() so the dispatch
    block in pid_controller.py can call either interchangeably.
    """

    def __init__(self, config: dict):
        self.params = NMPCParams.from_config(config)
        # Use CasADi + IPOPT if available, fall back to SLSQP
        if _CASADI_AVAILABLE:
            try:
                self.solver = CasADiNMPCSolver(self.params)
                self._solver_backend = 'casadi_ipopt'
                logger.info("NMPC: using CasADi + IPOPT solver")
            except Exception as exc:
                logger.warning("NMPC: CasADi init failed (%s), falling back to SLSQP", exc)
                self.solver = SLSQPNMPCSolver(self.params)
                self._solver_backend = 'scipy_slsqp'
        else:
            self.solver = SLSQPNMPCSolver(self.params)
            self._solver_backend = 'scipy_slsqp'
        self._consecutive_failures = 0
        self._fallback_active = False
        self._last_steering = 0.0
        self._frames_since_reset = 0

    def compute_steering(
        self,
        e_lat: float,
        e_heading: float,
        current_speed: float,
        last_delta_norm: float,
        kappa_ref: float,
        v_target: float,
        v_max: float,
        dt: float,
        kappa_horizon=None,
        grade_rad: float = 0.0,
        prediction_dt: Optional[float] = None,
    ) -> dict:
        """
        Compute NMPC steering for one frame.  Signature mirrors MPCController.

        Args:
            e_lat: lateral error (m)
            e_heading: heading error (rad)
            current_speed: vehicle speed (m/s)
            last_delta_norm: previous normalized steering [-1, 1]
            kappa_ref: reference curvature at current position (1/m)
            v_target: desired speed (m/s)
            v_max: track speed limit (m/s)
            dt: frame timestep (s) — not used for MPC prediction (params.dt is used)
            kappa_horizon: optional array of N curvatures from trajectory planner
            grade_rad: road grade in radians (positive = uphill)

        Returns:
            dict: steering_normalized, accel, nmpc_feasible, solve_time_ms,
                  nmpc_fallback_active, nmpc_consecutive_failures,
                  nmpc_cost, nmpc_iterations, e_lat_input, e_heading_input,
                  kappa_ref_used
        """
        p = self.params
        N = p.horizon

        # Build kappa horizon (N values)
        if kappa_horizon is not None and len(kappa_horizon) > 0:
            kappa_arr = np.empty(N)
            n_copy = min(len(kappa_horizon), N)
            kappa_arr[:n_copy] = kappa_horizon[:n_copy]
            kappa_arr[n_copy:] = kappa_horizon[n_copy - 1] if n_copy > 0 else kappa_ref
        else:
            kappa_arr = np.full(N, kappa_ref)

        # Startup warmup: relax first-step rate constraint so SLSQP can cold-start
        # freely from an unfavourable PP-inherited state.
        last_d = last_delta_norm
        if self._frames_since_reset < p.warmup_frames:
            last_d = 0.0

        result = self.solver.solve(
            e_lat, e_heading, current_speed, last_d,
            kappa_arr, v_target, v_max, p.dt,
            grade_rad=grade_rad,
            prediction_dt=prediction_dt,
        )

        self._frames_since_reset += 1

        # Failure: budget exceeded or infeasible
        is_timeout = result.get('solve_time_ms', 0.0) > p.max_solve_ms
        failed = (not result['feasible']) or is_timeout
        if failed:
            self._consecutive_failures += 1
            if self._consecutive_failures >= p.max_consecutive_failures:
                self._fallback_active = True
                logger.warning(
                    "NMPC: fallback active — failures=%d, solve_ms=%.1f, feasible=%s",
                    self._consecutive_failures,
                    result.get('solve_time_ms', 0.0),
                    result['feasible'],
                )
            result['steering_normalized'] = self._last_steering
        else:
            self._consecutive_failures = 0
            self._fallback_active = False
            self._last_steering = result['steering_normalized']

        return {
            'steering_normalized': result['steering_normalized'],
            'accel': result.get('accel', 0.0),
            'nmpc_feasible': result['feasible'],
            'solve_time_ms': result.get('solve_time_ms', 0.0),
            'nmpc_fallback_active': self._fallback_active,
            'nmpc_consecutive_failures': self._consecutive_failures,
            'nmpc_cost': result.get('nmpc_cost', float('nan')),
            'nmpc_iterations': result.get('nmpc_iterations', 0),
            'e_lat_input': e_lat,
            'e_heading_input': e_heading,
            'kappa_ref_used': float(kappa_arr[0]),
            'nmpc_prediction_dt': result.get('prediction_dt_used', p.dt),
        }

    def reset(self) -> None:
        """Reset solver state (e.g., on regime switch or after e-stop)."""
        self.solver._warm_u = None
        self._consecutive_failures = 0
        self._fallback_active = False
        self._last_steering = 0.0
        self._frames_since_reset = 0

    @property
    def should_fallback_to_lmpc(self) -> bool:
        """True when NMPC is in fallback state and params.fallback_lmpc is set."""
        return self._fallback_active and self.params.fallback_lmpc
