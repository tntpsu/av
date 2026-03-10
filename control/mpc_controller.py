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
import time
from dataclasses import dataclass, fields
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
    max_steer_rad: float = 0.5236           # 30°

    # State cost weights
    q_lat: float = 10.0                     # lateral tracking
    q_heading: float = 5.0                  # heading tracking
    q_speed: float = 1.0                    # speed tracking

    # Terminal cost multipliers
    q_lat_terminal_scale: float = 3.0
    q_heading_terminal_scale: float = 3.0

    # Input cost weights
    r_steer: float = 0.1                    # steering magnitude
    r_accel: float = 0.05                   # accel magnitude
    r_steer_rate: float = 1.0              # steering rate — PRIMARY COMFORT KNOB

    # Constraints
    delta_rate_max: float = 0.5             # max Δδ_norm per step
    max_accel: float = 1.2                  # m/s²
    max_decel: float = 2.4                  # m/s²
    v_min: float = 1.0                      # m/s
    v_max: float = 15.0                     # m/s

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

    # Curvature preview horizon (feedforward from trajectory planner)
    curvature_preview_enabled: bool = False   # Default OFF — safe for existing configs
    curvature_preview_gain: float = 1.0       # Scale factor for preview curvature

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
        self._nx = 3   # [e_lat, e_heading, v]
        self._nu = 2   # [δ_norm, a]
        self._prob: Optional[osqp.OSQP] = None
        self._warm_x = None
        self._warm_y = None
        self._nz = 0
        self._nc = 0
        self._build_qp()

    def _get_horizon(self, speed: float) -> int:
        if self.p.speed_adaptive_horizon and speed >= self.p.speed_adaptive_threshold_mps:
            return self.p.speed_adaptive_n_high
        return self.p.horizon

    # ----- QP construction -----

    def _build_qp(self):
        """Build OSQP problem matrices. Called once at init and on horizon change."""
        N = self._N
        nx, nu = self._nx, self._nu
        nz = (nx + nu) * N + nx
        self._nz = nz

        # --- P (Hessian): block diagonal ---
        P_diag = np.zeros(nz)
        for k in range(N):
            base = k * (nx + nu)
            # State cost at step k
            P_diag[base + 0] = self.p.q_lat
            P_diag[base + 1] = self.p.q_heading
            P_diag[base + 2] = self.p.q_speed
            # Input cost at step k
            P_diag[base + nx + 0] = self.p.r_steer
            P_diag[base + nx + 1] = self.p.r_accel

        # Terminal state cost
        term_base = N * (nx + nu)
        P_diag[term_base + 0] = self.p.q_lat * self.p.q_lat_terminal_scale
        P_diag[term_base + 1] = self.p.q_heading * self.p.q_heading_terminal_scale
        P_diag[term_base + 2] = self.p.q_speed

        # Steering rate penalty: r_steer_rate * (δ[k+1] - δ[k])²
        # This adds off-diagonal terms. We build a triplet list for P.
        rows, cols, vals = [], [], []
        for i in range(nz):
            if P_diag[i] != 0.0:
                rows.append(i)
                cols.append(i)
                vals.append(P_diag[i])

        # Steering rate: affects δ_norm at step k (index k*(nx+nu)+nx)
        # and δ_norm at step k+1 (index (k+1)*(nx+nu)+nx)
        for k in range(N - 1):
            idx_k = k * (nx + nu) + nx       # δ_norm at step k
            idx_k1 = (k + 1) * (nx + nu) + nx  # δ_norm at step k+1
            r_sr = self.p.r_steer_rate

            # (δ[k+1] - δ[k])² = δ[k+1]² - 2·δ[k]·δ[k+1] + δ[k]²
            # Add to diagonal
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
            # x[k] block (columns k*(nx+nu) .. k*(nx+nu)+nx-1)
            x_col = k * (nx + nu)
            row[0, x_col + 0] = 1.0    # e_lat → e_lat
            row[0, x_col + 1] = _EPS   # v*dt (updated per-solve)
            row[1, x_col + 1] = 1.0    # e_heading → e_heading
            row[2, x_col + 2] = 1.0    # v → v

            # u[k] block
            u_col = k * (nx + nu) + nx
            row[1, u_col + 0] = _EPS   # (v/L)*δ_max*dt (updated per-solve)
            row[2, u_col + 1] = _EPS   # dt (updated per-solve)

            # -I for x[k+1]
            x_next_col = (k + 1) * (nx + nu) if k < N - 1 else N * (nx + nu)
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

        # ---- 4. State bounds (speed only): for k = 0..N ----
        self._state_row_start = nx + N * nx + N * nu
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
        self._rate_row_start = nx + N * nx + N * nu + (N + 1)
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
              v_target: float, v_max: float, dt: float) -> dict:
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

        Returns:
            dict with: steering_normalized, accel, feasible, solve_time_ms,
                        predicted_trajectory (N+1, 3)
        """
        t0 = time.perf_counter()
        N = self._N
        nx, nu = self._nx, self._nu
        nz = self._nz

        # Check for horizon change
        new_N = self._get_horizon(v)
        if new_N != N:
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

        # --- Update dynamics in A matrix and c vector ---
        # For each step k, linearize around operating point.
        # We update A's nonzero entries in-place via the CSC data array.
        A_new = self._A.tolil()
        l_new = self._l.copy()
        u_new = self._u.copy()

        c_vec = []  # affine terms for dynamics
        for k in range(N):
            dyn_row = nx + k * nx  # row index in A for dynamics block k
            x_col = k * (nx + nu)
            u_col = k * (nx + nu) + nx
            x_next_col = (k + 1) * (nx + nu) if k < N - 1 else N * (nx + nu)
            kk = kappa[k]

            # A_k matrix (state transition)
            # e_lat: e_lat + v*e_heading*dt  (Frenet: no κ term)
            A_new[dyn_row + 0, x_col + 0] = 1.0
            A_new[dyn_row + 0, x_col + 1] = v_safe * dt        # ∂e_lat/∂e_heading
            A_new[dyn_row + 0, x_col + 2] = 0.0                # linearized: ignore ∂/∂v for now

            # e_heading: e_heading + (v/L)*δ_norm*δ_max*dt - κ*v*dt
            A_new[dyn_row + 1, x_col + 0] = 0.0
            A_new[dyn_row + 1, x_col + 1] = 1.0
            A_new[dyn_row + 1, x_col + 2] = 0.0

            # v: v + a*dt
            A_new[dyn_row + 2, x_col + 0] = 0.0
            A_new[dyn_row + 2, x_col + 1] = 0.0
            A_new[dyn_row + 2, x_col + 2] = 1.0

            # B_k matrix (input)
            steer_gain = (v_safe / self.p.wheelbase_m) * self.p.max_steer_rad * dt
            A_new[dyn_row + 0, u_col + 0] = 0.0       # δ doesn't directly affect e_lat
            A_new[dyn_row + 0, u_col + 1] = 0.0       # a doesn't affect e_lat
            A_new[dyn_row + 1, u_col + 0] = steer_gain  # (v/L)*δ_max*dt
            A_new[dyn_row + 1, u_col + 1] = 0.0
            A_new[dyn_row + 2, u_col + 0] = 0.0
            A_new[dyn_row + 2, u_col + 1] = dt

            # -I for x[k+1]
            A_new[dyn_row + 0, x_next_col + 0] = -1.0
            A_new[dyn_row + 1, x_next_col + 1] = -1.0
            A_new[dyn_row + 2, x_next_col + 2] = -1.0

            # Affine term c_k (curvature feedforward)
            # NOTE: c_lat must be 0.  In Frenet error coordinates, curvature
            # affects only heading (ė_ψ = v·κ_car − v·κ_road).  The lateral
            # equation is ė_y = v·sin(e_ψ) ≈ v·e_ψ — no κ term.  A nonzero
            # c_lat creates phantom lateral drift that causes limit-cycle
            # oscillation on curved roads (see Phase 2.8 root-cause analysis).
            c_lat = 0.0
            c_head = -kk * v_safe * dt              # −κ·v·dt
            c_v = 0.0
            c_vec.append(np.array([c_lat, c_head, c_v]))

        A_csc = A_new.tocsc()

        # --- Update constraint bounds ---
        # 1. Initial state
        l_new[0] = e_lat
        l_new[1] = e_heading
        l_new[2] = v_safe
        u_new[0] = e_lat
        u_new[1] = e_heading
        u_new[2] = v_safe

        # 2. Dynamics: l = u = -c_k (equality constraints)
        for k in range(N):
            row_start = nx + k * nx
            c_k = c_vec[k]
            l_new[row_start:row_start + nx] = -c_k
            u_new[row_start:row_start + nx] = -c_k

        # 3. Input bounds (already set in template, but update v_max)
        # Input bounds are static — no update needed

        # 4. State bounds (speed): update v_max
        state_start = self._state_row_start
        for k in range(N + 1):
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
        for k in range(N):
            base = k * (nx + nu)
            # Speed tracking: q_speed * (v - v_target)² = q_speed * v² - 2*q_speed*v_target*v + const
            # Linear term: -2 * q_speed * v_target (but OSQP uses 0.5*z'Pz + q'z, so factor is -q_speed*v_target)
            q_new[base + 2] = -self.p.q_speed * v_target

        # Terminal speed tracking
        term_base = N * (nx + nu)
        q_new[term_base + 2] = -self.p.q_speed * v_target

        # --- Solve ---
        try:
            self._prob.update(Ax=A_csc.data, q=q_new, l=l_new, u=u_new)
        except Exception:
            # If structure changed, need full setup
            self._prob = osqp.OSQP()
            self._prob.setup(
                self._P, q_new, A_csc, l_new, u_new,
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
                'predicted_trajectory': np.zeros((N + 1, 3)),
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
        predicted = np.zeros((N + 1, 3))
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

    def compute_steering(self, e_lat: float, e_heading: float,
                         current_speed: float, last_delta_norm: float,
                         kappa_ref: float, v_target: float,
                         v_max: float, dt: float,
                         kappa_horizon=None) -> dict:
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
        if self.params.bias_enabled and current_speed >= self.params.bias_min_speed:
            alpha = self.params.bias_alpha
            self._bias_ema = (1.0 - alpha) * self._bias_ema + alpha * e_lat
            raw_correction = -self.params.bias_kappa_gain * self._bias_ema
            self._bias_correction = float(np.clip(
                raw_correction,
                -self.params.bias_max_correction,
                self.params.bias_max_correction,
            ))

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
        kappa_horizon = kappa_corrected_horizon

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

        # Use the configured MPC prediction step (params.dt), NOT the frame dt.
        # The solver plans N steps of params.dt each (e.g. 20×0.1=2.0 s horizon).
        # Passing the frame dt (0.033 s) shrinks the horizon to 0.66 s, making
        # the terminal cost dominate and causing aggressive overcorrection.
        result = self.solver.solve(
            e_lat, e_heading, current_speed, last_delta_norm,
            kappa_horizon, v_target, v_max, self.params.dt
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
            'e_lat_input': e_lat,
            'e_heading_input': e_heading,
            'kappa_ref_used': kappa_ref + self._bias_correction,
            'kappa_bias_correction': self._bias_correction,
            'kappa_bias_ema': self._bias_ema,
            'kappa_preview_used': bool(_preview_used),
            'kappa_preview_range': float(np.ptp(kappa_corrected_horizon)) if _preview_used else 0.0,
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
