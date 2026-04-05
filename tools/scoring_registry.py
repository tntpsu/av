"""
Centralized scoring thresholds for the AV stack.

Single source of truth for metric thresholds shared across scoring,
testing, and diagnostic modules. Pure constants — no project imports.

Consumers:
    tests/conftest.py           — COMFORT_GATES dict
    tools/drive_summary_core.py — penalty thresholds, OOL detection
    tools/debug_visualizer/backend/layer_health.py   — control/traj/ACC thresholds
    tools/debug_visualizer/backend/issue_detector.py  — OOL, catastrophic, MPC/ACC budget
    tools/debug_visualizer/backend/acc_pipeline.py    — ACC tab (PhilViz)
    tools/analyze/acc_pipeline_analysis.py            — ACC pipeline diagnostic CLI
"""

from __future__ import annotations

# ── Comfort gates (S1-M39) ───────────────────────────────────────────────────
ACCEL_P95_GATE_MPS2: float = 3.0       # m/s²  — EMA-filtered accel P95
JERK_P95_GATE_MPS3: float = 6.0        # m/s³  — commanded jerk P95
LATERAL_P95_GATE_M: float = 0.40       # m     — curvature-adjusted lateral error P95
CENTERED_PCT_MIN: float = 70.0         # %     — minimum % frames within centered band
CENTERED_BAND_M: float = 0.5           # m     — half-width of centered band (±0.5m)
STEERING_JERK_GATE: float = 20.0       # norm/s² — steering jerk max (test gate)

# ── Out-of-lane / safety ─────────────────────────────────────────────────────
OUT_OF_LANE_THRESHOLD_M: float = 0.5   # m     — lateral error beyond this = out of lane
CATASTROPHIC_ERROR_M: float = 2.0       # m     — lateral error beyond this = catastrophic
MIN_CONSECUTIVE_OOL: int = 10           # frames — sustained OOL event minimum duration

# ── Scoring penalty thresholds ───────────────────────────────────────────────
CURVATURE_FLOOR_COEFF: float = 3.0      # arc-chord tracking error floor coefficient
STEERING_JERK_PENALTY_CAP: float = 18.0 # norm/s² — penalty applied above this cap
HEADING_PENALTY_FLOOR_DEG: float = 10.0 # degrees — heading RMSE penalty floor

# ── Layer score pass gate ────────────────────────────────────────────────────
LAYER_SCORE_MIN_PASS: float = 95.0      # —  — every layer must score >= this to pass

# ── Layer health thresholds (shared with triage_engine via re-export) ────────
PERCEPTION_CONF_FLOOR: float = 0.1      # confidence below this → hard fallback
TRAJ_REF_ERROR_MAX_M: float = 2.0       # lateral ref error > this → full penalty
CTRL_LATERAL_ERROR_MAX: float = 2.0     # lateral error > this → full penalty
CTRL_JERK_GATE: float = JERK_P95_GATE_MPS3  # alias for layer_health consumers
LOOKAHEAD_MIN_M: float = 5.0            # lookahead below this is suspiciously short
LOOKAHEAD_MAX_M: float = 25.0           # lookahead above this is unusually long

# ── Derived thresholds ───────────────────────────────────────────────────────
JERK_ALERT_MPS3: float = JERK_P95_GATE_MPS3 * 0.75  # 4.5 — 75% of gate; triggers flag
CTRL_JERK_ALERT: float = JERK_ALERT_MPS3            # alias for layer_health consumers
LOOKAHEAD_CONCERN_M: float = 8.0        # "short at speed" alert threshold

# ── MPC thresholds ───────────────────────────────────────────────────────────
MPC_SOLVE_TIME_BUDGET_MS: float = 5.0   # ms — solve time budget
MPC_SOLVE_TIME_ALERT_MS: float = 8.0    # ms — slow solve alert
MPC_SOLVE_TIME_P95_MS_GATE: float = 5.0       # ms — P95 solve time gate
MPC_SOLVE_TIME_MARGINAL_MS: float = 3.0       # ms — marginal solve time threshold
MPC_INFEASIBILITY_RATE_GATE: float = 0.005    # — max infeasibility rate
MPC_FALLBACK_RATE_GATE: float = 0.005         # — max fallback rate
MPC_REGIME_CHATTER_PER_MIN: float = 6.0       # transitions/min — chatter threshold
MPC_STEERING_OSC_RATE_GATE: float = 0.30      # — steering sign-change rate gate
MPC_HEADING_ERROR_P95_RAD: float = 0.25       # rad — heading error P95 gate
MPC_MIN_SPEED_DEFAULT_MPS: float = 3.0        # m/s — DEPRECATED: replaced by lateral accel budget
MPC_MAX_CURVATURE_DEFAULT: float = 0.020      # 1/m — DEPRECATED: replaced by lateral accel budget

# ── Regime lateral acceleration budget (replaces speed/curvature proxies) ────
MPC_MAX_LATERAL_ACCEL_MPS2: float = 1.5       # m/s² — κ×v² budget; MPC→PP above this
MPC_LATERAL_ACCEL_HYSTERESIS_MPS2: float = 0.3  # m/s² — dead band ±0.3 around threshold
MPC_MIN_SPEED_ABSOLUTE_MPS: float = 2.0       # m/s — hard floor; MPC needs nonzero speed

# ── NMPC thresholds (Step 5) ─────────────────────────────────────────────────
NMPC_SOLVE_TIME_BUDGET_MS: float = 20.0       # ms — hard solve budget (scipy SLSQP warm-started)
NMPC_SOLVE_TIME_ALERT_MS: float = 15.0        # ms — soft alert threshold (approaching budget)
NMPC_SOLVE_TIME_P95_MS_GATE: float = 20.0     # ms — P95 gate for analysis report pass/fail
NMPC_FALLBACK_RATE_GATE: float = 0.01         # — max acceptable NMPC→LMPC fallback rate (1%)
NMPC_INFEASIBILITY_RATE_GATE: float = 0.01    # — max acceptable infeasibility rate
NMPC_REGIME_CHATTER_PER_MIN: float = 6.0      # LMPC↔NMPC transitions/min — chatter threshold
NMPC_MIN_SPEED_MPS: float = 20.0              # m/s — LMPC→NMPC upshift threshold (lmpc_max_speed_mps)

# ── Grade compensation ──────────────────────────────────────────────────────
GRADE_MAX_SAFE_PCT: float = 10.0           # %     — max safe road grade
GRADE_EMA_ALPHA: float = 0.3              # —     — EMA smoothing for grade signal
GRADE_CLAMP_RAD: float = 0.15             # rad   — max grade input to MPC (~8.5°)
DOWNHILL_SPEED_MARGIN_MPS: float = 1.0    # m/s   — allowable overspeed margin on downhill
GRADE_FF_GAIN_DEFAULT: float = 1.8       # —     — calibrated for Unity WheelCollider physics (promoted to base)
GRADE_FF_GAIN_UNITY: float = 1.8         # —     — calibrated for Unity WheelCollider physics
GRADE_THROTTLE_BUDGET_MIN_RATIO: float = 0.5   # — — min acceptable (effective_max_accel - gravity) / max_accel
GRADE_THROTTLE_SATURATION_RATE: float = 0.10   # — — max acceptable fraction of graded frames at throttle > 0.95
GRADE_JERK_RELAXATION_GAIN: float = 2.0        # — — jerk limit bonus per unit gravity_accel
GRADE_STEERING_DAMPING_GAIN: float = 5.0       # — — steering alpha reduction per rad of grade
GRADE_TRANSITION_BLEND_M: float = 3.0          # m — cosine blend radius at grade transitions (TrackBuilder)

# ── ACC — Hard safety gates (Tier 1) ─────────────────────────────────────────
ACC_COLLISION_GATE: int = 0                   # events — zero tolerance; score → 0
ACC_TTC_CRITICAL_S: float = 1.5              # s  — e-stop trigger (same path as OOL)
ACC_NEAR_MISS_GAP_M: float = 2.0             # m  — gap < s0; inside minimum gap (orange)

# ── ACC — Graduated safety thresholds (Tier 2) ───────────────────────────────
ACC_TTC_MIN_GATE_S: float = 2.0              # s  — promotion gate (must stay above)
ACC_TTC_WARNING_S: float = 2.5              # s  — start of warning zone (2.0–2.5s)
ACC_TTC_COMFORTABLE_S: float = 4.0          # s  — comfortable headway (green)
ACC_DESIRED_GAP_FRACTION_WARN: float = 0.5  # —  — < 50% desired gap = IDM struggling
ACC_NEAR_MISS_PENALTY_PTS: float = 15.0     # pts — per near-miss event (Safety layer)
ACC_TTC_WARNING_PENALTY_PER_PCT: float = 0.3  # pts — per % of ACC frames in warning zone

# ── ACC — Comfort and quality (Tier 3) ───────────────────────────────────────
ACC_GAP_RMSE_GATE_M: float = 35.0           # m  — full-run gap RMSE gate (includes startup phase).
                                            #      IDM equilibrium gap = target_gap / sqrt(1-(v/v0)^4).
                                            #      At v_lead=12, v0=15: eq_gap≈26m vs target≈20m → steady-state
                                            #      gap_error≈6–8m. With ego-from-rest startup, full-run RMSE≈30m.
                                            #      Gate at 35m catches broken ACC (gap never closes, RMSE≈50m+)
                                            #      while passing correctly-converging ACC (RMSE≈28–32m).
ACC_JERK_P95_GATE_MPS3: float = 4.0         # m/s³ — following jerk (tighter than 6.0 free-flow)
ACC_DETECTION_RATE_GATE: float = 0.95       # —  — min detection rate when lead present
ACC_EMERGENCY_BRAKE_GAP_FACTOR: float = 1.5  # —  — gap < factor×speed → emergency brake
ACC_MIN_ACTIVE_FRAME_RATE: float = 0.10     # —  — min fraction for ACC section to activate

# ── Oscillation amplitude growth ─────────────────────────────────────────────
OSCILLATION_AMPLITUDE_GROWTH_THRESHOLD_MPS: float = 0.001  # m/s — RMS growth slope below this is benign
OSCILLATION_AMPLITUDE_GROWTH_SCALE: float = 5000.0         # penalty per unit slope (0.001→0, 0.003→10)
OSCILLATION_AMPLITUDE_GROWTH_MAX_PENALTY: float = 15.0     # max points deducted in Control layer

# ── Inter-frame control extrapolation ─────────────────────────────────────────
INTERFRAME_STALE_GT_THRESHOLD_MS: float = 50.0     # ms — skip inter-frame if GT older than this
INTERFRAME_E_LAT_DIVERGENCE_THRESHOLD_M: float = 0.2  # m — flag if inter-frame e_lat diverges from camera
INTERFRAME_MAX_UPDATES_PER_CYCLE: int = 3          # max inter-frame updates between camera frames

# ── L_eff online wheelbase estimation ───────────────────────────────────────
LEFF_NOMINAL_M: float = 2.5                  # m  — kinematic wheelbase (no adaptation)
LEFF_BOUNDS_MIN_M: float = 1.5               # m  — hard safety clamp (60% of nominal)
LEFF_BOUNDS_MAX_M: float = 8.0               # m  — hard safety clamp (320% of nominal; high due to actuator delay)
LEFF_EXCITATION_THRESHOLD: float = 0.01      # —  — min |v·δ·δ_max·dt| per frame for RLS update

# ── Dynamic bicycle model / tire estimation ────────────────────────────────
TIRE_CF_NOMINAL: float = 40000.0              # N/rad — front cornering stiffness nominal
TIRE_CR_NOMINAL: float = 40000.0              # N/rad — rear cornering stiffness nominal
TIRE_CF_BOUNDS_MIN: float = 15000.0           # N/rad — hard lower safety clamp
TIRE_CF_BOUNDS_MAX: float = 80000.0           # N/rad — hard upper safety clamp
TIRE_CR_BOUNDS_MIN: float = 15000.0           # N/rad — hard lower safety clamp
TIRE_CR_BOUNDS_MAX: float = 80000.0           # N/rad — hard upper safety clamp
TIRE_SLIP_ANGLE_LINEAR_MAX_RAD: float = 0.087 # rad (~5°) — linear tire region limit
TIRE_SLIP_ANGLE_SATURATION_RAD: float = 0.15  # rad (~8.6°) — saturation onset
TIRE_EKF_INNOVATION_P95_GATE: float = 0.05    # rad/s — yaw rate innovation P95 gate
TIRE_EKF_INNOVATION_DIVERGENCE: float = 0.15  # rad/s — innovation above this = divergent
TIRE_EKF_IMU_YAW_RATE_P95_GATE: float = 0.30      # rad/s — IMU yaw rate P95 sanity check
TIRE_EKF_IMU_SIGNAL_PRESENCE_GATE: float = 0.95    # min fraction of frames with non-zero IMU
TIRE_UNDERSTEER_GRADIENT_NOMINAL: float = 0.002  # rad/(m/s²) — expected K_us
TIRE_UNDERSTEER_GRADIENT_MAX: float = 0.01    # rad/(m/s²) — anomalous understeer threshold

# ── Vehicle geometry sanity bounds ───────────────────────────────────────────
VEHICLE_LF_MIN_M: float = 0.5                # m — min distance CoG to front axle
VEHICLE_LF_MAX_M: float = 3.0                # m — max distance CoG to front axle
VEHICLE_LR_MIN_M: float = 0.5                # m — min distance CoG to rear axle
VEHICLE_LR_MAX_M: float = 3.0                # m — max distance CoG to rear axle
VEHICLE_MASS_MIN_KG: float = 500.0            # kg — min vehicle mass
VEHICLE_MASS_MAX_KG: float = 5000.0           # kg — max vehicle mass
VEHICLE_IZ_MIN_KGMM: float = 200.0           # kg*m² — min yaw inertia
VEHICLE_IZ_MAX_KGMM: float = 10000.0         # kg*m² — max yaw inertia
GEOMETRY_OVERRIDE_FRAME_DEADLINE: int = 30    # frames — must receive Unity params within this

# ── Blind perception gates ────────────────────────────────────────────────────
PERCEPTION_BLIND_RATE_GATE: float = 0.05       # —     — max fraction of frames with no lane detection
PERCEPTION_BLIND_CONSECUTIVE_GATE: int = 5     # frames — consecutive no-detection before blind flag
PERCEPTION_BLIND_PENALTY_MAX: float = 25.0     # pts   — max perception layer deduction for blind

# ── Benign stale reasons ─────────────────────────────────────────────────────
BENIGN_STALE_REASONS: frozenset[str] = frozenset({"left_lane_low_visibility"})

# ── Convenience dict matching conftest shape ─────────────────────────────────
COMFORT_GATES: dict[str, float] = {
    "accel_p95_filtered_max":  ACCEL_P95_GATE_MPS2,
    "commanded_jerk_p95_max":  JERK_P95_GATE_MPS3,
    "lateral_p95_max":         LATERAL_P95_GATE_M,
    "centered_pct_min":        CENTERED_PCT_MIN,
    "out_of_lane_events_max":  0,
    "emergency_stops_max":     0,
    "steering_jerk_max_max":   STEERING_JERK_GATE,
}
