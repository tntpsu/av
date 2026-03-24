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
MPC_MIN_SPEED_DEFAULT_MPS: float = 3.0        # m/s — default MPC min speed (Step 4)
MPC_MAX_CURVATURE_DEFAULT: float = 0.020      # 1/m — default curvature guard (R50, Step 4)

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
ACC_GAP_RMSE_GATE_M: float = 0.50           # m  — steady-following gap RMSE gate
ACC_JERK_P95_GATE_MPS3: float = 4.0         # m/s³ — following jerk (tighter than 6.0 free-flow)
ACC_DETECTION_RATE_GATE: float = 0.95       # —  — min detection rate when lead present
ACC_EMERGENCY_BRAKE_GAP_FACTOR: float = 1.5  # —  — gap < factor×speed → emergency brake
ACC_MIN_ACTIVE_FRAME_RATE: float = 0.10     # —  — min fraction for ACC section to activate

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
