"""
Triage engine for PhilViz Phase 5.

Matches failure patterns against a library of known issue signatures,
computes per-layer attribution, and generates an ordered action checklist.

Pattern library: maps (signal conditions) → (code pointer, config lever, fix hint).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import h5py
import yaml

from backend.layer_health import (
    CTRL_JERK_GATE, CTRL_JERK_ALERT,
    LOOKAHEAD_CONCERN_M, BENIGN_STALE_REASONS,
    TIRE_EKF_INNOVATION_DIVERGENCE,
    TIRE_SLIP_ANGLE_SATURATION_RAD,
)


def _load_longitudinal_limits() -> tuple[float, float]:
    """Load longitudinal limits from config for command-domain jerk computation."""
    config_path = Path(__file__).resolve().parents[3] / "config" / "av_stack_config.yaml"
    max_accel = 2.5
    max_decel = 3.0
    if not config_path.exists():
        return max_accel, max_decel
    try:
        with config_path.open("r") as f:
            cfg = yaml.safe_load(f) or {}
        long_cfg = ((cfg.get("control") or {}).get("longitudinal") or {})
        max_accel = float(long_cfg.get("max_accel", max_accel))
        max_decel = float(long_cfg.get("max_decel", max_decel))
    except Exception:
        pass
    return max_accel, max_decel


def _compute_commanded_jerk_p95(
    throttle: np.ndarray | None,
    brake: np.ndarray | None,
    timestamps: np.ndarray | None,
    max_accel: float,
    max_decel: float,
) -> float:
    if throttle is None or brake is None:
        return 0.0
    t_arr = np.asarray(throttle, dtype=float).reshape(-1)
    b_arr = np.asarray(brake, dtype=float).reshape(-1)
    n = min(t_arr.size, b_arr.size)
    if timestamps is not None:
        ts_arr = np.asarray(timestamps, dtype=float).reshape(-1)
        n = min(n, ts_arr.size)
    if n <= 1:
        return 0.0
    t_arr = t_arr[:n]
    b_arr = b_arr[:n]
    if timestamps is not None:
        ts_arr = np.asarray(timestamps, dtype=float).reshape(-1)[:n]
    else:
        ts_arr = np.arange(n, dtype=float) * (1.0 / 30.0)
    dt = np.diff(ts_arr)
    finite_positive = dt[np.isfinite(dt) & (dt > 1e-6)]
    fallback_dt = float(np.median(finite_positive)) if finite_positive.size > 0 else (1.0 / 30.0)
    dt = np.where(dt > 1e-6, dt, fallback_dt)
    net_accel = (t_arr * float(max_accel)) - (b_arr * float(max_decel))
    cmd_jerk = np.divide(
        np.diff(net_accel),
        dt,
        out=np.zeros_like(dt, dtype=float),
        where=dt > 1e-6,
    )
    if cmd_jerk.size == 0:
        return 0.0
    return float(np.percentile(np.abs(cmd_jerk), 95))


# ── Pattern Library ───────────────────────────────────────────────────────────
# severity: "safety" > "instability" > "comfort"
PATTERNS = [
    {
        "id": "safety_ool_event",
        "name": "Out-of-lane event",
        "severity": "safety",
        "code_pointer": "av_stack.py:check_safety_limits() (~line 4100)",
        "config_lever": "safety.emergency_stop_error",
        "fix_hint": "Review lateral tracking quality. If consistent OOL on curves: reduce target speed or increase lookahead.",
        "check": lambda m: m.get("out_of_lane_events", 0) >= 1,
    },
    {
        "id": "perc_stale_cascade",
        "name": "Perception stale cascade",
        "severity": "instability",
        "code_pointer": "av_stack.py:apply_lane_ema_gating() (~line 2341)",
        "config_lever": "perception.low_visibility_fallback_max_consecutive_frames",
        "fix_hint": "Reduce stale TTL from 8 → 5 frames. Or raise model_fallback_confidence_hard_threshold.",
        "check": lambda m: m.get("stale_rate", 0) > 0.15 and m.get("lateral_p95", 0) > 0.3,
    },
    {
        "id": "perc_low_confidence",
        "name": "Persistent low detection confidence",
        "severity": "instability",
        "code_pointer": "perception/inference.py:detect() — confidence scoring",
        "config_lever": "perception.model_fallback_confidence_hard_threshold",
        "fix_hint": "Check camera exposure / track lighting. If systematic: retrain segmentation model.",
        "check": lambda m: m.get("mean_perception_confidence", 1.0) < 0.4,
    },
    {
        "id": "perc_single_lane_only",
        "name": "Single-lane detection only (>30% frames)",
        "severity": "instability",
        "code_pointer": "av_stack.py:blend_lane_pair_with_previous() (~line 2500)",
        "config_lever": "perception.lane_center_gate_m",
        "fix_hint": "Reduce lane_center_gate_m (0.3 → 0.2) to allow wider lane center updates.",
        "check": lambda m: m.get("single_lane_rate", 0) > 0.30,
    },
    {
        "id": "perc_blind_no_detection",
        "name": "Perception blind (no lane detection >5% frames)",
        "severity": "safety",
        "code_pointer": "av_stack/orchestrator.py:_pf_score_perception_health()",
        "config_lever": "N/A — seg model or CV fallback must detect lanes",
        "fix_hint": "Segmentation model returned zero masks. Check model checkpoint, camera exposure, or retrain on scenario imagery.",
        "check": lambda m: m.get("blind_perception_rate", 0) > 0.05,
    },
    {
        "id": "traj_ref_rate_clamped",
        "name": "Trajectory jitter (ref rate clamp active >20% frames)",
        "severity": "instability",
        "code_pointer": "trajectory/inference.py:compute_reference_point() (~line 445)",
        "config_lever": "trajectory.ref_x_rate_limit",
        "fix_hint": "Increase ref_x_rate_limit (0.22 → 0.30) to reduce clamp frequency.",
        "check": lambda m: m.get("ref_rate_clamp_rate", 0) > 0.20,
    },
    {
        "id": "traj_overcorrection",
        "name": "Trajectory overcorrection (clamp + high lateral error)",
        "severity": "instability",
        "code_pointer": "trajectory/inference.py:compute_curve_feedforward() (~line 510)",
        "config_lever": "trajectory.curve_feedforward_gain",
        "fix_hint": "Reduce curve_feedforward_gain. Clamping + high error together suggest feedforward is overcorrecting.",
        "check": lambda m: m.get("ref_rate_clamp_rate", 0) > 0.15 and m.get("lateral_p95", 0) > 0.3,
    },
    {
        "id": "traj_short_lookahead",
        "name": "Lookahead too short for current speed",
        "severity": "comfort",
        "code_pointer": "trajectory/inference.py:get_lookahead_distance() (~line 380)",
        "config_lever": "trajectory.reference_lookahead_speed_table",
        "fix_hint": "Add/raise speed table entry for current target speed. Short lookahead at speed causes oscillation.",
        "check": lambda m: m.get("short_lookahead_rate", 0) > 0.05,
    },
    {
        "id": "ctrl_jerk_spike",
        "name": f"Longitudinal jerk spikes (P95 > {CTRL_JERK_ALERT} m/s³)",
        "severity": "comfort",
        "code_pointer": "control/pid_controller.py:apply_jerk_limit() (~line 1800)",
        "config_lever": "control.longitudinal.jerk_cooldown_frames",
        "fix_hint": f"Jerk P95 exceeds {CTRL_JERK_ALERT} m/s³ (75% of {CTRL_JERK_GATE} m/s³ comfort gate). Increase jerk_cooldown_frames (8 → 12) or reduce max_jerk_max (4.0 → 3.0).",
        "check": lambda m: m.get("commanded_jerk_p95", 0) > CTRL_JERK_ALERT,
    },
    {
        "id": "ctrl_throttle_surge",
        "name": "Throttle surge (rate > limit >5% frames)",
        "severity": "comfort",
        "code_pointer": "control/pid_controller.py:apply_throttle_rate_limit() (~line 2100)",
        "config_lever": "control.longitudinal.throttle_rate_limit",
        "fix_hint": "Reduce throttle_rate_limit (0.04 → 0.03) for smoother acceleration.",
        "check": lambda m: m.get("throttle_surge_rate", 0) > 0.05,
    },
    {
        "id": "ctrl_steering_jerk",
        "name": "Steering jerk near or above cap",
        "severity": "comfort",
        "code_pointer": "control/pid_controller.py:apply_steering_jerk_limit() (~line 1650)",
        "config_lever": "control.lateral.pp_max_steering_jerk",
        "fix_hint": "If near cap (>17): system is healthy, cap working. If above 20: reduce pp_feedback_gain.",
        "check": lambda m: m.get("steering_jerk_max", 0) > 17.0,
    },
    {
        "id": "ctrl_lateral_oscillation",
        "name": "Lateral oscillation (positive osc_slope)",
        "severity": "instability",
        "code_pointer": "control/pid_controller.py:pure_pursuit_steer() (~line 1400)",
        "config_lever": "control.lateral.pp_feedback_gain / trajectory.mpc.mpc_r_steer_rate_scheduling_enabled",
        "fix_hint": (
            "If PP-mode: reduce pp_feedback_gain (0.10→0.08) or increase reference_smoothing. "
            "If MPC-mode at speed: enable r_steer_rate scheduling (mpc_r_steer_rate_scheduling_enabled: true). "
            "Scheduling is curvature-gated. Also verify mpc_elat_ramp_rate_m_per_frame is 0.0 (ramp limiter causes limit cycles)."
        ),
        "check": lambda m: m.get("osc_slope", -1) > 0.0,
    },
    {
        "id": "traj_curvature_error",
        "name": "Curvature measurement outliers (>2% frames)",
        "severity": "comfort",
        "code_pointer": "trajectory/utils.py:estimate_curvature() (~line 120)",
        "config_lever": "trajectory.curve_context_curvature_gain",
        "fix_hint": "High curvature outlier rate suggests noisy perception input. Check lane polynomial quality.",
        "check": lambda m: m.get("curvature_outlier_rate", 0) > 0.02,
    },
    # MPC patterns
    {
        "id": "mpc_infeasible_burst",
        "name": "MPC infeasible QP bursts (>0.5% frames)",
        "severity": "instability",
        "code_pointer": "control/mpc_controller.py:MPCSolver.solve()",
        "config_lever": "mpc_q_lat / mpc_r_steer_rate (note: r_steer_rate may be speed-scheduled — check mpc_r_steer_rate_effective in HDF5)",
        "fix_hint": "MPC QP solver returning infeasible. Check constraint bounds or reduce q_lat to widen feasible region. If r_steer_rate scheduling is active, high effective rate may over-constrain the QP.",
        "check": lambda m: m.get("mpc_available", False) and m.get("mpc_infeasible_rate", 0) > 0.005,
    },
    {
        "id": "mpc_solve_time_budget",
        "name": "MPC solve time exceeds budget (P95 > 5ms)",
        "severity": "instability",
        "code_pointer": "control/mpc_controller.py:MPCSolver.solve()",
        "config_lever": "mpc_horizon / mpc_max_iter",
        "fix_hint": "MPC P95 solve time > 5ms. Reduce horizon or max_iter to stay within real-time budget.",
        "check": lambda m: m.get("mpc_available", False) and m.get("mpc_solve_p95_ms", 0) > 5.0,
    },
    {
        "id": "mpc_fallback_cascade",
        "name": "MPC fallback to PP active (>0.5% frames)",
        "severity": "safety",
        "code_pointer": "control/mpc_controller.py:MPCController.compute()",
        "config_lever": "mpc_max_consecutive_failures",
        "fix_hint": "MPC falling back to PP. Check infeasibility source — may need constraint relaxation.",
        "check": lambda m: m.get("mpc_available", False) and m.get("mpc_fallback_rate", 0) > 0.005,
    },
    {
        "id": "regime_budget_exceeded",
        "name": "MPC active beyond lateral accel budget (>1% MPC frames)",
        "severity": "instability",
        "code_pointer": "control/regime_selector.py:RegimeSelector._compute_desired()",
        "config_lever": "regime.mpc_max_lateral_accel_mps2 / regime.mpc_lateral_accel_hysteresis_mps2",
        "fix_hint": "MPC active while κ×v² exceeds budget. Lateral demand exceeds linear tire region. Check hold timer or lower budget threshold.",
        "check": lambda m: m.get("mpc_available", False) and m.get("regime_budget_exceeded_rate", 0) > 0.01,
    },
    {
        "id": "mpc_regime_chatter",
        "name": "Regime chatter (>6 switches/min)",
        "severity": "comfort",
        "code_pointer": "control/regime_selector.py:RegimeSelector.update()",
        "config_lever": "regime.mpc_lateral_accel_hysteresis_mps2",
        "fix_hint": "Frequent PP↔MPC switches. Widen hysteresis band or increase blend ramp duration.",
        "check": lambda m: m.get("mpc_available", False) and m.get("mpc_regime_changes_per_min", 0) > 6.0,
    },
    {
        "id": "mpc_heading_error_exceeds_model",
        "name": "MPC heading error P95 > 0.25 rad",
        "severity": "instability",
        "code_pointer": "control/mpc_controller.py:MPCSolver._build_dynamics()",
        "config_lever": "mpc_q_heading",
        "fix_hint": "MPC heading error exceeds kinematic model validity. Check ground-truth sign convention or increase q_heading.",
        "check": lambda m: m.get("mpc_available", False) and m.get("mpc_heading_error_p95", 0) > 0.25,
    },
    {
        "id": "mpc_kappa_ref_mismatch",
        "name": "MPC kappa vs trajectory curvature mismatch (P95 > 0.01)",
        "severity": "instability",
        "code_pointer": "control/mpc_controller.py:MPCController._get_kappa_ref()",
        "config_lever": "mpc_kappa_source",
        "fix_hint": "MPC curvature reference diverges from trajectory path curvature. Check kappa feedforward pipeline.",
        "check": lambda m: m.get("mpc_available", False) and m.get("mpc_kappa_divergence_p95", 0) > 0.01,
    },
    # NMPC-specific patterns (regime == 2, Step 5)
    {
        "id": "nmpc_infeasible_burst",
        "name": "NMPC SLSQP infeasible (>0.5% of NMPC frames)",
        "severity": "instability",
        "code_pointer": "control/nmpc_controller.py:NMPCSolver.solve()",
        "config_lever": "nmpc_max_iter / nmpc_horizon",
        "fix_hint": "NMPC SLSQP infeasible. Try reducing nmpc_horizon or loosening constraint bounds (delta_max, rate).",
        "check": lambda m: m.get("nmpc_available", False) and m.get("nmpc_infeasible_rate", 0) > 0.005,
    },
    {
        "id": "nmpc_solve_time_budget",
        "name": "NMPC solve time exceeds budget (P95 > 20ms)",
        "severity": "instability",
        "code_pointer": "control/nmpc_controller.py:NMPCSolver.solve()",
        "config_lever": "nmpc_horizon / nmpc_max_iter",
        "fix_hint": "NMPC P95 solve time > 20ms. Reduce nmpc_horizon or nmpc_max_iter to stay within real-time budget.",
        "check": lambda m: m.get("nmpc_available", False) and m.get("nmpc_solve_p95_ms", 0) > 20.0,
    },
    {
        "id": "nmpc_fallback_cascade",
        "name": "NMPC fallback to LMPC active (>0.5% NMPC frames)",
        "severity": "instability",
        "code_pointer": "control/nmpc_controller.py:NMPCController.compute_steering()",
        "config_lever": "nmpc_max_consecutive_failures / nmpc_fallback_lmpc",
        "fix_hint": "NMPC falling back to LMPC. Check infeasibility source or increase nmpc_max_consecutive_failures.",
        "check": lambda m: m.get("nmpc_available", False) and m.get("nmpc_fallback_rate", 0) > 0.005,
    },
    # Heading gate / SignalIntegrity (Step 5)
    {
        "id": "heading_gate_stuck_on_curve",
        "name": "Heading-zero gate stuck on curve approach (SignalIntegrity penalty)",
        "severity": "instability",
        "code_pointer": "trajectory/inference.py:_update_heading_zero_gate()",
        "config_lever": "trajectory.traj_heading_zero_gate_curvature_preview_off_threshold",
        "fix_hint": (
            "Gate suppressing heading on >20% of curve-approach frames (κ>0.0005). "
            "ON thresholds: center_a<0.004 AND heading<2°. "
            "OFF conditions: center_a>0.008, heading>3.5°, map_preview_κ>0.001, or time_to_curve<1.35s. "
            "For gentle curves (autobahn R600, κ=0.00167): all OFF conditions fail until well inside the curve. "
            "Fix: lower traj_heading_zero_gate_curvature_preview_off_threshold (0.001→0.0005) "
            "so the map preview fires earlier."
        ),
        "check": lambda m: m.get("heading_suppression_rate_on_curves", 0.0) > 0.20,
    },
    # Grade compensation patterns
    {
        "id": "grade_overspeed_downhill",
        "name": "Downhill overspeed (>5% of downhill frames)",
        "severity": "safety",
        "code_pointer": "av_stack/orchestrator.py:_pf_run_speed_governor()",
        "config_lever": "control.grade_ema_alpha / trajectory.mpc.mpc_grade_clamp_rad",
        "fix_hint": "Increase brake authority on downhill. Check grade_ema_alpha smoothing or grade_clamp_rad limit.",
        "check": lambda m: m.get("grade_available", False) and m.get("downhill_overspeed_rate", 0) > 0.05,
    },
    {
        "id": "grade_compensation_missing",
        "name": "Grade compensation inactive on graded terrain",
        "severity": "instability",
        "code_pointer": "av_stack/orchestrator.py:_pf_validate_frame()",
        "config_lever": "control.grade_ema_alpha",
        "fix_hint": "Grade > 2% but compensation active < 50%. Verify grade_rad is passed to controller.",
        "check": lambda m: m.get("grade_available", False) and m.get("grade_max_pct", 0) > 2.0 and m.get("grade_comp_active_rate", 1.0) < 0.50,
    },
    {
        "id": "grade_mpc_infeasible",
        "name": "MPC infeasible on graded terrain",
        "severity": "instability",
        "code_pointer": "control/mpc_controller.py:MPCSolver.solve()",
        "config_lever": "trajectory.mpc.mpc_grade_clamp_rad",
        "fix_hint": "Grade > 2% and MPC infeasibility > 5%. Increase grade_clamp_rad or MPC accel budget.",
        "check": lambda m: (m.get("grade_available", False) and m.get("grade_max_pct", 0) > 2.0
                            and m.get("mpc_available", False) and m.get("mpc_infeasible_rate", 0) > 0.05),
    },
    # Grade E2E diagnostic patterns (Step 3B)
    {
        "id": "grade_pitch_signal_dead",
        "name": "Pitch telemetry dead while road grade has signal",
        "severity": "instability",
        "code_pointer": "unity/AVSimulation/Assets/Scripts/AVBridge.cs:pitchRad",
        "config_lever": "N/A — Unity WheelCollider suspension keeps chassis level",
        "fix_hint": "pitch_rad variance ~0 while road_grade has signal. Using roadGrade as authoritative source.",
        "check": lambda m: m.get("grade_available", False) and m.get("pitch_signal_dead", False),
    },
    {
        "id": "grade_speed_below_target",
        "name": "Speed >2 m/s below target for >30% of frames",
        "severity": "comfort",
        "code_pointer": "control/pid_controller.py:LongitudinalController",
        "config_lever": "control.longitudinal.grade_ff_gain",
        "fix_hint": "Speed consistently below target. Check grade compensation or throttle authority. Increase grade_ff_gain.",
        "check": lambda m: m.get("speed_below_target_rate", 0) > 0.30,
    },
    {
        "id": "grade_throttle_budget_starved",
        "name": "Throttle budget starved on grade",
        "severity": "instability",
        "category": "Control",
        "config_lever": "control.longitudinal.max_accel / control.longitudinal.grade_ff_gain",
        "fix_hint": "Grade gravity consuming >50% of max_accel budget. Check effective_max_accel computation.",
        "check": lambda m: m.get("grade_available", False) and m.get("grade_throttle_budget_min", 1.0) < 0.5,
    },
    {
        "id": "grade_planner_false_cap",
        "name": "Speed planner false cap on straight",
        "severity": "instability",
        "category": "Trajectory",
        "config_lever": "trajectory.curve_speed_preview_min_curvature",
        "fix_hint": "Speed planner capping on straight segments. Check curve preview distance-to-curve threading.",
        "check": lambda m: m.get("planner_false_cap_rate", 0) > 0.10,
    },
    {
        "id": "grade_lateral_osc_growth",
        "name": "Lateral oscillation growth on grade",
        "severity": "instability",
        "category": "Control",
        "config_lever": "control.lateral.grade_steering_damping_gain",
        "fix_hint": "Lateral error variance growing 2x+ on graded segments. Check grade-proportional steering damping.",
        "check": lambda m: m.get("grade_available", False) and m.get("grade_osc_growth", 0) > 2.0,
    },
    {
        "id": "flat_oscillation_not_grade",
        "name": "Oscillation in flat/straight segments — grade damping is wrong lever",
        "severity": "instability",
        "category": "Control",
        "config_lever": "control.lateral.pp_feedback_gain / MPC regime stability",
        "fix_hint": (
            "Flat-section lateral P95 is 50%+ higher than grade sections — oscillation is "
            "NOT grade-driven. Grade damping changes will regress flat performance. "
            "Investigate: (1) high-speed PP instability on straights, "
            "(2) forced MPC→PP regime resets (see forced_pp_transition_count), "
            "(3) PP feedback gain / lookahead at target speed."
        ),
        "check": lambda m: (
            m.get("grade_available", False)
            and m.get("flat_worse_than_grade_ratio", 0.0) > 1.5
            and m.get("grade_lat_p95_m", 1.0) < 0.30  # grade is actually OK
        ),
    },
    {
        "id": "forced_pp_regime_reset",
        "name": "Forced MPC→PP reset at high speed (false teleport guard)",
        "severity": "instability",
        "category": "Control",
        "config_lever": "transport continuity / teleport discontinuity guard",
        "fix_hint": (
            "Regime selector was reset to PP at high speed by the teleport guard firing on "
            "a frame-drop position jump. PP at >10 m/s is unstable. "
            "Fix: (1) verify coherent packet continuity and skipped-frame timing, "
            "(2) ensure the teleport guard only fires on motion inconsistent with expected transport continuity, "
            "(3) do NOT tune PP gains until forced resets disappear."
        ),
        "check": lambda m: m.get("forced_pp_transition_count", 0) > 0,
    },
    {
        "id": "grade_perception_dropout",
        "name": "Perception confidence drop on grade",
        "severity": "instability",
        "category": "Perception",
        "config_lever": "perception (training data)",
        "fix_hint": "Perception confidence 50%+ lower on graded vs flat. Segmentation model may need grade-diverse training data.",
        "check": lambda m: m.get("grade_available", False) and m.get("grade_perception_conf_ratio", 1.0) < 0.5,
    },
    {
        "id": "grade_throttle_saturated",
        "name": "Throttle saturated on grade",
        "severity": "comfort",
        "category": "Control",
        "config_lever": "control.longitudinal.max_accel",
        "fix_hint": "Throttle at >95% for >10% of graded frames. Motor authority insufficient for grade + acceleration.",
        "check": lambda m: m.get("grade_available", False) and m.get("grade_throttle_sat_rate", 0) > 0.10,
    },
    # MPC hunting / oscillation (Step 3D)
    {
        "id": "mpc_steering_oscillation",
        "name": "MPC steering oscillation (hunting)",
        "severity": "instability",
        "category": "Control",
        "code_pointer": "control/mpc_controller.py:MPCSolver.solve()",
        "config_lever": (
            "trajectory.mpc.mpc_r_steer_rate_scheduling_enabled (rate penalty) / "
            "trajectory.mpc.mpc_q_lat / trajectory.mpc.mpc_r_steer_rate"
        ),
        "fix_hint": (
            "MPC sign-changes >30%. Fix: enable r_steer_rate scheduling "
            "(penalizes steering rate at speed, curvature-gated). If already enabled, "
            "increase mpc_r_steer_rate_speed_gain. Also verify mpc_elat_ramp_rate_m_per_frame is 0.0."
        ),
        "check": lambda m: m.get("mpc_available", False) and m.get("mpc_steering_osc_rate", 0) > 0.30,
    },
    # GT boundary corruption (Step 3D)
    {
        "id": "gt_boundary_corrupt",
        "name": "GT boundary values corrupt (>50m)",
        "severity": "safety",
        "category": "Perception",
        "code_pointer": "unity/AVSimulation/Assets/Scripts/GroundTruthReporter.cs",
        "config_lever": "N/A — Unity GT reporter bug at track boundary",
        "fix_hint": "GT boundary values >50m detected. Add sanity clamp in GroundTruthReporter or filter in Python e-stop logic.",
        "check": lambda m: m.get("gt_boundary_corrupt_frames", 0) > 0,
    },
    # Curve late turn-in / lookahead floor rescue (Step 4)
    {
        "id": "traj_curve_late_turn_in",
        "name": "Curve late turn-in (lookahead floor rescue)",
        "severity": "comfort",
        "category": "Trajectory",
        "code_pointer": "trajectory/inference.py:_compute_lookahead()",
        "config_lever": "trajectory.curve_local_commit_distance_ready_m (auto-derived from speed)",
        "fix_hint": (
            "Lookahead drops abruptly at curve onset — COMMIT phase firing late. "
            "Auto-derive raises curve_local_commit_distance_ready_m with target speed. "
            "If still present: check curve_auto_derive=true and target_speed in config."
        ),
        "check": lambda m: m.get("floor_rescue_rate", 0.0) > 0.05,
    },
    {
        "id": "highway_mild_curve_underactivation",
        "name": "Highway mild-curve under-activation",
        "severity": "instability",
        "category": "Trajectory",
        "code_pointer": "trajectory/utils.py: curve activation and lookahead regime",
        "config_lever": "dynamic mild-curve activation / dynamic lookahead shaping",
        "fix_hint": (
            "Mild map-backed curve is present while curve recognizer stays STRAIGHT, "
            "lookahead remains long, and actual lane offset stays small. "
            "This is a reference-geometry problem, not a transport/perception issue. "
            "Lift the fix into dynamic speed+curvature+distance-to-curve activation and lookahead shaping."
        ),
        "check": lambda m: (
            m.get("highway_mild_curve_underactivation_rate", 0.0) > 0.02
            and m.get("highway_mild_curve_reference_geometry_mismatch_on_high_error_rate", 0.0) > 0.50
            and m.get("highway_mild_curve_transport_fallback_overlap_on_high_error_rate", 1.0) < 0.10
            and m.get("highway_mild_curve_poor_perception_overlap_on_high_error_rate", 1.0) < 0.10
        ),
    },
    {
        "id": "highway_preactivation_curve_authority_missing",
        "name": "Highway pre-activation authority missing",
        "severity": "instability",
        "category": "Trajectory",
        "code_pointer": "trajectory/utils.py + av_stack/orchestrator.py: early mild-curve authority before local ENTRY",
        "config_lever": "shared highway pre-activation speed/curvature/distance weighting",
        "fix_hint": (
            "The curve is materially relevant at highway speed before local curve state is authoritative, "
            "but no bounded early authority is active. Add pre-activation lookahead/speed/curvature authority "
            "before re-tuning steering or stretching the active-curve preserve path."
        ),
        "check": lambda m: (
            m.get("highway_preactivation_curve_authority_missing_rate", 0.0) > 0.05
            and m.get("highway_mild_curve_reference_geometry_mismatch_on_high_error_rate", 0.0) > 0.50
            and m.get("highway_mild_curve_transport_fallback_overlap_on_high_error_rate", 1.0) < 0.10
            and m.get("highway_mild_curve_poor_perception_overlap_on_high_error_rate", 1.0) < 0.10
        ),
    },
    {
        "id": "local_curve_reference_shadow_insufficient",
        "name": "Local curve reference shadow-only when authority is needed",
        "severity": "instability",
        "category": "Trajectory",
        "code_pointer": "trajectory/utils.py + av_stack/orchestrator.py",
        "config_lever": "highway local-arc shadow promotion / bounded authority",
        "fix_hint": (
            "The local map-backed arc is being computed, but it remains shadow-only while the planner/local-arc delta is materially large. "
            "Promote the local arc to bounded authority in the highway failure regime before more speed-cap or generic steering tuning."
        ),
        "check": lambda m: (
            m.get("local_curve_reference_shadow_insufficient_rate", 0.0) > 0.05
            and m.get("highway_mild_curve_transport_fallback_overlap_on_high_error_rate", 1.0) < 0.10
            and m.get("highway_mild_curve_poor_perception_overlap_on_high_error_rate", 1.0) < 0.10
        ),
    },
    {
        "id": "wrong_target_distractor_reference_divergence",
        "name": "Rejected distractor still pulls reference ownership",
        "severity": "instability",
        "category": "Trajectory",
        "code_pointer": "av_stack/orchestrator.py + trajectory/utils.py",
        "config_lever": "distractor-guarded bounded local-reference takeover",
        "fix_hint": (
            "A wrong-lane or opposite-direction actor is correctly rejected for ACC, but the planner/local-arc "
            "divergence still grows in the same window. Add a sustained guarded-bounded local-reference backstop "
            "instead of widening target capture or increasing generic steering authority."
        ),
        "check": lambda m: (
            m.get("wrong_target_distractor_reference_divergence_rate", 0.0) > 0.05
            and m.get("highway_mild_curve_transport_fallback_overlap_on_high_error_rate", 1.0) < 0.10
            and m.get("highway_mild_curve_poor_perception_overlap_on_high_error_rate", 1.0) < 0.10
        ),
    },
    {
        "id": "wrong_target_straight_reference_drift",
        "name": "Rejected distractor still bends straight-road reference",
        "severity": "instability",
        "category": "Trajectory",
        "code_pointer": "av_stack/orchestrator.py",
        "config_lever": "straight wrong-target reference guard",
        "fix_hint": (
            "A wrong-lane or opposite-direction actor is correctly rejected, but the straight-road "
            "lane_positions reference still drifts away from the ego lane. Anchor the straight reject "
            "guard to at-car selected-lane GT before increasing generic lateral authority."
        ),
        "check": lambda m: m.get("wrong_target_straight_reference_drift_rate", 0.0) > 0.05,
    },
    {
        "id": "ego_lane_contract_invalid",
        "name": "Selected lane anchor diverges from ego-lane contract",
        "severity": "instability",
        "category": "Trajectory",
        "code_pointer": "unity/AVSimulation/Assets/Scripts/GroundTruthReporter.cs + AVBridge.cs",
        "config_lever": "explicit ego-lane selection contract",
        "fix_hint": (
            "The reject-only guards are anchored to selected-lane GT, but selected-lane semantics diverge from the explicit ego-lane contract. "
            "Export ego-lane selection explicitly and anchor reject-only guards to ego-lane GT before raising authority."
        ),
        "check": lambda m: m.get("ego_lane_contract_invalid_rate", 0.0) > 0.05,
    },
    {
        "id": "high_speed_mild_curve_active_authority_insufficient",
        "name": "High-speed mild-curve active authority insufficient",
        "severity": "instability",
        "category": "Trajectory",
        "code_pointer": "control/speed_governor.py + control/mpc_controller.py",
        "config_lever": "active mild-curve speed feasibility + active-state curve authority",
        "fix_hint": (
            "The mild curve is already locally active, but the active-state speed cap and curvature authority are still too weak. "
            "Tighten active mild-curve feasibility and bound same-sign curvature amplification before changing generic steering gains."
        ),
        "check": lambda m: (
            m.get("high_speed_mild_curve_active_authority_insufficient_rate", 0.0) > 0.05
            and m.get("highway_mild_curve_transport_fallback_overlap_on_high_error_rate", 1.0) < 0.10
            and m.get("highway_mild_curve_poor_perception_overlap_on_high_error_rate", 1.0) < 0.10
        ),
    },
    {
        "id": "active_mild_curve_preserve_speed_gated",
        "name": "Active mild-curve authority still speed-gated",
        "severity": "instability",
        "category": "Trajectory",
        "code_pointer": "control/mpc_controller.py",
        "config_lever": "active mild-curve authority dynamic speed gate",
        "fix_hint": (
            "The curve is already locally active, but the dedicated mild-curve authority path never turns on because "
            "its speed gate starts above the ACC-follow operating band. Lower the dynamic authority on-threshold or "
            "make it state-aware before changing generic steering gains."
        ),
        "check": lambda m: (
            m.get("active_mild_curve_preserve_speed_gated_rate", 0.0) > 0.05
            and m.get("highway_mild_curve_transport_fallback_overlap_on_high_error_rate", 1.0) < 0.10
            and m.get("highway_mild_curve_poor_perception_overlap_on_high_error_rate", 1.0) < 0.10
        ),
    },
    {
        "id": "curve_sustain_collapse_rearm_cycle",
        "name": "Curve sustain collapse / REARM cycling",
        "severity": "instability",
        "category": "Trajectory",
        "code_pointer": "trajectory/utils.py:1116-1196",
        "config_lever": "dynamic sustain effect / ENTRY-to-REARM sustain hysteresis",
        "fix_hint": (
            "Curve arming is active, but sustain remains too weak and the scheduler alternates "
            "ENTRY for ~2 frames then REARM for ~1 frame. Strengthen dynamic sustain for active mild curves "
            "instead of re-tuning track-specific thresholds."
        ),
        "check": lambda m: (
            m.get("curve_sustain_collapse_rearm_cycle_rate", 0.0) > 0.05
            and m.get("highway_mild_curve_transport_fallback_overlap_on_high_error_rate", 1.0) < 0.10
            and m.get("highway_mild_curve_poor_perception_overlap_on_high_error_rate", 1.0) < 0.10
        ),
    },
    {
        "id": "mpc_curvature_bias_cancellation",
        "name": "MPC curvature bias cancellation",
        "severity": "instability",
        "category": "Control",
        "code_pointer": "control/mpc_controller.py:742-764",
        "config_lever": "trajectory.mpc.mpc_bias_* / dynamic bias guard",
        "fix_hint": (
            "MPC receives the correct map/reference curvature, then cancels too much of it with bias correction. "
            "Clamp bias as a fraction of |kappa_ref| for active curves, or disable the highway bias override."
        ),
        "check": lambda m: (
            m.get("mpc_curvature_bias_cancellation_rate", 0.0) > 0.05
            and m.get("highway_mild_curve_transport_fallback_overlap_on_high_error_rate", 1.0) < 0.10
            and m.get("highway_mild_curve_poor_perception_overlap_on_high_error_rate", 1.0) < 0.10
        ),
    },
    {
        "id": "mpc_gt_cross_track_semantic_mismatch",
        "name": "MPC GT cross-track semantic mismatch",
        "severity": "instability",
        "category": "Control",
        "code_pointer": "unity/AVSimulation/Assets/Scripts/AVBridge.cs + av_stack/orchestrator.py",
        "config_lever": "split lookahead GT from at-car GT; keep lookahead debug-only",
        "fix_hint": (
            "MPC high-error windows are dominated by large lookahead GT cross-track while at-car GT stays small. "
            "Do not use preview lane-center geometry as current e_lat. Export and consume a true at-car selected-lane-center signal."
        ),
        "check": lambda m: (
            m.get("mpc_gt_cross_track_semantic_mismatch_rate", 0.0) > 0.05
            and m.get("highway_mild_curve_transport_fallback_overlap_on_high_error_rate", 1.0) < 0.10
            and m.get("highway_mild_curve_poor_perception_overlap_on_high_error_rate", 1.0) < 0.10
        ),
    },
    {
        "id": "mpc_gt_cross_track_vehicle_frame_semantic_mismatch",
        "name": "MPC GT uses vehicle-frame lane-center geometry",
        "severity": "instability",
        "category": "Control",
        "code_pointer": "unity/AVSimulation/Assets/Scripts/GroundTruthReporter.cs + control/pid_controller.py",
        "config_lever": "control must consume road-frame selected-lane cross-track, not vehicle-frame lane-center X",
        "fix_hint": (
            "LMPC is consuming vehicle-frame selected-lane-center geometry as e_lat while also consuming heading error separately. "
            "Export a true road-frame selected-lane cross-track at the car and make that the authoritative control GT."
        ),
        "check": lambda m: (
            m.get("mpc_gt_cross_track_vehicle_frame_semantic_mismatch_rate", 0.0) > 0.05
            and m.get("highway_mild_curve_transport_fallback_overlap_on_high_error_rate", 1.0) < 0.10
            and m.get("highway_mild_curve_poor_perception_overlap_on_high_error_rate", 1.0) < 0.10
        ),
    },
    {
        "id": "mpc_gt_cross_track_absolute_coordinate_mismatch",
        "name": "MPC GT cross-track absolute-coordinate misuse",
        "severity": "instability",
        "category": "Control",
        "code_pointer": "unity/AVSimulation/Assets/Scripts/GroundTruthReporter.cs + av_stack/orchestrator.py",
        "config_lever": "export true at-car lane-center offset; keep reference-path geometry debug-only",
        "fix_hint": (
            "The at-car GT signal is too large to be a current cross-track offset while actual lane-center offset stays small. "
            "Compute at-car GT from the vehicle's current pose on the road, not from the time-based reference path."
        ),
        "check": lambda m: (
            m.get("mpc_gt_cross_track_absolute_coordinate_mismatch_rate", 0.0) > 0.05
            and m.get("highway_mild_curve_transport_fallback_overlap_on_high_error_rate", 1.0) < 0.10
            and m.get("highway_mild_curve_poor_perception_overlap_on_high_error_rate", 1.0) < 0.10
        ),
    },
    # ── Tire estimation patterns (dynamic bicycle MPC) ──────────────────────
    {
        "id": "tire_saturation",
        "name": "Tire saturation (slip angle exceeds linear range)",
        "severity": "instability",
        "category": "Control",
        "code_pointer": "control/mpc_controller.py:_update_tire_ekf()",
        "config_lever": "mpc_tire_ekf_slip_saturation_rad / target speed",
        "fix_hint": "Tire operating beyond linear range. Reduce target speed or widen tire saturation bounds.",
        "check": lambda m: m.get("tire_saturation_rate", 0) > 0.02,
    },
    {
        "id": "tire_ekf_divergence",
        "name": "EKF tire estimation diverging (innovation exceeds threshold)",
        "severity": "instability",
        "category": "Control",
        "code_pointer": "control/mpc_controller.py:_update_tire_ekf()",
        "config_lever": "mpc_tire_ekf_measurement_noise / mpc_tire_ekf_process_noise",
        "fix_hint": "Increase measurement noise R (trust model more) or decrease process noise Q.",
        "check": lambda m: m.get("tire_ekf_divergence_rate", 0) > 0.05,
    },
    {
        "id": "tire_understeer_anomaly",
        "name": "Anomalous understeer gradient (K_us > 0.010)",
        "severity": "comfort",
        "category": "Control",
        "code_pointer": "control/mpc_controller.py:_compute_understeer_gradient()",
        "config_lever": "mpc_tire_cf_nominal / mpc_tire_cr_nominal",
        "fix_hint": "C_f/C_r estimates may have drifted. Consider resetting EKF or adjusting nominal values.",
        "check": lambda m: m.get("tire_understeer_gradient_max", 0) > 0.010,
    },
    # ── ACC patterns (Step 5 — priority-0 when following_too_close) ──────────
    {
        "id": "acc_following_too_close",
        "name": "ACC following too close (TTC warning or near-miss)",
        "severity": "safety",
        "category": "Safety",
        "code_pointer": "control/acc_controller.py:ACCController._idm()",
        "config_lever": "acc.time_headway_s (1.5→2.0); verify radar Doppler σ",
        "fix_hint": (
            "acc_ttc_warning_pct > 5% or near-miss events > 0. "
            "time_headway_s too small for track speed, or radar range-rate noise "
            "causing IDM to underestimate closing speed. "
            "Increase time_headway_s (1.5→2.0) and verify radar SNR threshold."
        ),
        "check": lambda m: m.get("acc_active", False) and (
            m.get("acc_ttc_warning_pct", 0.0) > 5.0
            or m.get("acc_near_miss_events", 0) > 0
        ),
    },
    {
        "id": "acc_gap_tracking_poor",
        "name": "ACC gap tracking poor (RMSE above gate)",
        "severity": "comfort",
        "category": "Control",
        "code_pointer": "control/acc_controller.py:ACCController._idm()",
        "config_lever": "acc.time_headway_s ↑; acc.max_accel_mps2 ↓",
        "fix_hint": (
            "acc_gap_rmse_m > 0.50m. IDM oscillating around desired gap. "
            "Increase time_headway_s for a more conservative desired gap, "
            "or reduce max_accel_mps2 to damp oscillation."
        ),
        "check": lambda m: m.get("acc_active", False) and m.get("acc_gap_rmse_m", 0.0) > 0.50,
    },
    {
        "id": "acc_detection_dropout",
        "name": "ACC radar detection rate below gate",
        "severity": "instability",
        "category": "Perception",
        "code_pointer": "control/radar_sensor.py:ForwardRadarSensor.read_frame()",
        "config_lever": "acc.detection_range_m ↑; check RADAR_HALF_ANGLE_DEG",
        "fix_hint": (
            "acc_detection_rate < 95%. SphereCast FOV too narrow or lead vehicle "
            "occlusion at curve. Increase detection_range_m or widen the half-angle "
            "in the Unity RadarSensor component."
        ),
        "check": lambda m: m.get("acc_active", False) and m.get("acc_detection_rate", 1.0) < 0.95,
    },
    {
        "id": "acc_idm_hunting",
        "name": "ACC IDM hunting (gap error sign oscillation)",
        "severity": "comfort",
        "category": "Control",
        "code_pointer": "control/acc_controller.py:ACCController._idm()",
        "config_lever": "acc.time_headway_s ↑; acc.comfortable_decel_mps2 ↑",
        "fix_hint": (
            "Gap error sign reverses > 20 times/min in ACC-active frames. "
            "IDM is underdamped — asymmetry between comfortable_decel_mps2 and "
            "max_accel_mps2 is driving oscillation. "
            "Increase time_headway_s and comfortable_decel_mps2."
        ),
        "check": lambda m: m.get("acc_active", False) and m.get("acc_gap_sign_changes_per_min", 0.0) > 20.0,
    },
    # ── Inter-frame control extrapolation ────────────────────────────────────
    {
        "id": "interframe_e_lat_divergence",
        "name": "Inter-frame GT diverges from camera GT (e_lat drift between frames)",
        "severity": "instability",
        "category": "Control",
        "code_pointer": "av_stack/orchestrator.py:_run_interframe_update()",
        "config_lever": "control.interframe_extrapolation.max_e_lat_jump_m ↓; check GT freshness",
        "fix_hint": (
            "Inter-frame e_lat diverges > 0.2m from the next camera frame's GT. "
            "The lightweight MPC is using stale curvature or the GT signal is noisy. "
            "Check interframe_last_e_lat vs next camera mpc_e_lat. "
            "Reduce max_interframe_updates or tighten stale_vehicle_state_ms."
        ),
        "check": lambda m: (
            m.get("interframe_active_pct", 0.0) > 5.0
            and m.get("interframe_e_lat_divergence_mean", 0.0) > 0.2
        ),
    },
    {
        "id": "camera_pipeline_bottleneck",
        "name": "Camera pipeline bottleneck (inter-frame at max every cycle)",
        "severity": "instability",
        "category": "Control",
        "code_pointer": "av_stack/orchestrator.py:run() loop",
        "config_lever": "Investigate camera delivery rate; increase max_interframe_updates",
        "fix_hint": (
            "Inter-frame updates are hitting max_interframe_updates on > 80% of "
            "camera cycles, meaning the camera pipeline is consistently slow. "
            "This is expected at 7.5 Hz camera rate with max_updates=3. "
            "If rate improves, this pattern should disappear. "
            "Consider increasing max_interframe_updates if control quality is still poor."
        ),
        "check": lambda m: (
            m.get("interframe_active_pct", 0.0) > 5.0
            and m.get("interframe_at_max_pct", 0.0) > 80.0
        ),
    },
    {
        "id": "interframe_rate_throttled",
        "name": "Inter-frame rate throttled (effective rate << target)",
        "severity": "instability",
        "category": "Control",
        "code_pointer": "av_stack/orchestrator.py:_run_interframe_update()",
        "config_lever": "Check for time.sleep() inside _run_interframe_update(); verify min_interframe_dt_s",
        "fix_hint": (
            "Inter-frame is active but effective rate is below 60% of target "
            "(camera_hz × 4). Common cause: a time.sleep() inside the inter-frame "
            "function using camera frame_interval instead of min_interframe_dt_s. "
            "The main loop rate limiter should own all timing — inter-frame functions "
            "should return immediately."
        ),
        "check": lambda m: (
            m.get("interframe_active_pct", 0.0) > 5.0
            and m.get("interframe_at_max_pct", 0.0) < 10.0
            and m.get("interframe_rate_ratio", 1.0) < 0.6
        ),
    },
    {
        "id": "ekf_measurement_degraded",
        "name": "EKF yaw rate measurement degraded (IMU signal missing or derived fallback)",
        "severity": "instability",
        "category": "Control",
        "code_pointer": "control/mpc_controller.py:_update_tire_ekf()",
        "config_lever": "mpc_tire_ekf_use_imu_yaw_rate / mpc_tire_ekf_imu_yaw_rate_r",
        "fix_hint": (
            "IMU yaw rate signal missing or using derived fallback for >5% of MPC frames. "
            "Check angular_velocity pipeline from Unity. Verify angularVelocity.y is non-zero "
            "in vehicle state packets."
        ),
        "check": lambda m: (
            m.get("imu_yaw_rate_missing_pct", 0.0) > 5.0
        ),
    },
    {
        "id": "geometry_override_missing",
        "name": "Dynamic model running without Unity geometry override (symmetric fallback)",
        "severity": "instability",
        "category": "Control",
        "code_pointer": "av_stack/orchestrator.py:_apply_unity_geometry()",
        "config_lever": "mpc_vehicle_lf_m / mpc_vehicle_lr_m (fallback defaults)",
        "fix_hint": (
            "Unity did not send dynamicModelParamsReady=true. "
            "MPC using l_f=l_r=1.125m → K_us=0 → EKF drives C_f/C_r to bounds. "
            "Check CarController.ComputeDynamicModelParams() is called in FixedUpdate."
        ),
        "check": lambda m: (
            m.get("dynamic_model_active_pct", 0.0) > 50.0
            and not m.get("geometry_override_active", True)
        ),
    },
]

SEVERITY_ORDER = {"safety": 0, "instability": 1, "comfort": 2}


@dataclass
class PatternMatch:
    pattern_id:        str
    name:              str
    severity:          str
    occurrences:       int
    pct_of_recording:  float
    code_pointer:      str
    config_lever:      str
    fix_hint:          str
    example_frames:    list


class TriageEngine:
    def __init__(self, recording_path: Path) -> None:
        self.path = recording_path

    def generate_triage(self) -> dict:
        """Compute full triage report. Returns JSON-ready dict."""
        metrics = self._extract_aggregate_metrics()
        health_summary = self._get_layer_health_summary()
        stale_data = self._get_stale_events()

        matched = self._match_patterns(metrics)
        attribution = self._compute_attribution(health_summary)
        checklist = self._build_checklist(matched)
        overall = self._overall_health(health_summary)

        return {
            "recording_id":     self.path.stem,
            "total_frames":     metrics.get("total_frames", 0),
            "attribution":      attribution,
            "overall_health":   round(overall, 3),
            "stale_event_count": stale_data.get("event_count", 0),
            "matched_patterns": [
                {
                    "pattern_id":       m.pattern_id,
                    "name":             m.name,
                    "severity":         m.severity,
                    "occurrences":      m.occurrences,
                    "pct_of_recording": round(m.pct_of_recording, 4),
                    "code_pointer":     m.code_pointer,
                    "config_lever":     m.config_lever,
                    "fix_hint":         m.fix_hint,
                    "example_frames":   m.example_frames,
                }
                for m in matched
            ],
            "action_checklist": checklist,
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _extract_aggregate_metrics(self) -> dict:
        """Read HDF5 and compute aggregate metrics needed by pattern checks."""
        m: dict = {}
        with h5py.File(self.path, 'r') as f:
            def arr(key):
                if key not in f:
                    return None
                return np.array(f[key][:])

            n = self._frame_count(f)
            m["total_frames"] = n

            # Perception
            conf = arr("perception/confidence")
            stale = arr("perception/using_stale_data")   # actual field name
            n_lanes = arr("perception/num_lanes_detected")
            if conf is not None:
                m["mean_perception_confidence"] = float(np.nanmean(conf))
            if stale is not None:
                m["stale_rate"] = float(np.mean(stale.astype(bool)))
            if n_lanes is not None:
                m["single_lane_rate"] = float(np.mean(n_lanes == 1))
            blind = arr("perception/blind_active")
            if blind is not None:
                m["blind_perception_rate"] = float(np.mean(blind.astype(bool)))

            # Trajectory
            clamped = arr("trajectory/diag_ref_x_rate_limit_active")  # actual field name
            lookahead = arr("control/pp_lookahead_distance")           # lives in control group
            if clamped is not None:
                m["ref_rate_clamp_rate"] = float(np.mean(clamped.astype(bool)))
            if lookahead is not None:
                m["short_lookahead_rate"] = float(np.mean(lookahead < LOOKAHEAD_CONCERN_M))

            # Control
            lat_err = arr("control/lateral_error")
            throttle = arr("control/throttle")
            brake = arr("control/brake")
            control_ts = arr("control/timestamps")
            if control_ts is None:
                control_ts = arr("control/timestamp")
            jerk_cap = arr("control/longitudinal_jerk_capped")
            accel_cap = arr("control/longitudinal_accel_capped")
            steer_jerk_delta = arr("control/steering_jerk_limited_delta")
            estop = arr("control/emergency_stop")
            if lat_err is not None:
                abs_err = np.abs(lat_err)
                m["lateral_p95"] = float(np.percentile(abs_err, 95))
                x = np.arange(len(abs_err))
                m["osc_slope"] = float(np.polyfit(x, abs_err, 1)[0])
            max_accel, max_decel = _load_longitudinal_limits()
            m["commanded_jerk_p95"] = _compute_commanded_jerk_p95(
                throttle=throttle,
                brake=brake,
                timestamps=control_ts,
                max_accel=max_accel,
                max_decel=max_decel,
            )
            if jerk_cap is not None:
                m["longitudinal_jerk_cap_rate"] = float(np.mean(np.abs(jerk_cap) > 0.5))
            if accel_cap is not None:
                m["longitudinal_accel_cap_rate"] = float(np.mean(np.abs(accel_cap) > 0.5))
            if steer_jerk_delta is not None:
                # steering jerk max = largest single-frame steering change applied by limiter
                m["steering_jerk_max"] = float(np.max(np.abs(steer_jerk_delta)))
            # out_of_lane: count rising edges on emergency_stop as proxy
            if estop is not None:
                diff = np.diff(estop.astype(int))
                m["out_of_lane_events"] = int(np.sum(diff > 0))

            # MPC metrics
            regime = arr("control/regime")
            if regime is not None:
                mpc_mask = regime >= 0.5
                m["mpc_available"] = True
                m["mpc_active_rate"] = float(np.mean(mpc_mask)) if len(mpc_mask) > 0 else 0.0

                mpc_feasible = arr("control/mpc_feasible")
                m["mpc_infeasible_rate"] = (
                    float(1.0 - np.mean(mpc_feasible[mpc_mask]))
                    if mpc_feasible is not None and np.any(mpc_mask) else 0.0
                )

                mpc_solve = arr("control/mpc_solve_time_ms")
                m["mpc_solve_p95_ms"] = (
                    float(np.percentile(mpc_solve[mpc_mask], 95))
                    if mpc_solve is not None and np.any(mpc_mask) else 0.0
                )

                mpc_fallback = arr("control/mpc_fallback_active")
                m["mpc_fallback_rate"] = (
                    float(np.mean(mpc_fallback[mpc_mask]))
                    if mpc_fallback is not None and np.any(mpc_mask) else 0.0
                )

                mpc_failures = arr("control/mpc_consecutive_failures")
                m["mpc_max_consecutive_failures"] = (
                    int(np.max(mpc_failures)) if mpc_failures is not None else 0
                )

                # Lateral accel budget exceeded rate
                a_lat_arr = arr("control/regime_lateral_accel_mps2")
                a_thr_arr = arr("control/regime_lateral_accel_threshold_mps2")
                if a_lat_arr is not None and a_thr_arr is not None and np.any(mpc_mask):
                    exceeded = a_lat_arr[mpc_mask] > a_thr_arr[mpc_mask]
                    m["regime_budget_exceeded_rate"] = float(np.mean(exceeded))
                else:
                    m["regime_budget_exceeded_rate"] = 0.0

                # Regime chatter
                regime_changes = int(np.sum(np.diff(regime) != 0)) if len(regime) > 1 else 0
                duration_s = len(regime) * 0.033
                m["mpc_regime_changes_per_min"] = float(
                    regime_changes / max(duration_s / 60.0, 0.001)
                )

                # Heading error P95
                mpc_e_heading = arr("control/mpc_e_heading")
                m["mpc_heading_error_p95"] = (
                    float(np.percentile(np.abs(mpc_e_heading[mpc_mask]), 95))
                    if mpc_e_heading is not None and np.any(mpc_mask) else 0.0
                )

                # Kappa divergence
                mpc_kappa = arr("control/mpc_kappa_ref")
                path_kappa = arr("trajectory/path_curvature")
                if mpc_kappa is not None and path_kappa is not None and np.any(mpc_mask):
                    ml = min(len(mpc_kappa), len(path_kappa))
                    m["mpc_kappa_divergence_p95"] = float(
                        np.percentile(
                            np.abs(mpc_kappa[:ml] - path_kappa[:ml])[mpc_mask[:ml]], 95
                        )
                    )
                else:
                    m["mpc_kappa_divergence_p95"] = 0.0
            else:
                m["mpc_available"] = False

            # NMPC metrics (regime == 2)
            nmpc_mask = regime >= 1.5 if regime is not None else None
            if nmpc_mask is not None and np.any(nmpc_mask):
                m["nmpc_available"] = True

                nmpc_feasible = arr("control/nmpc_feasible")
                m["nmpc_infeasible_rate"] = (
                    float(1.0 - np.mean(nmpc_feasible[nmpc_mask]))
                    if nmpc_feasible is not None else 0.0
                )

                nmpc_solve = arr("control/nmpc_solve_time_ms")
                m["nmpc_solve_p95_ms"] = (
                    float(np.percentile(nmpc_solve[nmpc_mask], 95))
                    if nmpc_solve is not None else 0.0
                )

                nmpc_fallback = arr("control/nmpc_fallback_active")
                m["nmpc_fallback_rate"] = (
                    float(np.mean(nmpc_fallback[nmpc_mask]))
                    if nmpc_fallback is not None else 0.0
                )
            else:
                m["nmpc_available"] = False

            # Tire estimation metrics (dynamic bicycle MPC)
            tire_slip_f = arr("control/mpc_tire_slip_angle_front")
            tire_slip_r = arr("control/mpc_tire_slip_angle_rear")
            tire_innov = arr("control/mpc_tire_ekf_innovation")
            tire_us = arr("control/mpc_tire_understeer_gradient")
            dyn_active = arr("control/mpc_dynamic_model_active")

            if dyn_active is not None and np.any(dyn_active > 0.5):
                dyn_mask = dyn_active > 0.5
                # Saturation rate
                if tire_slip_f is not None and tire_slip_r is not None:
                    sat = (np.abs(tire_slip_f[dyn_mask]) > TIRE_SLIP_ANGLE_SATURATION_RAD) | (
                        np.abs(tire_slip_r[dyn_mask]) > TIRE_SLIP_ANGLE_SATURATION_RAD
                    )
                    m["tire_saturation_rate"] = float(np.mean(sat)) if len(sat) > 0 else 0.0
                # EKF divergence rate
                if tire_innov is not None:
                    div = np.abs(tire_innov[dyn_mask]) > TIRE_EKF_INNOVATION_DIVERGENCE
                    m["tire_ekf_divergence_rate"] = float(np.mean(div)) if len(div) > 0 else 0.0
                # Understeer gradient max
                if tire_us is not None:
                    m["tire_understeer_gradient_max"] = float(np.max(tire_us[dyn_mask]))

            # Heading gate suppression on curves (SignalIntegrity, Step 5)
            # Uses approach_threshold=0.0005 to match drive_summary_core.py scoring.
            hzg = arr("trajectory/diag_heading_zero_gate_active")
            traj_curv = arr("trajectory/reference_point_curvature")
            if hzg is not None and traj_curv is not None:
                _n = min(len(hzg), len(traj_curv))
                _gate = (hzg[:_n] > 0.5)
                _on_curve = np.abs(traj_curv[:_n]) > 0.0005
                _curve_frames = int(np.sum(_on_curve))
                m["heading_suppression_rate_on_curves"] = float(
                    np.sum(_gate & _on_curve) / max(_curve_frames, 1)
                )
            else:
                m["heading_suppression_rate_on_curves"] = 0.0

            # Grade metrics
            road_grade = arr("vehicle/road_grade")
            if road_grade is not None:
                m["grade_available"] = True
                m["grade_max_pct"] = float(np.max(np.abs(road_grade))) * 100.0
                downhill_mask = road_grade < -0.02
                speed = arr("vehicle/speed")
                speed_limit = arr("vehicle/speed_limit")
                if np.any(downhill_mask) and speed is not None and speed_limit is not None:
                    dh_speed = speed[downhill_mask]
                    dh_limit = speed_limit[downhill_mask]
                    m["downhill_overspeed_rate"] = float(np.mean(dh_speed > dh_limit + 1.0))
                else:
                    m["downhill_overspeed_rate"] = 0.0
                grade_comp = arr("control/grade_compensation_active")
                graded_mask = np.abs(road_grade) > 0.001
                graded_count = int(np.sum(graded_mask))
                if graded_count > 0 and grade_comp is not None:
                    m["grade_comp_active_rate"] = float(np.mean(grade_comp[graded_mask] > 0.5))
                else:
                    m["grade_comp_active_rate"] = 1.0
                # Throttle budget on grade (Step 3C)
                eff_max_accel = arr("control/effective_max_accel")
                grade_throttle = arr("control/throttle")
                graded = np.abs(road_grade) > 0.02
                if np.any(graded):
                    if eff_max_accel is not None and grade_throttle is not None:
                        gravity_load = 9.81 * np.abs(np.sin(road_grade[graded]))
                        budget = (eff_max_accel[graded] - gravity_load) / max(float(m.get("max_accel", 1.2)), 0.1)
                        m["grade_throttle_budget_min"] = float(np.min(budget))
                        m["grade_throttle_sat_rate"] = float(np.mean(grade_throttle[graded] > 0.95))

                # Speed planner false cap on straight
                limiter = arr("control/speed_governor_active_limiter_code")
                curv_preview = arr("control/curvature_preview_abs")
                grade_speed = arr("vehicle/speed")
                if limiter is not None and curv_preview is not None and grade_speed is not None:
                    planner_on_straight = (limiter == 5) & (curv_preview < 0.003) & (grade_speed < 10.0)
                    m["planner_false_cap_rate"] = float(np.mean(planner_on_straight))

                # Windowed lateral oscillation on grade vs flat
                grade_lat_err = arr("control/lateral_error")
                flat_mask = np.abs(road_grade) <= 0.02
                if grade_lat_err is not None and np.any(graded) and np.sum(graded) > 60:
                    graded_err = np.abs(grade_lat_err[graded])
                    win = 20
                    if len(graded_err) > 2 * win:
                        vars_early = float(np.var(graded_err[:win]))
                        vars_late = float(np.var(graded_err[-win:]))
                        m["grade_osc_growth"] = float(vars_late / max(vars_early, 1e-6))
                    # Compare grade vs flat P95 error — if flat >> grade, damping is wrong lever
                    if np.any(flat_mask) and np.sum(flat_mask) > 30:
                        flat_err = np.abs(grade_lat_err[flat_mask])
                        grade_p95 = float(np.percentile(graded_err, 95))
                        flat_p95 = float(np.percentile(flat_err, 95))
                        m["flat_lat_p95_m"] = flat_p95
                        m["grade_lat_p95_m"] = grade_p95
                        # ratio > 1.5 means flat is 50%+ worse than grade → grade damping is not the fix
                        m["flat_worse_than_grade_ratio"] = float(flat_p95 / max(grade_p95, 0.01))

                # Per-wheel diagnostics (Step 3D)
                wheel_force = arr("vehicle/wheel_contact_force")
                sideways_slip = arr("vehicle/wheel_sideways_slip")
                contact_normal_y = arr("vehicle/wheel_contact_normal_y")
                if wheel_force is not None and wheel_force.ndim == 2 and wheel_force.shape[1] == 4:
                    front_force = (wheel_force[:, 0] + wheel_force[:, 1]) / 2
                    rear_force = (wheel_force[:, 2] + wheel_force[:, 3]) / 2
                    m["wheel_force_front_rear_ratio"] = float(
                        np.mean(front_force / np.maximum(rear_force, 1.0)))
                if contact_normal_y is not None and contact_normal_y.ndim == 2 and np.any(graded):
                    graded_normals = contact_normal_y[graded]
                    m["grade_contact_normal_y_mean"] = float(np.mean(graded_normals))
                if sideways_slip is not None and sideways_slip.ndim == 2:
                    left_slip = (sideways_slip[:, 0] + sideways_slip[:, 2]) / 2
                    right_slip = (sideways_slip[:, 1] + sideways_slip[:, 3]) / 2
                    m["lateral_slip_asymmetry_p95"] = float(
                        np.percentile(np.abs(left_slip - right_slip), 95))

                # MPC steering oscillation on grade (Step 3D)
                regime = arr("control/regime")
                ctrl_steer = arr("control/steering")
                if regime is not None and ctrl_steer is not None:
                    mpc_mask = regime >= 0.5
                    if np.sum(mpc_mask) > 10:
                        mpc_steer = ctrl_steer[mpc_mask]
                        sign_changes = np.sum(np.diff(np.sign(mpc_steer)) != 0)
                        m["mpc_steering_osc_rate"] = float(sign_changes / len(mpc_steer))

                # Perception on grade vs flat
                grade_conf = arr("perception/confidence")
                flat = np.abs(road_grade) <= 0.02
                if grade_conf is not None and np.any(graded) and np.any(flat):
                    m["grade_perception_conf_ratio"] = float(
                        np.mean(grade_conf[graded]) / max(float(np.mean(grade_conf[flat])), 0.01))

            else:
                m["grade_available"] = False

            # GT boundary sanity check (Step 3D)
            for gt_key in ("vehicle/gt_left_boundary", "vehicle/gt_right_boundary",
                           "vehicle/groundTruthLeftBoundary", "vehicle/groundTruthRightBoundary"):
                gt_vals = arr(gt_key)
                if gt_vals is not None:
                    corrupt_count = int(np.sum(np.abs(gt_vals) > 50.0))
                    m["gt_boundary_corrupt_frames"] = m.get("gt_boundary_corrupt_frames", 0) + corrupt_count

            # MPC steering oscillation (non-grade context — Step 3D)
            if "mpc_steering_osc_rate" not in m:
                regime = arr("control/regime")
                ctrl_steer = arr("control/steering")
                if regime is not None and ctrl_steer is not None:
                    mpc_mask = regime >= 0.5
                    if np.sum(mpc_mask) > 10:
                        mpc_steer = ctrl_steer[mpc_mask]
                        sign_changes = np.sum(np.diff(np.sign(mpc_steer)) != 0)
                        m["mpc_steering_osc_rate"] = float(sign_changes / len(mpc_steer))

            # Forced MPC→PP transition detection: regime resets at high speed due to
            # teleport guard false-firing on frame drops. PP at >10 m/s is unstable.
            if regime is None:
                regime = arr("control/regime")
            _tp_detected = arr("control/teleport_detected")
            _spd = arr("vehicle/speed")
            if regime is not None and len(regime) > 1:
                forced_pp = 0
                for _i in range(1, len(regime)):
                    was_mpc = float(regime[_i - 1]) >= 0.5
                    now_pp = float(regime[_i]) < 0.5
                    if was_mpc and now_pp:
                        _s = float(_spd[_i]) if _spd is not None and _i < len(_spd) else 0.0
                        if _tp_detected is not None:
                            if _i < len(_tp_detected) and float(_tp_detected[_i]) > 0.5:
                                forced_pp += 1
                        elif _s > 6.0:
                            forced_pp += 1
                m["forced_pp_transition_count"] = forced_pp

            # Pitch signal health
            pitch = arr("vehicle/pitch_rad")
            if pitch is not None and road_grade is not None:
                m["pitch_signal_dead"] = bool(
                    np.var(pitch) < 1e-6 and np.var(road_grade) > 1e-4
                )

            # Curve late turn-in / lookahead floor rescue (Step 4)
            _lookahead = arr("control/pp_lookahead_distance")
            _curvature = arr("control/curvature_primary_abs")
            if _lookahead is not None and _curvature is not None:
                min_len = min(len(_lookahead), len(_curvature))
                la = _lookahead[:min_len]
                kappa = _curvature[:min_len]
                in_curve_mask = kappa > 0.005  # any meaningful curvature
                if np.any(in_curve_mask):
                    # Floor rescue: abrupt lookahead drops (>2m in one frame) while in curve
                    la_diff = np.diff(la)
                    diff_in_curve = la_diff[in_curve_mask[1:]]
                    floor_rescue_mask = diff_in_curve < -2.0
                    m["floor_rescue_rate"] = float(np.mean(floor_rescue_mask)) if len(diff_in_curve) > 0 else 0.0
                    m["floor_rescue_count"] = int(np.sum(floor_rescue_mask))
                else:
                    m["floor_rescue_rate"] = 0.0
                    m["floor_rescue_count"] = 0

            # Mild-curve under-activation: high control error on a mild map-backed arc,
            # recognizer stayed STRAIGHT, lookahead stayed long, actual lane offset stayed small.
            road_center_offset = arr("vehicle/road_frame_lane_center_offset")
            ref_curvature = arr("trajectory/reference_point_curvature")
            ref_lookahead = arr("control/reference_lookahead_target")
            if (
                lat_err is not None
                and road_center_offset is not None
                and ref_curvature is not None
                and lookahead is not None
                and "control/curve_intent_state" in f
                and "control/curve_local_state" in f
            ):
                curve_intent_state = np.array([
                    v.decode("utf-8", errors="ignore").strip().upper()
                    if isinstance(v, (bytes, bytearray, np.bytes_))
                    else str(v).strip().upper()
                    for v in f["control/curve_intent_state"][:]
                ], dtype=object)
                curve_local_state = np.array([
                    v.decode("utf-8", errors="ignore").strip().upper()
                    if isinstance(v, (bytes, bytearray, np.bytes_))
                    else str(v).strip().upper()
                    for v in f["control/curve_local_state"][:]
                ], dtype=object)
                sync_fallback = arr("control/sync_packet_fallback_active")
                perception_conf = arr("perception/confidence")
                num_lanes = arr("perception/num_lanes_detected")
                curve_blocker = arr("control/curve_activation_blocker_mode")
                arm_phase_deficit = arr("control/curve_local_arm_phase_deficit")
                sustain_phase_raw = arr("control/curve_local_sustain_phase_raw")
                mpc_kappa_ref = arr("control/mpc_kappa_ref")
                mpc_bias = arr("control/mpc_kappa_bias_correction")
                mpc_feasible_arr = arr("control/mpc_feasible")
                mpc_fallback_arr = arr("control/mpc_fallback_active")

                min_len = min(
                    len(lat_err),
                    len(road_center_offset),
                    len(ref_curvature),
                    len(lookahead),
                    len(curve_intent_state),
                    len(curve_local_state),
                )
                if ref_lookahead is not None:
                    min_len = min(min_len, len(ref_lookahead))
                if sync_fallback is not None:
                    min_len = min(min_len, len(sync_fallback))
                if perception_conf is not None:
                    min_len = min(min_len, len(perception_conf))
                if num_lanes is not None:
                    min_len = min(min_len, len(num_lanes))
                if curve_blocker is not None:
                    min_len = min(min_len, len(curve_blocker))
                if arm_phase_deficit is not None:
                    min_len = min(min_len, len(arm_phase_deficit))
                if sustain_phase_raw is not None:
                    min_len = min(min_len, len(sustain_phase_raw))
                if mpc_kappa_ref is not None:
                    min_len = min(min_len, len(mpc_kappa_ref))
                if mpc_bias is not None:
                    min_len = min(min_len, len(mpc_bias))
                if mpc_feasible_arr is not None:
                    min_len = min(min_len, len(mpc_feasible_arr))
                if mpc_fallback_arr is not None:
                    min_len = min(min_len, len(mpc_fallback_arr))

                if min_len > 0:
                    lat_abs = np.abs(np.asarray(lat_err[:min_len], dtype=float))
                    road_abs = np.abs(np.asarray(road_center_offset[:min_len], dtype=float))
                    ref_curv_abs = np.abs(np.asarray(ref_curvature[:min_len], dtype=float))
                    pp_look = np.asarray(lookahead[:min_len], dtype=float)
                    ref_look = (
                        np.asarray(ref_lookahead[:min_len], dtype=float)
                        if ref_lookahead is not None
                        else np.full(min_len, np.nan, dtype=float)
                    )
                    fallback_arr = (
                        np.asarray(sync_fallback[:min_len], dtype=float)
                        if sync_fallback is not None
                        else np.zeros(min_len, dtype=float)
                    )
                    conf_arr = (
                        np.asarray(perception_conf[:min_len], dtype=float)
                        if perception_conf is not None
                        else np.full(min_len, 1.0, dtype=float)
                    )
                    lanes_arr = (
                        np.asarray(num_lanes[:min_len], dtype=float)
                        if num_lanes is not None
                        else np.full(min_len, 2.0, dtype=float)
                    )
                    blocker_arr = (
                        np.asarray(curve_blocker[:min_len])
                        if curve_blocker is not None
                        else np.array([""] * min_len, dtype=object)
                    )
                    blocker_arr = np.array([
                        v.decode("utf-8", errors="ignore").strip().lower()
                        if isinstance(v, (bytes, bytearray, np.bytes_))
                        else str(v).strip().lower()
                        for v in blocker_arr
                    ], dtype=object)
                    arm_deficit_arr = (
                        np.asarray(arm_phase_deficit[:min_len], dtype=float)
                        if arm_phase_deficit is not None
                        else np.full(min_len, np.nan, dtype=float)
                    )
                    sustain_arr = (
                        np.asarray(sustain_phase_raw[:min_len], dtype=float)
                        if sustain_phase_raw is not None
                        else np.full(min_len, np.nan, dtype=float)
                    )
                    mpc_kappa_arr = (
                        np.asarray(mpc_kappa_ref[:min_len], dtype=float)
                        if mpc_kappa_ref is not None
                        else np.full(min_len, np.nan, dtype=float)
                    )
                    ref_signed = np.asarray(ref_curvature[:min_len], dtype=float)
                    mpc_bias_arr = (
                        np.asarray(mpc_bias[:min_len], dtype=float)
                        if mpc_bias is not None
                        else (mpc_kappa_arr - ref_signed)
                    )
                    mpc_feasible_mask = (
                        np.asarray(mpc_feasible_arr[:min_len], dtype=float) > 0.5
                        if mpc_feasible_arr is not None
                        else np.ones(min_len, dtype=bool)
                    )
                    mpc_fallback_mask = (
                        np.asarray(mpc_fallback_arr[:min_len], dtype=float) > 0.5
                        if mpc_fallback_arr is not None
                        else np.zeros(min_len, dtype=bool)
                    )
                    high_err = lat_abs >= 0.5
                    mild_curve = (ref_curv_abs >= 0.0015) & (ref_curv_abs <= 0.0035)
                    small_offset = road_abs <= 0.12
                    recognizer_inactive = np.array([
                        (curve_intent_state[i] not in {"ENTRY", "COMMIT"})
                        and (curve_local_state[i] not in {"ENTRY", "COMMIT"})
                        for i in range(min_len)
                    ], dtype=bool)
                    local_entry = np.array(
                        [curve_local_state[i] == "ENTRY" for i in range(min_len)],
                        dtype=bool,
                    )
                    local_rearm = np.array(
                        [curve_local_state[i] == "REARM" for i in range(min_len)],
                        dtype=bool,
                    )
                    long_look = (pp_look >= 8.0) | (ref_look >= 12.0)
                    poor_perception = (conf_arr < 0.5) | (lanes_arr < 2.0)
                    fallback_mask = fallback_arr > 0.5
                    underactivated = high_err & mild_curve & recognizer_inactive & long_look & small_offset
                    arm_ready = np.isfinite(arm_deficit_arr) & (arm_deficit_arr <= 0.01)
                    sustain_collapse = (
                        high_err
                        & mild_curve
                        & (local_entry | local_rearm)
                        & arm_ready
                        & np.isfinite(sustain_arr)
                        & (sustain_arr <= 0.12)
                    )
                    state_change = np.zeros(min_len, dtype=np.int8)
                    state_change[1:] = np.array(
                        [curve_local_state[i] != curve_local_state[i - 1] for i in range(1, min_len)],
                        dtype=np.int8,
                    )
                    transition_density = np.convolve(
                        state_change, np.ones(5, dtype=np.int8), mode="same"
                    )
                    rearm_presence = np.convolve(
                        local_rearm.astype(np.int8), np.ones(5, dtype=np.int8), mode="same"
                    )
                    rearm_cycle = (
                        sustain_collapse
                        & (transition_density >= 2)
                        & (rearm_presence >= 1)
                        & np.array([v == "state_hold" for v in blocker_arr], dtype=bool)
                    )
                    ref_safe = np.maximum(np.abs(ref_signed), 1e-6)
                    kappa_ratio = np.divide(
                        np.abs(mpc_kappa_arr),
                        ref_safe,
                        out=np.zeros_like(ref_safe),
                        where=np.isfinite(np.abs(mpc_kappa_arr)),
                    )
                    bias_cancel = (
                        high_err
                        & mild_curve
                        & mpc_feasible_mask
                        & ~mpc_fallback_mask
                        & (kappa_ratio <= 0.50)
                        & np.isfinite(mpc_bias_arr)
                        & (np.abs(mpc_bias_arr) >= 0.5 * np.abs(ref_signed))
                    )

                    high_err_count = int(np.sum(high_err))
                    m["highway_mild_curve_underactivation_rate"] = float(np.mean(underactivated))
                    preactivation_weight_arr = arr("control/curve_preactivation_authority_weight")
                    preactivation_active_arr = arr("control/curve_preactivation_authority_active")
                    if (
                        preactivation_weight_arr is not None
                        and preactivation_active_arr is not None
                    ):
                        pre_len = min(
                            len(preactivation_weight_arr),
                            len(preactivation_active_arr),
                            len(high_err),
                            len(mild_curve),
                            len(small_offset),
                            len(poor_perception),
                            len(fallback_mask),
                        )
                        pre_weight = np.asarray(preactivation_weight_arr[:pre_len], dtype=np.float64)
                        pre_active = np.asarray(preactivation_active_arr[:pre_len], dtype=np.float64) > 0.5
                        pre_high_err = high_err[:pre_len]
                        pre_mild_curve = mild_curve[:pre_len]
                        pre_small_offset = small_offset[:pre_len]
                        pre_poor_perception = poor_perception[:pre_len]
                        pre_fallback = fallback_mask[:pre_len]
                        pre_candidate = (
                            pre_high_err
                            & pre_mild_curve
                            & pre_small_offset
                            & ~pre_poor_perception
                            & ~pre_fallback
                            & np.isfinite(pre_weight)
                            & (pre_weight >= 0.35)
                        )
                        pre_missing = pre_candidate & ~pre_active
                        m["highway_preactivation_curve_authority_missing_rate"] = (
                            float(np.sum(pre_missing) / max(int(np.sum(pre_high_err)), 1))
                        )
                    local_ref_active_arr = arr("control/local_curve_reference_active")
                    active_preserve_weight_arr = arr("control/mpc_kappa_active_curve_preserve_weight")
                    curve_cap_speed_arr = arr("control/speed_governor_curve_cap_speed")
                    speed_series_arr = arr("vehicle/speed")
                    if (
                        local_ref_active_arr is not None
                        and active_preserve_weight_arr is not None
                        and curve_cap_speed_arr is not None
                        and speed_series_arr is not None
                    ):
                        active_len = min(
                            len(local_ref_active_arr),
                            len(active_preserve_weight_arr),
                            len(curve_cap_speed_arr),
                            len(speed_series_arr),
                            len(high_err),
                            len(mild_curve),
                            len(small_offset),
                            len(poor_perception),
                            len(fallback_mask),
                            len(curve_local_state_arr),
                        )
                        local_ref_active = np.asarray(local_ref_active_arr[:active_len], dtype=np.float64) > 0.5
                        preserve_weight = np.asarray(
                            active_preserve_weight_arr[:active_len], dtype=np.float64
                        )
                        cap_speed = np.asarray(curve_cap_speed_arr[:active_len], dtype=np.float64)
                        speed_now = np.asarray(speed_series_arr[:active_len], dtype=np.float64)
                        active_high_err = high_err[:active_len]
                        active_mild_curve = mild_curve[:active_len]
                        active_small_offset = small_offset[:active_len]
                        active_poor_perception = poor_perception[:active_len]
                        active_fallback = fallback_mask[:active_len]
                        local_active = np.array(
                            [
                                curve_local_state_arr[i] in {"ENTRY", "COMMIT"}
                                for i in range(active_len)
                            ],
                            dtype=bool,
                        )
                        active_failure = (
                            active_high_err
                            & active_mild_curve
                            & local_active
                            & local_ref_active
                            & active_small_offset
                            & ~active_poor_perception
                            & ~active_fallback
                        )
                        local_ref_shadow_only_arr = arr("control/local_curve_reference_shadow_only")
                        local_ref_raw_delta_arr = arr("control/local_curve_reference_raw_delta_m")
                        local_ref_capped_delta_arr = arr("control/local_curve_reference_capped_delta_m")
                        active_shadow_issue = np.zeros(active_len, dtype=bool)
                        if (
                            local_ref_shadow_only_arr is not None
                            and local_ref_raw_delta_arr is not None
                            and local_ref_capped_delta_arr is not None
                        ):
                            local_ref_shadow_only = (
                                np.asarray(local_ref_shadow_only_arr[:active_len], dtype=np.float64) > 0.5
                            )
                            local_ref_raw_delta = np.asarray(
                                local_ref_raw_delta_arr[:active_len], dtype=np.float64
                            )
                            local_ref_capped_delta = np.asarray(
                                local_ref_capped_delta_arr[:active_len], dtype=np.float64
                            )
                            local_ref_cap_ratio = np.divide(
                                local_ref_capped_delta,
                                local_ref_raw_delta,
                                out=np.full(active_len, np.nan, dtype=np.float64),
                                where=np.abs(local_ref_raw_delta) > 1e-6,
                            )
                            active_shadow_issue = (
                                active_failure
                                & local_ref_shadow_only
                                & np.isfinite(local_ref_raw_delta)
                                & (local_ref_raw_delta >= 0.25)
                                & np.isfinite(local_ref_cap_ratio)
                                & (local_ref_cap_ratio <= 0.10)
                            )
                        radar_candidate_present_arr = arr("vehicle/radar_fwd_candidate_present")
                        radar_reject_reason_arr = arr("vehicle/radar_fwd_reject_reason")
                        wrong_target_divergence = np.zeros(active_len, dtype=bool)
                        if (
                            radar_candidate_present_arr is not None
                            and radar_reject_reason_arr is not None
                            and local_ref_raw_delta_arr is not None
                        ):
                            radar_candidate_present = (
                                np.asarray(radar_candidate_present_arr[:active_len], dtype=np.float64)
                                > 0.5
                            )
                            radar_reject_reason = np.asarray(
                                [
                                    s.decode("utf-8", errors="ignore") if isinstance(s, (bytes, np.bytes_)) else str(s)
                                    for s in radar_reject_reason_arr[:active_len]
                                ],
                                dtype=object,
                            )
                            local_ref_raw_delta = np.asarray(
                                local_ref_raw_delta_arr[:active_len], dtype=np.float64
                            )
                            wrong_target_divergence = (
                                active_failure
                                & radar_candidate_present
                                & np.array(
                                    [
                                        radar_reject_reason[i] in {"wrong_lane", "opposite_direction"}
                                        for i in range(active_len)
                                    ],
                                    dtype=bool,
                                )
                                & np.isfinite(local_ref_raw_delta)
                                & (local_ref_raw_delta >= 0.75)
                            )
                        active_preserve_low = (
                            active_failure
                            & ~active_shadow_issue
                            & np.isfinite(preserve_weight)
                            & (preserve_weight <= 0.25)
                        )
                        active_cap_ineffective = (
                            active_failure
                            & ~active_shadow_issue
                            & (
                                (~np.isfinite(cap_speed))
                                | (~np.isfinite(speed_now))
                                | (cap_speed <= 0.0)
                                | (cap_speed >= (speed_now - 0.20))
                            )
                        )
                        active_issue = active_failure & ~active_shadow_issue & (
                            active_preserve_low | active_cap_ineffective
                        )
                        m["local_curve_reference_shadow_insufficient_rate"] = (
                            float(np.sum(active_shadow_issue) / max(int(np.sum(active_high_err)), 1))
                        )
                        m["wrong_target_distractor_reference_divergence_rate"] = (
                            float(np.sum(wrong_target_divergence) / max(int(np.sum(active_high_err)), 1))
                        )
                        m["wrong_target_straight_reference_drift_rate"] = float(
                            summary.get("wrong_target_contract", {}).get(
                                "straight_reference_drift_rate_pct",
                                0.0,
                            )
                            or 0.0
                        ) / 100.0
                        m["ego_lane_contract_invalid_rate"] = float(
                            summary.get("ego_lane_contract", {}).get(
                                "selected_vs_ego_cross_track_mismatch_rate_pct",
                                0.0,
                            )
                            or 0.0
                        ) / 100.0
                        m["high_speed_mild_curve_active_authority_insufficient_rate"] = (
                            float(np.sum(active_issue) / max(int(np.sum(active_high_err)), 1))
                        )
                        m["active_mild_curve_preserve_speed_gated_rate"] = float(
                            summary.get("highway_mild_curve_contract", {}).get(
                                "active_mild_curve_authority_speed_gated_on_high_error_rate",
                                0.0,
                            ) or 0.0
                        ) / 100.0
                    m["highway_mild_curve_reference_geometry_mismatch_on_high_error_rate"] = (
                        float(np.sum(high_err & small_offset) / max(high_err_count, 1))
                    )
                    m["highway_mild_curve_transport_fallback_overlap_on_high_error_rate"] = (
                        float(np.sum(high_err & fallback_mask) / max(high_err_count, 1))
                    )
                    m["highway_mild_curve_poor_perception_overlap_on_high_error_rate"] = (
                        float(np.sum(high_err & poor_perception) / max(high_err_count, 1))
                    )
                    m["highway_mild_curve_recognizer_inactive_on_high_error_rate"] = (
                        float(np.sum(high_err & recognizer_inactive) / max(high_err_count, 1))
                    )
                    m["highway_mild_curve_long_lookahead_on_high_error_rate"] = (
                        float(np.sum(high_err & long_look) / max(high_err_count, 1))
                    )
                    m["curve_sustain_collapse_rearm_cycle_rate"] = (
                        float(np.sum(rearm_cycle) / max(high_err_count, 1))
                    )
                    m["mpc_curvature_bias_cancellation_rate"] = (
                        float(np.sum(bias_cancel) / max(high_err_count, 1))
                    )
                    mpc_gt_at_car = arr("control/mpc_gt_cross_track_at_car_m")
                    mpc_gt_vehicle = arr("control/mpc_gt_cross_track_vehicle_frame_at_car_m")
                    if mpc_gt_vehicle is None:
                        mpc_gt_vehicle = mpc_gt_at_car
                    mpc_gt_road = arr("control/mpc_gt_cross_track_road_frame_at_car_m")
                    mpc_gt_lookahead = arr("control/mpc_gt_cross_track_lookahead_m")
                    if mpc_gt_at_car is not None and mpc_gt_lookahead is not None:
                        n_gt = min(len(mpc_gt_at_car), len(mpc_gt_lookahead), len(high_err))
                        if mpc_gt_vehicle is not None:
                            n_gt = min(n_gt, len(mpc_gt_vehicle))
                        if mpc_gt_road is not None:
                            n_gt = min(n_gt, len(mpc_gt_road))
                        gt_at_car = np.abs(np.asarray(mpc_gt_at_car[:n_gt], dtype=np.float64))
                        gt_vehicle = (
                            np.abs(np.asarray(mpc_gt_vehicle[:n_gt], dtype=np.float64))
                            if mpc_gt_vehicle is not None
                            else gt_at_car.copy()
                        )
                        gt_road = (
                            np.abs(np.asarray(mpc_gt_road[:n_gt], dtype=np.float64))
                            if mpc_gt_road is not None
                            else None
                        )
                        gt_lookahead = np.abs(np.asarray(mpc_gt_lookahead[:n_gt], dtype=np.float64))
                        high_err_gt = high_err[:n_gt]
                        semantic_mismatch = (
                            high_err_gt
                            & (gt_at_car <= 0.12)
                            & (gt_lookahead >= 0.50)
                        )
                        m["mpc_gt_cross_track_semantic_mismatch_rate"] = (
                            float(np.sum(semantic_mismatch) / max(int(np.sum(high_err_gt)), 1))
                        )
                        if gt_road is not None:
                            vehicle_frame_semantic = (
                                high_err_gt
                                & (gt_vehicle >= 0.50)
                                & (gt_road <= 0.12)
                                & (np.abs(gt_vehicle - gt_road) >= 0.35)
                            )
                            m["mpc_gt_cross_track_vehicle_frame_semantic_mismatch_rate"] = (
                                float(np.sum(vehicle_frame_semantic) / max(int(np.sum(high_err_gt)), 1))
                            )
                        road_offset = arr("vehicle/road_frame_lane_center_offset")
                        if road_offset is not None:
                            n_abs = min(n_gt, len(road_offset))
                            road_abs = np.abs(np.asarray(road_offset[:n_abs], dtype=np.float64))
                            high_err_abs = high_err[:n_abs]
                            absolute_mismatch = (
                                high_err_abs
                                & (gt_at_car[:n_abs] >= 1.0)
                                & (road_abs <= 0.12)
                            )
                            m["mpc_gt_cross_track_absolute_coordinate_mismatch_rate"] = (
                                float(np.sum(absolute_mismatch) / max(int(np.sum(high_err_abs)), 1))
                            )

            # Speed below target rate
            _speed = arr("vehicle/speed")
            if _speed is not None:
                m["speed_below_target_rate"] = float(
                    np.mean(_speed < 10.0)
                )

            # ACC metrics (Step 5)
            acc_active_raw = arr("vehicle/acc_active")
            if acc_active_raw is not None:
                acc_active_arr = acc_active_raw[:n].astype(float)
                acc_active_pct = float(np.mean(acc_active_arr > 0.5))
                m["acc_active"] = acc_active_pct >= 0.10
                if m["acc_active"]:
                    acc_mask = acc_active_arr > 0.5
                    m["acc_active_pct"] = acc_active_pct * 100.0

                    ttc_raw = arr("vehicle/acc_ttc_s")
                    if ttc_raw is not None:
                        ttc_arr = ttc_raw[:n].astype(float)
                        ttc_acc = ttc_arr[acc_mask & np.isfinite(ttc_arr)]
                        if ttc_acc.size > 0:
                            m["acc_ttc_warning_pct"] = float(np.mean(ttc_acc < 2.5) * 100.0)

                    dist_raw = arr("vehicle/radar_fwd_distance_m")
                    if dist_raw is not None:
                        dist_arr = dist_raw[:n].astype(float)
                        nm_frames = acc_mask & (dist_arr > 0.0) & (dist_arr < 2.0)
                        run_nm, nm_count = 0, 0
                        for f_flag in nm_frames:
                            if f_flag:
                                run_nm += 1
                                if run_nm == 3:
                                    nm_count += 1
                            else:
                                run_nm = 0
                        m["acc_near_miss_events"] = nm_count

                    gap_err_raw = arr("vehicle/acc_gap_error_m")
                    if gap_err_raw is not None:
                        ge_arr = gap_err_raw[:n].astype(float)
                        ge_acc = ge_arr[acc_mask & np.isfinite(ge_arr)]
                        if ge_acc.size > 0:
                            m["acc_gap_rmse_m"] = float(np.sqrt(np.mean(ge_acc ** 2)))
                            # Sign changes per minute (hunting indicator)
                            sign_changes = int(np.sum(np.diff(np.sign(ge_acc)) != 0))
                            duration_min = max(1.0, len(ge_acc) / 30.0 / 60.0)
                            m["acc_gap_sign_changes_per_min"] = sign_changes / duration_min

                    detected_raw = arr("vehicle/radar_fwd_detected")
                    if detected_raw is not None:
                        det_arr = detected_raw[:n].astype(float)
                        lead_frames = acc_mask
                        if np.any(lead_frames):
                            m["acc_detection_rate"] = float(np.mean(det_arr[lead_frames] > 0.5))
            else:
                m["acc_active"] = False

            # Inter-frame control extrapolation metrics
            if_active_raw = arr("control/interframe_active")
            if if_active_raw is not None:
                if_arr = if_active_raw[:n].astype(float)
                if_mask = if_arr > 0.5
                m["interframe_active_pct"] = float(np.mean(if_mask) * 100.0)

                if np.any(if_mask):
                    # Check if inter-frame at max updates consistently
                    if_updates = arr("control/interframe_updates_this_cycle")
                    if if_updates is not None:
                        up_arr = if_updates[:n].astype(float)
                        # Frames where inter-frame was active and at max (3)
                        at_max = if_mask & (up_arr >= 3.0)
                        if_active_count = max(1, int(np.sum(if_mask)))
                        m["interframe_at_max_pct"] = float(np.sum(at_max) / if_active_count * 100.0)

                    # Rate ratio: effective rate vs target (camera × 4)
                    if_total_arr = arr("control/interframe_total_count")
                    _ts = control_ts if control_ts is not None else arr("control/timestamps")
                    if if_total_arr is not None and len(if_total_arr) > 0:
                        total_if = int(if_total_arr[n - 1]) if n > 0 else 0
                        duration_s = float(_ts[n - 1]) if _ts is not None and n > 0 and len(_ts) > 0 else 1.0
                        if duration_s > 0:
                            eff_hz = (n + total_if) / duration_s
                            cam_hz = n / duration_s
                            target_hz = cam_hz * 4.0 if cam_hz > 0 else 30.0
                            m["interframe_rate_ratio"] = eff_hz / target_hz if target_hz > 0 else 1.0

                    # e_lat divergence: compare interframe_last_e_lat to next-frame mpc_e_lat
                    if_e_lat = arr("control/interframe_last_e_lat")
                    mpc_e_lat = arr("control/mpc_e_lat")
                    if if_e_lat is not None and mpc_e_lat is not None:
                        if_el = if_e_lat[:n].astype(float)
                        mpc_el = mpc_e_lat[:n].astype(float)
                        # Compare inter-frame e_lat to NEXT camera frame's mpc_e_lat
                        divergence = np.abs(if_el[:-1] - mpc_el[1:])[if_mask[:-1]]
                        if len(divergence) > 0:
                            m["interframe_e_lat_divergence_mean"] = float(np.mean(divergence))

        return m

    def _match_patterns(self, metrics: dict) -> list:
        """Check all patterns against extracted metrics. Return sorted matches."""
        matches = []
        n = metrics.get("total_frames", 1) or 1
        rate_key_map = {
            "perc_stale_cascade":    "stale_rate",
            "perc_low_confidence":   "stale_rate",
            "perc_single_lane_only": "single_lane_rate",
            "traj_ref_rate_clamped": "ref_rate_clamp_rate",
            "traj_short_lookahead":  "short_lookahead_rate",
            "mpc_infeasible_burst":  "mpc_infeasible_rate",
            "mpc_solve_time_budget": "mpc_solve_p95_ms",
            "mpc_fallback_cascade":  "mpc_fallback_rate",
            "mpc_regime_chatter":    "mpc_regime_changes_per_min",
            "mpc_heading_error_exceeds_model": "mpc_heading_error_p95",
            "mpc_kappa_ref_mismatch": "mpc_kappa_divergence_p95",
            "nmpc_infeasible_burst":  "nmpc_infeasible_rate",
            "nmpc_solve_time_budget": "nmpc_solve_p95_ms",
            "nmpc_fallback_cascade":  "nmpc_fallback_rate",
            "heading_gate_stuck_on_curve": "heading_suppression_rate_on_curves",
            "grade_overspeed_downhill": "downhill_overspeed_rate",
            "grade_compensation_missing": "grade_comp_active_rate",
            "grade_mpc_infeasible": "mpc_infeasible_rate",
            "grade_throttle_budget_starved": "grade_throttle_budget_min",
            "grade_planner_false_cap": "planner_false_cap_rate",
            "grade_lateral_osc_growth": "grade_osc_growth",
            "grade_perception_dropout": "grade_perception_conf_ratio",
            "grade_throttle_saturated": "grade_throttle_sat_rate",
            "mpc_steering_oscillation": "mpc_steering_osc_rate",
            "gt_boundary_corrupt": "gt_boundary_corrupt_frames",
            "highway_mild_curve_underactivation": "highway_mild_curve_underactivation_rate",
            "highway_preactivation_curve_authority_missing": "highway_preactivation_curve_authority_missing_rate",
            "ego_lane_contract_invalid": "ego_lane_contract_invalid_rate",
            "local_curve_reference_shadow_insufficient": "local_curve_reference_shadow_insufficient_rate",
            "wrong_target_distractor_reference_divergence": "wrong_target_distractor_reference_divergence_rate",
            "wrong_target_straight_reference_drift": "wrong_target_straight_reference_drift_rate",
            "high_speed_mild_curve_active_authority_insufficient": "high_speed_mild_curve_active_authority_insufficient_rate",
            "active_mild_curve_preserve_speed_gated": "active_mild_curve_preserve_speed_gated_rate",
            "mpc_gt_cross_track_semantic_mismatch": "mpc_gt_cross_track_semantic_mismatch_rate",
            "mpc_gt_cross_track_vehicle_frame_semantic_mismatch": "mpc_gt_cross_track_vehicle_frame_semantic_mismatch_rate",
            "mpc_gt_cross_track_absolute_coordinate_mismatch": "mpc_gt_cross_track_absolute_coordinate_mismatch_rate",
            "tire_saturation": "tire_saturation_rate",
            "tire_ekf_divergence": "tire_ekf_divergence_rate",
            "tire_understeer_anomaly": "tire_understeer_gradient_max",
            "perc_blind_no_detection": "blind_perception_rate",
        }
        for pat in PATTERNS:
            try:
                fired = pat["check"](metrics)
            except Exception:
                fired = False
            if not fired:
                continue
            rate = metrics.get(rate_key_map.get(pat["id"], ""), 0.0) or (1 / n)
            matches.append(PatternMatch(
                pattern_id=pat["id"],
                name=pat["name"],
                severity=pat["severity"],
                occurrences=max(1, int(n * rate)),
                pct_of_recording=rate,
                code_pointer=pat["code_pointer"],
                config_lever=pat["config_lever"],
                fix_hint=pat["fix_hint"],
                example_frames=[],
            ))
        matches.sort(key=lambda m: (SEVERITY_ORDER[m.severity], -m.occurrences))
        return matches

    def _build_checklist(self, matches: list) -> list:
        return [f"{i}. [{m.config_lever}] {m.fix_hint}"
                for i, m in enumerate(matches, 1)]

    def _compute_attribution(self, health_summary: dict) -> dict:
        """Attribution = per-layer fraction of unhealthy frames, normalized to 1.0."""
        raw = {
            layer: health_summary.get(layer, {}).get("pct_unhealthy", 0.0)
            for layer in ("perception", "trajectory", "control")
        }
        total = sum(raw.values())
        if total < 1e-9:
            return {"perception": 0.33, "trajectory": 0.33, "control": 0.34}
        return {k: round(v / total, 3) for k, v in raw.items()}

    def _overall_health(self, health_summary: dict) -> float:
        scores = [health_summary.get(layer, {}).get("mean_score", 0.8)
                  for layer in ("perception", "trajectory", "control")]
        return float(np.mean(scores))

    def _get_layer_health_summary(self) -> dict:
        from backend.layer_health import LayerHealthAnalyzer
        return LayerHealthAnalyzer(self.path).compute()["summary"]

    def _get_stale_events(self) -> dict:
        from backend.blame_tracer import BlameTracer
        data = BlameTracer(self.path).find_stale_propagation()
        # Exclude benign stale reasons that reflect designed backup-hold behaviour.
        filtered = [ev for ev in data.get('events', [])
                    if ev.get('stale_reason', '') not in BENIGN_STALE_REASONS]
        return {'event_count': len(filtered), 'events': filtered}

    def _frame_count(self, f: h5py.File) -> int:
        for key in ("vehicle/timestamp", "control/steering", "perception/confidence"):
            if key in f:
                return len(f[key])
        return 0
