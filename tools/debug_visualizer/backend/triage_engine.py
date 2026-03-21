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
        "config_lever": "control.lateral.pp_feedback_gain",
        "fix_hint": "Reduce pp_feedback_gain (0.10 → 0.08). If persists: increase reference_smoothing (0.75 → 0.80).",
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
        "config_lever": "mpc_q_lat / mpc_r_steer_rate",
        "fix_hint": "MPC QP solver returning infeasible. Check constraint bounds or reduce q_lat to widen feasible region.",
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
        "id": "mpc_regime_chatter",
        "name": "Regime chatter (>6 switches/min)",
        "severity": "comfort",
        "code_pointer": "control/regime_selector.py:RegimeSelector.update()",
        "config_lever": "regime.hysteresis_speed_mps",
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
        "config_lever": "trajectory.mpc.mpc_q_lat / trajectory.mpc.mpc_r_steer_rate",
        "fix_hint": "MPC steering sign-changes >30% of MPC frames. Lower q_lat or raise r_steer_rate to reduce aggressiveness.",
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

                # Windowed lateral oscillation on grade
                grade_lat_err = arr("control/lateral_error")
                if grade_lat_err is not None and np.any(graded) and np.sum(graded) > 60:
                    graded_err = np.abs(grade_lat_err[graded])
                    win = 20
                    if len(graded_err) > 2 * win:
                        vars_early = float(np.var(graded_err[:win]))
                        vars_late = float(np.var(graded_err[-win:]))
                        m["grade_osc_growth"] = float(vars_late / max(vars_early, 1e-6))

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

            # Speed below target rate
            _speed = arr("vehicle/speed")
            if _speed is not None:
                m["speed_below_target_rate"] = float(
                    np.mean(_speed < 10.0)
                )

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
