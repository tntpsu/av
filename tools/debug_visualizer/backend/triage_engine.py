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
                mpc_mask = regime >= 1
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
