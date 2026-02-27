"""
Layer health scoring for PhilViz Phase 3.

Computes per-frame health scores (0.0–1.0) for each stack layer
(Perception, Trajectory, Control) based on HDF5 signal quality.

Scoring is intentionally simple: weighted linear combination of
normalized signals, clamped to [0, 1]. No ML required.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import h5py
import yaml


# ── Thresholds ────────────────────────────────────────────────────────────────
# Tune these before tuning weights. They define "bad" for each signal.
PERCEPTION_CONF_FLOOR   = 0.1    # confidence below this → hard fallback territory
TRAJ_REF_ERROR_MAX_M    = 2.0    # lateral ref error > this → full penalty
CTRL_LATERAL_ERROR_MAX  = 2.0    # lateral error > this → full penalty (= safety.max_lateral_error)
CTRL_JERK_GATE          = 6.0    # commanded_jerk_p95 comfort gate (m/s³)
LOOKAHEAD_MIN_M         = 5.0    # lookahead below this is suspiciously short
LOOKAHEAD_MAX_M         = 25.0   # lookahead above this is unusually long

# Shared alert thresholds imported by triage_engine and blame_tracer
CTRL_JERK_ALERT      = CTRL_JERK_GATE * 0.75   # 4.5 m/s³ — 75% of comfort gate; triggers flag + triage pattern
LOOKAHEAD_CONCERN_M  = 8.0                      # Lookahead "short at speed" alert; distinct from LOOKAHEAD_MIN_M (absolute floor)
BENIGN_STALE_REASONS = frozenset({'left_lane_low_visibility'})  # Designed backup-hold states; suppress warnings


class LayerHealthAnalyzer:
    def __init__(self, recording_path: Path) -> None:
        self.path = recording_path
        self.longitudinal_max_accel, self.longitudinal_max_decel = self._load_longitudinal_limits()

    def compute(self) -> dict:
        """
        Load all needed HDF5 arrays once, compute per-frame scores,
        return JSON-ready dict.
        """
        with h5py.File(self.path, 'r') as f:
            n_frames = self._frame_count(f)
            frames = []
            for i in range(n_frames):
                pf = self._score_perception(f, i)
                tf = self._score_trajectory(f, i)
                cf = self._score_control(f, i)
                frames.append({
                    "frame_idx":         i,
                    "timestamp":         self._scalar(f, "vehicle/timestamp", i,
                                                      default=float(i) / 20.0),
                    "perception_score":  round(pf["score"], 3),
                    "perception_flags":  pf["flags"],
                    "trajectory_score":  round(tf["score"], 3),
                    "trajectory_flags":  tf["flags"],
                    "control_score":     round(cf["score"], 3),
                    "control_flags":     cf["flags"],
                })

        summary = self._summarize(frames)
        return {
            "frame_count": n_frames,
            "layers": ["perception", "trajectory", "control"],
            "frames": frames,
            "summary": summary,
        }

    # ── Per-layer scorers ─────────────────────────────────────────────────────

    def _score_perception(self, f: h5py.File, i: int) -> dict:
        score = 1.0
        flags = []

        # Stale frame (weight 0.30) — hard cap
        # Recorded as perception/using_stale_data (boolean int)
        stale = bool(self._scalar(f, "perception/using_stale_data", i, default=0))
        if stale:
            score -= 0.30
            flags.append("stale")

        # Detection confidence (weight 0.35)
        conf = self._scalar(f, "perception/confidence", i, default=0.5)
        conf_norm = max(0.0, min(1.0,
                        (conf - PERCEPTION_CONF_FLOOR) / (1.0 - PERCEPTION_CONF_FLOOR)))
        score -= 0.35 * (1.0 - conf_norm)
        if conf < PERCEPTION_CONF_FLOOR + 0.1:
            flags.append("low_confidence")

        # Num lanes detected (weight 0.20): 2=ideal, 1=partial, 0=failed
        n_lanes = int(self._scalar(f, "perception/num_lanes_detected", i, default=2))
        lane_score = {2: 1.0, 1: 0.5, 0: 0.0}.get(n_lanes, 0.5)
        score -= 0.20 * (1.0 - lane_score)
        if n_lanes == 1:
            flags.append("single_lane")
        elif n_lanes == 0:
            flags.append("no_lanes")

        # Perception clamp events (weight 0.15) — recorded as count, treat >0 as fired
        clamp = self._scalar(f, "perception/perception_clamp_events", i, default=0)
        if clamp > 0:
            score -= 0.15
            flags.append("gate_fired")

        return {"score": max(0.0, min(1.0, score)), "flags": flags}

    def _score_trajectory(self, f: h5py.File, i: int) -> dict:
        score = 1.0
        flags = []

        # Near-band lateral error (weight 0.40)
        # diag_postclip_abs_mean_0_8m = mean absolute x deviation in 0-8m band
        ref_err = self._scalar(f, "trajectory/diag_postclip_abs_mean_0_8m", i, default=0.0)
        ref_norm = max(0.0, min(1.0, ref_err / TRAJ_REF_ERROR_MAX_M))
        score -= 0.40 * ref_norm
        if ref_err > 0.3:
            flags.append("ref_error_high")

        # Ref rate limit active (weight 0.25) — diag_ref_x_rate_limit_active
        clamped = bool(self._scalar(f, "trajectory/diag_ref_x_rate_limit_active", i,
                                    default=0))
        if clamped:
            score -= 0.25
            flags.append("ref_rate_clamped")

        # Lookahead validity (weight 0.20) — PP lookahead lives in control/pp_lookahead_distance
        lookahead = self._scalar(f, "control/pp_lookahead_distance", i, default=10.0)
        if lookahead < LOOKAHEAD_MIN_M or lookahead > LOOKAHEAD_MAX_M:
            score -= 0.20
            flags.append("lookahead_out_of_range")

        # Curve anticipation active — informational only (no score impact)
        curv_active = bool(self._scalar(f, "control/curve_anticipation_active", i,
                                        default=0))
        if curv_active:
            flags.append("curve_anticipation")

        return {"score": max(0.0, min(1.0, score)), "flags": flags}

    def _score_control(self, f: h5py.File, i: int) -> dict:
        score = 1.0
        flags = []

        # Lateral error (weight 0.40)
        lat_err = abs(self._scalar(f, "control/lateral_error", i, default=0.0))
        lat_norm = max(0.0, min(1.0, lat_err / CTRL_LATERAL_ERROR_MAX))
        score -= 0.40 * lat_norm
        if lat_err > 0.4:
            flags.append("lateral_error_high")

        # Longitudinal jerk (weight 0.30) from command-domain acceleration derivative.
        jerk = self._commanded_jerk_at(f, i)
        jerk_norm = max(0.0, min(1.0, jerk / CTRL_JERK_GATE))
        score -= 0.30 * jerk_norm
        if jerk > CTRL_JERK_ALERT:
            flags.append("jerk_spike")

        jerk_cap_active = bool(self._scalar(f, "control/longitudinal_jerk_capped", i, default=0.0) > 0.5)
        if jerk_cap_active:
            flags.append("jerk_cap_active")

        # Steering jerk limiter active (weight 0.10)
        steer_jerk_active = bool(self._scalar(f, "control/steering_jerk_limited_active",
                                              i, default=0))
        if steer_jerk_active:
            score -= 0.10
            flags.append("steer_jerk_limited")

        # Longitudinal accel cap — flag only.
        accel_cap_active = bool(self._scalar(f, "control/longitudinal_accel_capped", i, default=0.0) > 0.5)
        if accel_cap_active:
            flags.append("accel_high")

        return {"score": max(0.0, min(1.0, score)), "flags": flags}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _scalar(self, f: h5py.File, path: str, i: int, default: float = 0.0) -> float:
        """Safely read scalar at index i from an HDF5 dataset."""
        if path not in f:
            return default
        try:
            val = float(f[path][i])
            if not np.isfinite(val):
                return default
            return val
        except (IndexError, ValueError):
            return default

    def _load_longitudinal_limits(self) -> tuple[float, float]:
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

    def _commanded_jerk_at(self, f: h5py.File, i: int) -> float:
        if i <= 0:
            return 0.0
        throttle_now = self._scalar(f, "control/throttle", i, default=0.0)
        throttle_prev = self._scalar(f, "control/throttle", i - 1, default=0.0)
        brake_now = self._scalar(f, "control/brake", i, default=0.0)
        brake_prev = self._scalar(f, "control/brake", i - 1, default=0.0)
        accel_now = (throttle_now * self.longitudinal_max_accel) - (brake_now * self.longitudinal_max_decel)
        accel_prev = (throttle_prev * self.longitudinal_max_accel) - (brake_prev * self.longitudinal_max_decel)

        dt = None
        if "control/timestamps" in f:
            try:
                dt = float(f["control/timestamps"][i]) - float(f["control/timestamps"][i - 1])
            except (IndexError, ValueError, TypeError):
                dt = None
        elif "control/timestamp" in f:
            try:
                dt = float(f["control/timestamp"][i]) - float(f["control/timestamp"][i - 1])
            except (IndexError, ValueError, TypeError):
                dt = None
        if dt is None or not np.isfinite(dt) or dt <= 1e-6:
            dt = 1.0 / 30.0
        jerk = (accel_now - accel_prev) / dt
        if not np.isfinite(jerk):
            return 0.0
        return abs(float(jerk))

    def _frame_count(self, f: h5py.File) -> int:
        """Infer frame count from the first available dataset."""
        for key in ("vehicle/timestamp", "control/steering", "perception/confidence"):
            if key in f:
                return len(f[key])
        raise ValueError(
            "Cannot determine frame count — no known dataset found in HDF5."
        )

    def _summarize(self, frames: list) -> dict:
        """Aggregate per-layer stats across all frames."""
        summary = {}
        for layer in ("perception", "trajectory", "control"):
            scores = [fr[f"{layer}_score"] for fr in frames]
            if not scores:
                summary[layer] = {}
                continue
            arr = np.array(scores)
            summary[layer] = {
                "mean_score":    round(float(arr.mean()), 3),
                "min_score":     round(float(arr.min()), 3),
                "pct_green":     round(float((arr >= 0.8).mean()), 3),
                "pct_yellow":    round(float(((arr >= 0.5) & (arr < 0.8)).mean()), 3),
                "pct_red":       round(float((arr < 0.5).mean()), 3),
                "pct_unhealthy": round(float((arr < 0.8).mean()), 3),
            }
        return summary
