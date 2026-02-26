"""
Blame tracer for PhilViz Phase 4.

Given a target frame and metric, walks backward through layer health
scores to find the earliest degradation event and assigns primary blame
to the originating layer.

Also finds stale propagation events: stale perception → trajectory drift
→ control error, with lag estimates.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import h5py

from backend.layer_health import BENIGN_STALE_REASONS  # noqa: F401 — future-proofs stale filtering


BASELINE_WINDOW   = 20    # frames before target to compute baseline
LOOKBACK          = 20    # frames to search backward from target
DEGRADATION_DELTA = 0.20  # score drop threshold to count as "degradation"
MIN_CAUSAL_LAG    = 2     # upstream degradation must precede target by at least this many frames


@dataclass
class BlameLink:
    frame_idx:  int
    layer:      str       # "perception" | "trajectory" | "control"
    signal:     str
    value:      float
    baseline:   float
    delta:      float     # positive = degraded


class BlameTracer:
    def __init__(self, recording_path: Path) -> None:
        self.path = recording_path
        self._health_cache: Optional[list] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def trace_blame(
        self,
        target_frame: int,
        metric: str = "lateral_error",
        lookback: int = LOOKBACK,
    ) -> dict:
        """
        Walk backward from target_frame to find the originating layer.
        Returns a JSON-ready dict.
        """
        health = self._get_health()
        if not health or target_frame >= len(health):
            return {"error": f"Frame {target_frame} out of range (total={len(health)})"}

        n = len(health)
        start = max(0, target_frame - lookback)
        baseline_start = max(0, target_frame - lookback - BASELINE_WINDOW)
        baseline_end   = max(0, target_frame - lookback)

        # Compute baseline per layer from the window before the lookback region
        baselines = {}
        for layer in ("perception", "trajectory", "control"):
            key = f"{layer}_score"
            window = [health[i][key] for i in range(baseline_start, baseline_end) if i < n]
            baselines[layer] = float(np.mean(window)) if window else 0.8

        # Walk forward from start — find the EARLIEST degradation per layer
        # (forward walk ensures we get the origin frame, not the most recent spike)
        first_degradation: dict[str, int] = {}
        for i in range(start, target_frame + 1):
            for layer in ("perception", "trajectory", "control"):
                key = f"{layer}_score"
                score = health[i][key]
                baseline = baselines[layer]
                if (baseline - score) >= DEGRADATION_DELTA and layer not in first_degradation:
                    first_degradation[layer] = i

        # Build chain: one link per degraded layer, earliest-first
        chain: list[BlameLink] = []
        for layer in ("perception", "trajectory", "control"):
            if layer not in first_degradation:
                continue
            fi = first_degradation[layer]
            score = health[fi][f"{layer}_score"]
            baseline = baselines[layer]
            flags = health[fi].get(f"{layer}_flags", [])
            signal = flags[0] if flags else f"{layer}_score"
            chain.append(BlameLink(
                frame_idx=fi,
                layer=layer,
                signal=signal,
                value=round(score, 3),
                baseline=round(baseline, 3),
                delta=round(baseline - score, 3),
            ))

        chain.sort(key=lambda x: x.frame_idx)

        # Primary blame = earliest degrading layer with sufficient causal lag
        primary_blame = "unknown"
        lag_frames = 0
        if chain:
            earliest = chain[0]
            if target_frame - earliest.frame_idx >= MIN_CAUSAL_LAG:
                primary_blame = earliest.layer
                lag_frames = target_frame - earliest.frame_idx

        confidence = self._compute_confidence(chain, target_frame)

        return {
            "target_frame":  target_frame,
            "target_metric": metric,
            "primary_blame": primary_blame,
            "lag_frames":    lag_frames,
            "confidence":    round(confidence, 2),
            "chain": [
                {
                    "frame_idx": lnk.frame_idx,
                    "layer":     lnk.layer,
                    "signal":    lnk.signal,
                    "value":     lnk.value,
                    "baseline":  lnk.baseline,
                    "delta":     lnk.delta,
                }
                for lnk in chain
            ],
        }

    def find_stale_propagation(self, max_lookahead: int = 15) -> dict:
        """
        Find stale perception events and measure their downstream cascade
        on trajectory and control for up to max_lookahead frames.
        Returns JSON-ready dict.
        """
        health = self._get_health()
        if not health:
            return {"event_count": 0, "events": []}

        stale_flags = [("stale" in fr.get("perception_flags", [])) for fr in health]
        events = []
        i = 0
        while i < len(stale_flags):
            if not stale_flags[i]:
                i += 1
                continue

            start = i
            while i < len(stale_flags) and stale_flags[i]:
                i += 1
            duration = i - start

            # Baselines: 10 frames before the event
            def _baseline(layer_key: str) -> float:
                window = [health[j][layer_key]
                          for j in range(max(0, start - 10), start)]
                return float(np.mean(window)) if window else 0.8

            baseline_traj = _baseline("trajectory_score")
            baseline_ctrl = _baseline("control_score")

            timeline = []
            max_traj_delta = 0.0
            max_ctrl_delta = 0.0
            traj_lag = 0
            ctrl_lag = 0

            for offset in range(max_lookahead):
                fi = start + offset
                if fi >= len(health):
                    break
                fr = health[fi]
                traj_delta = max(0.0, baseline_traj - fr["trajectory_score"])
                ctrl_delta = max(0.0, baseline_ctrl - fr["control_score"])
                timeline.append({
                    "frame":      fi,
                    "perc_stale": stale_flags[fi] if fi < len(stale_flags) else False,
                    "traj_delta": round(traj_delta, 3),
                    "ctrl_delta": round(ctrl_delta, 3),
                })
                if traj_delta > max_traj_delta:
                    max_traj_delta = traj_delta
                    traj_lag = offset
                if ctrl_delta > max_ctrl_delta:
                    max_ctrl_delta = ctrl_delta
                    ctrl_lag = offset

            events.append({
                "stale_start_frame":     start,
                "stale_duration":        duration,
                "trajectory_lag_frames": traj_lag,
                "control_lag_frames":    ctrl_lag,
                "max_trajectory_delta":  round(max_traj_delta, 3),
                "max_control_delta":     round(max_ctrl_delta, 3),
                "propagation_timeline":  timeline,
                "stale_reason":          "",  # filled below
            })

        # Annotate each event with the stale reason at its start frame so the
        # frontend can suppress known-benign events (e.g. left_lane_low_visibility).
        try:
            with h5py.File(self.path, "r") as f:
                if "perception/stale_reason" in f:
                    ds = f["perception/stale_reason"]
                    for ev in events:
                        idx = ev["stale_start_frame"]
                        if idx < len(ds):
                            r = ds[idx]
                            ev["stale_reason"] = (r.decode("utf-8") if isinstance(r, bytes) else str(r or "")).strip('\x00').strip()
        except Exception:
            pass  # stale_reason already defaults to ""

        return {"event_count": len(events), "events": events}

    def find_trajectory_degradation(self, max_lookahead: int = 15) -> dict:
        """
        Find trajectory degradation events (ref_rate_clamped, ref_error_high,
        lookahead_out_of_range) and measure their downstream impact on control
        health for up to max_lookahead frames.
        Returns JSON-ready dict.
        """
        health = self._get_health()
        if not health:
            return {"event_count": 0, "events": []}

        ACTIVE_FLAGS = {"ref_rate_clamped", "ref_error_high", "lookahead_out_of_range"}
        degraded = [
            bool(set(fr.get("trajectory_flags", [])) & ACTIVE_FLAGS)
            for fr in health
        ]

        events = []
        i = 0
        while i < len(degraded):
            if not degraded[i]:
                i += 1
                continue

            start = i
            while i < len(degraded) and degraded[i]:
                i += 1
            duration = i - start

            # Primary flag: most frequent active flag across the event window
            window_flags = []
            for j in range(start, start + duration):
                window_flags.extend(set(health[j].get("trajectory_flags", [])) & ACTIVE_FLAGS)
            primary_flag = max(set(window_flags), key=window_flags.count) if window_flags else "unknown"

            def _baseline(key: str) -> float:
                window = [health[j][key] for j in range(max(0, start - 10), start)]
                return float(np.mean(window)) if window else 0.8

            baseline_traj = _baseline("trajectory_score")
            baseline_ctrl = _baseline("control_score")

            max_traj_delta = 0.0
            max_ctrl_delta = 0.0
            traj_lag = 0
            ctrl_lag = 0

            for offset in range(max_lookahead):
                fi = start + offset
                if fi >= len(health):
                    break
                traj_delta = max(0.0, baseline_traj - health[fi]["trajectory_score"])
                ctrl_delta = max(0.0, baseline_ctrl - health[fi]["control_score"])
                if traj_delta > max_traj_delta:
                    max_traj_delta = traj_delta
                    traj_lag = offset
                if ctrl_delta > max_ctrl_delta:
                    max_ctrl_delta = ctrl_delta
                    ctrl_lag = offset

            events.append({
                "traj_start_frame":  start,
                "traj_duration":     duration,
                "primary_flag":      primary_flag,
                "traj_lag_frames":   traj_lag,
                "control_lag_frames": ctrl_lag,
                "max_traj_delta":    round(max_traj_delta, 3),
                "max_control_delta": round(max_ctrl_delta, 3),
            })

        return {"event_count": len(events), "events": events}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_health(self) -> list:
        """Lazy-load layer health (cached within this instance)."""
        if self._health_cache is None:
            from backend.layer_health import LayerHealthAnalyzer
            result = LayerHealthAnalyzer(self.path).compute()
            self._health_cache = result["frames"]
        return self._health_cache

    def _compute_confidence(self, chain: list, target_frame: int) -> float:
        """Estimate confidence: higher if all three layers degrade in causal order."""
        if not chain:
            return 0.0
        layers_present = [lnk.layer for lnk in chain]
        canonical = ["perception", "trajectory", "control"]
        present = [l for l in canonical if l in layers_present]
        order_score = len(present) / 3.0
        # Reduce confidence if lag is very short (may be coincidence)
        lag = target_frame - chain[0].frame_idx if chain else 0
        lag_penalty = 0.2 if lag < 3 else 0.0
        return max(0.0, min(1.0, order_score - lag_penalty + 0.1 * len(chain)))
