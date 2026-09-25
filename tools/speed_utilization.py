#!/usr/bin/env python3
"""Speed-limit utilisation — did the car actually use the road it was given?

Every existing layer scores what the car did *wrong* (lane error, jerk, e-stops).
Nothing scores whether it did the job. A car parked in the lane scores 100 on
every layer; hill_highway scored 97.6 while driving at 15 mph on a 25 mph road;
s_loop drives at half its posted limit. This module reports that.

Two ratios, both from fields present in every recording:

  vs_limit   = v / speed_limit(s)                 — product truth
  vs_allowed = v / min(speed_limit(s), target)     — system truth: isolates governor
                                                     / tracking-budget defects from the
                                                     deliberately low research target

Eligible frames exclude the first STARTUP_S seconds, frames following a lead
(ACC active — the lead sets the speed), emergency-stop frames, and frames
approaching a LOWER posted limit (speed_limit_preview < speed_limit), so
legitimate braking for a curve is not counted as under-utilisation.

Attribution: at under-utilised frames the binding cap is whichever of
target_speed_raw / velocity_profile_speed / curve_cap_speed / comfort_speed is
the minimum — that is the "why is it slow" column.

Report-only (2026-09-25): no score is changed. Gate proposal for later, after a
week of numbers: vs_allowed median >= 0.85 and < 10 % of eligible time under 0.70.

Usage:
    python3 tools/speed_utilization.py <recording.h5>
    from tools.speed_utilization import compute_speed_utilization
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np

STARTUP_S = 10.0
UNDER_UTILISED_RATIO = 0.70
PROPOSED_GATE_MEDIAN = 0.85
PROPOSED_GATE_UNDER_PCT = 10.0

_CAP_FIELDS = {
    "target": "control/target_speed_raw",
    "velocity_profile": "control/velocity_profile_speed_mps",
    "curve_cap": "control/speed_governor_curve_cap_speed",
    "comfort": "control/speed_governor_comfort_speed",
    "preview": "control/speed_governor_preview_speed",
}


def _arr(f: h5py.File, key: str, n: int) -> Optional[np.ndarray]:
    if key not in f:
        return None
    a = np.asarray(f[key][:n], dtype=float)
    return a if a.size == n else None


def compute_speed_utilization(path: str | Path) -> Optional[Dict]:
    """Return the utilisation summary for one recording, or None if the recording
    lacks speed / speed_limit."""
    with h5py.File(path, "r") as f:
        if "vehicle/speed" not in f or "vehicle/speed_limit" not in f or "vehicle/timestamps" not in f:
            return None
        n = len(f["vehicle/speed"])
        v = _arr(f, "vehicle/speed", n)
        lim = _arr(f, "vehicle/speed_limit", n)
        ts = _arr(f, "vehicle/timestamps", n)
        prev = _arr(f, "vehicle/speed_limit_preview", n)
        acc = _arr(f, "vehicle/acc_active", n)
        estop = _arr(f, "control/emergency_stop", n)
        caps = {k: _arr(f, key, n) for k, key in _CAP_FIELDS.items()}
        limiter = None
        if "control/speed_governor_active_limiter" in f:
            raw = f["control/speed_governor_active_limiter"][:n]
            limiter = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in raw])

    if v is None or lim is None or ts is None:
        return None

    rel = ts - ts[0]
    eligible = (rel > STARTUP_S) & (lim > 0.5) & np.isfinite(v) & np.isfinite(lim)
    if acc is not None:
        eligible &= acc < 0.5
    if estop is not None:
        eligible &= estop < 0.5
    if prev is not None:
        # Not braking toward a lower limit ahead.
        eligible &= ~(np.isfinite(prev) & (prev > 0.5) & (prev < lim - 1e-6))

    n_elig = int(eligible.sum())
    out: Dict = {
        "n_frames": int(n),
        "n_eligible": n_elig,
        "eligible_pct": 100.0 * n_elig / max(1, n),
        "startup_s": STARTUP_S,
        "under_ratio": UNDER_UTILISED_RATIO,
        "proposed_gate": {"vs_allowed_median_min": PROPOSED_GATE_MEDIAN, "under_pct_max": PROPOSED_GATE_UNDER_PCT},
    }
    if n_elig < 30:
        out["status"] = "insufficient_eligible_frames"
        return out

    target = caps["target"]
    allowed = lim.copy()
    if target is not None:
        t_ok = np.isfinite(target) & (target > 0.5)
        allowed = np.where(t_ok, np.minimum(lim, target), lim)

    vs_limit = v[eligible] / lim[eligible]
    vs_allowed = v[eligible] / allowed[eligible]

    def _pct(a: np.ndarray, q: float) -> float:
        return float(np.percentile(a, q))

    under_mask = vs_allowed < UNDER_UTILISED_RATIO
    attribution: Dict[str, float] = {}
    if under_mask.any():
        idx = np.flatnonzero(eligible)[under_mask]
        # Binding cap = the smallest finite positive cap at each under-utilised frame.
        names = [k for k in ("target", "velocity_profile", "curve_cap", "comfort", "preview") if caps.get(k) is not None]
        if names:
            stack = np.vstack([np.where(np.isfinite(caps[k][idx]) & (caps[k][idx] > 0.5), caps[k][idx], np.inf) for k in names])
            winner = np.argmin(stack, axis=0)
            has_cap = np.isfinite(stack.min(axis=0))
            counts = np.bincount(winner[has_cap], minlength=len(names))
            total = max(1, int(has_cap.sum()))
            attribution = {names[i]: 100.0 * int(counts[i]) / total for i in range(len(names)) if counts[i] > 0}
        if limiter is not None:
            vals, cnts = np.unique(limiter[idx], return_counts=True)
            attribution["_active_limiter_field"] = {str(a): 100.0 * int(c) / len(idx) for a, c in zip(vals, cnts)}

    out.update({
        "status": "ok",
        "speed_median_mps": float(np.median(v[eligible])),
        "speed_limit_median_mps": float(np.median(lim[eligible])),
        "allowed_median_mps": float(np.median(allowed[eligible])),
        "vs_limit_median": float(np.median(vs_limit)),
        "vs_limit_p10": _pct(vs_limit, 10),
        "vs_allowed_median": float(np.median(vs_allowed)),
        "vs_allowed_p10": _pct(vs_allowed, 10),
        "under_utilised_pct": 100.0 * float(under_mask.mean()),
        "binding_cap_at_under_utilised_pct": attribution,
        "proposed_gate_pass": bool(np.median(vs_allowed) >= PROPOSED_GATE_MEDIAN and 100.0 * float(under_mask.mean()) <= PROPOSED_GATE_UNDER_PCT),
    })
    return out


def format_summary(s: Optional[Dict]) -> str:
    if s is None:
        return "   Speed Utilisation: n/a (no speed / speed_limit fields)"
    if s.get("status") != "ok":
        return f"   Speed Utilisation: n/a ({s.get('status')}; eligible frames {s.get('n_eligible', 0)})"
    attr = s["binding_cap_at_under_utilised_pct"]
    attr_txt = ", ".join(f"{k} {v:.0f}%" for k, v in attr.items() if not k.startswith("_")) or "—"
    gate = "would PASS" if s["proposed_gate_pass"] else "would FAIL"
    return (
        f"   Speed Utilisation (report-only): vs posted limit {s['vs_limit_median']:.2f} (p10 {s['vs_limit_p10']:.2f}) · "
        f"vs allowed {s['vs_allowed_median']:.2f} (p10 {s['vs_allowed_p10']:.2f}) · "
        f"under 0.70 for {s['under_utilised_pct']:.0f}% of eligible time ({s['eligible_pct']:.0f}% of frames eligible)\n"
        f"   Speed Utilisation detail: median speed {s['speed_median_mps']:.1f} m/s vs limit {s['speed_limit_median_mps']:.1f} / allowed {s['allowed_median_mps']:.1f} · "
        f"binding cap when slow: {attr_txt} · proposed gate (≥0.85 vs allowed, ≤10% under): {gate}"
    )


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    s = compute_speed_utilization(argv[1])
    print(format_summary(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
