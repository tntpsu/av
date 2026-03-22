"""
Oscillation attribution for recordings (PhilViz + CLI).

Computes lead–lag between perception center, trajectory reference, lateral error,
and steering; classifies oscillation subtype; emits prioritized fix recommendations.
Uses the same failure_frame semantics as drive_summary_core (executive_summary).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

SCHEMA_VERSION = "oscillation_attribution_v1"

# Band for oscillation energy (Hz) — steering / error chatter on highway
OSC_BAND_LO_HZ = 0.2
OSC_BAND_HI_HZ = 2.0

# Canonical track id / name for `tracks/hill_highway.yml` (see `name: hill_highway`).
HILL_HIGHWAY_SLUG = "hill_highway"


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _parse_hdf5_metadata_attr(raw: Any) -> Dict[str, Any]:
    """Parse ``attrs['metadata']`` JSON written by ``DataRecorder``."""
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        try:
            out = json.loads(raw)
            return out if isinstance(out, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _slug_match_hill_highway(text: str) -> bool:
    """True if text references the hill_highway track (slug or common variants)."""
    if not text or not str(text).strip():
        return False
    s = str(text).lower().replace("-", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    return HILL_HIGHWAY_SLUG in s or "hillhighway" in s.replace("_", "")


def infer_hill_highway_from_hdf5(
    f: h5py.File,
    recording_basename: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Detect hill-highway scenario from recording metadata (no CLI flag required).

    Primary signals (in order):
    - ``attrs['metadata']`` → ``recording_provenance.track_id``, ``recording_name``,
      ``notes``, ``candidate_label``, ``run_command``
    - ``meta/runtime_config_json`` substring (config overlay path often contains track slug)

    Returns:
        (matched, source_key, detail_dict)
    """
    detail: Dict[str, Any] = {}
    meta = _parse_hdf5_metadata_attr(f.attrs.get("metadata"))

    prov = meta.get("recording_provenance")
    if not isinstance(prov, dict):
        prov = {}

    candidates: List[Tuple[str, str]] = []
    tid = str(prov.get("track_id", "") or "").strip()
    if tid:
        candidates.append(("track_id", tid))
    rname = str(meta.get("recording_name", "") or "").strip()
    if rname:
        candidates.append(("recording_name", rname))
    if recording_basename:
        candidates.append(("filename", recording_basename))
    for key in ("notes", "candidate_label", "run_command"):
        val = prov.get(key)
        if val is None:
            val = meta.get(key)
        s = str(val or "").strip()
        if s:
            candidates.append((key, s))

    for source_key, value in candidates:
        if _slug_match_hill_highway(value):
            detail["matched_field"] = source_key
            detail["matched_value"] = value[:200]
            return True, source_key, detail

    # Optional: runtime config snapshot may reference overlay or track slug
    try:
        if "meta" in f and "runtime_config_json" in f["meta"]:
            rc = f["meta/runtime_config_json"]
            blob = rc[0] if hasattr(rc, "shape") and len(rc.shape) > 0 else rc[()]
            blob_s = (
                blob.decode("utf-8", errors="replace")
                if isinstance(blob, (bytes, bytearray))
                else str(blob)
            )
            if _slug_match_hill_highway(blob_s):
                detail["matched_field"] = "meta/runtime_config_json"
                detail["matched_value"] = "substring hill_highway"
                return True, "runtime_config_json", detail
    except Exception:
        pass

    return False, "none", detail


def _resolve_hill_co_tuning(
    override: Optional[bool],
    *,
    metadata_matched: bool,
    curve_heuristic: bool,
) -> Tuple[bool, str]:
    """
    Whether to append hill-highway co-tuning bullet, and why.

    override:
        None = auto (metadata OR curve heuristic)
        True = force on
        False = force off (suppress bullet entirely)
    """
    if override is True:
        return True, "force_on"
    if override is False:
        return False, "force_off"
    if metadata_matched:
        return True, "metadata_track"
    if curve_heuristic:
        return True, "curve_fraction_heuristic"
    return False, "none"


def _read_1d(f: h5py.File, path: str, n: int) -> Optional[np.ndarray]:
    if path not in f:
        return None
    d = f[path]
    m = min(int(d.shape[0]), n)
    return np.asarray(d[:m], dtype=np.float64)


def get_failure_frame_from_summary(recording_path: Path) -> Optional[int]:
    """First failure frame from drive_summary (OOL / emergency stop)."""
    from drive_summary_core import analyze_recording_summary

    summ = analyze_recording_summary(recording_path, analyze_to_failure=False)
    es = summ.get("executive_summary") or {}
    ff = es.get("failure_frame")
    return int(ff) if ff is not None else None


def _bandpass_energy_ratio(
    y: np.ndarray,
    time_s: np.ndarray,
    lo_hz: float,
    hi_hz: float,
) -> float:
    """Fraction of signal variance in [lo_hz, hi_hz] via FFT (simple)."""
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(time_s, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(t)
    if np.sum(mask) < 32:
        return 0.0
    y = y[mask] - np.nanmean(y[mask])
    t = t[mask]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0 / 30.0
    if dt <= 0:
        dt = 1.0 / 30.0
    n = len(y)
    spec = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, d=dt)
    power = np.abs(spec) ** 2
    total = float(np.sum(power)) + 1e-12
    band = float(np.sum(power[(freqs >= lo_hz) & (freqs <= hi_hz)]))
    return band / total


def _crosscorr_lag_frames(a: np.ndarray, b: np.ndarray) -> Optional[int]:
    """
    Lag (frames) of b relative to a maximizing correlation of first differences.
    Positive lag => b lags a.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 24:
        return None
    a = a[m]
    b = b[m]
    da = np.diff(a)
    db = np.diff(b)
    if len(da) < 16 or len(db) < 16:
        return None
    n = min(len(da), len(db))
    da = da[:n] - np.mean(da[:n])
    db = db[:n] - np.mean(db[:n])
    corr = np.correlate(da, db, mode="full")
    lags = np.arange(-n + 1, n)
    peak = int(np.argmax(np.abs(corr)))
    return int(lags[peak])


def _classify_subtype(
    *,
    limiter_active_rate: float,
    ref_jump_p95_m: float,
    perc_ref_rms_m: float,
    blend_in_transition_rate: float,
    runaway: bool,
) -> str:
    """Single subtype label for oscillation character."""
    if blend_in_transition_rate > 0.25 and runaway:
        return "regime_blend"
    if limiter_active_rate > 0.12 and runaway:
        return "limiter_dominant"
    if ref_jump_p95_m > 0.12:
        return "reference_step"
    if perc_ref_rms_m > 0.04:
        return "perception_jitter"
    if runaway:
        return "control_error_loop"
    return "unclassified"


def _primary_layer_from_lags(
    lag_perc_to_ref: Optional[int],
    lag_ref_to_err: Optional[int],
    lag_err_to_steer: Optional[int],
) -> Tuple[str, str]:
    """
    Heuristic primary layer from cross-correlation lags (diff domain).
    Returns (layer, rationale).
    """
    # Negative lag_perc_to_ref => ref moves after perception (smoothing) — normal.
    # Large |lag_err_to_steer| with limiter elsewhere => control.
    parts: List[str] = []
    pl: str = "unknown"

    if lag_perc_to_ref is not None:
        parts.append(f"perc→ref lag={lag_perc_to_ref}f")
        if abs(lag_perc_to_ref) > 8:
            pl = "trajectory"
    if lag_ref_to_err is not None:
        parts.append(f"ref→err lag={lag_ref_to_err}f")
    if lag_err_to_steer is not None:
        parts.append(f"err→steer lag={lag_err_to_steer}f")
        if abs(lag_err_to_steer) > 6:
            pl = "control"

    if pl == "unknown" and lag_perc_to_ref is not None and abs(lag_perc_to_ref) <= 3:
        pl = "perception"

    rationale = "; ".join(parts) if parts else "insufficient data for lag attribution"
    return pl, rationale


def _recommendations(
    subtype: str,
    primary_layer: str,
    *,
    runaway: bool,
    include_hill_co_tuning: bool,
) -> List[Dict[str, str]]:
    """Prioritized human-readable fixes; hill_highway adds context."""
    out: List[Dict[str, str]] = []

    if subtype == "limiter_dominant":
        out.append(
            {
                "priority": "1",
                "area": "control",
                "action": (
                    "Reduce steering authority gap: review rate/jerk limits and "
                    "curve_entry_assist / limiter schedule vs demand on this road grade."
                ),
            }
        )
        out.append(
            {
                "priority": "2",
                "area": "control",
                "action": (
                    "Inspect per-frame steering_first_limiter_stage_code and "
                    "steering_before_limits vs steering during oscillation windows."
                ),
            }
        )
    elif subtype == "perception_jitter":
        out.append(
            {
                "priority": "1",
                "area": "perception",
                "action": (
                    "Stabilize lane center: tune exponential smoothing, "
                    "clamp events, and low-visibility fallback blend."
                ),
            }
        )
    elif subtype == "reference_step":
        out.append(
            {
                "priority": "1",
                "area": "trajectory",
                "action": (
                    "Smooth reference: increase ref smoothing alpha or tighten "
                    "reference jump clamps / lookahead stability."
                ),
            }
        )
    elif subtype == "regime_blend":
        out.append(
            {
                "priority": "1",
                "area": "control",
                "action": (
                    "Tune PP↔MPC regime blend: reduce blend band width or align "
                    "handoff when lateral error oscillates."
                ),
            }
        )
    elif subtype == "control_error_loop" and runaway:
        out.append(
            {
                "priority": "1",
                "area": "control",
                "action": (
                    "PID / feedforward: reduce aggressive feedback on this profile; "
                    "add damping or lower P gain in oscillation band."
                ),
            }
        )

    if primary_layer == "trajectory" and subtype not in ("reference_step",):
        out.append(
            {
                "priority": str(len(out) + 1),
                "area": "trajectory",
                "action": "Review reference smoothing lag vs perception updates (trajectory layer).",
            }
        )

    if include_hill_co_tuning:
        out.append(
            {
                "priority": str(len(out) + 1),
                "area": "stack",
                "action": (
                    "Hill highway profile: sustained grade + curvature often couples "
                    "speed, feasibility caps, and steering limits — co-tune "
                    "curve_mode speed cap, trajectory curvature contract, and control limits."
                ),
            }
        )

    if not out:
        out.append(
            {
                "priority": "1",
                "area": "stack",
                "action": (
                    "Collect pre-failure segment with full telemetry; re-run attribution "
                    "with --pre-failure-only after confirming oscillation_amplitude_runaway."
                ),
            }
        )

    return out


def analyze_oscillation_attribution(
    recording_path: Path | str,
    *,
    pre_failure_only: bool = True,
    hill_highway_override: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Full oscillation attribution JSON for PhilViz / CLI.

    Args:
        recording_path: Path to HDF5 recording.
        pre_failure_only: If True, analyze only [0, failure_frame] inclusive when known.
        hill_highway_override: If None, ``co_tuning_guidance`` is set from HDF5 metadata
            (``attrs['metadata']`` track_id / recording name) or curve heuristic.
            If True/False, force the hill-highway co-tuning bullet on/off.
    """
    path = Path(recording_path)
    if not path.is_file():
        return {"error": f"Recording not found: {path}", "schema_version": SCHEMA_VERSION}

    warnings: List[str] = []

    from drive_summary_core import analyze_recording_summary

    summ = analyze_recording_summary(path, analyze_to_failure=False)
    es = summ.get("executive_summary") or {}
    failure_frame = es.get("failure_frame")
    failure_frame = int(failure_frame) if failure_frame is not None else None
    cs = summ.get("control_smoothness") or {}
    runaway = bool(cs.get("oscillation_amplitude_runaway", False))
    curve_fraction = _safe_float(cs.get("oscillation_curve_fraction", 0.0))
    zc_hz = _safe_float(cs.get("oscillation_zero_crossing_rate_hz", 0.0))
    rms_slope = _safe_float(cs.get("oscillation_rms_growth_slope_mps", 0.0))

    curve_heuristic = bool(curve_fraction > 0.25 and runaway)

    with h5py.File(path, "r") as f:
        meta_ok, meta_src, meta_detail = infer_hill_highway_from_hdf5(f, path.name)
        n_ctrl = 0
        for key in ("control/steering", "control/lateral_error", "vehicle/timestamps"):
            if key in f:
                n_ctrl = int(f[key].shape[0])
                break
        if n_ctrl <= 0:
            return {"error": "No control series in recording", "schema_version": SCHEMA_VERSION}

        steer = _read_1d(f, "control/steering", n_ctrl)
        lat = _read_1d(f, "control/lateral_error", n_ctrl)
        ts = _read_1d(f, "vehicle/timestamps", n_ctrl)
        if ts is None:
            ts = _read_1d(f, "control/timestamps", n_ctrl)
        if ts is None:
            warnings.append("No timestamps; using uniform 30 Hz")
            ts = np.arange(n_ctrl, dtype=np.float64) / 30.0
        else:
            ts = ts - ts[0]

        mpc_e = _read_1d(f, "control/mpc_e_lat", n_ctrl)
        mpc_on = _read_1d(f, "control/mpc_using_ground_truth", n_ctrl)
        if mpc_e is not None and mpc_on is not None and lat is not None:
            n = min(len(mpc_e), len(mpc_on), len(lat))
            m = mpc_on[:n] > 0.5
            e_combined = np.where(m, mpc_e[:n], lat[:n])
        else:
            e_combined = lat

        ref_x = _read_1d(f, "trajectory/reference_point_x", n_ctrl)
        ref_raw = _read_1d(f, "trajectory/reference_point_raw_x", n_ctrl)

        left = _read_1d(f, "perception/left_lane_line_x", n_ctrl)
        right = _read_1d(f, "perception/right_lane_line_x", n_ctrl)
        perc_center: Optional[np.ndarray] = None
        if left is not None and right is not None:
            n = min(len(left), len(right))
            perc_center = (left[:n] + right[:n]) / 2.0

        before = _read_1d(f, "control/steering_before_limits", n_ctrl)
        limiter = _read_1d(f, "control/steering_first_limiter_stage_code", n_ctrl)
        blend = _read_1d(f, "control/regime_blend_weight", n_ctrl)

        gt_curv = _read_1d(f, "ground_truth/path_curvature", n_ctrl)

    # Align lengths
    arrays = [steer, ts, e_combined]
    if ref_x is not None:
        arrays.append(ref_x)
    if perc_center is not None:
        arrays.append(perc_center)
    n = min(len(a) for a in arrays if a is not None)
    steer = steer[:n]
    ts = ts[:n]
    e_combined = e_combined[:n]
    if ref_x is not None:
        ref_x = ref_x[:n]
    if ref_raw is not None:
        ref_raw = ref_raw[:n]
    if perc_center is not None:
        perc_center = perc_center[:n]
    if before is not None:
        before = before[:n]
    if limiter is not None:
        limiter = limiter[:n]
    if blend is not None:
        blend = blend[:n]
    if gt_curv is not None:
        gt_curv = gt_curv[:n]

    end = n
    if pre_failure_only and failure_frame is not None:
        end = min(n, int(failure_frame) + 1)
    if end < 32:
        warnings.append("Segment too short for attribution; using full recording")
        end = n

    sl = slice(0, end)
    steer_w = steer[sl]
    ts_w = ts[sl]
    err_w = e_combined[sl]
    ref_w = ref_x[sl] if ref_x is not None else None
    perc_w = perc_center[sl] if perc_center is not None else None
    before_w = before[sl] if before is not None else None
    limiter_w = limiter[sl] if limiter is not None else None
    blend_w = blend[sl] if blend is not None else None
    ref_raw_w = ref_raw[sl] if ref_raw is not None else None

    dt = float(np.median(np.diff(ts_w))) if len(ts_w) > 1 else 1.0 / 30.0

    # Metrics
    limiter_active_rate = 0.0
    if limiter_w is not None and len(limiter_w) > 0:
        limiter_active_rate = float(np.mean(limiter_w > 0.5))
    elif before_w is not None and len(before_w) == len(steer_w):
        limiter_active_rate = float(np.mean(np.abs(before_w - steer_w) > 0.02))

    ref_jump_p95_m = 0.0
    if ref_w is not None and len(ref_w) > 2:
        ref_jump_p95_m = float(np.percentile(np.abs(np.diff(ref_w)), 95))

    perc_ref_rms_m = 0.0
    if perc_w is not None and ref_w is not None and len(perc_w) == len(ref_w):
        perc_ref_rms_m = float(np.sqrt(np.nanmean((perc_w - ref_w) ** 2)))

    blend_in_transition_rate = 0.0
    if blend_w is not None and len(blend_w) > 0:
        blend_in_transition_rate = float(np.mean((blend_w > 0.01) & (blend_w < 0.99)))

    subtype = _classify_subtype(
        limiter_active_rate=limiter_active_rate,
        ref_jump_p95_m=ref_jump_p95_m,
        perc_ref_rms_m=perc_ref_rms_m,
        blend_in_transition_rate=blend_in_transition_rate,
        runaway=runaway,
    )

    lag_perc_ref = None
    if perc_w is not None and ref_w is not None:
        lag_perc_ref = _crosscorr_lag_frames(perc_w, ref_w)
    lag_ref_err = None
    if ref_w is not None:
        lag_ref_err = _crosscorr_lag_frames(ref_w, err_w)
    lag_err_steer = _crosscorr_lag_frames(err_w, steer_w)

    primary_layer, lag_rationale = _primary_layer_from_lags(
        lag_perc_ref, lag_ref_err, lag_err_steer
    )
    # Subtype overrides heuristic layer (spectral/limiter evidence wins).
    if subtype == "limiter_dominant":
        primary_layer = "control"
    elif subtype == "perception_jitter":
        primary_layer = "perception"
    elif subtype == "reference_step":
        primary_layer = "trajectory"
    elif subtype == "regime_blend":
        primary_layer = "control"

    e_band_steering = _bandpass_energy_ratio(steer_w, ts_w, OSC_BAND_LO_HZ, OSC_BAND_HI_HZ)
    e_band_err = _bandpass_energy_ratio(err_w, ts_w, OSC_BAND_LO_HZ, OSC_BAND_HI_HZ)

    include_hill, hill_source = _resolve_hill_co_tuning(
        hill_highway_override,
        metadata_matched=meta_ok,
        curve_heuristic=curve_heuristic,
    )

    recs = _recommendations(
        subtype,
        primary_layer,
        runaway=runaway,
        include_hill_co_tuning=include_hill,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "recording": str(path.name),
        "pre_failure_only": pre_failure_only,
        "failure_frame": int(failure_frame) if failure_frame is not None else None,
        "analysis_end_frame": int(end - 1),
        "hill_highway": {
            "override": hill_highway_override,
            "co_tuning_guidance": include_hill,
            "source": hill_source,
            "metadata_match": bool(meta_ok),
            "metadata_field": meta_src if meta_ok else None,
            "curve_heuristic": curve_heuristic,
            "detail": meta_detail,
        },
        "oscillation_summary": {
            "oscillation_amplitude_runaway": runaway,
            "oscillation_curve_fraction": curve_fraction,
            "oscillation_zero_crossing_rate_hz": zc_hz,
            "oscillation_rms_growth_slope_mps": rms_slope,
        },
        "metrics": {
            "limiter_active_rate": round(limiter_active_rate, 4),
            "reference_jump_p95_m": round(ref_jump_p95_m, 4),
            "perception_reference_rms_m": round(perc_ref_rms_m, 4),
            "regime_blend_transition_rate": round(blend_in_transition_rate, 4),
            "band_energy_ratio_steering_0p2_2hz": round(e_band_steering, 4),
            "band_energy_ratio_error_0p2_2hz": round(e_band_err, 4),
        },
        "lead_lag_frames": {
            "perception_to_reference": lag_perc_ref,
            "reference_to_error": lag_ref_err,
            "error_to_steering": lag_err_steer,
        },
        "lag_rationale": lag_rationale,
        "subtype": subtype,
        "primary_layer": primary_layer,
        "recommended_fixes": recs,
        "warnings": warnings,
    }


def main() -> int:
    import argparse
    import json

    p = argparse.ArgumentParser(description="Oscillation attribution (CLI parity with PhilViz).")
    p.add_argument("recording", type=Path, help="HDF5 recording path")
    p.add_argument(
        "--full-run",
        action="store_true",
        help="Analyze entire recording (default: pre-failure segment only)",
    )
    p.add_argument(
        "--force-hill-highway",
        action="store_true",
        help="Force hill-highway co-tuning guidance on (default: auto from recording metadata)",
    )
    p.add_argument(
        "--no-hill-highway",
        action="store_true",
        help="Disable hill-highway co-tuning guidance (overrides metadata)",
    )
    p.add_argument("--json", action="store_true", help="Print JSON only")
    args = p.parse_args()
    if args.force_hill_highway and args.no_hill_highway:
        p.error("Use only one of --force-hill-highway / --no-hill-highway")
    hill_override: Optional[bool] = None
    if args.force_hill_highway:
        hill_override = True
    elif args.no_hill_highway:
        hill_override = False
    pre = not args.full_run
    out = analyze_oscillation_attribution(
        args.recording,
        pre_failure_only=pre,
        hill_highway_override=hill_override,
    )
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(json.dumps(out, indent=2))
    return 0 if "error" not in out else 1


if __name__ == "__main__":
    raise SystemExit(main())
