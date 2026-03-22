#!/usr/bin/env python3
"""
Cadence attribution from HDF5 recordings: wait vs pipeline, queue depth, recommendations.

See docs/plans/cadence_breakdown_tool_plan.md.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORDINGS_DIR = REPO_ROOT / "data" / "recordings"
DEFAULT_AV_STACK_CONFIG = REPO_ROOT / "config" / "av_stack_config.yaml"

SCHEMA_VERSION = "cadence_breakdown_v1"


def load_target_loop_hz_from_av_stack_config(
    config_path: Path | None = None,
    *,
    fallback_hz: float = 20.0,
) -> float:
    """Read ``stack.target_loop_hz`` from the AV stack YAML (repo default: config/av_stack_config.yaml)."""
    path = config_path if config_path is not None else DEFAULT_AV_STACK_CONFIG
    if not path.is_file():
        return fallback_hz
    try:
        import yaml
    except ImportError:  # pragma: no cover - PyYAML is a declared dependency
        return fallback_hz
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            return fallback_hz
        stack = data.get("stack")
        if isinstance(stack, dict):
            raw = stack.get("target_loop_hz")
            if raw is not None:
                v = float(raw)
                if math.isfinite(v) and v > 0.1:
                    return v
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        pass
    return fallback_hz


def find_latest_recording(recordings_dir: Path | None = None) -> Path | None:
    """Most recently modified *.h5 under data/recordings."""
    base = recordings_dir or RECORDINGS_DIR
    if not base.is_dir():
        return None
    candidates = sorted(base.glob("*.h5"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _read_metadata_sync_policy(f: h5py.File) -> str:
    raw = f.attrs.get("metadata")
    if raw is None:
        return ""
    try:
        meta_str = raw.decode("utf-8", "ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
        meta = json.loads(meta_str)
        return str(meta.get("stream_sync_policy", "") or "").lower()
    except Exception:
        return ""


def _read_dataset(f: h5py.File, path: str) -> Optional[np.ndarray]:
    if path not in f:
        return None
    return np.asarray(f[path][:], dtype=np.float64)


def finite_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def _median_finite(a: np.ndarray) -> Optional[float]:
    x = np.asarray(a, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return float(np.median(x))


def _pearson(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    m = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(m)) < 5:
        return None
    x = a[m]
    y = b[m]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _recommendations(
    *,
    availability: dict[str, bool],
    wall_ms: np.ndarray,
    wait_ms: np.ndarray,
    pipeline_ms: np.ndarray,
    e2e_latency_ms: Optional[np.ndarray],
    queue_depth: Optional[np.ndarray],
    frame_id_delta: Optional[np.ndarray],
    unity_render_dt_ms: Optional[np.ndarray],
    ts_minus_rt_ms: Optional[np.ndarray],
    severe_ms: float,
    e2e_mismatch_max_ms: float,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []

    if not availability.get("e2e_mono"):
        out.append(
            {
                "code": "MISSING_E2E",
                "severity": "warn",
                "detail": "No control/e2e_* monotonic timestamps; using control/timestamps only. "
                "Re-record with a stack that logs e2e mono fields for full attribution.",
            }
        )

    if availability.get("e2e_mono") and e2e_latency_ms is not None and pipeline_ms is not None:
        m = np.isfinite(pipeline_ms) & np.isfinite(e2e_latency_ms)
        if np.any(m):
            diff = np.abs(pipeline_ms[m] - e2e_latency_ms[m])
            if float(np.max(diff)) > e2e_mismatch_max_ms:
                out.append(
                    {
                        "code": "E2E_MISMATCH",
                        "severity": "warn",
                        "detail": f"max|pipeline_ms - e2e_latency_ms| = {float(np.max(diff)):.2f} ms "
                        f"(threshold {e2e_mismatch_max_ms:.1f} ms). Treat breakdown as suspect.",
                    }
                )

    if queue_depth is not None:
        q = queue_depth[np.isfinite(queue_depth)]
        if q.size > 10:
            qmax = float(np.max(q))
            p95 = float(np.percentile(q, 95))
            if qmax > 5.0 and p95 > 0.8 * qmax:
                out.append(
                    {
                        "code": "QUEUE_BACKLOG",
                        "severity": "alert",
                        "detail": f"Front queue depth p95={p95:.1f} near max={qmax:.1f}. "
                        "Consumer likely slower than producer; consider async ingest or lower targetFPS.",
                    }
                )

    if frame_id_delta is not None:
        d = frame_id_delta[np.isfinite(frame_id_delta)]
        if d.size > 10:
            if float(np.mean(d)) > 1.05 or float(np.percentile(d, 95)) >= 2.0:
                out.append(
                    {
                        "code": "FRAME_SKIPS",
                        "severity": "warn",
                        "detail": f"stream_front_frame_id_delta mean={float(np.mean(d)):.3f} "
                        f"p95={float(np.percentile(d, 95)):.2f}. Processing cadence below Unity frame steps.",
                    }
                )

    sev = np.isfinite(wall_ms) & (wall_ms > severe_ms)
    if int(np.sum(sev)) >= 5:
        w_med = _median_finite(wait_ms[sev])
        p_med = _median_finite(pipeline_ms[sev] if pipeline_ms is not None else np.array([]))
        if w_med is not None and p_med is not None:
            if w_med > p_med:
                out.append(
                    {
                        "code": "WAIT_INPUT_DOMINATES",
                        "severity": "warn",
                        "detail": "On severe frames, median wait_input_ms exceeds median pipeline_ms. "
                        "Suspect sync HTTP / GIL / asyncio scheduling more than CV kernels.",
                    }
                )
            elif p_med > w_med:
                out.append(
                    {
                        "code": "PIPELINE_DOMINATES",
                        "severity": "warn",
                        "detail": "On severe frames, median pipeline_ms exceeds median wait_input_ms. "
                        "Profile perception/planning/control; add perf_* HDF5 fields when available.",
                    }
                )

    if unity_render_dt_ms is not None:
        u = unity_render_dt_ms[np.isfinite(unity_render_dt_ms)]
        if u.size > 10 and float(np.percentile(u, 95)) > 25.0:
            w_all = _median_finite(wait_ms)
            if w_all is None or w_all < 45.0:
                out.append(
                    {
                        "code": "UNITY_DT_SPIKE",
                        "severity": "info",
                        "detail": f"Unity render Δt (from vehicle/unity_delta_time) p95="
                        f"{float(np.percentile(u, 95)):.1f} ms with modest wait_input. "
                        "Check Unity main thread / GPU readback.",
                    }
                )

    if ts_minus_rt_ms is not None:
        y = ts_minus_rt_ms[np.isfinite(ts_minus_rt_ms)]
        x = np.arange(y.size, dtype=np.float64)
        if y.size > 30:
            slope, _ = np.polyfit(x, y, 1)
            if float(slope) < -0.3:
                out.append(
                    {
                        "code": "LAGGING_REALTIME",
                        "severity": "info",
                        "detail": f"stream_front_timestamp_minus_realtime_ms trends negative "
                        f"(slope ~ {float(slope):.3f} ms/frame). Capture time lagging wall clock.",
                    }
                )

    if not out:
        out.append(
            {
                "code": "NO_FLAGS",
                "severity": "info",
                "detail": "No cadence rule fired with current thresholds.",
            }
        )
    return out


def analyze_cadence(
    recording: Path | str,
    *,
    severe_ms: float = 200.0,
    target_hz: Optional[float] = None,
    stack_config_path: Optional[Path] = None,
    pre_failure_only: bool = False,
    e2e_mismatch_max_ms: float = 5.0,
) -> dict[str, Any]:
    """
    Load a recording and return cadence_breakdown_v1 JSON dict (JSON-serializable).

    If ``target_hz`` is None, it is taken from ``stack.target_loop_hz`` in
    ``stack_config_path`` (default ``config/av_stack_config.yaml``), then ``fallback_hz``
    inside :func:`load_target_loop_hz_from_av_stack_config`.
    """
    path = Path(recording)
    if not path.is_file():
        raise FileNotFoundError(f"Recording not found: {path}")

    resolved_hz = (
        float(target_hz)
        if target_hz is not None
        else load_target_loop_hz_from_av_stack_config(stack_config_path)
    )

    warnings: list[str] = []

    with h5py.File(path, "r") as f:
        sync_policy = _read_metadata_sync_policy(f)

        ctrl_ts = _read_dataset(f, "control/timestamps")
        sent = _read_dataset(f, "control/e2e_control_sent_mono_s")
        inputs = _read_dataset(f, "control/e2e_inputs_ready_mono_s")
        front = _read_dataset(f, "control/e2e_front_ready_mono_s")
        vehicle_mono = _read_dataset(f, "control/e2e_vehicle_ready_mono_s")
        e2e_lat = _read_dataset(f, "control/e2e_latency_ms")
        perf_perception_ms = _read_dataset(f, "control/perf_perception_ms")
        perf_planning_ms = _read_dataset(f, "control/perf_planning_ms")
        perf_control_ms = _read_dataset(f, "control/perf_control_ms")
        perf_hdf5_write_ms = _read_dataset(f, "control/perf_hdf5_write_ms")
        perf_wait_input_ms_ds = _read_dataset(f, "control/perf_wait_input_ms")

        qd = _read_dataset(f, "vehicle/stream_front_queue_depth")
        fid_delta = _read_dataset(f, "vehicle/stream_front_frame_id_delta")
        unity_dt = _read_dataset(f, "vehicle/stream_front_unity_dt_ms")
        unity_delta_s = _read_dataset(f, "vehicle/unity_delta_time")
        ts_rt = _read_dataset(f, "vehicle/stream_front_timestamp_minus_realtime_ms")

        n_candidates = []
        if ctrl_ts is not None:
            n_candidates.append(int(ctrl_ts.shape[0]))
        if sent is not None:
            n_candidates.append(int(sent.shape[0]))
        if inputs is not None:
            n_candidates.append(int(inputs.shape[0]))
        if qd is not None:
            n_candidates.append(int(qd.shape[0]))
        if not n_candidates:
            raise ValueError("Recording has no control/timestamps or e2e datasets")

        n = min(n_candidates)
        if len(set(n_candidates)) > 1:
            warnings.append(
                f"Aligned lengths differ min={n}; truncated control/vehicle series to common prefix."
            )

        def trim(a: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if a is None:
                return None
            if a.shape[0] >= n:
                return np.array(a[:n], dtype=np.float64)
            warnings.append("Dataset shorter than n; padding with NaN not done — check recording.")
            return np.array(a, dtype=np.float64)

        ctrl_ts = trim(ctrl_ts)
        sent = trim(sent)
        inputs = trim(inputs)
        front = trim(front)
        vehicle_mono = trim(vehicle_mono)
        e2e_lat = trim(e2e_lat)
        perf_perception_ms = trim(perf_perception_ms)
        perf_planning_ms = trim(perf_planning_ms)
        perf_control_ms = trim(perf_control_ms)
        perf_hdf5_write_ms = trim(perf_hdf5_write_ms)
        perf_wait_input_ms_ds = trim(perf_wait_input_ms_ds)
        qd = trim(qd)
        fid_delta = trim(fid_delta)
        unity_dt = trim(unity_dt)
        unity_delta_s = trim(unity_delta_s)
        ts_rt = trim(ts_rt)

        end_slice = n
        if pre_failure_only:
            try:
                from drive_summary_core import analyze_recording_summary

                summ = analyze_recording_summary(path, analyze_to_failure=True)
                ff = (summ.get("executive_summary") or {}).get("failure_frame")
                if ff is not None:
                    end_slice = min(n, int(ff) + 1)
            except Exception as exc:  # pragma: no cover - optional dep path
                warnings.append(f"pre_failure_only: could not load failure frame ({exc})")

        def slice_n(a: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if a is None:
                return None
            return np.array(a[:end_slice], dtype=np.float64)

        ctrl_ts = slice_n(ctrl_ts)
        sent = slice_n(sent)
        inputs = slice_n(inputs)
        front = slice_n(front)
        vehicle_mono = slice_n(vehicle_mono)
        e2e_lat = slice_n(e2e_lat)
        perf_perception_ms = slice_n(perf_perception_ms)
        perf_planning_ms = slice_n(perf_planning_ms)
        perf_control_ms = slice_n(perf_control_ms)
        perf_hdf5_write_ms = slice_n(perf_hdf5_write_ms)
        perf_wait_input_ms_ds = slice_n(perf_wait_input_ms_ds)
        qd = slice_n(qd)
        fid_delta = slice_n(fid_delta)
        unity_dt = slice_n(unity_dt)
        unity_delta_s = slice_n(unity_delta_s)
        ts_rt = slice_n(ts_rt)
        n = end_slice

    availability = {
        "control_timestamps": ctrl_ts is not None,
        "e2e_mono": sent is not None and inputs is not None,
        "e2e_latency_ms": e2e_lat is not None,
        "queue_depth": qd is not None,
        "frame_id_delta": fid_delta is not None,
        "unity_dt": unity_dt is not None,
        "unity_delta_time": unity_delta_s is not None,
        "timestamp_minus_realtime": ts_rt is not None,
        "front_vehicle_mono": front is not None and vehicle_mono is not None,
        "perf_perception_ms": perf_perception_ms is not None,
    }

    wall_loop_period_ms = np.full(n, np.nan, dtype=np.float64)
    wait_input_ms = np.full(n, np.nan, dtype=np.float64)
    pipeline_ms = np.full(n, np.nan, dtype=np.float64)
    input_sync_skew_ms = np.full(n, np.nan, dtype=np.float64)

    use_sent_for_wall = sent is not None and (
        bool(np.isfinite(sent).all())
        or (n > 0 and float(np.sum(np.isfinite(sent))) > 0.9 * n)
    )
    if use_sent_for_wall:
        s = sent
        for i in range(1, n):
            if np.isfinite(s[i]) and np.isfinite(s[i - 1]):
                wall_loop_period_ms[i] = float((s[i] - s[i - 1]) * 1000.0)
    elif ctrl_ts is not None:
        t = ctrl_ts
        for i in range(1, n):
            if np.isfinite(t[i]) and np.isfinite(t[i - 1]):
                wall_loop_period_ms[i] = float((t[i] - t[i - 1]) * 1000.0)

    if sent is not None and inputs is not None:
        s = sent
        inp = inputs
        for i in range(1, n):
            if np.isfinite(inp[i]) and np.isfinite(s[i - 1]):
                wait_input_ms[i] = float((inp[i] - s[i - 1]) * 1000.0)
        for i in range(n):
            if np.isfinite(s[i]) and np.isfinite(inp[i]):
                pipeline_ms[i] = float((s[i] - inp[i]) * 1000.0)

    if e2e_lat is not None:
        m = np.isfinite(e2e_lat)
        pipeline_ms[m] = e2e_lat[m]

    if front is not None and vehicle_mono is not None:
        for i in range(n):
            if np.isfinite(front[i]) and np.isfinite(vehicle_mono[i]):
                input_sync_skew_ms[i] = float(abs(front[i] - vehicle_mono[i]) * 1000.0)

    severe_mask = np.isfinite(wall_loop_period_ms) & (wall_loop_period_ms > severe_ms)
    normal_mask = np.isfinite(wall_loop_period_ms) & ~severe_mask

    def stats_for(mask: np.ndarray, series: np.ndarray) -> dict[str, Any]:
        return finite_stats(series[mask & np.isfinite(series)])

    series_map = {
        "wall_loop_period_ms": wall_loop_period_ms,
        "wait_input_ms": wait_input_ms,
        "pipeline_ms": pipeline_ms,
        "input_sync_skew_ms": input_sync_skew_ms,
    }
    if qd is not None:
        series_map["stream_front_queue_depth"] = qd
    if fid_delta is not None:
        series_map["stream_front_frame_id_delta"] = fid_delta
    if unity_dt is not None:
        series_map["stream_front_unity_dt_ms"] = unity_dt
    unity_render_dt_ms: Optional[np.ndarray] = None
    if unity_delta_s is not None:
        unity_render_dt_ms = np.full(n, np.nan, dtype=np.float64)
        m = np.isfinite(unity_delta_s) & (unity_delta_s > 0) & (unity_delta_s < 1.0)
        unity_render_dt_ms[m] = unity_delta_s[m] * 1000.0
        series_map["unity_render_frame_dt_ms"] = unity_render_dt_ms

    if perf_perception_ms is not None:
        series_map["perf_perception_ms"] = perf_perception_ms
    if perf_planning_ms is not None:
        series_map["perf_planning_ms"] = perf_planning_ms
    if perf_control_ms is not None:
        series_map["perf_control_ms"] = perf_control_ms
    if perf_hdf5_write_ms is not None:
        series_map["perf_hdf5_write_ms"] = perf_hdf5_write_ms
    if perf_wait_input_ms_ds is not None:
        series_map["perf_wait_input_ms"] = perf_wait_input_ms_ds

    stats_all = {name: finite_stats(arr) for name, arr in series_map.items()}
    stats_severe = {name: stats_for(severe_mask, arr) for name, arr in series_map.items()}
    stats_normal = {name: stats_for(normal_mask, arr) for name, arr in series_map.items()}

    wall_finite = wall_loop_period_ms[np.isfinite(wall_loop_period_ms)]
    if wall_finite.size > 0:
        mean_period_s = float(np.mean(wall_finite)) / 1000.0
        control_mean_hz = float(1.0 / mean_period_s) if mean_period_s > 1e-6 else None
    else:
        control_mean_hz = None

    rates_hz: dict[str, Any] = {"control_mean": control_mean_hz}
    if wall_finite.size > 0:
        rates_hz["control_p10_ms"] = float(np.percentile(wall_finite, 10))
        rates_hz["control_p50_ms"] = float(np.percentile(wall_finite, 50))
        rates_hz["control_p95_ms"] = float(np.percentile(wall_finite, 95))

    severe_count = int(np.sum(severe_mask))
    denom = int(np.sum(np.isfinite(wall_loop_period_ms)))
    severe_rate = float(severe_count / denom) if denom > 0 else 0.0

    target_period_ms = 1000.0 / resolved_hz if resolved_hz > 0 else None
    headroom_hint = None
    if control_mean_hz and resolved_hz > 0:
        headroom_hint = f"Mean loop {control_mean_hz:.2f} Hz vs target {resolved_hz:.1f} Hz."

    corr_hints: dict[str, Any] = {}
    c = _pearson(wait_input_ms[severe_mask], wall_loop_period_ms[severe_mask])
    if c is not None:
        corr_hints["pearson_wait_vs_wall_severe"] = c

    recs = _recommendations(
        availability={k: bool(v) for k, v in availability.items()},
        wall_ms=wall_loop_period_ms,
        wait_ms=wait_input_ms,
        pipeline_ms=pipeline_ms,
        e2e_latency_ms=e2e_lat,
        queue_depth=qd,
        frame_id_delta=fid_delta,
        unity_render_dt_ms=unity_render_dt_ms,
        ts_minus_rt_ms=ts_rt,
        severe_ms=severe_ms,
        e2e_mismatch_max_ms=e2e_mismatch_max_ms,
    )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "recording": str(path.resolve()),
        "frame_count": int(n),
        "sync_policy": sync_policy or None,
        "target_hz": float(resolved_hz),
        "severe_ms_threshold": float(severe_ms),
        "availability": availability,
        "warnings": warnings,
        "rates_hz": rates_hz,
        "headroom_hint": headroom_hint,
        "target_period_ms": target_period_ms,
        "severe": {"rate": severe_rate, "count": severe_count, "denom": denom},
        "stats_all": stats_all,
        "stats_severe": stats_severe,
        "stats_normal": stats_normal,
        "correlation_hints": corr_hints,
        "recommendations": recs,
    }
    return _json_safe(payload)


def _maybe_plot(
    prefix: Path,
    wait_ms: np.ndarray,
    wall_ms: np.ndarray,
    severe_mask: np.ndarray,
    qd: Optional[np.ndarray],
    pipeline_ms: np.ndarray,
    unity_dt: Optional[np.ndarray],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    prefix.parent.mkdir(parents=True, exist_ok=True)
    m = np.isfinite(wait_ms) & np.isfinite(wall_ms)
    if np.sum(m) > 5:
        plt.figure(figsize=(6, 5))
        sc = plt.scatter(wait_ms[m], wall_ms[m], c=severe_mask[m].astype(float), s=8, alpha=0.5)
        plt.xlabel("wait_input_ms")
        plt.ylabel("wall_loop_period_ms")
        plt.title("Cadence: wait vs loop period")
        plt.colorbar(sc, label="severe")
        plt.tight_layout()
        plt.savefig(prefix.with_name(prefix.name + "_scatter.png"), dpi=120)
        plt.close()

    # Downsampled time series
    max_pts = 2000
    n = wall_ms.size
    if n > 0:
        idx = np.linspace(0, n - 1, num=min(n, max_pts), dtype=int)
        plt.figure(figsize=(8, 4))
        plt.plot(idx, wall_ms[idx], label="wall_loop_period_ms", alpha=0.8)
        plt.plot(idx, pipeline_ms[idx], label="pipeline_ms", alpha=0.8)
        if qd is not None:
            plt.plot(idx, qd[idx], label="queue_depth", alpha=0.5)
        plt.legend()
        plt.xlabel("frame index (subsampled)")
        plt.tight_layout()
        plt.savefig(prefix.with_name(prefix.name + "_series.png"), dpi=120)
        plt.close()

    if unity_dt is not None and np.sum(np.isfinite(unity_dt)) > 5:
        plt.figure(figsize=(6, 4))
        u = unity_dt[np.isfinite(unity_dt)]
        plt.hist(u, bins=40, alpha=0.7, label="all")
        plt.xlabel("stream_front_unity_dt_ms")
        plt.tight_layout()
        plt.savefig(prefix.with_name(prefix.name + "_unity_dt_hist.png"), dpi=120)
        plt.close()


def analyze_cadence_with_plots(
    recording: Path | str,
    plot_prefix: Optional[Path] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """analyze_cadence plus optional PNGs when matplotlib is installed."""
    payload = analyze_cadence(recording, **kwargs)
    if plot_prefix is not None:
        # Re-load minimal arrays for plotting (avoid storing in payload)
        with h5py.File(recording, "r") as f:
            n = int(payload["frame_count"])
            sent = _read_dataset(f, "control/e2e_control_sent_mono_s")
            inputs = _read_dataset(f, "control/e2e_inputs_ready_mono_s")
            ctrl_ts = _read_dataset(f, "control/timestamps")
            e2e_lat = _read_dataset(f, "control/e2e_latency_ms")
            qd = _read_dataset(f, "vehicle/stream_front_queue_depth")
            unity_dt = _read_dataset(f, "vehicle/stream_front_unity_dt_ms")
        sent = sent[:n] if sent is not None else None
        inputs = inputs[:n] if inputs is not None else None
        wall = np.full(n, np.nan)
        wait = np.full(n, np.nan)
        pipe = np.full(n, np.nan)
        if sent is not None:
            for i in range(1, n):
                if np.isfinite(sent[i]) and np.isfinite(sent[i - 1]):
                    wall[i] = (sent[i] - sent[i - 1]) * 1000.0
        elif ctrl_ts is not None:
            ctrl_ts = ctrl_ts[:n]
            for i in range(1, n):
                if np.isfinite(ctrl_ts[i]) and np.isfinite(ctrl_ts[i - 1]):
                    wall[i] = (ctrl_ts[i] - ctrl_ts[i - 1]) * 1000.0
        if sent is not None and inputs is not None:
            for i in range(1, n):
                if np.isfinite(inputs[i]) and np.isfinite(sent[i - 1]):
                    wait[i] = (inputs[i] - sent[i - 1]) * 1000.0
            for i in range(n):
                if np.isfinite(sent[i]) and np.isfinite(inputs[i]):
                    pipe[i] = (sent[i] - inputs[i]) * 1000.0
        if e2e_lat is not None:
            e2e_lat = e2e_lat[:n]
            m = np.isfinite(e2e_lat)
            pipe[m] = e2e_lat[m]
        qd = qd[:n] if qd is not None else None
        unity_dt = unity_dt[:n] if unity_dt is not None else None
        sev = np.isfinite(wall) & (wall > float(kwargs.get("severe_ms", 200.0)))
        _maybe_plot(Path(plot_prefix), wait, wall, sev, qd, pipe, unity_dt)
    return payload


def _print_report(payload: dict[str, Any]) -> None:
    print(f"Cadence breakdown ({payload.get('schema_version')})")
    print(f"  Recording: {payload.get('recording')}")
    print(f"  Frames: {payload.get('frame_count')}  sync_policy={payload.get('sync_policy')}")
    for w in payload.get("warnings") or []:
        print(f"  WARNING: {w}")
    rh = payload.get("rates_hz") or {}
    print(
        f"  Rate: mean={rh.get('control_mean')} Hz  "
        f"p50_period~={rh.get('control_p50_ms')} ms  "
        f"p95_period~={rh.get('control_p95_ms')} ms  (target {payload.get('target_hz')} Hz)"
    )
    sev = payload.get("severe") or {}
    print(f"  Severe (>{payload.get('severe_ms_threshold')} ms): {sev.get('rate', 0)*100:.2f}%  ({sev.get('count')}/{sev.get('denom')})")
    if payload.get("headroom_hint"):
        print(f"  {payload['headroom_hint']}")
    print("\n  Stats (all frames):")
    for name, st in (payload.get("stats_all") or {}).items():
        if st.get("count", 0) == 0:
            continue
        p50 = st.get("p50")
        p95 = st.get("p95")
        mx = st.get("max")
        p50s = f"{p50:.2f}" if isinstance(p50, (int, float)) and math.isfinite(float(p50)) else str(p50)
        p95s = f"{p95:.2f}" if isinstance(p95, (int, float)) and math.isfinite(float(p95)) else str(p95)
        mxs = f"{mx:.2f}" if isinstance(mx, (int, float)) and math.isfinite(float(mx)) else str(mx)
        print(f"    {name}: n={st['count']} p50={p50s} p95={p95s} max={mxs}")
    print("\n  Recommendations:")
    for r in payload.get("recommendations") or []:
        print(f"    [{r.get('severity')}] {r.get('code')}: {r.get('detail')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cadence attribution from HDF5 recording.")
    parser.add_argument("recording", nargs="?", help="Path to .h5")
    parser.add_argument("--latest", action="store_true", help="Use latest *.h5 in data/recordings")
    parser.add_argument("--output-json", default="", help="Write JSON bundle")
    parser.add_argument("--severe-ms", type=float, default=200.0)
    parser.add_argument(
        "--target-hz",
        type=float,
        default=None,
        metavar="HZ",
        help="Expected control-loop Hz for headroom hints (default: stack.target_loop_hz from "
        "--stack-config or config/av_stack_config.yaml).",
    )
    parser.add_argument(
        "--stack-config",
        default="",
        metavar="PATH",
        help="YAML with stack.target_loop_hz; used when --target-hz is omitted.",
    )
    parser.add_argument("--plot", default="", help="PNG path prefix (optional)")
    parser.add_argument("--pre-failure-only", action="store_true")
    args = parser.parse_args()

    if args.latest:
        rec = find_latest_recording()
    elif args.recording:
        rec = Path(args.recording)
    else:
        rec = find_latest_recording()
    if rec is None or not Path(rec).exists():
        raise SystemExit("No recording found. Pass a path or use --latest with data in data/recordings.")

    stack_path: Optional[Path] = (
        Path(args.stack_config).expanduser() if str(args.stack_config).strip() else None
    )

    plot_kw: dict[str, Any] = {
        "severe_ms": args.severe_ms,
        "target_hz": args.target_hz,
        "pre_failure_only": args.pre_failure_only,
        "stack_config_path": stack_path,
    }
    if args.plot:
        payload = analyze_cadence_with_plots(rec, plot_prefix=Path(args.plot), **plot_kw)
    else:
        payload = analyze_cadence(rec, **plot_kw)
    _print_report(payload)
    if args.output_json:
        outp = Path(args.output_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)


if __name__ == "__main__":
    main()
