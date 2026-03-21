"""Tests for tools/analyze/cadence_breakdown.py."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_ANALYZE_PKG = _REPO / "tools" / "analyze"
if str(_ANALYZE_PKG) not in sys.path:
    sys.path.insert(0, str(_ANALYZE_PKG))

from cadence_breakdown import (  # noqa: E402
    SCHEMA_VERSION,
    analyze_cadence,
    find_latest_recording,
    load_target_loop_hz_from_av_stack_config,
    main,
)


def _write_synthetic_recording(
    path: Path,
    *,
    queue_backlog: bool = False,
    frame_skip: bool = False,
    n: int = 6,
) -> None:
    """Minimal HDF5 with e2e + vehicle streams."""
    # Wall periods ~50ms; last step has a large gap for severity testing
    sent = np.zeros(n, dtype=np.float64)
    t = 0.0
    for i in range(n):
        sent[i] = t
        if i < n - 1:
            dt = 0.21 if i == n - 2 else 0.05
            t += dt
    inputs = sent - 0.001  # 1ms pipeline
    front = inputs - 0.0005
    vehicle_m = inputs - 0.0002
    e2e_lat = (sent - inputs) * 1000.0
    ctrl_ts = sent.copy()

    qd = np.ones(n, dtype=np.float64) * 10.0
    if queue_backlog:
        qd = np.linspace(100.0, 120.0, n).astype(np.float64)

    fid = np.ones(n, dtype=np.float64)
    if frame_skip:
        fid[:] = 2.0

    unity_dt = np.ones(n, dtype=np.float64) * 16.7
    ts_rt = np.zeros(n, dtype=np.float64)

    with h5py.File(path, "w") as f:
        f.attrs["metadata"] = '{"stream_sync_policy": "aligned"}'
        g = f.create_group("control")
        g.create_dataset("timestamps", data=ctrl_ts.astype(np.float32))
        g.create_dataset("e2e_control_sent_mono_s", data=sent)
        g.create_dataset("e2e_inputs_ready_mono_s", data=inputs)
        g.create_dataset("e2e_front_ready_mono_s", data=front)
        g.create_dataset("e2e_vehicle_ready_mono_s", data=vehicle_m)
        g.create_dataset("e2e_latency_ms", data=e2e_lat.astype(np.float32))
        v = f.create_group("vehicle")
        v.create_dataset("stream_front_queue_depth", data=qd.astype(np.float32))
        v.create_dataset("stream_front_frame_id_delta", data=fid.astype(np.float32))
        v.create_dataset("stream_front_unity_dt_ms", data=unity_dt.astype(np.float32))
        v.create_dataset("stream_front_timestamp_minus_realtime_ms", data=ts_rt.astype(np.float32))


def test_analyze_cadence_derived_and_severe(tmp_path: Path) -> None:
    path = tmp_path / "cadence_synth.h5"
    _write_synthetic_recording(path)
    out = analyze_cadence(path, severe_ms=200.0, target_hz=30.0)
    assert out["schema_version"] == SCHEMA_VERSION
    assert out["frame_count"] == 6
    assert out["availability"]["e2e_mono"] is True
    sev = out["severe"]
    assert sev["denom"] == 5  # wall periods for i=1..n-1
    assert sev["count"] >= 1
    wall = out["stats_all"]["wall_loop_period_ms"]
    assert wall["max"] is not None and wall["max"] >= 200.0
    assert len(out["recommendations"]) >= 1


def test_queue_backlog_recommendation(tmp_path: Path) -> None:
    path = tmp_path / "cadence_queue.h5"
    _write_synthetic_recording(path, queue_backlog=True, n=20)
    out = analyze_cadence(path)
    codes = {r["code"] for r in out["recommendations"]}
    assert "QUEUE_BACKLOG" in codes


def test_frame_skip_recommendation(tmp_path: Path) -> None:
    path = tmp_path / "cadence_skip.h5"
    _write_synthetic_recording(path, frame_skip=True, n=20)
    out = analyze_cadence(path)
    codes = {r["code"] for r in out["recommendations"]}
    assert "FRAME_SKIPS" in codes


def test_missing_e2e_degraded(tmp_path: Path) -> None:
    path = tmp_path / "cadence_min.h5"
    n = 4
    ts = np.array([0.0, 0.04, 0.08, 0.30], dtype=np.float32)
    with h5py.File(path, "w") as f:
        g = f.create_group("control")
        g.create_dataset("timestamps", data=ts)
        f.create_group("vehicle")
    out = analyze_cadence(path, severe_ms=200.0)
    assert out["availability"]["e2e_mono"] is False
    codes = {r["code"] for r in out["recommendations"]}
    assert "MISSING_E2E" in codes


@pytest.mark.parametrize(
    "name",
    [
        "golden_gt_20260214_oval_latest_20s.h5",
        "golden_gt_20260214_sloop_latest_45s.h5",
        "golden_gt_20260215_sloop_latest_30s.h5",
    ],
)
def test_golden_smoke_if_present(name: str) -> None:
    root = Path(__file__).resolve().parents[1]
    p = root / "data" / "recordings" / name
    if not p.is_file():
        pytest.skip(f"Golden not in workspace: {p}")
    out = analyze_cadence(p)
    assert out["frame_count"] > 0
    assert out["schema_version"] == SCHEMA_VERSION


def test_find_latest_recording_missing_dir(tmp_path: Path) -> None:
    assert find_latest_recording(tmp_path / "nope") is None


def test_load_target_loop_hz_from_minimal_yaml(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("stack:\n  target_loop_hz: 17.5\n", encoding="utf-8")
    assert load_target_loop_hz_from_av_stack_config(cfg) == 17.5


def test_load_target_loop_hz_fallback_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    assert load_target_loop_hz_from_av_stack_config(missing, fallback_hz=99.0) == 99.0


def test_analyze_cadence_resolves_target_hz_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "cadence_synth.h5"
    _write_synthetic_recording(path)
    cfg = tmp_path / "stack.yaml"
    cfg.write_text("stack:\n  target_loop_hz: 25.0\n", encoding="utf-8")
    out = analyze_cadence(path, stack_config_path=cfg)
    assert out["target_hz"] == 25.0


def test_cli_writes_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "cli.h5"
    _write_synthetic_recording(path)
    json_path = tmp_path / "out.json"
    monkeypatch.setattr("sys.argv", ["cadence_breakdown.py", str(path), "--output-json", str(json_path)])
    main()
    assert json_path.is_file()
    assert json_path.read_text().startswith("{")
