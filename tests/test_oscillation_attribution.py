"""Tests for oscillation attribution (CLI + PhilViz parity)."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from tools.oscillation_attribution import (
    SCHEMA_VERSION,
    _classify_subtype,
    _slug_match_hill_highway,
    analyze_oscillation_attribution,
    infer_hill_highway_from_hdf5,
)


def test_slug_match_hill_highway() -> None:
    assert _slug_match_hill_highway("hill_highway")
    assert _slug_match_hill_highway("recording_hill_highway_20260101")
    assert not _slug_match_hill_highway("highway_65")


def test_infer_hill_highway_from_metadata_track_id(tmp_path: Path) -> None:
    path = tmp_path / "meta_run.h5"
    meta = {
        "recording_name": "test",
        "recording_provenance": {"track_id": "hill_highway"},
    }
    with h5py.File(path, "w") as f:
        f.attrs["metadata"] = json.dumps(meta)
    with h5py.File(path, "r") as f:
        ok, src, detail = infer_hill_highway_from_hdf5(f, path.name)
    assert ok is True
    assert src == "track_id"
    assert detail.get("matched_field") == "track_id"


def test_infer_hill_highway_from_filename(tmp_path: Path) -> None:
    path = tmp_path / "recording_hill_highway_20260101.h5"
    with h5py.File(path, "w") as f:
        f.attrs["metadata"] = json.dumps({"recording_name": "x"})
    with h5py.File(path, "r") as f:
        ok, src, _ = infer_hill_highway_from_hdf5(f, path.name)
    assert ok is True
    assert src == "filename"


def test_classify_subtype_mapping() -> None:
    assert (
        _classify_subtype(
            limiter_active_rate=0.2,
            ref_jump_p95_m=0.01,
            perc_ref_rms_m=0.01,
            blend_in_transition_rate=0.0,
            runaway=True,
        )
        == "limiter_dominant"
    )
    assert (
        _classify_subtype(
            limiter_active_rate=0.0,
            ref_jump_p95_m=0.2,
            perc_ref_rms_m=0.01,
            blend_in_transition_rate=0.0,
            runaway=False,
        )
        == "reference_step"
    )
    assert (
        _classify_subtype(
            limiter_active_rate=0.0,
            ref_jump_p95_m=0.01,
            perc_ref_rms_m=0.1,
            blend_in_transition_rate=0.0,
            runaway=False,
        )
        == "perception_jitter"
    )


def test_analyze_returns_schema(tmp_path: Path) -> None:
    path = tmp_path / "osc_attr.h5"
    n = 96
    t = np.arange(n, dtype=np.float64) * (1.0 / 30.0)
    steer = (0.1 * np.sin(2 * np.pi * 0.7 * t)).astype(np.float32)
    lat = (0.15 * np.sin(2 * np.pi * 0.7 * t + 0.2)).astype(np.float32)
    left = np.full(n, 1.0, dtype=np.float32)
    right = np.full(n, -1.0, dtype=np.float32)
    ref_x = (0.05 * np.sin(2 * np.pi * 0.5 * t)).astype(np.float32)
    ref_raw = ref_x + (0.02 * np.random.RandomState(1).standard_normal(n)).astype(np.float32)

    with h5py.File(path, "w") as f:
        f.create_dataset("vehicle/timestamps", data=t)
        f.create_dataset("vehicle/speed", data=np.full(n, 8.0, dtype=np.float32))
        f.create_dataset("control/steering", data=steer)
        f.create_dataset("control/lateral_error", data=lat)
        f.create_dataset("control/heading_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/total_error", data=lat)
        f.create_dataset("control/pid_integral", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/emergency_stop", data=np.zeros(n, dtype=np.int8))
        f.create_dataset("ground_truth/left_lane_line_x", data=np.full(n, -3.5, dtype=np.float32))
        f.create_dataset("ground_truth/right_lane_line_x", data=np.full(n, 3.5, dtype=np.float32))
        f.create_dataset("ground_truth/path_curvature", data=np.full(n, 0.001, dtype=np.float32))
        f.create_dataset("perception/left_lane_line_x", data=left)
        f.create_dataset("perception/right_lane_line_x", data=right)
        f.create_dataset("trajectory/reference_point_x", data=ref_x)
        f.create_dataset("trajectory/reference_point_raw_x", data=ref_raw)

    out = analyze_oscillation_attribution(path, pre_failure_only=False, hill_highway_override=True)
    assert "error" not in out
    assert out["schema_version"] == SCHEMA_VERSION
    assert "recommended_fixes" in out
    assert isinstance(out["recommended_fixes"], list)
    assert "lead_lag_frames" in out
    assert "subtype" in out
    assert "primary_layer" in out
    assert "hill_highway" in out
    assert out["hill_highway"]["source"] == "force_on"
    # Hill-highway flag should add stack-level guidance
    texts = " ".join(r["action"] for r in out["recommended_fixes"])
    assert "Hill highway" in texts or "hill" in texts.lower()


def test_pre_failure_truncation_uses_failure_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "trunc.h5"
    n = 100
    t = np.arange(n, dtype=np.float64) * 0.033
    with h5py.File(path, "w") as f:
        f.create_dataset("vehicle/timestamps", data=t)
        f.create_dataset("vehicle/speed", data=np.full(n, 5.0, dtype=np.float32))
        f.create_dataset("control/steering", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/lateral_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/heading_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/total_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/pid_integral", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/emergency_stop", data=np.zeros(n, dtype=np.int8))
        f.create_dataset("ground_truth/left_lane_line_x", data=np.full(n, -3.5, dtype=np.float32))
        f.create_dataset("ground_truth/right_lane_line_x", data=np.full(n, 3.5, dtype=np.float32))

    def fake_summary(p: Path, analyze_to_failure: bool = False, h5_file=None):
        return {
            "executive_summary": {"failure_frame": 40},
            "control_smoothness": {
                "oscillation_amplitude_runaway": False,
                "oscillation_curve_fraction": 0.0,
                "oscillation_zero_crossing_rate_hz": 0.0,
                "oscillation_rms_growth_slope_mps": 0.0,
            },
        }

    monkeypatch.setattr("drive_summary_core.analyze_recording_summary", fake_summary)

    out = analyze_oscillation_attribution(path, pre_failure_only=True)
    assert out.get("analysis_end_frame") == 40
    assert out.get("failure_frame") == 40
