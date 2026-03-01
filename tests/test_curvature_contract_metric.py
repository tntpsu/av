from pathlib import Path

import h5py
import numpy as np

from av_stack.orchestrator import _curvature_map_authority_lost
from tools.drive_summary_core import analyze_recording_summary


def _write_recording(path: Path, map_authority_lost: np.ndarray | None) -> None:
    n = 100
    t = np.linspace(0.0, 3.3, n, dtype=np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("vehicle/timestamps", data=t)
        f.create_dataset("vehicle/speed", data=np.full(n, 8.0, dtype=np.float32))
        f.create_dataset("control/steering", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/lateral_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/heading_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/total_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/pid_integral", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/emergency_stop", data=np.zeros(n, dtype=np.int8))
        f.create_dataset("control/curvature_primary_abs", data=np.full(n, 0.02, dtype=np.float32))
        f.create_dataset(
            "control/curvature_primary_source",
            data=np.array(["map_track"] * n, dtype=h5py.string_dtype(encoding="utf-8", length=32)),
        )
        f.create_dataset("control/curvature_source_diverged", data=np.zeros(n, dtype=np.int8))
        f.create_dataset(
            "control/curvature_source_divergence_abs",
            data=np.full(n, 0.1, dtype=np.float32),
        )
        f.create_dataset(
            "control/curvature_contract_consistent_all", data=np.ones(n, dtype=np.int8)
        )
        f.create_dataset("control/map_health_ok", data=np.ones(n, dtype=np.int8))
        f.create_dataset("control/track_match_ok", data=np.ones(n, dtype=np.int8))
        f.create_dataset("ground_truth/left_lane_line_x", data=np.full(n, -3.5, dtype=np.float32))
        f.create_dataset("ground_truth/right_lane_line_x", data=np.full(n, 3.5, dtype=np.float32))
        if map_authority_lost is not None:
            f.create_dataset(
                "control/curvature_map_authority_lost",
                data=map_authority_lost.astype(np.int8),
            )


def test_map_authority_metric_semantics() -> None:
    assert _curvature_map_authority_lost(0.02, "map_track") is False
    assert _curvature_map_authority_lost(0.02, "lane_context") is True
    assert _curvature_map_authority_lost(None, "lane_context") is False


def test_drive_summary_reports_map_authority_lost_rate(tmp_path: Path) -> None:
    recording = tmp_path / "map_authority_lost.h5"
    lost = np.array([0] * 50 + [1] * 50, dtype=np.int8)
    _write_recording(recording, lost)

    summary = analyze_recording_summary(recording)
    health = summary["curvature_contract_health"]

    assert health["curvature_map_authority_lost_rate"] == 50.0
    assert any(
        "Map curvature authority lost" in issue
        for issue in summary["executive_summary"].get("key_issues", [])
    )


def test_drive_summary_back_compat_without_map_authority_field(tmp_path: Path) -> None:
    recording = tmp_path / "legacy_no_map_authority.h5"
    _write_recording(recording, None)

    summary = analyze_recording_summary(recording)
    health = summary["curvature_contract_health"]

    assert health["curvature_map_authority_lost_rate"] == 0.0
