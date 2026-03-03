"""
Tests for PhilViz MPC integration (Phase 2.4b).

Uses synthetic HDF5 fixtures to verify:
- drive_summary_core MPC health summary
- Triage engine MPC patterns
- Layer health MPC flags
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest


def _create_synthetic_hdf5(
    path: Path,
    n_frames: int = 300,
    include_mpc: bool = True,
    mpc_start_frame: int = 100,
    infeasible_frames: list | None = None,
    fallback_frames: list | None = None,
    solve_time_ms: float = 1.0,
):
    """Create a minimal HDF5 recording with optional MPC fields."""
    with h5py.File(path, "w") as f:
        # Minimal required fields
        t = np.arange(n_frames, dtype=np.float64) / 30.0
        f.create_dataset("vehicle/timestamp", data=t)
        f.create_dataset("perception/confidence", data=np.full(n_frames, 0.8, dtype=np.float32))
        f.create_dataset("perception/num_lanes_detected", data=np.full(n_frames, 2, dtype=np.int8))
        f.create_dataset("perception/using_stale_data", data=np.zeros(n_frames, dtype=np.int8))
        f.create_dataset("control/steering", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/lateral_error", data=np.random.randn(n_frames).astype(np.float32) * 0.05)
        f.create_dataset("control/throttle", data=np.full(n_frames, 0.3, dtype=np.float32))
        f.create_dataset("control/brake", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/timestamps", data=t)

        if include_mpc:
            # Regime: 0 = PP, 1 = LMPC
            regime = np.zeros(n_frames, dtype=np.int8)
            regime[mpc_start_frame:] = 1
            f.create_dataset("control/regime", data=regime)

            blend = np.zeros(n_frames, dtype=np.float32)
            blend[mpc_start_frame:] = 1.0
            f.create_dataset("control/regime_blend_weight", data=blend)

            f.create_dataset("control/mpc_e_lat", data=np.random.randn(n_frames).astype(np.float32) * 0.01)
            f.create_dataset("control/mpc_e_heading", data=np.random.randn(n_frames).astype(np.float32) * 0.02)
            f.create_dataset("control/mpc_using_ground_truth", data=np.ones(n_frames, dtype=np.float32))

            feasible = np.ones(n_frames, dtype=np.int8)
            if infeasible_frames:
                for fi in infeasible_frames:
                    if fi < n_frames:
                        feasible[fi] = 0
            f.create_dataset("control/mpc_feasible", data=feasible)

            solve = np.full(n_frames, solve_time_ms, dtype=np.float32)
            f.create_dataset("control/mpc_solve_time_ms", data=solve)

            fb = np.zeros(n_frames, dtype=np.int8)
            if fallback_frames:
                for fi in fallback_frames:
                    if fi < n_frames:
                        fb[fi] = 1
            f.create_dataset("control/mpc_fallback_active", data=fb)

            failures = np.zeros(n_frames, dtype=np.int16)
            f.create_dataset("control/mpc_consecutive_failures", data=failures)

            f.create_dataset("control/mpc_kappa_ref", data=np.zeros(n_frames, dtype=np.float32))
            f.create_dataset("control/mpc_gt_cross_track_m", data=np.zeros(n_frames, dtype=np.float32))
            f.create_dataset("control/mpc_gt_heading_error_rad", data=np.zeros(n_frames, dtype=np.float32))


@pytest.fixture
def mpc_recording(tmp_path):
    path = tmp_path / "rec_mpc.h5"
    _create_synthetic_hdf5(path, n_frames=300, include_mpc=True, mpc_start_frame=100)
    return path


@pytest.fixture
def pp_only_recording(tmp_path):
    path = tmp_path / "rec_pp.h5"
    _create_synthetic_hdf5(path, n_frames=300, include_mpc=False)
    return path


@pytest.fixture
def mpc_infeasible_recording(tmp_path):
    path = tmp_path / "rec_infeasible.h5"
    # 2% of 200 MPC frames = 4 infeasible frames
    infeasible = [110, 150, 200, 250]
    _create_synthetic_hdf5(
        path, n_frames=300, include_mpc=True, mpc_start_frame=100,
        infeasible_frames=infeasible,
    )
    return path


@pytest.fixture
def mpc_fallback_recording(tmp_path):
    path = tmp_path / "rec_fallback.h5"
    _create_synthetic_hdf5(
        path, n_frames=300, include_mpc=True, mpc_start_frame=100,
        fallback_frames=[150, 151, 152, 153, 154],
    )
    return path


# ── Test 1: Summary MPC health present ──────────────────────────────────

def test_summary_mpc_health_present(mpc_recording):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
    from drive_summary_core import analyze_recording_summary

    summary = analyze_recording_summary(mpc_recording)
    mpc_health = summary.get("mpc_health")
    assert mpc_health is not None, "mpc_health should be present for MPC recording"
    assert mpc_health["mpc_frames"] == 200
    assert mpc_health["pp_frames"] == 100
    assert mpc_health["mpc_rate"] > 0.6
    assert mpc_health["feasibility_rate"] == 1.0
    assert mpc_health["feasibility_gate_pass"] is True
    assert mpc_health["solve_time_p95_ms"] is not None
    assert mpc_health["solve_time_gate_pass"] is True


# ── Test 2: Summary MPC health absent for PP ────────────────────────────

def test_summary_mpc_health_absent_for_pp(pp_only_recording):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
    from drive_summary_core import analyze_recording_summary

    summary = analyze_recording_summary(pp_only_recording)
    assert summary.get("mpc_health") is None


# ── Test 3: Triage MPC infeasible pattern ───────────────────────────────

def test_triage_mpc_infeasible_pattern(mpc_infeasible_recording):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools', 'debug_visualizer'))
    from backend.triage_engine import TriageEngine

    engine = TriageEngine(mpc_infeasible_recording)
    result = engine.generate_triage()
    pattern_ids = [m["pattern_id"] for m in result["matched_patterns"]]
    assert "mpc_infeasible_burst" in pattern_ids, (
        f"Expected mpc_infeasible_burst pattern, got: {pattern_ids}"
    )


# ── Test 4: Triage MPC patterns skip PP ─────────────────────────────────

def test_triage_mpc_patterns_skip_pp(pp_only_recording):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools', 'debug_visualizer'))
    from backend.triage_engine import TriageEngine

    engine = TriageEngine(pp_only_recording)
    result = engine.generate_triage()
    mpc_patterns = [
        m for m in result["matched_patterns"]
        if m["pattern_id"].startswith("mpc_")
    ]
    assert len(mpc_patterns) == 0, f"PP-only recording should have no MPC patterns, got: {mpc_patterns}"


# ── Test 5: Layer health MPC flags ──────────────────────────────────────

def test_layer_health_mpc_flags(mpc_fallback_recording):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools', 'debug_visualizer'))
    from backend.layer_health import LayerHealthAnalyzer

    analyzer = LayerHealthAnalyzer(mpc_fallback_recording)
    result = analyzer.compute()
    # Check that fallback frames have mpc_fallback flag
    fallback_frame_idx = 150  # first fallback frame
    frame_data = result["frames"][fallback_frame_idx]
    assert "mpc_fallback" in frame_data["control_flags"], (
        f"Expected mpc_fallback flag at frame {fallback_frame_idx}, "
        f"got: {frame_data['control_flags']}"
    )


# ── Test 6: Layer health PP unchanged ───────────────────────────────────

def test_layer_health_pp_unchanged(pp_only_recording):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools', 'debug_visualizer'))
    from backend.layer_health import LayerHealthAnalyzer

    analyzer = LayerHealthAnalyzer(pp_only_recording)
    result = analyzer.compute()
    # PP-only recording should have no MPC flags
    for frame in result["frames"]:
        mpc_flags = [f for f in frame["control_flags"] if f.startswith("mpc_")]
        assert len(mpc_flags) == 0, (
            f"PP-only recording should have no MPC flags, got {mpc_flags} at frame {frame['frame_idx']}"
        )
