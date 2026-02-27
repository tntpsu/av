from pathlib import Path
import sys

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools" / "debug_visualizer"))

from backend.diagnostics import analyze_trajectory_vs_steering
from backend.triage_engine import TriageEngine
from backend.layer_health import LayerHealthAnalyzer, CTRL_JERK_ALERT


def _write_command_jerk_recording(path: Path) -> None:
    n = 40
    t = np.arange(n, dtype=np.float64) * 0.1

    # Alternate throttle to create large command-domain jerk spikes.
    throttle = np.zeros(n, dtype=np.float32)
    throttle[1::2] = 1.0
    brake = np.zeros(n, dtype=np.float32)

    with h5py.File(path, "w") as f:
        f.create_dataset("vehicle/timestamp", data=t)
        f.create_dataset("vehicle/speed", data=np.linspace(0.0, 5.0, n, dtype=np.float32))

        # Minimal trajectory/control signals needed by diagnostics path.
        f.create_dataset("trajectory/reference_point_x", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("trajectory/reference_point_y", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("trajectory/reference_point_heading", data=np.zeros(n, dtype=np.float32))

        f.create_dataset("control/timestamps", data=t)
        f.create_dataset("control/steering", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/lateral_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/heading_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/emergency_stop", data=np.zeros(n, dtype=np.int8))
        f.create_dataset("control/throttle", data=throttle)
        f.create_dataset("control/brake", data=brake)

        # Cap flags stay zero to ensure commanded jerk is not derived from these.
        f.create_dataset("control/longitudinal_jerk_capped", data=np.zeros(n, dtype=np.int8))
        f.create_dataset("control/longitudinal_accel_capped", data=np.zeros(n, dtype=np.int8))


def test_diagnostics_and_triage_use_command_domain_jerk(tmp_path: Path) -> None:
    recording = tmp_path / "commanded_jerk_semantics.h5"
    _write_command_jerk_recording(recording)

    diagnostics = analyze_trajectory_vs_steering(recording)
    comfort = diagnostics.get("comfort", {})

    assert float(comfort.get("commanded_jerk_p95", 0.0)) > CTRL_JERK_ALERT
    assert float(comfort.get("jerk_cap_rate_pct", 0.0)) == 0.0

    triage_metrics = TriageEngine(recording)._extract_aggregate_metrics()
    assert float(triage_metrics.get("commanded_jerk_p95", 0.0)) > CTRL_JERK_ALERT
    assert float(triage_metrics.get("longitudinal_jerk_cap_rate", 0.0)) == 0.0


def test_layer_health_flags_commanded_jerk_spike_without_cap_flags(tmp_path: Path) -> None:
    recording = tmp_path / "layer_health_jerk_signal.h5"
    _write_command_jerk_recording(recording)

    out = LayerHealthAnalyzer(recording).compute()
    frames = out.get("frames", [])

    assert frames
    assert any("jerk_spike" in (fr.get("control_flags") or []) for fr in frames)
