from pathlib import Path

import h5py
import numpy as np

from tools.debug_visualizer.backend.summary_analyzer import analyze_recording_summary


def test_oscillation_growth_metrics_detect_runaway(tmp_path: Path) -> None:
    recording_path = tmp_path / "summary_osc_growth.h5"
    n_frames = 300
    dt = 0.1
    t = np.arange(n_frames, dtype=np.float64) * dt

    # 0.5 Hz oscillation with linearly increasing amplitude.
    amp = 0.02 + (0.004 * t)
    lateral_error = amp * np.sin(2.0 * np.pi * 0.5 * t)

    with h5py.File(recording_path, "w") as f:
        f.create_dataset("vehicle/timestamps", data=t)
        f.create_dataset("vehicle/speed", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/steering", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/lateral_error", data=lateral_error.astype(np.float32))
        f.create_dataset("control/heading_error", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/total_error", data=lateral_error.astype(np.float32))
        f.create_dataset("control/pid_integral", data=np.zeros(n_frames, dtype=np.float32))
        f.create_dataset("control/emergency_stop", data=np.zeros(n_frames, dtype=np.int8))

    summary = analyze_recording_summary(recording_path, analyze_to_failure=False)
    smooth = summary["control_smoothness"]

    assert smooth["oscillation_zero_crossing_rate_hz"] > 0.3
    assert smooth["oscillation_rms_growth_slope_mps"] > 0.002
    assert smooth["oscillation_amplitude_runaway"] is True
