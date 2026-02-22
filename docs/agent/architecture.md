# AV Stack — Agent Memory: Architecture

**Last updated:** 2026-02-21

---

## System Architecture Overview

```
Unity (C#)
  Camera frame (30 FPS) ──────► FastAPI Bridge (bridge/server.py)
  Vehicle state ─────────────►        │
  Ground truth path ─────────►        │ HTTP
                                       ▼
                               av_stack.py (orchestrator)
                                       │
                    ┌──────────────────┼───────────────────┐
                    ▼                  ▼                    ▼
             Perception         Trajectory            Control
          perception/           trajectory/          control/
          inference.py          inference.py      pid_controller.py
                    │                  │                    │
                    └──────────────────┴────────────────────┘
                                       │
                               HDF5 Recorder (async thread)
                               data/recorder.py
                                       │
                         data/recordings/*.h5
```

---

## Components

### 1. Bridge — `bridge/server.py`
- FastAPI async server; Unity speaks HTTP to it
- Endpoints: `POST /api/camera_frame`, `POST /api/vehicle_state`; returns `control_command`
- Maintains queues for front-center and topdown cameras
- Tracks telemetry: frame drops, decode latency, Unity time gaps
- CORS enabled for Unity WebGL; slow-request logging at 50ms threshold

### 2. Orchestrator — `av_stack.py` (5,420 lines — high complexity, read carefully before editing)
Handles all pipeline coordination:
- Polls bridge for latest frame + vehicle state
- Lane detection gating: EMA smoothing, jump clamping, outlier rejection
- `clamp_lane_center_and_width()`, `apply_lane_ema_gating()`, `blend_lane_pair_with_previous()`
- Calls perception → trajectory → control in sequence each frame
- Queues frames for async HDF5 write
- Manages safety states (emergency stop, stale perception decay)

### 3. Perception — `perception/inference.py` (489 lines)
**Primary:** Segmentation CNN (PyTorch, 320×640 input)
**Fallback:** CV pipeline (color masking → edges → Hough → RANSAC polynomial)
- Auto-falls back to CV if model untrained or confidence below threshold
- RANSAC params: residual_px=6.0, min_inliers=10, max_trials=40
- Output: lane polynomial coefficients + confidence score

### 4. Trajectory — `trajectory/inference.py` (1,143 lines)
Rule-based planner (no ML):
1. Lane coefficients → reference point (lane center midpoint)
2. EMA smoothing (α ≤ 0.85, reduced on curves to preserve intent)
3. Jump clamping (0.5m/frame), rate limiting (3× on sharp turns)
4. Heading zero gate (suppress heading error on straights, curvature < 0.012 rad⁻¹)
5. Speed planning: jerk-limited, curve-preview anticipatory decel (1.5× lookahead)
   - `v_max = √(a_lat / |κ|)` at 0.20g lateral limit
6. Output: (ref_x, ref_y, heading, target_speed)

Supporting modules:
- `trajectory/speed_planner.py` — jerk-limited speed profile (max_jerk: 0.7–4.0 m/s³)
- `trajectory/speed_governor.py` — curve-aware speed cap

### 5. Control — `control/pid_controller.py` (3,901 lines)
**Active mode: Pure Pursuit (default)**

Steering pipeline:
```
PP geometric computation
  → Rate limit (pp_max_steering_rate: 0.4)
  → Jerk limit (pp_max_steering_jerk: 30.0)
  → Sign flip check
  → Hard clip (max_steering: 0.7)
  → Smoothing
  → Drift correction (pp_feedback_gain: 0.10)
  → Stale decay (pp_stale_decay: 0.98)
```

Alternative modes (not currently active): PID, Stanley

**Longitudinal control:**
- Speed PID: kp=0.25, ki=0.02, kd=0.01
- Throttle rate limit: 0.04/frame; brake: 0.05/frame
- Jerk cap: 1.0–4.0 m/s³ (variable, distance-to-target based)
- 8-frame jerk cooldown post-cap
- Comfort gates: accel_p95 ≤ 3.0 m/s², jerk_p95 ≤ 6.0 m/s³

Alternative: `control/mpc_controller.py` (not currently used)

### 6. Recorder — `data/recorder.py`
- Async threaded HDF5 writes (30-frame buffer)
- Datasets: camera images (gzip), timestamps, vehicle state, perception, trajectory, control (250+ telemetry fields), ground truth
- Stall detection on write thread

### 7. Analysis Tools — `tools/analyze/` (50+ scripts)
Key tools:
- `analyze_drive_overall.py --latest` — primary summary report
- `analyze_recording_comprehensive.py` — detailed diagnostics
- `run_ab_batch.py` — A/B batch testing harness (5+ runs each)
- `counterfactual_layer_swap.py` — layer isolation (perception/trajectory/control)
- `replay_control_locked.py` / `replay_trajectory_locked.py` — replay with fixed layers
- `replay_perception.py` — perception-only replay
- `analyze_oscillation_root_cause.py` — steering stability diagnosis

### 8. PhilViz — `tools/debug_visualizer/`
- Flask server (port 5001)
- Frame-by-frame playback with lane overlay, trajectory, GT path
- Steering waterfall, signal chain diagnostics panel
- Polynomial inspector, re-run detection
- Export: PNG frames, JSON reports

---

## Data Flow (Normal Mode, per frame)

```
1. Unity captures camera frame (30 FPS)
2. Unity POSTs frame + vehicle_state to FastAPI bridge
3. av_stack.py fetches latest frame and state
4. Perception: image → lane polynomial coefficients
5. av_stack.py: EMA gating, jump clamping, outlier rejection on lane output
6. Trajectory: lanes → ref point (smoothed) + target speed
7. Control (Pure Pursuit): ref point → steering + throttle/brake
8. av_stack.py: rate/jerk limit, safety clip, comfort gate
9. Bridge returns control command to Unity
10. HDF5 recorder saves full frame (async)
```

---

## Configuration System

**File:** `config/av_stack_config.yaml` (~17 KB, 100+ parameters)

Key sections:
- `control.lateral.*` — steering mode, rate/jerk limits, gains (60+ params)
- `control.longitudinal.*` — speed PID, accel/jerk caps, slew rates (40+ params)
- `trajectory.*` — lookahead, smoothing alpha, heading gate threshold (20+ params)
- `safety.*` — max speed, lane width, emergency stop conditions
- `perception.*` — RANSAC params, confidence thresholds, fallback mode

Loading: auto from `config/av_stack_config.yaml` or `--config` CLI arg.

**Warning:** 100+ parameters creates a large tuning surface. Changes should use A/B batch testing (≥5 runs) for validation.

---

## Unity Bridge (C#)

Location: `unity/AVSimulation/Assets/Scripts/`
- Camera capture at 30 FPS → HTTP POST to bridge
- Vehicle state (pose, velocity, heading, curvature) → HTTP POST
- Ground truth path reporting (optional)
- Receives steering/throttle/brake → applies to vehicle physics

---

## Coordinate System

- Unity world space: X-right, Y-up, Z-forward
- Camera image: pixels, 480×640 (H×W)
- Perception output: polynomial coefficients in pixel space
- Trajectory ref point: meters in vehicle-relative frame
- Control: normalized steering [-1, 1], throttle/brake [0, 1]

---

## Test Suite

98 test files in `tests/`, 126+ passing tests:
- `test_control.py` (79 KB) — most comprehensive, covers all control modes
- `test_trajectory.py` — smoothing, heading gate, speed planning
- `test_perception_*.py` — CV pipeline, coordinate conversion
- `test_integration_scenarios.py` — end-to-end
- `test_config_alignment.py` — S1-M39 comfort gate validation

Run: `pytest tests/ -v`
