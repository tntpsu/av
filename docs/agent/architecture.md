# AV Stack — Agent Memory: Architecture

**Last updated:** 2026-03-16

---

## System Architecture Overview

```
Unity (C#)
  Camera frame (30 FPS) ──────► FastAPI Bridge (bridge/server.py)
  Vehicle state ─────────────►        │
  Ground truth signals ───────►        │ HTTP
                                        ▼
                                av_stack/ (orchestrator package)
                                av_stack/orchestrator.py
                                        │
             ┌──────────────────────────┼───────────────────────┐
             ▼                          ▼                        ▼
        Perception                 Trajectory                Control
     perception/                  trajectory/             control/
     inference.py                 inference.py         pid_controller.py
                                       │                        │
                                  Track YAML            control/
                                  (HD map)           mpc_controller.py
                                       │            control/regime_selector.py
                                       ▼
                               HDF5 Recorder (async thread)
                               data/recorder.py → data/recordings/*.h5
```

---

## Core Architectural Philosophy: Map + Perception

**The system uses both map geometry and real-time perception as complementary signals. This is intentional and correct.**

### Why maps are the right architecture

Track YAML files are HD maps — pre-built road geometry definitions, not requiring the car to drive the road first. This mirrors how production AV stacks work:

- **Waymo / Cruise / Mobileye:** HD map pre-built from survey vehicles, downloaded before every drive
- **This system:** Track YAML pre-built from simulation geometry, loaded at startup via `--track-yaml`

The map and perception serve different, non-overlapping roles:

| Signal source | Provides | Does NOT provide |
|---|---|---|
| **Map (track YAML)** | Curvature (exact), arc-distance to curves, speed cap geometry, MPC feedforward κ | Real-time vehicle position, lane center lateral offset |
| **Perception (segmentation)** | Lateral error, heading error, lane center position | Road geometry, future curvature, curve locations |

### Why perception alone cannot provide reliable curvature

Curvature estimation from camera images requires differentiating lane position twice over distance — a second-derivative operation that amplifies noise quadratically. At R15 (κ=0.067), this produces a 91.2% curvature undercall (P50 ratio 0.52) that no amount of model training eliminates. This is a fundamental geometric signal quality limit, not a training shortcoming.

**Corollary:** parameters that depend on curvature (speed cap, MPC feedforward, curve scheduler arming) should use map curvature as their primary source. Perception curvature is suitable only as a fallback when no map is loaded.

### What the map enables

- **Arc-distance curve scheduler:** arms turn-entry COMMIT phase at an exact geometry distance before the curve (10m normal, 12m tight), independent of sensor noise
- **Curvature preview for MPC:** `_map_curvature_at_distance()` samples signed κ from track profile at 1m intervals ahead — eliminates polynomial noise that caused oscillation in Phase 2.6
- **Speed cap geometry:** `v_max = √(g_cap × 9.8 / κ_map)` is exact at every point on the track
- **Track mismatch detection:** `map_health_ok` flag catches config/runtime mismatches

### Track YAML as map — practical workflow

The YAML is generated once from the simulation's known road geometry (not from driving). For new tracks:
1. Define geometry in `tracks/<name>.yml` (segments: straight/arc with radius, angle, direction, speed_limit)
2. Run with `--track-yaml tracks/<name>.yml` — the orchestrator auto-derives the track name and loads the profile
3. `reference_lookahead_entry_track_name` in overlay configs is explicit but redundant (auto-derived from `--track-yaml`)

---

## Components

### 1. Bridge — `bridge/server.py`
- FastAPI async server; Unity speaks HTTP to it
- Endpoints: `POST /api/camera_frame`, `POST /api/vehicle_state`; returns `control_command`
- Maintains queues for front-center and top-down cameras
- Tracks telemetry: frame drops, decode latency, Unity time gaps
- CORS enabled; slow-request logging at 50ms threshold

### 2. Orchestrator — `av_stack/` package (~5,724 lines total)

**`av_stack.py` is gone — replaced by the `av_stack/` package (S2-M3, 2026-02-24).**
`av_stack/__init__.py` re-exports all public names; 30+ consumers required no changes.

| Module | Contents |
|---|---|
| `av_stack/orchestrator.py` | `AVStack` class + 13 `_pf_*` sub-methods |
| `av_stack/lane_gating.py` | 9 EMA/jump/outlier lane gating functions |
| `av_stack/config.py` | `ControlConfig`, `TrajectoryConfig`, `SafetyConfig`, `load_config` |
| `av_stack/speed_helpers.py` | 8 speed/slew helper functions |

**`_pf_*` sub-method decomposition:**
- `_pf_validate_frame` — duplicate check, teleport guard, control_dt
- `_pf_run_perception` — lane detection, debug viz, segmentation mask
- `_pf_compute_lane_geometry` — speed/curvature/curve-phase/lookahead/speed-limits
- `_pf_apply_lane_gating` — jump detection, EMA/alpha-beta gating, instability validation
- `_pf_score_perception_health` — health scoring, stale fallback
- `_pf_build_perception_output` — PerceptionOutput dataclass construction
- `_pf_run_speed_governor` — path curvature, speed governor, horizon guardrail
- `_pf_plan_trajectory` — trajectory planning, reference point, oracle, vehicle-frame override
- `_pf_compute_steering` — controller.compute_control, speed prevention, launch ramp
- `_pf_apply_safety` — emergency stop latch, lateral error bounds, out-of-bounds
- `_pf_send_control` — bridge send + trajectory visualization
- `_pf_record` — HDF5 data recording
- `_pf_update_frame_state` — teleport countdown, position tracking, periodic logging

**Map integration in orchestrator:**
- `_load_reference_entry_track_profile()` — loads track YAML at startup, extracts curve list + total length
- `_map_curvature_at_distance()` — samples exact signed curvature at any arc distance ahead
- `_map_health_snapshot()` — validates map/runtime track match, lookup success rate, odometer jump rate
- Map odometer updated each frame from vehicle position; loop closure handled automatically

### 3. Perception — `perception/inference.py` (489 lines)
**Primary (active):** Segmentation CNN (PyTorch, 320×640 input) — `detection_method = "segmentation"` 100% of frames in validated runs. CLAUDE.md note "CV fallback active" is outdated.
**Fallback:** CV pipeline (color masking → edges → Hough → RANSAC polynomial) — only with `--use-cv` flag.

Output: lane polynomial coefficients + confidence score. Used for **lateral position** only. Curvature from lane polynomials is unreliable (see philosophy section above) — map curvature is the primary source.

### 4. Trajectory — `trajectory/inference.py` (1,143 lines)
Rule-based planner (no ML). Per-frame pipeline:
1. Lane coefficients → reference point (lane center midpoint)
2. EMA smoothing (α ≤ 0.85, reduced on curves)
3. Jump clamping (0.5m/frame), rate limiting (3× on sharp turns)
4. Heading zero gate (suppress on straights, κ < 0.012 rad⁻¹)
5. Speed planning: `v_max = √(a_lat / |κ_map|)` at configurable g-limit
6. Output: (ref_x, ref_y, heading, target_speed)

Supporting modules:
- `trajectory/speed_planner.py` — jerk-limited speed profile, cap-tracking, dip-frame fixes
- `trajectory/speed_governor.py` — curve-aware speed cap, cap-tracking, hard ceiling

### 5. Control — `control/pid_controller.py` (3,901 lines)

**Active lateral mode: Pure Pursuit (default)**

Steering pipeline:
```
Map feedforward (pp_map_ff_gain: 0.8 × κ_map × wheelbase)
  → PP geometric computation
  → Rate limit (pp_max_steering_rate: 0.4)
  → Jerk limit (pp_max_steering_jerk: 18.0)
  → Sign flip check
  → Hard clip (max_steering: 0.7)
  → Drift correction (pp_feedback_gain: 0.10)
  → Stale decay (pp_stale_decay: 0.98)
```

**MPC lateral mode (active when regime selector triggers):**
- `control/mpc_controller.py` — Linear MPC, OSQP QP solver, 3 classes: `MPCParams`, `MPCSolver`, `MPCController`
- `control/regime_selector.py` — PP↔MPC hysteresis: upshift at 11 m/s, downshift at 9.5 m/s
- MPC uses GT lane center (`groundTruthLaneCenterX`) for e_lat — not perception polynomials
- Enabled via `control.regime.enabled: true` in overlay configs (`mpc_sloop.yaml`, `mpc_highway.yaml`, `mpc_mixed.yaml`)

**Phase 2.8 MPC pipeline fixes (2026-03-12):**
- 2.8.1: Recovery mode suppression (skip ×1.2/×1.5 scaling when MPC active)
- 2.8.2: Steering feedback fix (`update_last_steering_norm` after all post-hoc modifications)
- 2.8.3: MPC-aware rate limiter (`mpc_max_steering_rate_per_frame: 0.15 rad/frame`)
- 2.8.4: Delay compensation (linear extrapolation: `predicted_e_lat = raw_e_lat + v × raw_e_heading × delay_frames × dt`)
- 10 new HDF5 diagnostic fields for MPC pipeline health

**MPC config key path (critical pitfall):**
Always use `self._full_config.get('trajectory', {}).get('mpc', {}).get('mpc_*', default)`. Top-level access silently returns default — caused a systematic 2pt regression in Phase 2.7.

**Longitudinal control:**
- Speed PID: kp=0.25, ki=0.02, kd=0.01
- Throttle rate limit: 0.04/frame; brake: 0.05/frame
- Jerk cap: 1.0–4.0 m/s³ (distance-to-target based)
- Cap-tracking: corrects under-speed during curve exits (`cap_tracking_enabled: true`)

### 6. Curve Phase Scheduler (trajectory-owned)

Controls lookahead contraction as the car approaches curves. Three phases:
- **FAR:** normal lookahead table, map-curvature preview only
- **ENTRY:** begins contracting lookahead at `curve_local_phase_distance_start_m` (10m normal, 12m tight) before curve
- **COMMIT:** aggressive error-severity-scaled contraction fires from `curve_local_commit_distance_ready_m` (8m)

**Mode:** `phase_active` (base default as of 2026-03-13, promoted from overlays)

**Why commit_distance_ready_m is track-specific (NOT in base):** 8.0m works on s_loop and mixed_radius because their approach straights are long enough that 8m is a small fraction of the approach. On hairpin (80m straights, back-to-back R20+R15 arcs with no gap between), 8m COMMIT + 10m ENTRY arm covers most of the short straight — the scheduler stays latched through the straight, producing the same premature-latch oscillation that Workstream C1 fixed on s_loop. Empirically validated 2026-03-13: promoting commit=8.0 to base caused hairpin to regress from 79→59/100 with an e-stop. Reverted to base=3.0; sloop and mixed_radius overlays retain 8.0 as track-specific overrides.

**Arc-distance gate:** `_reference_entry_track_profile` is auto-loaded from `--track-yaml`. When loaded, ENTRY/COMMIT fire at exact geometry-derived distances rather than perception-derived curvature estimates. This is the single most impactful robustness improvement for turn scheduling — eliminates the 16+ frame late turn-in observed on tracks without a loaded profile.

### 7. Configuration System

**Base:** `config/av_stack_config.yaml` — all defaults; all tracks inherit this

**Track overlays** (merge on top of base at runtime via `--config`):

| Overlay | Track | Key overrides |
|---|---|---|
| `config/mpc_sloop.yaml` | s_loop | MPC enabled, s_loop-specific MPC weights, map FF off |
| `config/mpc_highway.yaml` | highway_65 | MPC enabled, 15 m/s speed settings, r_steer_rate=2.0 |
| `config/mpc_mixed.yaml` | mixed_radius | MPC enabled, q_lat=0.5, r_steer_rate=2.0, g_cap=0.20 (keeps R40 in PP) |

**Overlays should only contain genuinely track-specific parameters:**
- Speed targets / max speed
- MPC weights (q_lat, r_steer_rate) tuned for the track's operating-point curvature range
- g-cap overrides for specific radius constraints (e.g., mixed_radius R40)

**Scheduling geometry (phase_distance, commit_distance, lookahead tables, scheduler mode) belongs in base** — these were validated identically across s_loop and mixed_radius and promoted to base on 2026-03-13.

**Config loading pitfall — HDF5 new field registration (6 locations required):**
(1) create_dataset, (2) list init, (3) per-frame append, (4) resize loop, (5) write, (6) ControlCommandData construction in orchestrator.py. Missing any location produces silent zeros.

### 8. Recorder — `data/recorder.py`
- Async threaded HDF5 writes (30-frame buffer)
- 250+ telemetry fields: camera images (gzip), timestamps, vehicle state, perception, trajectory, control, ground truth, MPC pipeline diagnostics
- Stall detection on write thread

### 9. Analysis Tools — `tools/analyze/`

| Tool | Purpose |
|---|---|
| `analyze_drive_overall.py --latest` | Primary 14-section summary report |
| `run_ab_batch.py` | A/B config comparison (≥5 runs each) |
| `mpc_pipeline_analysis.py --latest` | Phase 2.8 MPC pipeline health (regime, fix status, state quality, worst frames) |
| `counterfactual_layer_swap.py` | Layer isolation (perception/trajectory/control attribution) |
| `replay_perception_locked.py` | Replay with frozen perception — isolates trajectory/control |

### 10. PhilViz — `tools/debug_visualizer/`
- Flask server (port 5001), debug=False — **must restart after any backend edit**
- Tabs: Summary, Diagnostics, Triage, Issues, Layers, Compare, Chain, Gates, **MPC Pipeline** (added 2026-03-13)
- MPC Pipeline tab: regime distribution, 2.8.1-2.8.4 fix status, state quality cards, severe frame inspector
- `backend/mpc_pipeline.py` — returns `has_mpc: false` for PP-only recordings (safe for all tracks)

---

## Data Flow (per frame, normal mode)

```
1. Unity captures camera frame (30 FPS)
2. Unity POSTs frame + vehicle_state + GT signals to FastAPI bridge
3. Orchestrator fetches latest frame and state
4. _pf_run_perception: image → lane polynomial coefficients (segmentation CNN)
5. _pf_compute_lane_geometry: map curvature lookup + curve-phase scheduler update
6. _pf_apply_lane_gating: EMA smoothing, jump clamping, outlier rejection
7. _pf_run_speed_governor: v_max from map κ, cap-tracking
8. _pf_plan_trajectory: ref point (smoothed) + lookahead (phase-contracted)
9. _pf_compute_steering: PP or MPC → steering + throttle/brake
10. _pf_apply_safety: rate/jerk limit, emergency stop latch, lateral bounds
11. _pf_send_control: bridge returns command to Unity
12. _pf_record: HDF5 async write (250+ fields)
```

**Key signal sources per step:**
- Step 5 curvature: **map primary**, perception fallback
- Step 8 lookahead: phase-contracted via arc-distance gate (map-driven)
- Step 9 MPC e_lat: `groundTruthLaneCenterX` (GT lane center) — not perception polynomial

---

## Coordinate System

- Unity world space: X-right, Y-up, Z-forward
- Camera image: 480×640 pixels (H×W)
- Perception output: polynomial coefficients in pixel space
- Trajectory ref point: meters in vehicle-relative frame
- Control: normalized steering [-1, 1], throttle/brake [0, 1]
- GT lane center error: `groundTruthLaneCenterX` — vehicle-vs-lane in camera frame (+right = car right of center)

---

## Test Suite

**Current: 735 passing, ~10 skipped** (as of 2026-03-13)

Key test files:
- `tests/test_comfort_gate_replay.py` — 32 tests, two tiers (synthetic + golden recordings), all 5 tracks
- `tests/test_mpc_controller.py` — 14 tests: correctness, constraints, performance, fallback
- `tests/test_regime_selector.py` — 11 tests: hysteresis, blend ramp
- `tests/test_control.py` (79 KB) — comprehensive PP mode coverage
- `tests/test_config_integration.py` — config loading, overlay merging, field validation

Golden recordings (all 5 registered in `tests/fixtures/golden_recordings.json`):

| Track | Score | Notes |
|---|---|---|
| s_loop | 95.6 | PP only (4–7 m/s, below MPC threshold) |
| highway_65 | 96.2 | MPC active ~80% of frames |
| hairpin_15 | 79.0 | PP only; empirical R15 ceiling; gate set at 75 |
| sweeping_highway | 93.5 | MPC active ~75–77% |
| mixed_radius | 93.9 | PP↔MPC hybrid; q_lat=0.5 + r_steer_rate=2.0 |

---

## Known Pitfalls

1. **`--track-yaml` is mandatory.** Without it the vehicle falls through the road (315,000N impulse crash).
2. **MPC q_lat operating-point sensitivity:** q_lat=10.0 (base) works on highway (κ≈0.003) but causes hunting on moderate curves (κ=0.007). Highway and mixed_radius overlays use q_lat=0.5. s_loop MPC never activates so q_lat is irrelevant for s_loop.
3. **Regime mask in analysis scripts:** `mpc_mask = regime >= 0.5` (not `>= 1.5`). Regime encoding: 0=PP, 1=LINEAR_MPC, 2=NONLINEAR_MPC. Using `>= 1.5` silently drops all current MPC frames.
4. **Curvature undercall is fundamental, not a training bug.** Camera lane polynomials see κ≈0.52× actual on R15. This is geometric — second-derivative amplification of noise. Map curvature is the correct source; do not attempt to fix via retraining.
5. **Scoring cap mechanism:** `critical_cap=79.0` fires when Safety or Trajectory layer score is 60–79.9 (yellow). Cadence has zero weight in scoring. Score of exactly 79 is almost always a Trajectory-yellow cap, not cadence.
6. **Smith predictor ring buffer:** abandoned in Phase 2.8.4 — PP steering values at PP→MPC transitions corrupted predictions. Current implementation uses simple linear extrapolation. Ring buffer code kept for diagnostics only.
7. **PhilViz requires manual restart** after editing any backend file. Flask runs `debug=False` — no auto-reload.

---

## CI / Testing Infrastructure (T-033)

### Test Layers

| Layer | File | Tests | What it catches |
|---|---|---|---|
| Comfort gate pass/fail | `test_comfort_gate_replay.py` | 32 | Gate threshold violations (accel, jerk, lateral, e-stops) |
| Scoring regression | `test_scoring_regression.py` | 25 | Silent scoring drift from pipeline/config changes |
| Full suite | `pytest tests/ -v` | 774+ | All code-level regressions |

### Scoring Regression Pipeline

```
Golden HDF5 recordings (5 tracks, on-disk)
        │
        ▼
test_scoring_regression.py
  ├── Re-scores each via analyze_recording_summary()
  ├── Compares against tests/fixtures/scoring_baselines.json (frozen metrics)
  ├── Asserts: score tolerance, layer health, comfort gates, metric deltas
  └── Writes: data/reports/gates/latest_scoring_regression.json
        │
        ▼
conftest.py gate bundle writer
  ├── Reads latest_scoring_regression.json
  └── Embeds regression_deltas into gate_report.json
        │
        ▼
PhilViz Gates tab
  └── Renders color-coded regression delta table (green/yellow/red)
```

### Config Change Detector

`tools/ci/check_config_regression.py` — runs in CI on PRs (informational, not blocking).

Monitors: `config/av_stack_config.yaml`, `config/mpc_*.yaml`
Classifies params as scoring-critical (lateral control, speed governor, comfort, regime, curve scheduling) vs. other.

### CI Workflow (`.github/workflows/tests.yml`)

Steps in order:
1. Runbook sync check
2. `pytest tests/ -v` — full suite
3. `pytest tests/test_scoring_regression.py -v` — scoring regression (blocking)
4. Config change report (PR-only, informational)
5. Coverage report

### When to update baselines

After any intentional scoring formula change or new golden recording registration:
1. Update `tests/fixtures/scoring_baselines.json` with new metric values
2. Update `BASELINE_SCORES` in `tests/conftest.py` if overall scores changed
3. Run full suite to confirm green
