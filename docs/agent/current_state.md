# AV Stack — Agent Memory: Current State

**Last updated:** 2026-02-21
**Current milestone:** S1-M39 (Longitudinal comfort tuning — in validation)

---

## Active Development Focus

### Milestone S1-M39 — Longitudinal Comfort Tuning

**Goal:** Meet comfort gates for a 60-second s_loop run:
- Accel P95 ≤ 3.0 m/s²
- Jerk P95 ≤ 6.0 m/s³
- Max steering jerk reduced from 93–123 → 47–65 (achieved in M38)
- Emergency stops: 0 frames
- Lateral RMSE: ≤ 0.40m (currently ~0.32m ✓)
- Centered frames: ≥ 70% ✓

**Recent changes in M39:**
- Jerk cooldown: 8 frames post-jerk cap event
- Throttle rate limit: 0.04/frame; brake rate limit: 0.05/frame
- Jerk cap range: 1.0–4.0 m/s³ (variable, distance-to-target based)
- Speed planner: max_accel=2.5, max_decel=3.0, max_jerk=2.0
- Config test alignment: `test_config_alignment.py`

**Validation method:** 3× s_loop runs + `analyze_drive_overall.py --latest`

---

## Completed Milestones (Stage 1)

| Milestone | Description | Status |
|---|---|---|
| S1-M33 | Pure Pursuit control mode | ✓ Done |
| S1-M35 | PID pipeline bypass in PP mode | ✓ Done |
| S1-M36 | Trajectory ref tracking + speed tuning | ✓ Done |
| S1-M37 | Dynamic per-radius speed control | ✓ Done |
| S1-M38 | Steering jerk reduction (PP mode) | ✓ Done |
| S1-M39 | Longitudinal comfort tuning | ⚡ Validating |

---

## Incomplete / In-Progress Items

### From `docs/TODO.md` — Stack Isolation

These are the planned next major work items after S1-M39:

| Stage | Description | Status |
|---|---|---|
| Stage 1 | Stack isolation for perception layer replay | Planned |
| Stage 2 | Trajectory-locked replay harness | Planned |
| Stage 3 | Control-locked sensitivity analysis | Planned |
| Stage 4 | Deterministic latency injection profiles | Planned |
| Stage 5 | Cross-layer counterfactual analysis | Planned |
| Stage 6 | PhilViz upstream/downstream clarity improvements | Planned |

### Known Incomplete Items

1. **PhilViz (debug_visualizer):** Recent fixes in M39, but `CONSOLIDATION_PLAN.md` exists — suggests planned refactor
2. **`highway_65.yml` track:** New file (untracked in git) — likely a new test scenario not yet integrated
3. **`docs/README_ANALYSIS.md`:** New untracked file — new documentation, purpose unclear
4. **MPC controller (`control/mpc_controller.py`):** Implemented but not active; alternative to PP/PID/Stanley
5. **Segmentation model:** Training pipeline exists but model may not be trained (CV fallback is the de facto path)

---

## Active Configuration State

Primary config: `config/av_stack_config.yaml`

Key active values (S1-M39):
```yaml
control:
  lateral:
    control_mode: pure_pursuit
    pp_max_steering_rate: 0.4
    pp_max_steering_jerk: 30.0
    pp_feedback_gain: 0.10
    stale_decay: 0.98
  longitudinal:
    kp: 0.25 / ki: 0.02 / kd: 0.01
    max_jerk_min: 1.0 / max_jerk_max: 4.0
    jerk_cooldown_frames: 8
    throttle_rate_limit: 0.04
    brake_rate_limit: 0.05
    target_speed: 12.0 m/s
trajectory:
  reference_smoothing: 0.80
  dynamic_reference_lookahead: true
  lookahead_distance: 20.0 m
safety:
  max_speed: 12.0 m/s
```

---

## Git Status (at session start)

**Modified (staged/unstaged):**
- `CONFIG_GUIDE.md`
- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/TODO.md`
- `tools/debug_visualizer/README.md`
- `tools/debug_visualizer/data_loader.js`
- `tools/debug_visualizer/server.py`
- `tools/debug_visualizer/style.css`
- `tools/debug_visualizer/visualizer.js`

**Untracked (new files):**
- `docs/README_ANALYSIS.md`
- `tracks/highway_65.yml`

→ These changes likely belong to an in-progress commit for S1-M39.

---

## Known Risks / Fragile Areas

1. **`av_stack.py` complexity** — 5,420 lines with deeply layered lane gating logic. Changes here have wide blast radius.
2. **Curvature measurement quality** — Speed planning and steering are both curvature-sensitive. Errors cascade.
3. **Perception stale fallback** — Complex multi-condition blending when lane detection fails multiple consecutive frames.
4. **Temporal sync** — Camera frames and vehicle state can arrive with different timestamps; sync logic is critical.
5. **Parameter surface** — 100+ config params. Untested combinations can produce unstable behavior.
6. **Test suite vs. live behavior** — Tests are offline unit/integration tests; real Unity runs can surface issues not caught in tests.

---

## Diagnostic Workflow

Standard debugging cycle:
```
1. Run: ./start_av_stack.sh --launch-unity --unity-auto-play --duration 60
2. Analyze: python tools/analyze/analyze_drive_overall.py --latest
3. Inspect frames: python tools/debug_visualizer/server.py
4. Layer isolation: python tools/analyze/counterfactual_layer_swap.py
5. A/B test config change: python tools/analyze/run_ab_batch.py --runs 5
6. Update config, repeat
```
