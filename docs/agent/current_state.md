# AV Stack — Agent Memory: Current State

**Last updated:** 2026-02-23
**Current milestone:** S2-M2 (Config parameter documentation — complete)

---

## Active Development Focus

S2-M2 is complete. `docs/CONFIG_GUIDE.md` documents all 328 config parameters in three tiers
(primary levers / subsystem tuning / inactive legacy). Inline YAML comments added for 17
Tier 1 parameters. Reduces per-session tuning friction.
Next: S2-M3 (av_stack.py module decomposition) or T-012 (PhilViz CONSOLIDATION_PLAN).

---

## Completed Milestones (Stage 1)

| Milestone | Description | Status |
|---|---|---|
| S1-M33 | Pure Pursuit control mode | ✓ Done |
| S1-M35 | PID pipeline bypass in PP mode | ✓ Done |
| S1-M36 | Trajectory ref tracking + speed tuning | ✓ Done |
| S1-M37 | Dynamic per-radius speed control | ✓ Done |
| S1-M38 | Steering jerk reduction (PP mode) | ✓ Done |
| S1-M39 | Longitudinal comfort tuning | ✓ Done (2026-02-22) |

### S1-M39 Final Validation Results (2026-02-22)

| Track | Score | Lateral RMSE | Accel P95 | Cmd Jerk P95 | Steer Jerk | E-stops |
|---|---|---|---|---|---|---|
| s_loop (60s) | 95.6/100 | 0.203m ✓ | 1.32 m/s² ✓ | 1.44 m/s³ ✓ | 18.4 (at cap) ✓ | 0 ✓ |
| highway_65 (60s) | 96.2/100 | 0.044m ✓ | 1.08 m/s² ✓ | 1.44 m/s³ ✓ | 18.0 (at cap) ✓ | 0 ✓ |

All M39 comfort gates pass. Oscillation slope negative (stable) on both tracks.

### Key M39 Metric Fixes (committed 2026-02-22, `255dd3b`)

- **Gap-filtered `steering_jerk_max`**: Unity frame-drop artifacts (40–104 phantom
  values) eliminated via 1.5× median dt two-sided filter. Raw preserved as
  `steering_jerk_max_raw`.
- **Score recalibration**: jerk penalty zero at/below cap (18.0), thresholds
  raised to 20/22 so they only fire on genuine limiter failures.
- **`run_ab_batch` fixes**: correct flat summary dict lookup; added RMSE,
  osc_slope, steer_jerk_max, cmd_jerk_p95 columns; None-safe formatting.

---

## Incomplete / In-Progress Items

### Stack Isolation (S1-M40 — in progress)

| Stage | Description | Status |
|---|---|---|
| Stage 1 | Perception-locked replay (`replay_perception_locked.py`) | ✓ Done (2026-02-23) |
| Stage 2 | Trajectory-locked replay (`replay_trajectory_locked.py`) | ✓ Done (pre-existing) |
| Stage 3 | Control-locked sensitivity analysis (`replay_control_locked.py`) | ✓ Done (pre-existing) |
| Stage 4 | Deterministic latency injection | ✓ Done (built into trajectory-locked via `--lock-latency-ms`) |
| Stage 5 | Cross-layer counterfactual analysis (`counterfactual_layer_swap.py`) | ✓ Done (pre-existing, now includes Stage 1 matrix) |
| Stage 6 | PhilViz upstream/downstream clarity improvements | ✓ Done (2026-02-23) |

All three replay harnesses are now integrated into `counterfactual_layer_swap.py`.
The Stage-5 scorecard now attributes errors to perception, trajectory, or control.
PhilViz now has a **Chain** tab showing the per-frame causal chain (Perception → Trajectory → Control)
with stale-propagation banners and replay-mode locked-layer context.

### Perception-locked replay: key design notes

- Monkeypatches `self.perception.detect` to return locked 4-tuple `(lane_coeffs, confidence, detection_method, num_lanes_detected)`
- `lane_coeffs` reconstructed from `perception/lane_line_coefficients` vlen float32 HDF5 dataset (flattened degree-2 poly coeffs, 3 per lane)
- Single-lane cases disambiguated using `perception/left_lane_line_x` / `perception/right_lane_line_x` scalars
- EMA gating in av_stack still runs live — locks only the raw detection input
- Falls back to live perception if lock data unavailable for a frame

### Known Incomplete Items

1. **PhilViz (debug_visualizer):** `CONSOLIDATION_PLAN.md` exists — planned refactor (T-012)
2. **MPC controller (`control/mpc_controller.py`):** Implemented but not active; alternative to PP
3. **Segmentation model:** CV fallback is de facto perception path; model untrained
4. **Speed error RMSE on highway:** 1.19 m/s — structural (vehicle accelerates ~15s to reach 12 m/s target); not a control bug

---

## Active Configuration State

Primary config: `config/av_stack_config.yaml`

Key active values (post S1-M39):
```yaml
control:
  lateral:
    control_mode: pure_pursuit
    pp_max_steering_rate: 0.4
    pp_max_steering_jerk: 18.0     # confirmed working via gap-filtered metric
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

## Known Risks / Fragile Areas

1. **`av_stack.py` complexity** — 5,420 lines with deeply layered lane gating logic. Changes here have wide blast radius.
2. **Curvature measurement quality** — Speed planning and steering are both curvature-sensitive. Errors cascade.
3. **Perception stale fallback** — Complex multi-condition blending when lane detection fails multiple consecutive frames.
4. **Temporal sync** — Camera frames and vehicle state can arrive with different timestamps; sync logic is critical.
5. **Parameter surface** — 100+ config params. Untested combinations can produce unstable behavior.
6. **Test suite vs. live behavior** — Tests are offline unit/integration tests; real Unity runs can surface issues not caught in tests.
7. **Unity frame timing jitter** — dt std≈0.046s vs mean≈0.053s; double-ticks (0.067s) and long gaps (0.3s) occur regularly. Any metric using double-differentiation must filter gap frames.

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
