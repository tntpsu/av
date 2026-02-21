# TODO: Near-Term (Phase 1 Completion)

## Pure Pursuit Control (S1-M33) — Completed

- [x] Implement `pure_pursuit` control mode in `LateralController`
- [x] Add PP parameters (pp_feedback_gain, pp_min_lookahead, pp_ref_jump_clamp, pp_stale_decay)
- [x] Add PP telemetry to HDF5 and metadata dict
- [x] Wire PP config through `VehicleController` and `av_stack.py`
- [x] Write 10 unit tests for PP (geometry, symmetry, oscillation, stale, jump, metadata)
- [x] Update PhilViz v63 with control mode indicator and PP diagnostics
- [x] E2E validation: 2 runs on s-loop, PP completes full 40s with 0 oscillation
- [x] Revert counterproductive PID tuning, keep bug fixes

## PP Steering Pipeline Bypass (S1-M35) — Completed

- [x] Bypass PID-era jerk limiting, EMA smoothing, sign flip override, adaptive rate computation in PP mode
- [x] Extract turn feasibility governor computation for PP path (speed governor needs it)
- [x] Add `pp_max_steering_rate` param (simple fixed rate limit, default 0.4)
- [x] Set `pp_feedback_gain: 0.0` for pure geometric initial validation
- [x] Add `pp_pipeline_bypass_active` telemetry field to HDF5 + PhilViz v65
- [x] 66 unit tests passing (3 new PP pipeline, 10 existing PP, 53 PID/longitudinal)
- [x] E2E validation: F571 (vs F295 pre-bypass), 0% jerk/smoothing active, oscillation eliminated
- [x] Tuned lookahead: `reference_lookahead_scale_min: 0.45`, `reference_lookahead_tight_scale: 0.55`

## Trajectory Ref Tracking + Speed Tuning (S1-M36) — Completed

- [x] Increase `ref_x_rate_limit: 0.12→0.22` and scaling factors for S-turn transitions
- [x] Add drift-rate-based dynamic rate limit relaxation in trajectory smoothing
- [x] Lower `ref_x_rate_limit_curvature_min: 0.003→0.001` and increase scale max to 5.0
- [x] Reduce `curve_mode_speed_cap_mps: 8.0→7.5`
- [x] Increase `max_steering: 0.6→0.7` for more PP authority
- [x] E2E validation: Full 40s s-loop, 733 frames, 0 out-of-lane events
- [x] **Next:** Reduce lateral error RMSE — achieved via S1-M37 (0.55m → 0.32m)
- [x] **Next:** Evaluate `pp_feedback_gain` — set to 0.10 in S1-M37

## Dynamic Per-Radius Speed Control + PP Tuning (S1-M37) — Completed

- [x] Add `curvature_calibration_scale: 2.5` to comfort governor (compensates for pipeline under-measurement)
- [x] Lower `comfort_governor_max_lat_accel_g: 0.25→0.20` (physically meaningful passenger comfort)
- [x] Add curvature history tracking (max of last 5 frames) to prevent speed spikes during transitions
- [x] Remove ad-hoc curvature cap block from `av_stack.py` (duplicate of comfort governor)
- [x] Remove unused `curve_mode_speed_cap_mps` and associated dead variables/telemetry
- [x] Raise curvature bins `max_lateral_accel` to 3.0 m/s² (safety-only, comfort governor is binding)
- [x] Set `pp_feedback_gain: 0.0→0.10` for steady-state drift correction
- [x] 9 new unit tests (calibration scale, curvature history, per-radius speed), 124 total passing
- [x] 3x E2E validation: All 100% in lane, lat RMSE 0.32m (target <0.40m achieved)
- [x] **Next:** Steering jerk reduction — achieved via S1-M38 (max 93-123 → 47-65, P95 21-25)
- [ ] **Next:** Comfort tuning (longitudinal accel/jerk P95)
- [ ] **Next:** Rate limit sweep: `pp_max_steering_rate` at 0.25, 0.35, 0.45

## PP Steering Jerk Reduction (S1-M38) — Completed

- [x] Add `pp_max_steering_jerk: 30.0` parameter (per s²) to `LateralController` and `VehicleController`
- [x] Replace hard-clip rate limiter with jerk-limited rate ramp in PP bypass path
- [x] Add max_steering ceiling anticipation to prevent jerk from hard clip at boundary
- [x] Track `_pp_last_steering_rate` post-clip (decays on stale perception)
- [x] Wire parameter through `av_stack.py` config and `av_stack_config.yaml`
- [x] Add `pp_steering_jerk_limited` and `pp_effective_steering_rate` telemetry to HDF5 + PhilViz v68
- [x] 2 new unit tests (jerk bounded, rate ramp convergence), 126 total passing
- [x] 3x E2E validation: All 100% in lane, jerk max 47-65 (from 93-123), P95 21-25 (target <30), lat RMSE 0.34m (no regression)

## Stack Isolation TODOs (Priority Execution Plan)

This section is the active execution backlog for layer-by-layer validation.
Use this before any further control tuning.

### Stage 0: Baseline Protocol (Must pass before experiments)

- [ ] Lock standard scenario template for all comparisons:
  - Track: `tracks/s_loop.yml`
  - Duration: `40s`
  - Entry window: `25m -> 33m`
  - Analysis scope: `to-failure`
- [ ] Standardize A/B decision rule:
  - Use repeated A/B (`>=5` repeats/arm)
  - Decision from median/p25/p75, never single run
- [ ] Standardize required report fields:
  - first centerline-cross frame
  - first failure frame
  - time-to-failure
  - feasibility class
  - authority gap / transfer ratio / stale %

Acceptance test:
- [ ] One reference A/B report generated with the exact template above.

### Stage 1: Perception Layer Isolation

Available now:
- `tools/replay_perception.py`
- `tools/reprocess_recording.py` (legacy path)

TODOs:
- [ ] Define canonical "perception replay pack" output format for downstream use.
- [ ] Add scripted comparison output (replay vs original) focused on:
  - lane center drift
  - stale events + stale reasons
  - lane-side sign violations
- [ ] Add pass/fail gate for perception replay reproducibility.

Acceptance tests:
- [ ] For a reference recording, replay output reproduces key failure window signals.
- [ ] Perception regression test fails if stale% or lane-side violations exceed threshold.

### Stage 2: Trajectory Layer Isolation

Available now:
- PhilViz trajectory diagnostics
- `tools/analyze/analyze_curve_entry_feasibility.py`

TODOs:
- [ ] Implement trajectory-locked replay harness (placeholder:
      `tools/analyze/replay_trajectory_locked.py`).
- [ ] Add cross-lock sensitivity comparator (temporary standalone):
      `tools/analyze/compare_crosslock_sensitivity.py`
  - Inputs: self-lock and cross-lock summary JSON files
  - Outputs: amplification ratios, upstream/downstream sensitivity call
  - Purpose: quantify trajectory-reference sensitivity relative to replay baseline
- [ ] Add trajectory quality gate independent of controller tuning:
  - reference smoothness
  - GT-reference deviation
  - curve-entry consistency
- [ ] Add trajectory-only comparison mode in analysis output.

Acceptance tests:
- [ ] Same trajectory can be replayed across controller variants.
- [ ] Trajectory metrics remain stable when only controller params change.
- [ ] Cross-lock comparator identifies when cross-lock deltas exceed self-lock
      baseline by configured ratio thresholds.

### Stage 3: Control Layer Isolation

Available now:
- `tools/analyze/run_ab_batch.py`
- Steering limiter waterfall instrumentation
- GT follower baseline
- `tools/analyze/replay_control_locked.py` (implemented scaffold)

TODOs:
- [x] Implement control-only replay harness on fixed trajectory/timebase
      (`tools/analyze/replay_control_locked.py`, scaffold version).
- [x] Extend control-locked summary to include post-safety throttle/brake
      attribution and override counters.
- [ ] Add control stage-gates:
  - no regression in failure timing medians
  - no regression in stale% during entry
  - no regression in comfort caps
- [x] Tighten centerline failure onset to first intrusion in analyzer and visualizer diagnostics.
- [x] Rework dynamic comfort policy to use `g_lat` hard guard + smoothed `g_jerk` soft penalty telemetry.
- [x] Align downstream hard clip with comfort-aware dynamic authority (`dynamic_curve_hard_clip_boost_*`).
- [x] Validate hard-clip lift in canonical 25s sanity run before starting gain/cap A/B sweep.
- [x] Add control-lock sensitivity comparator for self/cross matrix reporting
      (`tools/analyze/compare_controllock_sensitivity.py`).
- [ ] Replace GT-dependent boundary emergency stop trigger with perception/trajectory
      boundary inference (keep left/right off-road symmetry without GT).
- [ ] Add automated "reject promotion" rule when authority metrics improve but
      failure medians worsen.

Acceptance tests:
- [ ] Control-only harness distinguishes limiter bottleneck from upstream errors.
- [ ] Batch runner outputs automatic pass/reject for control candidate.

### Stage 4: Integration + Timing Robustness

Current gap:
- acceptance-gated candidate promotion runs not yet automated in CI workflow.

TODOs:
- [x] Add deterministic latency injection profiles (0ms/50ms/100ms).
- [x] Add deterministic perception perturbation profiles (seeded).
- [x] Build integration stress runner
      (`tools/analyze/run_latency_noise_suite.py`).
- [x] Add cross-seed robustness gate before parameter promotion.
- [x] Add latency/noise stress promotion policy:
  - fail if worst_profile / baseline steering-mean ratio exceeds threshold
  - require explicit rationale for any waived profile

Acceptance tests:
- [ ] Candidate config passes baseline + latency/noise stress gates.
- [ ] Failure ordering remains explainable in diagnostics.

### Stage 5: Cross-Layer Counterfactuals

TODOs:
- [ ] Build layer-swap evaluator (placeholder:
- [x] Build layer-swap evaluator (placeholder:
      `tools/analyze/counterfactual_layer_swap.py`) supporting:
  - baseline perception + treatment control
  - treatment perception + baseline control
  - baseline trajectory + treatment control
- [x] Fold cross-lock sensitivity comparator into unified isolation reporting
      (do not leave as long-term standalone one-off).
- [x] Add "upstream/downstream attribution scorecard" to final report.

Acceptance tests:
- [x] Tool can isolate which layer contributes most to failure delta.
- [x] At least one historical regression is correctly re-attributed.

### Stage 6: PhilViz Upstream/Downstream Clarity

TODOs:
- [x] Add GT-vs-perception lateral error side-by-side at hotspots.
- [x] Add perception->trajectory attribution metrics.
- [x] Add failure-frame upstream state card.
- [x] Add causal event timeline view.
- [x] Add graphical steering waterfall chart.

Acceptance tests:
- [x] For each failed run, PhilViz can answer:
  - "What failed first?"
  - "What was downstream consequence?"
  - "Which single next lever is justified?"

### Promotion Policy (Always-on)

- [ ] No new parameter promoted without:
  - repeated A/B significance by robust stats
  - no regression on failure medians
  - documented upstream/downstream call
  - rollback condition written in report

## Phase 1: Finish Lane-Keeping + Longitudinal Fidelity
- **Goal:** lock stable lane keeping with acceptable comfort on r20/r30.
- **Test gate:** centered >= 70% on r20/r30, lateral_rmse <= 0.40m,
  accel_p95 <= 25 m/s^2, jerk_p95 <= 400 m/s^3, lat_jerk_p95 <= 5.0,
  emergency_stop_frame == 0.

## ✅ Recent Accomplishments (2026-02-10)

### Perception Stability Improvements
- **Fixed RANSAC bug**: Corrected xs/ys coordinate swap in polynomial fitting
- **Tuned RANSAC parameters**: threshold 10.0px, min_inliers 8 for robust outlier rejection
- **Increased perception smoothing**: lane_center_ema_alpha 0.15 to reduce drift impact
- **Tightened perception gating**: lane_center_gate_m 0.3m to catch smaller jumps
- **Added lane line change clamp**: 0.25m to prevent sudden lane position jumps

### Control Stability Improvements
- **Added lateral deadband (0.05m)**: Prevents steering when car is centered but perception drifts
- **Lowered feedforward threshold**: curve_feedforward_curvature_min 0.001 to activate with low curvature
- **Improved trajectory smoothing**: lane_position_smoothing_alpha 0.75 for smoother ref_x

### Results
- ✅ **Stable straight-line driving**: No oscillation on straights
- ✅ **Successfully reaches curves**: No overcorrection before curve entry
- ⚠️ **Remaining issue**: Curve following needs improvement (understeering in curves)

## Current Focus: S-Loop Turn Clearance (S1-M32)

### Completed (2026-02-20)
- [x] **A1:** Curvature guard on heading zeroing (trajectory_planner.py) — prevents false straight-road classification on curve approach
- [x] **A2:** Heading from lane-center history when lane_coeffs unavailable
- [x] **A3:** Curvature-aware smoothing alpha reduction + rate-limit relaxation
- [x] **A4:** Curvature preview at 1.5× lookahead for feedforward priming
- [x] **B1:** Curvature-based speed cap enabled (8.0 m/s default)
- [x] **B2:** Curvature preview → early deceleration (0.25g comfort limit)
- [x] **C1:** Control curvature_smoothing_alpha 0.7 → 0.5
- [x] **C2:** Curvature preview blended into feedforward (30% blend)
- [x] **C3:** Straight→curve integral decay (5-frame flush)
- [x] **V1–V5:** PhilViz auto-triage package (signal chain, suppression, speed-curvature, compare, scoring)
- [x] **D1:** Baseline signal chain diagnostic script

### Pending (Requires Live Unity)
- [ ] **D2:** A/B validation (3 repeats, canonical start, Mann-Whitney p<0.10)
- [ ] **D3:** Control fine-tuning if comfort/oscillation issues after D2
- [ ] **D4:** Full s-loop multi-turn validation + Stage 1 promotion gate

## Current Tuning Targets (Iterations)
- **Comfort caps:** accel_p95 <= 3.0 m/s^2, jerk_p95 <= 6.0 m/s^3.
- **Speed tracking:** planned vs actual RMSE <= 1.0 m/s, overspeed <= 10%.
- **Iteration budget:** 5 runs before structural changes.
- **Allowed levers:** longitudinal gains, slew/limits, planner feedforward.
- **Scenario:** s_loop.yml, 40s duration.

### Work items
- **Longitudinal dt:** use real dt in longitudinal control loop and log it.
- **Planner feedforward:** add planner accel feedforward term into throttle/brake.
- **Accel/jerk caps:** enforce explicit accel/jerk limits in longitudinal control.
- **Align limits:** make speed planner limits match controller caps; avoid double-smoothing.
- **Lateral comfort cap:** add lateral accel cap (v^2 * curvature) feeding into speed limits.
- **Verify:** run curve sweep (r20/r30/r40/r60) and confirm comfort metrics.

## Phase 2: Curvature Conditioning (Post Phase 1)
- **Start after:** Phase 1 gate is met.
- **Why:** reduce curvature spikes and stabilize comfort metrics.
- **Work items:**
  - Add curvature smoothing/preview window (10–20 m path window).
  - Use smoothed curvature for lateral jerk metrics and speed planning.
  - Validate with curve sweep (r20/r30/r40/r60) and report lateral jerk P95.

## Phase 3: Behavior + Lead Vehicle Integration (Post Phase 2)
- **Start after:** Phase 2 curvature smoothing is validated.
- **Why:** enable realistic car-following and future traffic features.
- **Work items:**
  - Add a behavior layer that outputs desired speed based on lead vehicle (IDM/ACC style).
  - Feed desired speed into the speed planner to generate jerk‑limited deceleration.
  - Add lead‑vehicle regression tests in `tools/analyze/` (simulated lead profile).
