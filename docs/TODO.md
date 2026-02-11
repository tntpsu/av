# TODO: Near-Term (Phase 1 Completion)

## Phase 1: Finish Lane-Keeping + Longitudinal Fidelity
- **Goal:** lock stable lane keeping with acceptable comfort on r20/r30.
- **Test gate:** centered >= 70% on r20/r30, lateral_rmse <= 0.40m,
  accel_p95 <= 25 m/s^2, jerk_p95 <= 400 m/s^3, lat_jerk_p95 <= 5.0,
  emergency_stop_frame == 0.

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
