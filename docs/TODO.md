# TODO: Post Lane-Keeping Roadmap

## Phase 2: Curvature Conditioning
- **Start after:** lane keeping stability is acceptable (target: time_in_lane >= 95%, lateral_rmse <= 0.35m).
- **Why:** reduces curvature spikes that cause lateral jerk even with a good speed profile.
- **Work items:**
  - Add curvature smoothing/preview window (e.g., 10–20 m path window).
  - Use smoothed curvature for lateral jerk metrics and speed planning.
  - Validate with curve sweep (r20/r30/r40/r60) and report lateral jerk P95.

## Phase 3: Behavior + Lead Vehicle Integration
- **Start after:** Phase 1 speed planner is stable and Phase 2 curvature smoothing is in place.
- **Why:** future feature to handle slower lead vehicles requires behavior + planning integration.
- **Work items:**
  - Add a behavior layer that outputs desired speed based on lead vehicle (IDM/ACC style).
  - Feed desired speed into the speed planner to generate jerk‑limited deceleration.
  - Add lead‑vehicle regression tests in `tools/analyze/` (simulated lead profile).
