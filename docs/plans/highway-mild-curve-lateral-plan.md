# Highway Mild-Curve Lateral Plan

## Goal

Fix the highway lateral oscillation/swerving on gentle map-backed arcs in a way that is:

1. robust on `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`
2. diagnosable in PhilViz/CLI without manual reverse-engineering
3. transferable to other mild-curvature highway tracks, not just this one recording

This plan is based on:
- `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_183057.h5`

## Current Root Cause

The current run is clean enough to trust transport and perception:

- transport completeness/fallback: `100.0% / 0.0%`
- coherence pass: `100.0%`
- lane detection: `100.0%`
- stale perception: `0.0%`

The highway failure is now a lateral/reference contract problem:

1. the map/reference says a mild arc exists at about `kappa ~= 0.002`
2. highway curve recognition remains inactive:
   - `curve_intent_state = STRAIGHT`
   - `curve_local_state = STRAIGHT`
3. lookahead remains long:
   - `reference_lookahead_target` on high-error frames: about `14.4 m`
   - `pp_lookahead_distance` on high-error frames: about `9.4 m`
4. MPC is feasible and not clipped/rate-limited
5. steering remains too soft while the controller chases a far-ahead point
6. lateral error grows into an oscillatory chase

Important nuance:

- `control/lateral_error` gets large
- but `vehicle/road_frame_lane_center_offset` stays small, about `4-5 cm` p95 on high-error frames

That means the car is not physically near lane departure. The stack is mostly wrong about the reference geometry it should be tracking through the mild arc.

## Why Current Tooling Is Not Enough

Today the tools expose symptoms, but not the actual contract failure.

### What exists today

Generic signals exist in:
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/backend/triage_engine.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/backend/issue_detector.py`
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`

Examples:
- `ctrl_lateral_oscillation`
- `mpc_steering_oscillation`
- `curve late turn-in (lookahead floor rescue)`
- analyzer warnings like:
  - oscillation amplitude growth
  - curve intent late arm

### What is missing

Nothing today directly says:

1. mild highway arc is present in the reference
2. curve recognizer stayed `STRAIGHT`
3. lookahead stayed in straight/high-speed mode
4. actual road-frame lane offset stayed small
5. therefore the root cause is under-activated mild-curve reference shaping, not perception or transport

That missing attribution is the first thing this plan fixes.

## Success Criteria

Functional:

1. highway mild arcs with `reference_point_curvature ~= 0.002` are recognized as curves early enough
2. lookahead contracts for that regime before large oscillatory error appears
3. lateral RMSE/P95 materially improve on the same scenario
4. no regression in transport, safety, or ACC ownership

Attribution:

1. PhilViz and CLI can directly identify mild-curve under-activation
2. issue detector and triage surface a dedicated root-cause event, not just generic oscillation
3. actual lane offset vs controller lateral error is visible side-by-side

## Phase H1: Add Explicit Attribution First

### Why

If the next runtime change works, we need to know why. If it fails, we need to know whether it failed because:

1. mild-curve recognition still did not arm
2. recognition armed but lookahead stayed long
3. lookahead shortened but actual lane offset still oscillated

### Files

- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/server.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/backend/issue_detector.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/backend/triage_engine.py`

### New first-class metrics

Add per-frame/summary metrics for:

1. `mild_curve_present`
   - based on map/reference curvature band, not GT-only
   - target band should cover highway `~0.002`

2. `mild_curve_recognition_inactive`
   - true when:
     - mild curve present
     - `curve_intent_state == STRAIGHT`
     - `curve_local_state == STRAIGHT`

3. `reference_geometry_mismatch`
   - true when:
     - `|control/lateral_error|` high
     - `|vehicle/road_frame_lane_center_offset|` small

4. `mild_curve_long_lookahead`
   - true when:
     - mild curve present
     - lookahead remains in straight/high-speed band

5. `mild_curve_underactivated_tracking_window`
   - composite diagnostic:
     - mild curve present
     - recognizer inactive
     - long lookahead
     - high controller lateral error
     - small actual lane offset

### New PhilViz cards

Add:

1. `Highway Curve Contract`
   - mild-curve present rate
   - recognition-active rate
   - under-activation rate
   - long-lookahead-on-mild-curve rate

2. `Reference Geometry Mismatch`
   - `control/lateral_error` P50/P95
   - `vehicle/road_frame_lane_center_offset` P50/P95
   - mismatch rate

3. `Mild Curve Steering Response`
   - `reference_point_curvature`
   - `mpc_kappa_ref`
   - `pp_lookahead_distance`
   - steering command

### New timeline rows

Add rows for:

1. `mild_curve_present`
2. `mild_curve_recognition_inactive`
3. `reference_geometry_mismatch`
4. `mild_curve_long_lookahead`
5. `road_frame_lane_center_offset`
6. `control/lateral_error`

### New issue detector item

Add a dedicated issue:

- `highway_mild_curve_underactivation`

Condition:

1. mild curve present
2. `curve_intent_state` inactive
3. `curve_local_state` inactive
4. long lookahead retained
5. `|control/lateral_error|` exceeds threshold
6. `|vehicle/road_frame_lane_center_offset|` remains small

This should be a direct root-cause issue, not a derived summary.

### New triage pattern

Add to:
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/backend/triage_engine.py`

Pattern:
- `highway_mild_curve_underactivation`

Triage output should say plainly:

> Mild map-backed highway arc is present, but curve recognition stayed STRAIGHT and lookahead remained in straight mode. Large controller lateral error with small actual lane offset indicates a reference geometry mismatch, not transport or perception failure.

Config levers should point to:
- `/Users/philiptullai/Documents/Coding/av/config/av_stack_config.yaml`
- `/Users/philiptullai/Documents/Coding/av/config/mpc_highway.yaml`
- `/Users/philiptullai/Documents/Coding/av/config/acc_highway.yaml`

## Phase H2: Short-Lived Hypothesis Validation in Highway Overlays

### Why

This phase is not the final architecture.

It exists to answer one question quickly and cleanly:

> If the stack recognizes the mild highway arc earlier and shortens lookahead for that regime, does the swerving/oscillation collapse on the clean transport run?

The overlay changes in this phase are diagnostic scaffolding. They are allowed only as a temporary proof step before the behavior is lifted into dynamic core logic.

The current base thresholds are tuned above the actual highway arc in this run.

Observed in the recording:
- `reference_point_curvature ~= 0.002`

Current base thresholds still include:
- `curve_phase_preview_enter_threshold: 0.01`
- `curve_phase_preview_exit_threshold: 0.008`
- `curve_phase_preview_curvature_min: 0.0025`
- `curve_phase_path_curvature_min: 0.003`
- `curve_local_phase_path_curvature_min: 0.003`
- `curve_local_entry_severity_curvature_min: 0.003`
- `reference_lookahead_entry_preview_curvature_min: 0.003`

That is too high for this regime.

### Files

- `/Users/philiptullai/Documents/Coding/av/config/mpc_highway.yaml`
- `/Users/philiptullai/Documents/Coding/av/config/acc_highway.yaml`

### Required changes

Override highway-specific mild-curve thresholds so they cover the observed arc band.

Targets to override:

1. `curve_phase_preview_enter_threshold`
2. `curve_phase_preview_exit_threshold`
3. `curve_phase_preview_curvature_min`
4. `curve_phase_path_curvature_min`
5. `curve_local_phase_path_curvature_min`
6. `curve_local_entry_severity_curvature_min`
7. `reference_lookahead_entry_preview_curvature_min`
8. `local_curve_reference_curvature_min`

Also add temporary highway-overlay shaping for:

1. `reference_lookahead_speed_table_entry`
2. if needed, `reference_lookahead_speed_table`
3. if needed, mild-curve-specific owner scaling in the highway overlay

The key is not globally shortening lookahead. It is:

1. shortening only for the gentle arc regime
2. only when curve recognition is actually active
3. using the overlay only to validate the diagnosis on the existing run/scenario

### Acceptance gate for H2

On the rerun:

1. mild-curve-present frames should no longer have near-zero recognizer activation
2. `mild_curve_recognition_inactive` should materially drop
3. startup and transport remain clean

### Exit rule for H2

Do not stop at a passing overlay-tuned highway run.

If H2 fixes the swerving, the next phase is mandatory:
- move the logic into dynamic core behavior
- remove reliance on track-specific mild-curve thresholds as the real solution

## Phase H3: Lift the Fix Into Dynamic Core Logic

### Why

The final stack should not depend on `highway`-specific magic numbers for mild arcs.

The real fix should be transferable across tracks and should derive behavior from:

1. current speed
2. map-backed curvature ahead
3. distance/time to next curve
4. curve persistence / expected heading change
5. reference confidence and coherence

That means the overlay values from H2 are only allowed to serve as a proof step.

### Exact blocker isolated from the clean transport run

The blocker is not generic “highway thresholds too high.” The concrete failing gate is:

- `/Users/philiptullai/Documents/Coding/av/trajectory/utils.py`
  - `STRAIGHT -> ENTRY` requires `local_arm_phase_raw >= entry_on_effective`

On the bad windows from:
- `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_183057.h5`

the scheduler already knows the curve is locally relevant:

1. `curve_preview_far_phase = 1.0`
2. `curve_local_arm_ready = 1`
3. `curve_local_commit_ready = 1`
4. `curve_local_reentry_ready = 1` on most high-error frames
5. `distance_to_next_curve_start_m = 0.0` on the worst late windows

but it still stays `STRAIGHT` because:

1. `curve_local_arm_phase_raw p50 ~= 0.028`
2. `curve_local_entry_on_effective p50 ~= 0.349`

So the exact problem is:

1. the stack recognizes the curve as near
2. but the active arm-phase signal is built mostly from curvature-normalized terms only
3. `curve_local_phase_use_time_term` is off, so the saturated time/proximity signal does not help arm
4. the arm-phase never gets close to the `ENTRY` threshold
5. the scheduler never leaves `STRAIGHT`

That is the behavior H3 must fix.

### Files

- `/Users/philiptullai/Documents/Coding/av/config/av_stack_config.yaml`
- `/Users/philiptullai/Documents/Coding/av/trajectory/utils.py`
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/server.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`

### Required changes

Replace fixed mild-curve thresholds/lookahead behavior with dynamic regime logic.

### H3 execution order

1. Add blocker attribution first.
2. Change the scheduler arm-phase construction, not just overlay thresholds.
3. Re-run the same highway scenario.
4. Only after the scheduler activates correctly, adjust dynamic lookahead shaping.

This sequencing matters because the isolated blocker is upstream of lookahead ownership. If `curve_local_state` never enters `ENTRY`, no amount of downstream owner tuning will make the right path active.

### H3.0 Add first-class blocker attribution

Before changing runtime behavior, add direct blocker metrics so the next run can answer why activation did or did not happen.

Files:
- `/Users/philiptullai/Documents/Coding/av/trajectory/utils.py`
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
- `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
- `/Users/philiptullai/Documents/Coding/av/data/recorder.py`
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/server.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`

Required fields:

1. `curve_activation_blocker_mode`
   - examples:
     - `arm_phase_below_entry_threshold`
     - `reentry_not_ready`
     - `startup_lockout`
     - `force_straight`
     - `none`

2. `curve_local_arm_phase_deficit`
   - `entry_on_effective - local_arm_phase_raw`

3. `curve_local_arm_effect_score`
4. `curve_local_arm_effect_heading_term`
5. `curve_local_arm_effect_lateral_shift_term`
6. `curve_local_arm_effect_time_support_term`

Required summary/PhilViz items:

1. blocker mode distribution on high-error mild-curve frames
2. arm-phase raw vs effective entry threshold
3. effect-score components side-by-side with current curvature terms

The point is to make the next run say directly:
- “the curve was near, but arm-phase was 0.03 vs threshold 0.35”

not just:
- “recognizer stayed STRAIGHT”

#### H3.1 Dynamic mild-curve activation

Do not rely on fixed values like:

- `curve_phase_preview_curvature_min`
- `curve_phase_path_curvature_min`
- `curve_local_phase_path_curvature_min`
- `curve_local_entry_severity_curvature_min`

as the final owner of behavior.

Instead derive activation from a dynamic signal such as:

1. map/reference curvature magnitude
2. expected heading change over preview horizon
3. lateral displacement significance over preview horizon
4. distance/time to next curve
5. speed

Practical direction:

Build a new dynamic arm-effect score that can arm `ENTRY` for long-radius arcs when they are locally relevant.

Required runtime change:

1. keep the existing local readiness gates:
   - `curve_local_arm_ready`
   - `curve_local_commit_ready`
   - `curve_local_reentry_ready`

2. change the construction of `local_arm_phase_raw`
   - current code is effectively curvature-dominated
   - it must become an effect-based score for arming

Recommended arm-effect inputs:

1. map/reference curvature magnitude
2. expected heading change over the local arm horizon
3. expected lateral shift over the local arm horizon
4. time-to-curve support when the curve is near and locally ready
5. speed

Recommended derived terms:

1. `heading_change_est_rad`
   - example shape: `preview_curvature_abs * effective_distance_start_m`

2. `lateral_shift_est_m`
   - example shape: `0.5 * preview_curvature_abs * effective_distance_start_m^2`

3. `time_support_term`
   - only active when:
     - local readiness is already true
     - preview/path evidence is nonzero
   - this is not a blind "turn on time term globally" change

4. `arm_effect_score`
   - suggested shape:
     - `max(heading_term, lateral_shift_term, curvature_term)`
     - then blend/boost with time support only when locally ready

Critical guardrail:

1. use the dynamic arm-effect score only for arming `STRAIGHT -> ENTRY`
2. do not use it as the sustain term through straights
3. keep sustain anchored to actual local path/in-curve evidence so we do not recreate straight-latching on looped tracks

This is the core correction for the isolated blocker.

#### H3.2 Dynamic lookahead shaping

Replace the final reliance on fixed highway overlay tables with a dynamic function:

- `L = f(speed, curvature_severity, distance_to_curve, time_to_curve, curve_progress)`

Expected behavior:

1. straight road: long lookahead
2. mild highway arc at speed: moderate contraction
3. tighter curve: stronger contraction
4. exit: smooth unwind

But do this only after H3.1 is working.

Reason:

1. today the owner is already `phase_active`
2. but it cannot matter because `curve_local_state` never enters `ENTRY`
3. fixing lookahead shaping before activation is correct would be downstream tuning against the wrong owner state

Implementation target after H3.1 passes:

1. keep `reference_lookahead_owner_mode = phase_active`
2. make the entry contraction depend on:
   - dynamic arm-effect score
   - speed
   - distance/time to curve
   - curve progress
3. keep unwind smooth and progress-based

#### H3.3 Dynamic local severity

Entry severity should be based on:

1. curvature persistence over distance
2. previewed heading change
3. speed-scaled relevance

not only raw curvature threshold crossing.

Specific change:

1. today `curve_local_entry_severity` is too small on the mild arc
   - observed high-error p50: about `0.02`
2. severity needs to reflect curve effect, not only normalized raw curvature/rise

Recommended severity inputs:

1. heading-change estimate
2. lateral-shift estimate
3. preview/path curvature persistence
4. speed-scaled relevance

Severity should influence:

1. `curve_local_entry_on_effective`
2. local distance/time horizons
3. later lookahead shaping

But severity alone is not enough; H3.1 still has to change the arming signal itself.

#### H3.4 Keep the tooling aligned with the dynamic logic

The debug tools must show:

1. why the dynamic activator fired
2. what dynamic severity bucket/range applied
3. how that changed lookahead

so the new logic is directly attributable.

Add explicit timeline/debug values for:

1. `curve_local_arm_phase_raw`
2. `curve_local_entry_on_effective`
3. `curve_local_arm_phase_deficit`
4. `curve_local_arm_effect_score`
5. `curve_activation_blocker_mode`

That is the minimum needed to avoid another blind H2-style iteration.

### Acceptance gate for H3

The dynamic lift is complete only if:

1. the overlay values from H2 are no longer required for the behavior
2. mild-curve recognition activates on the highway scenario for the right reason
3. the dominant blocker on high-error mild-curve frames is no longer `arm_phase_below_entry_threshold`
4. `curve_local_state` materially enters `ENTRY`/`COMMIT` on the mild arc windows
5. long lookahead on high-error mild-curve frames materially drops
6. the same logic is defensible on other mild-curvature tracks
7. PhilViz/CLI can explain activation from dynamic inputs, not hidden per-track numbers

Quantitative target for the first H3 pass:

1. `curve_recognition_inactive_on_high_error_rate` should fall well below the current `100%`
2. `long_lookahead_on_high_error_rate` should fall well below the current `100%`
3. `reference_geometry_mismatch_on_high_error_rate` should materially drop from the current `100%`
4. transport must remain at the current clean state

## Phase H4: Revalidate and Reclassify the Swerving

### Live validation order

1. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`
2. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h8_curve_catchup.yml`

Reference recordings:
- `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_183057.h5`
- `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_094053.h5`

### What to measure

1. `control/lateral_error` RMSE/P95
2. `vehicle/road_frame_lane_center_offset` RMSE/P95
3. `mild_curve_underactivated_tracking_window` rate
4. `curve_intent_state` / `curve_local_state` activation on mild curves
5. `pp_lookahead_distance` on mild curves
6. oscillation zero-cross interval and RMS growth
7. whether the behavior still holds after removing temporary H2 overlay scaffolding
8. `curve_activation_blocker_mode` distribution
9. `curve_local_arm_phase_deficit` P50/P95 on mild-curve high-error windows

### Pass condition

We should consider the root issue fixed only if:

1. lateral error drops materially
2. road-frame actual lane offset stays low
3. mild-curve under-activation issue disappears or becomes rare
4. transport remains at current clean state
5. the behavior still holds once the solution is represented as dynamic core logic

## File-by-file implementation checklist for H3

### `/Users/philiptullai/Documents/Coding/av/trajectory/utils.py`

1. Add dynamic arm-effect helper(s).
2. Keep sustain logic conservative and path/in-curve based.
3. Add blocker classification and arm-phase deficit outputs.
4. Ensure `local_arm_phase_raw` can rise on mild high-speed arcs when locally ready.

### `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`

1. Propagate new blocker/effect fields into `reference_point` and `control_command`.
2. Do not add track-name branching.
3. Keep transport and ACC paths untouched.

### `/Users/philiptullai/Documents/Coding/av/config/av_stack_config.yaml`

1. Add any new generic dynamic-effect parameters.
2. Do not add highway-only permanent thresholds for this fix.

### `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`

1. Add blocker-mode rollups.
2. Add arm-phase-deficit summaries.
3. Update `highway_mild_curve_contract` to state whether activation failed because of arm-phase deficit.

### `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`

1. Print blocker attribution directly in the highway mild-curve section.
2. Stop recommending generic threshold tuning if the blocker is arm-phase deficit.

### `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`

1. Add an activation-blocker card/timeline.
2. Show arm-phase raw vs threshold visually.

## Tests for H3

1. Unit tests in `/Users/philiptullai/Documents/Coding/av/tests/` for:
   - dynamic arm-effect score increasing on mild high-speed arcs
   - `STRAIGHT -> ENTRY` when curve is near and effect is materially relevant
   - no false sustain on straight segments with far preview only

2. Summary/PhilViz contract tests for:
   - blocker mode
   - arm-phase deficit
   - effect-score reporting

3. Live validation on:
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`
   - then `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h8_curve_catchup.yml`

## Phase H4: Split Lookahead GT Geometry From At-Car MPC Cross-Track

### Why

The remaining lateral issue after H3 is a semantic contract bug, not another gain problem.

Current path:

1. Unity exports `groundTruthLaneCenterX` at lookahead distance.
2. `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py` treats it as current MPC cross-track.
3. `/Users/philiptullai/Documents/Coding/av/control/pid_controller.py` uses it as `e_lat`.

That makes MPC steer against future lane-center geometry instead of true at-car lane-center error.

### New contract

Split the telemetry into two explicit signals:

1. `groundTruthLaneCenterXLookahead`
   - selected lane center at the configured GT lookahead distance
   - debug/perception-alignment metric only

2. `groundTruthLaneCenterXAtCar`
   - selected lane center at the vehicle's current position in vehicle coordinates
   - authoritative MPC/Stanley cross-track source

Keep legacy `groundTruthLaneCenterX` as a backward-compatible alias for the lookahead value, but stop using it as the runtime control input.

### Files

- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/GroundTruthReporter.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/AVBridge.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CarController.cs`
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`
- `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
- `/Users/philiptullai/Documents/Coding/av/data/recorder.py`
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
- `/Users/philiptullai/Documents/Coding/av/control/pid_controller.py`
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/backend/issue_detector.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/backend/triage_engine.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`

### Runtime changes

1. Add a lane-aware at-car GT helper in Unity.
2. Export both lookahead and at-car selected-lane-center metrics.
3. Use the at-car metric for:
   - `reference_point['gt_cross_track_m']`
   - MPC `e_lat`
   - Stanley GT cross-track when GT is active
4. Preserve the lookahead metric separately for debug/perception evaluation.
5. Record which source the controller actually used:
   - `at_car`
   - `legacy_lookahead`
   - `perception_fallback`

### Recorder and analysis changes

Record:

1. `ground_truth/lane_center_x_at_car`
2. `control/mpc_gt_cross_track_at_car_m`
3. `control/mpc_gt_cross_track_lookahead_m`
4. `control/mpc_gt_cross_track_source_code`

Add a dedicated summary/issue path for:

- `mpc_gt_cross_track_semantic_mismatch`

It should trigger when:

1. `|control/mpc_gt_cross_track_lookahead_m|` is large
2. `|control/mpc_gt_cross_track_at_car_m|` is small
3. controller high-error windows overlap that mismatch
4. transport/perception remain clean

### Acceptance gates

On `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`:

1. `control/mpc_gt_cross_track_source_code` should be `at_car` on the active MPC path.
2. The semantic-mismatch issue should not fire.
3. High-error windows should no longer be dominated by large lookahead GT with small at-car GT.
4. Transport must remain:
   - completeness `100%`
   - fallback `0%`
5. If path-tracking score remains poor after the runtime fix, treat that as a separate scoring/metric contract cleanup, not as evidence the runtime fix failed.

## Tests

### Analyzer/debug tests

1. Add drive-summary contract tests for:
   - `mild_curve_present`
   - `reference_geometry_mismatch`
   - `mild_curve_underactivated_tracking_window`
   - dynamic activation reason / bucket summaries

2. Add issue detector tests for:
   - triggering `highway_mild_curve_underactivation`
   - not triggering it when actual lane offset is also large

3. Add triage tests ensuring the new pattern emits:
   - root cause
   - config levers
   - distinguishing rationale vs generic oscillation
   - explicit dynamic activator/long-lookahead explanation after H3

4. Add PhilViz/CLI parity tests for the new rates and modes.

### Runtime tests

1. Unit tests for dynamic mild-curve activation:
   - low-speed straight should stay inactive
   - high-speed mild arc at `kappa ~= 0.002` should activate
   - tighter curve should activate more strongly

2. Unit tests for dynamic lookahead shaping:
   - increasing speed on same mild arc should reduce lookahead relevance threshold
   - increasing curvature persistence should shorten lookahead
   - exit/unwind should lengthen lookahead smoothly

3. Contract tests for debug surfaces:
   - dynamic activation reason is recorded
   - dynamic lookahead regime is recorded

### Live analysis checks

After the first rerun, explicitly inspect:

1. frames where `|control/lateral_error| > 0.7`
2. overlap with:
   - `mild_curve_present`
   - `curve_intent_state != STRAIGHT`
   - `curve_local_state != STRAIGHT`
   - `vehicle/road_frame_lane_center_offset`

If the next run still shows:
- small actual lane offset
- large controller lateral error

then keep working on reference geometry/recognition.

If the next run instead shows:
- actual lane offset growing with the error

then the problem has transitioned into a true control authority/tuning problem and can be treated separately.

## Recommended execution order

1. Implement H1 first: attribution and triage
2. Implement H2: temporary overlay-based hypothesis validation
3. If H2 confirms the diagnosis, implement H3 immediately: dynamic core logic lift
4. Re-run the same scenario with the dynamic path
5. Only after that decide whether the remaining swerving is still a reference problem or now a true controller tuning problem

## Answer to the current tooling question

No. The current tools do not make this root cause obvious enough.

Today they tell you:

1. oscillation is growing
2. curve intent arms late
3. lookahead is long/oscillatory in generic terms

They do not directly tell you:

1. the arc is mild but real
2. the recognizer stayed inactive
3. the controller error is large while the actual lane offset stays small
4. therefore the failure is a mild-curve reference/activation bug, not perception or transport

That is why this plan includes the issue/triage/debug additions as required work, not optional cleanup.

## H5. Free-Flow Mild-Curve Authority At Full Highway Speed

### Why this is a separate phase

The wrong-target reject scenarios are now passing their ACC contracts, but they still expose
a separate free-flow lateral weakness at full highway speed:

1. `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260327_235457.h5`
2. `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260328_000128.h5`

Those runs are not ACC failures. They are mild-curve ENTRY runs at `~14-15 m/s` where:

1. `curve_local_state = ENTRY`
2. `curve_local_phase ~= 0.30`
3. `trajectory/reference_point_curvature ~= 0.002`
4. `control/mpc_kappa_ref ~= 0.0015`
5. actual lane offset stays small while controller lateral error grows

To isolate that cleanly, use a dedicated no-traffic reproducer:

- `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h11_freeflow_mild_curve.yml`

### H5 goal

Make the highway stack dynamically preserve enough map-backed curvature authority at full
free-flow speed on mild active curves without regressing slower ACC-following runs.

### H5 runtime direction

Do not add more track-specific overlays here.

Use a dynamic MPC authority-preservation rule keyed by:

1. current speed
2. active curve state (`ENTRY`/`COMMIT`)
3. local gate weight
4. actual map/reference curvature

Specifically:

1. tighten the existing active-curve bias-preservation guard as speed rises into the
   full highway band
2. preserve more of `kappa_ref` on mild active curves instead of allowing the bias path
   to soften it to the old `~0.75` ratio
3. record the effective preserve ratio and whether the dynamic path was active

### H5 telemetry

Record:

1. `control/mpc_kappa_active_curve_preserve_ratio`
2. `control/mpc_kappa_active_curve_preserve_active`
3. `control/mpc_kappa_active_curve_preserve_weight`

### H5 acceptance gates

On `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h11_freeflow_mild_curve.yml`:

1. trajectory score materially improves from the current baseline
2. high-error mild-curve windows materially decrease
3. `control/mpc_kappa_ref / trajectory/reference_point_curvature` rises above the current
   `~0.75` floor during those windows
4. no new limiter/fallback path becomes the dominant cause

Non-regression:

1. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`
2. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h8_curve_catchup.yml`

should remain clean on transport, ACC, and lane-keeping.

## H6. Early Curve Authority And Highway-Regime Consistency

### Root cause this phase resolves

H5 improved the dedicated free-flow reference run:

1. `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260328_091606.h5`

but it did not survive the non-regression sweep:

1. `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260328_091847.h5`
2. `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260328_092058.h5`
3. `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260328_092308.h5`

The dedicated root cause is:

1. H5 only improves active mild-curve bias preservation after local curve authority is already
   strong enough.
2. The dominant H8/H10 failures happen earlier, while the stack is still effectively straight or
   locally not ready.
3. In those windows:
   - `curve_local_state` is still `STRAIGHT` or `local_not_ready`
   - lookahead stays long
   - preserve weight is zero or very small
   - free-flow speed remains too high under `/Users/philiptullai/Documents/Coding/av/config/acc_highway.yaml`
4. So the H5 fix is out of play when the actual failure begins.

This phase must therefore resolve:

1. early curve authority before full local activation
2. `mpc_highway` vs `acc_highway` regime inconsistency
3. speed-feasibility / curve-cap coupling before the vehicle reaches the bad mild-curve window

### H6 design principles

1. Do not add another track-specific overlay hack.
2. Do not solve this by raising generic steering aggressiveness.
3. Do not solve this by further widening the active-curve preserve path alone.
4. Make the stack dynamically prepare for a mild highway curve earlier when speed and map
   curvature make the curve materially relevant.
5. Keep H9/H10 as wrong-target contract checks; they are not primary trajectory references.

### H6.0 Required attribution before more tuning

Add first-class attribution for the regime that H5 missed.

Record and surface:

1. `control/curve_preactivation_authority_weight`
2. `control/curve_preactivation_authority_active`
3. `control/curve_preactivation_blocker_mode`
4. `control/curve_preactivation_preview_weight`
5. `control/curve_preactivation_speed_weight`
6. `control/curve_preactivation_curvature_weight`
7. `control/curve_preactivation_distance_weight`
8. `control/curve_preactivation_kappa_floor`
9. `control/curve_preactivation_lookahead_target`
10. `control/curve_preactivation_speed_cap_target`

Add issue/triage:

1. `highway_preactivation_curve_authority_missing`
2. `highway_curve_entry_overspeed_before_local_ready`

The next rerun must tell you directly:

1. the curve was map-visible
2. local activation was not yet authoritative
3. pre-activation authority was or was not active
4. the active speed/lookahead targets were or were not reduced early enough

### H6.1 Unify the highway runtime regime first

Before more control tuning, remove the misleading split between:

1. `/Users/philiptullai/Documents/Coding/av/config/mpc_highway.yaml`
2. `/Users/philiptullai/Documents/Coding/av/config/acc_highway.yaml`

Current evidence shows H11 was validated under `mpc_highway`, while H8/H10 failed under
`acc_highway`.

Implementation requirements:

1. Make `acc_highway` inherit the same high-speed mild-curve authority behavior used by
   `mpc_highway`.
2. Remove any accidental dependence on different free-flow speed ownership for lateral
   validation.
3. Document which parameters are intentionally allowed to differ for ACC:
   - target speed
   - ACC settings
   - transport stack settings
4. Document which lateral/trajectory authority parameters must remain shared between
   `mpc_highway` and `acc_highway`.

Files to inspect/update:

1. `/Users/philiptullai/Documents/Coding/av/config/mpc_highway.yaml`
2. `/Users/philiptullai/Documents/Coding/av/config/acc_highway.yaml`
3. `/Users/philiptullai/Documents/Coding/av/config/av_stack_config.yaml`

Acceptance gate:

1. a clean H11/H10 comparison should no longer differ primarily because one config has the fix
   and the other does not

### H6.2 Add pre-activation curve authority

This is the main runtime change.

Add a bounded pre-activation authority path that can act before `curve_local_state` becomes
fully active.

The path should be driven by:

1. current speed
2. map/reference curvature magnitude
3. distance/time to the next curve start
4. preview relevance / far-preview phase
5. local readiness state

The path should be allowed to operate only when:

1. map curvature is coherent
2. curve is ahead and same-path relevant
3. speed is in the highway regime
4. transport/perception contracts are healthy
5. the vehicle is not already in a strong active local curve state

What it should control:

1. a bounded earlier `kappa_ref` preservation floor
2. a bounded earlier lookahead contraction target
3. a bounded earlier speed-cap target for mild curves

What it must not do:

1. pretend the curve is tighter than it is
2. reintroduce the old false-activation problem on straights
3. arm on noise or one-frame curvature spikes
4. override wrong-target or transport contracts

Likely implementation files:

1. `/Users/philiptullai/Documents/Coding/av/trajectory/utils.py`
2. `/Users/philiptullai/Documents/Coding/av/control/pid_controller.py`
3. `/Users/philiptullai/Documents/Coding/av/control/mpc_controller.py`
4. `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`

### H6.3 Tie pre-activation to speed feasibility, not just steering

H10 proved this is not only a curvature-ownership problem. It also becomes an entry-speed
problem before local activation is fully ready.

Implementation requirements:

1. add a mild-curve pre-cap that can act before full local activation when the map says the
   curve is materially relevant
2. ensure the pre-cap is bounded and releases cleanly once local curve authority takes over
3. keep the existing active curve-cap path authoritative once local phase is ready
4. record whether the active speed reduction came from:
   - pre-activation cap
   - active local curve cap
   - ACC
   - governor only

Files to inspect/update:

1. `/Users/philiptullai/Documents/Coding/av/trajectory/utils.py`
2. `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
3. `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
4. `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`

### H6.4 Validation order

Validate in this order.

1. H11 reference first
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h11_freeflow_mild_curve.yml`
   - goal: confirm the pre-activation path improves the dedicated mild-curve free-flow case

2. H10 reject-contract non-regression
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h10_oncoming_straight_reject.yml`
   - goal: `wrong_target_contract = PASS`
   - secondary goal: trajectory no longer fails because pre-activation authority/speed cap engages

3. H8 ACC-follow non-regression
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h8_curve_catchup.yml`
   - goal: preserve the H8 lead-continuity fix while removing the new lateral regression

4. H2 steady-follow guardrail
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`
   - goal: do not harm the clean highway baseline

Do not commit runtime changes until all four pass.

### H6.5 Acceptance gates

H11:

1. score remains at or above the new improved band
2. high-error mild-curve frames stay near zero
3. pre-activation authority activates before the old failure window

H10:

1. `wrong_target_contract = PASS`
2. no ACC contamination
3. no emergency stop from curve entry
4. no out-of-lane at the prior boundary-failure region

H8:

1. score returns to the clean band
2. ACC detection/continuity stays healthy
3. high-error mild-curve frames return close to the old clean reference

H2:

1. score stays near the current clean baseline
2. no new lateral oscillation growth beyond the existing acceptable band

### H6.6 Rollback rule

If H11 improves but either H8 or H10 remains in the `79` band, do not commit the runtime patch.

That means the change is still regime-specific rather than robust.

### Recommended execution order

1. implement H6.0 attribution first
2. unify the highway runtime regime in H6.1
3. implement bounded pre-activation authority in H6.2
4. tie that path into speed feasibility in H6.3
5. run the full H11/H10/H8/H2 validation order
6. only then decide whether the patch is promotable

## H7. Distractor-Sensitive Planner Divergence Backstop

### H7.0 Current authoritative root cause

After the H8 fix, the remaining highway blocker is now isolated to H10 on the current branch.

Authoritative current recordings:

1. H8 fixed:
   - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260328_172454.h5`
   - score `97.3`
2. H10 current-branch failure:
   - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260328_175543.h5`
   - score `79.0`
3. H10 stronger-authority falsification run:
   - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260328_180200.h5`
   - score `79.0`
   - reduced early error, but ended with late offroad emergency stop

What is now proven:

1. ACC wrong-target rejection is not the blocker.
   - `wrong_target_contract = PASS`
   - no association
   - no continuity hold
   - no ACC contamination
2. transport is not the blocker.
   - completeness `100%`
   - fallback `0%`
3. stale GT mismatch is not the blocker on the current H10 branch state.
   - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260328_175543.h5`
   - `MPC GT CROSS-TRACK CONTRACT: issue=NO`
4. the remaining failure is reference/ownership instability under a rejected distractor.

Concrete evidence from `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260328_175543.h5`:

1. high-error frames: `84`
2. first high-error frame: `487`
3. on those high-error frames:
   - `trajectory/reference_point_method = local_curve_reference` on `82/84`
   - `control/local_curve_reference_mode = bounded` on `82/84`
   - `control/local_curve_reference_shadow_promotion_reason = speed_curvature_raw_delta` on `82/84`
   - `control/local_curve_reference_blend_weight p50 ~= 0.35`
   - `control/local_curve_reference_raw_delta_m p50 ~= 1.27 m`
   - `control/local_curve_reference_capped_delta_m p50 ~= 0.44 m`

That means:

1. the local curve reference is already active
2. the stack is already trying to correct the planner
3. but the correction is still bounded too softly when planner-vs-local divergence is materially large

The failed stronger-authority experiment in
`/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260328_180200.h5`
proved the next important point:

1. simply raising bounded authority later is not robust
2. it reduced early high-error frames
3. but it ended with late offroad failure

So the robust fix must not be a blind larger blend floor.

### H7.1 Robust fix direction

The correct fix is a distractor-sensitive reference backstop, not a generic steering-aggression
increase and not a generic larger bounded-authority knob.

The design target:

1. detect when a rejected wrong-target actor is present
2. detect when the planner reference is diverging materially from the valid local map-backed arc
3. temporarily demote the distractor-sensitive planner path before the error is fully seeded
4. promote a guarded map-backed bounded local reference earlier and more authoritatively
5. unwind cleanly once divergence collapses

This must preserve:

1. H8 fixed behavior
2. H2 non-regression
3. H10 wrong-target reject contract
4. H11 free-flow lateral quality

### H7.2 New contract and attribution

Add a first-class issue:

1. `wrong_target_distractor_reference_divergence`

Trigger requirements:

1. wrong-target contract passes:
   - reject reason in `{wrong_lane, opposite_direction}`
   - no association
   - no ACC contamination
2. planner/local disagreement is materially large:
   - `local_curve_reference_raw_delta_m` above threshold
3. local curve geometry is valid:
   - mild active map curve
   - local curve reference valid
4. transport/perception freshness are clean enough
5. either:
   - planner reference is still primary in the onset window
   - or bounded local reference is active but still capped far below the raw delta

Files:

1. `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
2. `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
3. `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/backend/issue_detector.py`
4. `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/backend/triage_engine.py`

Telemetry to elevate:

1. `trajectory/reference_point_method`
2. `control/local_curve_reference_mode`
3. `control/local_curve_reference_shadow_promotion_reason`
4. `control/local_curve_reference_raw_delta_m`
5. `control/local_curve_reference_capped_delta_m`
6. ratio `capped_delta / raw_delta`
7. wrong-target reject reason and candidate presence

### H7.3 Runtime backstop design

Do not add another generic global authority knob.

Instead add a new bounded backstop mode, for example:

1. `distractor_guarded_bounded`

Behavior:

1. only available when:
   - wrong-target reject contract is active
   - candidate present is true
   - reject reason is `wrong_lane` or `opposite_direction`
   - local curve reference is valid
   - planner/local raw delta exceeds threshold
2. it promotes local reference earlier than the standard bounded shadow promotion
3. it can temporarily override the normal `local_curve_reference_max_blend` ceiling with a
   guarded, sustained authority floor
4. it must use hysteresis and minimum dwell so it does not chatter
5. it must unwind back to ordinary bounded/shadow behavior after divergence shrinks

This should live primarily in:

1. `/Users/philiptullai/Documents/Coding/av/trajectory/utils.py`
2. `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`

Core requirements:

1. sustained trigger, not one-frame trigger
2. reject-contract aware
3. no effect when no distractor is present
4. no effect on H8/H2 normal following
5. no full `active` local-arc ownership unless separately justified

### H7.4 Sustained trigger and unwind semantics

Add a small state machine for the backstop.

Suggested trigger inputs:

1. `wrong_target_candidate_present`
2. `wrong_target_reject_reason in {wrong_lane, opposite_direction}`
3. `local_curve_reference_valid`
4. `curve_local_state in {STRAIGHT, REARM, ENTRY}`
5. `local_curve_reference_raw_delta_m`
6. `current_speed_mps`
7. `preview/road curvature`

Suggested policy:

1. arm when the above conditions hold for `N` consecutive frames
2. enter guarded bounded mode with:
   - higher bounded blend floor than ordinary shadow promotion
   - optional lower lookahead ceiling in the same window
3. stay active for at least `M` frames
4. exit only after:
   - reject candidate gone or no longer wrong-target
   - and raw delta falls below exit threshold for `K` frames

This avoids:

1. one-frame snap decisions
2. the unsafe late over-correction seen in
   `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260328_180200.h5`

### H7.5 Bounded-authority policy, not blind larger blend

The failed experiment proved that a naive larger blend floor is not sufficient.

The robust bounded-authority policy should be:

1. onset authority:
   - stronger than current `0.35`
   - but only in distractor-guarded mode
2. progress-aware authority:
   - strongest near onset
   - fades as the planner/local delta shrinks
3. cap-aware authority:
   - allow higher bounded correction only when the cap/raw ratio is clearly too small
4. no late free escalation:
   - do not keep increasing authority deeper into the curve body without renewed evidence

That means the authority floor should depend on:

1. sustained raw delta
2. active reject contract
3. curve progress
4. delta collapse progress

Not just:

1. raw delta magnitude alone

### H7.6 Lookahead coordination

The backstop should coordinate with lookahead ownership.

If guarded bounded mode is active:

1. prevent straight-style long lookahead from reappearing in the onset window
2. keep the shorter local curve entry target coherent with the bounded local reference
3. do not let lookahead stay long while bounded local arc is trying to correct a large planner
   divergence

Files:

1. `/Users/philiptullai/Documents/Coding/av/trajectory/utils.py`
2. `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`

### H7.7 Implementation checklist

1. Add new contract/issue:
   - `wrong_target_distractor_reference_divergence`
2. Add new control telemetry:
   - guarded bounded active
   - guarded bounded reason
   - guarded bounded authority floor
   - guarded bounded dwell count
   - guarded bounded raw delta trigger
   - guarded bounded exit trigger
3. Implement sustained trigger in `/Users/philiptullai/Documents/Coding/av/trajectory/utils.py`
4. Wire the new reference mode into `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
5. Record new telemetry in:
   - `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
   - `/Users/philiptullai/Documents/Coding/av/data/recorder.py`
6. Update tooling surfaces
7. Add focused tests:
   - onset trigger
   - sustained dwell
   - hysteresis exit
   - no activation without reject candidate
   - no activation on normal H8/H2 cases

### H7.8 Validation order

Validate in this exact order.

1. H10 first
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h10_oncoming_straight_reject.yml`
   - goal:
     - wrong-target contract still `PASS`
     - score exits the `79` band
     - no late offroad failure
2. H8
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h8_curve_catchup.yml`
   - goal:
     - stay in the clean band
     - no continuity regression
3. H2
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`
   - goal:
     - no baseline regression
4. H11
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h11_freeflow_mild_curve.yml`
   - goal:
     - no free-flow lateral regression

### H7.9 Acceptance gates

H10:

1. wrong-target contract remains `PASS`
2. no ACC contamination
3. no emergency stop
4. high-error frames materially below the current `84`
5. no late offroad event

H8:

1. score remains in the clean band
2. no return of the old mild-curve underactivation failure

H2:

1. score remains near the clean baseline
2. no new oscillation growth

H11:

1. free-flow lateral score does not regress materially

### H7.10 Rollback rule

If H10 improves early but still ends in late offroad, do not commit.

That exact failure mode already happened in
`/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260328_180200.h5`,
and it is not acceptable to keep iterating authority increases without a sustained guarded
backstop design.

## H8. GT Cross-Track Semantic Cleanup And Revalidation

### H8.0 Current authoritative root cause

The current H10 blocker is not another bounded-authority tuning problem.

Current authoritative recordings:

1. H10 current repeat:
   - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260328_191720.h5`
   - score `79.0`
2. H8 clean:
   - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260328_190818.h5`
   - score `98.0`
3. H11 clean:
   - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260328_191338.h5`
   - score `96.9`

What is now proven:

1. wrong-target rejection is passing on H10
   - contract pass
   - no ACC contamination
   - no continuity hold
2. the new distractor guard is safe but dormant
   - `control/reference_distractor_guard_active = 0` through the repeat H10 runs
3. the remaining high H10 lateral error is being driven by the current GT cross-track signal
   - H10 high-error frames:
     - `control/mpc_gt_cross_track_at_car_m p50 ~= -1.09 m`
     - `control/mpc_gt_cross_track_lookahead_m p50 ~= -1.31 m`
     - `control/lateral_error p50 ~= 0.65 m`
4. the comparator we previously leaned on is not the same contract
   - `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/AVBridge.cs:1361`
   - `roadFrameLaneCenterOffset` is computed from:
     - `roadCenterAtLookahead - roadCenterAtCar`
     - projected into the road frame
   - so it is a lookahead road-center offset term, not an at-car selected-lane cross-track
5. the current MPC GT input is still semantically wrong for control
   - `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/GroundTruthReporter.cs:1041`
   - `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/GroundTruthReporter.cs:1516`
   - `GetLaneCenterAtCar()` is built from `GetLanePositionsVehicle()`
   - `GetLanePositionsVehicle()` projects lane geometry onto `carRight`
   - this means the exported “at-car lane center x” is a vehicle-frame projection, not a pure road-frame cross-track
6. LMPC then consumes that signal as `e_lat` while also consuming `e_heading`
   - `/Users/philiptullai/Documents/Coding/av/control/pid_controller.py:5615`
   - `raw_e_lat = -float(gt_cross_track)`
   - `raw_e_heading = float(gt_heading)`

So the current control path is mixing:

1. a heading-contaminated vehicle-frame lane-center signal as `e_lat`
2. a separate heading-error signal as `e_heading`

That is the real cross-turn root cause.

The remaining H10 `79.0` can no longer be treated as evidence for another authority increase until this
semantic contract is corrected.

### H8.1 Robust fix direction

The robust fix is to replace the current GT cross-track control contract with a true road-frame
selected-lane cross-track-at-car signal.

Do not:

1. keep tuning bounded authority around a contaminated `e_lat`
2. compare controller GT input to `roadFrameLaneCenterOffset`
3. keep using `groundTruthLaneCenterXAtCar` as the authoritative MPC cross-track state

Do:

1. export a true road-frame selected-lane cross-track at the car
2. keep the current vehicle-frame lane-center X only for debug/perception alignment
3. switch MPC and analyzer contracts to the new road-frame signal
4. then rerun H11/H2/H8/H9/H10 before any new lateral tuning

### H8.2 New ground-truth contract

Add two explicit GT families and stop mixing them:

1. debug vehicle-frame selected-lane-center geometry
   - keep:
     - `groundTruthLaneCenterXAtCar`
     - `groundTruthLaneCenterXLookahead`
   - mark them as debug/perception-alignment only
2. authoritative control GT in road frame
   - add:
     - `groundTruthSelectedLaneCrossTrackRoadFrameAtCar`
     - optional debug companion:
       - `groundTruthSelectedLaneCenterOffsetRoadFrame`
       - `groundTruthRoadFrameLateralOffsetAtCar`

Definition:

1. `road_frame_lateral_offset_at_car`
   - car lateral position from road center at the closest road point
2. `selected_lane_center_offset_from_road_center`
   - lane-center offset for `currentLane`
   - already derivable from road width and lane index
3. `selected_lane_cross_track_road_frame_at_car`
   - `road_frame_lateral_offset_at_car - selected_lane_center_offset_from_road_center`

That is the signal LMPC should consume as `e_lat`.

### H8.3 Unity producer changes

Files:

1. `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/GroundTruthReporter.cs`
2. `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/AVBridge.cs`
3. `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CarController.cs`

Implementation:

1. In `GroundTruthReporter`, add a new method that returns the road-frame selected-lane cross-track
   at the car.
2. Use:
   - closest-point road direction
   - car world position
   - road width
   - `currentLane`
3. Do not derive it by projecting a lane-center world point onto `carRight`.
4. Keep the current `GetLaneCenterAtCar()` path untouched for debug compatibility, but rename/comment it
   as vehicle-frame geometry, not control GT.
5. In `AVBridge`, export the new field explicitly into `currentState`.
6. In `CarController`, add matching state fields and serialization.

### H8.4 Orchestrator and controller changes

Files:

1. `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
2. `/Users/philiptullai/Documents/Coding/av/control/pid_controller.py`
3. `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
4. `/Users/philiptullai/Documents/Coding/av/data/recorder.py`

Implementation:

1. In orchestrator GT ingestion:
   - prefer the new road-frame selected-lane cross-track field
   - keep vehicle-frame fields separately for debug
2. In `pid_controller`, use the new road-frame field for:
   - `gt_cross_track`
   - `mpc_gt_cross_track_m`
   - `raw_e_lat`
3. Preserve:
   - `mpc_gt_cross_track_at_car_m`
   - `mpc_gt_cross_track_lookahead_m`
   - source-code telemetry
   but mark them as debug-only support metrics
4. Add explicit telemetry:
   - `mpc_gt_cross_track_road_frame_at_car_m`
   - `mpc_gt_cross_track_vehicle_frame_at_car_m`
   - `mpc_gt_cross_track_control_source_code`

### H8.5 Tooling and issue cleanup

Files:

1. `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
2. `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
3. `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/backend/issue_detector.py`
4. `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/backend/triage_engine.py`
5. `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`

Required changes:

1. stop using `vehicle/road_frame_lane_center_offset` as the primary comparator for MPC GT semantic checks
2. add a direct issue:
   - `mpc_gt_cross_track_vehicle_frame_semantic_mismatch`
3. trigger it when:
   - control is sourced from vehicle-frame GT
   - heading error is nontrivial
   - road-frame selected-lane cross-track stays materially smaller
4. update `mpc_gt_cross_track_contract` to report:
   - control GT source
   - road-frame selected-lane cross-track stats
   - vehicle-frame GT stats
   - absolute delta between them
5. relabel:
   - `vehicle/road_frame_lane_center_offset`
   as a lookahead road-center term, not selected-lane at-car error

### H8.6 Tests

Files:

1. `/Users/philiptullai/Documents/Coding/av/tests/test_control_command_hdf5.py`
2. `/Users/philiptullai/Documents/Coding/av/tests/test_drive_summary_contract.py`
3. `/Users/philiptullai/Documents/Coding/av/tests/test_issue_detector.py`
4. add or extend Unity-side regression coverage if available for GT export semantics

Required tests:

1. road-frame selected-lane cross-track equals zero when the car is centered in the selected lane
2. vehicle-frame lane-center X can diverge under heading error while the road-frame signal stays bounded
3. controller prefers the road-frame GT signal when available
4. analyzer flags vehicle-frame semantic mismatch only when the wrong source is still used
5. no recorder/schema regressions

### H8.7 Validation order

Validate in this order after the contract replacement:

1. H11
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h11_freeflow_mild_curve.yml`
   - goal:
     - no regression on the clean free-flow reference
2. H2
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`
   - goal:
     - no regression on the clean ACC baseline
3. H8
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h8_curve_catchup.yml`
   - goal:
     - keep the current clean band
4. H9
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h9_adjacent_lane_parallel_reject.yml`
   - goal:
     - wrong-target contract still `PASS`
5. H10
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h10_oncoming_straight_reject.yml`
   - goal:
     - wrong-target contract still `PASS`
     - GT semantic issue gone
     - determine whether the remaining `79.0` was a metric/control-contract artifact or a real
       active-state tracking problem

### H8.8 Acceptance gates

H11/H2/H8:

1. no material score regression
2. no new transport or GT semantic issues

H9/H10:

1. wrong-target contract remains `PASS`
2. no ACC contamination
3. new MPC GT contract shows road-frame source in control
4. vehicle-frame semantic mismatch issue is `NO`

Decision gate after H10:

1. if H10 score exits the `79` band:
   - the remaining issue was primarily GT control semantics
   - do not add new authority tuning
2. if H10 still scores `79` with the corrected road-frame control GT:
   - then the remaining issue is a real active-state lateral-control problem
   - only then return to authority/weighting work

### H8.9 Rollback rule

Do not keep any runtime patch that changes lateral authority if the controller is still consuming the
vehicle-frame GT cross-track signal as `e_lat`.

That would be tuning around the wrong state definition.

## H9. Post-H8 Root Cause And Robust Fix

Status after the fresh 2026-03-29 reruns:

1. The GT control-source semantic fix worked.
2. Wrong-target ACC rejection still passes.
3. The remaining `59`/`79` failures are now real reference-authority failures, not GT-source failures.

Evidence:

1. `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260329_111508.h5`
   - `WRONG-TARGET CONTRACT = PASS`
   - `MPC GT CROSS-TRACK CONTRACT = issue NO`
   - control source on high-error frames: `road_frame_at_car`
2. `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260329_111255.h5`
   - wrong-target contract still `PASS`
   - issue detector now flags repeated `wrong_target_distractor_reference_divergence`
3. `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260329_110822.h5`
   - same-family mild-curve authority failure exists without a wrong-target distractor

### H9.1 Dedicated root cause

The remaining failures share one control-layer weakness:

1. planner/local reference divergence becomes materially large
2. local curve reference is already active or promoted
3. bounded local-reference authority remains too weak relative to the raw delta
4. the controller stays on the wrong owner long enough to seed a real road-frame drift

Wrong-target traffic makes this worse in H9/H10:

1. ACC correctly rejects the actor
2. but the rejected actor still correlates with large planner/local delta
3. current bounded local-reference blend stays around `0.31-0.35` while raw delta reaches `~3.8m`
4. that is not enough authority to arrest the divergence

### H9.2 Debug tooling requirements

The tooling must make this obvious without manual HDF5 work.

Required surfaces:

1. issue detector:
   - `wrong_target_distractor_reference_divergence`
   - active when rejected actor is present, raw delta is large, and bounded local-reference blend is still too weak
2. analyzer/summary:
   - add a dedicated wrong-target reference-divergence contract
   - do not hide it under the mild-curve contract when the failure window is mostly straight/pre-entry
3. summary metrics:
   - `local_curve_reference_raw_delta_m`
   - `local_curve_reference_capped_delta_m`
   - `local_curve_reference_blend_weight`
   - `local_curve_reference_guarded_bounded_active`
   - `local_curve_reference_guarded_bounded_reason`
   - wrong-target reject mode on the same frames

### H9.3 Runtime fix direction

Do not add more generic steering authority first.

Implement a stateful reference-owner escalation:

1. introduce a `distractor_guarded_active_bounded` mode
2. arm it when all are true:
   - rejected wrong-target candidate present
   - planner/local raw delta above threshold
   - local curve reference valid
   - bounded local-reference blend still below the required floor
   - transport/perception remain clean
3. give it:
   - higher bounded blend floor than current shadow promotion
   - minimum dwell
   - hysteretic exit
4. keep it separate from:
   - ACC reject logic
   - generic mild-curve speed-cap logic

### H9.4 Validation order

1. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h9_adjacent_lane_parallel_reject.yml`
2. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h10_oncoming_straight_reject.yml`
3. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h8_curve_catchup.yml`
4. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`
5. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h11_freeflow_mild_curve.yml`

Acceptance gates:

1. H9/H10 wrong-target contract remains `PASS`
2. H9/H10 no longer show repeated `wrong_target_distractor_reference_divergence`
3. H8/H2/H11 do not regress
4. no late offroad introduced by stronger bounded authority

## H10. Oscillation Root Cause Split And Robust Resolution Plan

Status after the 2026-03-29 reruns:

1. We have been adding useful debug surfaces, but not enough to keep the failure families separate.
2. The real oscillation family is `H2` and `H8`.
3. `H9` and `H10` are still valuable, but they are reject-contract scenarios first and should not be the primary driver for the core oscillation fix.

### H10.1 Deep root cause

There are two different problems on the current branch, and they must stop being treated as one:

1. Core oscillation family: `H2` and `H8`
   - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260329_110822.h5`
   - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260329_111041.h5`
   - both are on mild curves with:
     - `curve_local_state = ENTRY`
     - `MPC GT CROSS-TRACK CONTRACT = issue NO`
     - transport clean
     - actual lane offset still small
   - but:
     - `active_state_tracking_failure_on_high_error_rate` stays high
     - `mpc_kappa_active_curve_preserve_weight p50 = 0.000`
     - `speed_governor_curve_cap_reason_mode_on_active_state_failure = commit`
     - `mpc_kappa_ratio_to_reference p50` remains too soft
   - this is the actual oscillation problem

2. Reject-contract family: `H9` and `H10`
   - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260329_111255.h5`
   - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260329_111508.h5`
   - wrong-target contract passes
   - ACC contamination is absent
   - these runs still expose real trajectory/reference issues, but they are not the right primary tuning anchor for the oscillation fix

The specific oscillation root cause in `H2/H8` is:

1. The active mild-curve preserve path is implemented inside the MPC bias-guard logic in `/Users/philiptullai/Documents/Coding/av/control/mpc_controller.py`.
2. That preserve path only gains authority when its internal dynamic gate becomes nonzero.
3. On the failing `H2/H8` runs, the effective ACC-follow speed is about `12.5-12.7 m/s`.
4. The current preserve speed gate in `/Users/philiptullai/Documents/Coding/av/control/mpc_controller.py` is:
   - `bias_active_curve_preserve_speed_on = 13.0`
   - `bias_active_curve_preserve_speed_full = 15.0`
5. So the main active-state oscillation family sits below the preserve-on speed threshold.
6. Result:
   - `curve_local_state` is already active
   - local curve reference is already active or promoted
   - but active mild-curve curvature preservation is still effectively off
   - controller keeps tracking a too-soft active reference and grows oscillation

This is the wrong-layer mistake we need to avoid:

1. We should not keep tuning around `H9/H10` reject-only scenarios as if they defined the core oscillation fix.
2. We should not keep using generic steering-authority or speed-cap changes to compensate for an inactive active-state preserve path.

### H10.2 What the tools already catch

Good:

1. transport and GT semantic failures are now easy to exclude
2. wrong-target reject contracts are explicit
3. issue detector can now surface:
   - `wrong_target_distractor_reference_divergence`

Insufficient:

1. analyzer/summary still over-compress `H2/H8` and `H9/H10` into one highway mild-curve bucket
2. tooling does not directly say why active preserve weight is zero
3. tooling does not classify scenario families before interpreting the trajectory issue
4. legacy `curve intent late arm` still shows up too prominently relative to the authoritative active-state issue

### H10.3 Tooling work required before more runtime tuning

Files:

1. `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
2. `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
3. `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/backend/issue_detector.py`
4. `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/backend/triage_engine.py`
5. `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`

Required changes:

1. Add scenario-family classification:
   - `freeflow_mild_curve`
   - `acc_follow_mild_curve`
   - `wrong_target_reject_only`
2. Add a first-class issue:
   - `active_mild_curve_preserve_speed_gated`
3. Trigger it when:
   - `curve_local_state in {ENTRY, COMMIT}`
   - high lateral error on mild curve
   - `mpc_kappa_active_curve_preserve_weight` is near zero
   - local curve reference is active or promoted
   - transport and GT contracts are clean
4. Record the gating reason explicitly:
   - `speed_below_on`
   - `curvature_below_on`
   - `gate_below_min`
   - `state_inactive`
   - `bias_path_inactive`
5. Split the highway mild-curve summary into:
   - `active_authority_issue`
   - `wrong_target_reference_divergence_issue`
6. Demote legacy `curve intent late arm` on runs where:
   - local curve state is already active on the dominant high-error frames

### H10.4 Robust runtime fix for the real oscillation family

Do not fix this with another generic steering or speed-cap tweak.

Implement a dedicated active-state mild-curve authority path that is not owned by the bias guard.

Files:

1. `/Users/philiptullai/Documents/Coding/av/control/mpc_controller.py`
2. `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
3. `/Users/philiptullai/Documents/Coding/av/control/pid_controller.py`
4. `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
5. `/Users/philiptullai/Documents/Coding/av/data/recorder.py`

Required behavior:

1. Introduce an active mild-curve authority weight that is independent of bias-sign logic.
2. Base it on:
   - current speed
   - mild-curve curvature magnitude
   - local curve state
   - local curve gate weight
   - local-reference validity
3. Allow it to activate in the ACC-follow regime, not only at `>= 13.0 m/s`.
4. Apply it to a direct `kappa_ref` floor or blend-floor path, not just to bias-correction clipping.
5. Keep it bounded:
   - mild curves only
   - active local states only
   - clean transport / GT contract only
   - no effect on straights

Concrete direction:

1. Create `active_mild_curve_authority_weight`
2. Compute it from dynamic effect, not a hard `13.0 -> 15.0 m/s` ramp alone
3. Use it to ensure `mpc_kappa_ref` cannot collapse below a stateful fraction of coherent road/reference curvature once `ENTRY` is active
4. Preserve a safe unwind on exit

### H10.5 Separate runtime fix for reject-only distractor scenarios

This remains necessary, but it is secondary to the core oscillation fix.

Files:

1. `/Users/philiptullai/Documents/Coding/av/trajectory/utils.py`
2. `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`

Required behavior:

1. keep `distractor_guarded_active_bounded`
2. only arm it when:
   - wrong-target reject contract passes
   - rejected candidate is present
   - planner/local raw delta is sustained and large
   - bounded local-reference authority is still too weak
3. do not let this become the main oscillation fix path for `H2/H8`

### H10.6 Validation order

Run in this order:

1. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`
2. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h8_curve_catchup.yml`
3. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h11_freeflow_mild_curve.yml`
4. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h9_adjacent_lane_parallel_reject.yml`
5. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h10_oncoming_straight_reject.yml`

Rationale:

1. prove the real oscillation fix first on `H2/H8`
2. protect the known-good free-flow case in `H11`
3. only then evaluate whether reject-only scenarios still need their separate guarded backstop

### H10.7 Acceptance gates

Core oscillation family:

1. `H2` score exits the `59` band
2. `H8` stays out of the `79` band
3. `active_mild_curve_preserve_speed_gated` goes to near zero on the dominant high-error windows
4. `mpc_kappa_active_curve_preserve_weight` or its replacement no longer stays at zero in active mild-curve windows
5. no GT or transport regressions

Reject-only family:

1. wrong-target contract remains `PASS`
2. no ACC contamination
3. if trajectory still fails there, it is classified under the dedicated wrong-target divergence contract, not the core oscillation contract

### H10.8 Rollback rule

Do not keep any new runtime patch that:

1. improves `H9/H10` only
2. leaves `H2/H8` in the same oscillation band
3. or regresses `H11`

That would be another wrong-layer fix.

### H10.9 Current validated status

Validated on current branch:

1. `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260329_121005.h5` (`H2`)
   - score `98.7`
   - scenario family `acc_follow_mild_curve`
   - no highway mild-curve contract issue
2. `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260329_121213.h5` (`H8`)
   - score `95.3`
   - scenario family `acc_follow_mild_curve`
   - remaining issue is active-state mild-curve authority, not stale legacy curve-intent or GT semantics
3. `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260329_121411.h5` (`H11`)
   - score `98.3`
   - scenario family `freeflow_mild_curve`
   - no highway mild-curve contract issue
4. `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260329_121605.h5` (`H9`)
   - score `59.0`
   - wrong-target contract `PASS`
   - dominant family is not mild-curve active-state authority
   - issue detector points to:
     - `trajectory_suppressed_curve_entry`
     - `emergency_stop`
     - `centerline_cross`
5. `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260329_121752.h5` (`H10`)
   - score `79.0`
   - wrong-target contract `PASS`
   - issue detector points to:
     - `wrong_target_distractor_reference_divergence`
     - `high_speed_mild_curve_active_authority_insufficient`

Implications:

1. The core oscillation fix is now validated on the true `H2/H8/H11` family.
2. `H9` is a separate straight/high-speed suppressed-entry problem.
3. `H10` is a separate reject-only mild-curve problem after ACC reject logic already passes.
4. Do not use `H9/H10` as the primary anchor for more `H2/H8` oscillation tuning.

### H10.10 Debug/triage gaps closed

Closed on this branch:

1. scenario-family split now classifies by scenario intent instead of incidental reject-mask overlap
2. stale legacy `Curve intent late arm` is suppressed when the authoritative owner path exists
3. false `mpc_gt_cross_track_absolute_coordinate_mismatch` issue emission is blocked when control already uses `road_frame_at_car`

Remaining tooling follow-up:

1. elevate `trajectory_suppressed_curve_entry` into the summary layer for `H9`
2. expose `wrong_target_distractor_reference_divergence` directly in summary/analyzer for `H10`, not only in issue detection

### H10.11 Lessons locked in

Do not repeat these mistakes on later tuning passes:

1. Split scenario families before tuning.
   - `H2/H8/H11` are the real mild-curve oscillation family.
   - `H9/H10` are wrong-target reject scenarios.
   - Do not mix them into one root-cause loop.
2. Validate semantic contracts before authority tuning.
   - ACC reject contract
   - GT control-source contract
   - ego-lane contract
3. Do not compare unlike signals in tooling.
   - at-car anchors must be compared to at-car anchors
   - lookahead/path reference must not be used as a proxy for at-car lane-center semantics
4. When runtime exports guard-native error signals, use those directly in the analyzer and issue detector.

### H11 Reject-Only Input Isolation

Root cause after the latest reject-only deep dive:

1. The output-side guard already saturates on `H9/H10`.
2. The remaining failure is upstream:
   - `lane_positions` center/width is still being perturbed by the rejected distractor
   - overriding only the final reference point does not fully isolate the trajectory stack

Implementation:

1. Add `reference_distractor_input_guard` in `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
2. Arm it only when:
   - wrong-target reject contract is live
   - current state is `STRAIGHT`/`REARM`
   - at-car ego-lane anchor is available
   - lane-position center/width error is sustained
3. When active:
   - replace planner/reference `lane_positions` with a synthetic ego-lane pair
   - suppress `lane_coeffs` for planning/reference generation
   - seed center-history to the synthetic ego-lane center so stale heading does not leak through
4. Export HDF5 telemetry for:
   - active state
   - reason
   - dwell
   - center/width error
   - trigger weight
   - synthetic lane pair
   - lane-coeff suppression
   - center-history seeding

Tooling:

1. Surface the new input guard in `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
2. Print it in `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
3. Keep wrong-target contract quality gated on:
   - divergence
   - straight drift
   - not merely on reject-pass

Validation order for this phase:

1. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h9_adjacent_lane_parallel_reject.yml`
2. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h10_oncoming_straight_reject.yml`
3. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`
4. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h8_curve_catchup.yml`
5. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h11_freeflow_mild_curve.yml`

Acceptance:

1. `H9/H10` wrong-target contract remains `PASS`
2. input-guard active rate is materially above zero on the reject-only onset windows
3. suppressed-lane-coeffs rate is materially above zero on those windows
4. `H2/H8/H11` do not regress

### Recovery baseline

Current branch state is not commit-safe.

1. Last clean overall commit:
   - `/Users/philiptullai/Documents/Coding/av` `3a4ff34`
   - this preserved the safe reporting split and did not change runtime behavior
2. Last committed runtime checkpoint:
   - `/Users/philiptullai/Documents/Coding/av` `19a1da1`
   - `3a4ff34` sits on top of it and only changes docs/reporting
3. Dirty worktree after `3a4ff34` contains mixed runtime experiments across:
   - active mild-curve authority
   - reject-only guarded reference behavior
   - GT / lane-anchor telemetry
   - planner-input distractor isolation
4. Recovery rule:
   - do not commit the current dirty bundle
   - preserve safe docs/tooling lessons separately
   - rebuild runtime fixes from the `3a4ff34` baseline with one scenario family at a time
