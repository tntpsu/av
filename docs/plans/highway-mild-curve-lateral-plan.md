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
