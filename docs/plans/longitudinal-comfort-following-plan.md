# Longitudinal Comfort And Following Plan

## Goal

Fix longitudinal reporting and gap-convergence logic without reintroducing unsafe ACC behavior.

This plan is driven by two reference runs:

- clean baseline:
  - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_230150.h5`
- failed catch-up experiment:
  - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_234730.h5`

The clean baseline is safe and high quality overall. The failed experiment proves that a blind positive-acceleration catch-up assist is the wrong fix.

## Executive Conclusion

The current `over_conservative_trailing` classification is not comparing the baseline run against the actual IDM equilibrium the controller is using.

Today we report convergence against:

- `vehicle/acc_target_gap_m = s0 + v*T`

But the controller behavior is governed by IDM:

- `a_idm = a_max * [1 - (v/v0)^4 - (s*/gap)^2]`
- `s* = s0 + max(0, v*T + v*Δv/(2*sqrt(a*b)))`

That means:

1. `acc_target_gap_m` is a useful diagnostic.
2. It is not the steady-state equilibrium gap when `v` is close to `v0`.
3. So the current reporting overstates “failure to converge.”

On the clean baseline, late steady-state frames show:

- actual gap p50:
  - `27.55 m`
- simple target gap `s0+vT` p50:
  - `20.11 m`
- actual minus simple target:
  - `+7.43 m`
- ego speed p50:
  - `12.04 m/s`
- final target speed p50:
  - `12.41 m/s`
- closing rate p50:
  - `0.02 m/s`

With those values, the IDM interaction term is still large enough that the controller is already near a deceleration equilibrium. So the clean run is not evidence that “the controller should obviously close 7 m more.”

The failed experiment confirms this:

- adding blind catch-up acceleration did not improve convergence
- it reduced robustness
- it produced:
  - detection dropouts
  - emergency brake events
  - collapsed-gap stop
  - off-road failure

So the next step is not another runtime catch-up tweak. It is a contract and attribution cleanup first, then an explicitly bounded acquisition-mode design if policy still needs to change.

## Current Evidence

### Clean baseline

Reference:
- `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_230150.h5`

What is true:

1. Safety is good.
- collisions: `0`
- near misses: `0`
- TTC p05: `26.58 s`
- minimum gap: `27.3 m`

2. ACC owns the longitudinal target cleanly.
- `vehicle/acc_target_speed_mps` and `control/target_speed_final` match in steady ACC_ACTIVE operation

3. Late steady-state closure is extremely small.
- late closing rate p50:
  - `0.02 m/s`
- late `acc_target_speed - ego_speed` p50:
  - `0.36 m/s`
- late `acc_target_speed - estimated_lead_speed` p50:
  - `0.38 m/s`

4. The currently reported “too conservative” gap is against the wrong semantic target.
- late actual gap minus `s0+vT` p50:
  - `+7.43 m`
- but this is not the same as “7.43 m above the IDM equilibrium gap”

### Failed catch-up experiment

Reference:
- `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_234730.h5`

What is true:

1. The additive catch-up assist was unsafe.
- score: `59.0`
- first `gap <= 10m`: frame `570`
- lead-collision override starts: frame `954`
- ACC states degraded into:
  - `DETECTION_LOSS`
  - `EMERGENCY_BRAKE`
  - `COLLAPSED_GAP_STOP`

2. It did not even solve the reported convergence problem.
- actual / target gap p50 got worse:
  - `41.59 / 21.63 m`
- `gap > target + 10m` got worse:
  - `65.3%`

3. Detection robustness degraded materially.
- radar detection rate:
  - `81.3%`
- max no-detect run:
  - `26` frames

So a blind additive acceleration bonus is ruled out.

## Root Cause

### RC1. Convergence reporting is using the wrong target semantic

Current reporting in:
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`

uses:
- `acc_target_gap_m = s0 + v*T`

as if it were the authoritative tracking target.

That is not correct for IDM steady-state behavior when:

1. speed is already close to free-flow target
2. interaction term is still non-trivial
3. closure reserve to the lead is small

This makes `over_conservative_trailing` too eager and can push the team toward unsafe fixes.

### RC2. ACC currently has no explicit, bounded acquisition mode

The controller has:

1. emergency / cutout / loss states
2. one generic `ACC_ACTIVE` state for all following

It does not distinguish:

1. far-gap acquisition
2. settled following

That means any attempt to “close faster” by modifying `ACC_ACTIVE` risks changing steady-state safety behavior and interacts badly with radar dropouts.

### RC3. We are missing the telemetry needed to tell whether we are:

1. tracking IDM equilibrium correctly
2. lead-speed limited
3. policy limited
4. detection limited

Right now we only have enough telemetry to argue about `s0+vT`, which is not sufficient.

## Required Design Direction

### Principle 1

Do not tune ACC convergence against `s0+vT` alone.

### Principle 2

Do not add positive acceleration directly into the generic `ACC_ACTIVE` IDM output.

### Principle 3

If we want faster long-gap closure, add a distinct and bounded acquisition policy with:

1. explicit entry/exit rules
2. explicit closure cap
3. explicit stability gates
4. explicit diagnostics

## Workstreams

### L4. ACC Gap Contract Correction

Goal:
- make reporting and debugging compare actual gap to the right longitudinal semantics

Files:
- `/Users/philiptullai/Documents/Coding/av/control/acc_controller.py`
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
- `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
- `/Users/philiptullai/Documents/Coding/av/data/recorder.py`
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`

Implementation:
1. Export first-class IDM telemetry:
- `acc_idm_dynamic_gap_m`
- `acc_idm_equilibrium_gap_m`
- `acc_idm_accel_mps2`
- `acc_lead_speed_estimate_mps`
- `acc_closure_reserve_mps`
- `acc_convergence_mode`

2. Redefine convergence reporting around three contracts:
- simple target gap:
  - `s0 + v*T`
- dynamic gap:
  - `s*`
- equilibrium-limited regime:
  - actual gap relative to the implied IDM steady-state envelope

3. Change summary classifications:
- `tracking_simple_gap`
- `tracking_dynamic_gap`
- `lead_limited_tracking`
- `policy_limited_tracking`
- `detection_limited_following`
- `compressed`

4. Stop promoting `over_conservative_trailing` as a top issue unless:
- detection is stable
- no recent loss/cutout
- actual gap stays materially above the corrected tracking envelope
- and the condition persists after the startup acquisition window

Acceptance:
- the clean baseline no longer gets a misleading top-level convergence failure
- the summary explains whether the gap is large because of policy, lead speed, or IDM equilibrium

### L5. Detection-Stability Contract For Following

Goal:
- make it obvious when radar instability invalidates convergence analysis

Files:
- `/Users/philiptullai/Documents/Coding/av/control/radar_sensor.py`
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
- `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
- `/Users/philiptullai/Documents/Coding/av/data/recorder.py`
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`

Implementation:
1. Record detection stability metrics:
- `acc_detection_stable_frames`
- `acc_recent_detection_loss`
- `acc_detection_loss_event_delta`
- `acc_no_detect_run_length`

2. Gate convergence attribution:
- if detection health is poor, classify as `detection_limited_following`
- do not recommend policy tuning from those windows

3. Add a dedicated summary block:
- `ACC Detection Contract`

Acceptance:
- the failed catch-up run is labeled detection-limited / invalid for convergence tuning

### L6. Explicit Far-Acquisition Mode Design

Goal:
- if policy change is still needed after L4-L5, add a separate acquisition mode rather than mutating generic IDM following

Files:
- `/Users/philiptullai/Documents/Coding/av/control/acc_controller.py`
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
- tests and tooling files above

Design:
1. Add explicit acquisition state or submode:
- `ACC_ACQUIRE`

2. Enter only when all are true:
- lead detected and stable for N frames
- no recent detection loss
- no recent emergency/collapsed-gap stop
- gap above corrected acquisition threshold
- TTC healthy with wide margin
- closure reserve below minimum desired reserve

3. Acquisition policy should be:
- lead-speed-referenced
- closure-capped
- bounded by free-flow target
- not an additive raw acceleration bonus

4. Example output policy:
- `target_speed = min(free_flow_target, lead_speed_estimate + closure_cap_mps)`
- where `closure_cap_mps` is modest and state-dependent

5. Exit when:
- gap reaches acquisition exit envelope
- detection becomes unstable
- closing rate exceeds cap
- any safety guard approaches warning band

Telemetry:
- `acc_acquire_active`
- `acc_acquire_reason`
- `acc_acquire_closure_cap_mps`
- `acc_acquire_exit_reason`

Acceptance:
- any new acquisition mode must preserve:
  - no collisions
  - no near misses
  - no lead-collision override
  - no detection-health regression
- and materially improve only the corrected convergence metric

### L7. Recommendation Cleanup

Goal:
- keep top-level recommendations aligned with the corrected contract

Files:
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`

Implementation:
1. Replace the current unconditional “ACC following too conservative” recommendation trigger.
2. Use corrected regime attribution from L4-L5.
3. Prefer specific wording:
- `lead-limited following`
- `policy-limited acquisition`
- `detection-limited following`

Acceptance:
- clean safe runs are not pushed toward unsafe “close faster” tuning unless the corrected contract proves policy under-closing

## Validation Order

1. Implement L4 only.
- no runtime ACC behavior change

2. Re-run:
- `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`

3. Confirm whether the clean baseline still shows a real convergence problem after corrected semantics.

4. Implement L5.
- make failed / unstable runs explicitly non-actionable for convergence tuning

5. Only if a real policy-limited gap remains, implement L6.

6. Then update L7 recommendation wording.

## Required Output After L4-L5

For any highway ACC run, the tools should answer directly:

1. Is the run safe behind the lead?
2. Is the large gap simply `s0+vT` vs IDM equilibrium semantics?
3. Is the car lead-speed limited?
4. Is the car policy limited?
5. Is the run invalid for convergence tuning because detection stability is bad?

## Immediate Recommendation

Do not make another runtime ACC convergence change before L4 and L5 are in place.

The current evidence supports:

1. reporting contract bug
2. missing acquisition-mode architecture
3. unsafe outcome from blind catch-up assist

It does not support another direct tuning pass inside the existing `ACC_ACTIVE` state.

### L8. Curve Catch-Up Lead-Continuity Contract

Goal:
- make ACC robust to same-lane lead tracking through curves without making opposite-lane or wrong-lane vehicles capturable

Reference failure:
- `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260327_095121.h5`

Observed failure chain:
1. H8 is still a real catch-up scenario:
   - lead starts `200 m` ahead at `10 m/s`
   - ego target is `15 m/s`
2. Radar detects the lead and gap closes correctly down to roughly `61 m`, `47 m`, `29 m`, `20 m`, `15 m`, then `~8 m`
3. During curve-entry / curved-road geometry, the forward radar repeatedly drops to zero for `10-28` frame runs
4. ACC exits to `FREE_FLOW` / `DETECTION_LOSS`
5. longitudinal ownership falls back to the speed governor
6. the lead reappears only briefly in `EMERGENCY_BRAKE`, then disappears again
7. first `lead_collision_override_active` appears at frame `715`

Primary root cause hypothesis:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/AVBridge.cs` uses a strict `±5°` forward-cone check
- on `R500` curves, same-lane lead azimuth can exceed that even when the lead is still the correct ACC target
- producer-side radar continuity is therefore too brittle for curve-following

Files:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/AVBridge.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/LeadVehicle.cs`
- `/Users/philiptullai/Documents/Coding/av/control/radar_sensor.py`
- `/Users/philiptullai/Documents/Coding/av/control/acc_controller.py`
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
- `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
- `/Users/philiptullai/Documents/Coding/av/data/recorder.py`
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/server.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`

Implementation:

#### L8.1 Producer-side radar-continuity attribution

Record why a lead was not returned on each frame, not just `detected=0`.

Add fields:
- `radar_fwd_candidate_present`
- `radar_fwd_reject_reason`
- `radar_fwd_target_azimuth_deg`
- `radar_fwd_target_heading_delta_deg`
- `radar_fwd_target_same_lane_confidence`
- `radar_fwd_target_lane_offset_m`
- `radar_fwd_target_arc_distance_m`

Required reject reasons:
- `out_of_range`
- `out_of_cone`
- `wrong_lane`
- `opposite_direction`
- `occluded_or_invalid`
- `no_candidate`

Acceptance:
- H8 can tell us whether the dropout is really `out_of_cone`
- we no longer have to infer producer failure from zeroed radar fields

#### L8.2 Same-lane target association contract

Do not let ACC depend on a pure forward-cone detector.

Producer-side target selection must prefer:
1. same-lane lead candidate
2. positive arc-distance ahead
3. similar heading / travel direction
4. bounded lateral offset relative to ego lane center

Reject:
1. opposite-lane traffic
2. oncoming traffic
3. adjacent-lane wrong target
4. geometrically visible but non-followable objects

Important design rule:
- do not fix H8 by blindly widening `radarHalfAngleDeg`
- widen coverage only if it is paired with same-lane / same-direction gating

Acceptance:
- same-lane curved lead remains capturable
- opposite-lane/oncoming candidate is explicitly rejected with `wrong_lane` or `opposite_direction`

Detailed implementation:
1. Keep the existing raw `±5°` cone as the strict radar acceptance cone.
2. Add a second, wider association envelope that is only used when all of these are true:
   - `same_lane_confidence >= assoc_same_lane_confidence_min`
   - `abs(target_heading_delta_deg) <= assoc_same_direction_heading_delta_max_deg`
   - `target_arc_distance_m > assoc_min_arc_ahead_m`
   - `target_arc_distance_m <= radarMaxRangeM + assoc_arc_slack_m`
   - `abs(target_azimuth_deg) <= assoc_half_angle_deg`
3. When the wider association envelope is used, mark the output source explicitly as `same_lane_association`.
4. Never use the wider association envelope for:
   - negative or near-zero arc distance
   - low same-lane confidence
   - large opposite-direction heading deltas
5. Keep all existing reject-reason telemetry and extend it with source/hold telemetry rather than replacing it.

Files:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/AVBridge.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CarController.cs`
- `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
- `/Users/philiptullai/Documents/Coding/av/data/recorder.py`
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_drive_summary_contract.py`

#### L8.3 Short-horizon lead continuity track

ACC should not drop straight to free-flow on every brief producer miss if continuity confidence is still high.

Add a small lead-continuity layer between raw radar return and ACC consumption:
- hold last accepted lead track for a short bounded horizon
- decay confidence with time and geometry error
- use predicted arc-distance / lead-speed estimate during the hold
- force exit immediately if same-lane confidence collapses

This is not multi-object fusion.
It is a bounded continuity bridge for the one authoritative same-lane lead.

Suggested output fields:
- `acc_lead_track_active`
- `acc_lead_track_source`
- `acc_lead_track_age_ms`
- `acc_lead_track_confidence`
- `acc_lead_track_hold_reason`
- `acc_lead_track_drop_reason`

Acceptance:
- brief H8 producer misses do not hand ownership back to free-flow immediately
- lead continuity does not persist through wrong-lane / opposite-direction evidence

Detailed implementation:
1. Keep one producer-owned forward-lead continuity track in Unity.
2. Seed or refresh the track only from:
   - `accepted_raw`
   - `same_lane_association`
3. Track state:
   - `active`
   - `source`
   - `age_ms`
   - `confidence`
   - `last_gap_m`
   - `last_range_rate_mps`
   - `last_same_lane_confidence`
   - `hold_reason`
   - `drop_reason`
4. Permit hold only when all of these remain true:
   - last accepted target was same-lane and same-direction
   - current candidate still exists
   - current candidate remains same-lane eligible
   - hold age stays under `continuity_hold_max_ms`
   - predicted gap remains positive and bounded
5. Force-drop immediately on:
   - `wrong_lane`
   - `opposite_direction`
   - `target_arc_distance_m <= 0`
   - confidence decay below `continuity_hold_min_confidence`
6. Surface held detections to Python as normal forward-radar detections, but with explicit source telemetry so summaries can distinguish:
   - raw detection
   - same-lane association
   - continuity hold
7. Do not move the continuity logic into ACC state handling first. Fix the producer contract before adding consumer-side state complexity.

New telemetry:
- `radar_fwd_track_active`
- `radar_fwd_track_source`
- `radar_fwd_track_age_ms`
- `radar_fwd_track_confidence`
- `radar_fwd_track_hold_reason`
- `radar_fwd_track_drop_reason`
- `radar_fwd_association_eligible`

Acceptance gates:
- H8 failure mode is no longer `out_of_cone_same_lane`
- H2 remains clean
- no held frame is labeled with `wrong_lane` or `opposite_direction`
- continuity-hold usage is bounded and attributable

#### L8.4 Consumer-state changes

Update ACC state semantics so temporary continuity holds are distinct from true no-lead loss.

Add or derive:
- `LEAD_TRACK_HOLD`
- `detection_limited_following`
- `association_limited_following`

Behavior:
1. raw no-detect with strong continuity confidence:
   - hold bounded lead track
   - do not immediately ramp to free-flow
2. raw no-detect with poor same-lane confidence or stale track:
   - enter `DETECTION_LOSS` / `FREE_FLOW` as today

Acceptance:
- H8 no longer oscillates between `ACC_ACTIVE` and `DETECTION_LOSS` on short curve misses
- true lead loss still exits quickly

#### L8.5 Tooling and triage

Add direct issue types:
- `lead_continuity_curve_dropout`
- `wrong_lane_target_rejected`
- `opposite_direction_target_rejected`
- `lead_track_hold_exhausted`

PhilViz/analyzer cards:
- `Lead Continuity Contract`
- `Lead Association Contract`

They should answer directly:
1. Did we have a real same-lane candidate?
2. Why was it rejected?
3. Did continuity hold bridge the miss?
4. Did ACC fall back because the lead was truly gone, or because producer association failed?

#### L8.6 Validation order

1. Reproduce H8 on current baseline:
- `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h8_curve_catchup.yml`

2. Add attribution only and confirm dominant reject reason.

3. Implement same-lane association without continuity hold.

4. Re-run H8.

5. Implement bounded continuity hold only if H8 still fails due to short producer misses.

6. Re-run H8 and then a same-lane steady baseline:
- `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`

7. Add explicit wrong-target tests before declaring success.

Concrete execution order:
1. Expand plan and implement telemetry fields first.
2. Implement same-lane association.
3. Rebuild Unity player and run:
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h8_curve_catchup.yml`
4. If H8 still shows `out_of_cone_same_lane`, enable bounded continuity hold.
5. Re-run:
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h8_curve_catchup.yml`
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`
6. Only after H8/H2 are stable, add dedicated wrong-target scenarios or synthetic tests.

#### L8.7 Required new scenarios

Add scenarios or validation cases for:
1. same-lane curve catch-up:
   - existing `H8`
2. opposite-lane oncoming pass on curve
3. opposite-lane oncoming pass on straight
4. wrong-lane same-direction vehicle if the track setup supports it

Promotion gate:
- fixing H8 must not make opposite-lane/oncoming vehicles capturable by ACC

#### L8.8 Current simulator capability gap

We do not fully have first-class oncoming-traffic support today.

What exists now:
1. `LeadVehicleConfig` in `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/TrackConfig.cs` supports:
   - `startDistanceM`
   - `speedProfileType`
   - `speedMps`
   - `laneOffsetM`
2. `LeadVehicle.Initialise(...)` in `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/LeadVehicle.cs` always configures `TrackWaypointFollower` to move in the track waypoint order.
3. `TrackWaypointFollower` in `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/TrackWaypointFollower.cs` always advances forward along the sampled path.

What is missing:
1. a scenario/config field for travel direction
2. reverse traversal of the track path
3. a first-class notion of when an oncoming actor should meet and pass the ego
4. validation semantics for "correctly rejected by ACC"

So:
1. we can place a vehicle in another lane with `lane_offset_m`
2. we cannot yet make that actor drive toward the ego on the same track with a supported scenario contract

#### L8.9 Simulator extension for opposite-lane/oncoming coverage

Goal:
- add first-class scenario support for opposite-direction actors so ACC wrong-target coverage is reproducible and attributable

Files:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/TrackConfig.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/TrackLoader.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/LeadVehicle.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/TrackWaypointFollower.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/AVBridge.cs`
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`

Implementation:

1. Extend lead-vehicle scenario config.
- Add fields to `LeadVehicleConfig`:
  - `travelDirection = "same"` where allowed values are `same | opposite`
  - `spawnArcDistanceM`
  - `despawnArcDistanceM` or `passUntilArcDistanceM`
  - optional `activationTimeS`
  - optional `targetPassArcDistanceM`
- Keep `startDistanceM` as a compatibility alias for same-direction cases.

2. Extend YAML parsing.
- In `TrackLoader.ParseLeadVehicleKeyValue(...)`, parse:
  - `travel_direction`
  - `spawn_arc_distance_m`
  - `despawn_arc_distance_m`
  - `activation_time_s`
  - `target_pass_arc_distance_m`
- Define compatibility behavior:
  - if `travel_direction` is omitted, default to `same`
  - if only `start_distance_m` is given, map it to the current same-direction path

3. Add reverse traversal to `TrackWaypointFollower`.
- Add an explicit direction multiplier or enum.
- Support:
  - forward arc advance
  - reverse arc advance
  - correct heading/orientation for reverse path traversal
  - correct sign convention for `laneOffsetM` relative to the actual travel direction
- `SetArcDistance(...)` should remain arc-based, but movement must be able to decrease arc distance when direction is `opposite`.

4. Make oncoming initialization deterministic.
- `LeadVehicle.Initialise(...)` should accept the new direction and spawn fields.
- For `travelDirection = opposite`:
  - spawn the actor at `spawnArcDistanceM` on the path
  - move it with reverse traversal
  - use `laneOffsetM` for the opposite lane relative to its travel direction
- Do not infer pass timing from random initial placement.

5. Add explicit pass timing semantics.
- Preferred contract:
  - `target_pass_arc_distance_m` means "the oncoming actor should cross the ego near this ego arc-distance"
- Implementation options:
  - compute actor spawn from ego start, actor speed, and target pass arc
  - or allow direct `spawn_arc_distance_m` and use `target_pass_arc_distance_m` only as a validation check
- Keep this deterministic; do not rely on approximate human tuning of distances.

6. Add producer telemetry for wrong-target scenarios.
- Extend `AVBridge` telemetry so wrong-target scenarios can be verified directly:
  - candidate direction class
  - association-eligibility outcome
  - reject reason
  - relative arc/ahead status
- Expected reject reasons in oncoming coverage:
  - `opposite_direction`
  - `wrong_lane`

7. Add scenario-level validation helpers.
- For each wrong-target scenario, the analysis should answer:
  - was a candidate present?
  - was it same-lane eligible?
  - was it correctly rejected?
  - did ACC ever enter following because of it?

#### L8.10 How to initialize oncoming traffic and guarantee the pass point

Use an explicit deterministic contract, not manual trial-and-error.

Recommended scenario contract:
1. ego starts at a known arc distance:
   - existing `start_distance`
2. oncoming actor declares:
   - `travel_direction: opposite`
   - `lane_offset_m`
   - `speed_mps`
   - `target_pass_arc_distance_m`
3. the loader/runtime computes `spawn_arc_distance_m` so the actor and ego arrive at the pass point at the same time

For example:
1. choose an ego start and speed profile
2. choose a pass point on the track, such as a point in the middle of a curve or on a straight
3. choose oncoming speed
4. solve:
   - ego travel time from ego start to pass point
   - oncoming travel time from actor spawn to pass point in reverse direction
5. initialize the actor so those times match

Acceptance:
1. pass-point error stays within a small tolerance, e.g. `< 5 m`
2. the actor remains in the opposite lane throughout the pass
3. ACC never accepts the target as followable
4. reject reason is stable and attributable

#### L8.11 Required wrong-target coverage set

Add these scenarios after the simulator extension lands:

1. `highway_oncoming_straight_reject`
- same road
- opposite lane
- opposite direction
- pass point on a straight
- expected:
  - candidate may be visible
  - reject mode `opposite_direction` or `wrong_lane`
  - ACC never enters following

2. `highway_oncoming_curve_reject`
- same road
- opposite lane
- opposite direction
- pass point on an `R500` curve
- expected:
  - same-lane association remains disabled
  - continuity hold remains disabled
  - ACC never enters following

3. `highway_adjacent_lane_same_direction_reject`
- adjacent lane
- same direction
- nearby enough to be geometrically tempting
- expected:
  - reject mode `wrong_lane`
  - no continuity hold
  - no ACC following

4. keep `highway_h8_curve_catchup.yml` as the positive control
- same-lane
- same direction
- curve catch-up
- expected:
  - association may activate
  - ACC stays active
  - no collision override

#### L8.12 Validation order for wrong-target robustness

1. Implement config + reverse traversal only.
2. Add one straight oncoming scenario and verify the actor actually passes the ego at the planned arc point.
3. Add telemetry assertions before ACC assertions.
4. Run:
   - positive control `H8`
   - oncoming straight reject
   - oncoming curve reject
   - adjacent-lane same-direction reject
5. Only pass the phase if:
   - `H8` still follows correctly
   - oncoming and wrong-lane scenarios are explicitly rejected
   - no reject scenario ever produces `ACC_ACTIVE` because of the wrong target

#### L8.13 Interim adjacent-lane same-direction coverage

This is the right immediate path before mixed traffic.

Reason:
1. the current simulator is still single-actor
2. adjacent-lane same-direction coverage does not require reverse traversal
3. it tests the same wrong-target association contract we care about
4. it isolates lane-association mistakes without adding multi-actor ambiguity

What we can support with the current actor model:
1. one vehicle in the adjacent lane
2. moving in the same waypoint direction as ego
3. with deterministic arc-distance placement and speed

That is enough to answer:
1. does ACC wrongly capture a nearby same-direction vehicle in the other lane?
2. does same-lane association stay off?
3. does continuity hold stay off?

Recommended interim scenario contract:
1. keep one actor only
2. set:
   - `travel_direction: same`
   - `lane_offset_m` to the adjacent lane center
   - `spawn_arc_distance_m` or `start_distance_m`
   - `speed_mps`
3. choose one of two regimes:
   - `adjacent_lane_parallel_reject`
   - `adjacent_lane_overtake_reject`

Preferred first case:
- `adjacent_lane_parallel_reject`
- actor stays near ego longitudinally for long enough to be a tempting false target
- expected:
  - candidate may be visible
  - reject mode `wrong_lane`
  - `association_eligible_rate_pct = 0`
  - `same_lane_association_rate_pct = 0`
  - `continuity_hold_rate_pct = 0`
  - ACC never enters following because of it

This should land before mixed traffic because it gives us a clean wrong-lane rejection test with much lower simulator complexity.

### L9. ACC Robustness Scope

The right target is not “similar to a top company because we widened radar coverage.”

The right target is:
1. explicit same-lane lead association
2. short-horizon track continuity
3. direct rejection of wrong-lane / opposite-direction candidates
4. attribution that makes failures obvious

This is directionally similar to how production systems are structured:
- object association first
- continuity / tracking second
- controller consumption third

What we have today is still much simpler than a top-tier production stack.
Top programs usually rely on:
1. multi-object tracking
2. lane association against map / path geometry
3. same-direction gating
4. continuity through temporary sensor misses
5. multi-sensor fusion, not a single narrow forward-cone producer

So the answer is:
- yes, the direction in L8 is aligned with a serious architecture
- no, the current `±5° cone -> zero return -> free-flow` behavior is not production-grade

### L10. Mixed-Traffic Scenario Support

Goal:
- support one authoritative same-lane lead while additional traffic actors exist in adjacent or opposite lanes

This is the first point where we can truthfully test:
1. follow a real lead
2. while a different vehicle is present in the other lane
3. without letting ACC switch to the wrong target

This should come after L8, not before it.

Reason:
1. L8 proves wrong-target rejection in isolation
2. mixed traffic adds actor enumeration, association ranking, and pass-timing complexity
3. if we skip isolation, failures will be harder to attribute

#### L10.1 Required simulator contract

Files:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/TrackConfig.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/TrackLoader.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/LeadVehicle.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/TrackWaypointFollower.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/AVBridge.cs`

Implementation:
1. replace singular `lead_vehicle` ownership with list-based actor ownership
2. add `traffic_actors:` scenario syntax with per-actor fields:
   - `role: lead | traffic`
   - `travel_direction: same | opposite`
   - `lane_offset_m`
   - `speed_profile`
   - `spawn_arc_distance_m`
   - optional `target_pass_arc_distance_m`
   - optional `despawn_arc_distance_m`
3. preserve backward compatibility by mapping legacy `lead_vehicle:` into one `traffic_actor` with role `lead`

#### L10.2 Actor roles and association rules

Producer-side association must explicitly separate:
1. followable lead candidates
2. visible but non-followable traffic actors

Rules:
1. only one actor may be the followable ACC target at a time
2. `role=lead` is not sufficient by itself; the actor must also be:
   - same-lane
   - same-direction
   - positive arc-distance ahead
3. `role=traffic` actors are never followable by ACC
4. oncoming or wrong-lane actors are always reject-only inputs

#### L10.3 Mixed-traffic initialization

Use deterministic actor setup.

Example mixed-traffic case:
1. actor A:
   - `role: lead`
   - `travel_direction: same`
   - same lane
   - slower speed
2. actor B:
   - `role: traffic`
   - `travel_direction: opposite` or `same`
   - other lane
   - timed to pass or ride near ego during lead following

Initialization rule:
1. choose the ego start point
2. choose the lead-follow segment
3. choose when the secondary actor should be nearest ego
4. solve spawn arc distances from those timing constraints

Do not initialize mixed traffic by hand-tuning random distances until it "looks right."

#### L10.4 Required mixed-traffic coverage

Add after L8 passes:

1. `mixed_follow_lead_oncoming_straight`
- same-lane lead ahead
- oncoming actor in opposite lane on straight
- expected:
  - ACC follows lead
  - oncoming actor stays reject-only

2. `mixed_follow_lead_oncoming_curve`
- same-lane lead ahead
- oncoming actor in opposite lane on curve
- expected:
  - ACC follows lead
  - oncoming actor never becomes association-eligible

3. `mixed_follow_lead_adjacent_same_direction`
- same-lane lead ahead
- nearby same-direction actor in adjacent lane
- expected:
  - ACC follows the same-lane lead
  - adjacent-lane actor is rejected as `wrong_lane`

#### L10.5 Promotion gate

Do not declare mixed-traffic robustness complete until:
1. single-actor H8 still passes
2. single-actor wrong-target reject cases still pass
3. mixed-traffic cases keep ACC locked to the correct same-lane lead
4. no mixed-traffic case produces:
   - `same_lane_association` from the wrong actor
   - continuity hold on the wrong actor
   - `ACC_ACTIVE` against an adjacent-lane or oncoming target
