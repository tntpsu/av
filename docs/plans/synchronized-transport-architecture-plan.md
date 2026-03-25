# Synchronized Transport Architecture Plan

## Goal

Move closed-loop AV control off independently fetched `latest` snapshots and onto a coherent, FIFO-consumed simulation packet stream with explicit contracts, continuity accounting, and debug visibility.

This is intended to be robust and transferable across tracks. The design must hold for:
- highway free-flow and ACC scenarios
- hill/high-grade tracks
- local-road / curve-heavy tracks
- future multi-sensor additions (topdown, radar, map, prediction)

## Why This Plan Exists

The current stack can misclassify normal motion as a teleport because the control loop consumes camera and vehicle state as separate `latest` snapshots rather than a single coherent sample.

Concrete evidence from `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260325_142159.h5`:
- `vehicle/stream_front_queue_depth` is pinned at `48` for the entire run.
- The loop target is `13.0 Hz`, observed mean is about `11.97 Hz`, with p95 wall period about `152 ms`.
- The front stream frequently skips `8-12` Unity frames between processed samples.
- At highway speed, those skipped samples produce normal pose deltas around `2.0-2.35 m`.
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py` currently applies `_is_teleport_jump()` on raw consecutive processed positions with a fixed `2.0 m` threshold.
- That false-fires the post-jump cooldown, halves `trajectory/reference_point_velocity` to `7.5 m/s`, and the controller then brakes against that false low reference.

The immediate bug can be mitigated with a dt-aware teleport guard. That is necessary but not sufficient. The architectural fix is to stop creating mismatched/skipped processed samples for closed-loop control.

## Current Architecture and Failure Mode

### Current closed-loop ingest

Python main loop:
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
  - fetches latest front camera
  - fetches latest vehicle state
  - processes them as one control sample

Bridge client:
- `/Users/philiptullai/Documents/Coding/av/bridge/client.py`
  - `get_latest_camera_frame_with_metadata()`
  - `_sync_get_latest_vehicle_state()` / `get_latest_vehicle_state()`
  - optional parallel fetch of the two `latest` endpoints

Bridge server:
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`
  - keeps `latest_camera_frame` / `latest_camera_frames`
  - keeps `latest_vehicle_state`
  - also maintains FIFO queues, but the main closed-loop path does not use them

### Current transport weaknesses

1. Camera and vehicle state are not consumed as a single contract.
2. `latest` semantics make skipped Unity frames invisible to downstream logic until pose deltas are large.
3. Camera queue saturation is normal operating mode, not a detectable failure.
4. Teleport/jump detection is trying to infer continuity after the fact instead of receiving continuity as part of the input contract.
5. The stack has no single packet identifier for “the control sample we acted on.”

## Target Architecture

### Core design

Introduce a synchronized simulation packet contract:
- one packet per Unity simulation step or per accepted bundling unit
- packet keyed by `unity_frame_count`
- consumed in FIFO order for closed-loop control
- includes all required control inputs plus transport metadata

Closed-loop control should move from:
- `latest camera` + `latest vehicle state`

to:
- `next synchronized packet`

### Packet contents

Each packet must include, at minimum:
- `schema_version`
- `packet_id`
- `unity_frame_count`
- `unity_time`
- `bundle_policy` / `sync_policy`
- `front_camera`
  - image payload or reference
  - source timestamp
  - source frame id
- `vehicle_state`
  - all current state fields needed by perception/planning/control
  - source timestamp
- optional components
  - `topdown_camera`
  - `unity_feedback`
  - future `radar`, `map_pose`, `prediction_inputs`
- continuity and transport diagnostics
  - packet age at consume time
  - front/vehicle component age
  - component frame alignment delta
  - skipped Unity frames since last consumed packet
  - bundle queue depth
  - bundle drop count
  - component orphan counts
  - bundle fallback flag + reason

### Design rules

1. Closed-loop control uses FIFO packet consumption only.
2. `latest` endpoints remain available for debug/UI, not as the primary control path.
3. Every fallback must be explicit and recorded.
4. A packet is either:
   - complete and consumed, or
   - incomplete and rejected/degraded with a reason.
5. Teleport detection becomes a secondary safety guard, not the primary continuity detector.

## Recommended Implementation Strategy

Implement this in staged rollout so the stack can be validated safely.

### Phase 0: Contract definition and shadow plumbing

#### Deliverables
- Define a canonical packet schema in code and docs.
- Add a bridge-side bundler keyed by `unity_frame_count`.
- Keep current control path active.
- Record bundle diagnostics in shadow mode.

#### Files to change
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`
- `/Users/philiptullai/Documents/Coding/av/bridge/client.py`
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
- `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
- `/Users/philiptullai/Documents/Coding/av/data/recorder.py`
- `/Users/philiptullai/Documents/Coding/av/docs/analysis/contracts/` if you want a formal JSON schema alongside code

#### Server-side implementation details

Add a bundler data structure in `/Users/philiptullai/Documents/Coding/av/bridge/server.py`:
- `pending_packets_by_unity_frame: dict[int, PartialPacket]`
- `assembled_packet_queue: deque(maxlen=N)`
- `latest_packet: Optional[Packet]`
- counters:
  - `packet_drop_count`
  - `packet_orphan_camera_count`
  - `packet_orphan_vehicle_count`
  - `packet_timeout_count`
  - `packet_partial_publish_count`

Define a `PartialPacket` structure that can accumulate:
- front camera component
- topdown component
- vehicle state component
- unity feedback component

Bundle assembly rule:
- primary key: `unity_frame_count`
- completeness requirement for closed-loop packet:
  - front camera present
  - vehicle state present
- optional components may be attached if present within the same frame id or bounded tolerance

Timeout rule:
- if one component for a frame never arrives within a configurable timeout or age window, mark orphan and evict it
- do not silently pair it to a different frame

New endpoints:
- `GET /api/sim/packet/latest`
- `GET /api/sim/packet/next`
- optionally raw-camera-compatible packet endpoints for reduced decode overhead if needed

Retain existing endpoints:
- `/api/camera/latest/raw`
- `/api/camera/next/raw`
- `/api/vehicle/state/latest`
- `/api/vehicle/state/next`

But mark them as:
- legacy/debug inputs
- not the primary closed-loop contract once rollout advances

#### Client-side implementation details

Add packet fetch methods in `/Users/philiptullai/Documents/Coding/av/bridge/client.py`:
- `get_next_sync_packet()`
- `get_latest_sync_packet()`

Add a policy enum or string mode:
- `latest_parallel`
- `packet_shadow`
- `packet_fifo_active`
- optional `packet_fifo_with_latest_fallback`

Important: if fallback is supported, it must record:
- `sync_packet_fallback_active=True`
- `sync_packet_fallback_reason`
- `sync_packet_expected_frame_count`
- `sync_packet_missing_components_mask`

#### Orchestrator shadow-mode implementation

In `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`:
- keep current ingest path for control in Phase 0
- also fetch shadow packet diagnostics or consume them via a low-overhead side path
- compare:
  - current control sample frame id
  - bundle frame id
  - component ages
  - skipped-frame counts
  - whether the current sample would have been rejected by the packet contract

Do not change runtime behavior yet.

### Phase 1: Recorder and diagnostics contract

#### Goal

Make packet continuity a first-class recorded contract, not an inferred story.

#### New `VehicleState` / recorder fields

Add fields in `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py` and register them in `/Users/philiptullai/Documents/Coding/av/data/recorder.py`.

Required fields:
- `sync_packet_mode`
- `sync_packet_schema_version`
- `sync_packet_id`
- `sync_packet_unity_frame_count`
- `sync_packet_complete`
- `sync_packet_fallback_active`
- `sync_packet_fallback_reason_code`
- `sync_packet_queue_depth`
- `sync_packet_drop_count`
- `sync_packet_orphan_camera_count`
- `sync_packet_orphan_vehicle_count`
- `sync_packet_timeout_count`
- `sync_packet_skipped_unity_frames`
- `sync_packet_age_ms`
- `sync_front_age_ms`
- `sync_vehicle_age_ms`
- `sync_front_vehicle_frame_delta`
- `sync_front_vehicle_time_delta_ms`
- `sync_packet_missing_front`
- `sync_packet_missing_vehicle`

Also add explicit cooldown/jump provenance fields to close the current observability gap:
- `control/post_jump_cooldown_active`
- `control/post_jump_cooldown_frames_remaining`
- `control/post_jump_reason_code`
- `control/reference_velocity_effective`
- `control/teleport_detected`
- `control/teleport_jump_m`
- `control/teleport_expected_motion_m`
- `control/teleport_motion_ratio`

This makes future root-cause analysis much cheaper.

### Phase 2: Debug tools and analysis updates

#### PhilViz

Update:
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/server.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`

Add a `Transport & Contracts` section with:
- Sync packet mode
- Packet completeness rate
- Fallback active rate
- Packet queue depth p50/p95/max
- Packet drop/orphan/timeout counters
- Skipped Unity frames p50/p95/max
- Front vs vehicle age p50/p95/max
- Front vs vehicle frame delta p50/p95/max
- Effective reference velocity vs planned/final target speed
- Post-jump cooldown active rate and episode table

Add timeline overlays for:
- `sync_packet_complete`
- `sync_packet_fallback_active`
- `sync_packet_skipped_unity_frames`
- `control/post_jump_cooldown_active`
- `control/teleport_detected`
- `control/reference_velocity_effective`

Add one issue classifier for this exact failure mode:
- `transport_false_teleport_cooldown`
  - condition: teleport/cooldown episodes coincide with high skipped-frame counts or stale packet episodes and effective ref speed drop

#### CLI analyzers

Update:
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/cadence_breakdown.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/verify_recording_run.py`

Required new outputs:
- packet completeness summary
- fallback summary
- orphan/timeout summary
- skipped-frame severity summary
- effective reference velocity drop attribution
- classification of brake episodes into:
  - real speed cap
  - ACC demand
  - post-jump cooldown
  - feasibility backstop
  - unknown

#### Acceptance story

Analyzer and PhilViz must agree on the same definitions and numbers for:
- packet completeness
- packet fallback rate
- skipped-frame p95
- post-jump cooldown rate
- effective reference velocity drop count

### Phase 3: Active closed-loop packet consumption

#### Goal

Move control onto `packet_fifo_active`.

#### Orchestrator changes

In `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`:
- replace `get_latest_camera_and_vehicle_parallel()` with `get_next_sync_packet()` when active mode is enabled
- derive control dt from packet continuity / timestamps, not inferred latest behavior
- track `last_consumed_packet_id` and `last_consumed_unity_frame`
- detect missing packets directly from packet ids or unity frame count deltas

#### Safety / fallback policy

If packet stream degrades:
- configurable policy by severity

Example levels:
- Level 0: packet complete, proceed normally
- Level 1: one optional component missing, proceed with degraded diagnostics only
- Level 2: required component missing, hold previous control or enter safe degraded mode
- Level 3: repeated packet timeout/drop, trigger explicit transport fault handling

Transport fault handling must not silently masquerade as a teleport.

### Phase 4: Teleport guard demotion and redesign

After active packet mode is stable:
- keep teleport guard
- make it consume packet continuity info and expected motion
- only fire as a true discontinuity detector

Required guard inputs:
- packet dt
- skipped Unity frames
- expected motion from speed and dt
- explicit reset/respawn events if available from Unity

New rule shape:
- a large pose delta with matching skipped-frame/time evidence is not teleport by itself
- a large pose delta inconsistent with dt, speed, and continuity is teleport

### Phase 5: Rollout and promotion

Rollout order:
1. shadow mode on highway + ACC scenario
2. shadow mode on hill/highway
3. shadow mode on one local-road/curve track
4. active mode on controlled highway scenario
5. active mode across promotion matrix

Do not switch all tracks at once.

## Detailed Test Plan

### Unit tests

#### Bridge server
- packet assembly by matching `unity_frame_count`
- orphan camera eviction
- orphan vehicle eviction
- packet timeout behavior
- queue overflow/drop accounting
- packet latest vs next semantics

Suggested file:
- `/Users/philiptullai/Documents/Coding/av/tests/test_bridge_sync_packets.py`

#### Bridge client
- parse and consume sync packet responses
- fallback flag propagation
- mode switching behavior

Suggested file:
- `/Users/philiptullai/Documents/Coding/av/tests/test_bridge_client_sync_packets.py`

#### Orchestrator
- active mode consumes packet FIFO
- fallback path records explicit reason
- dt derived from packet continuity
- teleport guard uses expected-motion inputs when enabled

Suggested file:
- `/Users/philiptullai/Documents/Coding/av/tests/test_orchestrator_sync_policy.py`

#### Recorder contract
- new sync packet fields exist in HDF5
- old recordings still load cleanly
- schema version is written

Suggested file:
- `/Users/philiptullai/Documents/Coding/av/tests/test_control_command_hdf5.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_drive_summary_contract.py`

#### Analyzer / PhilViz parity
- analyzer and PhilViz backend compute identical packet summary values
- issue classification matches CLI output

Suggested file:
- `/Users/philiptullai/Documents/Coding/av/tests/test_drive_summary_parity.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_visualizer_transport_parity.py`

### Integration tests

1. Synthetic bundle assembly replay
- feed camera and vehicle streams with controlled jitter
- verify bundler assembles correct packets
- verify expected orphan counts

2. Throughput degradation test
- artificially slow consumer
- verify packet drop accounting rises
- verify no false teleport trigger in shadow mode

3. Fallback semantics test
- withhold vehicle state for several frames
- verify fallback flag and reason are recorded
- verify no silent latest substitution unless explicitly enabled

### Live validation matrix

#### Stage A: Shadow-mode validation

Tracks/scenarios:
- `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`
- `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h8_curve_catchup.yml`
- `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/hill_g1_grade_following.yml`
- one local-road track that exercises curves and lower speed

Gates:
- no behavior regression in shadow mode
- packet completeness > 99% for required components
- fallback active rate near zero on nominal runs
- packet queue/drop story explains existing cadence behavior
- no unexplained effective reference speed drops

#### Stage B: Active-mode validation

Highway first.

Success gates:
- no periodic false cooldown episodes
- no repeated `reference_velocity_effective = 0.5 * target_speed` unless explicitly due to safety policy
- no transport-caused overspeed brake pulses
- packet drop/orphan rates within defined budget
- analyzer and PhilViz agree on transport metrics

#### Stage C: Cross-track promotion

Required before promotion:
- highway free-flow pass
- ACC scenario pass
- hill/highway pass
- local-road pass
- no new transport fault patterns in PhilViz/triage

## Error Handling Contracts

These need to be explicit.

### Required component missing
- packet marked incomplete
- reason code written
- active mode uses configured degraded policy
- no silent substitution

### Queue overflow
- increment packet drop count
- record severity event
- do not hide it behind latest semantics

### Timestamp/frame inconsistency
- if camera and vehicle disagree on `unity_frame_count` beyond tolerance, do not pair silently
- record orphan or mismatch event

### Bridge restart / Unity reconnect
- first packet after reconnect must be marked as discontinuity
- controllers may reset explicitly
- this should be distinct from teleport detection

## Risks and Tradeoffs

### Risks introduced by the new architecture

1. Higher implementation complexity in the bridge
- Mitigation: shadow mode first, formal packet schema, bounded scope

2. Potential increase in control latency if FIFO queue is allowed to grow
- Mitigation: keep packet queue shallow, record queue age/depth, fail loudly rather than silently lagging

3. More explicit transport faults may initially look like regressions
- Mitigation: this is a feature; the stack should surface real continuity failures instead of hiding them

4. Migration bugs from legacy latest paths to active packet path
- Mitigation: dual-mode rollout, parity tests, packet shadow mode before active

## Other Fragility Areas Found During Analysis

These are not the same bug, but they are the same class of weak-contract risk.

### 1. Control command transport is latest-wins and lossy

Files:
- `/Users/philiptullai/Documents/Coding/av/bridge/client.py`

Current behavior:
- control commands are queued with `maxsize=1`
- unsent command is dropped on every new enqueue
- no explicit ack in the recording for which command was actually applied

Risk:
- command drops are acceptable only if explicitly treated as latest-wins by contract
- today they are mostly invisible

Recommendation:
- record control command sequence id, send timestamp, apply ack timestamp if Unity can return it
- expose dropped-command counts in recorder/PhilViz

### 2. Trajectory transport is also latest-wins and lossy

Files:
- `/Users/philiptullai/Documents/Coding/av/bridge/client.py`

Risk:
- lower than control because it is mainly visualization/debug
- but if any runtime behavior depends on Unity consuming the latest trajectory, the contract should be explicit that this is best-effort only

Recommendation:
- mark trajectory transport as visualization-only in docs and code comments unless/until runtime depends on it

### 3. Front and topdown streams do not have a strict coherence contract

Files:
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/server.py`

Risk:
- overlay trust can degrade if topdown/front are assumed aligned when they are not

Recommendation:
- carry topdown in the same packet contract if used for runtime/triage alignment
- otherwise mark topdown as advisory/debug-only with explicit frame delta metrics

### 4. Speed-target contract is split across multiple fields

Observed in the highway analysis:
- `control/target_speed_final`
- `control/target_speed_post_limits`
- `trajectory/reference_point_velocity`

Risk:
- operator sees one speed story while controller acts on another

Recommendation:
- define one canonical `reference_velocity_effective`
- classify every difference from desired target with a named cause

### 5. Scenario intent vs road-limit intent is not obvious enough

Example:
- road limit says `65 mph`
- scenario/config target is `15 m/s` steady-follow ACC

Risk:
- users interpret behavior as a speed-limit bug when it is actually scenario intent plus transport fault

Recommendation:
- record and surface one run-intent card in PhilViz:
  - track
  - scenario
  - mode (free-flow / ACC-follow / stop-go / etc.)
  - configured target speed
  - lead vehicle enabled

### 6. Optional recorder fields are doing too much silent compatibility work

Files:
- `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`

Risk:
- tools degrade gracefully, but sometimes too gracefully, which can hide missing-contract situations

Recommendation:
- add explicit schema versioning for transport/continuity contract groups
- prefer `field unavailable` over silently treating missing as zero when diagnosing transport issues

## Solutions for Other Contract Fragilities

These should be built into the transport re-architecture plan, not left as follow-on cleanup. They are the same class of problem: hidden lossy behavior, ambiguous ownership, or weak runtime contracts.

### 1. Control command transport contract

#### Problem
- `/Users/philiptullai/Documents/Coding/av/bridge/client.py` uses a `maxsize=1` latest-wins queue for control commands.
- Commands can be superseded before send.
- The recording does not clearly tell the operator which command Unity actually applied.

#### Recommendation
Keep latest-wins if you want low-latency control semantics, but make it explicit and observable.

#### Implementation
Add command transport ids and ack fields:
- Python command fields
  - `control_command_seq_id`
  - `control_command_generated_mono_s`
  - `control_command_enqueued_mono_s`
  - `control_command_sent_mono_s`
  - `control_command_superseded_before_send`
- Unity ack/apply fields
  - `unity_feedback.last_control_command_seq_id_received`
  - `unity_feedback.last_control_command_seq_id_applied`
  - `unity_feedback.last_control_command_apply_unity_frame`
  - `unity_feedback.last_control_command_apply_unity_time`

Add recorder fields:
- `control/control_command_seq_id`
- `control/control_command_superseded_count`
- `control/control_command_transport_latency_ms`
- `control/control_command_apply_latency_ms`
- `control/control_command_ack_gap_frames`

Add PhilViz diagnostics:
- `Control Transport` card:
  - superseded command rate
  - send error count
  - command apply latency p50/p95
  - last-applied seq id drift

Add tests:
- unit: superseded commands increment count and latest seq wins
- integration: Unity ack for applied command matches last applied seq id
- analyzer parity: apply latency in PhilViz matches HDF5-derived CLI summary

### 2. Trajectory transport contract

#### Problem
- `/Users/philiptullai/Documents/Coding/av/bridge/client.py` also uses latest-wins async transport for trajectory data.
- It is unclear whether this path is visualization-only or runtime-significant.

#### Recommendation
Make the contract explicit.

#### Implementation
Two allowed modes:
- `visualization_only`
- `runtime_required`

If `visualization_only`:
- keep latest-wins
- record drops
- do not use trajectory delivery as evidence of runtime planner correctness

If `runtime_required` in the future:
- add seq ids and apply ack exactly like control transport

Recorder fields:
- `trajectory/transport_mode`
- `trajectory/transport_seq_id`
- `trajectory/transport_dropped_count`
- `trajectory/transport_apply_ack_seq_id`

PhilViz:
- show `Visualization Only` badge when appropriate
- never mix dropped-trajectory transport with runtime path failures unless mode is `runtime_required`

Tests:
- verify visualization-only mode does not fail drive-summary health
- verify runtime-required mode enforces ack coverage if enabled

### 3. Front/topdown coherence contract

#### Problem
- Front and topdown currently have diagnostics, but not a strict coherence contract.
- It is easy to over-trust the overlay when the streams are not tightly aligned.

#### Recommendation
Make topdown an explicit optional packet component with trust gating.

#### Implementation
Add packet-level coherence fields:
- `sync_topdown_present`
- `sync_topdown_age_ms`
- `sync_topdown_front_frame_delta`
- `sync_topdown_front_time_delta_ms`
- `sync_topdown_trust`
- `sync_topdown_trust_reason_code`

Policy:
- topdown is advisory/debug unless `sync_topdown_trust == true`
- if outside tolerance, UI can still display it, but must label it `UNTRUSTED`
- runtime logic must not depend on topdown unless a future explicit mode enables it

PhilViz:
- `Top-Down Contract` card with trust, frame delta, time delta, and trust reason
- frame-level overlay warning when topdown is untrusted

Tests:
- parity tests for trusted/untrusted classification
- backward compatibility on recordings without topdown contract fields

### 4. Canonical speed-target contract

#### Problem
The stack currently exposes multiple speed targets that can diverge:
- `control/target_speed_final`
- `control/target_speed_post_limits`
- `trajectory/reference_point_velocity`

This creates operator confusion and hides the actual control truth.

#### Recommendation
Introduce a canonical speed decision ledger and one explicit effective controller target.

#### Implementation
Add a per-frame speed-waterfall contract:
- `speed_target_desired_mps`
- `speed_target_acc_override_mps`
- `speed_target_governor_mps`
- `speed_target_post_limits_mps`
- `speed_target_reference_effective_mps`
- `speed_target_controller_effective_mps`
- `speed_target_effective_reason_code`
- `speed_target_effective_owner`

Reason code examples:
- `free_flow_target`
- `acc_follow`
- `curve_cap`
- `feasibility_backstop`
- `post_jump_cooldown`
- `fallback_unknown`

PhilViz:
- `Speed Intent` card
- waterfall row with the canonical effective target highlighted
- red failure highlight when effective target is unexpectedly below desired target

Analyzer:
- classify every brake episode by effective target reason
- flag inconsistent stories such as `desired=15`, `effective=7.5`, `curve_cap=off`, `acc=off`

Tests:
- unit tests for mutually exclusive reason selection
- integration test that the analyzer labels post-jump cooldown correctly

### 5. Scenario/run-intent contract

#### Problem
Road speed limit, scenario intent, and controller target are too easy to confuse.
A 65 mph road with a 15 m/s ACC-follow scenario looks like a bug if run intent is not surfaced.

#### Recommendation
Record run intent as a first-class contract from startup through the HDF5 and PhilViz summary.

#### Implementation
At startup, capture and persist:
- `run_intent_mode` (`free_flow`, `acc_follow`, `stop_go`, `hard_brake`, etc.)
- `run_config_path`
- `run_track_path`
- `run_scenario_name`
- `run_target_speed_mps`
- `run_speed_limit_expected_mps`
- `run_lead_vehicle_enabled`
- `run_lead_vehicle_speed_mps`
- `run_lead_vehicle_start_distance_m`

Suggested sources:
- startup script / orchestrator config parsing
- scenario YAML loader
- track metadata

PhilViz summary:
- `Run Intent` card
- clear statement of:
  - road speed limit
  - configured target speed
  - scenario type
  - whether lead following is active

Analyzer:
- `intent mismatch` warning when operator expectation is likely to be wrong, e.g. free-flow road limit much higher than configured target speed

Tests:
- startup/config parsing tests
- HDF5 metadata contract tests
- summary rendering tests

### 6. Schema and missing-field contract

#### Problem
Optional fields and graceful defaults currently make some missing-contract situations look like real zeros.
That is dangerous for transport debugging.

#### Recommendation
Version the recorder contract by feature group and distinguish `missing`, `unavailable`, and `zero`.

#### Implementation
Add schema-version groups, for example:
- `transport_contract_schema_version`
- `speed_contract_schema_version`
- `intent_contract_schema_version`

Policy:
- required diagnostic fields for a feature group must be present together when that schema version is declared
- analyzer must output `UNAVAILABLE` rather than silently coercing to `0` when required group fields are missing
- legacy recordings remain supported, but with explicit `legacy/no-contract` labeling

PhilViz:
- render `N/A (legacy recording)` or `N/A (schema missing)` explicitly
- avoid showing `0` when the field was absent

Tests:
- old-recording compatibility tests
- schema-group completeness tests
- analyzer/PhilViz parity tests for unavailable vs zero

## Rollout Integration for 1-6

Build these into the main rollout, not after it:

### Phase 0 additions
- define transport, speed, and run-intent schema groups
- define reason-code enums centrally

### Phase 1 additions
- record control command seq/ack contract
- record speed decision ledger
- record run intent metadata
- record explicit schema versions

### Phase 2 additions
- add PhilViz cards for:
  - Control Transport
  - Transport & Contracts
  - Speed Intent
  - Run Intent
  - Top-Down Contract
- add CLI parity checks for all new summaries

### Phase 3 additions
- active FIFO packet mode only promotes if:
  - packet completeness passes
  - command apply ack continuity passes
  - no unexplained effective target drops

### Promotion gates additions
- no false post-jump cooldown episodes
- no hidden command supersession drift beyond budget
- no analyzer/UI disagreement on packet or speed-contract metrics
- legacy recordings still load with explicit unavailable labels


## Recommended Rollout Decision

The right path is:
1. keep a smarter teleport guard as temporary protection
2. do not treat that as the architectural fix
3. build the synchronized packet contract in shadow mode
4. promote closed-loop control to FIFO packets only after shadow metrics are stable

## Definition of Done

This work is done when:
- closed-loop control no longer depends on separate `latest` snapshots
- packet continuity is explicit and recorded
- false teleport/cooldown episodes disappear on highway nominal runs
- PhilViz and CLI explain transport faults directly
- other silent latest-wins contracts are either instrumented or explicitly demoted to debug-only status

## Suggested Immediate Execution Order

1. Phase 0 contract + bridge-side bundler in shadow mode
2. Phase 1 recorder/PhilViz/analyzer contract
3. Shadow validation on the highway scenario that exposed the problem
4. Phase 3 active-mode consumption on highway only
5. Teleport guard redesign after packet continuity proves out
6. Follow-up work on command/trajectory/topdown contract gaps

## Post-Stabilization Cleanup Phase

After the synchronized packet path is active, validated, and promoted, do a surgical cleanup pass to reduce future operator confusion.

This cleanup must happen only after:
- packet FIFO active mode is the closed-loop default on promoted tracks
- transport metrics are stable in both PhilViz and CLI analyzers
- no false post-jump cooldown episodes remain in the validation matrix

### Cleanup goals

1. Remove or demote legacy assumptions that still imply `latest` snapshot semantics are authoritative.
2. Remove duplicate or ambiguous summaries when a new canonical transport/speed/intent story exists.
3. Keep backward compatibility for old recordings, but make legacy status explicit.
4. Reduce the number of UI cards, analyzer fields, and helper paths that tell overlapping or conflicting stories.

### Cleanup scope

#### 1. Legacy endpoint and mode cleanup

Files:
- `/Users/philiptullai/Documents/Coding/av/bridge/client.py`
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`

Actions:
- audit all remaining uses of:
  - `get_latest_camera_frame_with_metadata()`
  - `get_latest_vehicle_state()`
  - `get_latest_camera_and_vehicle_parallel()`
- keep them only where they are truly debug-only or compatibility-only
- rename or comment them clearly as legacy/debug paths if they remain
- remove dead fallback branches that are no longer exercised after promotion

#### 2. Debug-tool/card cleanup

Files:
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/server.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`

Actions:
- remove or relabel cards/rows that duplicate the new canonical transport contract
- ensure only one card is the source of truth for:
  - transport continuity
  - effective speed target
  - run intent
  - topdown trust
- mark old metrics as `legacy` where needed instead of showing them as peers to canonical metrics

#### 3. Analyzer cleanup

Files:
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/cadence_breakdown.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/verify_recording_run.py`

Actions:
- remove redundant heuristics that were only needed because transport continuity was inferred rather than explicit
- prefer packet-contract-backed attribution over indirect queue-only attribution
- retain legacy fallbacks only for old recordings, labeled explicitly as legacy mode

#### 4. Schema cleanup

Files:
- `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
- `/Users/philiptullai/Documents/Coding/av/data/recorder.py`

Actions:
- identify fields that became obsolete once canonical packet/speed/intent fields exist
- avoid immediate deletion if they are useful for historical comparisons
- instead:
  - mark obsolete fields as deprecated in code comments and docs
  - stop surfacing them prominently in tools
  - plan deletion only after a defined deprecation window if desired

### Cleanup tests

Add one final cleanup-oriented review/test pass:
- analyzer/PhilViz parity still passes
- legacy recordings still load with explicit legacy labels
- no UI tab shows two competing “source of truth” metrics for the same concept
- no promoted code path still depends on legacy latest-only control ingest

### Cleanup acceptance criteria

The cleanup phase is complete when:
- the promoted transport path is the only active closed-loop authority
- remaining legacy paths are explicitly debug-only or compatibility-only
- PhilViz and CLI show one canonical story for transport, speed intent, and run intent
- obsolete or overlapping metrics no longer confuse operator triage

## Exact Implementation Checklist: Patch Sets A+B+C

This section is the execution checklist for the first implementation slice only:
- Patch Set A: contract foundations, no runtime behavior change
- Patch Set B: bridge-side packet bundler in shadow mode
- Patch Set C: client + orchestrator shadow consumption

The objective of A+B+C is:
- define the packet contract
- assemble coherent packets on the bridge
- record packet continuity in HDF5
- keep the existing closed-loop behavior unchanged

Do not change active control ownership or replace the current ingest path in this slice.

### Preconditions

1. Start from commit `f627d44` or later.
2. Do not include local artifacts under `/Users/philiptullai/Documents/Coding/av/data/recordings/`.
3. Keep the current `latest_parallel` path as the active control path during all of A+B+C.

### Patch Set A: Contract Foundations

#### A1. Add central transport contract enums and schema constants

Files:
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`
- `/Users/philiptullai/Documents/Coding/av/bridge/client.py`
- `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`

Implementation:
- define a packet schema version constant, for example:
  - `SYNC_PACKET_SCHEMA_VERSION = 1`
- define string or integer enums for:
  - packet mode
    - `latest_parallel`
    - `packet_shadow`
    - `packet_fifo_active`
  - packet fallback reason
    - `none`
    - `missing_front_camera`
    - `missing_vehicle_state`
    - `packet_timeout`
    - `packet_queue_overflow`
    - `packet_legacy_path`
  - post-jump reason
    - `none`
    - `teleport_guard`
    - `explicit_reset`
    - `unknown`
- define canonical effective-target reason enums for later use, but do not wire behavior yet:
  - `free_flow_target`
  - `acc_follow`
  - `curve_cap`
  - `feasibility_backstop`
  - `post_jump_cooldown`
  - `fallback_unknown`

Notes:
- keep reason definitions centralized so recorder, analyzer, and PhilViz do not drift.
- if you add a new helper module for shared enums, document it in this plan and import from one place only.

#### A2. Extend `VehicleState` with packet shadow fields

Files:
- `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`

Fields to add:
- `sync_packet_mode`
- `sync_packet_schema_version`
- `sync_packet_id`
- `sync_packet_unity_frame_count`
- `sync_packet_complete`
- `sync_packet_fallback_active`
- `sync_packet_fallback_reason_code`
- `sync_packet_queue_depth`
- `sync_packet_drop_count`
- `sync_packet_orphan_camera_count`
- `sync_packet_orphan_vehicle_count`
- `sync_packet_timeout_count`
- `sync_packet_skipped_unity_frames`
- `sync_packet_age_ms`
- `sync_front_age_ms`
- `sync_vehicle_age_ms`
- `sync_front_vehicle_frame_delta`
- `sync_front_vehicle_time_delta_ms`
- `sync_packet_missing_front`
- `sync_packet_missing_vehicle`

Also add explicit post-jump continuity fields now so shadow analysis has the real cause:
- `post_jump_cooldown_active`
- `post_jump_cooldown_frames_remaining`
- `post_jump_reason_code`
- `reference_velocity_effective`
- `teleport_expected_motion_m`
- `teleport_motion_ratio`

Rules:
- choose explicit default values that cannot be confused with real measurements where possible.
- if legacy support requires numeric defaults, ensure analyzers later can distinguish schema missing from real zero.

#### A3. Register new HDF5 datasets

Files:
- `/Users/philiptullai/Documents/Coding/av/data/recorder.py`

Implementation:
- register every new field in all recorder registration paths already used for control/vehicle data.
- keep dataset names stable and flat, following current repo conventions.
- recommended groups:
  - `vehicle/sync_packet_*`
  - `control/post_jump_*`
  - `control/reference_velocity_effective`
  - `control/teleport_expected_motion_m`
  - `control/teleport_motion_ratio`

Acceptance:
- old recordings must still open cleanly in readers that treat missing datasets as unavailable.

### Patch Set B: Bridge-Side Packet Bundler

#### B1. Add pending packet assembly structures

Files:
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`

Implementation:
- add a `PartialPacket` structure keyed by `unity_frame_count`
- add:
  - `pending_packets_by_unity_frame`
  - `assembled_packet_queue`
  - `latest_packet`
  - counters for drop/orphan/timeout/partial publish

Packet contract for A+B+C:
- required for a complete packet:
  - front camera
  - vehicle state
- optional:
  - topdown
  - unity feedback

#### B2. Assemble packets without changing legacy endpoints

Files:
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`

Implementation:
- on camera arrival, attach camera component to the matching partial packet by `unity_frame_count` if available
- on vehicle state arrival, attach vehicle component similarly
- once complete, move packet to:
  - `latest_packet`
  - `assembled_packet_queue`

Required metadata on each packet:
- packet id
- schema version
- unity frame count
- unity time
- component source timestamps
- component frame ids
- packet queue depth at publish
- component age fields or enough raw timing to derive them

Important:
- do not remove or alter current behavior of:
  - `/api/camera/latest/raw`
  - `/api/camera/next/raw`
  - `/api/vehicle/state/latest`
  - `/api/vehicle/state/next`

#### B3. Add packet endpoints

Files:
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`

Endpoints:
- `GET /api/sim/packet/latest`
- `GET /api/sim/packet/next`

Behavior:
- `latest` returns the most recent assembled packet
- `next` pops from assembled FIFO queue
- response must include:
  - complete/fallback flags
  - missing component flags
  - queue/drop/orphan counters

#### B4. Add bounded timeout/orphan handling

Files:
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`

Implementation:
- if a partial packet never receives its matching required component within a bounded frame/time window:
  - increment orphan/timeout counters
  - evict it
- do not silently match a component to a different frame

Notes:
- keep timeout policy simple in A+B+C
- avoid introducing recovery heuristics yet

### Patch Set C: Client + Orchestrator Shadow Consumption

#### C1. Add client packet fetch methods

Files:
- `/Users/philiptullai/Documents/Coding/av/bridge/client.py`

Implementation:
- add:
  - `get_latest_sync_packet()`
  - `get_next_sync_packet()`
- parse packet fields into plain Python structures
- add a client mode field with current default:
  - `latest_parallel`
- allow `packet_shadow` mode to fetch packet diagnostics while keeping current control ingest active

Do not:
- switch the orchestrator’s active control path yet

#### C2. Add orchestrator shadow packet capture

Files:
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`

Implementation:
- during the current main loop, after obtaining the current active inputs, also fetch packet shadow data
- record shadow values into the outgoing `VehicleState`
- compute:
  - packet completeness
  - packet fallback flag
  - skipped Unity frames
  - front/vehicle age
  - front/vehicle frame delta
  - front/vehicle time delta
- record `reference_velocity_effective`
  - this should be the value the longitudinal controller actually tracks
- record explicit post-jump cooldown status each frame

Important:
- do not alter the current control dt, perception path, planner path, or controller path in A+B+C
- this patch set is instrumentation-only from a runtime behavior standpoint

#### C3. Upgrade teleport provenance only, not teleport behavior

Files:
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
- `/Users/philiptullai/Documents/Coding/av/av_stack/speed_helpers.py`

Implementation:
- record:
  - observed jump distance
  - expected motion distance from speed and dt
  - ratio between observed and expected motion
- keep the actual teleport behavior unchanged in A+B+C

Reason:
- the first slice is about proving continuity, not changing safety behavior

### Tests to Add or Update

#### New test files

Add:
- `/Users/philiptullai/Documents/Coding/av/tests/test_bridge_sync_packets.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_bridge_client_sync_packets.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_orchestrator_sync_policy.py`

#### Existing test files to update

Update:
- `/Users/philiptullai/Documents/Coding/av/tests/test_control_command_hdf5.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_drive_summary_contract.py`

#### Exact unit test checklist

`test_bridge_sync_packets.py`
- packet assembles when matching camera + vehicle components share `unity_frame_count`
- packet does not assemble when one required component is missing
- orphan camera increments orphan counter
- orphan vehicle increments orphan counter
- packet timeout increments timeout counter
- queue overflow increments packet drop count
- `/latest` does not consume packet
- `/next` consumes packet FIFO

`test_bridge_client_sync_packets.py`
- client parses latest packet response
- client parses next packet response
- missing/partial packet fields map to fallback flags correctly
- packet mode enum wiring works for `latest_parallel` and `packet_shadow`

`test_orchestrator_sync_policy.py`
- `packet_shadow` records packet fields without changing active ingest path
- `reference_velocity_effective` is recorded from the real controller reference path
- post-jump cooldown fields are recorded when cooldown is active
- skipped Unity frames are recorded from packet shadow metadata

`test_control_command_hdf5.py`
- new sync packet datasets exist
- new post-jump datasets exist
- new effective-reference dataset exists

`test_drive_summary_contract.py`
- legacy recordings with no sync packet datasets still load
- missing packet datasets are treated as unavailable, not as real zeros

### Validation Commands

Run after A+B+C land:

Syntax:
```bash
python3 -m py_compile /Users/philiptullai/Documents/Coding/av/bridge/server.py \\
  /Users/philiptullai/Documents/Coding/av/bridge/client.py \\
  /Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py \\
  /Users/philiptullai/Documents/Coding/av/data/formats/data_format.py \\
  /Users/philiptullai/Documents/Coding/av/data/recorder.py
```

Targeted tests:
```bash
pytest /Users/philiptullai/Documents/Coding/av/tests/test_bridge_sync_packets.py \\
  /Users/philiptullai/Documents/Coding/av/tests/test_bridge_client_sync_packets.py \\
  /Users/philiptullai/Documents/Coding/av/tests/test_orchestrator_sync_policy.py \\
  /Users/philiptullai/Documents/Coding/av/tests/test_control_command_hdf5.py \\
  /Users/philiptullai/Documents/Coding/av/tests/test_drive_summary_contract.py -q
```

Live shadow validation:
```bash
./start_av_stack.sh --build-unity-player --skip-unity-build-if-clean --run-unity-player \\
  --config /Users/philiptullai/Documents/Coding/av/config/acc_highway.yaml \\
  --track-yaml /Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml \\
  --duration 90
```

Then inspect the newest recording with:
```bash
python /Users/philiptullai/Documents/Coding/av/tools/analyze/verify_recording_run.py --latest
python /Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py --latest
```

### Acceptance Gates for A+B+C

1. Runtime behavior must be unchanged relative to current ingest mode.
2. Sync packet datasets must exist in the new recording.
3. Packet shadow metrics must explain:
   - packet completeness
   - skipped Unity frames
   - front/vehicle age mismatch
   - post-jump cooldown provenance
4. Legacy recordings must still analyze cleanly.
5. No analyzer or recorder regression is allowed in existing ACC tooling.

### Explicit Non-Goals for A+B+C

Do not do any of the following in this slice:
- switch active control ingest to FIFO packets
- retune teleport thresholds
- change post-jump cooldown behavior
- redesign control/trajectory apply acks
- modify PhilViz UI beyond what is required to keep new fields loadable in later slices

Those belong in later patch sets after shadow data proves the transport story.
