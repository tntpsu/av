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

#### Phase 3A finding: metadata-only active mode is insufficient

The first `packet_fifo_active` attempt proved an important architectural point:
- consuming packet metadata from `/api/sim/packet/next`
- while still consuming camera pixels from `/api/camera/next/raw`
- and vehicle state from `/api/vehicle/state/next`

is not robust enough.

Observed failure modes from live highway validation:
- packet fallback remained too high even after stale-packet draining
- active mode still depended on reconstructing one control sample from three independent FIFO streams
- false post-jump cooldown episodes remained transport-linked
- the system improved queue pressure but did not eliminate continuity ambiguity

Conclusion:
- Patch Set F must not stop at metadata FIFO
- the next required step is a **single coherent packet payload endpoint**
- active mode should consume one authoritative packet payload per control sample

### Phase 3B: Coherent packet payload endpoint

#### Goal

Replace the current active-mode “packet metadata + separate camera/state FIFOs” design with:
- one bridge-side queued packet payload
- one client fetch
- one authoritative control sample

This is the first active-mode shape that is strong enough to remove transport ambiguity instead of shifting it around.

#### Contract decision

For active closed-loop control, the bridge must expose a payload endpoint that returns:
1. packet metadata
2. the full vehicle state snapshot that belongs to that packet
3. the front camera image bytes for that same packet

The orchestrator must no longer align these from separate endpoints.

#### Endpoint design

Add a new bridge endpoint:
- `GET /api/sim/packet/next/raw`

Optional companion endpoint for debugging:
- `GET /api/sim/packet/latest/raw`

Recommended response format:
- `multipart/mixed`

Parts:
1. `application/json`
   - packet metadata
   - full vehicle state
   - image metadata:
     - `camera_id`
     - `timestamp`
     - `frame_id`
     - `unity_frame_count`
     - `unity_time`
     - `shape`
     - `dtype`
2. `application/octet-stream`
   - raw RGB image bytes

Reason for `multipart/mixed`:
- preserves raw image transport
- avoids base64 inflation
- keeps full vehicle state and packet metadata together
- avoids header-size abuse for large JSON

Do not use:
- three separate FIFO requests
- a JSON-only payload with base64 image bytes for active mode

JSON+base64 is acceptable only for debug fallback endpoints, not the closed-loop path.

#### Bridge server changes

Files:
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`

Implementation:
- extend the assembled packet queue to store full payload contents, not metadata-only
- each queued active packet must contain:
  - packet metadata
  - full vehicle state snapshot
  - front image ndarray or raw bytes
  - image shape/dtype metadata
- keep the existing metadata-only snapshot helpers for shadow/debug use
- add a bounded active payload queue distinct from any debug/history queue if needed

Required queue/accounting fields:
- `sync_packet_payload_queue_depth`
- `sync_packet_payload_drop_count`
- `sync_packet_payload_bytes`
- `sync_packet_payload_oldest_age_ms`

Required policy:
- oldest payloads may be dropped only with explicit accounting
- if payload bytes exceed budget, record a distinct payload-drop reason
- no silent conversion from “payload missing” to “packet complete”

Suggested new fallback reason codes:
- `payload_timeout`
- `payload_queue_overflow`
- `payload_missing_front_bytes`
- `payload_decode_failed`

If new reason codes are added:
- update `/Users/philiptullai/Documents/Coding/av/sync_contract.py`
- update analyzer/PhilViz parity expectations

#### Client changes

Files:
- `/Users/philiptullai/Documents/Coding/av/bridge/client.py`

Implementation:
- add `get_next_sync_packet_payload_raw()`
- parse `multipart/mixed`
- decode the JSON part
- decode raw RGB bytes into ndarray using the packet’s shape/dtype metadata
- return one object:
  - `image`
  - `timestamp`
  - `frame_id`
  - `camera_meta`
  - `vehicle_state_dict`
  - `packet_meta`
  - `front_ready_mono_s`
  - `vehicle_ready_mono_s`
  - `inputs_ready_mono_s`

Active mode must then use this packet payload fetch instead of:
- `get_next_sync_packet()`
- `get_next_camera_frame_with_metadata()`
- `get_next_vehicle_state()`

Keep the current metadata-only packet methods for:
- shadow mode
- debugging
- compatibility

But they must not be the promoted active ingest path after this phase.

#### Orchestrator changes

Files:
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`

Implementation:
- replace the current active-mode bundle reconstruction with one payload fetch call
- active mode path becomes:
  1. `packet_payload = bridge.get_next_sync_packet_payload_raw()`
  2. extract `image`, `timestamp`, `vehicle_state_dict`, `packet_meta`
  3. compute continuity from packet ids / unity frame deltas
  4. pass packet metadata through recorder/analyzer fields

Remove from active mode:
- per-tick camera FIFO resync logic
- per-tick vehicle FIFO resync logic
- packet metadata drain-to-latest logic

Those may remain in debug utilities, but not in the promoted active path.

Required active-mode behavior:
- if payload fetch fails:
  - record explicit fallback reason
  - either use labeled legacy fallback or hold-safe policy, based on config
- if payload fetch succeeds:
  - do not call camera/vehicle FIFO endpoints for that control cycle

#### Recorder/data contract changes

Files:
- `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
- `/Users/philiptullai/Documents/Coding/av/data/recorder.py`

Add fields for payload-specific diagnostics:
- `sync_packet_payload_queue_depth`
- `sync_packet_payload_drop_count`
- `sync_packet_payload_oldest_age_ms`
- `sync_packet_payload_bytes`
- `sync_packet_payload_fallback_reason_code`

Rules:
- keep current `sync_packet_*` continuity fields
- do not overload metadata queue and payload queue into one ambiguous field
- analyzers must distinguish:
  - metadata queue depth
  - payload queue depth

#### PhilViz and analyzer updates required for Phase 3B

Files:
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/verify_recording_run.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/server.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`

Add or update:
- payload queue depth card/row
- payload drop count/rate
- payload fallback reason breakdown
- explicit label for:
  - `metadata shadow`
  - `metadata active`
  - `payload active`

PhilViz must show, for active runs:
- whether the control sample came from a coherent payload
- whether active mode fell back to legacy/latest
- whether packet continuity was inferred from payload ids or degraded fallback

Do not leave operators guessing whether active mode is “real active” or “active with legacy fallback”.

#### Testing plan for Phase 3B

Add or extend tests in:
- `/Users/philiptullai/Documents/Coding/av/tests/test_bridge_sync_packets.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_bridge_client_sync_packets.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_orchestrator_sync_policy.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_control_command_hdf5.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_drive_summary_contract.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_drive_summary_parity.py`

Bridge tests:
- packet payload is published only when image bytes and vehicle state both exist
- `next/raw` consumes FIFO payloads
- `latest/raw` returns the newest payload without consuming
- payload queue overflow increments payload drop counters
- payload bytes budget enforcement increments the right reason code

Client tests:
- multipart parser reconstructs ndarray correctly
- malformed JSON part fails cleanly
- malformed image byte count fails cleanly
- payload reason codes propagate into fallback semantics

Orchestrator tests:
- active payload mode consumes one fetch per control cycle
- active payload mode does not call camera FIFO or vehicle FIFO endpoints on success
- continuity bookkeeping uses payload packet ids / unity frames
- fallback path is explicitly labeled when payload fetch fails

Recorder/analyzer tests:
- new payload diagnostics are written
- old recordings still load
- analyzer and PhilViz agree on payload queue depth, drop rate, and fallback rate

#### Live validation order for Phase 3B

1. Highway ACC scenario:
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`
2. Highway curve catch-up:
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h8_curve_catchup.yml`
3. Hill/highway:
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/hill_g1_grade_following.yml`

Only after those pass:
- one local-road / curve track

#### Acceptance gates for Phase 3B

Nominal highway ACC run must show:
- `packet_mode = packet_fifo_active`
- `packet_completeness_rate >= 99%`
- `fallback_active_rate <= 5%`
- `payload_drop_count = 0` on nominal runs
- materially lower post-jump cooldown rate than shadow baseline
- materially lower effective reference drop rate than shadow baseline
- no transport-caused e-stop or boundary failure

Also require:
- analyzer and PhilViz agree on payload metrics
- `sync_packet_payload_queue_depth` stays bounded and low
- any remaining cooldown episode is attributable to true safety logic, not transport ambiguity

#### Risks to call out explicitly

1. Memory pressure
- active payload queues now hold image bytes, not just metadata
- bounded queue sizes and byte budgets are mandatory

2. HTTP parsing overhead
- `multipart/mixed` parsing adds CPU cost
- this is acceptable only if it removes higher-cost fallback churn and false cooldown behavior

3. Legacy path confusion
- if legacy fallback remains enabled, tools must show when it was used
- otherwise operators will misread runs as “true active” when they were not

4. Top-down / extra sensors
- do not pull top-down into the required active payload in this phase
- keep the first coherent payload limited to front camera + vehicle state
- add optional components only after the core contract is stable

### Phase 3C: Freshness-bounded coherent packet consumption

#### Goal

Keep the coherent payload contract from Phase 3B, but stop making control consume stale packets just because they are coherent.

The active consumer must move from:
- `oldest coherent packet in FIFO order`

to:
- `newest coherent packet within a freshness budget`

This is the required next step because strict FIFO payload mode can still fail even when:
- `packet_completeness_rate = 100%`
- `fallback_active_rate = 0%`

if the consumed packets are several seconds old.

#### Problem statement from live validation

The first coherent-payload active runs showed:
- coherence improved
- fallback collapsed toward zero
- but payload backlog grew large enough that the controller was acting on stale packets

Observed failure pattern:
- payload queue depth stayed high
- payload oldest age reached multi-second values
- skipped Unity frames remained high
- false post-jump cooldown episodes still appeared
- highway run still ended in lateral failure / boundary breach

Conclusion:
- Phase 3B solved **coherence**
- it did not solve **freshness**

The control ingest contract therefore needs two guarantees:
1. coherent sample
2. freshness-bounded sample

#### Contract decision

Closed-loop control must consume:
- the most recent coherent payload packet whose age is within a configurable freshness budget

If multiple stale packets are queued:
- drop them explicitly
- record that they were dropped for staleness
- do not process them one by one

If no coherent payload is fresh enough:
- record explicit `payload_stale_timeout`
- then either:
  - use labeled legacy fallback, or
  - hold-safe policy,
depending on configuration

Do not:
- silently process stale coherent packets
- silently drain to latest without accounting
- silently convert “no fresh packet available” into a normal packet timeout

#### Freshness policy

Introduce an active consume policy with three modes:

1. `fifo_strict`
- existing behavior for debugging
- consume oldest payload packet in queue
- never use as promoted control mode after this phase

2. `freshest_within_budget`
- target promoted mode
- drain queued payload packets up to the newest packet whose age is <= budget
- count all older skipped payload packets explicitly

3. `latest_debug`
- debug-only
- return latest payload snapshot without FIFO accounting
- not for promotion

Recommended default for promoted highway active mode:
- `freshest_within_budget`

Recommended initial freshness budgets:
- `payload_consume_max_age_ms = 250`
- `payload_consume_warn_age_ms = 125`
- `payload_consume_max_drain_count = 128`

Those values should be config-driven, not hardcoded.

#### Bridge server changes

Files:
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`
- `/Users/philiptullai/Documents/Coding/av/sync_contract.py`

Implementation:

Keep the coherent payload queue from Phase 3B, but add:
- packet publish wall time as canonical freshness source
- byte-bounded queue accounting
- stale-drop accounting support

Add server-side helper(s):
- `_snapshot_sync_packet_payload_queue_state(...)`
- optional `_get_latest_fresh_payload(...)` only if freshness filtering is done server-side

Recommended division of responsibility:
- server continues to own packet assembly and payload FIFO queue
- client/orchestrator owns the active consume policy and stale-drain accounting

Reason:
- keeps bridge simple
- keeps policy visible in Python recorder/analyzer
- avoids hiding dropped stale payloads inside the bridge

Add new contract constants in `/Users/philiptullai/Documents/Coding/av/sync_contract.py`:
- `PACKET_CONSUME_POLICY_FIFO_STRICT`
- `PACKET_CONSUME_POLICY_FRESHEST_WITHIN_BUDGET`
- `PACKET_CONSUME_POLICY_LATEST_DEBUG`

Add new fallback / drop reason codes:
- `payload_stale_timeout`
- `payload_stale_drop`
- `payload_age_budget_exceeded`

#### Client changes

Files:
- `/Users/philiptullai/Documents/Coding/av/bridge/client.py`

Implementation:

Keep:
- `get_next_sync_packet_payload_raw()`
- `get_latest_sync_packet_payload_raw()`

Add:
- `get_fresh_sync_packet_payload_raw(...)`

Suggested return shape:
- `payload`
- `drained_payload_count`
- `stale_dropped_count`
- `max_drained_payload_age_ms`
- `selected_payload_age_ms`
- `consume_policy`
- `freshness_budget_ms`
- `fallback_active`
- `fallback_reason_code`

Client algorithm for `freshest_within_budget`:
1. call `get_next_sync_packet_payload_raw()`
2. continue draining while payloads remain available and drain count < budget
3. retain the newest payload seen
4. compute packet age from payload metadata publish wall time
5. mark older drained packets as stale-dropped
6. if newest payload age <= `payload_consume_max_age_ms`, return it
7. otherwise return no active payload with:
   - `fallback_active = True`
   - `fallback_reason_code = payload_stale_timeout`

Important:
- every drained packet must be accounted for
- do not drop stale payloads invisibly
- do not depend on raw queue depth alone; use actual selected packet age

#### Orchestrator changes

Files:
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
- `/Users/philiptullai/Documents/Coding/av/config/av_stack_config.yaml`
- `/Users/philiptullai/Documents/Coding/av/config/acc_highway.yaml`

Implementation:

Replace the current active payload fetch in `_capture_active_sync_packet_inputs()` with policy-driven fetch:
- read `sync_packet_consume_policy`
- read `sync_packet_payload_max_age_ms`
- read `sync_packet_payload_warn_age_ms`
- read `sync_packet_payload_max_drain_count`

Active success path:
1. fetch freshest coherent payload within budget
2. use only the selected payload for this control cycle
3. record selected payload age and drained stale counts
4. compute continuity from the selected payload packet id / Unity frame count

Active degraded path:
1. if no fresh payload available:
   - label fallback as `payload_stale_timeout`
2. if legacy fallback allowed:
   - fetch legacy latest path
   - record fallback provenance explicitly
3. if legacy fallback not allowed:
   - hold-safe / skip cycle as configured

Add config keys:
- `sync_packet_consume_policy`
- `sync_packet_payload_max_age_ms`
- `sync_packet_payload_warn_age_ms`
- `sync_packet_payload_max_drain_count`
- `sync_packet_payload_allow_stale_debug`

Recommended initial config:
- highway active path only:
  - `sync_packet_consume_policy: freshest_within_budget`
  - `sync_packet_payload_max_age_ms: 250`
  - `sync_packet_payload_warn_age_ms: 125`
  - `sync_packet_payload_max_drain_count: 128`

Do not promote this globally until highway gates pass.

#### Recorder/data contract changes

Files:
- `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
- `/Users/philiptullai/Documents/Coding/av/data/recorder.py`

Add payload freshness fields:
- `sync_packet_consume_policy`
- `sync_packet_payload_selected_age_ms`
- `sync_packet_payload_selected_fresh`
- `sync_packet_payload_warn_age_exceeded`
- `sync_packet_payload_stale_drop_count`
- `sync_packet_payload_drained_count`
- `sync_packet_payload_max_drained_age_ms`

Keep existing payload queue fields from Phase 3B:
- `sync_packet_payload_queue_depth`
- `sync_packet_payload_drop_count`
- `sync_packet_payload_oldest_age_ms`
- `sync_packet_payload_bytes`
- `sync_packet_payload_fallback_reason_code`

Contract rule:
- queue depth alone is not authoritative for transport health
- selected payload age is authoritative for active control freshness

#### Analyzer changes

Files:
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/verify_recording_run.py`

Add to `transport_contract` summary:
- selected payload age p50/p95/max
- payload selected-fresh rate
- warn-age exceeded rate
- stale-drop count max
- drained count p95/max
- max drained payload age p95
- consume policy mode

Add causal attribution:
- distinguish:
  - `coherent_and_fresh`
  - `coherent_but_stale`
  - `fallback_due_to_missing_payload`
  - `fallback_due_to_stale_timeout`

Update brake/reference-drop attribution:
- if post-jump cooldown coincides with stale selected payload age or stale-timeout fallback,
  classify as transport freshness fault, not only generic teleport guard

#### PhilViz changes

Files:
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/server.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`

Add to `Transport & Contracts` card:
- consume policy
- selected payload age p50/p95/max
- selected-fresh rate
- stale-drop count
- drained payload count
- max drained payload age
- warn-age exceeded rate

Add frame-level sync fields:
- `sync_packet_consume_policy`
- `sync_packet_payload_selected_age_ms`
- `sync_packet_payload_selected_fresh`
- `sync_packet_payload_warn_age_exceeded`
- `sync_packet_payload_stale_drop_count`
- `sync_packet_payload_drained_count`
- `sync_packet_payload_max_drained_age_ms`

Add a new timeline overlay or diagnostics row:
- `selected_payload_age_ms`
- `stale_drop_event`
- `fallback_reason`

The operator must be able to answer:
- “did we consume a coherent packet?”
- “was it fresh enough?”
- “how many stale packets were skipped to get it?”

#### Testing plan for Phase 3C

Add or extend tests in:
- `/Users/philiptullai/Documents/Coding/av/tests/test_bridge_client_sync_packets.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_orchestrator_sync_policy.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_control_command_hdf5.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_drive_summary_contract.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_drive_summary_latency_sync.py`

Client tests:
- fresh payload within age budget is selected
- multiple stale payloads are drained and counted
- stale timeout produces explicit fallback reason
- drain count cap is respected

Orchestrator tests:
- active mode uses `freshest_within_budget` when configured
- selected payload age fields are written into `vehicle_state_dict`
- legacy fallback is labeled `payload_stale_timeout` when no fresh payload exists
- strict FIFO mode remains available only for debug tests

Recorder tests:
- new selected-age and stale-drop fields are persisted to HDF5

Analyzer tests:
- transport summary includes freshness fields
- old recordings still show `UNAVAILABLE` cleanly
- analyzer/PhilViz parity on freshness metrics

#### Live validation order for Phase 3C

1. Highway ACC nominal:
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`
2. Highway catch-up / curve:
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h8_curve_catchup.yml`
3. Hill/highway:
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/hill_g1_grade_following.yml`

Only after highway passes:
- rerun one local-road / curve track for regression check

#### Acceptance gates for Phase 3C

Nominal highway active run must show:
- `packet_mode = packet_fifo_active`
- `sync_packet_consume_policy = freshest_within_budget`
- `packet_completeness_rate >= 99%`
- `fallback_active_rate <= 5%`
- `payload_selected_fresh_rate >= 95%`
- `selected_payload_age_ms p95 <= payload_consume_max_age_ms`
- `stale_drop_count > 0` is acceptable only if selected payload freshness stays in budget
- materially lower false cooldown rate than strict FIFO payload baseline
- materially lower effective reference drop rate than strict FIFO payload baseline
- no transport-caused boundary failure / e-stop

Strict rejection conditions:
- payload oldest age multi-second backlog persists while selected payload age is also stale
- fallback collapses to zero but stale packet age remains high
- active control still processes packets that exceed the configured age budget

#### Risks and tradeoffs

1. This is intentionally closer to “use the newest” than strict FIFO.
- that is correct
- but only because the selected item is now a coherent packet, not separate latest snapshots

2. Over-aggressive draining can hide throughput problems if not recorded.
- stale-drop accounting is mandatory

3. Too-tight age budgets can cause excessive fallback on slower hardware.
- that is why warn-age and hard-age must both be recorded

4. Payload queue depth may remain high even if selected payload freshness is good.
- selected payload age is the primary health metric
- queue depth remains secondary context

5. Command transport remains separate
- this phase fixes ingest continuity, not command apply/ack continuity
- command transport contract work still remains a separate patch set

### Phase 3D: Server-side freshness-bounded coherent packet selection

#### Goal

Keep the coherent payload contract from Phase 3B and the freshness semantics from Phase 3C, but move the freshness-selection work out of the client loop and into the bridge.

The active control loop must make:
- one request per cycle
- for one coherent packet payload
- already selected to satisfy the configured freshness policy

This phase exists because Phase 3C proved:
1. the control contract was correct
2. the placement was wrong

The client-side drain loop fixed stale-packet behavior, but it collapsed loop throughput by doing repeated payload fetches and repeated multipart parsing inside one control cycle.

#### Problem statement from live validation

The Phase 3C live run showed:
- false cooldowns dropped to effectively zero
- effective reference drops collapsed
- safety recovered
- but loop rate fell to about `1.62 Hz`

Observed pattern:
- selected payload age was good
- stale coherent packets were being dropped correctly
- but front queue saturation returned
- packet completeness dropped
- fallback rose
- the control loop spent too much wall time draining payloads in Python

Conclusion:
- the freshness policy must stay
- the client-side drain implementation must not

#### Contract decision

The bridge must return one authoritative coherent payload response for active control:
- selected according to the configured consume policy
- with freshness/drain accounting included in the same response

The active client must no longer:
- poll multiple payload packets per cycle
- reconstruct freshness policy in Python
- do repeated multipart parsing to chase a fresh packet

The client should instead:
1. issue one request
2. receive one selected coherent payload
3. receive the server's freshness-selection accounting
4. execute control on that one selected payload

#### API design

Files:
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`
- `/Users/philiptullai/Documents/Coding/av/bridge/client.py`
- `/Users/philiptullai/Documents/Coding/av/sync_contract.py`

Recommended primary endpoint:
- `GET /api/sim/packet/next/raw`

Recommended behavior change:
- keep the endpoint path stable
- add query parameters that define consume policy and freshness budget

Required query parameters:
- `consume_policy`
- `max_age_ms`
- `warn_age_ms`
- `max_drain_count`

Recommended supported values:
- `fifo_strict`
- `freshest_within_budget`
- `latest_debug`

Alternative acceptable design:
- add `/api/sim/packet/next/raw/selected`

Use that alternative only if the existing endpoint would become too ambiguous.

#### Bridge server changes

Files:
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`
- `/Users/philiptullai/Documents/Coding/av/sync_contract.py`

Implementation:

Keep the coherent payload queue from Phase 3B.

Move the freshness-selection loop into the server:
1. read the requested consume policy and age budget
2. inspect queued coherent payload packets
3. drain stale packets server-side up to `max_drain_count`
4. select the newest packet that satisfies the freshness policy
5. return the selected payload plus selection accounting in one response

Add a dedicated selector helper, for example:
- `_select_sync_packet_payload(...)`

Server selection responsibilities:
- compute selected payload age from packet publish wall time
- count drained payloads
- count stale-dropped payloads
- compute max drained payload age
- classify freshness failure reason
- distinguish:
  - queue empty
  - timeout waiting for coherent payload
  - coherent payload available but stale
  - drain cap exceeded

Required response metadata:
- `consume_policy`
- `selected_packet_id`
- `selected_unity_frame_count`
- `selected_payload_age_ms`
- `selected_payload_fresh`
- `selected_payload_warn_age_exceeded`
- `drained_payload_count`
- `stale_dropped_count`
- `max_drained_payload_age_ms`
- `selection_fallback_active`
- `selection_fallback_reason_code`
- `server_queue_depth_after_select`
- `server_oldest_payload_age_ms_after_select`

The selected payload response must still include:
- raw front image bytes
- vehicle state payload
- coherent packet metadata

The critical architectural rule is:
- one server request returns both the selected payload and the accounting for how it was selected

#### Client changes

Files:
- `/Users/philiptullai/Documents/Coding/av/bridge/client.py`

Implementation:

Demote the current client-side `freshest_within_budget` drain helper to:
- debug-only
- or remove it after Phase 3D is stable

Add or update a single-shot fetch helper, for example:
- `get_selected_sync_packet_payload_raw(...)`

Client responsibilities:
1. issue one request with policy parameters
2. parse one multipart response
3. return selected payload and server-supplied accounting unchanged
4. do no additional drain loop in Python

Client must treat server freshness accounting as source of truth for:
- selected payload age
- stale-drop count
- drained count
- fallback reason

#### Orchestrator changes

Files:
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
- `/Users/philiptullai/Documents/Coding/av/config/av_stack_config.yaml`
- `/Users/philiptullai/Documents/Coding/av/config/acc_highway.yaml`

Implementation:

Replace client-side drain-based active fetch in `_capture_active_sync_packet_inputs()` with:
- one selected coherent payload request per control cycle

Configuration stays the same at the orchestrator level:
- `sync_packet_consume_policy`
- `sync_packet_payload_max_age_ms`
- `sync_packet_payload_warn_age_ms`
- `sync_packet_payload_max_drain_count`

But those fields are now passed through to the bridge selector rather than applied in the client.

Active success path:
1. request selected coherent payload
2. use selected payload for control
3. record server selection accounting into `vehicle_state_dict`
4. compute continuity only from the selected payload metadata

Active degraded path:
1. if server reports `selection_fallback_active`
   - record the exact reason
2. if configured, use explicit legacy fallback
3. otherwise hold-safe / skip cycle

Orchestrator must not:
- make extra payload requests to chase freshness
- reinterpret server stale-drop accounting
- silently downgrade a stale-timeout to generic packet timeout

#### Recorder/data contract changes

Files:
- `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
- `/Users/philiptullai/Documents/Coding/av/data/recorder.py`

Keep the Phase 3C freshness fields.

Add or clarify provenance fields so it is explicit these came from server-side selection:
- `sync_packet_payload_selection_source`
- `sync_packet_payload_selection_fallback_active`
- `sync_packet_payload_selection_fallback_reason_code`
- `sync_packet_payload_server_queue_depth_after_select`
- `sync_packet_payload_server_oldest_age_ms_after_select`

Contract rule:
- selected payload age remains the authoritative freshness metric
- queue depth remains supporting context
- selection provenance must say whether freshness policy was satisfied by active packet mode or by fallback

#### Analyzer changes

Files:
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/verify_recording_run.py`

Add transport summary fields for Phase 3D:
- selection source mode
- selected payload age p50/p95/max
- selected fresh rate
- stale-drop count max
- drained count p95/max
- queue depth after select p95/max
- oldest payload age after select p95/max
- fallback reason breakdown from server-side selection

Add causal attribution:
- `coherent_and_fresh_selected_server_side`
- `coherent_but_stale_server_side`
- `selection_fallback_due_to_stale_timeout`
- `selection_fallback_due_to_empty_queue`
- `selection_fallback_due_to_drain_cap`

Add an explicit throughput verdict:
- if freshness metrics are good but loop rate collapses, classify as control-loop throughput failure
- if loop rate is good but selected payload age is bad, classify as transport freshness failure

#### PhilViz changes

Files:
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/server.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`

Extend the `Transport & Contracts` card with:
- selection source
- selected payload age p50/p95/max
- selected-fresh rate
- stale-dropped count
- drained count
- queue depth after select
- oldest payload age after select
- selection fallback reason histogram

Add frame-level fields:
- `sync_packet_payload_selection_source`
- `sync_packet_payload_selection_fallback_active`
- `sync_packet_payload_selection_fallback_reason_code`
- `sync_packet_payload_server_queue_depth_after_select`
- `sync_packet_payload_server_oldest_age_ms_after_select`

Add timeline overlays:
- `selected_payload_age_ms`
- `stale_drop_event_count`
- `server_queue_depth_after_select`
- `selection_fallback_reason`

The operator must be able to answer:
1. did the bridge return a coherent packet?
2. was it fresh?
3. how many packets were dropped server-side to get it?
4. did the control loop still fall behind anyway?

#### Testing plan for Phase 3D

Add or extend tests in:
- `/Users/philiptullai/Documents/Coding/av/tests/test_bridge_sync_packets.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_bridge_client_sync_packets.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_orchestrator_sync_policy.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_control_command_hdf5.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_drive_summary_contract.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_drive_summary_latency_sync.py`

Bridge server tests:
- selects freshest coherent payload within budget
- drains stale packets and returns accounting
- returns explicit stale-timeout reason when only stale payloads exist
- respects drain-count cap
- reports queue depth and oldest age after selection correctly

Client tests:
- parses one selected-payload response correctly
- does not loop-fetch in active mode
- preserves server selection accounting unchanged

Orchestrator tests:
- active mode performs one payload request per cycle
- selection accounting is written into `vehicle_state_dict`
- fallback path preserves server reason codes

Recorder/analyzer tests:
- new selection provenance fields persist to HDF5
- analyzer summary includes server-side selection metrics
- legacy recordings still show `UNAVAILABLE` cleanly
- PhilViz/backend parity holds on selection metrics

#### Live validation order for Phase 3D

1. Highway ACC nominal:
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`
2. Highway catch-up / curve:
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h8_curve_catchup.yml`
3. Hill/highway:
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/hill_g1_grade_following.yml`

Run each in:
1. shadow packet mode baseline
2. Phase 3D active selected-payload mode

Do not change teleport guard semantics in this phase.
Do not change command transport in this phase.

#### Acceptance gates for Phase 3D

Promotable highway active run must show:
- `packet_mode = packet_fifo_active`
- `sync_packet_consume_policy = freshest_within_budget`
- one payload request per control cycle in active mode
- `packet_completeness_rate >= 95%`
- `fallback_active_rate <= 5%`
- `payload_selected_fresh_rate >= 95%`
- `selected_payload_age_ms p95 <= payload_consume_max_age_ms`
- materially lower loop-time overhead than the Phase 3C client-drain run
- control loop frequency materially above the Phase 3C `1.62 Hz` failure
- no transport-caused false cooldown episodes
- no transport-caused boundary failure / e-stop

Strict rejection conditions:
- selected payload age is good but control loop rate still collapses badly
- server-side selection still requires repeated client fetches
- queue depth after select remains near full while selected payload age and loop rate are both poor

#### Risks and tradeoffs

1. Server-side freshness selection concentrates more work in the bridge.
- that is acceptable if it removes repeated client parsing and preserves loop rate

2. If multipart response handling is the dominant cost, server-side selection alone may not be enough.
- that is the escalation gate for a lower-overhead transport

3. Queue-depth semantics become more nuanced.
- queue depth after selection is more useful than raw queue depth before selection

4. Legacy fallback can still hide real transport failures if overused.
- fallback reason reporting remains mandatory

5. If Phase 3D still cannot meet cadence targets with one-request-per-cycle semantics, the next step is not another client/server selection tweak.
- the next step is a lower-overhead active transport for image payloads, such as shared memory or a binary streaming channel

### Phase 3E: Upstream coherent packet completeness hardening

#### Goal

Keep the Phase 3D server-side freshness selector and target the next real failure:
- coherent packet freshness is now acceptable
- loop throughput is back near target
- but coherent packet completeness is still too low
- fallback is still too high
- `packet_timeout` and orphan counts still dominate

This phase exists to raise:
- `packet_completeness_rate`
- selected-fresh availability
- control-path stability

without reintroducing stale payload processing.

#### Problem statement from live validation

The Phase 3D highway run showed:
- selected payload age p95 stayed in budget
- server queue after select stayed near zero
- false cooldown rate dropped materially
- loop rate recovered from the Phase 3C `1.62 Hz` failure

But it also showed:
- `packet_completeness_rate ≈ 65%`
- `fallback_active_rate ≈ 35%`
- fallback mode still dominated by `packet_timeout`
- orphan counts were still large
- skipped Unity frames stayed high

Observed consequence:
- transport was no longer the primary source of stale control samples
- but the control loop still spent a large fraction of frames outside the coherent packet path
- later in the drive, behavior degraded and the run still failed

#### Architectural diagnosis

The remaining problem is upstream contract fragility:
- front camera upload and vehicle-state upload are still independent HTTP producers
- the bridge is still assembling coherent packets after the fact by joining on `unity_frame_count`
- that join is good enough to prove freshness semantics
- it is not good enough yet to guarantee high completeness under load

Current exact-match join path:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CameraCapture.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/AVBridge.cs`
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`

This phase therefore targets:
1. source-side packet coherence
2. bounded join-window behavior
3. explicit completion / miss accounting

Not:
1. another freshness-policy tweak
2. another fallback heuristic
3. another teleport-guard change

#### Contract decision

The preferred promoted contract is:
- Unity publishes packet components with a shared packet key and shared capture intent
- the bridge assembles complete coherent packets with bounded ambiguity
- active control still consumes one selected coherent payload per cycle

Minimum acceptable improvement path:
1. harden the existing split uploads with a better source-side key and bounded join window
2. expose precise completion / miss accounting

Preferred long-term path:
1. move toward Unity publishing a true frame bundle contract for front camera + vehicle state
2. reduce bridge-side orphan/join ambiguity as much as possible

#### Recommended implementation approach

##### Stage 3E.1: Harden the source-side packet key and timing contract

Files:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CameraCapture.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/AVBridge.cs`
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`
- `/Users/philiptullai/Documents/Coding/av/sync_contract.py`

Implementation:

Add a source-generated packet key that is stronger than bare `unity_frame_count`.

Recommended fields:
- `packet_key`
- `unity_frame_count`
- `unity_time`
- `capture_wall_clock_ms` or equivalent source monotonic time
- `capture_sequence_id`

Requirements:
1. camera and vehicle state created for the same simulation step must carry the same `packet_key`
2. the bridge must use `packet_key` as primary join key
3. `unity_frame_count` remains a diagnostic / continuity field, not the only join key

Why:
- this removes ambiguity from frame-count reuse / drift
- this lets the bridge distinguish “late partner for same packet” from “different sample”

##### Stage 3E.2: Add a bounded join window instead of open-ended exact-match waiting

Files:
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`

Implementation:

For each partial packet:
1. start a bounded join window when the first component arrives
2. wait only within that window for the missing component
3. classify the outcome explicitly:
   - complete packet
   - camera-only miss
   - vehicle-only miss
   - timed-out partial packet
   - superseded by newer packet

Recommended server fields:
- `join_window_ms`
- `join_completed`
- `join_timeout`
- `join_superseded`
- `join_missing_component`
- `join_wait_time_ms`

Contract rule:
- the bridge must stop treating every missing match as the same generic timeout
- completion and miss modes must be separable in diagnostics

##### Stage 3E.3: Prefer source-side pairing over bridge-side rescue

Files:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CameraCapture.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/AVBridge.cs`
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`

Implementation:

Do not yet move full raw-image bundling into the Unity-side vehicle-state post in this phase.

Instead:
1. ensure both producers stamp the same packet key
2. ensure both producers publish the same step identity
3. align producer timing so the bridge sees matching pairs within a bounded window

If Stage 3E.1 + 3E.2 still cannot raise completeness enough, the next escalation is:
- a true source-side bundled publish endpoint

That escalation should be held until this cheaper contract hardening is measured.

#### Bridge server changes

Files:
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`

Implementation details:

1. Replace `pending_packets_by_unity_frame` primary indexing with:
   - `pending_packets_by_packet_key`

2. Keep secondary indexing for diagnostics:
   - `unity_frame_count`
   - `packet_id`

3. Add packet-part arrival metadata:
   - first-arrival time
   - component arrival deltas
   - join wait time

4. Add bounded eviction reasons:
   - `orphan_camera_timeout`
   - `orphan_vehicle_timeout`
   - `partial_superseded`
   - `join_window_exceeded`

5. Add summary counters:
   - `sync_packet_complete_count`
   - `sync_packet_partial_count`
   - `sync_packet_join_timeout_count`
   - `sync_packet_superseded_count`
   - `sync_packet_missing_camera_count`
   - `sync_packet_missing_vehicle_count`

6. Preserve the Phase 3D selector on top of the improved complete-packet queue.

#### Unity-side producer changes

Files:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CameraCapture.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/AVBridge.cs`

Implementation details:

1. Generate one step-level packet identity per simulation/update cycle.
2. Stamp camera upload and vehicle-state upload with the same identity.
3. Record source-side queue pressure separately for:
   - camera upload queue
   - vehicle-state send cadence
4. Add source-side counters:
   - camera packet stamped count
   - vehicle packet stamped count
   - source packet key mismatch count
   - source packet publish lag

Important:
- do not silently generate independent ids in each producer
- the identity must come from one shared step contract

#### Client and orchestrator changes

Files:
- `/Users/philiptullai/Documents/Coding/av/bridge/client.py`
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`

Implementation details:

Client:
- no major policy change
- continue one selected payload request per cycle
- expose the new completion/miss metadata returned by the bridge

Orchestrator:
- record the new join/completeness provenance
- distinguish:
  - no coherent packet available
  - coherent packet unavailable because source pair never completed
  - coherent packet unavailable because join window expired

This distinction matters because it tells us whether to fix:
- source production timing
- bridge join semantics
- or downstream fallback policy

#### Recorder/data contract changes

Files:
- `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
- `/Users/philiptullai/Documents/Coding/av/data/recorder.py`

Add fields for:
- `sync_packet_key`
- `sync_packet_join_wait_ms`
- `sync_packet_join_completed`
- `sync_packet_join_timeout`
- `sync_packet_join_superseded`
- `sync_packet_missing_camera_count`
- `sync_packet_missing_vehicle_count`
- `sync_packet_complete_count`
- `sync_packet_partial_count`
- `sync_packet_join_timeout_count`
- `sync_packet_superseded_count`

Contract rule:
- completeness must be diagnosable from explicit join outcomes
- not inferred only from fallback rate

#### Analyzer changes

Files:
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/verify_recording_run.py`

Add to `transport_contract`:
- join-completed rate
- join-timeout rate
- join-superseded rate
- missing-camera rate
- missing-vehicle rate
- join-wait p50/p95/max
- completeness source mode

Add causal attribution:
- `fallback_due_to_incomplete_source_pair`
- `fallback_due_to_join_timeout`
- `fallback_due_to_join_supersession`

Add correlation checks:
- whether lateral/perception degradation spikes coincide with:
  - completeness collapse
  - join timeout bursts
  - fallback bursts

This is where the late-drive swerving should be evaluated after transport hardening:
- not as a separate guess first
- but as a correlated symptom against transport completeness and fallback episodes

#### PhilViz changes

Files:
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/server.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`

Extend `Transport & Contracts` with:
- join-completed rate
- join-timeout rate
- join-superseded rate
- missing-camera / missing-vehicle rates
- join wait p95
- completeness source mode

Add timeline rows:
- `join_timeout_event`
- `join_superseded_event`
- `missing_camera_event`
- `missing_vehicle_event`
- `fallback_reason`

Add one explicit correlation view:
- transport fallback vs lateral error / stale perception / out-of-lane markers

The operator should be able to answer:
1. did the packet fail because the source pair never formed?
2. did fallback bursts line up with swerving?
3. was the late-drive instability downstream of completeness loss or independent of it?

#### Testing plan for Phase 3E

Add or extend tests in:
- `/Users/philiptullai/Documents/Coding/av/tests/test_bridge_sync_packets.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_bridge_client_sync_packets.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_orchestrator_sync_policy.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_control_command_hdf5.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_drive_summary_contract.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_drive_summary_latency_sync.py`

Recommended new focused test:
- `/Users/philiptullai/Documents/Coding/av/tests/test_sync_packet_source_completeness.py`

Test cases:
1. matching packet key from camera + vehicle creates complete packet
2. mismatched packet key does not join
3. join timeout is recorded explicitly
4. superseded partial packet is recorded explicitly
5. server selector still returns one request / one payload
6. recorder persists new join/completeness provenance
7. analyzer reports join/completeness metrics correctly

#### Live validation order for Phase 3E

1. Highway ACC nominal:
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`
2. Highway catch-up / curve:
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h8_curve_catchup.yml`
3. Hill/highway:
   - `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/hill_g1_grade_following.yml`

Validation comparisons:
1. Phase 3D active selected-payload baseline
2. Phase 3E hardened completeness path

Required comparisons:
- packet completeness
- fallback rate
- join-timeout rate
- join-superseded rate
- false cooldown rate
- effective reference drop rate
- out-of-lane / failure
- late-drive swerving correlation

#### Acceptance gates for Phase 3E

Promotable highway active run must show:
- `packet_completeness_rate >= 90%` in first pass, target `>= 95%`
- `fallback_active_rate <= 10%` in first pass, target `<= 5%`
- selected payload age p95 still within budget
- join-timeout and superseded rates materially below the Phase 3D baseline
- false cooldown rate materially below the Phase 3D baseline
- no transport-caused boundary failure / e-stop

Additional correlation gate:
- if late-drive swerving still appears, it must be shown whether it coincides with transport fallback bursts
- if it does not, treat swerving as a separate follow-on control/perception issue

Strict rejection conditions:
- completeness stays in the current ~65% band
- fallback remains dominated by `packet_timeout`
- join metrics are unavailable or ambiguous
- transport fixes mask the swerving question instead of clarifying it

#### Execution status: implemented and measured

Implementation landed across:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CameraCapture.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/AVBridge.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CarController.cs`
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
- `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
- `/Users/philiptullai/Documents/Coding/av/data/recorder.py`
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/verify_recording_run.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/server.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`

Focused validation passed:
- `29` targeted tests

Live measurement:
- previous Phase 3D server-selection baseline:
  - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260325_225113.h5`
- Phase 3E result:
  - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260325_232219.h5`

Measured outcome:
- `packet_completeness_rate`: `65.2% -> 90.5%`
- `fallback_active_rate`: `34.8% -> 9.5%`
- `join_source_mode`: `packet_key`
- `join_key_present_rate`: `90.5%`
- selected payload age p95 stayed in budget: `72.8 ms`
- run stayed in bounds and finished safely with score `97.0`

What is still wrong:
- false transport-linked cooldown remains high:
  - `false_teleport_cooldown_rate ≈ 20.9%`
  - `post_jump_cooldown_active_rate ≈ 23.3%`
- effective reference drops remain high:
  - `31.3%`
- fallback is still dominated by `packet_timeout`
- late oscillation still appears in the highway run

Conclusion:
- Phase 3E improved completeness enough to prove the source-side packet identity was the right direction.
- Phase 3E did not resolve the remaining highway instability.
- The next bottleneck is no longer packet freshness or packet identity.
- The next bottleneck is the teleport/post-jump continuity path, with a secondary follow-up on residual packet timeouts.

Correlation note for the late-drive swerving:
- in `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260325_232219.h5`, late high-lateral-error frames overlapped:
  - fallback on `16.7%` of those frames
  - post-jump cooldown on `44.4%`
  - either fallback or cooldown on `55.6%`
- perception confidence stayed nominal in those frames

Interpretation:
- the late swerving is only partly explained by residual fallback
- it aligns more strongly with post-jump cooldown / forced reset behavior than with packet incompleteness itself
- that means the next phase should focus on teleport guard demotion/redesign before treating the swerving as a purely lateral-controller issue

#### Risks and tradeoffs

1. This phase increases source-side and bridge-side contract complexity.
- that is justified only because the current split-producer exact-match contract is too fragile

2. If Unity-side packet identity is not truly shared, the bridge diagnostics will look better than the actual contract.
- source-generated shared packet identity is mandatory

3. If completeness still does not rise after Stage 3E.1 and 3E.2, the next escalation is a true source-side bundled publish path.
- do not over-invest in join heuristics beyond that point

4. Late-drive swerving may still remain after transport completeness is fixed.
- that is acceptable
- but this phase must make it measurable whether swerving is caused by transport fallback or not

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

Execution status: implemented and measured

- previous Phase 3E baseline:
  - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260325_232219.h5`
- Phase 4 result:
  - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_085015.h5`

Measured outcome:
- `post_jump_cooldown_active_rate`: `23.3% -> 0.0%`
- `false_teleport_cooldown_rate`: `20.9% -> 0.0%`
- `effective_reference_velocity_drop_rate`: `31.3% -> 4.0%`
- no forced MPC→PP reset remained; MPC stayed active for `94.1%` of frames

What improved:
- the teleport/post-jump redesign did the intended job
- the guard is no longer misclassifying normal transport continuity as teleport
- late high-lateral-error frames no longer overlap cooldown; fallback overlap in the late third dropped to `5.7%`

What is still wrong:
- the run is not promotable
- overall score dropped to `79.0`
- ACC/following behavior became the dominant problem:
  - near-miss events: `1`
  - minimum gap observed: `0.1 m`
  - ACC jerk P95: `53.9 m/s^3`
- residual packet timeout fallback is still nontrivial:
  - `packet_completeness_rate`: `89.0%`
  - `fallback_active_rate`: `11.0%`

Conclusion:
- Phase 4 resolved the false teleport / forced reset bottleneck.
- The late swerving seen before should no longer be treated as a transport-cooldown artifact on this path.
- The next bottleneck is now highway lateral/following behavior under sustained MPC plus the remaining packet-timeout fallback tail.

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

### Phase 3F: Atomic source packet-key ownership and timeout-tail hardening

Phase 3E materially improved completeness, but the remaining timeout tail shows that the source-side packet identity is still not atomic. The current Unity side still lets vehicle state read the camera's latest packet key rather than emitting one per-update packet identity owned by the same simulation tick.

This phase hardens the remaining timeout tail by moving packet identity ownership to the source update itself and making join failures directly attributable.

#### Goal

Keep the good parts of Phase 3D and Phase 3E:

- coherent selected payload
- server-side freshness selection
- explicit transport diagnostics

Replace the remaining weak point:

- vehicle state referencing `LatestSyncPacketKey`

with:

- one source-owned packet id minted once per AVBridge update
- same id stamped on both vehicle state and camera upload for that update

#### Stage 3F.1: Mint packet key at the update owner

Files:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/AVBridge.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CameraCapture.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CarController.cs`

Implementation checklist:
- [ ] Move packet-key ownership to `AVBridge.cs` at the start of each send/update cycle
- [ ] Mint one packet id/key per bridge update using:
  - update sequence
  - unity frame count
  - optional monotonic tie-breaker if needed
- [ ] Pass that exact key into the camera capture/send path
- [ ] Stamp that same key directly into the vehicle-state payload for the same update
- [ ] Stop treating `cameraCapture.LatestSyncPacketKey` as the authoritative vehicle-state source
- [ ] Keep camera-side debug/latest fields if useful, but demote them to observability only

Required source metadata:
- [ ] `source_packet_key`
- [ ] `source_packet_owner = avbridge_update`
- [ ] `source_packet_camera_scheduled`
- [ ] `source_packet_vehicle_scheduled`
- [ ] `source_packet_camera_sent`
- [ ] `source_packet_vehicle_sent`

#### Stage 3F.2: Distinguish packet-key presence from join success

Files:
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`
- `/Users/philiptullai/Documents/Coding/av/sync_contract.py`

Implementation checklist:
- [ ] Add explicit join-failure reason codes instead of collapsing everything into `packet_timeout`
- [ ] Distinguish:
  - `camera_key_missing`
  - `vehicle_key_missing`
  - `packet_key_mismatch`
  - `camera_component_late`
  - `vehicle_component_late`
  - `join_window_expired`
  - `superseded_camera`
  - `superseded_vehicle`
- [ ] Join by packet key only for active mode
- [ ] Keep Unity-frame fallback only for shadow/debug paths and label it as such
- [ ] Emit a per-selected-frame reason field, not just cumulative timeout counters

#### Stage 3F.3: Record direct per-frame timeout attribution

Files:
- `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
- `/Users/philiptullai/Documents/Coding/av/data/recorder.py`
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`

Implementation checklist:
- [ ] Add recorder fields for:
  - `sync_packet_selection_result`
  - `sync_packet_join_failure_reason`
  - `sync_packet_join_failure_side`
  - `sync_packet_source_key_present_camera`
  - `sync_packet_source_key_present_vehicle`
  - `sync_packet_selected_packet_key`
  - `sync_packet_selected_packet_id`
  - `sync_packet_timeout_event_delta`
- [ ] Preserve cumulative orphan/timeout counters, but do not rely on them as the only evidence
- [ ] Ensure every fallback control frame has one direct reason field explaining why it fell back

#### Stage 3F.4: Debug tool and analyzer upgrades

Files:
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/verify_recording_run.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/server.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/backend/triage_engine.py`

Implementation checklist:
- [ ] Add a `Packet Join Contract` summary showing:
  - source key present rate by side
  - join success rate
  - fallback selected-frame rate
  - failure reason split
  - failure side split
- [ ] Add timeline/event markers for:
  - key missing on camera side
  - key missing on vehicle side
  - key mismatch
  - join-window expiry
  - selected fallback frame
- [ ] Update triage so it can state:
  - `packet_timeout` due to camera-side missing key
  - `packet_timeout` due to vehicle-side missing key
  - not just “timeout count high”
- [ ] Keep cumulative timeout/orphan charts, but visually demote them behind per-frame selected-fallback attribution

#### Stage 3F.5: Tests

Add or update:
- `/Users/philiptullai/Documents/Coding/av/tests/test_bridge_sync_packets.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_bridge_client_sync_packets.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_orchestrator_sync_policy.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_control_command_hdf5.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_drive_summary_contract.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_visualizer_transport_parity.py`

Exact test checklist:
- [ ] matching source-owned packet key from camera + vehicle joins successfully
- [ ] vehicle key missing yields `join_failure_side=vehicle` and `join_failure_reason=vehicle_key_missing`
- [ ] camera key missing yields `join_failure_side=camera` and `join_failure_reason=camera_key_missing`
- [ ] mismatched key does not join and is labeled `packet_key_mismatch`
- [ ] selected fallback frame writes direct reason fields into HDF5
- [ ] CLI analyzer and PhilViz report the same reason split and side split
- [ ] legacy recordings still load with fields marked unavailable

#### Live validation order for Phase 3F

1. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`
2. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h8_curve_catchup.yml`
3. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/hill_g1_grade_following.yml`

For each, compare against the current Phase 4 baseline:
- `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_085015.h5`

#### Acceptance gates for Phase 3F

1. Source key present rate:
- camera `100%`
- vehicle `100%`

2. Selected fallback rate is materially below the current `11%` tail on the highway ACC baseline.

3. No selected fallback frame is unlabeled.

4. Joined packets remain fresh:
- selected payload age stays within current budget

5. Analyzer and PhilViz can identify whether the remaining timeout tail is:
- camera-side
- vehicle-side
- key mismatch
- join-window expiry

6. Cumulative timeout churn may remain nonzero, but direct selected-frame fallback attribution must be accurate enough that control-path loss is no longer inferred indirectly.
