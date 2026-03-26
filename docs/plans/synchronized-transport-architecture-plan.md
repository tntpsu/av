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


### Phase 3G: Same-tick packet bundling and coherence-gated selection

Phase 3F improved packet-key completeness, but the validation run showed a more serious contract failure: packets were marked `complete` while the front camera and vehicle state were still separated by roughly `0.63-0.75 s` and `37-45` Unity frames. That is not a fallback problem. It is a packet coherence problem.

This phase fixes that by making the source-side packet key represent one bounded update/capture instant, then refusing to use packets for active control when they are key-complete but time-incoherent.

#### Goal

Keep:
- source-owned packet keys
- server-side freshness selection
- direct join-failure attribution from Phase 3F

Add the missing contract:
- one packet key must represent one same-tick bundle, not a queued future camera frame
- active selection must require coherence, not just completeness

#### Root cause to fix

The current Unity-side flow is:
1. `AVBridge` mints a key and stamps vehicle state immediately.
2. `CameraCapture` queues that key in `pendingSyncPacketContexts`.
3. A later camera capture dequeues the oldest context and stamps that old key onto a newer frame.

That makes key matching succeed while temporal coherence fails.

Validation evidence from `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_111746.h5`:
- `sync_front_vehicle_time_delta_ms p50/p95/max = 627.3 / 716.8 / 750.1`
- `sync_front_vehicle_frame_delta p50/p95/max = 37 / 43 / 45`
- `sync_packet_join_wait_ms p50/p95/max = 659.1 / 749.2 / 784.4`
- packet join source was still `packet_key`

That is the regression this phase must prevent.

#### Stage 3G.1: Make packet keys represent one bounded source tick

Files:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/AVBridge.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CameraCapture.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CarController.cs`

Implementation checklist:
- [ ] Replace the unbounded FIFO `pendingSyncPacketContexts` handoff with a same-tick packet context contract.
- [ ] `AVBridge.cs` should mint one packet context per update and attach:
  - `packet_key`
  - `owner_update_id`
  - `owner_unity_frame_count`
  - `owner_unity_time`
  - `created_realtime`
- [ ] `CameraCapture.cs` must not consume an arbitrarily old queued context.
- [ ] Add a bounded context match rule before using a context for a captured frame:
  - `abs(captureUnityFrameCount - ownerUnityFrameCount) <= camera_context_max_frame_delta`
  - `abs(captureUnityTime - ownerUnityTime) <= camera_context_max_time_delta_s`
- [ ] If the oldest queued context violates the budget, drop it as stale and record a source-side stale-context counter.
- [ ] If no valid context exists for the current capture, either:
  - skip stamping a key and record `camera_context_missing_for_capture`, or
  - explicitly mark the capture as unbundled debug-only
- [ ] Do not allow a packet key to survive across dozens of later camera frames.

Recommended default budgets for the first pass:
- `camera_context_max_frame_delta = 2`
- `camera_context_max_time_delta_s = 0.050`

Required new source diagnostics:
- [ ] `source_packet_context_queue_depth`
- [ ] `source_packet_context_dropped_stale_count`
- [ ] `source_packet_context_missing_count`
- [ ] `source_packet_context_frame_delta`
- [ ] `source_packet_context_time_delta_ms`

#### Stage 3G.2: Add bridge-side coherence classification

Files:
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`
- `/Users/philiptullai/Documents/Coding/av/sync_contract.py`

Implementation checklist:
- [ ] Add explicit coherence reason codes separate from join-failure codes:
  - `coherent`
  - `front_vehicle_time_delta_budget_exceeded`
  - `front_vehicle_frame_delta_budget_exceeded`
  - `join_wait_budget_exceeded`
  - `component_age_budget_exceeded`
  - `missing_component_timestamp`
- [ ] Compute a `packet_coherence_pass` verdict in `_snapshot_sync_packet(...)`.
- [ ] Add configurable budgets for active control:
  - `sync_packet_max_front_vehicle_time_delta_ms`
  - `sync_packet_max_front_vehicle_frame_delta`
  - `sync_packet_max_join_wait_ms`
  - optional `sync_packet_max_component_age_ms`
- [ ] Preserve the raw diagnostics even when coherence fails.
- [ ] Distinguish:
  - packet complete and coherent
  - packet complete but incoherent
  - packet fallback / missing component

#### Stage 3G.3: Gate active selection on coherence, not completeness alone

Files:
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`
- `/Users/philiptullai/Documents/Coding/av/bridge/client.py`
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
- `/Users/philiptullai/Documents/Coding/av/config/av_stack_config.yaml`
- `/Users/philiptullai/Documents/Coding/av/config/acc_highway.yaml`

Implementation checklist:
- [ ] Update `_select_sync_packet_payload(...)` so active control only treats a packet as selectable when it is both:
  - fresh
  - coherent
- [ ] If a packet is fresh but incoherent, return fallback with an explicit coherence reason, not `complete`.
- [ ] Add a selection result split:
  - `complete_coherent`
  - `complete_incoherent`
  - `fallback`
  - `legacy`
- [ ] Do not silently pass through key-complete but incoherent packets to control.
- [ ] Record whether fallback was due to:
  - missing component
  - join failure
  - coherence failure
  - age failure

#### Stage 3G.4: Persist coherence diagnostics into HDF5

Files:
- `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
- `/Users/philiptullai/Documents/Coding/av/data/recorder.py`
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`

Implementation checklist:
- [ ] Add fields for:
  - `sync_packet_coherence_pass`
  - `sync_packet_coherence_reason_code`
  - `sync_packet_complete_but_incoherent`
  - `sync_packet_front_vehicle_time_delta_budget_exceeded`
  - `sync_packet_front_vehicle_frame_delta_budget_exceeded`
  - `sync_packet_join_wait_budget_exceeded`
  - `sync_packet_front_age_ms`
  - `sync_packet_vehicle_age_ms`
  - `sync_packet_component_age_budget_exceeded`
  - `sync_packet_source_context_frame_delta`
  - `sync_packet_source_context_time_delta_ms`
- [ ] Ensure legacy recordings load with these fields as unavailable, not zero.

#### Stage 3G.5: PhilViz and analyzer changes

Files:
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/verify_recording_run.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/server.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/backend/triage_engine.py`

Implementation checklist:
- [ ] Add a `Packet Coherence` summary card showing:
  - coherence pass rate
  - complete-but-incoherent rate
  - coherence reason mode
  - front↔vehicle dt p50/p95/max
  - frame-delta p50/p95/max
  - join-wait p50/p95/max
  - front age p50/p95
  - vehicle age p50/p95
- [ ] Add a timeline row for:
  - `selected packet coherence pass`
  - `coherence reason`
  - `complete but incoherent`
- [ ] Add a `Source Bundle Contract` card showing:
  - source context queue depth
  - stale context drop count
  - context missing count
  - source context frame/time delta
- [ ] Update triage wording so it can say:
  - `selected packet was complete but incoherent`
  - `camera frame lagged vehicle state by 43 Unity frames`
  - instead of only reporting high fallback or timeout counts
- [ ] Update the transport rollup to stop presenting completeness as sufficient for active control.

#### Stage 3G.6: Tests

Add or update:
- `/Users/philiptullai/Documents/Coding/av/tests/test_bridge_sync_packets.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_bridge_client_sync_packets.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_orchestrator_sync_policy.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_control_command_hdf5.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_drive_summary_contract.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_visualizer_transport_parity.py`

Exact test checklist:
- [ ] same-tick camera + vehicle pair passes coherence budgets
- [ ] packet key match with excessive front/vehicle time delta is labeled `complete_but_incoherent`
- [ ] packet key match with excessive frame delta is labeled `complete_but_incoherent`
- [ ] join-wait budget exceedance is labeled correctly
- [ ] stale queued packet contexts are dropped at the source and counted
- [ ] active selector refuses incoherent packets and returns explicit fallback attribution
- [ ] HDF5 persists coherence verdict and reason fields
- [ ] CLI analyzer and PhilViz agree on coherence pass/failure counts
- [ ] legacy recordings still load with coherence fields marked unavailable

#### Live validation order for Phase 3G

1. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`
2. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h8_curve_catchup.yml`
3. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/hill_g1_grade_following.yml`

For each run, compare against:
- clean pre-3F baseline: `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_094053.h5`
- failed 3F run: `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_111746.h5`

#### Acceptance gates for Phase 3G

1. Active selected packets must satisfy coherence budgets at >= `99%` of selected frames.
2. `complete_but_incoherent` rate must be near zero in steady-state highway runs.
3. `front_vehicle_time_delta_ms p95` must be back in the original band, roughly `< 100 ms`.
4. `front_vehicle_frame_delta p95` must be back in the original band, roughly `<= 4`.
5. `join_wait_ms p95` must be back below `100 ms`.
6. Startup lateral error on `highway_h2_steady` must not regress relative to `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_094053.h5`.
7. The transport summary must explicitly distinguish:
   - incomplete fallback
   - complete but incoherent rejection
   - coherent selected packet

#### Priority order after this plan lands

1. Implement same-tick source bundling first.
2. Then add bridge-side coherence gating.
3. Then rerun `highway_h2_steady`.
4. Only after coherence is fixed should any startup lateral/controller retuning be considered.

### Phase 3H: Source-side completeness hardening for `vehicle_component_late`

Phase 3G fixed the coherence regression, but the next validation run showed that active control is still losing too many packets because the source-side bundle is not completing reliably enough. The dominant selected fallback mode is now `vehicle_component_late`, not incoherence.

Validation evidence from `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_125008.h5`:
- `packet_completeness_rate = 75.2%`
- `fallback_active_rate = 24.8%`
- `coherence_pass_rate = 75.2%`
- `complete_but_incoherent_rate = 0.0%`
- `join_failure_reason_mode = vehicle_component_late`
- `join_failure_side_mode = vehicle`
- `source_key_present_camera_rate = 86.1%`
- `source_key_present_vehicle_rate = 89.1%`

This phase is about restoring completeness without weakening the coherence gate.

#### Goal

Keep:
- source-owned packet keys
- same-tick context selection
- coherence-gated active selection
- explicit join-failure attribution

Fix the remaining weak point:
- too many selected frames still fall back because the keyed vehicle-side component arrives too late or does not get keyed/sent in time

#### Priority and relation to swerving

This is the high-priority step before any deep lateral/swerve cleanup.

Reason:
1. The Phase 3G rerun removed the startup transport-coherence failure.
2. The current active path still falls back on roughly one quarter of frames.
3. Late-drive swerving should not be treated as a clean lateral signal until packet completeness is back near promotion quality.

So the execution order should remain:
1. fix source-side completeness
2. rerun `highway_h2_steady`
3. only then decide whether any remaining swerving is truly lateral/control work

#### Working hypothesis to validate

The current data suggests two distinct remaining losses:
1. source packet keys are not present on both sides often enough
2. even when a key exists, the vehicle-side keyed component is still the dominant late side

That means the likely failure surface is not the bridge selector anymore. It is upstream:
- AVBridge packet lifecycle
- vehicle-state send timing relative to camera send timing
- source-side supersession and skipped update behavior

The next phase must prove which of those is dominant instead of treating all misses as generic `vehicle_component_late`.

#### Stage 3H.1: Add a first-class source bundle lifecycle in Unity

Files:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/AVBridge.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CameraCapture.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CarController.cs`

Implementation checklist:
- [ ] Create an explicit source bundle record owned by `AVBridge.cs` for each update that is intended to produce a control packet.
- [ ] The record should contain:
  - `packet_key`
  - `packet_owner_update_id`
  - `packet_owner_unity_frame_count`
  - `packet_owner_unity_time`
  - `bundle_created_realtime`
  - `bundle_deadline_realtime`
  - `camera_requested`
  - `camera_captured`
  - `camera_enqueued`
  - `camera_sent`
  - `vehicle_state_built`
  - `vehicle_state_sent`
  - `bundle_closed_reason`
- [ ] Vehicle state must be stamped from this same bundle record, not from any independent “latest” state.
- [ ] Camera capture should attach to the same bundle record and update it in-place rather than copying partial context into a lossy queue.
- [ ] Add explicit source-side close reasons:
  - `complete`
  - `camera_missing`
  - `vehicle_missing`
  - `camera_late`
  - `vehicle_late`
  - `superseded_before_send`
  - `bundle_deadline_expired`
- [ ] Keep the number of inflight bundle records bounded and observable.

The intent is to replace “camera and vehicle happened to share a key” with “one source bundle either completed or failed for a known reason.”

#### Stage 3H.2: Make vehicle-side send timing directly attributable

Files:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/AVBridge.cs`
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`
- `/Users/philiptullai/Documents/Coding/av/sync_contract.py`

Implementation checklist:
- [ ] Add source-side timestamps/flags that let the bridge distinguish:
  - vehicle state was never built for key `K`
  - vehicle state was built but never sent
  - vehicle state was sent after bundle deadline
  - vehicle state was superseded by a newer source bundle before send
- [ ] Record a source-side vehicle lifecycle reason in the vehicle-state payload.
- [ ] Mirror the equivalent camera lifecycle reason in the camera upload.
- [ ] On the bridge, map `vehicle_component_late` into a more exact selected-frame cause using source lifecycle metadata:
  - `vehicle_not_built`
  - `vehicle_not_sent`
  - `vehicle_sent_after_deadline`
  - `vehicle_superseded_before_send`
- [ ] Do the same for camera-side misses so the attribution is symmetric.

This is the key debugging improvement: selected fallback must answer “what actually happened upstream for this packet key?” not just “vehicle was late.”

#### Stage 3H.3: Add bounded bundle deadline semantics

Files:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/AVBridge.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CameraCapture.cs`
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`

Implementation checklist:
- [ ] Define one source-side bundle deadline budget for active control packets.
- [ ] A bundle that cannot complete within the deadline must be closed explicitly and counted as incomplete.
- [ ] Do not allow partially completed bundles to drift indefinitely and later be interpreted as normal timeouts.
- [ ] Pass the source-side deadline status into the bridge so the selected-frame failure reason reflects the actual producer-side outcome.

Recommended first-pass budgets:
- `source_bundle_deadline_ms = 100`
- `source_bundle_max_inflight = 4`

The exact values can be adjusted later, but the contract must be explicit now.

#### Stage 3H.4: Differentiate key-missing from key-present-but-late

Files:
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`
- `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
- `/Users/philiptullai/Documents/Coding/av/data/recorder.py`
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`

Implementation checklist:
- [ ] Split the selected-frame attribution into:
  - `source_key_missing_camera`
  - `source_key_missing_vehicle`
  - `key_present_camera_only`
  - `key_present_vehicle_only`
  - `key_present_both_vehicle_late`
  - `key_present_both_camera_late`
  - `source_bundle_deadline_expired`
- [ ] Record those as direct per-frame selected-frame fields, not just summary counters.
- [ ] Keep the current join-failure reason codes, but add the more exact selected-frame contract reason beside them.

This prevents the current ambiguity where `vehicle_component_late` may still include multiple upstream producer failures.

#### Stage 3H.5: Recorder and HDF5 contract additions

Files:
- `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
- `/Users/philiptullai/Documents/Coding/av/data/recorder.py`

Add fields for:
- [ ] `sync_packet_selected_failure_contract_reason`
- [ ] `sync_packet_selected_failure_source_stage`
- [ ] `sync_packet_source_bundle_close_reason`
- [ ] `sync_packet_source_bundle_deadline_ms`
- [ ] `sync_packet_source_bundle_age_ms`
- [ ] `sync_packet_source_bundle_inflight_count`
- [ ] `sync_packet_source_vehicle_state_built`
- [ ] `sync_packet_source_vehicle_state_sent`
- [ ] `sync_packet_source_camera_requested`
- [ ] `sync_packet_source_camera_sent`
- [ ] `sync_packet_source_superseded_before_send`

Legacy behavior:
- [ ] Older recordings must load these as unavailable, not zero.

#### Stage 3H.6: PhilViz and analyzer upgrades

Files:
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/verify_recording_run.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/server.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/backend/triage_engine.py`

Implementation checklist:
- [ ] Add a `Source Bundle Completeness` summary card showing:
  - bundle completion rate
  - bundle close-reason split
  - source bundle inflight p50/p95/max
  - bundle deadline exceed rate
  - key-present rate by side
  - selected failure contract reason split
- [ ] Add timeline rows for:
  - `source bundle close reason`
  - `selected failure contract reason`
  - `vehicle state built/sent`
  - `camera requested/sent`
- [ ] Update transport triage wording so it can say:
  - `selected packet fallback because vehicle state was never sent for the keyed source bundle`
  - `selected packet fallback because source bundle expired before vehicle send`
  - `camera was keyed and sent, vehicle was keyed but superseded before send`
- [ ] Keep the existing coherence views, but treat completeness as a separate contract section rather than merging them.

This is the missing direct-attribution layer for the current fallback tail.

#### Stage 3H.7: Tests

Add or update:
- `/Users/philiptullai/Documents/Coding/av/tests/test_sync_packet_source_completeness.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_bridge_sync_packets.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_bridge_client_sync_packets.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_orchestrator_sync_policy.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_control_command_hdf5.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_drive_summary_contract.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_visualizer_transport_parity.py`

Exact test checklist:
- [ ] same source bundle emits keyed camera and keyed vehicle state with the same bundle record
- [ ] vehicle-not-built case is labeled distinctly from vehicle-sent-late
- [ ] vehicle-superseded-before-send case is labeled distinctly
- [ ] bundle deadline expiry is recorded and selected-frame attribution reflects it
- [ ] key present on both sides but vehicle late is preserved as a distinct class
- [ ] HDF5 writes the new bundle lifecycle and selected failure reason fields
- [ ] analyzer and PhilViz agree on bundle close-reason and selected failure splits
- [ ] legacy recordings still load cleanly with unavailable bundle-contract fields

#### Live validation order for Phase 3H

1. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`
2. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h8_curve_catchup.yml`
3. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/hill_g1_grade_following.yml`

Compare against:
- coherence-fixed run: `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_125008.h5`
- clean ACC baseline: `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_094053.h5`

#### Acceptance gates for Phase 3H

1. `packet_completeness_rate >= 95%` on the first pass, target `>= 99%`.
2. `fallback_active_rate <= 5%` on the first pass, target `<= 1%`.
3. `complete_but_incoherent_rate` remains near zero.
4. `join_failure_reason_mode` is no longer dominated by generic `vehicle_component_late`; the selected failure contract reason must identify a specific source lifecycle cause.
5. `source_key_present_camera_rate >= 99%`
6. `source_key_present_vehicle_rate >= 99%`
7. Startup lateral error on `highway_h2_steady` stays within the current clean band from `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_125008.h5`.
8. No new e-stop or out-of-lane regression is introduced while completeness improves.

#### Execution order for Phase 3H

1. Add source bundle lifecycle instrumentation first.
2. Add exact selected-frame failure attribution second.
3. Rerun `highway_h2_steady`.
4. Only if completeness is materially recovered should any remaining swerving be reclassified as a primary lateral issue.

### Phase 3I: Camera request lifecycle hardening for `camera_not_requested`

The next validation run after the bridge-side attribution patch showed that the transport ambiguity is gone, but the dominant remaining fallback cause shifted upstream again. The selected-frame fallback is no longer a generic bridge join failure. It is now mostly a camera lifecycle failure.

Validation evidence from `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_151116.h5`:
- `packet_completeness_rate = 74.3%`
- `fallback_active_rate = 25.7%`
- `join_failure_reason_mode = vehicle_component_late`
- `join_failure_side_mode = vehicle`
- `selected_failure_contract_reason_mode = camera_not_requested`
- `selected_failure_source_stage_mode = camera_capture`
- selected fallback frame counts:
  - `camera_not_requested = 206`
  - `camera_not_sent = 30`
- startup remained healthy:
  - `startup_abs_lateral_p95 = 0.018 m`
  - no startup boundary breach

This phase is now the highest-priority transport step before any serious attempt to interpret later highway swerving.

#### Goal

Keep:
- source-owned packet keys
- same-tick coherence gating
- bridge-side fallback attribution
- active-selection freshness budgets

Fix:
- source bundles that never issue a front-camera request
- camera-request bundles that do not progress to camera-send
- ambiguous open bundles that fall back without a producer-side terminal reason

#### Why this is higher priority than swerving

The current data is still transport-contaminated:
1. startup is fixed, so the old catastrophic transport failure is gone
2. fallback is still about one quarter of selected frames
3. the dominant selected failure is now `camera_not_requested`

That means late-drive swerving is still not cleanly attributable as a primary lateral/controller problem. The next clean question is whether the remaining transport miss is due to:
- AVBridge not requesting capture for a bundle
- CameraCapture dropping or never accepting a valid request
- request-to-capture timing drifting outside the bounded source-side budget

Until that is fixed, swerving should remain secondary.

#### Working hypothesis to validate

The current control path appears to create source bundles more often than `CameraCapture` successfully accepts them into the capture lifecycle.

The likely failure surfaces are:
1. `AVBridge.cs` creates a source bundle and stamps vehicle state, but the corresponding camera request is not issued for that bundle at all
2. `CameraCapture.cs` receives a request opportunity, but bounded context selection or queue turnover causes the bundle to be treated as never requested
3. the camera request was issued too late relative to the bundle deadline, so the selected-frame fallback is correctly showing a camera-stage miss

This phase must separate those cases explicitly.

#### Stage 3I.1: Make camera request issuance an explicit source-bundle state transition

Files:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/AVBridge.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CameraCapture.cs`

Implementation checklist:
- [ ] Move camera-request ownership to a single explicit transition in `AVBridge.cs`.
- [ ] For every source bundle created for an active control update, record one of:
  - `camera_request_issued`
  - `camera_request_skipped`
  - `camera_request_rejected`
- [ ] Add source-bundle fields for:
  - `camera_request_attempted`
  - `camera_request_accepted`
  - `camera_request_skipped_reason`
  - `camera_request_attempt_realtime`
  - `camera_request_accept_realtime`
- [ ] `CameraCapture.RegisterSyncPacketContext(...)` must return an acceptance result, not just mutate state implicitly.
- [ ] If the request is not accepted, close or mark the source bundle with a terminal producer-side reason immediately instead of leaving it effectively open.

The intent is to stop inferring “camera not requested” indirectly from missing downstream fields. The producer should state whether it attempted the request and whether the request entered camera capture state.

#### Stage 3I.2: Differentiate `not requested` from `requested but not accepted`

Files:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CameraCapture.cs`
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`
- `/Users/philiptullai/Documents/Coding/av/sync_contract.py`

Implementation checklist:
- [ ] Add distinct source-stage contract reasons:
  - `camera_request_not_attempted`
  - `camera_request_rejected`
  - `camera_request_expired_before_capture`
  - `camera_requested_not_sent`
- [ ] Keep `camera_not_requested` only for the strict case where `AVBridge` never attempted a request for that source bundle.
- [ ] Map `camera_request_rejected` to a dedicated selected failure source stage:
  - `camera_capture`
- [ ] Map `camera_request_expired_before_capture` to:
  - `source_bundle`
  - if the deadline expired before any valid capture could be associated
- [ ] Ensure the bridge can distinguish:
  - bundle reached `CameraCapture`
  - bundle never reached `CameraCapture`

This removes the current ambiguity where `camera_not_requested` may still be covering multiple producer-side failure classes.

#### Stage 3I.3: Add camera request deadline and close semantics

Files:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/AVBridge.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CameraCapture.cs`

Implementation checklist:
- [ ] Define a bounded budget from source-bundle creation to camera-request acceptance.
- [ ] Define a bounded budget from camera-request acceptance to camera-capture enqueue.
- [ ] If either budget is exceeded, close the source bundle explicitly with:
  - `camera_request_deadline_expired`
  - `camera_capture_deadline_expired`
- [ ] Do not leave these bundles open and let the bridge reinterpret them later as generic timeout-like misses.
- [ ] Record deadline exceed counters separately for:
  - request never attempted
  - request attempted but not accepted
  - accepted but not captured
  - captured but not sent

Recommended first-pass budgets:
- `camera_request_accept_budget_ms = 33`
- `camera_capture_enqueue_budget_ms = 50`

The exact thresholds can change later, but the producer-side contract must become explicit now.

#### Stage 3I.4: Make `CameraCapture` acceptance observable and bounded

Files:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CameraCapture.cs`

Implementation checklist:
- [ ] Change `RegisterSyncPacketContext(...)` so it returns a structured result:
  - accepted
  - rejected_queue_full
  - rejected_out_of_budget
  - rejected_invalid_context
- [ ] Record queue depth and rejection reason on the source bundle at registration time.
- [ ] If a request is rejected because a newer context superseded it, mark:
  - `camera_request_rejected_reason = superseded_before_capture`
- [ ] If a request is accepted, record:
  - acceptance frame/time
  - queue depth at acceptance
- [ ] Preserve the current same-tick capture matching logic, but stop treating acceptance as implicit success.

This is the critical instrumentation gap right now. The data tells us camera capture is the dominant failure stage, but not whether the bundle ever entered the camera queue successfully.

#### Stage 3I.5: Add per-frame recorder and HDF5 fields for camera request lifecycle

Files:
- `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
- `/Users/philiptullai/Documents/Coding/av/data/recorder.py`
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`

Add fields for:
- [ ] `sync_packet_source_camera_request_attempted`
- [ ] `sync_packet_source_camera_request_accepted`
- [ ] `sync_packet_source_camera_request_rejected_reason`
- [ ] `sync_packet_source_camera_request_attempt_age_ms`
- [ ] `sync_packet_source_camera_request_accept_age_ms`
- [ ] `sync_packet_source_camera_request_queue_depth`
- [ ] `sync_packet_selected_failure_contract_reason`
- [ ] `sync_packet_selected_failure_source_stage`

Compatibility rules:
- [ ] older recordings must show these as unavailable, not `0`
- [ ] selected-frame transport summaries must remain readable if only a subset of these fields exists

#### Stage 3I.6: PhilViz and analyzer additions

Files:
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/verify_recording_run.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/server.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/backend/triage_engine.py`

Implementation checklist:
- [ ] Add a `Camera Request Contract` card showing:
  - request attempted rate
  - request accepted rate
  - request rejected split
  - request deadline-expired rate
  - accepted-but-not-sent rate
  - queue depth at request p50/p95/max
- [ ] Add timeline rows for:
  - `camera request attempted`
  - `camera request accepted`
  - `camera request rejected reason`
  - `source bundle close reason`
  - `selected failure contract reason`
- [ ] Update transport rollups so the summary can say:
  - `selected packet fallback because AVBridge never attempted a camera request for the source bundle`
  - `selected packet fallback because CameraCapture rejected the source bundle request`
  - `selected packet fallback because camera request was accepted but no capture was enqueued before bundle deadline`
- [ ] Keep join-failure fields visible, but demote them below the higher-fidelity camera request contract when both are present.

This is the direct-attribution layer required to avoid another round of reverse-engineering from generic fallback counts.

#### Stage 3I.7: Tests

Add or update:
- `/Users/philiptullai/Documents/Coding/av/tests/test_bridge_sync_packets.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_bridge_client_sync_packets.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_orchestrator_sync_policy.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_control_command_hdf5.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_drive_summary_contract.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_visualizer_transport_parity.py`

Exact test checklist:
- [ ] source bundle with no request attempt is labeled `camera_request_not_attempted`
- [ ] request attempted but rejected by `CameraCapture` is labeled `camera_request_rejected`
- [ ] request accepted but capture never enqueued before deadline is labeled `camera_capture_deadline_expired`
- [ ] request accepted and capture sent completes without fallback
- [ ] HDF5 persists the new camera request lifecycle fields
- [ ] analyzer and PhilViz report the same camera request failure splits
- [ ] legacy recordings still load cleanly with unavailable request-lifecycle fields

#### Live validation order for Phase 3I

1. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`
2. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h8_curve_catchup.yml`
3. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/hill_g1_grade_following.yml`

Compare against:
- current attributed fallback run: `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_151116.h5`
- pre-attribution fallback run: `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_134012.h5`
- clean ACC baseline: `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_094053.h5`

#### Acceptance gates for Phase 3I

1. `selected_failure_contract_reason_mode` is no longer dominated by generic camera-stage ambiguity; it must identify one concrete request-lifecycle cause.
2. `camera_request_attempted_rate >= 99%` for active control bundles.
3. `camera_request_accepted_rate >= 99%` once attempted.
4. `camera_request_rejected_rate <= 1%`.
5. `packet_completeness_rate >= 95%` on the first pass, target `>= 99%`.
6. `fallback_active_rate <= 5%` on the first pass, target `<= 1%`.
7. `complete_but_incoherent_rate` remains near zero.
8. startup lateral performance on `highway_h2_steady` remains in the current clean band from `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_151116.h5`.

#### Execution order for Phase 3I

1. Add producer-side camera request attempt/accept/reject states first.
2. Add explicit camera request deadline and close reasons second.
3. Persist those fields into HDF5 and surface them in CLI + PhilViz.
4. Rerun `highway_h2_steady`.
5. Only if fallback is materially reduced should later swerving be treated as a primary lateral issue.

### Phase 3J: Deterministic camera request issuance and bundle abort semantics

Status after implementation and rerun:
- Implemented in transport/runtime/recorder/debug tooling.
- Focused validation passed:
  - `38 passed`
- Highway validation run:
  - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_172948.h5`
- Outcome:
  - not resolved
  - `packet_completeness_rate = 71.1%`
  - `fallback_active_rate = 28.9%`
  - `selected_failure_contract_reason_mode = camera_request_not_attempted`
  - `join_failure_reason_mode = vehicle_component_late`
- Important diagnostic conclusion:
  - the new abort/disposition fields are recorded correctly on successful packets
  - but the dominant remaining loss is now front-camera packets entering active selection without any source-bundle request metadata at all
  - that points to camera captures occurring with no matched/requested source bundle, not just missing abort labeling
  - the next step therefore has to harden the active-camera admission path, not lateral tuning

Phase 3I is now instrumented correctly. The validation rerun proved that the success path carries the new source-bundle camera request fields, so the remaining attribution is trustworthy.

Validation evidence from `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_161226.h5`:
- `packet_completeness_rate = 84.8%`
- `fallback_active_rate = 15.2%`
- `selected_failure_contract_reason_mode = camera_request_not_attempted`
- `selected_failure_source_stage_mode = source_bundle`
- `join_failure_reason_mode = vehicle_component_late`
- `complete_coherent` frames now show:
  - `camera_request_attempted = 100%`
  - `camera_request_accepted = 100%`

That means the remaining transport loss is no longer a tooling blind spot. The dominant residual miss is now: source bundles exist, vehicle-side data is emitted, but no camera request was ever attempted for those fallback bundles.

This is now the highest-priority transport fix before any serious attempt to analyze later swerving as a primary lateral issue.

#### Goal

Keep:
- source-owned packet keys
- same-tick coherence gating
- server-side freshness selection
- trustworthy camera request attribution from Phase 3I

Fix:
- active source bundles that never attempt a camera request
- bundles that are allowed to emit keyed vehicle state even though camera request issuance was skipped
- ambiguous open-bundle behavior that still falls through into `vehicle_component_late`

#### Why this is still higher priority than swerving

The current transport state is improved but still not clean enough for lateral diagnosis:
1. startup is healthy
2. coherence is healthy
3. the false teleport/reset path is gone
4. fallback is still `15.2%`
5. the dominant fallback contract is now a real producer-side miss: `camera_request_not_attempted`

So the remaining swerving cannot yet be treated as clean lateral evidence. The next clean question is whether AVBridge is:
- intentionally skipping request issuance,
- failing to reach the request attempt branch,
- or sending keyed vehicle bundles before request admission is resolved.

Until that is fixed, swerving remains secondary.

#### Core invariant for Phase 3J

For every active source bundle, exactly one of these must be true before vehicle send:
1. `camera_request_attempted = true`
2. bundle is explicitly aborted with a terminal producer-side reason and does **not** enter the active keyed vehicle path

The current failure mode violates that invariant by allowing the transport to observe:
- keyed vehicle-side bundle state
- no paired front camera
- `camera_request_not_attempted`

That must become structurally impossible in the steady-state active path.

#### Stage 3J.1: Make camera request issuance synchronous with bundle creation in AVBridge

Files:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/AVBridge.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CameraCapture.cs`

Implementation checklist:
- [ ] Move bundle creation and camera request attempt into one explicit AVBridge-owned transaction.
- [ ] In the same update where a source bundle is created, AVBridge must immediately do one of:
  - `request_attempted_and_accepted`
  - `request_attempted_and_rejected`
  - `request_skipped_with_reason`
- [ ] Stop allowing a bundle to remain in an implicit “camera maybe later” state for active transport.
- [ ] Record explicit skipped reasons at the point of decision, not later by inference.

Recommended skip/reject taxonomy:
- `camera_request_skipped_camera_unavailable`
- `camera_request_skipped_sync_disabled`
- `camera_request_skipped_budget_exceeded`
- `camera_request_skipped_bundle_superseded`
- `camera_request_rejected_queue_full`
- `camera_request_rejected_invalid_context`
- `camera_request_rejected_out_of_budget`

#### Stage 3J.2: Gate keyed vehicle emission on camera request admission

Files:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/AVBridge.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CarController.cs`

Implementation checklist:
- [ ] Do not send a keyed active vehicle-state bundle into the bridge unless camera request disposition is already resolved.
- [ ] If request was not attempted or was rejected, mark the source bundle aborted before vehicle send and use a terminal close reason.
- [ ] Do not allow the active keyed packet contract to be populated by a vehicle-side component for a bundle that never entered camera request state.
- [ ] If non-active legacy/debug transport still needs the vehicle sample, keep that path separate from the active sync packet contract.

Required explicit close reasons:
- `bundle_aborted_camera_request_not_attempted`
- `bundle_aborted_camera_request_rejected`
- `bundle_aborted_camera_request_budget_exceeded`
- `bundle_aborted_camera_capture_missing`

The active path should prefer dropping the bundle explicitly over creating a misleading `vehicle_component_late` timeout.

#### Stage 3J.3: Make CameraCapture admission results authoritative

Files:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CameraCapture.cs`

Implementation checklist:
- [ ] Keep the structured `RegisterSyncPacketContext(...)` result from Phase 3I.
- [ ] Extend it to carry an explicit disposition enum used by AVBridge:
  - `accepted`
  - `rejected_queue_full`
  - `rejected_out_of_budget`
  - `rejected_invalid_context`
  - `rejected_superseded`
- [ ] Guarantee AVBridge gets that disposition synchronously in the same update it creates the bundle.
- [ ] Do not rely on later queue inspection to infer whether a request ever entered camera capture state.

#### Stage 3J.4: Separate aborted bundles from timed-out bundles in the bridge

Files:
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`
- `/Users/philiptullai/Documents/Coding/av/sync_contract.py`

Implementation checklist:
- [ ] Add selected failure contract reasons for explicit producer aborts:
  - `bundle_aborted_camera_request_not_attempted`
  - `bundle_aborted_camera_request_rejected`
  - `bundle_aborted_camera_request_budget_exceeded`
- [ ] Keep `packet_timeout` for true join/arrival timeout only.
- [ ] Map source-aborted bundles to a distinct selected failure stage:
  - `source_bundle`
- [ ] Ensure bridge-side `vehicle_component_late` is no longer used for cases where the producer already knew the bundle was invalid before camera request admission.

The goal is to prevent producer-aborted bundles from being misread as bridge join failures.

#### Stage 3J.5: Add source-bundle issuance contract metrics to recorder and PhilViz

Files:
- `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
- `/Users/philiptullai/Documents/Coding/av/data/recorder.py`
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/server.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`

Add first-class fields for:
- [ ] `sync_packet_source_camera_request_skipped_reason`
- [ ] `sync_packet_source_bundle_aborted_before_vehicle_send`
- [ ] `sync_packet_source_bundle_abort_reason`
- [ ] `sync_packet_source_bundle_vehicle_send_blocked_by_camera_request`
- [ ] `sync_packet_source_camera_request_disposition_code`
- [ ] `sync_packet_source_camera_request_attempt_frame`
- [ ] `sync_packet_source_camera_request_accept_frame`

PhilViz/CLI additions:
- [ ] `Camera Request Contract` card:
  - attempted rate
  - accepted rate
  - skipped reason mode
  - rejected reason mode
  - aborted-before-vehicle-send rate
- [ ] `Bundle Abort Contract` card:
  - abort rate
  - abort reason split
  - keyed-vehicle-without-request rate
- [ ] timeline rows:
  - camera request disposition
  - bundle abort reason
  - vehicle send blocked by camera request

#### Stage 3J.6: Add one hard contract assertion to the active path

Files:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/AVBridge.cs`
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`

Implementation checklist:
- [ ] Add one explicit debug assertion/metric:
  - `keyed_vehicle_without_camera_request_attempt`
- [ ] This should be zero in steady-state active mode.
- [ ] Persist and surface it in summary tools even if it only increments on failure.

This is the single best guardrail to stop the current failure mode from silently returning.

#### Tests for Phase 3J

Unity-side behavior tests:
- [ ] source bundle with accepted request proceeds to keyed vehicle send
- [ ] source bundle with skipped request aborts before keyed vehicle send
- [ ] source bundle with rejected request aborts before keyed vehicle send
- [ ] source bundle with camera capture missing uses explicit abort reason

Bridge tests:
- [ ] producer-aborted bundle is classified as source-bundle abort, not `packet_timeout`
- [ ] `vehicle_component_late` is reserved for true arrival/join lateness
- [ ] keyed vehicle without request attempt triggers contract metric/assertion

Recorder/analyzer tests:
- [ ] new skipped/abort/disposition fields persist to HDF5
- [ ] analyzer and PhilViz agree on skipped vs rejected vs aborted counts
- [ ] legacy recordings still load cleanly with unavailable abort fields

#### Live validation order for Phase 3J

1. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`
2. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h8_curve_catchup.yml`
3. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/hill_g1_grade_following.yml`

Compare against:
- trusted Phase 3I instrumentation run: `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_161226.h5`
- earlier attributed run: `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_151116.h5`
- clean ACC ownership baseline: `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_094053.h5`

#### Acceptance gates for Phase 3J

1. `camera_request_not_attempted` is no longer the dominant selected failure contract mode.
2. `bundle_aborted_before_vehicle_send_rate` is explicit and low enough to explain any remaining misses.
3. `keyed_vehicle_without_camera_request_attempt_rate = 0`.
4. `packet_completeness_rate >= 92%` on the first pass, target `>= 97%`.
5. `fallback_active_rate <= 8%` on the first pass, target `<= 3%`.
6. `complete_but_incoherent_rate` remains near zero.
7. startup lateral performance remains in the clean band from `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_125008.h5`.
8. Only after those gates pass should late-drive swerving be reclassified as a likely primary lateral issue.

#### Execution order for Phase 3J

1. Add deterministic request issuance + explicit skip/reject taxonomy in Unity.
2. Gate keyed vehicle emission on request admission.
3. Add producer-abort classification in the bridge.
4. Add recorder + PhilViz + CLI contract surfaces.
5. Rerun `highway_h2_steady`.
6. If fallback materially drops, then build the dedicated swerving/lateral deep-dive plan from the cleaned transport run.

### Phase 3K: Active-camera admission gating and debug-only unbundled capture semantics

Status entering Phase 3K:
- `Phase 3J` is implemented and instrumented.
- Latest validation run:
  - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_172948.h5`
- Outcome:
  - `packet_completeness_rate = 71.1%`
  - `fallback_active_rate = 28.9%`
  - `selected_failure_contract_reason_mode = camera_request_not_attempted`
  - `join_failure_reason_mode = vehicle_component_late`
- Critical observation from HDF5:
  - `camera_request_not_attempted` frames show:
    - `sync_packet_source_bundle_camera_requested = 0`
    - `sync_packet_source_camera_request_attempted = 0`
    - `sync_packet_source_camera_request_accepted = 0`
    - `sync_packet_source_bundle_aborted_before_vehicle_send = 0`
    - skipped/disposition/abort strings empty

That means the dominant remaining loss is not “aborted bundle classification missing.” It is that front-camera captures are still reaching the active transport path with no admitted source bundle at all. This is higher priority than swerving analysis.

#### Goal

Keep:
- source-owned packet keys
- coherence gating
- server-side freshness selection
- `Phase 3J` request/abort attribution

Fix:
- camera captures with no admitted source bundle entering the active packet path
- fallback packet keys from `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CameraCapture.cs` being treated as control-relevant active packets
- ambiguity between active transport data and debug-only/unbundled capture data

#### Why this is still higher priority than swerving

The transport contract is still too lossy for trustworthy lateral diagnosis:
1. startup is healthy
2. coherence is healthy
3. false teleport/reset is gone
4. fallback is still `28.9%`
5. the dominant remaining miss is still a producer-side camera admission problem

Until unmatched camera captures are excluded from the active path, later swerving is still contaminated by transport loss and should not be treated as a primary lateral tuning target.

#### Core invariant for Phase 3K

For active transport, every selected front camera must have exactly one of these states:
1. paired to an admitted source bundle with `camera_request_attempted = true`
2. explicitly rejected from the active path and labeled `debug_only_unbundled_capture`

What must become impossible:
1. front camera enters active join path
2. no source bundle request metadata exists
3. bridge later reports `camera_request_not_attempted`

#### Stage 3K.1: Split active capture from unbundled debug capture at the source

Files:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CameraCapture.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/AVBridge.cs`

Implementation checklist:
- [ ] Add an explicit capture classification at enqueue time:
  - `active_bundled_capture`
  - `debug_unbundled_capture`
- [ ] If `SelectSyncPacketContextForCapture(...)` does not return a matched admitted context, do not build an active packet key from `BuildFallbackSyncPacketKey(...)` for the active path.
- [ ] Keep fallback packet keys only for debug/compatibility capture accounting, not active control transport.
- [ ] Make the capture item carry:
  - `active_transport_eligible`
  - `unbundled_debug_only`
  - `camera_capture_contract_reason`

Required reasons:
- `no_source_bundle_context`
- `source_bundle_context_budget_exceeded`
- `source_bundle_context_superseded`
- `source_bundle_context_mismatched`
- `source_bundle_context_missing`

#### Stage 3K.2: Make CameraCapture request admission authoritative for active upload

Files:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CameraCapture.cs`

Implementation checklist:
- [ ] Only call the active `/api/camera` upload path when `sourceBundleContext != null` and the bundle was admitted.
- [ ] If no admitted context exists, either:
  - skip upload entirely for active mode, or
  - upload with explicit `debug_unbundled_capture` classification that the bridge excludes from active packet assembly.
- [ ] Do not allow a front-camera frame with no admitted bundle to update active `latestSyncPacket*` state.
- [ ] Keep debug-only capture telemetry available so diagnostics still show camera throughput independent of active transport.

#### Stage 3K.3: Separate active and debug camera ingestion in the bridge

Files:
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`
- `/Users/philiptullai/Documents/Coding/av/sync_contract.py`

Implementation checklist:
- [ ] Add active-ingest eligibility fields to the camera component schema:
  - `source_bundle_active_transport_eligible`
  - `source_bundle_debug_unbundled_capture`
  - `camera_capture_contract_reason`
- [ ] Ensure `_attach_camera_component(...)` only feeds active packet assembly for `active_transport_eligible = true`.
- [ ] Keep debug-only/unbundled captures available to diagnostics, but outside active join/completeness math.
- [ ] Add a distinct selected failure / classification family for excluded debug captures:
  - `debug_unbundled_capture_excluded`
  - not `packet_timeout`
  - not `vehicle_component_late`

#### Stage 3K.4: Tighten join accounting so active completeness is measured against eligible captures only

Files:
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`

Implementation checklist:
- [ ] Remove debug-only/unbundled camera frames from active completeness denominators.
- [ ] Add separate metrics:
  - `active_camera_eligible_rate`
  - `debug_unbundled_capture_rate`
  - `active_camera_excluded_reason_mode`
- [ ] Keep existing active completeness/fallback metrics strictly about active-eligible transport.
- [ ] Prevent active failures from being attributed to camera request lifecycle when the frame never qualified for active transport in the first place.

#### Stage 3K.5: Recorder, PhilViz, and CLI attribution

Files:
- `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
- `/Users/philiptullai/Documents/Coding/av/data/recorder.py`
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/server.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`

Add first-class fields:
- [ ] `sync_packet_source_bundle_active_transport_eligible`
- [ ] `sync_packet_source_bundle_debug_unbundled_capture`
- [ ] `sync_packet_camera_capture_contract_reason`
- [ ] `sync_packet_active_camera_excluded`
- [ ] `sync_packet_active_camera_excluded_reason`

PhilViz/CLI additions:
- [ ] `Active Camera Admission` card:
  - eligible rate
  - excluded rate
  - excluded reason mode
- [ ] `Debug-Unbundled Capture` card:
  - rate
  - source context mismatch mode
  - whether any debug-only frame touched active transport
- [ ] timeline rows:
  - `camera_capture_contract_reason`
  - `active_transport_eligible`
  - `debug_unbundled_capture`

#### Stage 3K.6: Add one hard contract guardrail

Files:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CameraCapture.cs`
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`

Implementation checklist:
- [ ] Add a counter/guard:
  - `unbundled_camera_entered_active_path`
- [ ] This must be zero for active transport.
- [ ] Persist it and fail the transport summary if nonzero.

This is the single best guardrail for the current failure mode.

#### Tests for Phase 3K

Unity-side behavior tests:
- [ ] admitted source bundle produces `active_bundled_capture`
- [ ] missing source bundle produces `debug_unbundled_capture`
- [ ] mismatched/stale source bundle produces `debug_unbundled_capture`
- [ ] debug-only capture does not mutate active latest packet state

Bridge tests:
- [ ] debug-unbundled capture is excluded from active join assembly
- [ ] active completeness denominator uses eligible captures only
- [ ] `camera_request_not_attempted` is not emitted for excluded debug captures
- [ ] `unbundled_camera_entered_active_path` remains zero in normal flow

Recorder/analyzer tests:
- [ ] new admission/exclusion fields persist to HDF5
- [ ] PhilViz and CLI agree on eligible/excluded/debug-only counts
- [ ] legacy recordings still load with `UNAVAILABLE` values

#### Live validation order for Phase 3K

1. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`
2. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h8_curve_catchup.yml`
3. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/hill_g1_grade_following.yml`

Compare against:
- `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_172948.h5`
- `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_161226.h5`
- `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_094053.h5`

#### Acceptance gates for Phase 3K

1. `unbundled_camera_entered_active_path = 0`
2. `camera_request_not_attempted` is no longer the dominant selected failure contract mode
3. active completeness recovers to at least the pre-3J band on first pass:
   - `packet_completeness_rate >= 85%`
4. fallback drops materially from `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_172948.h5`:
   - first-pass target `fallback_active_rate <= 12%`
5. coherence remains healthy:
   - `complete_but_incoherent_rate ~= 0`
6. startup remains healthy:
   - no early boundary breach
7. only after these gates pass should later swerving be treated as likely primary lateral behavior

#### Execution order for Phase 3K

1. Split active vs debug-only camera admission in `CameraCapture`.
2. Prevent unmatched captures from entering the active packet path.
3. Teach the bridge to exclude debug-unbundled captures from active completeness/join accounting.
4. Add recorder + PhilViz + CLI admission/exclusion metrics.
5. Rerun `highway_h2_steady`.
6. If fallback materially drops, then do the dedicated swerving/lateral deep dive on the cleaned run.

### Phase 3L: Request-driven best-match source context capture

Status entering Phase 3L:
- `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_181048.h5`
- `packet_completeness_rate = 81.8%`
- `fallback_active_rate = 18.2%`
- `selected_failure_contract_reason_mode = camera_requested_not_sent`
- `join_failure_reason_mode = camera_component_late`
- `active_camera_excluded_reason_mode = no_source_bundle_context`
- the rebuilt player confirms the Phase 3K exclusion path is live, but admitted source bundles are still being lost when camera capture cadence and AVBridge update cadence drift

#### Why Phase 3L is the next priority

The remaining transport loss is now concentrated in the camera-side source lifecycle:
- the bridge is excluding unbundled camera frames correctly
- the dominant miss has shifted from `camera_request_not_attempted` to `camera_requested_not_sent`
- successful packets still need their source-admission fields preserved end-to-end
- `CameraCapture` is still draining all pending contexts and selecting only the newest one, which starves older admitted bundles whenever more than one accepted bundle is waiting

Until source-side capture matching is deterministic, later highway swerving is still contaminated by transport loss.

#### Core invariant for Phase 3L

Every accepted source bundle must get one bounded, deterministic chance to produce an active camera capture:
- accepted bundles cannot be discarded just because a newer bundle arrived first
- successful active packets must retain `source_bundle_active_transport_eligible = 1`
- captures that occur with no admissible source bundle remain debug-only and excluded from active transport

#### Stage 3L.1: Preserve successful source-admission fields through bridge payload selection

Files:
- `/Users/philiptullai/Documents/Coding/av/bridge/server.py`
- `/Users/philiptullai/Documents/Coding/av/tests/test_bridge_sync_packets.py`

Implementation checklist:
- [ ] Protect the active-admission success fields in `_sync_packet_payload_response_with_selection(...)`:
  - `source_bundle_active_transport_eligible`
  - `source_bundle_debug_unbundled_capture`
  - `camera_capture_contract_reason`
  - `active_camera_excluded_event_delta`
  - `active_camera_excluded_reason_code`
  - `unbundled_camera_entered_active_path_event_delta`
- [ ] Add a regression test proving a successful packet keeps `source_bundle_active_transport_eligible = true` when `selection_meta` carries default/empty values.
- [ ] Re-run the HDF5 success-path spot check after the next highway validation to confirm complete packets can actually carry active-admission truth.

#### Stage 3L.2: Replace newest-only context draining with best-match FIFO selection

Files:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CameraCapture.cs`

Implementation checklist:
- [ ] Stop clearing `pendingSyncPacketContexts` on every capture.
- [ ] Scan pending contexts in FIFO order and select the first in-budget bundle instead of the newest bundle.
- [ ] Retain newer not-yet-admissible bundles in the queue for later capture attempts.
- [ ] Expire only contexts that are genuinely stale:
  - deadline exceeded
  - frame delta beyond budget on the old side
  - time delta beyond budget on the old side
- [ ] Distinguish:
  - `no_source_bundle_context`
  - `source_bundle_context_pending`
  - `source_bundle_context_mismatched`
- [ ] Preserve queue-depth and stale-drop diagnostics so PhilViz/CLI can separate “no admitted bundle” from “admitted bundle dropped”.

#### Stage 3L.3: Add request-driven capture when admitted bundles are waiting

Files:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/CameraCapture.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/AVBridge.cs`

Implementation checklist:
- [ ] Add a request-driven capture mode that allows `CameraCapture` to capture earlier than the periodic interval when accepted source bundles are pending.
- [ ] Bound that mode with a minimum interval so the camera path does not run unconstrained.
- [ ] Use pending accepted bundle count as the trigger, not generic camera backlog.
- [ ] Keep the existing periodic capture path for debug observability and non-active use.
- [ ] Ensure request-driven captures do not mark unmatched/debug-only captures as active.

#### Stage 3L.4: Keep active transport admission explicit in recorder and tools

Files:
- `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
- `/Users/philiptullai/Documents/Coding/av/data/recorder.py`
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/server.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`

Implementation checklist:
- [ ] Keep the current Phase 3K admission/exclusion fields.
- [ ] Add summary metrics for:
  - active-admission success rate
  - request-driven capture assist rate
  - `source_bundle_context_pending` rate
  - `source_bundle_context_mismatched` rate
- [ ] Make `camera_requested_not_sent` and `camera_component_late` visible next to the new request-driven metrics so attribution remains direct.

#### Tests for Phase 3L

Python tests:
- [ ] bridge payload-response regression test for preserved active-admission fields
- [ ] existing orchestrator/HDF5 tests updated only if new summary fields are added

Unity behavior checks:
- [ ] pending accepted bundle can survive multiple captures until matched or expired
- [ ] newer bundle arrival does not automatically supersede an older still-in-budget bundle
- [ ] request-driven capture path runs when pending admitted bundles exist
- [ ] unmatched periodic capture remains debug-only

#### Live validation order for Phase 3L

1. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`
2. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h8_curve_catchup.yml`

Compare against:
- `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_181048.h5`
- `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_172948.h5`

#### Acceptance gates for Phase 3L

1. successful complete packets can carry `source_bundle_active_transport_eligible = 1`
2. `active_camera_excluded_reason_mode = no_source_bundle_context` drops materially from `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_181048.h5`
3. `selected_failure_contract_reason_mode = camera_requested_not_sent` is no longer dominant
4. `packet_completeness_rate >= 88%`
5. `fallback_active_rate <= 12%`
6. startup remains healthy:
  - no early boundary breach
7. only after these gates pass should the later highway swerving be treated as likely primary lateral behavior

#### Execution order for Phase 3L

1. Patch bridge payload selection so successful source-admission fields are not clobbered.
2. Replace newest-only source context draining with best-match FIFO selection.
3. Add bounded request-driven capture while admitted bundles are waiting.
4. Re-run `highway_h2_steady` with a forced Unity rebuild.
5. Inspect complete-packet HDF5 fields for preserved `active_transport_eligible`.
6. If fallback clears the target band, then resume the swerving/lateral deep dive.
