# AV Stack — Performance Brief

**Last updated:** 2026-08-12
**Status:** Baseline measured. Finding 1 investigated and closed (not a bug).
Finding 2 is the top open item.
**Source recording:** `data/recordings/recording_20260506_042045.h5` (720 frames,
`highway_h9_adjacent_lane_parallel_reject`, 2026-05-06 — the most recent fresh
Unity run at time of writing).

---

## Headline

**Unity rendering is the bottleneck. The Python stack is not.**

The control loop runs at **6.23 Hz** against a configured `target_loop_hz: 30`.
Of each 145.8 ms loop period, the stack spends **69.2 ms working and 74.6 ms
idle**, waiting for Unity to deliver the next camera frame. Unity's own render
period is 150.0 ms (6.7 FPS).

No amount of Python optimization raises the frame rate from here — the loop is
already waiting half of every cycle. The two levers that matter are (a) making
Unity render faster, and (b) **decoupling simulation time from wall-clock time
so render speed stops mattering** (see *Lockstep* below).

---

## How to reproduce this measurement

The project already ships the attribution tool. Do not write a one-off script.

```bash
python3 tools/analyze/cadence_breakdown.py <recording.h5>
python3 tools/analyze/cadence_breakdown.py --latest
python3 tools/analyze/cadence_breakdown.py <rec.h5> --output-json out.json
```

Per-stage timings are recorded in HDF5 under `control/`:
`perf_wait_input_ms`, `perf_perception_ms`, `perf_planning_ms`,
`perf_control_ms`, `perf_hdf5_write_ms`, `e2e_latency_ms`, `mpc_solve_time_ms`.

---

## Measured baseline (2026-05-06 recording, 720 frames)

| Stage | p50 | p95 | Share of loop |
|---|---|---|---|
| **Wall loop period** | **145.8 ms** | 237.4 ms | → 6.23 Hz |
| `perf_wait_input_ms` | 74.6 ms | 154.4 ms | 51% (idle) |
| `pipeline_ms` | 69.2 ms | 87.1 ms | 47% |
| ├─ `perf_perception_ms` | **49.6 ms** | 62.4 ms | **72% of pipeline** |
| ├─ `perf_planning_ms` | 6.6 ms | 9.3 ms | 10% of pipeline |
| ├─ `perf_control_ms` | 1.6 ms | 4.3 ms | 2% of pipeline |
| └─ `perf_hdf5_write_ms` | 0.01 ms | 0.03 ms | ~0% |
| **`unity_render_frame_dt_ms`** | **150.0 ms** | 172.0 ms | **the ceiling** |

Severe frames (>200 ms): 26.0% (187/719).

Tool's own verdicts on this recording:

- `[alert] QUEUE_BACKLOG` — front queue depth p95 = 48.0 at max 48.0
  — **FALSE POSITIVE**, see Finding 1
- `[alert] PACKET_FALLBACK` — sync packet fallback active on 100.00% of frames
  — **FALSE POSITIVE** (configured `packet_shadow` mode), see Finding 1
- `[warn] FRAME_SKIPS` — `stream_front_frame_id_delta` mean 1.071, p95 2.00
- `[warn] WAIT_INPUT_DOMINATES` — on severe frames, wait exceeds pipeline
  — **real**, and consistent with the Unity-bound headline

Treat the two alerts as tool bugs until T-PERF-METRIC-2 lands.

---

## Finding 1 — Camera transport is HEALTHY; two diagnostics are false positives

**Investigated 2026-08-12 and closed. Do not re-open without new evidence.**

An initial read of `cadence_breakdown.py` output suggested the camera FIFO was
saturated and serving ~9-second-stale frames (`stream_front_queue_depth` pinned
at 48/48, `stream_front_unity_dt_ms` p50 = 9,352 ms). **Both signals are
misleading. Actual frame age is fine.**

### The real number

`vehicle/stream_front_latest_age_ms` — **p50 = 39.5 ms, p95 = 95.4 ms, max =
151.3 ms.** Consumed frames are well under one Unity render period old. Control
is not eating stale frames.

### Why `stream_front_unity_dt_ms` is not frame age

`av_stack/orchestrator.py:10432` computes it as:

```python
stream_front_unity_dt_ms = float(timestamp - float(unity_time_value)) * 1000.0
```

That subtracts **Unity's simulation clock** (`unityTime`) from a
**different clock** (`timestamp`). It measures epoch offset plus relative drift,
not staleness. Evidence from `recording_20260506_042045.h5` (720 frames):

| Check | Result | Meaning |
|---|---|---|
| Value on frame 0 | **7,785 ms** | A backlog cannot be 7.8 s deep at frame 0 |
| corr(value, frame index) | **+0.946** | Tracks elapsed time, not queue state |
| Drift first → last | +2,691 ms over ~115 s | ≈2.3% clock drift between the two |
| corr(value, queue depth) | n/a — depth std = 0 | No relationship to the queue at all |

This is a textbook `feedback_diagnostic_labels_can_be_buggy` case: the metric
name implies frame age, the implementation is a cross-clock subtraction.

### Why queue depth 48/48 is by design

`bridge/server.py:3284` sets `queue_depth = len(queue_obj)` on a
`deque(maxlen=48)`. A full ring buffer is the *steady state* of a bounded deque:
appending past capacity evicts the oldest. Combined with a 39.5 ms consumed-frame
age, the reader is clearly taking the newest entry — i.e. the transport already
has latest-wins semantics. The queue holds 48 frames of recent history; it is
not a backlog the consumer must drain.

`cadence_breakdown.py`'s `[alert] QUEUE_BACKLOG` ("Consumer likely slower than
producer") fires on `depth ≈ capacity` and is therefore a **false positive** for
a ring buffer designed to sit full.

### Why 100% packet fallback is expected

`config/av_stack_config.yaml:1078` sets `sync_packet_mode: packet_shadow`, whose
own comment reads: *"Closed-loop remains on latest camera + latest vehicle.
Shadow packet path records continuity only."* The active FIFO packet path is
deliberately unused, so `sync_packet_fallback_active = 100%` is the configured
design, not a failure. `cadence_breakdown.py`'s `[alert] PACKET_FALLBACK`
("Control is relying on legacy/partial transport") describes intended behaviour
as a fault — also a **false positive**.

### Follow-up work this created

- **T-PERF-METRIC-1** — `stream_front_unity_dt_ms` should either be renamed to
  disclose that it is a clock-offset diagnostic, or recomputed against a
  same-domain timestamp. As written it will mislead every future reader.
  (Related: `feedback_field_naming_for_single_writer_provenance`.)
- **T-PERF-METRIC-2** — `tools/analyze/cadence_breakdown.py` should suppress
  `QUEUE_BACKLOG` when the queue is a bounded ring buffer (compare
  `latest_age_ms` against the render period instead of depth against capacity),
  and suppress `PACKET_FALLBACK` when `sync_packet_mode` is `packet_shadow`.

**Not** related to the `project_h5_harness_deferred` "bridge payload age 34 s"
symptom — that tripwire remains open and needs its own evidence.

---

## Finding 2 — Perception and Unity contend for one GPU

`config/av_stack_config.yaml` sets `use_gpu: true`, `prefer_mps: true`, and MPS
is available on this machine. `perception/device_utils.py::resolve_torch_device`
therefore places the segmentation model on the **same Apple GPU Unity renders
with**. Apple Silicon is a unified-memory SoC with a single MPS device — this is
direct contention, not a theoretical concern.

The active path is confirmed segmentation, not the CV fallback:
`perception/detection_method` = `segmentation` on 720/720 frames. (Note:
`CLAUDE.md` still describes the CV fallback as "de facto active" — stale for
ACC scenarios at minimum.)

49.6 ms p50 is slow for a 256×512 forward pass, consistent with contention plus
per-call MPS launch overhead. The code also builds the tensor on CPU
(`torch.from_numpy(resized.astype(np.float32) / 255.0)`) before transfer, adding
a sync point per frame.

**Cheap experiment (one run):** set `perception.use_gpu: false`, run one
scenario, and compare `unity_render_frame_dt_ms`. If Unity's 150 ms drops,
contention is confirmed and CPU-vs-GPU placement becomes a real scheduling
decision rather than an assumed win.

---

## Finding 3 — Small, safe wins (will NOT move FPS)

Listed because they are free, not because they matter at current cadence.

**`data/recorder.py` dataset-handle caching.** The recorder resolves
`self.h5_file["control/..."]` ~938 times per flush; each is an HDF5 link
traversal. Measured against a real recording:

```
h5py __getitem__ ×940:  uncached = 12.36 ms    cached = 0.038 ms   (325× faster)
```

Resolving handles once in `_create_datasets()` into a dict removes 12.3 ms per
flush. But flushes fire every `flush_every: 30` frames ≈ 2.3 s at current
cadence, so this is a **0.6% duty cycle** — zero-risk, near-zero payoff today.
Revisit if the loop ever reaches 30 Hz.

**`topdown_recording_interval_frames: 3`** — costs an extra bridge GET + decode
every third frame. The config comment already says to raise it when chasing
higher control Hz. Set to `0` for pure performance runs.

**Blind `time.sleep(self.frame_interval)`** at six sites in
`av_stack/orchestrator.py::run` (≈3729, 3772, 3805, 3856, 3862, 3867). When a
packet isn't ready the loop sleeps a *full* frame interval (33 ms at
`target_loop_hz: 30`) rather than retrying tightly, amplifying latency on every
miss.

### Refuted hypothesis — the recorder is not a bottleneck

`data/recorder.py` is 12,074 lines: `_create_datasets()` is 4,627 lines of 938
hand-written `create_dataset` calls, and `_write_control_commands()` is 3,657
lines declaring ~500 parallel lists. It looks like the obvious culprit.

It measures at `perf_hdf5_write_ms` p50 = **0.01 ms** in-loop, and ~13.6 ms of
GIL-holding work on the background flush thread every 2.3 s. Total impact well
under 1%.

Recorded here so nobody re-derives the same wrong hypothesis from line count.
(The file is still a maintainability problem — the 6-location pattern — but that
is a separate concern from performance.)

---

## Lockstep — the structural fix

Every mature AV simulator solves slow-host rendering the same way, and it is not
faster hardware.

The stack currently runs the simulator in **real-time mode**: Unity renders on a
wall clock and the stack scrambles to keep up. On constrained hardware that
converts into a *data-quality* problem — 6.7 FPS, 9-second-stale frames,
degraded control authority.

The alternative is **synchronous / lockstep** mode: the simulator advances a
fixed Δt and *waits* for the client. A slow machine then costs wall-clock time
and nothing else — physics, sensor timing, and control cadence stay correct.

Prior art:

- **AirSim** — lockstep "causes the simulator to not use a realtime clock, but
  instead advances the clock in steps with each sensor update sent to the
  controller. This way the controller thinks time is progressing smoothly no
  matter how long it takes the simulator to really process that update loop...
  AirSim can be used on slow machines that cannot process updates quickly."
- **CARLA** — synchronous mode "becomes specially relevant with slow client
  applications... the simulator waits until the client is ready to continue."
  With a hard caveat: *"Always run the simulator at fixed time-step when using
  the synchronous mode. Otherwise the physics engine will try to recompute at
  once all the time spent waiting for the client, this usually results in
  inconsistent or not very realistic physics."* CARLA also documents that
  GPU sensors (cameras) lag CPU sensors by a couple of frames — directly
  relevant to the temporal-sync fragile area in `CLAUDE.md`.
- **Unity** — supports this natively via `Time.captureDeltaTime`, which
  "decouples wall clock time from simulation time... `Time.time` increases at an
  interval of `captureDeltaTime` regardless of real time and the duration of a
  frame." Unity's own Perception package builds deterministic multi-sensor
  capture on exactly this primitive.
- **AWSIM** (Tier IV — the closest peer to this project) ships a user-facing
  Time Scale control and documents dropping to ×0.50 to reduce rendering delay.

### What it would take here

The stack is already structured for it: the bridge is request/response and the
orchestrator has an explicit per-frame boundary at `_process_frame`. Unity would
set `Time.captureDeltaTime = 1/30`, render one frame, publish, and block on a
step-ack from the orchestrator instead of free-running. `Time.fixedDeltaTime`
must be pinned in the same change (see the CARLA caveat above).

Estimated cost: a 60 s scenario would take roughly 140 s wall-clock on this Mac
Mini, and produce a clean 30 FPS recording with no stale frames and no queue
backlog.

### What it would unblock

- Nightly Unity seeding becomes viable on this hardware — which is the
  prerequisite for the 91-night frozen lateral sweep and the 89-night ACC
  blackout (`project_sweep_trajectory_flags`, `project_acc_sweep_baseline`).
- The acc-sweep pre-flight cadence gate (`8e95382`) becomes unnecessary —
  cadence stops being a function of machine load.
- `project_cadence_13fps` ("~13 FPS is normal now") stops being true by
  construction.

### The trade-off — state it before adopting

Lockstep gives up wall-clock realism, so it **cannot** catch timing bugs that
only appear under genuine real-time pressure. The
`feedback_solver_cold_start_determinism` work and the temporal-sync fragile area
live exactly there. Standard practice is to keep both modes and reserve
real-time for deliberate timing tests.

---

## Recommended order

1. ~~Finding 1 — FIFO staleness~~ **CLOSED 2026-08-12, not a bug.** Left two
   diagnostic-quality tasks behind (T-PERF-METRIC-1, T-PERF-METRIC-2).
2. **Finding 2** — one-run GPU-contention experiment (`use_gpu: false`, read
   `unity_render_frame_dt_ms`). Now the top open item: it is the only remaining
   lever that could move Unity's 150 ms render period without an architecture
   change.
3. **Lockstep** — scope via `/plan-feature`. The structural fix; Finding 2's
   result informs whether perception must also move off the GPU.

Finding 3 is not worth doing until the loop is fast enough for it to register.

## Tooling gap

There is no `/perf` skill. `tools/analyze/cadence_breakdown.py` is a good tool
that this session located only by grepping. It deserves a skill wrapper so
cadence attribution is a first-class move rather than a rediscovery.
