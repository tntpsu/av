# Cadence Performance — Implementation Analysis & Test Plan

**Created:** 2026-03-11
**Status:** Draft — Phase 1 instrumentation not yet implemented
**Related:** S2-M5 Step 2 (after S2-M5 Step 1 track coverage)

---

## 1. Executive Summary

The AV stack runs at ~19.4 Hz (51.6 ms/frame mean) against a 30 Hz Unity target.
**~5% of frames** are "severe" (control_dt > 200 ms).
**Root cause is Python-side, not Unity.** Unity's `unity_delta_time` is stable at 16.7 ms (60 fps smooth). The s_loop golden recording from 2026-02-22 also shows `severe_rate=5.0%` — this pattern is pre-existing and track-independent.

### Key measurement already in HDF5

| Field | Normal frame | Severe frame | Meaning |
|---|---|---|---|
| `wait_input` (derived from e2e timestamps) | ~44 ms | 44–117 ms | Time between consecutive `inputs_ready` events |
| `total_processing` (derived) | ~38 ms | ~39 ms | Perception + planning + control send |
| `unity_delta_time` | 16.7 ms | 16.7 ms | Unity render frame — always smooth |
| `stream_front_queue_depth` | ~116/120 | ~116/120 | Near-full throughout; Python is the bottleneck |

### Conclusion from existing data

The Python processing itself (perception ~0 ms, planning ~1 ms, total ~39 ms) is not the bottleneck. The gap is in `wait_input` — the interval between consecutive frames arriving for processing. At 60 Hz Unity the camera fires every 33 ms, but Python sees them at ~51 ms average. **One of three root causes:**

1. **Python GIL / async I/O**: FastAPI/asyncio event loop starved between frames — camera POST arrives but is blocked behind GIL-held computation from the previous frame
2. **HTTP round-trip overhead**: Unity sends the frame via UnityWebRequest, waits for response; the round-trip (Python process + HTTP response) takes >33 ms → Unity can't send the next frame until Python responds
3. **asyncio + uvicorn thread-pool scheduling**: On macOS, `asyncio` uses kqueue; task scheduling can introduce 10–50 ms wake-up latency when the event loop is idle

---

## 2. Existing Data — What We Can Already Extract

### 2.1 Per-frame Python phase breakdown (already in HDF5)

Using `e2e_inputs_ready_mono_s`, `e2e_front_ready_mono_s`, `e2e_vehicle_ready_mono_s`, `e2e_control_sent_mono_s`:

```
phase_perception = e2e_front_ready - e2e_inputs_ready        (image decode + model)
phase_vehicle    = e2e_vehicle_ready - e2e_inputs_ready       (state parse — parallel)
phase_planning   = e2e_control_sent - max(front, vehicle)     (trajectory + control)
wait_input       = inputs_ready[t] - control_sent[t-1]        (inter-frame idle)
total_processing = control_sent - inputs_ready                (end-to-end Python work)
```

**Immediate action**: Add a `tools/analyze/cadence_breakdown.py` script that:
1. Loads any HDF5 recording
2. Computes the four phases above
3. Stratifies by normal / severe (control_dt > 200 ms)
4. Prints P50/P95/max per phase + scatter plot of wait_input vs control_dt

This script requires no code changes and can run TODAY.

### 2.2 Queue depth and timestamp lag (already in HDF5)

- `stream_front_queue_depth`: near-full (116/120) means images accumulate → Python is slower than Unity's 30 fps camera
- `stream_front_timestamp_minus_realtime_ms`: ~ −128 ms lag from frame capture to processing start
- `stream_front_frame_id_delta` > 1: Python is already skipping frames (processing ~1.27 Unity frames apart)

These fields are already rich enough to characterize the bottleneck without any new instrumentation.

---

## 3. New Instrumentation — What to Add

### 3.1 Python per-layer wall-clock timing (HDF5 fields)

Add 6 new float fields to `ControlCommandData` in `data/formats/data_format.py`:

| HDF5 field | Computed in | Meaning |
|---|---|---|
| `perf_perception_ms` | orchestrator `_pf_run_perception` | Wall clock for segmentation inference + post-process |
| `perf_planning_ms` | orchestrator `_pf_run_trajectory` | Trajectory + speed governor |
| `perf_control_ms` | orchestrator `_pf_send_control` | PP/MPC solve + rate limiter |
| `perf_hdf5_write_ms` | orchestrator, after async flush | HDF5 write overhead (async — should be near 0) |
| `perf_bridge_roundtrip_ms` | bridge/server.py, POST handler | Time from HTTP request received to response sent |
| `perf_wait_input_ms` | derived from e2e timestamps | inter-frame idle (already computable; add as explicit field) |

**Implementation:** In each `_pf_*` method, wrap the body with `_t0 = time.perf_counter()` and store `(time.perf_counter() - _t0) * 1000` in the frame context dict, then wire to `ControlCommandData` in `_pf_build_control_command_data`.

Follows the existing 6-location HDF5 registration pattern:
1. `create_dataset` in `recorder.py`
2. list init
3. per-frame append
4. resize loop
5. write
6. `ControlCommandData` construction in `orchestrator.py`

### 3.2 Unity-side timing (new fields in AVBridge vehicle state JSON)

Add to the `VehicleState` C# class in `AVBridge.cs`:

| JSON field | C# source | Meaning |
|---|---|---|
| `unity_delta_time_ms` | `Time.deltaTime * 1000` | Render frame duration (raw) |
| `unity_smooth_delta_time_ms` | `Time.smoothDeltaTime * 1000` | Unity's EMA-smoothed frame time |
| `unity_realtime_since_startup` | `Time.realtimeSinceStartup` | Monotonic clock at state-send time |
| `unity_physics_delta_time_ms` | `Time.fixedDeltaTime * 1000` | Physics step size (diagnostic) |
| `unity_state_send_duration_ms` | `(Time.realtimeSinceStartup - updateStartRealtime) * 1000` | State + JSON serialize + HTTP time from Unity side |

These fields flow through `bridge/server.py` to `ControlCommandData` (new `VehicleStateData` fields) and are recorded in HDF5.

**Why this helps:** If `unity_smooth_delta_time_ms` spikes, it's Unity-side. If it stays flat while `control_dt` spikes, the problem is confirmed Python-side. Also: `unity_state_send_duration_ms` directly measures whether the HTTP send from Unity is the bottleneck.

### 3.3 PhilViz — Cadence Triage Panel

New panel in `tools/debug_visualizer/backend/`:
**File:** `tools/debug_visualizer/backend/cadence.py`
**Route:** `/api/analysis/cadence` (added to `server.py`)

#### Panel layout (4 cards)

**Card 1 — Frame Rate Health**
```
Control rate:   19.4 Hz  (target 30 Hz)   [color: warn if <25, alert if <15]
Severe frames:   5.0%    (target <2%)      [color: warn if >2, alert if >5]
Unity rate:     60.0 Hz  (smooth)          [color: always green if >55]
Queue depth:   116/120   (frames)          [color: alert if >100]
```

**Card 2 — Per-Phase Breakdown (P50 / P95)**
```
                P50     P95    MAX
wait_input      44ms    97ms   300ms   ← bottleneck indicator
perception       0ms     1ms    12ms
planning         1ms     3ms    15ms
control          0ms     1ms     5ms
total_proc      38ms    42ms    55ms
```

**Card 3 — Severe Frame Inspector**
Table of the 10 most recent severe frames with: frame #, control_dt, wait_input, total_proc, perception_ms, was it the first severe in a burst?

**Card 4 — Lag & Skew**
```
Frame-to-frame ID delta:  1.27   (1.0 = no skip, >1 = Python skips frames)
Timestamp lag:          -128ms   (capture-to-process latency)
```

#### PhilViz integration checklist
- [ ] `backend/cadence.py` — computes all 4 cards from latest HDF5
- [ ] `server.py` — registers `/api/analysis/cadence` route
- [ ] `index.html` — adds "Cadence" tab to nav bar
- [ ] `visualizer.js` — fetches + renders cadence panel
- [ ] Restart PhilViz after any of the above changes

---

## 4. Root Cause Experiments

### Experiment A: HTTP round-trip hypothesis (no code change needed)

**Method:** Compare `unitySendRealtime` (timestamp when Unity sends the state) with `e2e_inputs_ready_mono_s` (when Python receives the camera frame). If the gap is >33 ms on severe frames, the HTTP layer is the bottleneck.

Currently `unitySendRealtime` is transmitted in the vehicle state JSON. Python reads it in `bridge/server.py`. **Add logging** to print `time.monotonic() - unity_send_realtime` on the Python side for any frame where `control_dt > 200 ms`.

**Expected result:** ~5–10 ms round-trip on normal frames; if severe frames show >50 ms here, HTTP is implicated.

### Experiment B: GIL hypothesis

**Method:** Profile one 60s run using `py-spy record --subprocesses` while the AV stack runs:
```bash
py-spy record -o /tmp/avstack.svg --pid $(pgrep -f "uvicorn") --duration 60
```
If the flame graph shows long stalls in the numpy/torch path during `wait_input`, GIL contention is confirmed.

### Experiment C: asyncio event loop lag

**Method:** In `bridge/server.py`, add:
```python
import asyncio, time
@app.middleware("http")
async def log_event_loop_lag(request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    if time.perf_counter() - t0 > 0.15:
        logger.warning("SLOW_REQUEST %.1fms %s", (time.perf_counter()-t0)*1000, request.url.path)
    return response
```
If slow requests cluster on `/api/camera` but NOT on `/api/vehicle/state`, it's camera-specific processing.

### Experiment D: Decoupled camera receiver (architectural fix candidate)

**Hypothesis:** The current flow is synchronous — Unity POSTs the camera frame and **waits** for a 200 response before it can send the next frame. This creates a 1-frame pipeline depth.

**Proposed fix:** Camera receiver returns 202 Accepted immediately (enqueue the frame), and processing runs in a background task. This decouples Unity's 30 Hz send rate from Python's processing speed.

**Risk:** Frame reordering is possible if processing varies. Must use monotonic sequence numbers and always process the latest frame (drop older if a newer arrives during processing).

**A/B test:** A = current (synchronous), B = async enqueue. Compare control_dt P95, severe_rate, lateral RMSE over 5 runs each.

---

## 5. Test Plan

### Phase 1 (no code changes, ~1 day)

- [ ] **T1.1** Write `tools/analyze/cadence_breakdown.py` — phase extraction from any HDF5
- [ ] **T1.2** Run on 3 existing recordings (hairpin_15, mixed_radius, s_loop golden) to confirm Python-side root cause
- [ ] **T1.3** Run Experiment A (HTTP round-trip from logs)
- [ ] **T1.4** Run Experiment B (py-spy profiling on a fresh 60s run)
- [ ] **T1.5** Document findings and confirm/refute the 3 hypotheses

### Phase 2 (new HDF5 fields, ~0.5 day)

- [ ] **T2.1** Add `perf_perception_ms`, `perf_planning_ms`, `perf_control_ms`, `perf_wait_input_ms` to HDF5 (6-location pattern)
- [ ] **T2.2** Add `unity_delta_time_ms`, `unity_smooth_delta_time_ms`, `unity_state_send_duration_ms` to vehicle state JSON
- [ ] **T2.3** Run 1 validation run; confirm new fields populate correctly
- [ ] **T2.4** `pytest tests/ -v` — confirm no regressions (fields follow 6-location rule)

### Phase 3 (PhilViz panel, ~1 day)

- [ ] **T3.1** Implement `backend/cadence.py` with 4-card structure
- [ ] **T3.2** Add `/api/analysis/cadence` route
- [ ] **T3.3** Add Cadence tab to PhilViz UI
- [ ] **T3.4** Verify renders correctly on an existing recording
- [ ] **T3.5** Verify severe frame inspector shows correct top-10 frames

### Phase 4 (fix + A/B test, ~1 day)

Based on Phase 1 findings, implement the winning fix (most likely Experiment D — async enqueue):

- [ ] **T4.1** Implement async camera receiver in `bridge/server.py`
- [ ] **T4.2** A/B test: 5 runs each configuration on s_loop
- [ ] **T4.3** Confirm: severe_rate drops from ~5% to <2%, no RMSE regression
- [ ] **T4.4** If A/B confirms improvement, promote to both configs
- [ ] **T4.5** Full comfort gate test suite — `pytest tests/test_comfort_gate_replay.py -v`

---

## 6. Success Criteria

| Metric | Current | Target |
|---|---|---|
| Control rate | 19.4 Hz | ≥ 25 Hz |
| Severe frame rate (control_dt > 200 ms) | ~5% | < 2% |
| Control_dt P95 | ~100 ms | < 60 ms |
| Unity delta_time P95 | 16.7 ms (no change expected) | ≤ 18 ms |
| Lateral RMSE | baseline per track | no regression (±0.02 m) |
| Comfort gate score | per track baseline | no regression |

---

## 7. What NOT to Do

- **Do not tune Unity quality settings** — already at level 0 (lowest). No headroom there.
- **Do not lower Unity target FPS** — Python already running behind 30 Hz; reducing Unity FPS would starve Python further
- **Do not add Python threads naively** — PyTorch is not thread-safe; perception must stay single-threaded
- **Do not optimize perception model** — already ~0 ms; not the bottleneck
- **Do not add more HDF5 fields** without following the 6-location rule

---

## 8. Files to Touch

| File | Change |
|---|---|
| `data/formats/data_format.py` | Add 6 `perf_*` fields to `ControlCommandData` |
| `data/recorder.py` | 6-location registration for new fields |
| `av_stack/orchestrator.py` | Timer wraps in `_pf_run_perception`, `_pf_run_trajectory`, `_pf_send_control` |
| `unity/AVSimulation/Assets/Scripts/AVBridge.cs` | Add `unity_delta_time_ms`, `unity_smooth_delta_time_ms`, `unity_state_send_duration_ms` to `VehicleState` class and populate them |
| `bridge/server.py` | Read new Unity timing fields; optionally Experiment D async enqueue |
| `tools/analyze/cadence_breakdown.py` | New script — phase analysis, no HDF5 writes |
| `tools/debug_visualizer/backend/cadence.py` | New PhilViz panel backend |
| `tools/debug_visualizer/server.py` | Register cadence route |
| `tools/debug_visualizer/index.html` | Cadence tab |
| `tools/debug_visualizer/static/visualizer.js` | Cadence panel render |
