# Plan A — Pareto perf (items 1–3): raw camera, Torch device (MPS/CUDA), top-down / cadence

**Status (2026-02-17):** **Implemented in repo** — raw `GET /api/camera/latest/raw` + `/next/raw`, client `use_raw_camera` + `stack.use_raw_camera_transport`, `perception.prefer_mps` + `resolve_torch_device`, `stack.target_loop_hz`, `stack.topdown_recording_interval_frames`. Tests: `tests/test_bridge_raw_camera.py`, `tests/test_torch_device_resolve.py`, `tests/test_stack_config_yaml.py`.

**Consumer ingest (2026-03-21):** `stack.camera_prefetch` (daemon thread + dedicated `Session` overlaps camera HTTP with the stack loop), `stack.clear_bridge_camera_queue_on_start` + `POST /api/camera/drain_queue`, configurable FIFO cap via `stack.bridge_max_camera_queue` / env `AV_BRIDGE_MAX_CAMERA_QUEUE` (start scripts export from YAML), deque overflow increments `camera_drop_count` for `stream_front_drop_count`. Tests: `tests/test_bridge_ingest_consumer.py`.

**Goal:** Cut redundant JPEG work, use Apple Silicon GPU when available, and reduce optional top-down + pacing load **without removing features** (same algorithms, same outputs when enabled).

**Related analysis:** conversation Pareto (JPEG round-trip, GPU, top-down rate), `docs/plans/cadence-performance-plan.md`.

---

## A1. Raw camera transport (remove decode→encode→decode)

### Problem
- Bridge decodes JPEG → `ndarray`, then `/api/camera/latest` **re-encodes JPEG + base64**; Python client **decodes again**.

### Design
- Add **`GET /api/camera/latest/raw`** (and **`GET /api/camera/next/raw`** for FIFO parity):
  - Body: **C-contiguous `uint8`** RGB row-major (`H*W*3` bytes).
  - Metadata in **ASCII response headers** (no JSON body):  
    `X-AV-Shape: H,W,C`, `X-AV-Timestamp`, `X-AV-Frame-Id`, `X-AV-Queue-Depth`, `X-AV-Queue-Capacity`, `X-AV-Drop-Count`, `X-AV-Decode-In-Flight`, `X-AV-Latest-Age-Ms`, `X-AV-Last-Arrival-Time`, `X-AV-Last-Realtime`, `X-AV-Last-Unscaled`, `X-AV-Camera-Id`.
- **`UnityBridgeClient`**: if `use_raw_camera` (default **on**, overridable via config/env), call raw endpoint first; on **404** or parse error, **fall back** to existing JSON+JPEG path (old bridge binaries).
- **External tools** (browsers, curl scripts) keep using `/api/camera/latest` JSON unchanged.

### Implementation steps
1. `bridge/server.py`: implement raw handlers; reuse queue item shape for `/next/raw`.
2. `bridge/client.py`: `_get_latest_camera_raw()`, integrate into `get_latest_camera_frame_with_metadata` and `get_next_camera_frame`.
3. `av_stack/orchestrator.py`: pass `use_raw_camera` from config into `UnityBridgeClient`.
4. `config/av_stack_config.yaml`: `stack.use_raw_camera_transport: true`.
5. Server help text / README one-liner.

### Tests
- **Unit:** `tests/test_bridge_raw_camera.py` — set module globals + `TestClient`, assert raw bytes round-trip shape `(4,5,3)` and headers match seeded timestamp/frame_id.
- **Unit:** client parser tests with `requests.Response` mock (optional).
- **Regression:** existing bridge tests if any; run `pytest tests/test_bridge_raw_camera.py tests/test_torch_device_resolve.py` (and full suite if fast).

### Success criteria
- Pixel identity: raw array **equals** JSON path on same frame (allow small JPEG path to differ; compare raw vs raw only, or compare raw vs decode of same buffer).
- Fallback: with raw disabled or 404, JSON path still works.

### Expected gain
- Often **~5–25 ms CPU/frame** saved (machine-dependent); helps **queue drain** and **effective Hz**.

---

## A2. Torch device: CUDA **or** Apple MPS (`mps`)

### Problem
- `LaneDetectionInference` used **`cuda` else `cpu`**, so **MacBook Apple Silicon** stayed on CPU for segmentation.

### Design
- New helper `perception/device_utils.py`: `resolve_torch_device(use_gpu: bool, prefer_mps: bool) -> torch.device`
  - Order: `cpu` if `not use_gpu`; else **`cuda`** if available; else **`mps`** if `prefer_mps` and `torch.backends.mps.is_available()`; else **`cpu`**.
- Wire **`perception.use_gpu`** (default true) and **`perception.prefer_mps`** (default true) from YAML into `LaneDetectionInference`.
- **No autocast/FP16 in Plan A** (MPS/CUDA mixed support); optional follow-up.

### Risks / mitigations
- Rare MPS op gaps: if model fails at runtime, document **set `prefer_mps: false`** to force CPU for that env.
- Intel Mac: MPS unavailable → CPU (unchanged).

### Tests
- `tests/test_torch_device_resolve.py`: monkeypatch `torch.cuda.is_available` / `torch.backends.mps.is_available` and assert chosen device.

### Expected gain
- On M1/M2/M3, segmentation often **much faster than CPU** (varies by model); **no change** on CUDA Linux.

---

## A3. Top-down fetch interval + Python **loop Hz** (match stack capability)

### Problem
- Every recorded frame can trigger **another** full camera fetch for `top_down` (JPEG path today) → doubles bridge load for diagnostics.
- Python loop assumes **30 Hz** `frame_interval` while stack may only sustain **~19 Hz**.

### Design
1. **`stack.topdown_recording_interval_frames`** (int):
   - **`1`**: current behavior (fetch every recording frame).
   - **`N > 1`**: fetch top-down every N-th **record** tick only; other frames record **without** fresh top-down (HDF5 top-down slots empty or stale policy as today when missing).
   - **`0`**: **never** fetch top-down for recording (saves bandwidth; PhilViz top-down column empty for that run).
2. **`stack.target_loop_hz`** (float, default **30**): sets `self.target_fps` / `frame_interval` in `AVStack` so Python **does not spin faster** than configured; **align with Unity `CameraCapture.targetFPS`** (e.g. **20–24**) when stack cannot hold 30 Hz.

### Unity note
- Lower **`CameraCapture.targetFPS`** on Unity side to the **same** value as `target_loop_hz` for steady cadence and smaller queues.

### Tests
- **Unit:** logic for “should fetch topdown this tick” (pure function or small orchestrator method tested via minimal mock).
- **Config:** load yaml fragment and assert `AVStack` picks up `target_loop_hz` (optional integration test with mocked bridge).

### Expected gain
- Top-down interval: **~(1 − 1/N)** of top-down HTTP+decode cost** on recording path.
- Lower loop Hz: **less busy-wait**, **fewer duplicate polls**, **backpressure** aligned with reality (does not speed up perception; reduces wasted wakeups).

---

## Cadence guidance (30 Hz vs slower)

| Unity `targetFPS` | `stack.target_loop_hz` | When |
|-------------------|-------------------------|------|
| 30 | 30 | Machine sustains ≤33 ms end-to-end. |
| 24 | 24 | Comfortable middle ground. |
| 20 | 20 | Close to ~19 Hz sustained (your recent run). |

**Rule of thumb:** set both to the **same** number; use **`cadence_breakdown.py`** after changes to verify mean Hz and queue depth.

---

## MacBook GPU checklist

1. **Apple Silicon** (M1+): install **PyTorch build with MPS** (official wheels). `prefer_mps: true`.
2. **Intel Mac:** no MPS → CPU or external GPU not typical for PyTorch on Mac.
3. Verify: log line on startup should show **`mps`** when segmentation loads.

---

## Rollout order (implementation)

1. Raw camera (server + client + config) + tests.  
2. Device resolve + perception wiring + tests.  
3. Stack YAML (`target_loop_hz`, `topdown_recording_interval_frames`, `use_raw_camera_transport`) + orchestrator wiring + tests.

Then **E2E:** Unity + bridge + stack, compare `cadence_breakdown.py` before/after.
