# Plan B — Pareto perf (items 4–6): mask postprocess, recorder tails, architecture

**Goal:** Further reduce CPU spikes and tail latency after Plan A is in place. **No feature removal** — same outputs, faster or smoother implementations.

**Prerequisite:** Plan A merged and E2E baseline captured (`tools/analyze/cadence_breakdown.py`).

### Priority note — defer Plan B while the consumer is backlog-bound

If `cadence_breakdown` shows **`QUEUE_BACKLOG`** (front queue depth pinned near max), **do not invest in Plan B yet** (especially **B1** perception micro-optimizations). A full frame queue means the **consumer cannot drain frames**; shaving milliseconds inside perception does not fix sustained backlog. **Focus on the consumer path first:** ingest scheduling, async/deferred work, bridge/client loop alignment with Unity `targetFPS`, and Plan A transport (raw camera, device placement). Revisit **B2** (recorder flush tails) only when tails correlate with flush logs *after* queue health improves; **B3** when `wait_input` / HTTP remains dominant with healthy queue depth.

---

## B1. Vectorize / tune perception mask → lane (`perception/inference.py`)

### Problem
- `_mask_to_lane_data` uses **Python `for y in range(min_row, max_row, row_step)`** with `np.where` per conceptual row — scales with image height and branchiness.

### Design
- **Vectorized row statistics:** for each lane label, use **stride slicing** or `np.add.reduceat` / sorting by row index to compute **per-row median x** without a Python loop over all `y` (or restrict loop to **sparse rows** where mask has pixels).
- Keep **RANSAC** optional; when off, fast path only.
- Config already exposes **`segmentation_ransac_*`** and row band ratios — add optional **`segmentation_row_step`** if not present (larger step = less work).

### Tests
- **Golden parity:** same synthetic mask → **lane coefficients** (or fit points) **within epsilon** of current implementation on fixed seed.
- **Perf test (optional):** benchmark before/after on synthetic 480×640 mask (not flaky in CI — mark `@pytest.mark.slow` or skip).

### Expected gain
- Often **~1–5 ms** on CPU at 480p when row loop was hot.

### Risk
- Numerical tie-breaking in median vs current loop — document tolerance in test.

---

## B2. Recorder / HDF5 flush tuning (`data/recorder.py`)

### Problem
- Batched flush every **`flush_every`** frames still does **`h5_file.flush()`** and large writes → occasional **>50–200 ms** spikes (`[RECORDER_FLUSH_SLOW]`).

### Design (pick any subset)
1. **Config `recorder.flush_every`** (expose in YAML if not already) — increase for benchmark runs (e.g. 60–120) when full temporal resolution not needed.
2. **Compression / chunking:** HDF5 `compression="gzip", compression_opts=1` (or lzff if available) on **large** datasets — trade CPU for smaller I/O.
3. **Async flush backpressure:** if queue depth > threshold, **drop non-critical datasets** for one flush (feature-flagged) — only if product agrees (borderline “functionality”; prefer 1–2 first).

### Tests
- **Integration:** write N synthetic frames, assert file readable and **frame count** matches.
- **Regression:** existing recorder tests; add test that custom `flush_every` is respected.

### Expected gain
- Smoother **p95/p99** loop time; mean Hz may rise slightly if recorder was tail-bound.

---

## B3. Architecture: in-process bridge, async client, or shared memory

### Problem
- Separate **uvicorn** process + **sync `requests`** stack → serialization + localhost stack; aligns with **`wait_input`** dominance in some recordings.

### Design options (increasing invasiveness)
1. **httpx async + shared event loop** in a single “driver” process (still two processes unless bridge embedded).
2. **Embed FastAPI/uvicorn in-thread** with stack (one process) — **raw camera** becomes a function call (zero HTTP for camera).
3. **Shared memory ring buffer** for frames (POSIX / `multiprocessing.shared_memory`) — Unity bridge writer + stack reader.

### Tests
- **Contract tests:** same HTTP API surface if bridge remains addressable; or **smoke test** script for embedded mode.
- **E2E:** cadence + lane-keeping score unchanged within tolerance.

### Expected gain
- Highly variable; can overlap with Plan A raw path; best case **another ~2–10 ms** + fewer scheduling gaps.

### Risk
- Highest **engineering + debug** cost; do after A + B1 + B2 if metrics still show **wait_input** large.

---

## Suggested order

1. **B1** (localized, testable parity).  
2. **B2** (config + recorder tests).  
3. **B3** (prototype branch only after data says HTTP is still the bottleneck).

---

## Done criteria (Plan B overall)

- [ ] Mask path: tests pass with **parity bounds** documented.  
- [ ] Recorder: configurable flush / compression; no corrupt HDF5.  
- [ ] Architecture: design doc + spike PR OR explicit defer with metric gate.
