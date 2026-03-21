# Plan: `tools/analyze/cadence_breakdown.py`

**Status:** Implemented — `tools/analyze/cadence_breakdown.py`, PhilViz **Cadence** tab, gate bundle `cadence_breakdown/`.  
**Goal:** One command that turns any recording `.h5` into **cadence attribution** (where time goes each frame), **hotspot stratification**, and **actionable recommendations**—suitable for CLI, CI gates, and later **PhilViz / triage** ingestion.

**Related:** `docs/plans/cadence-performance-plan.md` (background + future `perf_*` fields), `tools/analyze/analyze_sync_acceptance.py` (patterns).

---

## 1. Problem statement

Operators need to distinguish:

| Question | Needs |
|----------|--------|
| Am I **behind** Unity’s camera rate? | Queue depth, frame-id deltas, effective Hz |
| Is **Python work** too heavy? | Time from inputs-ready → control-sent (`e2e_latency_ms`) |
| Is **idle/gap** between frames dominating? | `wait_input` = time from previous control-sent → this inputs-ready |
| Is **Unity** stuttering? | `stream_front_unity_dt_ms` / vehicle `unity_delta_time` vs wall metrics |
| Is **sync skew** (image vs state) large? | `\|front_ready − vehicle_ready\|` |

The HDF5 contract already has most signals; this tool **derives** the rest and **summarizes** them.

---

## 2. Inputs / outputs

### 2.1 CLI (proposed)

```text
python tools/analyze/cadence_breakdown.py [--latest | RECORDING.h5]
  [--output-json PATH]          # machine-readable bundle (schema below)
  [--severe-ms 200]             # wall loop period threshold for "severe" bucket
  [--target-hz HZ]              # optional; default = stack.target_loop_hz from config/av_stack_config.yaml
  [--plot PATH_PREFIX]          # optional PNGs (matplotlib); omit if not installed
  [--pre-failure-only]          # trim to frames before first failure (optional; reuse drive_summary policy if available)
```

Reuse helpers like `find_latest_recording()` from `analyze_sync_acceptance.py` (or import shared `tools/analyze/recording_utils.py` if you dedupe later).

### 2.2 Human-readable report (stdout)

Sections:

1. **Recording** — path, frame count, `stream_sync_policy` from `f.attrs["metadata"]` JSON if present.
2. **Rate health** — mean control rate from `Δ(control/timestamps)` or `Δ(e2e_control_sent_mono_s)`; compare to `--target-hz`.
3. **Severity** — fraction of frames with wall loop period `> severe-ms`; max period.
4. **Per-metric stats table** — P50 / P95 / P99 / max for each derived series (see §3).
5. **Stratified table** — same stats for **all frames** vs **severe** vs **normal** (complement of severe).
6. **Recommendations** — bullet list from rule engine (§5).
7. **Availability** — which optional datasets were missing (degraded mode).

### 2.3 JSON bundle (for debug tools)

Stable keys (version with `schema_version: "cadence_breakdown_v1"`):

- `recording`, `frame_count`, `target_hz`, `severe_ms_threshold`
- `availability`: `{ "e2e_mono": bool, "e2e_latency_ms": bool, "queue_depth": bool, "unity_dt": bool, ... }`
- `rates_hz`: `{ "control_mean": float, "control_p10": ... }` (optional percentiles)
- `severe`: `{ "rate": float, "count": int }`
- `stats_all`, `stats_severe`, `stats_normal`: nested dicts metric → `{count, mean, p50, p95, p99, max}`
- `correlation_hints`: e.g. Pearson or rank corr between `wait_input_ms` and `wall_loop_period_ms` on severe frames (optional, cheap)
- `recommendations`: `[{ "code": "...", "severity": "info|warn|alert", "detail": "..." }, ...]`

PhilViz / `run_gate_and_triage.py` can later load this JSON or call a shared Python function `analyze_cadence(recording: Path) -> dict`.

---

## 3. Data contract (HDF5 paths)

### 3.1 Required (minimal mode)

Aligned on **control row index** `i = 0 .. N-1`:

| Path | Use |
|------|-----|
| `control/timestamps` | Fallback wall period if mono missing; must match control row count |

If **only** timestamps exist, emit **rates + severe** from `np.diff(timestamps)`, and recommendations limited to cadence.

### 3.2 Strong mode (full breakdown)

| Path | Use |
|------|-----|
| `control/e2e_inputs_ready_mono_s` | Sync point for pipeline |
| `control/e2e_control_sent_mono_s` | End of pipeline; `wall_loop_period_ms[i] = (sent[i]-sent[i-1])*1000` for `i>0` |
| `control/e2e_front_ready_mono_s` | Parallel path A |
| `control/e2e_vehicle_ready_mono_s` | Parallel path B |
| `control/e2e_latency_ms` | Validate `≈ (sent - inputs_ready)*1000` |

**Note on older doc typo:** With `inputs_ready = max(front, vehicle)`, expressions like `front_ready - inputs_ready` are **non-positive**. Use instead:

- **`input_sync_skew_ms`** = `abs(front_ready - vehicle_ready) * 1000` (how “out of step” the two inputs are).
- **`wait_input_ms[i]`** = `(inputs_ready[i] - control_sent[i-1]) * 1000` for `i > 0`, else NaN.
- **`pipeline_ms[i]`** = `(control_sent[i] - inputs_ready[i]) * 1000` (should match `e2e_latency_ms[i]` within float noise).

### 3.3 Queue / skip / Unity (attribution)

Aligned on **vehicle** index — **must same length as control** for per-frame join (if lengths differ, document truncation rule: `min(len(control), len(vehicle))` and warn).

| Path | Use |
|------|-----|
| `vehicle/stream_front_queue_depth` | Backlog indicator |
| `vehicle/stream_front_frame_id_delta` | Skips (`> 1` ⇒ catching up / dropping) |
| `vehicle/stream_front_unity_dt_ms` | Unity render/physics dt proxy at sample time |
| `vehicle/stream_front_timestamp_minus_realtime_ms` | Capture vs wall lag |
| `vehicle/unity_delta_time` | If present, cross-check vs stream_front_unity_dt |

Optional: `vehicle/stream_topdown_*` — second table or summarized only (top-down is diagnostic).

---

## 4. Derived series (implement as vectorized NumPy)

For each frame index `i`:

| Name | Formula | Notes |
|------|---------|--------|
| `wall_loop_period_ms` | `(sent[i] - sent[i-1])*1000` | Prefer mono `e2e_control_sent_mono_s`; else `diff(control/timestamps)` |
| `wait_input_ms` | `(inputs[i] - sent[i-1])*1000` | “Idle” between frames from Python’s perspective |
| `pipeline_ms` | `(sent[i] - inputs[i])*1000` | Overwrite with `e2e_latency_ms` when present |
| `input_sync_skew_ms` | `abs(front[i]-vehicle[i])*1000` | Requires both mono series |
| `severe` | `wall_loop_period_ms > severe_ms` | Default 200 ms |

**Sanity check (warn in report):** `max(|pipeline_ms - e2e_latency_ms|)` should be tiny when both exist.

**Effective Hz:** `1000 / mean(wall_loop_period_ms)` over finite samples.

---

## 5. Recommendation rule engine (deterministic)

Emit structured codes so UI/tests can assert without string matching.

| Code | Condition (typical) | Recommendation text (example) |
|------|----------------------|--------------------------------|
| `QUEUE_BACKLOG` | `p95(queue_depth) > 0.8 * max_observed` or mean > threshold (e.g. >80% of observed max) | Bridge/consumer backlog; consider async camera ingest, faster processing, or lower `targetFPS`. |
| `FRAME_SKIPS` | `mean(frame_id_delta) > 1.05` or `p95(delta) >= 2` | Processing slower than Unity’s frame cadence; reduce load or increase skip tolerance intentionally. |
| `WAIT_INPUT_DOMINATES` | On severe frames, `median(wait_input_ms) > median(pipeline_ms)` | Inter-frame gap dominates; investigate **sync HTTP**, **GIL**, **asyncio scheduling**, not CV kernels alone. |
| `PIPELINE_DOMINATES` | On severe frames, `median(pipeline_ms) > median(wait_input_ms)` | Perception/planning/control hot; profile `e2e_latency_ms` spikes; enable future `perf_*` fields. |
| `UNITY_DT_SPIKE` | `p95(unity_dt_ms) > 25` while `wait_input` modest | Unity/GPU/main-thread hitches; check `CameraCapture` drops, quality, resolution. |
| `LAGGING_REALTIME` | Strong negative trend in `timestamp_minus_realtime_ms` | Processing falling behind wall clock vs capture timestamp. |
| `E2E_MISMATCH` | Large `|pipeline_ms - e2e_latency_ms|` | Recording bug or clock reset; treat breakdown as unreliable. |
| `MISSING_E2E` | No mono fields | Re-record with current stack or use timestamp-only degraded metrics. |

Thresholds should be **CLI-tunable** with documented defaults. Start conservative (few false alerts).

---

## 6. Plots (optional `--plot`)

If matplotlib available:

1. **Scatter:** `wait_input_ms` vs `wall_loop_period_ms`, colored by `severe`.
2. **Time series (downsampled):** `queue_depth`, `wall_loop_period_ms`, `pipeline_ms` on shared x = frame index.
3. **Histogram:** `unity_dt_ms` for normal vs severe.

Save as `{prefix}_scatter.png`, etc. Skip gracefully if matplotlib missing.

---

## 7. Testing

- **`tests/test_cadence_breakdown.py`**
  - Build a **minimal synthetic HDF5** in `tmp_path` with known mono timestamps and assert derived `wait_input_ms`, `pipeline_ms`, `severe` counts.
  - Assert JSON schema keys and recommendation codes for a crafted case (e.g. artificial high `wait_input`).
- **Golden smoke:** run CLI against repo golden GT files (read-only):
  - `data/recordings/golden_gt_20260214_oval_latest_20s.h5`
  - `data/recordings/golden_gt_20260214_sloop_latest_45s.h5`
  - `data/recordings/golden_gt_20260215_sloop_latest_30s.h5`  
  Assert exit code 0 and `frame_count > 0` (do not hard-code exact Hz—those files may evolve).

---

## 8. Integration roadmap

| Step | Work |
|------|------|
| **A** | Implement `cadence_breakdown.py` + unit tests |
| **B** | Add one line to `docs/plans/cadence-performance-plan.md` §2.1: “Implemented: `tools/analyze/cadence_breakdown.py`” |
| **C** | Optional: `run_gate_and_triage.py` invokes tool and attaches `cadence_breakdown.json` to gate bundle |
| **D** | Optional: PhilViz route `/api/analysis/cadence` reads same JSON or calls shared analyzer (see cadence-performance-plan §3.3) |
| **E** | When `perf_*` HDF5 fields land, extend §3 with new series and rules (`PERCEPTION_HOT`, `CONTROL_HOT`, …) |

---

## 9. Non-goals (v1)

- Replacing `analyze_sync_acceptance.py` (different SLO contract).
- Pre-failure-only analysis **unless** metadata exists to locate first failure cheaply; v1 can skip or use optional flag with TODO.
- Real-time streaming (offline HDF5 only).

---

## 10. Done criteria

- [x] Script runs on `--latest` and explicit path; `--output-json` produces stable schema.
- [x] Stratified stats (all / severe / normal) on synthetic H5.
- [x] Recommendation codes exercised: `QUEUE_BACKLOG`, `FRAME_SKIPS`, `MISSING_E2E`, etc.
- [x] `pytest tests/test_cadence_breakdown.py` passes.
- [x] PhilViz: `GET /api/recording/<path>/cadence-breakdown`, **Cadence** tab.
- [x] Gate bundle: `cadence_breakdown/<recording_id>.json` + `index.json`.

**Semantics note:** `vehicle/stream_front_unity_dt_ms` in HDF5 is **stream lag** (camera timestamp − Unity time), not `Time.deltaTime`. **UNITY_DT_SPIKE** uses `vehicle/unity_delta_time` → `unity_render_frame_dt_ms`.
