# Per-layer performance timings (HDF5) — implementation spec

**Status:** Implemented (replaces T2.1 checklist item in `cadence-performance-plan.md`).  
**Related:** `docs/plans/cadence-performance-plan.md` §3.1, `tools/analyze/cadence_breakdown.py`.

## Purpose

Per-frame wall times (Python `time.perf_counter()`, milliseconds) so recordings answer:

- Where does **`pipeline_ms` / `e2e_latency_ms`** go (perception vs planning vs control vs recorder ingest)?
- How does recorded **`perf_wait_input_ms`** compare to cadence-derived **`wait_input_ms`**?

## HDF5 datasets (under `control/`)

| Dataset | Source | Definition |
|--------|--------|------------|
| `perf_perception_ms` | `AVStack._process_frame` | `_pf_run_perception` only (inference + perception post in that call). |
| `perf_planning_ms` | same | `_pf_run_speed_governor` + `_pf_plan_trajectory`. |
| `perf_control_ms` | same | `_pf_compute_steering` + `_pf_apply_safety` + `_pf_send_control` (MPC/PID, limits, control + trajectory enqueue to bridge). |
| `perf_hdf5_write_ms` | `AVStack._record_frame` | Wall time for `DataRecorder.record_frame` only (buffer append + periodic queue handoff). **Does not** include background flush worker HDF5 I/O. |
| `perf_wait_input_ms` | `AVStack._process_frame` | `max(0, (inputs_ready_mono_s − prev_control_sent_mono_s) * 1000)`. First frame / missing mono → NaN. |

## Intentionally not included in v1

- **Lane geometry / gating / health / `perc_out` build** — time between perception and governor is *not* in the three layer buckets; see “Unaccounted time” below.
- **`perf_bridge_roundtrip_ms`** — requires instrumentation in `bridge/server.py` (FastAPI handler timing); left for a follow-up.

## Unaccounted time (sanity check)

For a given frame, the following should approximately hold:

\[
\text{pipeline\_ms} \approx \text{perf}_{\text{pre}} + \text{perf\_perception} + \text{perf\_planning} + \text{perf\_control} + \text{perf\_hdf5}
\]

where **`perf_pre`** is everything between `_pf_run_perception` return and `_pf_run_speed_governor` (geometry, gating, health, perception output struct). If the sum of the five stored metrics is much smaller than `e2e_latency_ms`, investigate **`perf_pre`** or add a future `perf_lane_pipeline_ms` field.

## Code touchpoints (6-location pattern)

1. `data/formats/data_format.py` — `ControlCommand` fields  
2. `data/recorder.py` — `create_dataset` loop + list init + append + column list + slice write  
3. `av_stack/orchestrator.py` — `_process_frame` timers, `latency_ctx`, `_record_frame` `record_frame` timing + `ControlCommand` kwargs  
4. `tools/analyze/cadence_breakdown.py` — optional `stats_all` entries when datasets exist  

## Optional: cProfile (function-level)

Does not replace HDF5 layer timings. Use:

```bash
./start_av_stack.sh --duration 30 --profile-cprofile
# or explicit path:
python av_stack/orchestrator.py --record --profile-cprofile /tmp/run.prof
```

Then: `python -m pstats /tmp/run.prof` → `sort cumulative` → `stats 40`.

**Note:** Profiling adds overhead; use short runs and compare trends, not absolute Hz.
