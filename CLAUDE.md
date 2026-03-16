# AV Stack — Claude Code Instructions

## Project Overview

Autonomous vehicle pipeline running inside a **Unity 2021.3 LTS** simulation.
Handles perception, trajectory planning, and vehicle control for lane-keeping.
Research/development system — not production safety-critical.

**Stage:** Stage 2 — Robustness, Infrastructure & Speed Expansion (active)
**Current milestone:** S2-M1 complete — automated comfort-gate CI

---

## Agent Protocol

Before starting any work session, read the agent docs:
1. `docs/agent/README.md` — project overview
2. `docs/agent/current_state.md` — active work and known issues
3. `docs/agent/tasks.md` — task list and priorities
4. `docs/agent/architecture.md` — component details (read before touching any layer)

After completing work, update the relevant `docs/agent/*.md` files.

---

## Critical Files — Handle With Care

| File | Lines | Notes |
|---|---|---|
| `av_stack.py` | 5,420 | Main orchestrator — DO NOT edit without reading relevant sections first. Lane gating logic has wide blast radius. |
| `config/av_stack_config.yaml` | ~17 KB | 100+ tuning params. All changes require A/B batch testing (≥5 runs). |
| `control/pid_controller.py` | 3,901 | Active controller. Pure Pursuit is the active lateral mode. |
| `trajectory/inference.py` | 1,143 | Rule-based planner. EMA smoothing and speed planning are curvature-sensitive. |
| `perception/inference.py` | 489 | Lane detection. CV fallback is de facto active path. |
| `bridge/server.py` | — | FastAPI Unity↔Python bridge. |
| `data/recorder.py` | — | Async HDF5 recording. |
| `tools/scoring_registry.py` | ~60 | Single source of truth for 25+ scoring thresholds. Update here, not in consumers. |

---

## Active Control Configuration

- **Lateral mode:** Pure Pursuit (not PID, not Stanley, not MPC)
- **PID pipeline:** Bypassed in PP mode (S1-M35)
- **Target speed:** 12.0 m/s
- **Perception:** CV fallback active (segmentation model likely untrained)

---

## Key Constraints

1. **Always use A/B batch testing (≥5 runs)** to validate config changes — use `run_ab_batch.py`
2. **Read before editing** — never modify source code without reading the relevant file first
3. **Do not modify `av_stack.py`** without understanding the lane gating logic in context
4. **Do not commit** unless explicitly asked
5. **Config changes only** — prefer YAML tuning over code changes when possible
6. **Run comfort gate tests** when touching the analysis pipeline — see Testing Protocol below

---

## Workflow Commands

```bash
# Run AV stack (60s on s_loop track)
./start_av_stack.sh --launch-unity --unity-auto-play --duration 60

# Ground truth baseline
./start_ground_truth.sh --track-yaml tracks/s_loop.yml --duration 60

# Analyze latest recording (primary summary)
python tools/analyze/analyze_drive_overall.py --latest

# Frame inspector / PhilViz (port 5001)
python tools/debug_visualizer/server.py

# A/B config comparison (minimum 5 runs each)
python tools/analyze/run_ab_batch.py --config-a A.yaml --config-b B.yaml --runs 5

# Full test suite
pytest tests/ -v
```

---

## Diagnostic Cycle

```
1. Run: ./start_av_stack.sh --launch-unity --unity-auto-play --duration 60
2. Analyze: python tools/analyze/analyze_drive_overall.py --latest
3. Inspect frames: python tools/debug_visualizer/server.py
4. Layer isolation: python tools/analyze/counterfactual_layer_swap.py
5. A/B test config change: python tools/analyze/run_ab_batch.py --runs 5
6. Update config, repeat
```

---

## Architecture Summary

```
Unity (C#)
  Camera frame (30 FPS) ──► FastAPI Bridge (bridge/server.py)
  Vehicle state ──────────►        │ HTTP
                                    ▼
                            av_stack.py (orchestrator)
                                    │
             ┌──────────────────────┼────────────────────┐
             ▼                      ▼                     ▼
        Perception            Trajectory              Control
    perception/               trajectory/           control/
    inference.py              inference.py       pid_controller.py
                                    │
                            HDF5 Recorder (async)
                            data/recorder.py → data/recordings/*.h5
```

Per-frame data flow: Unity → bridge → perception → EMA gating (av_stack.py) → trajectory → Pure Pursuit → rate/jerk limit → safety clip → control command back to Unity → async HDF5 write.

---

## Comfort Gates (S1-M39)

| Metric | Target |
|---|---|
| Accel P95 | ≤ 3.0 m/s² |
| Jerk P95 | ≤ 6.0 m/s³ |
| Lateral RMSE | ≤ 0.40 m |
| Centered frames | ≥ 70% |
| Emergency stops | 0 |

---

## Testing Protocol — Comfort Gate Regression

`tests/test_comfort_gate_replay.py` — 32 tests, two tiers, no Unity required.

**Run Tier 1 (synthetic, ~0.3 s) when you touch:**
- `tools/drive_summary_core.py` — metric computation
- `tools/analyze/run_gate_and_triage.py` — gate evaluation / validity logic
- `data/recorder.py` — HDF5 field names or schema

```bash
pytest tests/test_comfort_gate_replay.py -v -k "Synthetic or Boundary"
```

**Run the full suite (includes golden recording regression, ~0.5 s) when:**
- Merging any branch that touches the analysis pipeline
- Promoting a config change to a new milestone
- Registering new golden recordings after live validation runs

```bash
pytest tests/test_comfort_gate_replay.py -v
```

**To register new golden recordings** after a milestone validation:
1. Update `tests/fixtures/golden_recordings.json` with the new filenames and scores
2. Update `BASELINE_SCORES` in `tests/conftest.py`
3. Update `tests/fixtures/scoring_baselines.json` with new metric values
4. Re-run the full suite to confirm green

---

## Testing Protocol — Scoring Regression (T-033)

`tests/test_scoring_regression.py` — 25 tests (5 per track), no Unity required.
Catches silent scoring drift from changes to the scoring pipeline or config.

**Run when you touch:**
- `tools/drive_summary_core.py` — scoring formula, layer weights, penalty computation
- `config/av_stack_config.yaml` or any `config/mpc_*.yaml` — tuning parameters
- `control/pid_controller.py` or `trajectory/inference.py` — control/trajectory logic
- `tests/conftest.py` — baseline scores or tolerances

```bash
pytest tests/test_scoring_regression.py -v
```

**What it checks (per golden recording):**
1. Overall score within `±SCORE_TOLERANCES[track]` of frozen baseline
2. Every layer score (Perception, Trajectory, Control) ≥ 60 (not red)
3. Trajectory layer ≥ 80 (not yellow — prevents cap regressions)
4. Comfort gates: accel P95 ≤ 3.0, jerk P95 ≤ 6.0, e-stops = 0
5. Key metric deltas within frozen tolerances (adj RMSE ±0.03, accel ±0.5, jerk ±1.0)

**Baselines file:** `tests/fixtures/scoring_baselines.json` — frozen per-track metrics.
**Report artifact:** `data/reports/gates/latest_scoring_regression.json` — written after test run.

**To update baselines** after a scoring formula change:
1. Run `pytest tests/test_scoring_regression.py -v` — note actual values in output
2. Update `tests/fixtures/scoring_baselines.json` with new metric values
3. Update `BASELINE_SCORES` in `tests/conftest.py` if overall scores changed
4. Re-run to confirm green

**Config change detector** (informational, runs in CI on PRs):
```bash
python tools/ci/check_config_regression.py --base origin/main
python tools/ci/check_config_regression.py --base HEAD~1  # local check
python tools/ci/check_config_regression.py --critical-only  # scoring-critical params only
```

---

## Fragile Areas

- **`av_stack.py` lane gating** — `clamp_lane_center_and_width()`, `apply_lane_ema_gating()`, `blend_lane_pair_with_previous()` are deeply interdependent
- **Curvature measurement** — errors cascade to speed planning AND steering
- **Perception stale fallback** — multi-condition blending logic; can fail silently
- **Temporal sync** — camera frames and vehicle state have different timestamps
- **Parameter surface** — 100+ config params; untested combos can produce unstable behavior

---

## Tech Stack

| Layer | Tools |
|---|---|
| Simulation | Unity 2021.3 LTS (C#) |
| Core stack | Python 3 |
| ML / Vision | PyTorch ≥ 2.0, OpenCV ≥ 4.8, scikit-learn |
| API bridge | FastAPI + uvicorn (async) |
| Data storage | HDF5 via h5py |
| Numerics | NumPy, SciPy, Pandas |
| Testing | pytest (98 test files, 126+ passing) |
| Visualization | PhilViz (Flask, port 5001), Matplotlib |
| Config | YAML (`config/av_stack_config.yaml`) |
