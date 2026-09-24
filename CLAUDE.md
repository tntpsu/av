# AV Stack — Claude Code Instructions

## Project Overview

Autonomous vehicle pipeline running inside a **Unity 2021.3 LTS** simulation.
Handles perception, trajectory planning, and vehicle control for lane-keeping.
Research/development system — not production safety-critical.

**Stage:** Stage 2 — Robustness, Infrastructure & Speed Expansion (active)
**Current milestone:** S2-M1 complete — automated comfort-gate CI
**Last hand-written commit:** 2026-05-06 (`8e95382`). Since then the repo has run
nightly automation only. Confirm this line is still accurate before relying on it.

---

## Agent Protocol

Before starting any work session, read the agent docs:
1. `docs/agent/README.md` — project overview
2. `docs/agent/current_state.md` — active work and known issues
3. `docs/agent/tasks.md` — task list and priorities
4. `docs/agent/architecture.md` — component details (read before touching any layer)

Also read `docs/agent/performance.md` before any cadence/FPS/latency work — it
holds the measured per-stage baseline and the reproduction command
(`tools/analyze/cadence_breakdown.py`). Do not re-derive it by hand.

**These docs are not self-validating — check three clocks first.** Run alongside
the reads above:

```bash
git log -1 --date=short --pretty='%h %ad %s'   # clock 2: last real commit
ls -lt data/reports/*.txt | head -3            # clock 3a: automation output
ls -lt data/recordings/*.h5 | head -3          # clock 3b: automation INPUT
```

**Clock 3b is not sufficient on its own — freshness is PER-TRACK.** A globally
fresh pool can still be frozen for the consumer you care about. On 2026-09-07
the newest recording was that same morning and 28 had landed in 8 nights, yet
all six *lateral* base tracks were still frozen at 2026-08-15: every one of
those 28 was an ACC scenario. The ACC sweep had unfrozen and the lateral sweep
had not. Group by `track_id` and check the tracks the consumer in question
actually reads — use the `latest_per_track()` function from
`tools/nightly/sweep/PROMPT.md` verbatim. Report freshness per consumer
("ACC live since Aug-31, lateral frozen at Aug-15"), never as one global verdict.

Compare against the `**Last updated:**` line in `current_state.md` / `tasks.md`.
If they disagree, the artifacts win — but "the artifacts" means their *measured
data* (mtimes, scores, recording ages), not the narrative notes inside generated
reports. That prose is agent-written and can carry stale claims: on 2026-08-30
`process_health_*.md` asserted "no recordings newer than 2026-05-06" when the
newest was 2026-08-15. Verify any freshness claim against the filesystem before
repeating it — on 2026-09-07 `sweep_report.txt` opened with "newest recording is
2026-08-15 (23 days old)" when the newest was that morning (its conclusion was
right for its own six tracks; its stated evidence was not). Critically: if the
newest recording *for a given track* is more than a few days old, that sweep is
re-analyzing frozen recordings — its scores in `data/reports/sweep_report.txt`
or `acc_sweep_report.txt` are historical, and a nightly "delta" is a baseline
change, not a system change. Say so explicitly rather than reporting it as live.

After completing work, update the relevant `docs/agent/*.md` files — including
their `**Last updated:**` line.

**Testing-policy scope:** the user-level `CLAUDE.md` makes `TESTS.md` +
`/coverage-matrix` a ship gate across projects. **That gate does not apply
here.** This repo predates it and uses the two protocols below (comfort-gate
regression + scoring regression) as its coverage contract instead. Do not block
work on a missing `TESTS.md`.

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
- **Perception:** Segmentation model active but needs retraining (current checkpoint mislabels car hood as lane). Retraining pipeline ready: `./tools/segmentation/train_pipeline.sh`. See `training/TRAINING_GUIDE.md`.

---

## Key Constraints

1. **Always use A/B batch testing (≥5 runs)** to validate config changes — use `run_ab_batch.py`
2. **Read before editing** — never modify source code without reading the relevant file first
3. **Do not modify `av_stack.py`** without understanding the lane gating logic in context
4. **Do not commit** unless explicitly asked
5. **Config changes only** — prefer YAML tuning over code changes when possible
6. **Run comfort gate tests** when touching the analysis pipeline — see Testing Protocol below
7. **Log fixes after commits** — run `/log-fix` after any commit that fixes an issue (feeds `/process-health` Pareto)
8. **Physics-first design** — before adding any config parameter, ask: "What physical quantity does this approximate? Can we compute it directly?" Prefer `sqrt(8R×e_target)` over speed lookup tables.

---

## Workflow Commands

```bash
# Run AV stack (60s on s_loop track — builds player automatically)
./start_av_stack.sh --duration 60 --track-yaml tracks/s_loop.yml

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
1. Run: ./start_av_stack.sh --duration 60 --track-yaml tracks/s_loop.yml
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

> **The 60 / 80 above are drift-detection floors, NOT the pass bar.** They exist
> so this suite fires when scoring silently collapses. The bar for calling a run
> *good* — used by `/e2e`, `/sweep`, and `/validate` verdicts — is **every layer
> ≥ 95**. A run with Trajectory 94.6 passes this regression suite and is still a
> FAIL. Don't quote 60/80 as evidence a run is healthy.

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

## Testing Protocol — ACC Closed-Loop (2026-09-22)

`tests/test_acc_closedloop.py` — 60 tests, ~1.7 s, no Unity. Closes the
longitudinal loop (radar → ACC → owner-resolver → LongitudinalController → safety
clip → point-mass plant) with the **production controllers built from the real
merged YAML**. Harness in `tests/acc_closedloop_harness.py`; calibrated against a
known-good (H5) and a known-bad (G2) sweep result.

**Run when you touch:** `control/acc_controller.py`, the longitudinal path in
`control/pid_controller.py`, `_pf_resolve_longitudinal_target` or the ACC
routing in `av_stack/orchestrator.py`, or any `acc:` / `control.longitudinal:`
YAML key.

```bash
pytest tests/test_acc_closedloop.py -v
```

**Reading the result:** `TestG2StopOnGrade` runs the production config and must
pass. `TestG2Mechanism` / `TestFlatGroundJerkCooldownPin` deliberately run
with the two 2026-09-22 kill-switches in their legacy position
(`cutout_requires_no_lead=False`, `acc_jerk_cooldown_bypass_states=()`) so the
diagnosis stays on record — they must also pass. `TestFixFlags::
test_legacy_flags_reproduce_the_failure` is the rollback proof. **Pattern for
the next ACC failure:** add a `run_<scenario>()` to the harness, land a
strict-xfail reproducer, fix behind a config flag, remove the marker, keep the
mechanism tests. A 90 s Unity A/B is confirmation, not investigation.

**Radar frame (2026-09-22):** Unity's radar range is centre-to-centre — bumpers
touch at ~4.43 m. `ForwardRadarSensor` subtracts `acc.radar_range_offset_m`
(4.43); keep it equal to `scoring_registry.ACC_RADAR_RANGE_OFFSET_M`. The
recorded `radar_fwd_distance_m` is the sensor's FILTERED gap after that offset
(orchestrator.py:9760), and `recording_provenance.radar_range_offset_m` records
which frame a file is in — pre-09-22 files are centre-to-centre. Collisions come
from `vehicle/lead_collision_detected` (the `distance < 0` line is structurally
0); e-stop events exclude EMERGENCY_BRAKE reflex frames. The harness models both
frames (`_legacy_cfg`).

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
