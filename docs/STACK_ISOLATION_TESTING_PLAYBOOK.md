# Stack Isolation Testing Playbook

This playbook defines the standard method to evaluate and debug the AV stack
without over-indexing on one layer.

It maps current tools to each layer, identifies missing capabilities, and
defines a repeatable workflow for root-cause isolation.

---

## Why This Exists

Closed-loop AV failures can look like a controller problem while actually being:

- an upstream perception error,
- a trajectory generation instability,
- a timing/latency issue,
- or a true control authority limitation.

Tuning gains without layer isolation can improve one metric while making
real-world behavior worse.

Execution checklist:
- See `docs/TODO.md` under "Stack Isolation TODOs (Priority Execution Plan)" for
  staged implementation tasks and acceptance tests.

---

## Layer Isolation Matrix (Current State)

### 1) Perception Layer

- **Available now**
  - `tools/replay_perception.py` (offline replay on recorded frames)
  - `tools/reprocess_recording.py` (legacy replay path)
  - Segmentation/CV mode switches in `av_stack.py` and start scripts
- **What this answers**
  - "Did lane outputs change due to perception code/model changes?"
  - "Are failures reproducible offline on the same input frames?"
- **Current gap**
  - No strict "perception-only" replay contract output artifact used by downstream
    tools as a frozen input.

### 2) Trajectory Layer

- **Available now**
  - Trajectory quality diagnostics in PhilViz (`trajectory_vs_steering`)
  - `tools/analyze/analyze_curve_entry_feasibility.py` and backend equivalent
- **What this answers**
  - "Is the requested reference path feasible/consistent before failure?"
- **Current gap**
  - No first-class trajectory lock/replay mode in live stack.
  - No explicit trajectory-only harness that consumes frozen perception outputs.

### 3) Control Layer

- **Available now**
  - `control.lateral.control_mode` (`pid`/`stanley`)
  - `tools/analyze/run_ab_batch.py` repeated A/B with robust stats
  - Steering limiter waterfall instrumentation in recording + PhilViz
  - Ground truth follower (`tools/ground_truth_follower.py`) for controller
    bypass scenarios
- **What this answers**
  - "Are we rate/jerk/hard-clip/smoothing limited?"
  - "Does one control parameter improve medians across repeated trials?"
- **Current gap**
  - No control-only replay harness against frozen trajectory.

### 4) Integration / Timing Layer

- **Available now**
  - End-to-end closed-loop runs with `start_av_stack.sh`
  - To-failure truncation and event diagnostics in PhilViz
- **What this answers**
  - "Where does integrated behavior fail first?"
- **Current gap**
  - No explicit latency/noise injection test toggles.
  - No deterministic integration stress suite with seeded perturbation profiles.

### 5) Sweep / Batch Infrastructure

- **Available now**
  - `tools/analyze/run_ab_batch.py` (single-parameter repeated A/B)
  - `tools/analyze/tune_sweep.py`
  - `tools/analyze/curve_sweep.py`
  - `tools/parameter_sweep_tuning.py`
  - `tools/parameter_sweep_with_metrics.py`
- **Current gap**
  - Most sweeps are still full-stack runs, not layer-isolated sweeps.

---

## Standard Debug Workflow (Recommended Default)

Use this sequence for every major failure investigation:

1. **Detect and scope failure**
   - Use PhilViz with `analyze_to_failure=true`.
   - Use fixed distance window for entry comparisons (for example, `25m -> 33m`).
2. **Classify first-order bottleneck**
   - Use curve-entry feasibility + limiter waterfall + issue list.
3. **Run repeated A/B only after classification**
   - Use `tools/analyze/run_ab_batch.py` (minimum 5 repeats per arm).
   - Decide with median/p25/p75, not single-run outcomes.
4. **Layer challenge tests before additional tuning**
   - Replay perception on same recordings.
   - Run ground-truth follower baseline.
   - Compare with same analysis window and metrics.
5. **Promote only if stage-gate metrics improve**
   - Must improve failure timing and not regress key safety/quality metrics.

---

## PhilViz: What It Already Does Well

PhilViz currently has strong tools for:

- steering limiter root-cause attribution ("waterfall" stages),
- phase and curve-entry segmentation,
- to-failure analysis scope,
- issue/event detection with jump-to-frame,
- curve-entry feasibility classification (speed vs authority vs stale).

This is enough to avoid "blind tuning" and to reject many false improvements.

---

## PhilViz: Gaps To Close For Better Upstream/Downstream Attribution

High-value additions (recommended order):

1. **GT vs perception lateral error side-by-side at hotspots**
   - Distinguish "controller failed" vs "reference itself was wrong."
2. **Perception-to-trajectory attribution metrics**
   - Separate error from perception center estimate vs planner transform/smoothing.
3. **Failure-frame upstream state card**
   - Lanes detected, stale reason, GT/perception center mismatch at failure.
4. **Causal event timeline**
   - Ordered events: perception degradation -> trajectory deviation -> limiter
     saturation -> lane departure.
5. **Graphical waterfall chart**
   - Existing table is useful, but stacked deltas improve quick triage.

---

## Root-Cause Decision Rules (Practical)

Use these rules to avoid over-indexing on control:

- If GT/perception disagreement rises **before** limiter activity, investigate
  perception/trajectory first.
- If `steering_pre_rate_limit` is already low while lane departure grows,
  planner/reference quality is likely upstream.
- If `steering_pre_rate_limit` is high but post-limit collapses with stable
  perception, control authority/limits are primary.
- If a change improves authority-transfer metrics but worsens failure medians,
  it is likely over-aggressive or poorly timed.

---

## Minimum Test Layers We Should Have (Target State)

To align with top-tier controls methodology, add these explicit harnesses:

1. **Perception freeze artifact**
   - Save replayed perception outputs as a canonical dataset.
2. **Trajectory lock replay mode**
   - Consume frozen trajectory in closed-loop control tests.
3. **Control-only replay harness**
   - Re-run control policy against fixed trajectory/timebase.
4. **Latency/noise injection profiles**
   - Deterministic perturbations with seeds and severity levels.
5. **Cross-layer counterfactual evaluator**
   - "What if we swap only layer X from baseline into treatment?"

Script status:

- `tools/analyze/replay_trajectory_locked.py` (implemented scaffold)
- `tools/analyze/compare_crosslock_sensitivity.py` (planned temporary helper,
  then integrate into unified isolation reporting)
- `tools/analyze/replay_control_locked.py` (implemented scaffold)
- `tools/analyze/run_latency_noise_suite.py` (implemented; cross-seed gate + waiver policy)
- `tools/analyze/counterfactual_layer_swap.py` (implemented; unified Stage-5 report)

---

## Recommended Reporting Template For Any Investigation

Every report should include:

- Test matrix (what was locked vs live per layer)
- Window used (distance and to-failure setting)
- Robust stats (median/p25/p75)
- Primary classification and confidence
- Upstream/downstream call with evidence
- Next single change and stop/rollback criteria

---

## Quick Command Reference (Existing)

- Repeated A/B:
  - `python tools/analyze/run_ab_batch.py --param <dotted.key> --a <A> --b <B> --repeats 5 ...`
- Perception replay:
  - `python tools/replay_perception.py <recording.h5>`
- Trajectory-locked replay:
  - `python tools/analyze/replay_trajectory_locked.py <recording.h5> --disable-vehicle-frame-lookahead-ref`
- Latency/noise stress suite:
  - `python tools/analyze/run_latency_noise_suite.py <recording.h5> --noise-seeds 1337,2026 --min-seed-pass-rate 1.0 --output-prefix stage4_run`
- Control-locked replay:
  - `python tools/analyze/replay_control_locked.py <recording.h5>`
- Stage-5 counterfactual layer swap:
  - `python tools/analyze/counterfactual_layer_swap.py <baseline.h5> <treatment.h5> --output-prefix stage5_pair`
- Ground truth follower baseline:
  - `./start_ground_truth.sh --track-yaml <track> --duration 60 --speed 8.0`
- Curve-entry feasibility:
  - `python tools/analyze/analyze_curve_entry_feasibility.py <recording.h5> ...`
- Phase-to-failure:
  - `python tools/analyze/analyze_phase_to_failure.py <recording.h5> ...`

