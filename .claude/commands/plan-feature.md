Create a detailed, phased implementation plan for a new feature or architecture change. Captures industry context, specifies every file touch, includes debug tooling, tests, validation, and doc updates.

The user's request is: $ARGUMENTS

## Step 1 — Understand the problem

Read context before planning:
1. `docs/agent/current_state.md` — what's active, recent results, known issues
2. `docs/agent/architecture.md` — component relationships and data flow
3. `docs/agent/tasks.md` — existing priorities
4. `CLAUDE.md` — constraints, fragile areas, critical files
5. Any relevant memory files from `MEMORY.md`

If $ARGUMENTS references a specific recording or E2E result, read it to ground the plan in real data.

## Step 2 — Industry benchmarking

Before proposing ANY approach:
1. State what Waymo, Tesla, Comma.ai, Aurora, or Motional do for this problem
2. Identify the industry-standard solution vs ad-hoc alternatives
3. Explain why the standard approach applies (or doesn't) to our simulation context
4. **Reject per-track tuning** — solutions must generalize across all tracks

Output:
```
INDUSTRY CONTEXT:
  Standard approach: <what top companies do>
  Why it applies here: <reasoning>
  What we're NOT doing: <rejected ad-hoc alternatives and why>
```

## Step 3 — Fix Level Classification & Robustness Gate

Before designing anything, classify what kind of change this is and verify cross-track robustness:

### 3a — Classify the change level

```
FIX LEVEL CLASSIFICATION:
  Level: TUNING / CONFIG / CODE PATCH / ARCHITECTURE
    - TUNING: adjusting existing numeric params within their intended range
    - CONFIG: adding/changing config knobs (new params, new overlay keys)
    - CODE PATCH: changing priority/ordering/logic within existing architecture
    - ARCHITECTURE: changing how components interact, what signals drive decisions,
      or what abstractions exist
```

### 3b — Proxy stacking audit

Check whether the subsystem being modified has multiple parameters that approximate the same physical quantity:

```
PROXY STACKING AUDIT:
  Physical quantity being controlled: <what the params are really trying to decide>
  Parameters that approximate it:
    - <param1> in <file> — <what it controls>
    - <param2> in <file> — <what it controls>
    ...
  Per-track overrides for this subsystem: <count> overlays (<list>)
  Proxy stacking detected: yes/no
  → If yes: consider unifying around the actual physical quantity instead of
    adding another proxy parameter
```

### 3c — Cross-track robustness check

For EVERY track in `tracks/*.yml`, answer:
1. Does this change alter behavior on this track? (yes/no/marginal)
2. If yes, does it improve, degrade, or have no net effect?
3. Does this change eliminate any existing per-track overlay workaround?

```
CROSS-TRACK IMPACT:
  | Track          | Behavior change? | Effect        | Overlay eliminated? |
  |----------------|-----------------|---------------|---------------------|
  | highway_65     | no              | —             | —                   |
  | s_loop         | yes             | improved      | mpc_min_speed_mps   |
  | ...            | ...             | ...           | ...                 |
  
  Tracks improved: <count>
  Tracks degraded: <count> — BLOCK if any degraded without justification
  Overlays eliminated: <count>
```

**Rules:**
- If ARCHITECTURE level: the plan MUST show how it eliminates ≥1 per-track workaround
- If the change adds a new per-track overlay parameter: justify why it can't be auto-derived or unified
- If ≥2 overlays work around the same subsystem and the plan doesn't address it: flag and reconsider
- Prefer changes that make the system more physics-based (e.g., lateral accel budget) over heuristic-based (speed thresholds + curvature guards)

## Step 4 — Root cause grounding

Tie the plan to specific data, not hypotheses:
- Reference exact metric values from recent E2E runs or recordings
- Identify the specific frames/curves/segments where the problem manifests
- State the measurable success criterion (e.g., "steady-state e_lat P50 < 0.30m")
- State the regression guardrails (e.g., "highway score must not drop below 97")

Output:
```
ROOT CAUSE:
  Evidence: <specific metrics from specific recordings>
  Bottleneck: <what exactly limits the score>
  Success criterion: <measurable target>
  Regression guardrails: <what must NOT break>
```

## Step 4b — Mechanism Verification (MANDATORY before designing)

Before designing anything, verify the proposed approach actually changes behavior at the problem frames. This is a 30-second numerical check that prevents building the wrong solution.

1. **Extract existing signal values** at the 5 worst frames from the diagnosis:
   - What is the current value of the signal you plan to modify?
   - What outcome does it produce (e.g., Ld=3.28m, steering=0.05)?

2. **Compute what the proposed fix would produce** at those same frames:
   - If adding a ceiling/floor: is the proposed value MORE restrictive than existing?
   - If adding a scaling factor: does the scaling actually change the outcome?
   - If replacing a mechanism: does the new mechanism produce a different output?

3. **Compare and decide:**
   ```
   MECHANISM VERIFICATION
   ============================================================
   Frame    Existing signal    Proposed signal    Changes outcome?
   1595     Ld=3.28m          profile=5.66m      NO (ceiling is higher)
   ...
   
   Verdict: PROCEED / STOP (if proposed doesn't change ≥3 of 5 worst frames)
   ============================================================
   ```

**If the proposed mechanism doesn't change the outcome at the problem frames, STOP.** Re-diagnose — the bottleneck is elsewhere. Common traps:
- Ceiling on a signal that's already lower than your ceiling (lookahead profiler lesson)
- Floor on a signal that's already higher than your floor
- Scaling a parameter that's already saturated or bypassed
- Fixing a subsystem that's not the active controller at the problem frames (e.g., tuning MPC when PP is active)

## Step 4b.1 — Design Smell Gate (MANDATORY)

Run the 7-smell check from `/iterate` §2e4 against the proposed fix. A plan that triggers ANY smell without explicit architectural justification is BLOCKED — do not emit a `## Critical files modified` section until the smell is addressed or justified in writing.

```
DESIGN SMELL GATE (plan-side)
============================================================
  □ Smell 1 — Binary gate on continuous signal?
  □ Smell 2 — Proxy stacking (2+ params ≈ one quantity)?
  □ Smell 3 — Frame-rate dependent formula?
  □ Smell 4 — Static lookup table for physics quantity?
  □ Smell 5 — Post-hoc clamp / independent limiter?
  □ Smell 6 — Wrapper forwarding gap?
  □ Smell 7 — Load-bearing assumption removed without verification?

  Smells triggered: <list>
  Architectural justification (if any triggered): <text>
============================================================
```

**Smell 7 deserves special care in plans.** If the fix removes a proxy/projection/"contamination" term, the plan MUST include a Phase 0.5 step that empirically measures the removed term's closed-loop role (not just its algebraic meaning) on the recordings the plan cites:

- `corr(removed_term, d/dt(primary_error))` — is it carrying derivative/damping load?
- `corr(removed_term, error_at_t+Δ)` for Δ = 0.1–0.5s — is it acting as preview?
- Frequency content — does removing it shift the closed-loop pole structure?

If any of these correlations exceed 0.5 on stable frames, the term is structurally load-bearing. The plan must then include an *explicit substitute* for that role (derivative gain, preview horizon, etc.) BEFORE activation, not discover the gap at G2.

**Session precedent (2026-04-19):** `project_frenet_mpc_reference.md` passed Phase 0 (algebraic verification: candidate matched at-car truth at +0.9872 correlation, RMS 0.051m) but did NOT test the removed `Ld·sin(e_h)` term's dynamic role. Activation regressed H2 79→59 because that term was supplying PD-preview damping. A Phase 0.5 check would have caught this before the HDF5 plumbing was built.

## Step 4c — Prior Investigation Cross-Reference

Before designing from scratch, check whether prior investigations already characterized this root cause:

1. Search `docs/agent/tasks.md` for task IDs referencing this symptom
2. Search `MEMORY.md` for related project/feedback memories
3. Search `docs/agent/current_state.md` for "DEFERRED" items on this topic

```
PRIOR INVESTIGATION CHECK
============================================================
  Related tasks: <T-XXX — what it found, when>
  Related memories: <memory file — key finding>
  Prior root cause: <what was already determined>
  Prior attempts: <what was tried and why it didn't work>
  
  Does the current proposal conflict with prior findings? YES/NO
  If YES: explain why the new approach overcomes the prior limitation
============================================================
```

**If a prior investigation already characterized the root cause, surface that finding.** Don't re-derive what T-078 already proved — build on it.

## Step 5 — Design the implementation phases

Break into phases following this mandatory checklist. Not all phases apply to every feature — mark N/A where appropriate, but **explicitly justify** skipping any phase.

### Phase A: Config & Parameters
- New YAML keys in `config/av_stack_config.yaml`
- New fields in dataclasses (`MPCParams`, `ACCParams`, etc.) with `from_config()` mapping
- Overlay updates for affected track configs
- Auto-derive integration if the param should scale with track geometry

### Phase B: Core Algorithm
- The actual implementation (new class, modified method, etc.)
- Specify exact file, method name, insertion point (line number or "after X")
- State the mathematical model or algorithm with equations where relevant
- Identify the hot path and performance budget (e.g., "must complete in <1ms at 30Hz")

### Phase C: Data Plumbing
- Orchestrator changes (`av_stack/orchestrator.py`) — how data flows in
- Controller changes (`control/pid_controller.py`) — how results flow out
- Bridge changes if Unity data is needed (`bridge/server.py`, C# scripts)
- **Follow the 6-location HDF5 pattern** for any new recorded fields:
  1. `data/formats/data_format.py` — dataclass field
  2. `data/recorder.py` — `create_dataset`
  3. `data/recorder.py` — list init
  4. `data/recorder.py` — per-frame append
  5. `data/recorder.py` — resize loop + write
  6. `av_stack/orchestrator.py` — ControlCommand/VehicleState construction

### Phase D: Scoring & Gates
- New constants in `tools/scoring_registry.py`
- Gate thresholds with justification (why this value?)
- Guard tests in `tests/test_scoring_registry.py`

### Phase E: Diagnostic Tooling (MANDATORY — per diagnostic coverage rule)
Every new failure mode MUST appear in ALL THREE:
1. **`tools/debug_visualizer/backend/issue_detector.py`** — frame-level issue type(s)
2. **`tools/debug_visualizer/backend/triage_engine.py`** — pattern-level detection
3. **`tools/debug_visualizer/backend/layer_health.py`** — per-frame health flag + score deduction

Plus:
4. **`tools/analyze/mpc_pipeline_analysis.py`** or **`acc_pipeline_analysis.py`** — CLI diagnostic section
5. **PhilViz card** in relevant backend if the feature has visual diagnostics

### Phase F: Tests
- Unit tests for the core algorithm (Phase B)
- Integration tests if data flows through multiple components
- Scoring registry guard tests (Phase D)
- Regression tests — which existing test suites must still pass?
- Specify test file name and class name

### Phase G: E2E Validation
- Primary track to validate on (the one with the problem)
- Regression tracks (must not break)
- Expected score improvement with reasoning
- Use `/e2e <track>` for each validation run

### Phase H: Documentation & Memory
- `docs/agent/current_state.md` — update with results
- `docs/agent/tasks.md` — mark complete, add follow-ups
- `docs/agent/architecture.md` — update if component relationships changed
- `CLAUDE.md` — update critical files table, constraints, or fragile areas if needed
- Memory files — capture non-obvious learnings, pitfalls, or validated approaches

## Step 6 — Output the plan

Use Plan mode (EnterPlanMode) to present the plan. Structure:

```
IMPLEMENTATION PLAN: <feature name>
============================================================

INDUSTRY CONTEXT:
  ...

ROOT CAUSE:
  ...

PHASE A — Config & Parameters
  Files: <list with line numbers>
  Changes: <specific additions>
  
PHASE B — Core Algorithm
  Files: <list with line numbers>
  Algorithm: <description with math if relevant>
  Performance budget: <constraint>

PHASE C — Data Plumbing
  New HDF5 fields: <list>
  Orchestrator: <changes>
  Controller: <changes>
  Bridge: <changes if any>

PHASE D — Scoring & Gates
  New constants: <list with values and justification>

PHASE E — Diagnostic Tooling
  issue_detector: <new types>
  triage_engine: <new patterns>
  layer_health: <new flags>
  pipeline_analysis: <new section>

PHASE F — Tests
  New tests: <count and file>
  Regression suites: <which must pass>

PHASE G — E2E Validation
  Primary: /e2e <track> using <config>
  Regression: /e2e <track1>, /e2e <track2>
  Success: <target score>
  Guardrail: <minimum acceptable on regression tracks>

PHASE H — Documentation
  Files to update: <list>
  Learnings to capture: <list>

ESTIMATED SCOPE: <small/medium/large> (<N> files, <N> new tests)
============================================================
```

After the user approves the plan (ExitPlanMode), begin implementation phase by phase, marking tasks as completed along the way.

---

Critical files reference: `CLAUDE.md`
Diagnostic coverage rule: memory file `feedback_diagnostic_coverage.md`
HDF5 6-location pattern: `MEMORY.md` critical rules section
Scoring registry: `tools/scoring_registry.py`
