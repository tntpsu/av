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

## Step 3 — Root cause grounding

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

## Step 4 — Design the implementation phases

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

## Step 5 — Output the plan

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
