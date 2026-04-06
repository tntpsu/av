# Autonomous Engineering System

## Overview

This document describes the near-autonomous engineering system built for the AV stack. The system uses a network of specialized skills (slash commands) that form feedback loops for diagnosis, correction, validation, and continuous improvement.

**Current state:** ~80% autonomous for diagnose-fix-validate cycles. Human judgment still needed for architecture decisions and process stage classification.

---

## What We Can Do Now

### Autonomous Loops

```
                    /scores
                       │
            ┌──────────┼──────────┐
            ▼          ▼          ▼
         /pareto   /diagnose   /validate
            │          │          │
            ▼          ▼          ▼
     Prioritize    Root cause   Gate check
     by ROI        + blame      pass/fail
            │          │          │
            └──────────┼──────────┘
                       ▼
                   /iterate
              ┌────────┼────────┐
              ▼        ▼        ▼
          /plan-    Code/     /e2e
          feature   config    run+analyze
              │     change        │
              ▼        │         ▼
           Approve     │     /validate
              │        │         │
              └────────┼─────────┘
                       ▼
                   /log-fix
                       │
                       ▼
              /process-health
              (Pareto of where
               issues originate)
```

### Skill Capabilities

| Skill | What it does | Autonomy level |
|-------|-------------|----------------|
| `/scores` | Scoreboard + top 3 ROI fixes | Fully automatic |
| `/pareto` | 3-tier cross-track issue Pareto | Fully automatic |
| `/diagnose` | Root cause + signal chain blame | Fully automatic |
| `/trace curve_entry` | Per-corner signal chain blame | Fully automatic |
| `/iterate <track> <goal>` | Diagnose → fix → validate loop (max 5 iter) | Semi-auto (checkpoints for approval) |
| `/e2e <track>` | Build → run → analyze → verdict | Fully automatic |
| `/validate` | Gate checks (comfort, scoring, ACC) | Fully automatic |
| `/plan-feature` | Architecture plan with industry context | Semi-auto (human approval) |
| `/log-fix` | Record fix to improvement log | Semi-auto (human classifies root cause) |
| `/process-health` | Continuous improvement Pareto | Fully automatic |
| `/update-arch` | Keep architecture docs current | Fully automatic |
| `/instrument` | Add/remove debug probes | Fully automatic |
| `/run` | Resolve scenario → exact command | Fully automatic |

### Diagnostic Tools

| Tool | Purpose |
|------|---------|
| `trace_curve_entry.py` | Signal chain blame per curve event (PP floor, Ld, FF, steering) |
| `analyze_drive_overall.py` | Full recording analysis with 21 sections |
| `run_gate_and_triage.py` | Gate evaluation + triage packets |
| `build_failure_packet.py` | Failure frame isolation |
| `counterfactual_layer_swap.py` | Layer-level attribution |
| `acc_pipeline_analysis.py` | ACC-specific diagnostics |
| `mpc_pipeline_analysis.py` | MPC-specific diagnostics |

### Continuous Improvement Infrastructure

| Component | Purpose |
|-----------|---------|
| `improvement_log.json` | Historical fix database (process stage, root cause, detection method) |
| `/process-health` Pareto | Where issues originate (design 73%, implementation 27%) |
| Detection delay tracking | Measures how quickly tools find root causes (avg 2.5 → 1.8 iterations) |
| Physics-first design rule | CLAUDE.md constraint #8 — prevents proxy approximation issues |

---

## What's Near-Automatic (Human Approves, System Executes)

### The `/iterate` Loop

The autonomous improvement loop handles:
1. **Baseline establishment** — reads all recordings, builds cross-track table
2. **Holistic diagnosis** — runs `/diagnose`, cross-track correlation, signal chain blame
3. **Fix level classification** — TUNING/CONFIG/CODE/ARCHITECTURE with proxy audit
4. **Impact prediction** — maps fix against operating envelope (all tracks)
5. **Implementation** — edits code/config, runs tests
6. **Validation** — `/e2e` on target + boundary tracks
7. **Regression triage** — classifies any regressions (true/latent/artifact/tradeoff)
8. **Safety guards** — immediate revert on e-stops, pause on >5pt drops

**Human needed for:** Step 3 approval (checkpoint before implementing), architecture decisions, and accepting tradeoffs.

### The Design Loop

```
User: "improve trajectory on s_loop"
  → /iterate s_loop trajectory
    → /diagnose (automatic)
    → trace_curve_entry.py (automatic)
    → CHECKPOINT: "PP floor heavy rescue, propose lowering floor"
    → User: "proceed"
    → Implementation (automatic)
    → Tests (automatic)
    → /e2e s_loop (automatic)
    → /e2e mixed_radius boundary check (automatic)
    → Results comparison (automatic)
    → /log-fix (semi-auto: human classifies process stage)
```

---

## What's Still Manual

| Gap | Why | Path to Automation |
|-----|-----|-------------------|
| Architecture decisions | Requires judgment about tradeoffs, system-level thinking | Could use `/plan-feature` + LLM reasoning, but human approval is safety-critical |
| Process stage classification | "/log-fix" asks human to classify root cause stage | Could auto-classify from diff patterns (design: new params, implementation: bug fix) |
| Cross-track regression gate | Currently manual `/e2e` on each track | Build automated cross-track sweep after each `/iterate` |
| New trace event types | Only `curve_entry` exists | Build `regime_transition`, `heading_suppression`, `oscillation_onset` |
| Unity health management | Player segfaults after ~36 cycles | Auto-rebuild heuristic exists in `/iterate` guard 8f |

---

## Next Steps Toward Full Autonomy

### Phase 1: Cross-Track Regression Gate (immediate)

Build an automated cross-track scoring gate that runs after each `/iterate` completion:
```
/sweep — run /e2e on all tracks, compare to baselines, flag regressions
```
This is the #1 process improvement from `/process-health` (cited by 6/11 entries).

### Phase 2: Expand Trace Tooling (short-term)

Build trace event types for the remaining manual-diagnosis gaps:
- `regime_transition` — MPC↔PP handoff, blend asymmetry
- `heading_suppression` — heading gate, far_preview routing
- `oscillation_onset` — growth detection at steering frequency

Each saves ~2 iterations of diagnosis.

### Phase 3: Auto-Classification for /log-fix (medium-term)

Auto-detect process stage from git diff patterns:
- New config parameter → likely "design/proxy approximation"
- Bug fix in formula → "implementation/wrong-formula"
- New test → "testing/missing-coverage"
- New diagnostic tool → "tooling/missing-trace"

Reduce human input in `/log-fix` to confirmation only.

### Phase 4: Fully Autonomous Iterate (longer-term)

Remove the checkpoint in `/iterate` Step 2g for TUNING and CONFIG-level fixes. Only pause for CODE PATCH and ARCHITECTURE. The system has enough guard rails (safety revert, severity pause, regression detection) to handle routine tuning autonomously.

### Phase 5: Autonomous Roadmap Execution (vision)

```
/advance — execute the next milestone in the project roadmap

1. Read docs/agent/tasks.md for next priority
2. /plan-feature for the approach
3. Implement in phases
4. /iterate until all tracks pass
5. /process-health to verify no systemic issues
6. /log-fix for all changes
7. Commit, push, update roadmap
8. Pick up next task
```

This is the end state: the system reads the roadmap, plans the work, executes it, validates across all tracks, logs learnings, and advances to the next milestone. Human reviews at phase boundaries (milestone promotion) rather than per-fix.

---

## Architecture Principles That Enable Autonomy

1. **Physics-first design** — formulas over lookup tables, computed over configured. Reduces the parameter search space that the autonomous loop needs to explore.

2. **Proportional over discrete** — continuous signals (local_gate_weight) over state machines (ENTRY/COMMIT). Smoother control surfaces are easier to tune and less likely to produce edge cases.

3. **Signal chain blame** — every diagnostic traces from symptom to root cause to specific mechanism. The system can identify what to fix, not just that something is wrong.

4. **Cross-track envelope thinking** — every fix is evaluated against all track geometries before implementation. The operating envelope table prevents fixes that help one track and hurt another.

5. **Process health feedback** — the improvement log creates a meta-learning loop. The system learns not just how to fix issues, but how to prevent them (physics-first, trace tooling, regression gates).
