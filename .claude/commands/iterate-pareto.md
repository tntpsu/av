Autonomous improvement driven by the cross-track Pareto. Picks the highest-ROI fix, identifies the worst-affected track, and runs /iterate on it. Repeats until the top-of-pareto is below threshold.

The user's request is: $ARGUMENTS

## Step 0 — Parse arguments

- No args → run until top-of-pareto < 5 pts total, max 3 cycles
- A number (e.g., "1") → run exactly N cycles
- "--dry-run" → show what would be targeted without running

## Step 1 — Build Pareto

Run the same scoring extraction as /scores to get layer_score_breakdown for every track with a recent recording. For tracks with pre-change recordings (> 24h old or before last code commit), note them as stale.

Aggregate deductions across tracks:
```
Deduction Name                    Σ pts lost  Tracks affected  Worst track
Lateral Error RMSE (curv-adj)     29.9        5/5              mixed_radius (-6.9)
Steering Jerk                     20.0        1/5              hairpin_15 (-20.0)
Oscillation Growth                14.7        3/5              sweeping (-6.8)
...
```

## Step 1.5 — Prior Investigation Cross-Reference (MANDATORY)

Before selecting a target, check whether the top Pareto items have been investigated before:

1. **Run the convergence detector** on the top Pareto items' target metrics:
```bash
python3 tools/analyze/analyze_convergence.py
```
If ANY top Pareto item maps to a CRITICAL ceiling (3+ failed attempts with same root cause), **skip it** — it cannot be fixed by tuning or code patch. Flag it as ARCHITECTURE and move to the next Pareto item.

2. Search `docs/agent/tasks.md` for deferred tasks referencing the deduction name
3. Search `MEMORY.md` for related project/feedback memories
4. If a prior investigation exists, summarize: what was found, what was tried, why it was deferred

```
PRIOR INVESTIGATION CHECK
============================================================
  Convergence detector: <CLEAR / WARNING / CRITICAL per item>
  
  Pareto #1: <deduction> — prior task: <T-XXX or "none">
    Convergence: <CLEAR/WARNING/CRITICAL — N failed attempts, root cause>
    Prior finding: <root cause if known>
    Prior attempt: <what was tried>
    Status: <resolved / deferred / ceiling / not investigated>
  
  Pareto #2: <deduction> — prior task: <T-XXX or "none">
    Convergence: <status>
    ...
============================================================
```

**If a CRITICAL ceiling is detected:**
- Do NOT dispatch /iterate — it will waste E2E cycles hitting the same wall
- Auto-classify as ARCHITECTURE (skip Step 2a)
- Present the ceiling evidence and recommend `/plan-feature`
- Move to next Pareto item if user doesn't want architecture work now

**If a deferred investigation already characterized the root cause:**
- Do NOT dispatch /iterate with the same approach that was already tried
- Present the prior finding and ask: "Prior investigation found X. Should we try a different approach, or skip to the next Pareto item?"
- If the prior approach failed because of a specific limitation, the new approach must address that limitation explicitly

## Step 2 — Select target and classify fix level

Pick the #1 Pareto item (highest Σ pts lost). Identify:
- **Target track**: the track where this deduction is WORST
- **Target layer**: the layer this deduction belongs to
- **Points recoverable**: the deduction value on that track

If the target track has a stale recording, run `/e2e <track>` first to get fresh data.

### 2a — Fix-level classification (MANDATORY before dispatch)

Before dispatching to `/iterate`, classify the fix level. The hierarchy is:
1. **ARCHITECTURE** — component interactions, signal paths, what data drives decisions
2. **CODE PATCH** — logic changes within existing architecture
3. **CONFIG** — adding/changing config knobs
4. **TUNING** — adjusting existing numeric params within their intended range

**Quick classification rules (from Pareto data alone, no full diagnosis needed):**
- Symptom hits **4+ tracks** at similar severity → likely ARCHITECTURE or CODE PATCH (systemic)
- Symptom is in **Trajectory** layer with "late turn-in" issues → ARCHITECTURE (lookahead/anticipation)
- Symptom is in **Control** layer with "oscillation growth" on curved tracks → investigate if scoring artifact vs real issue
- Symptom is in **1-2 tracks** only → may be TUNING or CONFIG
- Prior `/iterate` cycles failed to fix this deduction → escalate one level

### 2b — Architecture gate

**If fix level is ARCHITECTURE:**
- Do NOT dispatch to `/iterate` (which will waste E2E cycles trying to tune)
- Instead, recommend `/plan-feature` with the Pareto evidence
- Present the architecture issue and ask the user

```
PARETO #1 CLASSIFIED AS ARCHITECTURE
════════════════════════════════════════
Issue: <deduction_name> — <Σ pts> across <N> tracks
Classification: ARCHITECTURE
Reason: <why architecture, not tuning>

This issue cannot be fixed by tuning existing parameters.
→ Recommended: /plan-feature <description>
→ Skip to Pareto #2? (if actionable at lower fix level)
════════════════════════════════════════
```

Wait for user decision before continuing. Options:
- Run `/plan-feature` for the architecture fix
- Skip to next Pareto item
- Override and attempt `/iterate` anyway

**If fix level is CODE PATCH, CONFIG, or TUNING:**
- Proceed to Step 3 (dispatch to `/iterate`)

Present:
```
PARETO-DRIVEN TARGET
════════════════════════════════════════
Pareto #1: <deduction_name> — <Σ pts> across <N> tracks
Fix level: <TUNING/CONFIG/CODE PATCH>
Worst track: <track> (<pts> pts lost)
Target: /iterate <track> <layer>
════════════════════════════════════════
```

## Step 3 — Run /iterate

Execute the full /iterate workflow on the selected target:
- All iterate guards apply (safety, severity, regression, etc.)
- Max iterations per cycle: 3 (to prevent burning too many e2e runs)
- If /iterate hits max iterations without meeting goal, move to next Pareto item

## Step 4 — Re-assess

After /iterate completes (success or max iterations):
1. Re-run the Pareto to see if rankings changed
2. Check if the fix helped OTHER tracks (cross-track benefit)
3. Log all fixes via /log-fix

Present:
```
PARETO UPDATE
════════════════════════════════════════
Before:
  #1 <deduction> — <pts> across <N> tracks
After:
  #1 <deduction> — <pts> across <N> tracks

Cycle result: <target track> <before> → <after> (<delta>)
Cross-track benefit: <list of other tracks improved>
════════════════════════════════════════
```

## Step 5 — Loop or stop

**Stop if ANY of:**
- Top-of-pareto Σ pts < 5 (diminishing returns)
- Max cycles reached
- /iterate failed with safety guard (e-stop, >5pt regression)
- All tracks have all layers ≥ 95
- All remaining Pareto items are classified ARCHITECTURE (need /plan-feature, not /iterate)

**Continue if:**
- Top-of-pareto Σ pts ≥ 5
- Cycles remaining
- No safety guards triggered

## Step 6 — Completion

```
ITERATE-PARETO COMPLETE
════════════════════════════════════════════════════════════════════
Cycles used: <N> / <max>
Starting Pareto #1: <deduction> — <pts>
Ending Pareto #1:   <deduction> — <pts>

Fixes applied:
  Cycle 1: <fix> on <track> — <delta>
  Cycle 2: <fix> on <track> — <delta>

Track improvements:
  <track>: <before> → <after> (<delta>)

Remaining Pareto:
  #1 <deduction> — <pts> across <N> tracks
  #2 <deduction> — <pts> across <N> tracks
  #3 <deduction> — <pts> across <N> tracks

Promotion checklist:
  [ ] /sweep to validate all tracks
  [ ] /log-fix for each fix
  [ ] /process-health to check improvement trends
  [ ] Commit and push
════════════════════════════════════════════════════════════════════
```

## References
- Pareto: `/pareto` skill
- Iterate: `/iterate` skill
- Scores: `/scores` skill
- Process health: `/process-health` skill
- Sweep: `/sweep` skill
- Convergence detector: `tools/analyze/analyze_convergence.py`
- Experiment journal: `data/reports/experiment_journal.json` (log via `/log-experiment`)
- Regime boundaries: `tools/analyze/analyze_regime_boundaries.py`
