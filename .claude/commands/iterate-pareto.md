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

## Step 2 — Select target

Pick the #1 Pareto item (highest Σ pts lost). Identify:
- **Target track**: the track where this deduction is WORST
- **Target layer**: the layer this deduction belongs to
- **Points recoverable**: the deduction value on that track

If the target track has a stale recording, run `/e2e <track>` first to get fresh data.

Present:
```
PARETO-DRIVEN TARGET
════════════════════════════════════════
Pareto #1: <deduction_name> — <Σ pts> across <N> tracks
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
