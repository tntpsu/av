Generate a cross-track issue Pareto showing the highest-ROI fixes across all tracks and scenarios.

The user's request is: $ARGUMENTS

## Step 1 — Gather scores for all tracks

Run the scoring extraction (same as /scores Step 2) to get layer_score_breakdown for every track with a recent recording. Use `analyze_recording_summary()` on the latest recording per track.

## Step 2 — Build Tier 1: Scoring Impact Pareto

For each track, extract all deductions with value > 0 from layer_score_breakdown.
Aggregate across tracks:
- Sum total points lost per deduction name
- Count how many tracks are affected
- Sort by total points lost (descending)

Present as:
```
TIER 1 — SCORING IMPACT (what's costing points)
═══════════════════════════════════════════════════
Rank  Deduction                          Σ pts lost  Tracks  Impact
1     <name>                             <sum>       N/M     ████████
2     <name>                             <sum>       N/M     ████
...
```

Use bar chart characters proportional to impact.

## Step 3 — Build Tier 2: Root Cause Clustering

Group deductions by root cause. Use these mappings:
- "Lateral Error RMSE" → run `python3 tools/analyze/trace_curve_entry.py <recording>` for blame (PP floor, Ld too long, lgw too low)
- "Oscillation Growth" → controller stability (MPC transition, PP gain, regime blend)
- "Heading Suppression Rate" → far_preview routing, heading rate limiter
- "Out Of Lane / Emergency Events" → safety (steering authority, track geometry)
- "Steering Jerk" → rate limiter, regime transition blend
- "Straight Sign Mismatch" → FF phase gating, curve state latch on straights

For tracks where Trajectory < 95, run the blame trace:
```bash
python3 tools/analyze/trace_curve_entry.py <recording>
```

Present as:
```
TIER 2 — ROOT CAUSE CLUSTERS
═══════════════════════════════════════════════════
Root Cause                  Deductions affected    Tracks    Status
<cause>                     <list>                 N/M       FIXED/OPEN/PARTIAL
```

Mark causes as FIXED if they were addressed in the improvement log (`data/reports/improvement_log.json`).

## Step 4 — Build Tier 3: Fix ROI

For each open root cause, estimate:
- Fix level: TUNING / CONFIG / CODE PATCH / ARCHITECTURE
- Points recoverable: sum of deductions this fix would address
- Tracks helped: count
- ROI = points / complexity (TUNING=1, CONFIG=2, CODE=3, ARCH=5)

Present as:
```
TIER 3 — FIX ROI (highest value first)
═══════════════════════════════════════════════════
Fix                          Level        Pts recoverable  Tracks  ROI
<description>                <level>      <pts>            N/M     <score>
```

## Step 5 — Output summary

```
TOP 3 HIGHEST-ROI FIXES
════════════════════════════════════════
1. <fix> — <pts> pts across <N> tracks (<level>)
2. <fix> — <pts> pts across <N> tracks (<level>)
3. <fix> — <pts> pts across <N> tracks (<level>)

Run /process-health for continuous improvement analysis.
```

## References
- Scoring: `tools/drive_summary_core.py`
- Blame trace: `tools/analyze/trace_curve_entry.py`
- Thresholds: `tools/scoring_registry.py`
- Improvement log: `data/reports/improvement_log.json`
