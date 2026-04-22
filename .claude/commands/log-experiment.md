Record an experiment outcome (success, failure, or regression) to the experiment journal. Unlike `/log-fix` which only tracks successful fixes, this tracks ALL attempts — especially failed ones — to enable spinning-wheels detection.

The user's request is: $ARGUMENTS

## Step 1 — Identify the experiment

From $ARGUMENTS, git log, or the current conversation, extract:
- **Target metric**: what was being improved (e.g., lateral_rmse, jerk_p95, overall_score)
- **Target track**: which track (e.g., s_loop, mixed_radius, or "all")
- **Approach**: what was tried (e.g., "Higher q_lat (2.0 → 5.0)")
- **Fix level**: TUNING / CONFIG / CODE_PATCH / ARCHITECTURE
- **Hypothesis**: why this was expected to work

## Step 2 — Record the outcome

- **IMPROVEMENT**: metric improved meaningfully (state before/after values)
- **NO_IMPROVEMENT**: metric unchanged or within noise
- **REGRESSION**: metric got worse

## Step 3 — Classify root cause of failure

If outcome is NO_IMPROVEMENT or REGRESSION, identify why:
- `model_plant_mismatch` — controller model doesn't match vehicle dynamics
- `oscillation` — change caused instability
- `cfl_stability` — numerical stability limit prevents the approach
- `transient_dominance` — steady-state fix but error is transient
- `solver_convergence` — optimizer can't find feasible solution
- `proxy_approximation` — tuning a proxy instead of the real quantity
- `ceiling_reached` — architectural limit of current approach
- Or a new category if none fits

## Step 4 — Append to journal

```bash
python3 -c "
import json
from pathlib import Path
journal_path = Path('data/reports/experiment_journal.json')
journal = json.loads(journal_path.read_text()) if journal_path.exists() else []
journal.append({
    'date': '<YYYY-MM-DD>',
    'target_metric': '<metric>',
    'target_track': '<track>',
    'approach': '<approach description>',
    'fix_level': '<TUNING/CONFIG/CODE_PATCH/ARCHITECTURE>',
    'hypothesis': '<why expected to work>',
    'outcome': '<IMPROVEMENT/NO_IMPROVEMENT/REGRESSION>',
    'outcome_status': '<OPEN/SUPERSEDED/RESOLVED>',   # lifecycle of the experiment, not the metric direction
    'metric_before': <float>,
    'metric_after': <float>,
    'root_cause_of_failure': '<cause or null>',
    'root_cause_detail': '<1-2 sentence explanation>',
    'iterations_spent': <int>,
    'files_changed': [<list of files>],
    'superseded_by': '<journal entry index or commit SHA, or null>',   # populated when outcome_status transitions to SUPERSEDED
    'resolved_by': '<commit SHA or null>',                              # populated when outcome_status transitions to RESOLVED
})
journal_path.write_text(json.dumps(journal, indent=2))
print(f'Logged experiment #{len(journal)}')
"
```

### `outcome_status` semantics

| Status | When to use |
|---|---|
| `OPEN` | Experiment result stands; the root cause has not been fixed or abandoned. **Default for new entries.** |
| `SUPERSEDED` | A later experiment replaced this one's approach (e.g., plan rewritten, term removed). The original REGRESSION entry stays for audit; the Pareto should not count it as an active problem. |
| `RESOLVED` | The metric / track has since moved back into the acceptable range — either by a follow-up commit or because the scenario was dropped from the goal set. |

This is separate from `outcome` (IMPROVEMENT/NO_IMPROVEMENT/REGRESSION), which is frozen at the moment the experiment ran. `outcome_status` evolves over time as the broader project state changes.

**Updating an existing entry's status** (to mark as superseded or resolved):

```bash
python3 -c "
import json, sys
from pathlib import Path
path = Path('data/reports/experiment_journal.json')
journal = json.loads(path.read_text())
idx = <entry_index>                       # 0-based
journal[idx]['outcome_status'] = '<SUPERSEDED|RESOLVED>'
journal[idx]['superseded_by'] = '<ref>'   # or 'resolved_by'
path.write_text(json.dumps(journal, indent=2))
print(f'Updated entry #{idx} → {journal[idx][\"outcome_status\"]}')
"
```

**Backfill rule:** entries written before this field existed are implicitly `OPEN`. `/process-health` should treat missing `outcome_status` as `OPEN` for backward compatibility.

### Scoring-change experiments: use `/revalidate` to capture pre/post

For experiments where the intervention is a **scoring code change** (not a controller/config change), `metric_before` / `metric_after` should come from `/revalidate <recording>` on the SAME recording, not from two different Unity runs. Two Unity runs introduce behavior variance that contaminates a pure scoring change. `/revalidate` holds the recording fixed and re-scores under old vs new code, giving an uncontaminated before/after.

Typical pattern:
1. Before landing the scoring change: `python3 tools/analyze/analyze_drive_overall.py <recording>` → capture `metric_before`
2. Land the scoring change
3. `/revalidate <recording>` → capture `metric_after` (the "Now" column in revalidate's diff table)
4. Log the experiment with both values; `root_cause_of_failure` is typically null since scoring-only changes are deterministic re-runs.

## Step 5 — Check convergence

After logging, run the convergence detector:
```bash
python3 tools/analyze/analyze_convergence.py --metric <metric> --track <track>
```

If a WARNING or CRITICAL alert appears, inform the user:
- WARNING: "2 attempts have failed for the same root cause — consider escalating"
- CRITICAL: "3+ attempts failed — this is an architectural ceiling, not tunable"

## When to use this skill

- After EVERY `/iterate` or `/e2e` cycle that targeted a specific metric improvement
- After config tuning experiments (even if "just trying something")
- After architecture experiments (2DOF, preview weighting, etc.)
- **Especially** after failures — these are the most valuable data points

The experiment journal feeds `/process-health` and the convergence detector (`tools/analyze/analyze_convergence.py`). Without logging failures, the spinning-wheels detector can't work.
