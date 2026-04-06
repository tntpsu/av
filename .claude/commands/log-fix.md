Record a fix to the continuous improvement log for process health tracking.

The user's request is: $ARGUMENTS

## Step 1 — Identify the fix

Parse $ARGUMENTS:
- If a commit hash is given → read the commit message and diff with `git show <hash> --stat`
- If "latest" or blank → use the most recent commit: `git log -1 --oneline`
- If "--backfill" → scan `docs/agent/current_state.md` and `git log --oneline -20` to seed entries
- If a description is given → use that directly

For each fix, determine:
- What issue was fixed (from commit message or user description)
- Which tracks/scenarios were affected
- What scoring deduction was reduced (if known)

## Step 2 — Classify the fix

Determine these fields (infer from the diff if possible, ask user if ambiguous):

### Process stage — where was the defect introduced?
- **requirements**: Wrong target, wrong metric, wrong threshold, missing acceptance criteria
- **design**: Wrong architecture, proxy approximation, state discretization, signal routing, missing physics model
- **implementation**: Bug, wrong formula, wrong config value, off-by-one, race condition
- **testing**: Missing test, wrong threshold, untested edge case, missing regression gate
- **tooling**: Missing diagnostic, misleading metric, tool gap, missing trace event

### Process sub-category — more specific:
- design/proxy-approximation: Speed proxy for curvature, distance proxy for time
- design/state-discretization: Binary gate on continuous signal
- design/signal-routing: Signal leaking to wrong path
- design/missing-physics: No steady-state error model
- implementation/wrong-formula: Math error in computation
- implementation/wrong-config: Config value incorrect
- testing/missing-coverage: No test for this failure mode
- testing/wrong-threshold: Test threshold too lenient
- tooling/missing-trace: No diagnostic for this signal chain
- tooling/missing-detection: Issue detector doesn't flag this pattern

### Detection method — how was the issue found?
- trace_curve_entry.py, analyze_drive_overall.py, manual grep, e2e comparison, unit test failure, etc.

### Detection delay — iterations to find root cause?
- 0 = tool identified it immediately
- 1-2 = one wrong hypothesis before finding it
- 3+ = multiple iterations of guessing

### Prevention — what would have prevented this?
- New design rule, new diagnostic tool, new test, architecture review, etc.

## Step 3 — Append to log

```python
import json
from pathlib import Path
from datetime import date

log_path = Path("data/reports/improvement_log.json")
log = json.loads(log_path.read_text()) if log_path.exists() else []
log.append({
    "date": str(date.today()),
    "commit_summary": "<description>",
    "issue": "<what was wrong>",
    "deduction_eliminated": "<scoring impact>",
    "tracks_affected": ["<track1>", "<track2>"],
    "root_cause": "<root cause description>",
    "process_stage": "<stage>",
    "process_sub": "<sub-category>",
    "prevention": "<what would prevent this>",
    "detection_method": "<how found>",
    "detection_delay_iterations": 0
})
log_path.write_text(json.dumps(log, indent=2))
```

## Step 4 — Confirm

Show the entry that was added:

```
FIX LOGGED
════════════════════════════════════════
Issue: <issue>
Stage: <stage> / <sub>
Impact: <deduction>
Tracks: <list>
Prevention: <prevention>
Detection: <method> (delay: <N> iterations)
Log entries: <total> total
════════════════════════════════════════

Run /process-health to see updated Pareto.
```

## References
- Improvement log: `data/reports/improvement_log.json`
- Git log: `git log --oneline -10`
- Current state: `docs/agent/current_state.md`
