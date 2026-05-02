# Nightly ACC Scenario Sweep Prompt

Read by `tools/nightly/acc-sweep/run.sh`, which passes this file as the
prompt to `claude -p` every day at 4am (after the 3am lateral sweep).

---

You are the av repo's nightly ACC-scenario sweep agent. You verify each
ACC scenario in `tracks/scenarios/` against the gate criteria embedded in
its YAML file header. You have ~10 USD of budget and a 90-min wall ceiling.

## Setup

```bash
HEARTBEAT="data/reports/acc_sweep_status.txt"
DATE=$(date +%Y-%m-%d)
mkdir -p data/reports
echo "step0_started $DATE $(date -Iseconds)" > "$HEARTBEAT"
```

## Step 1 — Read the playbook

Read `.claude/commands/acc-sweep.md`. It defines:
- Scenario inventory and the `Expected:` header parsing rules
- Disambiguation hierarchy (filename → ACC data presence → recency)
- Universal ACC gates (collisions, TTC) and per-scenario gates
- The PASS/FAIL/WARN/SKIPPED/AMBIGUOUS verdict matrix

Follow it exactly. **Default to `--quick` mode for nightly runs** — never
launch fresh Unity at 4am under launchd's no-GUI Background context.

## Step 2 — Update heartbeat per scenario

```bash
echo "scenario_<name>_running $(date -Iseconds)" >> "$HEARTBEAT"
# ... evaluate scenario ...
echo "scenario_<name>_done verdict=<PASS|FAIL|WARN|SKIPPED|AMBIGUOUS> reason=\"<short>\" $(date -Iseconds)" >> "$HEARTBEAT"
```

The wrapper's `compose_subject()` parses these `scenario_*_done verdict=`
lines to compute the email subject. Format the verdict token EXACTLY —
the wrapper does literal grep matching.

## Step 3 — Write the report

Write `data/reports/acc_sweep_report.txt` with the full per-scenario table
from `.claude/commands/acc-sweep.md` Step 6. Include:
- Per-scenario row (scenario, base track, recording age, verdict, reason)
- Failures section (one-line reason per FAIL)
- Warnings section
- Skipped section with re-seeding instruction (`/e2e tracks/scenarios/<name>.yml`)
- Ambiguous section

## Step 4 — Final summary line (optional, fallback only)

If you want, print one summary line at the end matching the format below
— but the wrapper computes the email subject from the heartbeat directly,
so this is informational only:

```
$DATE ACC_SWEEP gate=$GATE pass=$P fail=$F warn=$W skip=$S amb=$A total=$T
```

Then exit. **Do not** generate additional commentary — the wrapper's hard
timeout doesn't wait for you to wax thoughtful.

## What NOT to do

- Do not commit code or open PRs. ACC sweep is read-only — observation only.
- Do not run `/diagnose` on failures during the sweep — log the failure
  and let the human decide whether to investigate.
- Do not invoke `--fresh` mode (Unity launches) under any circumstance —
  this prompt runs at 4am via launchd; fresh Unity here is unvalidated.
- Do not modify `tracks/scenarios/*.yml` to "fix" a marginal Expected:
  threshold — the headers are deliberate; tightening or loosening them
  is a human decision.
- Do not create per-scenario baselines in `tests/fixtures/` — that's a
  separate refactor (see deferred items in `docs/agent/tasks.md`).
