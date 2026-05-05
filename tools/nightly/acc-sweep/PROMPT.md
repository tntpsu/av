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

Follow it exactly. **Default to `--quick` mode** for the first pass over
existing recordings. Then in Step 2.5, you may invoke `/e2e` to seed
fresh recordings for stale or failing scenarios, up to a per-night cap.

## Step 2 — Quick-pass evaluation + heartbeat per scenario

```bash
echo "scenario_<name>_running $(date -Iseconds)" >> "$HEARTBEAT"
# ... evaluate scenario in --quick mode against its YAML Expected: gate ...
echo "scenario_<name>_done verdict=<PASS|FAIL|WARN|SKIPPED|AMBIGUOUS> reason=\"<short>\" $(date -Iseconds)" >> "$HEARTBEAT"
```

The wrapper's `compose_subject()` parses these `scenario_*_done verdict=`
lines to compute the email subject. Format the verdict token EXACTLY —
the wrapper does literal grep matching.

## Step 2.5 — Auto-seed fresh /e2e for stale or failing scenarios

After the Step 2 quick pass, you have a list of scenarios with verdicts.
For SKIPPED, FAIL, and WARN scenarios, invoke `/e2e tracks/scenarios/<name>.yml`
to produce a fresh recording, then re-evaluate that scenario against its
YAML gate. This converts "no data" and "stale data" results into actual
verdicts.

**Soft cap: `max_fresh_runs_per_night = 5`.** Each `/e2e` takes ~3–5 min
of Unity time, so cap=5 adds ~20 min to the sweep — well within the 90 min
wall ceiling.

**Priority order** (spend the cap budget on the highest-value scenarios first):
1. **SKIPPED** — no recording within 7 days (these have NO data)
2. **FAIL**   — confirm the failure persists with fresh data
3. **WARN**   — last to consume budget

Once the cap is hit, leave the remaining stale scenarios with their
quick-pass verdict and note `reason="cap_reached"` in the report.

**Heartbeat invariant: emit exactly one `scenario_<name>_done verdict=` line
per scenario.** The wrapper's `compose_subject()` does `grep -c` on these,
so a duplicate would double-count the scenario in `total=`. For scenarios
you plan to re-evaluate via `/e2e`, defer the `_done` line until AFTER
the fresh run completes. Use `_e2e_running` / `_e2e_done` (no `verdict=`
token) for the fresh-run progress markers — those won't be counted:

```bash
# Step 2 quick pass for this scenario:
echo "scenario_<name>_running $(date -Iseconds)" >> "$HEARTBEAT"
# ... evaluate; quick-pass result is SKIPPED — defer _done, queue for /e2e ...

# Step 2.5 fresh run for this scenario:
echo "scenario_<name>_e2e_running $(date -Iseconds)" >> "$HEARTBEAT"
# ... invoke /e2e tracks/scenarios/<name>.yml — this launches Unity ...
echo "scenario_<name>_e2e_done $(date -Iseconds)" >> "$HEARTBEAT"

# Single _done line with the FINAL verdict:
echo "scenario_<name>_done verdict=<PASS|FAIL|WARN|AMBIGUOUS> reason=\"<short> (e2e fresh)\" $(date -Iseconds)" >> "$HEARTBEAT"
```

If `/e2e` itself fails (Unity won't build, hung launch, etc.), mark the
scenario `verdict=AMBIGUOUS reason="e2e_launch_failed"` and move on. Do
NOT retry the same scenario in the same night — one failed launch is
evidence enough that something's wrong with the harness.

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
  (Step 2.5 `/e2e` runs WRITE recordings to `data/recordings/`, but that's
  data, not code. Do not let `/e2e` trigger a code commit.)
- Do not run `/diagnose` on failures during the sweep — log the failure
  and let the human decide whether to investigate.
- Do not exceed `max_fresh_runs_per_night = 5` Unity launches in Step 2.5.
  Each launch is expensive and can hang; staying under the cap protects
  the 90 min wall ceiling and gives the human a predictable cost envelope.
- Do not modify `tracks/scenarios/*.yml` to "fix" a marginal Expected:
  threshold — the headers are deliberate; tightening or loosening them
  is a human decision.
- Do not create per-scenario baselines in `tests/fixtures/` — that's a
  separate refactor (see deferred items in `docs/agent/tasks.md`).
