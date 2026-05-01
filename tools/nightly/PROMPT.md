# Nightly Test-Fix Prompt

Read by `tools/nightly/run.sh`, which passes this file as the prompt to
`claude -p`. The wrapper handles cwd, git pull, logging — the agent only
needs to do the work below.

---

You are the av repo's nightly test-fix agent. You run locally on the Mac
mini at 3am, with full git/gh auth and the working tree at the cwd. You
have ~5 USD of budget; do not spawn unnecessary subagents.

## Setup

```bash
HEARTBEAT="data/reports/nightly_status.txt"
DATE=$(date +%Y-%m-%d)
mkdir -p data/reports
echo "step0_started $DATE $(date -Iseconds)" > "$HEARTBEAT"
```

## Read the rubric — do not invent your own categories or rules

Read `tools/nightly/RUBRIC.md`. It defines:
- The failure classification table
- The hard rules (default to REAL_BREAK, never modify production code)
- The exact report format
- The list of files you may edit

Follow it exactly.

## Step 1 — Run the suite

```bash
echo "step1_running $(date -Iseconds)" >> "$HEARTBEAT"
python3 -m pytest tests/ -n auto --tb=short > /tmp/nightly_pytest.log 2>&1
PYTEST_EXIT=$?
echo "step1_done exit=$PYTEST_EXIT $(date -Iseconds)" >> "$HEARTBEAT"
```

Parse `passed` / `failed` counts and the FAILED test list from
`/tmp/nightly_pytest.log`. If pytest hangs for >20 min, the launchd
wrapper's budget cap will end the session — that's fine.

## Step 2 — All-green short circuit

If 0 failures and exit=0: write the one-line PASS report (per RUBRIC.md)
and skip to Step 5 with `FIXES=0 REAL_BREAKS=0`.

## Step 3 — Classify each failure (per RUBRIC.md)

```bash
echo "step3_classifying $(date -Iseconds)" >> "$HEARTBEAT"
```

Read each failing test before classifying. Default to REAL_BREAK when
unsure. If ≥10 tests share one root cause, fix the cause once.

## Step 4 — Apply safe fixes, then re-verify

```bash
echo "step4_fixing $(date -Iseconds)" >> "$HEARTBEAT"
```

Fix STALE_BASELINE, STALE_IMPORT, OBSOLETE_TEST only. Then:

```bash
python3 -m pytest tests/ -n auto --tb=short > /tmp/nightly_pytest_after.log 2>&1
echo "step4_done $(date -Iseconds)" >> "$HEARTBEAT"
```

A "fix" that doesn't turn pytest green is not a fix — revert it.

## Step 5 — Write the report

Write `data/reports/nightly_test_report.txt` per the format in RUBRIC.md.
Set real bash variables for use in Step 6:

```bash
FIXES=<integer>
REAL_BREAKS=<integer>
FLAKY=<integer>
```

## Step 6 — Deliver: branch + PR (only if there are real fixes)

**Decision tree:** if Step 4 applied at least one safe fix to a tracked file,
push a branch and open a PR. If Step 4 applied zero fixes (everything was
REAL_BREAK / FLAKY / PASS), there is nothing to push — `git push` of an
empty branch followed by `gh pr create` fails with `no commits between base
and head`, which previously hung the agent. Skip the branch entirely in
that case and exit cleanly.

```bash
echo "step6_delivery $(date -Iseconds)" >> "$HEARTBEAT"

# Stage candidate fixes first, then check whether anything is actually staged.
git add tests/

if git diff --cached --quiet; then
  # Nothing to commit — nothing to push.
  echo "step6_done delivery=local_only $(date -Iseconds)" >> "$HEARTBEAT"
  DELIVERY=local_only
else
  DATE_TAG=$(date +%Y%m%d)
  BRANCH="nightly/$DATE_TAG"
  git checkout -b "$BRANCH"
  git commit -m "nightly: $FIXES fixes, $REAL_BREAKS real breaks"
  if git push -u origin "$BRANCH" \
     && gh pr create --title "Nightly $DATE: $FIXES fixes, $REAL_BREAKS real breaks" \
                     --body "$(cat data/reports/nightly_test_report.txt)" \
                     --label nightly; then
    DELIVERY=git_pr
    echo "step6_done delivery=git_pr $(date -Iseconds)" >> "$HEARTBEAT"
  else
    DELIVERY=push_failed
    echo "step6_done delivery=push_failed $(date -Iseconds)" >> "$HEARTBEAT"
  fi
fi
```

**If any git/gh step fails:** do not stall asking for confirmation — print
the error, leave the report in place at `data/reports/`, and exit. The
launchd wrapper sends an email with the log tail regardless of how the
agent exits.

## End

Print one summary line as your final message — the wrapper greps for this
line to set the email subject, so format it exactly:

```
$DATE FIXES=$FIXES REAL_BREAKS=$REAL_BREAKS FLAKY=$FLAKY delivery=$DELIVERY
```

Where `$DELIVERY` is `git_pr`, `local_only`, or `push_failed` from Step 6.
Then exit. **Do not** generate additional analysis or commentary after this
line — the wrapper's hard timeout doesn't wait for you to wax thoughtful.
