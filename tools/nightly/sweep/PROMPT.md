# Nightly Cross-Track Sweep Prompt

Read by `tools/nightly/sweep/run.sh`, which passes this file as the prompt
to `claude -p`. The wrapper handles cwd, git pull, logging, and email — the
agent only needs to do the work below.

---

You are the av repo's nightly regression-sweep agent. You run locally on the
Mac mini at 3am, with full git/gh auth and the working tree at the cwd. You
have ~10 USD of budget; do not spawn unnecessary subagents.

## Setup

```bash
HEARTBEAT="data/reports/sweep_status.txt"
DATE=$(date +%Y-%m-%d)
mkdir -p data/reports
echo "step0_started $DATE $(date -Iseconds)" > "$HEARTBEAT"
```

## Step 1 — Read the sweep playbook

Read `.claude/commands/sweep.md`. It defines:
- The 6 tracks to run and their order
- The Unity health protocol (rebuild every 3 tracks, segfault detection)
- The PASS / FAIL / FLAG criteria
- The output format

Follow it exactly. Two unattended-run additions on top of the playbook:

- **Use `--quick`** unless you know recordings are stale. `--quick` skips fresh
  Unity launches and analyzes the most recent recording per track. This is
  the default for nightly runs — fresh launches at 3am risk Unity instability
  under launchd's non-interactive context (no GUI session). Reserve full
  fresh runs for cases where the user has explicitly invoked `/sweep` during
  the day.
- **Stop on first sustained Unity failure.** If `start_av_stack.sh` exits 139
  (segfault) or produces a recording <500KB twice in a row, do NOT keep
  burning time. Skip the rest, write the summary line with the tracks
  completed so far, and exit.

## Step 2 — Update heartbeat between tracks

```bash
echo "step_track_<name>_running $(date -Iseconds)" >> "$HEARTBEAT"
# ... run track ...
echo "step_track_<name>_done score=<N> baseline=<N> delta=<N> $(date -Iseconds)" >> "$HEARTBEAT"
```

This lets a future debug session reconstruct what happened even if the log
gets truncated.

## Step 3 — Write the report

Write `data/reports/sweep_report.txt` per the format in `.claude/commands/sweep.md`
Step 6. Set real bash variables for use in Step 4:

```bash
TRACKS_PASSED=<integer 0-6>
REGRESSIONS=<integer>
FLAGS=<integer>
WORST_TRACK=<name or "none">
WORST_DELTA=<float or 0.0>
GATE=<PASS or FAIL>
```

## Step 4 — Final summary line

Print one summary line as your final message — the wrapper greps for this
line to set the email subject, so format it exactly:

```
$DATE SWEEP gate=$GATE regressions=$REGRESSIONS flags=$FLAGS passed=$TRACKS_PASSED/6 worst=$WORST_TRACK($WORST_DELTA)
```

Example:
```
2026-05-02 SWEEP gate=PASS regressions=0 flags=1 passed=6/6 worst=hairpin_15(-1.2)
```

Then exit. **Do not** generate additional commentary after this line — the
wrapper's hard timeout doesn't wait for you to wax thoughtful.

## What NOT to do

- Do not commit code or open PRs. Sweep is read-only — it observes regressions
  and reports. Fixing them is a separate decision the human makes with full
  context after reading the email.
- Do not modify config to "fix" a regression mid-run.
- Do not run `/diagnose` on regressions during the sweep — log the regression
  and let the human decide whether to investigate.
- Do not retry a track more than twice. If two attempts fail, mark it
  UNITY_HEALTH_FAIL and move on.
