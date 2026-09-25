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

## Step 1.5 — Find the latest recording per track (--quick)

**Before scanning any recordings**, use the `latest_per_track()` function VERBATIM
from the `reference_hdf5_track_id` memory. Do NOT write your own HDF5 scan code.
The correct attribute path is `f.attrs['metadata'] → json → ['recording_provenance']['track_id']`.
Any variation (top-level `meta.get('track_id')`, `f['recording_provenance'].attrs`, etc.)
returns None for every recording. This exact mistake has occurred FIVE times
(nights 17, 25, 33, 34, 76). If all tracks return NO RECORDING FOUND, the lookup
is wrong — do not assume recordings are absent.

The verbatim function is in the `reference_hdf5_track_id` memory (see MEMORY.md).
If a track's latest recording has < 200 frames, fall back to the next-most-recent
recording for that track.

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

## Step 3.5 — Update project memory as a ROLLUP, not an append

Target memory: `project_sweep_trajectory_flags.md`.

**Default action is to EDIT THE ROLLUP IN PLACE, not to add a dated section.**

1. Compare tonight's per-track scores against the `## Current state` table at
   the top of that memory.
2. **If they match** (the overwhelmingly common case — nights 2–91 were all
   identical): update only the night number, the date range, and the staleness
   arithmetic in that section. Update the frontmatter `description` the same way.
   **Write nothing else.** Do not add a `## <date> nightly sweep` section.
3. **If tonight differs** — any score moved, a new e-stop appeared, a track
   errored, a fresh recording was used, or you hit the HDF5 lookup bug — THEN
   append a short dated entry under `## Divergent nights` describing only what
   differed, and refresh the rollup.

Why: this memory reached 48 KB / 78 sections of near-identical text before being
compacted on 2026-08-12. It is loaded on every session in this project, so the
repetition costs real context and buries the handful of nights that actually
mattered. See the `feedback_nightly_memory_unbounded_append` memory.

The same rule applies to any other memory you touch tonight.


## Speed Utilisation column (report-only, added 2026-09-25)

`analyze_drive_overall.py` now prints a `SPEED UTILISATION (report-only)`
section. Put its "vs allowed" median in a `Util` column of the report table and,
for any track under 0.85, one line naming the binding cap (curve_cap /
velocity_profile / target / comfort). Do NOT treat it as a gate — the verdict
rule is unchanged (every layer ≥ 95). Expect s_loop ≈ 0.49, hill_highway ≈ 0.55,
mixed_radius ≈ 0.57, highway_65 ≈ 0.91 on the frozen pool; a change in these
numbers on a fresh recording is worth a "notable" line, a change on a frozen
recording is not (see the stale-diagnostics rule above).

## Retro step — you are NON-INTERACTIVE

A `Stop` hook (defined in `~/.claude/settings.json`, fires on *every* session)
will ask you to invoke `/retro` before exiting. Do it — with one override.

**The skill's step 4 says to present a proposal and ask `Proceed?`. Do NOT ask.**
This job runs under `claude -p` from launchd. There is no human on the other end
of the pipe; a question here is silently discarded, the session exits 0, and the
wrapper reports success while the lesson is lost.

Instead, **write the files yourself**, then report what you wrote:

- Dedupe first (skill step 3). Prefer updating an existing memory in place over
  creating a new one — most nightly lessons are corrections to a memory that has
  gone stale, not new rules.
- Cap at **2 new memory files** per night. If more candidates survive dedupe,
  write the 2 highest-value and name the rest in your output.
- Always update `MEMORY.md` after adding a file.
- "Nothing durable tonight" is a legitimate and common result — say it and stop.

Verified cost: the acc-sweep job did exactly this on four consecutive nights
(2026-09-05 through -08), losing every lesson at `exit=0`. See memory
`feedback_nightly_retro_proposes_into_void`.

## What NOT to do

- **Do not append a new dated section to `project_sweep_trajectory_flags.md`
  when the result is unchanged.** Update the rollup in place (Step 3.5).
- Do not commit code or open PRs. Sweep is read-only — it observes regressions
  and reports. Fixing them is a separate decision the human makes with full
  context after reading the email.
- Do not modify config to "fix" a regression mid-run.
- Do not run `/diagnose` on regressions during the sweep — log the regression
  and let the human decide whether to investigate.
- Do not retry a track more than twice. If two attempts fail, mark it
  UNITY_HEALTH_FAIL and move on.
- **Do not surface stale per-recording diagnostics as "notable."** In `--quick`
  mode the same HDF5 files are reanalyzed each night. Signals like
  sweeping_highway LMPC feasibility (98.75%), hill_highway MPC solve time
  (P95=3.99ms), and contract consistency rates are frozen in those recordings —
  they cannot have changed. Only flag a diagnostic if it is (a) new this run or
  (b) measurably worse than a prior sweep's same recording. This mistake occurred
  7 times (nights 4, 29, 37, 47, 52, 53, 57) before this rule was added.
