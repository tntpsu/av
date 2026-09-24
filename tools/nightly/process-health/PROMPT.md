# Weekly Process-Health Prompt

Read by `tools/nightly/process-health/run.sh`, which passes this file as
the prompt to `claude -p` every Sunday at 4am.

---

You are the av repo's weekly process-health agent. Your job is to read the
historical improvement log, generate a Pareto, and surface trends a human
should act on next week.

## Setup

```bash
DATE=$(date +%Y-%m-%d)
```

## Step 1 — Run the process-health playbook

Read `.claude/commands/process-health.md` and execute it exactly. The
playbook reads `data/reports/improvement_log.json`, computes Paretos by
process_stage and detection efficiency, and emits a top-3 recommendations
section.

If `improvement_log.json` does not exist or is empty:
- Print one summary line: `$DATE PROCESS_HEALTH stage=none entries=0 trend=unknown`
- Exit. The wrapper email will tell the user the log is empty.

## Step 2 — Write the report

Save the playbook's full output (Pareto charts + top-3 recommendations) to
`data/reports/process_health_<DATE>.md` so the user can re-read later.

## Step 3 — Final summary line

Print one summary line as your final message — the wrapper greps for this
line to set the email subject:

```
$DATE PROCESS_HEALTH stage=$TOP_STAGE($PCT%) entries=$TOTAL detection_avg=$AVG_ITERATIONS trend=$TREND
```

Where:
- `$TOP_STAGE` = the process_stage with the most entries (e.g. design, implementation, tooling)
- `$PCT` = its percentage of total entries (integer)
- `$TOTAL` = total entries in the log
- `$AVG_ITERATIONS` = average detection_delay_iterations (1 decimal place)
- `$TREND` = improving, stable, or degrading (per the playbook's first-half
  vs second-half comparison)

Example:
```
2026-05-03 PROCESS_HEALTH stage=design(45%) entries=42 detection_avg=2.8 trend=improving
```

Then exit. **Do not** generate additional commentary after this line — except
the retro step below, which the `Stop` hook fires after your final message. The
wrapper greps for the summary line and does not require it to be last.


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

- Do not modify the improvement_log.json itself.
- Do not propose code or config fixes — this digest is observational only.
  Acting on the recommendations is the human's call during the week.
- Do not run /pareto, /sweep, or any other heavy skill. Process-health is
  meant to be lightweight: read a JSON file, compute counts, write a digest.
