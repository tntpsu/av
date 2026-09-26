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

**Pre-flight gate.** Before doing ANY fresh `/e2e` work, check the env var:

```bash
if [ -n "${AV_NIGHTLY_NO_FRESH_E2E:-}" ]; then
  echo "[skill] Step 2.5 disabled by wrapper pre-flight (cadence too slow)"
  # Skip the entire Step 2.5 block. For each scenario that would have been
  # re-seeded, emit:
  echo "scenario_<name>_done verdict=SKIPPED reason=\"preflight_cadence_fail\" $(date -Iseconds)" >> "$HEARTBEAT"
  # ... then jump to Step 3.
fi
```

The wrapper sets this when its 10-second smoke test detects that Unity can't
sustain 30 FPS in the current execution context (typically launchd-at-4 AM
starvation). Honoring this saves 5 wasted Unity launches that would produce
garbage data scoring `n/a`.

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

## Step 3.5 — Update project memory as a ROLLUP, not an append

Target memory: `project_acc_sweep_baseline.md`.

**Default action is to EDIT THE ROLLUP IN PLACE, not to add a dated entry.**

1. Compare tonight's verdicts against the `## Current state` section at the top
   of that memory.
2. **If they match** (every night since 2026-05-14 has been an identical
   all-SKIPPED blackout): update only the night number, the pre-flight counter,
   the date, and the pool ages. Update the frontmatter `description` the same
   way. **Write nothing else.**
3. **If tonight differs** — any scenario produced a real verdict, a score moved,
   a scenario was freshly seeded, or a new crash signature appeared — THEN add a
   short dated entry under `## Divergent nights` describing only what differed,
   and refresh the rollup. A night that ends the blackout is a divergent night;
   record it fully.

Why: this memory reached 40 KB / 69 near-identical entries before being compacted
on 2026-08-12. It is loaded on every session in this project. See the
`feedback_nightly_memory_unbounded_append` memory.


## Report structure contract — the email renderer parses these

`tools/nightly/report_render.py` turns `acc_sweep_report.txt` into the HTML email.
Keep these exact shapes or the email degrades to the raw text:

- One results row per scenario: `ID  (track_id)  base_track  rec_age  VERDICT  score  sub-scores/note`
  (VERDICT ∈ PASS FAIL WARN SKIP AMB). The header line and `(Night N)` in the title.
- The canonical `GATE: PASS|FAIL` line and the `pass=N fail=N warn=N skip=N amb=N total=N` line.
- A `Changes from Night-N (date):` heading followed by one indented line per change,
  ending with a blank line.
- One block per FAIL/WARN starting with `ID — track_id [recording, age]`, whose
  indented body contains a `Root cause:` line and an `Action:` line (continuation
  lines indented further). Every `Action:` becomes a numbered "Next step" at the top
  of the email — write them as things a human can do tomorrow, not restatements of
  the failure.

## Gap-RMSE verdicts and grouping FAILs — read before writing "Root cause"

1. **Gate on `Post-conv RMSE vs EQ`, not `Gap Error RMSE`.** Card 2 now prints
   both. The old whole-run RMSE vs s* was unreachable by construction (IDM
   settles at s*/√(1−(v/v0)⁴); pool 17–24 m post-convergence on every steady
   scenario). H7/G1/A1 `Expected:` lines say "post-convergence gap RMSE vs IDM
   equilibrium ≤ 10m" since 2026-09-26. "n/a — never converged" is itself a
   finding (report why: detection loss, lead outran ego, run too short).
2. **Mechanism before grouping.** Three FAILs with the same *symptom* are not one
   cause. Before writing "same class as X", compare for each scenario:
   `acc_target_speed_mps` vs `target_speed_final` (is the ego being CAPPED below
   what ACC asks for → governor/tracking-budget problem), the sign of
   `acc_idm_accel_mps2` (asking to close but not closing → longitudinal
   authority problem), and `acc_idm_equilibrium_gap_m` vs the gap (at
   equilibrium → gate/spec problem). Night 45 grouped H7 (gate wording), A1
   (ego 0.35 m/s under target: speed_drag_gain, T-ACC-DRAG) and G1 (IDM
   equilibrium ≈ 1.3×s*) as "one uncalibrated-gate group"; only H7 was.
3. **A standing `bias (gap−EQ)` of +3–5 m is T-ACC-EQ-BIAS**, present on every
   scenario — report it, do not attribute it to the scenario under review.

## Detection-rate verdicts — read before FAILing anything on detection

1. **Gate only on what the scenario's `Expected:` line names.** `acc_pipeline_analysis`
   prints `[FAIL ≥95%]` next to detection on EVERY run; that is the tool's generic
   reference bar, not a sweep gate. H8's Expected is "smooth engage at ~60 m gap;
   no hunting; jerk P95 ≤ 4.0" — no detection criterion. Night 23 and night 43
   both FAILed H8 on it in error.
2. **Catch-up scenarios have run-length-dependent detection rates.** H7 and H8
   start the lead beyond radar range BY DESIGN (H8 header: "beyond radar range at
   start, ego drives free until ~27 s"). Whole-run detection = 1 − (out-of-range
   seconds / run length): 514 s → 86 %, 174 s → 62 %, with 100 % detection once
   engaged in both. A change in that number between runs of different duration is
   NOT a regression. Compare dropout events after engage, or the engaged-phase
   rate, and always state the run duration next to a detection percentage.
3. **Before attributing detection loss to the radar classifier, read
   `radar_fwd_reject_reason` and `radar_fwd_target_arc_distance_m` at the rejected
   frames.** `out_of_range` with arc distance > 150 m means the lead outran the
   ego — a speed/spec problem (A1: lead 20 vs ego 12; G1: ego held to 7.0 m/s on R100 by
   the PP tracking budget, T-GOV-TRACKING-BUDGET-SPEED). `opposite_direction` at 300 m+
   arc distance on a loop track is a car on the far side, correctly rejected.
   Heading deltas at accepted frames on A1/G1/H8 are 0.1–5.9°: the classifier is
   not the cluster's cause (2026-09-24 re-analysis).

## Scoring changes on 2026-09-22 — read before comparing against earlier nights

Three things changed that move scores WITHOUT any controller regression:

1. **Physical contact now forces the composite to 0.** `acc_pipeline_analysis`
   Card 5 counts `vehicle/lead_collision_detected` frames as collisions (the old
   `distance < 0` test was unsatisfiable — the Unity override clamps the range to
   0.1 m). Every earlier hill_g2 run, hill_g1 on 09-04/09-07 and highway_h5 on
   09-09 were physical contacts scored as "near-miss + e-stop" or PASS. Their
   re-scored composites will read 0. That is the scorer becoming honest, not a
   regression — say so in the report and do NOT open a T-SWEEP regression task
   for it.
2. **Near-miss is measured in the bumper frame.** `radar_fwd_distance_m` is
   centre-to-centre; bumpers touch at a reported ~4.43 m
   (`ACC_RADAR_RANGE_OFFSET_M`). Near-miss = reported gap < 6.43 m sustained.
   Expect near-miss counts to rise on close-following scenarios (H5, H6).
3. **The controller now sees the bumper gap** (`acc.radar_range_offset_m: 4.43`
   in `ForwardRadarSensor`), together with `acc.cutout_requires_no_lead`,
   `control.longitudinal.acc_jerk_cooldown_bypass_states`,
   `acc.emergency_brake_min_closing_mps: 0.1` and `acc.emergency_brake_abs_gap_m: 1.5`.
   **The recorded `radar_fwd_distance_m` is the controller's filtered gap** (not
   Unity's raw range), so in new recordings it reads ~4.4 m SMALLER than in
   pre-09-22 recordings of the same following distance. `recording_provenance.
   radar_range_offset_m` says which frame a file is in and the scorers use it.
   Gap-RMSE deltas vs pre-09-22 nights are baseline shifts, not regressions.
4. **E-stop events exclude EMERGENCY_BRAKE reflex frames.** The B1 bypass tags
   those with `emergency_stop=True`; only TTC_ESTOP / COLLAPSED_GAP_STOP / lateral
   stops count now. Near-miss excludes standstill frames (speed < 0.5 m/s).
   Expect e-stop counts to DROP on scenarios that stop behind a lead.

Card 3 also prints `Physical Contact: N frame(s)` — any N > 0 is a collision
regardless of the `Collision Frames` line above it.

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

Verified cost: on 2026-09-05, -06, -07 and -08 this exact job emitted a
well-formed proposal block ending in `Proceed?` and exited 0. All four nights'
lessons were lost — including three factual corrections to
`reference_hdf5_acc_schema_gaps.md`, which stayed wrong in the meantime. See
memory `feedback_nightly_retro_proposes_into_void`.

## What NOT to do

- **Do not append a new dated entry to `project_acc_sweep_baseline.md` when the
  result is unchanged.** Update the rollup in place (Step 3.5).
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
