Run ACC scenario sweep across `tracks/scenarios/*.yml` and verify each against the gate criteria embedded in its file header.

The user's request is: $ARGUMENTS

## Step 1 — Parse arguments

- No args → run all scenarios in `tracks/scenarios/`
- Scenario names → run only those (e.g., `H5 G2` or `highway_h5_stop_go hill_g2_stop_on_grade`)
- `--quick` (DEFAULT for nightly use) → analyze the latest matching recording per scenario; do NOT launch fresh Unity
- `--fresh` → launch fresh Unity for every scenario via `start_av_stack.sh --track-yaml tracks/scenarios/<scenario>.yml`. Use during the day, never at 3-5am under launchd (no GUI session, Unity stability unvalidated for 14 launches)
- `--scenario-list` → just print the scenario list and exit

## Step 2 — Inventory scenarios

```bash
ls tracks/scenarios/*.yml
```

For each scenario, the YAML file's header (commented `# ...` lines at top) declares the gate criteria as a free-text `Expected:` line, e.g.:

```
# H5 — Stop-and-go (highway_65)
# ...
# Expected: TTC min ≥ 2.0s; 0 collisions; detection ≥ 95% over ACC-active frames.
# base: tracks/highway_65.yml
```

Parse the `Expected:` line and the `# base: tracks/<X>.yml` line from each scenario.

## Step 3 — Find the recording per scenario (--quick mode)

Disambiguation challenge: ACC scenarios share `track_id` with their base track in `recording_provenance` metadata, so `track_id=highway_65` could be H5 ACC or pure lateral highway_65. Use this hierarchy to disambiguate:

1. **Filename heuristic**: if recording filename contains the scenario short-name (e.g., `recording_*h5*.h5`), trust it
2. **ACC data presence**: if recording has populated `radar_*` / `lead_*` HDF5 fields (non-zero detection count or non-empty range timeseries), treat as an ACC run on that base track
3. **Recency**: if multiple ACC recordings exist for one base track, take the most recent
4. **Best-effort**: a recording cannot be confidently mapped to one *specific* scenario (e.g., H5 vs H6 both on highway_65). Note ambiguity in the report and evaluate against the universal ACC gates rather than the scenario-specific `Expected:` line

If no recording on the base track is younger than 7 days, mark scenario as `SKIPPED — no recent recording (run /e2e tracks/scenarios/<name>.yml to seed)` and move on.

### Steering field gotcha (2026-05-03 finding)

When verifying Δsteering criteria from a scenario's `Expected:` line, **do
NOT use `control/calculated_steering_angle_deg`** — it's all-zero on every
normal AV-stack recording. That field is populated *only* by
`tools/ground_truth_follower.py`, a special tool that drives along
ground-truth lane centers; the regular planner/controller pipeline never
writes it (recorder writes 0 when the ControlCommand attribute is None).

Use one of these instead, depending on what you want to measure:

| Field                                      | What it is                                      |
|--------------------------------------------|-------------------------------------------------|
| `control/steering`                         | Final commanded steering [-1, 1] (use this for Δsteering) |
| `control/steering_pre_rate_limit`          | Output of the controller before rate limiting   |
| `control/steering_post_rate_limit`         | After rate limiter                              |
| `control/steering_post_jerk_limit`         | After jerk limiter                              |
| `control/steering_post_smoothing`          | After smoothing (final stage)                   |
| `control/steering_rate_limited_active`     | Bool — was rate limit clipping?                 |
| `control/steering_jerk_limited_active`     | Bool — was jerk limit clipping?                 |

If a scenario's Δsteering criterion is "≤ 0.10" with no units, treat as
the normalized [-1, 1] form on `control/steering`. The 2026-05-03 A2 WARN
("calculated_steering_angle_deg reads 0.0000 throughout") was a false
warning caused by reading the wrong field; A2's actual Δsteering is
verifiable from `control/steering` instead.

## Step 4 — Evaluate each recording against `Expected:`

Universal ACC gates (apply to every scenario, hard fails):
- 0 collisions (`ACC_COLLISION_GATE` from `tools/scoring_registry.py`)
- TTC min ≥ 2.0s during ACC-active frames (`ACC_TTC_MIN_GATE_S`)
- No emergency e-stops that weren't expected per scenario header

Scenario-specific gates (parsed from `Expected:` line, scenario-by-scenario):
- Detection rate threshold (e.g., `≥ 95%`)
- Specific TTC bounds (e.g., `≥ 2.5s`)
- Comfort criteria (e.g., `accel_p95 ≤ 3.0 m/s²`)

Use `tools/analyze/acc_pipeline_analysis.py --file <recording>` to extract the relevant metrics rather than rolling your own HDF5 reads.

## Step 5 — Per-scenario verdict

For each scenario, classify:

| Verdict | Meaning |
|---|---|
| **PASS** | All universal gates + all parsed `Expected:` gates met |
| **FAIL** | Any universal gate violated, OR any explicit `Expected:` criterion missed |
| **WARN** | Marginal (e.g., TTC_min between 2.0 and 2.5s); soft signal, not a gate failure |
| **SKIPPED** | No recent recording matched (--quick mode only) |
| **AMBIGUOUS** | Recording exists but can't be confidently mapped to this specific scenario |

## Step 6 — Output

For each scenario with a recording, run `python3 tools/analyze/acc_pipeline_analysis.py
--file <recording.h5>` and read **Card 5** for the composite ACC score
(0–100). Include it as a `Score` column alongside the existing `Verdict`.
The score is a continuous trend signal (Safety/Tracking/Behavior sub-layers,
weighted 50/30/20); the verdict is the binary pass-bar. **Both must appear** —
neither replaces the other. Score=`n/a` when ACC was inactive or fewer than
30 ACC-active frames.

```
ACC SCENARIO SWEEP
══════════════════════════════════════════════════════════════════════════
Scenario              Base                Rec age  Verdict  Score   Sub
A1 (autobahn_a1)      autobahn            <age>    PASS/FAIL/...  n/a    (ACC inactive)
H5 (highway_h5)       highway_65          <age>    FAIL     62.5   S25 T100 B100
H6 (highway_h6)       highway_65          <age>    FAIL     n/a    (<30 ACC frames)
G2 (hill_g2)          hill_highway        <age>    FAIL     73.7   S60 T90 B83
... (all scenarios)
══════════════════════════════════════════════════════════════════════════
GATE: PASS / FAIL  (FAIL if any scenario FAILed; SKIPPED do not block)

Failures:    <list with one-line reason or "none">
Warnings:    <list or "none">
Skipped:     <count> — re-seed with: /e2e tracks/scenarios/<name>.yml
Ambiguous:   <count> — base-track recording can't be uniquely attributed

Score color bands:  ≥90 GREEN · 70-89 YELLOW · 40-69 ORANGE · <40 RED
The Behavior sub-score (B) catches longitudinal oscillation / jerk / bang-bang
behaviors that gate verdicts miss — a scenario can pass all gates but still
report Behavior=42 (rough ride). Surface this in the Failures/Warnings text
when the score is low even if the verdict is PASS.
```

## Known limitations (V1)

- **No per-scenario baselines.** ACC scenarios are evaluated against the `Expected:` line embedded in each file, NOT against `tests/fixtures/scoring_baselines.json`. That file's per-track scores are lateral-only.
- **Disambiguation is best-effort.** Until a `recording_provenance.scenario_id` field exists in the HDF5 schema, the playbook uses recency + ACC-data-presence + filename heuristics. Recordings flagged AMBIGUOUS are honest about not knowing.
- **`/scores` and `/sweep` collisions remain.** Both currently bucket by base `track_id`. Use `/acc-sweep` for ACC-aware results until those tools learn the `--scenarios` flag (deferred).

## References

- Scenario YAMLs: `tracks/scenarios/*.yml`
- ACC gates: `tools/scoring_registry.py` (ACC_COLLISION_GATE, ACC_TTC_*, ACC_NEAR_MISS_*)
- ACC analysis CLI: `tools/analyze/acc_pipeline_analysis.py`
- ACC PhilViz tab: `tools/debug_visualizer/backend/acc_pipeline.py`
- Lateral sibling: `/sweep` (lateral base tracks only — does NOT include scenarios)
