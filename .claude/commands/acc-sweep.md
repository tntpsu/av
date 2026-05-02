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

```
ACC SCENARIO SWEEP
═══════════════════════════════════════════════════════════════
Scenario              Base                Recording age  Verdict
A1 (autobahn_a1)      autobahn            <age>          PASS/FAIL/WARN/SKIP
A2 (autobahn_a2)      autobahn            <age>          ...
H2 (highway_h2)       highway_65          <age>          ...
H3 (highway_h3)       highway_65          <age>          ...
H4 (highway_h4)       highway_65          <age>          ...
H5 (highway_h5)       highway_65          <age>          ...
H6 (highway_h6)       highway_65          <age>          ...
H7 (highway_h7)       highway_65          <age>          ...
H8 (highway_h8)       highway_65          <age>          ...
H9 (highway_h9)       highway_65          <age>          ...
H10 (highway_h10)     highway_65          <age>          ...
H11 (highway_h11)     highway_65          <age>          ...
G1 (hill_g1)          hill_highway        <age>          ...
G2 (hill_g2)          hill_highway        <age>          ...
═══════════════════════════════════════════════════════════════
GATE: PASS / FAIL  (FAIL if any scenario FAILed; SKIPPED do not block)

Failures:    <list with one-line reason or "none">
Warnings:    <list or "none">
Skipped:     <count> — re-seed with: /e2e tracks/scenarios/<name>.yml
Ambiguous:   <count> — base-track recording can't be uniquely attributed
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
