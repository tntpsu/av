Re-score historical recordings under the current scoring code to distinguish pipeline drift (scoring formula changed) from controller regression (behavior changed). Recommends whether frozen baselines need updating.

The user's request is: $ARGUMENTS

## When to run this skill

Run `/revalidate` in these situations:

1. **After scoring code lands.** Commits touching `tools/drive_summary_core.py`, `tools/scoring_registry.py`, `tools/analyze/run_gate_and_triage.py`, or anything that computes layer scores / gates / penalties change how every past recording would score. Without re-validation, `/iterate` and `/sweep` will compare post-fix scores against pre-fix baselines and conclude either "false regression" or "false improvement".

2. **Before trusting an old recording.** `/iterate` §1a2 ("Verify recordings match current codebase") mandates checking that source files weren't modified after the recording. `/revalidate` is the authoritative implementation of that check — it actually re-scores instead of just comparing timestamps.

3. **Before updating frozen baselines in `tests/fixtures/scoring_baselines.json`.** A baseline update should be justified: either (a) the scoring formula changed (pipeline drift, re-freeze is correct) or (b) the controller got better (genuine improvement, re-freeze celebrates real progress). `/revalidate` separates these.

4. **After `/log-experiment` outcomes that depend on pre/post scores** where the scoring code landed between recordings. Without re-validation, the `metric_before` / `metric_after` fields are incommensurable.

## Step 1 — Parse arguments

`$ARGUMENTS` parsing:
- A single recording path (e.g., `data/recordings/recording_20260421_230013.h5`) → revalidate that one recording
- A track name (e.g., `g2_stop_on_grade`) → find the most recent recording for that track and revalidate
- `--golden` → revalidate all 5 golden recordings listed in `tests/fixtures/golden_recordings.json`
- `--baselines` → revalidate every recording that backs `tests/fixtures/scoring_baselines.json`
- `--since <git-ref>` → revalidate every recording newer than the commit (typical: `--since HEAD~5` after a scoring change)
- Empty → default to `--golden`

## Step 2 — Establish the "as recorded" score

For each target recording, extract the original scores from whichever source has them:

1. **Embedded summary in HDF5** (if present): `meta/analysis_summary_json` or `meta/scoring_summary_json`
2. **Sibling JSON**: `data/reports/gates/<recording_stem>_*.json`
3. **Fixture files**: `tests/fixtures/scoring_baselines.json` if the recording is a baseline anchor
4. **Fallback**: if none present, note "no as-recorded score available" — revalidation becomes a one-sided snapshot, not a diff

Capture per recording:
- Overall score (as recorded)
- Per-layer scores (Safety, Trajectory, Control, Perception, LongitudinalComfort, SignalIntegrity) as recorded
- Recording date + git SHA at time of recording (from `meta/git_sha` if present)

## Step 3 — Re-score under current code

```bash
python3 tools/analyze/analyze_drive_overall.py <recording>
```

Capture the current-code scores into the same shape as Step 2.

Repeat for every recording in scope.

## Step 4 — Diff and classify

For each recording, build the comparison table:

```
RE-VALIDATION — <recording>
═══════════════════════════════════════════════════════════
Layer                 As recorded    Now       Δ       Classification
Overall               <val>          <val>     <Δ>     <class>
Safety                <val>          <val>     <Δ>     <class>
Trajectory            <val>          <val>     <Δ>     <class>
Control               <val>          <val>     <Δ>     <class>
Perception            <val>          <val>     <Δ>     <class>
LongitudinalComfort   <val>          <val>     <Δ>     <class>
SignalIntegrity       <val>          <val>     <Δ>     <class>
═══════════════════════════════════════════════════════════
```

**Classification rules (per layer):**

| Condition | Classification | Interpretation |
|---|---|---|
| `|Δ| < 0.5` | **NOISE** | Ignore — within numerical tolerance |
| `Δ > 0.5` AND scoring code in layer touched since recording | **PIPELINE DRIFT (+)** | Layer got more lenient / correctly stopped penalizing a benign pattern. Baseline re-freeze is correct. |
| `Δ < -0.5` AND scoring code in layer touched since recording | **PIPELINE DRIFT (-)** | Layer got stricter. Confirm the new penalty is well-founded; re-freeze if so. |
| `|Δ| ≥ 0.5` AND scoring code in layer UNTOUCHED since recording | **CONTROLLER CHANGE** | Only possible if the recording's controller/config differs from current — flag for separate investigation (should not happen when revalidating the same recording). |
| Bidirectional: `|Δ| ≥ 0.5` on SignalIntegrity while Trajectory unchanged | **SIGNAL-ONLY DRIFT** | Scoring-only change — most common after /iterate fixes to the scoring code. |

"Scoring code in layer touched" = grep the commits between recording's git SHA and HEAD for changes in the layer's compute function (e.g., `signal_integrity_heading_penalty`, `safety_emergency_stop_penalty`).

## Step 5 — Baseline recommendation

For each recording flagged as baseline-relevant (golden recording or tests/fixtures/scoring_baselines.json entry):

```
BASELINE RECOMMENDATION — <recording>
  Current baseline:  <as-frozen value>
  Re-validated:      <now value>
  Δ:                 <delta>
  Recommendation:    <UPDATE / HOLD / FLAG>
  Rationale:         <1 sentence>
```

- **UPDATE**: Δ is fully explained by PIPELINE DRIFT (+/-) and the drift is intentional → update `tests/fixtures/scoring_baselines.json` and `BASELINE_SCORES` in `tests/conftest.py` per CLAUDE.md "To update baselines" protocol.
- **HOLD**: Δ is NOISE or offsets cancel at Overall level — keep baselines frozen.
- **FLAG**: Δ is classified as CONTROLLER CHANGE or unexplained → do NOT re-freeze. Investigate with `/diagnose` or `/iterate`.

## Step 6 — Report

```
REVALIDATE — <scope>
═══════════════════════════════════════════════════════════════════
Recordings revalidated: <N>
Scoring changes since baseline: <list of commit SHAs + subjects>

Per-recording classification:
  <rec1>: PIPELINE DRIFT (+<Δ>) — SignalIntegrity heading penalty lifted
  <rec2>: NOISE — no meaningful change
  <rec3>: FLAG — Safety dropped 5.0 with no scoring change → diagnose

Baseline recommendations:
  <rec1>: UPDATE (scoring formula changed 2026-04-22, new score is correct)
  <rec2>: HOLD
  <rec3>: FLAG — do not re-freeze, run /diagnose first

Next recommended action: <skill>
  → If any UPDATE: refresh baselines per CLAUDE.md, then run `pytest tests/test_scoring_regression.py -v`
  → If any FLAG: run `/diagnose <recording>` or `/iterate <task>`
  → If all HOLD / NOISE: `/revalidate` is clean, proceed with planned work
═══════════════════════════════════════════════════════════════════
```

## Integration with other skills

- **`/commit`** triggers `/revalidate` when the commit touches scoring code (see `commit.md` file-touched triggers).
- **`/iterate`** §1a2 calls `/revalidate` instead of the timestamp-only check when the target recording is older than the last scoring-code touch.
- **`/plan-feature`** validation phase cites `/revalidate` for scoring-layer plans (e.g., new penalty formula, weight change).
- **`/log-experiment`** — for pure scoring-change experiments, the `metric_before` / `metric_after` fields should be captured via `/revalidate` on the same recording, not via two different Unity runs.
- **`/sweep`** is the right tool when running fresh Unity runs on all tracks. `/revalidate` is the complement — it re-scores existing recordings without burning Unity cycles.

## Anti-patterns

- **Don't re-run Unity.** `/revalidate` is about re-scoring existing recordings. If you find yourself running `./start_av_stack.sh`, you want `/e2e` or `/sweep`, not `/revalidate`.
- **Don't silently update baselines.** Every baseline change must cite a PIPELINE DRIFT classification or a deliberate controller improvement; a silent re-freeze hides regressions.
- **Don't revalidate in isolation.** If the scoring code changed, revalidate ALL golden recordings — a single recording may not exercise the changed code path.
- **Don't conflate drift with improvement.** Pipeline drift that raises a score is not the controller getting better. Only genuine behavior changes (re-run under current controller on the same scenario) count as controller improvements.

## References
- Scoring code: `tools/drive_summary_core.py`, `tools/scoring_registry.py`
- Baseline fixtures: `tests/fixtures/scoring_baselines.json`, `tests/fixtures/golden_recordings.json`, `tests/conftest.py` (`BASELINE_SCORES`)
- Baseline update protocol: `CLAUDE.md` § Testing Protocol — Scoring Regression
- Fresh-run complement: `/sweep` (all tracks), `/e2e` (single track)
- Related: `/log-experiment` (captures metric_before / metric_after), `/iterate` §1a2 (recording freshness check)
