Autonomous improvement loop: diagnose, fix, and validate until a goal is met across all affected tracks. Calls existing skills (`/e2e`, `/diagnose`, `/trace`, `/instrument`, `/plan-feature`) as needed.

The user's request is: $ARGUMENTS

## Step 0 — Parse request

Extract from $ARGUMENTS:
- **TARGET TRACK**: the primary track to improve (e.g., "mixed", "sloop")
- **GOAL**: score/layer target (default: "all layers >= 95")
- **MAX ITERATIONS**: hard cap (default: 5)

If no target track specified, ask the user.

---

## Step 1 — BASELINE (run once at start)

Before changing anything, establish the current state of ALL tracks.

### 1a — Read existing recordings

Check for recent recordings across all tracks:
```bash
ls -lt data/recordings/*.h5 | head -20
```

For each track that has a recent recording (< 24h old), run:
```bash
python3 tools/analyze/analyze_drive_overall.py <recording>
```

For tracks without recent recordings, note them as "no baseline."

### 1a2 — Verify recordings match current codebase

**CRITICAL:** Recordings made before code changes are INVALID baselines for validating those changes. Before trusting any existing recording:

1. Check when the recording was made: `ls -lt data/recordings/*.h5`
2. Check when key source files were last modified: `git log -1 --format=%ci control/regime_selector.py control/pid_controller.py`
3. If ANY code file was modified AFTER the recording was created → that recording cannot validate the code change. Re-run `/e2e` on that track first.

This prevents the mistake of: changing the regime selector, then "validating" on sloop with a recording made BEFORE the change — which shows old behavior, not new.

### 1b — Build baseline table

```
BASELINE SNAPSHOT (before any changes)
============================================================
Track           Score   Perception  Trajectory  Control   Status
highway_65      96.2    98.0        96.5        97.1      PASS
sloop           94.3    97.0        93.1        96.0      FAIL (trajectory < 95)
mixed_radius    79.0    97.0        79.1        96.0      FAIL (trajectory < 95)
sweeping        96.6    98.0        97.0        96.5      PASS
hairpin         79.0    ??          ??          ??        FAIL
hill            79.0    ??          ??          ??        FAIL
autobahn        ??      ??          ??          ??        no baseline
============================================================
```

### 1c — Build operating envelope

Run the physics-based regime boundary analysis instead of manually listing geometry:
```bash
python3 tools/analyze/analyze_regime_boundaries.py --all
```
This computes expected tracking error for PP, LMPC, and NMPC from first principles at each track's operating points, identifies the optimal controller per segment, and suggests regime config values. Use this to verify the regime selector config matches physics — if the tool says NMPC should handle a segment but config routes to PP, that's a regime misconfiguration, not a tuning problem.

If the suggested `mpc_max_map_curvature` or `nmpc_curvature_threshold` differ significantly from current config, prioritize fixing the regime boundary before tuning individual controller gains.

This envelope is the reference for predicting impact of any fix.

### 1d — Identify all tracks below goal

Don't just focus on the target — list every track that fails the goal. Check if they share symptoms.

---

## Step 2 — DIAGNOSE (holistic, cross-track)

### 2-pre — Prior Investigation Cross-Reference (MANDATORY)

Before diagnosing from scratch, check whether prior investigations already characterized this symptom:

1. **Run the convergence detector** to check for spinning-wheels patterns:
```bash
python3 tools/analyze/analyze_convergence.py --metric <target_metric> --track <target_track>
```
If this reports a CRITICAL ceiling (3+ failed attempts with the same root cause), **STOP.** Do not attempt another tuning/code fix. The metric has an architectural ceiling. Recommend `/plan-feature` to redesign the subsystem instead.

2. Search `docs/agent/tasks.md` for deferred tasks (T-XXX) referencing this symptom
3. Search memory files in `MEMORY.md` for related project/feedback memories
4. Search `docs/agent/current_state.md` for "DEFERRED" items on this topic

```
PRIOR INVESTIGATION CHECK
============================================================
  Convergence detector: <CLEAR / WARNING / CRITICAL>
    If WARNING+: <root cause>, <N failed attempts>, <approaches tried>
  Related tasks: <T-XXX — what it found, when, why deferred>
  Related memories: <memory file — key finding>
  Prior root cause: <what was already determined>
  Prior attempts: <what was tried and why it didn't work>
  
  Does the current symptom match the prior root cause? YES/NO
  If YES: skip redundant diagnosis, build on the prior finding
  If CRITICAL ceiling: STOP — recommend /plan-feature instead
============================================================
```

**If a prior investigation already characterized the root cause, do NOT re-diagnose.** State the prior finding, verify it still holds with current data, and proceed to fix-level triage. This prevents:
- Re-discovering what T-078 already proved (wasted E2E cycles)
- Building the wrong solution because you didn't see the prior evidence
- Contradicting prior findings without acknowledging the discrepancy
- Spinning wheels on an architectural ceiling that can't be tuned away

### 2a — Primary diagnosis on target track

Run ALL three diagnostic tools — analysis, triage engine, and trace:
```bash
# 1. Primary metrics
python3 tools/analyze/analyze_drive_overall.py <recording>

# 2. Triage engine (20+ pattern detectors — catches known issues instantly)
python3 tools/analyze/run_gate_and_triage.py <recording>

# 3. Signal chain blame (for lateral/trajectory issues)
python3 tools/analyze/trace_curve_entry.py <recording>
```

**Read triage output FIRST.** If the triage engine matches a known pattern, use that as the diagnosis — don't rediscover it manually. The triage engine knows about: floor_rescue, heading_oscillation, regime_blend, curvature_lag, and 15+ other patterns from PhilViz.

**Check §24–26 from the primary analysis** for regime/controller diagnostics:
- **§24 Curvature Distribution:** what κ values exist on this track, whether the map is quantized
- **§25 Per-Regime Error Breakdown:** RMSE per controller (PP/LMPC/NMPC) and per segment — identifies which controller owns the tracking error. Do NOT write ad-hoc HDF5 scripts for this data.
- **§26 Model-Plant Comparison:** steering command-vs-actual, reference alignment, heading prediction accuracy

4. Classify primary issue using the priority table from `/diagnose`
5. Run additional targeted tools only if triage didn't identify the pattern

### 2b — Cross-track symptom correlation

For EACH other track that is below goal (from step 1d):
- Does it show the same symptom category?
- Does it show the same root cause hint (even at lower severity)?

Methods:
- Read existing analysis output if available
- Run `python3 tools/analyze/analyze_drive_overall.py <recording>` on other failing tracks
- Compare: same subsystem flagged? Same issue type? Same curve geometry range?

```
CROSS-TRACK CORRELATION
============================================================
Primary symptom: <e.g., late curve turn-in>
Target track: mixed_radius (C3, R40, +60 frames late)

Other tracks with same/similar symptom:
  sloop:    <yes/no — describe if yes>
  hairpin:  <yes/no — describe if yes>
  hill:     <no — different issue category>

Symptom is: SYSTEMIC (N tracks) / ISOLATED (target only)
============================================================
```

### 2c — Complete investigation protocol (run ALL checks before presenting diagnosis)

The diagnosis must be COMPLETE before presenting to the user. Do NOT present a preliminary diagnosis that requires further investigation — that wastes an iteration.

**Step 2c1 — Signal sufficiency check:**
Check the HDF5 recording has all fields needed for the issue category:
```python
import h5py
f = h5py.File("<recording>", "r")
# Check fields based on issue category
```
If critical signals are missing → `/instrument` + `/e2e` BEFORE continuing.

**Step 2c2 — Root cause disambiguation plan:**
List ALL possible root causes for the symptom. For EACH, specify what check would confirm or rule it out:
```
DISAMBIGUATION PLAN
============================================================
Symptom: <what the metric shows>
Possible causes:
  A: <cause A> — check: <specific trace/analysis/data query>
  B: <cause B> — check: <specific trace/analysis/data query>
  C: <cause C> — check: <specific trace/analysis/data query>
============================================================
```

**Step 2c3 — Run ALL disambiguation checks:**
Execute ALL checks from the plan IN ONE PASS. Use parallel Agent calls when possible. Use existing recording data where possible — only re-run e2e if signals are missing.

Checks to consider per issue type:
- **Lateral error**: `trace_curve_entry.py` (blame), HDF5 lookahead/floor/FF analysis
- **Oscillation**: Check correlation with regime transitions (`trace_regime_transition.py`), grade angle, speed cap, floor activation
- **Jerk spikes**: Check frame drops (dt gaps), ceiling hits, regime handoffs
- **Heading suppression**: `trace_heading_suppression.py`, gate weight analysis
- **Safety (e-stops)**: Failure frame inspection, steering authority at curve apex

**Step 2c4 — Confirm root cause with evidence:**
```
ROOT CAUSE CONFIRMED
============================================================
Symptom: <metric>
Investigated:
  A: <cause A> — RULED OUT because <evidence>
  B: <cause B> — CONFIRMED because <evidence>
  C: <cause C> — RULED OUT because <evidence>

Root cause: <B — specific description>
Evidence: <specific frames, signals, correlations>
============================================================
```

Only proceed to Step 2e (fix level) after the root cause is CONFIRMED, not hypothesized.

### 2d — Regression origin check (MANDATORY if baseline regressed)

If the target track shows a score drop vs its frozen baseline (Δbl < -2.0 in scoreboard), check whether recent commits correlate with the regression. This is the difference between "the system has always had this limitation" and "we broke it last week" — the fix path is fundamentally different.

1. Find the last 10 commits that touched the subsystem the root cause implicates:
   ```bash
   git log --oneline -10 -- <relevant_files>
   ```

2. Check commit dates vs the baseline recording date:
   ```bash
   # Baseline freeze date (from scoring_baselines.json mtime or git log on the fixture):
   git log -1 --format=%ci tests/fixtures/scoring_baselines.json
   ```
   Commits made AFTER the baseline was frozen are suspects.

3. If ≥1 suspect commit exists AND the subsystem it touches matches the
   confirmed root cause from Step 2c4 → this becomes a "revert-or-forward-fix"
   decision point. Record suspect commits for presentation at Step 2g.

```
REGRESSION ORIGIN
============================================================
  Baseline date: <date>
  Current score vs baseline: <score> (Δbl <delta>)
  Suspect commits (touched root-cause subsystem after baseline):
    <sha> <date>  <title>
    <sha> <date>  <title>
  Bisect needed: YES/NO
  Revert viable: YES/NO (YES = reverting cleanly restores baseline)
============================================================
```

**If bisect is needed to identify which of multiple suspects caused it**, run the bisect as part of disambiguation — one `/e2e` per suspect commit with `git stash`/`git revert` between runs. The bisect result feeds into Step 2g fix options.

### 2d2 — Industry context check (MANDATORY)

Before proposing any fix approach, state how top AV companies handle this class of problem:

```
INDUSTRY CONTEXT:
  Problem class: <e.g., steering jerk, late curve turn-in, oscillation>
  Standard approach: <what Waymo/Aurora/Cruise/Comma do>
  Why it applies: <1 sentence>
  Anti-pattern to avoid: <ad-hoc fix that creates tech debt>
```

This prevents reinventing solutions with known-good industry patterns. Examples from this project:
- FF curvature floor (0.005→0.0): top companies use proportional atan(L×κ) with no floor
- Steering jerk: top companies use a universal output limiter after all controllers, not per-controller soft constraints
- Regime selection: top companies use κ×v² lateral acceleration budget, not speed thresholds + curvature guards

If the standard industry approach differs from the proposed fix, explain why — or adopt the standard approach.

### 2e — Classify fix level

Based on the full picture (not just one track):

```
FIX LEVEL CLASSIFICATION
============================================================
Level: TUNING / CONFIG / CODE PATCH / ARCHITECTURE

Evidence:
  - Symptom scope: <1 track / N tracks / all tracks in κ range>
  - Proxy stacking: <yes/no — are multiple params approximating same quantity?>
  - Per-track overlays working around this: <count> (<list>)
  
Rationale: <why this level, not higher or lower>
============================================================
```

Rules:
- If symptom is systemic (3+ tracks) → minimum CODE PATCH
- If proxy stacking detected → minimum ARCHITECTURE
- If 2+ overlays work around same subsystem → minimum ARCHITECTURE
- NEVER recommend per-track overlay as primary fix

### 2e2 — Regime transition completeness check

When diagnosing comfort issues (jerk spikes, steering discontinuities), check whether regime transition blending is symmetric:

- **Blend asymmetry:** Each controller regime (Stanley, MPC, NMPC) applies its own blend logic when IT is the active regime. But the "base" controller (PP) may not have blend logic for transitions TO it. This means Stanley→PP and MPC→PP transitions can produce full-step jumps in steering output.
- **How to detect:** Look for jerk spikes at regime transition frames. If `regime_blend_weight < 1.0` but the steering output jumps discontinuously, the blend is only applied in one direction.
- **Fix pattern:** Track `_last_regime_steering` at the orchestrator level and apply blend when returning to PP with `blend_weight < 1.0`.

This was learned when a 46.3 steering jerk spike on sloop at frame ~145 was traced to the Stanley→PP handoff — blend_weight was correctly ramping, but PP's code path had no blend logic.

### 2e3 — Signal locality check (preview vs at-car)

When a fix uses sensor signals for control decisions (e.g., curvature for regime selection), verify whether the signal needs lookahead:

- **At-car signals** (e.g., `curvature_map_abs` at current position) only react AFTER the car is already in the condition. For regime transitions that need lead time, this is too late.
- **Preview signals** (e.g., `curve_phase_preview_curvature_abs`, ~25m ahead) provide advance warning, allowing transitions to complete before the car enters the challenging geometry.
- **Best practice:** Use `max(at_car, preview)` so the guard fires on whichever sees the condition first.

This was learned when the map curvature guard initially used at-car curvature only — it fired too late on sloop, causing MPC to remain active into tight curves before the hold timer could complete the transition.

### 2e4 — Design smell checker (MANDATORY — run BEFORE proposing fix level)

Before proposing ANY fix, check the bottleneck mechanism for these 5 design smells. If ANY smell is detected, the fix level is minimum ARCHITECTURE — do NOT propose TUNING or CODE PATCH.

**Smell 1: Binary gate on continuous signal**
- Is the mechanism a binary on/off (state machine, threshold crossing)?
- Does the underlying signal vary continuously?
- Pattern: "gate oscillates", "threshold sensitivity", "release conditions fight activation"
- Fix pattern: Replace with proportional weight (0→1 smoothstep)
- Session examples: lookahead blend (EMA→local_gate_weight), heading gate (binary→proportional weight)

**Smell 2: Proxy stacking (multiple params ≈ one quantity)**
- Are there 2+ parameters that all approximate the same physical decision?
- Do they interact adversely (one overrides another, conflicting thresholds)?
- Pattern: "rate limit + jerk limit + ceiling clip", "speed table + curvature guard + per-track overlay"
- Fix pattern: Unify into one physics-based formula or planner
- Session examples: PP floor (speed table→sqrt(8R*e)), steering (3 limiters→motion profile)

**Smell 3: Frame-rate dependent formula**
- Does the formula use dt or dt² in a way that changes behavior on frame drops?
- Pattern: "jerk × dt²", "rate per frame", values that change at different FPS
- Fix pattern: Convert to physical units (per-second), verify dt-independence
- Session examples: jerk limiter (dt²→per-second), rate limit (per-frame→per-second)

**Smell 4: Static lookup table for physics quantity**
- Is a hand-tuned table approximating something computable from physics?
- Pattern: "speed table", "curvature table", "per-track config"
- Fix pattern: Replace with formula from first principles
- Session examples: PP floor table→sqrt(8R*e_target), entry table lowered

**Smell 5: Post-hoc clamp (independent limiter after the fact)**
- Is the mechanism a clamp/clip applied AFTER the main computation?
- Does it create discontinuities (step changes, ceiling hits)?
- Pattern: "ceiling creates jerk spike", "clamp overrides the controller"
- Fix pattern: Make the planner/controller aware of the constraint, plan smooth approach
- Session examples: steering ceiling→motion profile with demand-aware tapering

**Smell 6: Wrapper forwarding gap**
- Is the controller instantiated through a wrapper class (e.g., VehicleController→LateralController)?
- Are new config params added to the wrapper's constructor but NOT forwarded to the inner controller?
- Pattern: "config flag is true but feature is disabled at runtime", "unit tests pass but e2e doesn't"
- Fix pattern: Add a forwarding test, or use config-object passthrough to eliminate explicit forwarding
- Session example: 12 params (physics floor, motion profile) silently dropped by VehicleController

```
DESIGN SMELL CHECK
============================================================
Bottleneck mechanism: <what's being modified>

  □ Smell 1 — Binary gate?     <yes/no — describe if yes>
  □ Smell 2 — Proxy stacking?  <yes/no — list params if yes>
  □ Smell 3 — Frame-dependent? <yes/no — show formula if yes>
  □ Smell 4 — Static table?    <yes/no — what physics it approximates>
  □ Smell 5 — Post-hoc clamp?  <yes/no — what discontinuity it creates>

  Smells detected: <count>
  → If ≥ 1: minimum ARCHITECTURE. Use /plan-feature.
  → If 0: TUNING or CODE PATCH may be appropriate.
============================================================
```

**This check is MANDATORY.** Do NOT skip it. The process health Pareto shows 73% of issues originate at the design level — catching design smells here prevents 3 rounds of escalation.

### 2f — Multi-issue prioritization

If diagnosis found multiple issues:
```
Priority: safety > correctness > performance > comfort
```
Fix ONE issue per iteration. The exception: if two issues share a root cause, they're one fix.

### 2g — Present COMPLETE diagnosis to user — CHECKPOINT

The diagnosis presented here must be COMPLETE — all disambiguation done, root cause confirmed with evidence, design smells checked. The user should NOT need to say "investigate more."

```
ITERATION <N> — DIAGNOSIS (COMPLETE)
============================================================
Primary issue: <category>

ROOT CAUSE (confirmed):
  <description with specific evidence>
  Investigated & ruled out: <list of alternatives with why>

DESIGN SMELL CHECK:
  <results of 5-smell check>

CROSS-TRACK:
  <which other tracks share this root cause>

Fix level: <level>

FIX OPTIONS (enumerate ALL viable paths — do NOT collapse to one):

  If suspect commits exist from Step 2d:
    A. REVERT <sha> — restores baseline behavior
       Tradeoff: loses feature <X> that the commit added
       Risk: <low/med/high>
       Estimated effort: <minutes/hours>

    B. FORWARD-FIX — keep feature <X>, repair the regression
       Approach: <1-sentence summary>
       Risk: <low/med/high>
       Estimated effort: <hours/days>

  If architectural alternatives exist:
    C. <alternative approach>
       Tradeoff: <what it gains / loses>

  If only one viable path: state that explicitly and say why alternatives
  were ruled out.

Proposed approach (if only one path) OR recommended option (if multiple):
  <letter + 1-2 sentence rationale>

User decision required — do NOT proceed with implementation until user
picks an option (or confirms the single viable path).
============================================================
```

**WAIT for user confirmation before proceeding.** This is mandatory for all iterations. When multiple fix options exist, the user picks the option — do NOT auto-select even if one option is obviously cheaper, because the tradeoff (e.g., losing a feature) may matter more than the cost.

---

## Step 3 — PLAN (conditional on fix level)

### If ARCHITECTURE:
Run the `/plan-feature` skill workflow with the full cross-track evidence from step 2.
Present plan to user. Wait for approval.

### If CODE PATCH with unclear scope:
Run `/plan-feature` in lightweight mode — skip industry benchmarking, focus on:
- Exact files and methods to change
- Predicted impact on operating envelope
- Tests needed
Present to user. Wait for approval.

### If TUNING or CONFIG:
No formal plan needed. State:
```
PROPOSED FIX (TUNING/CONFIG)
============================================================
Parameter: <name> in <file>
Current value: <value>
Proposed value: <value>
Rationale: <why this value>

Predicted impact:
  κ < 0.005 (gentle):  <no effect / effect>
  κ 0.005-0.015 (mid): <no effect / effect>
  κ 0.015-0.025 (tight): <THIS IS THE TARGET>
  κ > 0.025 (hairpin):  <no effect / effect>
============================================================
```

---

## Step 4 — PREDICT

Before implementing, map the fix against the operating envelope from step 1c:

```
IMPACT PREDICTION
============================================================
Fix: <description>

Track           Min R   Max κ    Predicted effect     Confidence
highway_65      500     0.002    No change             High
sweeping        300     0.003    No change             High
autobahn        600     0.002    No change             High
hill            100     0.010    <effect>              <confidence>
mixed_radius    40      0.025    <TARGET — effect>     <confidence>
sloop           40      0.025    <effect>              <confidence>
hairpin         15      0.067    <effect>              <confidence>

Tracks near change boundary (MUST validate):
  - <track> — <why it's near the boundary>
  - <track> — <why it's near the boundary>
============================================================
```

"Tracks near change boundary" = tracks where the fix changes behavior at their operating point. These MUST be validated in step 6, even if they're currently passing.

---

## Step 5 — IMPLEMENT

Execute the fix based on the level:
- TUNING: edit config YAML
- CONFIG: edit config + dataclass
- CODE PATCH: edit source files
- ARCHITECTURE: follow the approved plan phase by phase

After implementation, run relevant unit tests:
```bash
pytest tests/ -v -k "<relevant_test_pattern>"
```

If tests fail, fix before proceeding to validation.

---

## Step 6 — VALIDATE

### 6a — Run target track
Use the `/e2e` skill workflow:
```
/e2e <target_track>
```

### 6a1 — Run duration sanity check

**CRITICAL:** Before running `/e2e`, verify the run duration matches the track geometry:

- **Non-looping tracks** (e.g., sloop — net 0° heading, car drives off end after ~1.5 laps): use `--duration 60` for a single clean lap. Duration 120 produces ~1740 frames that hit track-end geometry → OOL events and inflated RMSE.
- **Looping tracks** (e.g., mixed_radius, highway): can use longer durations, but prefer single-lap runs for consistency. Duration 120 on mixed produces 2-lap runs that average more tracking errors.
- **Default:** Use `--duration 60` unless you have a specific reason for longer runs. Single-lap scores are more stable and comparable across iterations.

Check track geometry: `tracks/<track>.yml` — if the waypoints don't form a closed circuit, the track doesn't loop.

### 6a2 — Validate code changes BEFORE tuning

**CRITICAL:** If the fix involves BOTH code changes and tuning changes, validate in two stages:

1. **Code-only**: Apply the code/architecture change with NO tuning tweaks. Run `/e2e` on target AND all boundary tracks. Record results.
2. **Code + tuning**: Only THEN layer tuning on top and re-validate.

This prevents a dangerous failure mode: tuning masks a code regression on boundary tracks. In this session, curve timing tuning improved mixed_radius PP tracking but degraded sloop by interacting with MPC reference trajectory generation — something invisible when only testing the target track.

If code-only already meets the goal, skip tuning entirely. Simpler changes are more robust.

### 6b — Run boundary tracks
For each track identified in step 4 as "near change boundary":
```
/e2e <boundary_track>
```

### 6c — Compare against baseline

```
VALIDATION RESULTS vs BASELINE
============================================================
Track           Baseline    Now     Delta    Status
<target>        79.0        ??      +??      <IMPROVED/SAME/REGRESSED>
<boundary_1>    96.2        ??      +??      <IMPROVED/SAME/REGRESSED>
<boundary_2>    94.3        ??      +??      <IMPROVED/SAME/REGRESSED>

Goal check: <target> all layers >= 95?  YES/NO
  Perception:  ?? (was ??)
  Trajectory:  ?? (was ??)
  Control:     ?? (was ??)
============================================================
```

---

## Step 7 — TRIAGE RESULTS

### If goal met AND no regressions:
```
ITERATION <N> — PASS
============================================================
Goal met: <target> all layers >= 95
No regressions detected.

Remaining tracks below goal: <list or "none">
→ If none: ALL DONE
→ If some: continue with next target (set target = worst failing track)
============================================================
```

### If goal NOT met (target still failing):
Record what improved and what didn't. Feed back into step 2 for next iteration.

### If regression detected on any track:
**Do NOT automatically reject.** Classify the regression:

| Type | Detection | Action |
|------|-----------|--------|
| **True regression** | Score dropped, fix directly caused it (same subsystem) | Revise approach — the fix is wrong for that geometry |
| **Exposed latent** | Score dropped, but different subsystem flagged than what we changed | Good finding — add to scope, fix both |
| **Score artifact** | Score dropped, but the previous score was masking a problem (e.g., MPC "smooth" on wrong line) | Update baseline, continue |
| **Acceptable tradeoff** | Small drop (<2 points) on passing track, large gain on target | Document justification, accept if all layers still >= 95 |
| **Convergence failure** | Same parameter change helps one track, hurts another (seen across 2+ iterations) | STOP TUNING — escalate to ARCHITECTURE |

```
REGRESSION TRIAGE
============================================================
Track: <track>
Score delta: <baseline> → <now> (<delta>)
Regression type: <type>
Evidence: <why this classification>
Action: <what to do>
============================================================
```

---

## Step 8 — GUARDS (checked every iteration)

### 8a — SAFETY (immediate stop + revert)
If ANY track shows:
- Emergency stops > 0 (that weren't in baseline)
- OOL events > 0 (that weren't in baseline)

→ **IMMEDIATELY revert the change.** Report to user. Do not continue.

### 8b — SEVERITY (pause + assess)
If any track score drops > 5 points from baseline:

→ **Pause.** Report the drop. Ask user whether to revert or investigate.

### 8c — SAME DIAGNOSIS (escalate fix level)
If the diagnosis in step 2 identifies the same root cause as the previous iteration:

→ The previous fix didn't work. Do NOT retry same approach.
→ Escalate fix level: TUNING → CONFIG → CODE PATCH → ARCHITECTURE
→ If already at ARCHITECTURE and same diagnosis repeats → stop, report to user.

### 8d — SCORE OSCILLATION (escalate to architecture)
Track scores across iterations. If over 3+ iterations:
- Track A improves then regresses then improves
- Track B regresses then improves then regresses
- Pattern: fixing one hurts the other

→ **STOP tuning.** This is proxy stacking.
→ Escalate to ARCHITECTURE with the oscillation evidence.
→ Run `/plan-feature` with the oscillation data.

### 8e — SCOPE CREEP (pause + re-plan)
If a TUNING fix has grown to touch 3+ files, or a CONFIG fix now requires code changes:

→ The fix level was misclassified. Pause.
→ Re-classify at the correct level. If ARCHITECTURE, run `/plan-feature`.

### 8f — UNITY HEALTH (rebuild, don't count)
If any `/e2e` run:
- Segfaults (exit 139)
- Produces recording < 500KB
- Shows stale Unity build warnings

→ Force Unity player rebuild: omit `--skip-unity-build-if-clean`
→ Retry the run. Do NOT count as an iteration.
→ After 6 total E2E runs in this session, force a rebuild on the next run.

### 8g — REVERT AVAILABLE (pause + ask)
If Step 2d found suspect commits AND the fix level is ARCHITECTURE or CODE PATCH
AND the user has not yet been presented with revert-vs-forward-fix options at Step 2g:

→ Pause. Do NOT proceed to Step 3 (PLAN). Return to Step 2g with fix options enumerated.
→ Rationale: architecture work is expensive; reverting a recent commit may be
  10 minutes and solves the immediate regression. The user should make that
  call explicitly before any planning work starts.

This guard is asymmetric from the others: most guards stop on failure. This
one stops on *option availability* — the skill should not silently eliminate
a cheap option without the user knowing it existed.

### 8h — MAX ITERATIONS (hard stop)
If iteration count reaches the max (default 5):

→ Stop. Report progress:
```
MAX ITERATIONS REACHED
============================================================
Iterations completed: 5
Starting baseline: <scores>
Current state: <scores>
Issues fixed: <list>
Issues remaining: <list>
Recommendation: <what to try next>
============================================================
```

---

## Step 9 — LOOP

If goal is not met and no guard has stopped the loop:
- Increment iteration counter
- Return to Step 2 (DIAGNOSE) with updated recordings

If goal IS met on target track:
- Check: are there other tracks below goal (from step 1d)?
- If yes: set target to the worst failing track, return to step 2
- If no: proceed to step 10

---

## Step 10 — COMPLETION

```
ITERATE COMPLETE
============================================================
Goal: <original goal>
Iterations used: <N> / <max>
Starting state:
  <track>: <baseline_score> → <final_score> (<delta>)
  <track>: <baseline_score> → <final_score> (<delta>)
  ...

Fixes applied:
  1. <fix description> (level: <level>)
  2. <fix description> (level: <level>)

Regressions: <none / list with classification>

Remaining work:
  - <any tracks still below goal>
  - <any exposed latent issues to address>

Promotion checklist:
  [ ] Update baselines: tests/fixtures/scoring_baselines.json
  [ ] Update docs: docs/agent/current_state.md
  [ ] Run full test suite: pytest tests/ -v
  [ ] Log fixes: /log-fix for each fix applied (updates improvement_log.json)
  [ ] Commit changes
============================================================
```

### Log fixes to improvement history

For EACH fix applied during the iterate loop, record it to the improvement log using `/log-fix`.
This feeds the `/process-health` Pareto that tracks where issues originate and how to prevent them.
Include: process stage (design/implementation/testing), root cause, detection method, and iterations to find.

---

## Reference

- Gate thresholds: `tools/scoring_registry.py`
- Track geometry: `tracks/*.yml`
- Field reference: `.claude/docs/hdf5_field_reference.md`
- Existing skills: `/e2e`, `/diagnose`, `/trace`, `/instrument`, `/plan-feature`, `/validate`
- Memory: check `MEMORY.md` for relevant project context and feedback
- Critical files: see `CLAUDE.md`
