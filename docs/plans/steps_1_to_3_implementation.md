# Steps 1–3 Implementation Plan

**Created:** 2026-03-10
**Scope:** Track Coverage (Step 1) → Automated CI Regression (Step 2) → Grade & Banking (Step 3)
**Execution model:** Each step is self-contained. Steps must be completed in order — later steps depend on earlier ones.

---

## Step 1 — Track Coverage Expansion (S2-M5)

### Goal

Prove the PP + MPC control architecture generalizes beyond s_loop (tight, 40–50m radii, 4–7 m/s) and highway_65 (gentle, 500m radii, 15 m/s). If it doesn't, find out where it breaks and fix it before adding more complexity.

### What We Have Today

| Track | Radii | Speed | Controller | Score |
|---|---|---|---|---|
| s_loop | 40–50m | 4–7 m/s (cap-tracked) | PP only | 95.7/100 |
| highway_65 | 500m | 15 m/s | PP + MPC (66% MPC) | 97.1/100 |

Existing unused tracks: `oval.yml` (R50, 180° arcs), `circle.yml` (R50, 360°). These exercise R50 but nothing else.

### New Tracks to Create

Three new track profiles that fill geometry gaps:

#### Track A: `hairpin_15.yml` — Tight Hairpin (stress test for PP)

**Purpose:** Extreme curvature (R15–R20) forces PP to its geometric limits. Validates the floor rescue behavior and cap-tracking at very low speed.

```yaml
name: hairpin_15
road_width: 7.2
lane_line_width: 0.2
sample_spacing: 0.5          # finer spacing for tight arcs
loop: true
start_distance: 3.0
speed_limit_mph: 15.0        # low — hairpin speed
segments:
  - type: straight
    length: 40
  - type: arc
    radius: 20
    angle_deg: 120
    direction: right
    speed_limit_mph: 10.0
  - type: straight
    length: 30
  - type: arc
    radius: 15
    angle_deg: 150
    direction: left
    speed_limit_mph: 8.0
  - type: straight
    length: 50
  - type: arc
    radius: 20
    angle_deg: 90
    direction: right
    speed_limit_mph: 10.0
  - type: straight
    length: 30
  - type: arc
    radius: 15
    angle_deg: 90
    direction: left
    speed_limit_mph: 8.0
```

**Expected behavior:**
- Speed capped to 2–4 m/s on R15 arcs (κ=0.067, much tighter than s_loop κ=0.025)
- PP only — speed never reaches MPC threshold
- Lateral error will be high (~0.5–0.8m) due to arc-chord geometry at R15
- May need `sample_spacing: 0.5` for accuracy on tight arcs

**Risks:**
- **R15 may be below PP's minimum viable radius.** If RMSE > 0.6m or e-stops occur, try R20 as the minimum.
- Cap-tracking may not decelerate fast enough for the R15 entry. Watch `cap_tracking_error_p95` and `overspeed_into_curve_rate`.
- `sample_spacing: 0.5` may cause TrackBuilder to generate a mesh with too many triangles. If Unity crashes or is very slow, raise to `1.0`.

#### Track B: `mixed_radius.yml` — Mixed Geometry (transition stress test)

**Purpose:** Tests regime transitions — vehicle must switch between PP and MPC as speed changes through varied geometry. This is the closest thing to a "real road" we have.

```yaml
name: mixed_radius
road_width: 7.2
lane_line_width: 0.2
sample_spacing: 1.0
loop: true
start_distance: 3.0
speed_limit_mph: 45.0
segments:
  # Long approach — vehicle gets up to MPC speed
  - type: straight
    length: 300
    speed_limit_mph: 45.0
  # Gentle highway curve — MPC handles this
  - type: arc
    radius: 200
    angle_deg: 90
    direction: left
    speed_limit_mph: 35.0
  # Transition zone
  - type: straight
    length: 100
    speed_limit_mph: 30.0
  # Medium curve — may trigger PP/MPC handoff during arc
  - type: arc
    radius: 80
    angle_deg: 90
    direction: right
    speed_limit_mph: 25.0
  # Short straight
  - type: straight
    length: 60
  # Tight curve — forces decel into PP range
  - type: arc
    radius: 40
    angle_deg: 90
    direction: left
    speed_limit_mph: 18.0
  # Recovery straight — vehicle re-accelerates, MPC re-engages
  - type: straight
    length: 200
    speed_limit_mph: 45.0
  # Sweeping curve back home
  - type: arc
    radius: 150
    angle_deg: 90
    direction: right
    speed_limit_mph: 35.0
```

**Expected behavior:**
- MPC active on straights and R200/R150 curves (speed ~12–15 m/s)
- PP takes over for R40 curve (speed drops below 10 m/s via cap-tracking)
- R80 curve is the critical transition zone — speed may hover near 10 m/s handoff threshold
- Regime selector's hysteresis and blend ramp tested by real geometry transitions

**Risks:**
- **Mid-curve regime handoff at R80.** If speed is ~10 m/s in the R80 arc, the regime selector may oscillate between PP and MPC. Watch `regime_selector_mode` transitions during R80. If oscillation occurs, the hysteresis band in `regime_selector.py` needs widening.
- **Curvature threshold mismatch.** Base config curvature thresholds (curve_phase, reference_lookahead) are tuned for s_loop/highway extremes. R80/R200 may fall in uncalibrated territory. Watch `curve_phase` state and `reference_lookahead_m` during these arcs.
- **Speed planner may not have entries for R80/R200 curvatures.** The speed governor's curve cap uses curvature→speed mapping. If R200 (κ=0.005) is above `curve_cap_curvature_min` threshold, it may unnecessarily cap speed on the gentle curve.

**Config overlay needed:** `config/mpc_mixed.yaml`

```yaml
# Mixed-radius track overlay
control:
  regime:
    enabled: true
trajectory:
  target_speed: 12.0          # moderate — not pure highway
```

#### Track C: `sweeping_highway.yml` — Long Sweeping Highway (sustained MPC test)

**Purpose:** Sustained high-speed MPC operation with gentle, long curves. Tests MPC bias estimator drift over 60s of continuous MPC operation. Tests oscillation growth over long straight segments (the highway_65 issue).

```yaml
name: sweeping_highway
road_width: 7.2
lane_line_width: 0.2
sample_spacing: 1.0
loop: true
start_distance: 3.0
speed_limit_mph: 65.0
segments:
  - type: straight
    length: 800
    speed_limit_mph: 65.0
  - type: arc
    radius: 300
    angle_deg: 45
    direction: left
    speed_limit_mph: 55.0
  - type: straight
    length: 400
    speed_limit_mph: 65.0
  - type: arc
    radius: 400
    angle_deg: 60
    direction: right
    speed_limit_mph: 60.0
  - type: straight
    length: 600
    speed_limit_mph: 65.0
  - type: arc
    radius: 250
    angle_deg: 75
    direction: left
    speed_limit_mph: 50.0
  - type: straight
    length: 300
    speed_limit_mph: 65.0
  - type: arc
    radius: 350
    angle_deg: 80
    direction: right
    speed_limit_mph: 60.0
```

**Expected behavior:**
- MPC active >90% of frames (speed ~15 m/s throughout)
- R300 curves: κ=0.0033, well within MPC tuning (highway R500: κ=0.002)
- R250: κ=0.004, slightly tighter — tests MPC's curvature preview effectiveness
- Long 800m straight: tests oscillation growth over ~53s at 15 m/s
- Bias estimator should converge within first lap

**Risks:**
- **800m straight may reveal oscillation runaway** (the pre-MPC highway failure mode). If MPC's oscillation slope is positive over 53s, the bias estimator's alpha may need adjustment.
- **R250 (κ=0.004) may need different MPC weights.** Highway config has `mpc_q_lat: 0.5` tuned for κ=0.002. At κ=0.004, the lateral cost may be too soft. Watch `mpc_e_lat_m` P95 on R250 arcs vs R300/R400.
- Loop geometry must close. The angles (45+60+75+80 = 260°) need supplementary arcs/straights to total 360°. **The implementing agent must validate the track closes** by checking that the total heading change equals 360° (sum of all arc `angle_deg` × direction_sign = 0 for a loop, or ±360 for single-direction).

**Config overlay:** Use `config/mpc_highway.yaml` as-is (target_speed: 15.0).

### Track ↔ Phase 2.8 Dependency Map

**IMPORTANT:** Not all three new tracks have the same dependencies. Run them in this order:

| Track | Phase 2.8 needed? | Can run now? | Why |
|---|---|---|---|
| `hairpin_15` | No | ✅ Yes | PP-only (speed 2–4 m/s, never reaches MPC threshold). Fully self-contained. |
| `sweeping_highway` | No | ✅ Yes | Pure highway MPC — same pipeline as highway_65. Phase 2.8 improves quality but is not required to pass. |
| `mixed_radius` | **Yes, for regime transitions** | ⚠️ Partial | PP and MPC both active. Without Phase 2.8, the PP↔MPC handoff may exhibit recovery-mode interference (RC-1) and missing rate limiter (RC-2). Run a baseline pass to document current behavior, but **do not register a golden recording until after Phase 2.8 is validated.** |

**Recommended execution order:**
1. Run `hairpin_15` and `sweeping_highway` first — these can be fully completed (1B → 1C → 1D → gate) without any Phase 2.8 work.
2. Run `mixed_radius` baseline pass to characterize current failure modes.
3. Complete Phase 2.8 (2.8.1–2.8.4 + 2.8.6 validation).
4. Re-run `mixed_radius` with Phase 2.8 fixes, then register golden recording.

**mixed_radius golden recording gate:** Score ≥ 85/100 AND `regime_transition_steering_discontinuity_max < 0.05` AND 0 e-stops. The discontinuity gate is the key one — it cannot be met without Phase 2.8.

---

### Implementation Sequence

#### Phase 1A: Create Track YAMLs (no code changes)

1. Create `tracks/hairpin_15.yml`, `tracks/mixed_radius.yml`, `tracks/sweeping_highway.yml`
2. **Validation step:** For each track, compute the total heading change to verify the loop closes:
   - Sum all `angle_deg` values, weighted by direction (+1 for left arcs if the track turns left overall, etc.)
   - A closed loop must have net heading = ±360°
   - If a track doesn't close, add a compensating arc/straight at the end
3. Create `config/mpc_mixed.yaml` overlay for the mixed-radius track

#### Phase 1B: Validate Each Track (live runs)

Run order: hairpin → mixed → sweeping (easiest failure first).

For each track:
```bash
# 1. Run 60s
./start_av_stack.sh \
  --build-unity-player \
  --skip-unity-build-if-clean \
  --config <config_for_track> \
  --track-yaml tracks/<track>.yml \
  --duration 60 > /tmp/run_<track>.log 2>&1

# 2. Analyze
source venv/bin/activate && PYTHONPATH=/Users/philiptullai/Documents/Coding/av \
  python tools/analyze/analyze_drive_overall.py --latest

# 3. Check for failures
# Look at: score, e-stops, lateral RMSE, oscillation slope, regime transitions
```

**Config selection per track:**

| Track | Config overlay | Why |
|---|---|---|
| hairpin_15 | `config/mpc_sloop.yaml` | Low speed, PP only — s_loop overlay is closest match |
| mixed_radius | `config/mpc_mixed.yaml` | Custom overlay (created in 1A) |
| sweeping_highway | `config/mpc_highway.yaml` | High speed, sustained MPC |

**Standard post-run analysis (every track, every run):**
```bash
# Primary summary — score, gates, curve events
source venv/bin/activate && PYTHONPATH=/Users/philiptullai/Documents/Coding/av \
  python tools/analyze/analyze_drive_overall.py --latest

# PhilViz — frame-level inspection
pkill -f "debug_visualizer/server.py" 2>/dev/null; sleep 1
source venv/bin/activate && PYTHONPATH=/Users/philiptullai/Documents/Coding/av \
  python tools/debug_visualizer/server.py > /tmp/philviz.log 2>&1 &
# Open http://localhost:5001 — load latest recording
```

**Key analyze_drive_overall.py fields to check per track:**

| Track | Fields to inspect first | Healthy range |
|---|---|---|
| `hairpin_15` | `cap_tracking_error_p95`, `overspeed_into_curve_rate`, `pp_floor_rescue_delta_max_m` | cap_err ≈ 0, overspeed = 0, floor_rescue < 2.0m |
| `mixed_radius` | `regime_selector_mode` transition count, `mpc_recovery_mode_suppressed` rate, `regime_transition_steering_discontinuity_max` | transitions < 20/run, discontinuity < 0.05 |
| `sweeping_highway` | `mpc_e_lat_m` P95 per segment, oscillation slope on 800m straight, `mpc_bias_estimator_converged` | e_lat P95 < 0.05m, slope ≤ 0, bias converged within lap 1 |

---

#### Per-Track PhilViz Triage

**hairpin_15 failures:**

| Symptom in analyze output | PhilViz: where to look | Likely cause | Fix |
|---|---|---|---|
| E-stop, lateral error > 1.5m on arc | Layers tab → click e-stop frame → Control panel, check `pp_curve_local_lookahead_post_floor` | R15 below PP minimum viable radius — arc-chord offset at 4.4m lookahead exceeds road width | Increase min radius: R15→R20 in `hairpin_15.yml`. If still failing at R20, raise to R25 and document minimum viable radius. |
| `cap_tracking_error_p95` > 0.5 m/s | Drive Summary → Speed section, `overspeed_into_curve_rate` | Cap-tracking decelerating too slowly before R15 entry | Increase `cap_tracking_decel_gain` in `mpc_sloop.yaml` overlay (e.g. 2.0 → 2.5). A/B test 3 runs each. |
| `pp_floor_rescue_delta_max_m` > 3.0m | Curve Events panel → click C1/C2, check `preLdStepMin` | Lookahead floor structurally too high for R15 geometry | Reduce `pp_curve_local_lookahead_floor_speed_table` floor at low speed. Note: floor rescue > 2.0m on R15 is expected — only fix if it causes e-stops. |
| Score low (< 75) but no e-stops | Comfort Gates section → check which gate fails | Likely jerk P95 — tight arcs cause high lateral jerk at PP floor | Accept if jerk is the only failing gate; document as "PP geometric ceiling at R15". |

**mixed_radius failures:**

| Symptom in analyze output | PhilViz: where to look | Likely cause | Fix |
|---|---|---|---|
| Steering jerk at regime transition | MPC Diagnostics tab → filter to transition frames, check `mpc_rate_limiter_active` | Phase 2.8 RC-2 (missing rate limiter) — MPC output is not smoothed at handoff | Expected without Phase 2.8. Document and proceed. Fix in 2.8.3. |
| MPC→PP transition then RMSE grows | Layers tab → PP section after transition, check `mpc_steering_feedback_error` | Phase 2.8 RC-1 (recovery mode corrupts MPC state) — feedback state wrong at re-entry | Expected without Phase 2.8. Document. Fix in 2.8.1 + 2.8.2. |
| Regime oscillating rapidly at R80 arc | MPC Diagnostics → regime_selector_mode time series, look for alternating PP/MPC | Speed hovering at PP/MPC threshold (~10 m/s) during R80 section | Widen hysteresis: increase `pp_max_speed_mps` to 11.0 or add `regime_hold_frames: 10` so regime can't switch more than once per 10 frames. |
| Score < 85 but no e-stops, all gates pass | Check if score is penalizing MPC section only | MPC quality on R80/R200 uncalibrated — `mpc_q_lat` tuned for highway | Try `mpc_q_lat: 1.0` in `mpc_mixed.yaml` (between s_loop=0 and highway=0.5). A/B 3 runs. |
| E-stop on PP section (low-speed R40) | Layers → click e-stop frame, Control panel | Same as s_loop R40 failure modes — floor rescue, PP turn-in | Apply same s_loop overlay tuning (mpc_sloop.yaml defaults). |

**sweeping_highway failures:**

| Symptom in analyze output | PhilViz: where to look | Likely cause | Fix |
|---|---|---|---|
| `mpc_e_lat_m` P95 > 0.05m on R250 arcs | MPC Diagnostics → filter to arc sections, compare `mpc_e_lat_m` on R300 vs R250 | κ=0.004 (R250) is at edge of MPC tuning range (tuned for κ=0.002) — `mpc_q_lat` may be too soft | Increase `mpc_q_lat`: 0.5 → 1.0 in `mpc_highway.yaml` overlay. A/B 3 runs. |
| Oscillation slope > 0 on 800m straight | Drive Summary → steering time series, check `oscillation_slope_per_frame` if available, else plot raw steering in PhilViz frame view | MPC bias estimator not converging / oscillation runaway (pre-2.7 failure mode) | Reduce `mpc_bias_alpha` (e.g. 0.01 → 0.005). If still growing, check `mpc_e_lat_m` vs `mpc_e_heading_m` correlation — if heading-led, the curvature preview may be over-correcting. |
| Bias estimator not converged after lap 1 | MPC Diagnostics → `mpc_bias_m` time series | Long straights take longer to build curvature evidence | Expected — bias estimator should converge by lap 2. If still drifting after lap 2, reduce `mpc_bias_alpha`. |
| Score < 90 on sweeping but no gates fail | Check `centered_pct` and `lateral_rmse` | MPC e_lat slightly elevated throughout — steady-state offset | Check `mpc_e_lat_m` mean (not P95). If mean > 0.01m, MPC has a steady-state bias. Add `mpc_integrator_gain` or verify `groundTruthLaneCenterX` sign convention hasn't drifted. |

---

**General rule:** If a failure is labeled "Expected without Phase 2.8" above — document the failure mode and score, then move on. Do NOT attempt to tune around Phase 2.8 root causes with config changes; the config workarounds will be invalidated when 2.8 lands.

4. Document the failure, fix, and re-run. Do not move to the next track until current one passes (except mixed_radius regime transition failures — these are deferred to Phase 2.8).

#### Phase 1C: Run 3x Validation Per Track

Once a single run passes (score ≥ 85, 0 e-stops), run 3 more times to check variance:

```bash
# Run 3x, analyze each
for i in 1 2 3; do
  ./start_av_stack.sh \
    --skip-unity-build-if-clean \
    --config <config> \
    --track-yaml tracks/<track>.yml \
    --duration 60 > /tmp/run_<track>_${i}.log 2>&1
  source venv/bin/activate && PYTHONPATH=/Users/philiptullai/Documents/Coding/av \
    python tools/analyze/analyze_drive_overall.py --latest
done
```

#### Phase 1D: Register Golden Recordings

For each new track that passes 3x validation:

1. Pick the median-scoring recording as the golden recording
2. Copy it to `data/recordings/` (it's already there from the run)
3. Update `tests/fixtures/golden_recordings.json`:
   ```json
   {
     "tracks": {
       "s_loop": "data/recordings/recording_20260222_202957.h5",
       "highway_65": "data/recordings/recording_20260222_214028.h5",
       "hairpin_15": "data/recordings/recording_<timestamp>.h5",
       "mixed_radius": "data/recordings/recording_<timestamp>.h5",
       "sweeping_highway": "data/recordings/recording_<timestamp>.h5"
     },
     "documented_scores": {
       "s_loop": 95.6,
       "highway_65": 96.2,
       "hairpin_15": <actual_score>,
       "mixed_radius": <actual_score>,
       "sweeping_highway": <actual_score>
     }
   }
   ```
4. Update `tests/conftest.py` `BASELINE_SCORES`:
   ```python
   BASELINE_SCORES: dict[str, float] = {
       "s_loop":           95.6,
       "highway_65":       96.2,
       "hairpin_15":       <actual_score>,
       "mixed_radius":     <actual_score>,
       "sweeping_highway": <actual_score>,
   }
   ```
5. Run comfort gate regression to confirm green:
   ```bash
   pytest tests/test_comfort_gate_replay.py -v
   ```

#### Phase 1D-post: mixed_radius Re-validation After Phase 2.8

**Trigger:** Run this after Phase 2.8 steps 2.8.1–2.8.4 are all validated on highway.

```bash
# Re-run mixed_radius with Phase 2.8 fixes in place
for i in 1 2 3; do
  ./start_av_stack.sh \
    --skip-unity-build-if-clean \
    --config config/mpc_mixed.yaml \
    --track-yaml tracks/mixed_radius.yml \
    --duration 60 > /tmp/run_mixed_post28_${i}.log 2>&1
  source venv/bin/activate && PYTHONPATH=/Users/philiptullai/Documents/Coding/av \
    python tools/analyze/analyze_drive_overall.py --latest
  python tools/analyze/mpc_pipeline_analysis.py --latest
done
```

**Gate for golden recording registration:**
- Score ≥ 85/100
- 0 e-stops
- `regime_transition_steering_discontinuity_max` < 0.05
- `mpc_rate_limiter_active` rate < 15% (limiter is a safety net, not compensating for missing fixes)
- `mpc_steering_feedback_error` P95 < 0.01

Once all three runs pass this gate, pick the median-scoring run and register it as the golden recording (same 1D procedure as the other tracks). Update `tests/fixtures/golden_recordings.json` and `tests/conftest.py` `BASELINE_SCORES`.

#### Phase 1E: Tests

**New test file:** `tests/test_track_coverage.py`

```python
"""
Tests that all track YAMLs are valid and parseable by the Python track loader.
Does NOT require Unity — validates geometry only.
"""

class TestTrackGeometry:
    """Validate track YAML geometry is self-consistent."""

    @pytest.mark.parametrize("track_name", [
        "s_loop", "highway_65", "hairpin_15", "mixed_radius", "sweeping_highway"
    ])
    def test_track_yaml_loads(self, track_name):
        """Track YAML exists and is parseable."""
        # Load YAML, verify all segments have required fields

    @pytest.mark.parametrize("track_name", [
        "s_loop", "highway_65", "hairpin_15", "mixed_radius", "sweeping_highway"
    ])
    def test_track_loop_closes(self, track_name):
        """For loop tracks, verify total heading change = ±360°."""
        # Walk segments, compute heading delta per arc
        # straight: heading_delta = 0
        # arc: heading_delta = angle_deg * (1 if left else -1)
        # abs(sum) should be 360.0

    @pytest.mark.parametrize("track_name", [
        "s_loop", "highway_65", "hairpin_15", "mixed_radius", "sweeping_highway"
    ])
    def test_track_curvature_in_range(self, track_name):
        """All arc curvatures (1/radius) are within supported range [0.001, 0.1]."""
        # κ < 0.001 is essentially straight (R > 1000m)
        # κ > 0.1 is R < 10m — below PP's minimum viable radius

    def test_track_segment_speed_limits_exist(self):
        """Every segment with an arc has a speed_limit_mph set."""
        # Missing speed limits cause silent fallback to track-level default,
        # which may be dangerously fast for tight arcs

    @pytest.mark.parametrize("track_name", [
        "hairpin_15", "mixed_radius", "sweeping_highway"
    ])
    def test_new_track_has_golden_recording(self, track_name):
        """New tracks have golden recordings registered (after 1D)."""
        # Skip if golden_recordings.json doesn't have this track yet
        # Once registered, verify the file exists
```

**Update existing tests:**
- `tests/test_comfort_gate_replay.py` will automatically pick up new golden recordings (it iterates `golden_recordings.json`). No changes needed to the test file itself.

#### Phase 1F: PhilViz Changes

**No PhilViz code changes needed for Step 1.** The existing analysis pipeline handles any track — recordings are track-agnostic HDF5 files. PhilViz will display data from any recording opened.

**However**, verify that:
- The "Curvature Contract Health" section in `analyze_drive_overall.py` correctly identifies curve turns on the new tracks (check C1/C3 entries in the output)
- If hairpin_15 has many short curves close together, the curve turn detector may merge them. This is informational, not a blocker.

### Acceptance Criteria — Step 1

**Step 1 has two completion gates — Part A (before Phase 2.8) and Part B (after Phase 2.8).**

#### Part A — hairpin_15 + sweeping_highway (no Phase 2.8 dependency)

| Criteria | Gate |
|---|---|
| 3 new track YAMLs exist and are valid | `test_track_geometry` all green |
| hairpin_15: 3× runs, 0 e-stops | Live validation logs |
| sweeping_highway: 3× runs, 0 e-stops | Live validation logs |
| Lateral RMSE ≤ 0.60m (hairpin), ≤ 0.30m (sweeping) | `analyze_drive_overall.py` output — sweeping is 55% arc so naturally higher than highway_65 |
| Score ≥ 75/100 on hairpin_15 | `analyze_drive_overall.py` output |
| Score ≥ 90/100 on sweeping_highway | `analyze_drive_overall.py` output |
| mixed_radius baseline run documented (failures noted, not required to pass) | Run log + analysis note |
| Golden recordings registered for hairpin_15 and sweeping_highway | `golden_recordings.json` updated |
| Comfort gate regression green (hairpin + sweeping recordings) | `pytest tests/test_comfort_gate_replay.py -v` |
| Existing s_loop and highway scores not regressed | Same test suite |

#### Part B — mixed_radius (requires Phase 2.8 complete)

| Criteria | Gate |
|---|---|
| mixed_radius: 3× runs, 0 e-stops | Live validation logs |
| Score ≥ 85/100 | `analyze_drive_overall.py` output |
| `regime_transition_steering_discontinuity_max` < 0.05 | `mpc_pipeline_analysis.py --latest` |
| `mpc_rate_limiter_active` rate < 15% | `mpc_pipeline_analysis.py --latest` |
| Golden recording registered for mixed_radius | `golden_recordings.json` updated |
| Comfort gate regression green (all 5 tracks) | `pytest tests/test_comfort_gate_replay.py -v` |

### Risks Summary — Step 1

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| R15 hairpin below PP minimum viable radius | Medium | Track A fails | Increase to R20; if still failing, R25. Document minimum viable radius. |
| Mixed-radius regime oscillation at R80 | Medium | Score drops, jerky handoff | Widen regime_selector hysteresis; add MPC hold timer |
| Sweeping highway doesn't close as a loop | High | Unity crashes or infinite road | Validate heading sum = 360° before first run |
| New track triggers untested curvature thresholds | Medium | Curve phase/lookahead misbehavior | Use base config defaults first; tune per-track overlay only if needed |
| Golden recordings too large for git | Low | Repo bloats | Recordings are typically 8–20MB; acceptable for 3 new files. Add to `.gitignore` if policy changes. |

---

## Step 2 — Automated A/B CI Regression (T-033)

### Goal

Any config or code change that touches the control/trajectory/analysis pipeline is automatically validated against golden recordings. A PR that regresses score by >2 points is blocked.

### What We Have Today

- GitHub Actions CI runs `pytest tests/ -v` on push to main/develop and PRs
- Comfort gate replay tests exist (`test_comfort_gate_replay.py`) — 19 tests, 2 tiers
- Golden recordings exist for s_loop and highway_65 (and new tracks from Step 1)
- `conftest.py` auto-bundles gate results to `data/reports/gates/` on green runs
- `run_ab_batch.py` can run A/B trials but requires Unity (not suitable for CI)

### Architecture Decision

**CI cannot run Unity** (requires macOS + Unity Editor/Player, not available on GitHub ubuntu runners). Therefore:

- **Tier 1 CI (GitHub Actions):** Replay-only tests against golden recordings. No live runs. Tests synthetic data + golden HDF5 files. This is what we already have.
- **Tier 2 CI (local gate):** Pre-push hook or manual script that runs live A/B trials on the developer's machine. Produces a gate report.
- **Tier 3 (future):** Self-hosted macOS runner with Unity Player for full live CI.

This plan implements Tier 1 (enhanced) + Tier 2 (new).

### Implementation Sequence

#### Phase 2A: Enhance CI Golden Regression Tests

**File:** `tests/test_comfort_gate_replay.py` — add per-track parametrized tests.

Currently the golden recording tests iterate tracks from the manifest. After Step 1, this automatically covers 5 tracks. But we need to add:

1. **Score delta check:** If score drifts by > SCORE_TOLERANCE (2.0 pts) from baseline, fail.
   - This already exists in the golden regression tests. Verify it works for new tracks.

2. **Metric-level regression checks** (new tests):

```python
class TestMetricRegression:
    """Per-metric regression checks against golden baselines."""

    @pytest.mark.parametrize("track_id", list(BASELINE_SCORES.keys()))
    def test_lateral_rmse_not_regressed(self, track_id):
        """Lateral RMSE should not increase by more than 20% from baseline."""

    @pytest.mark.parametrize("track_id", list(BASELINE_SCORES.keys()))
    def test_comfort_gates_pass(self, track_id):
        """All 7 comfort gates must pass on golden recording."""

    @pytest.mark.parametrize("track_id", list(BASELINE_SCORES.keys()))
    def test_no_emergency_stops(self, track_id):
        """Zero e-stops on golden recording."""
```

3. **Config hash check** (new, catches accidental config changes):

```python
def test_base_config_hash_stable():
    """Base config YAML hash hasn't changed unexpectedly.
    If this fails, it means av_stack_config.yaml was modified.
    Update the hash after intentional changes + re-validation."""
    import hashlib
    config_path = REPO_ROOT / "config" / "av_stack_config.yaml"
    actual_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()[:16]
    # Store expected hash in conftest.py, update after validated changes
    assert actual_hash == EXPECTED_CONFIG_HASH, (
        f"Base config changed (hash {actual_hash} != {EXPECTED_CONFIG_HASH}). "
        "If intentional, update EXPECTED_CONFIG_HASH after re-validating all tracks."
    )
```

**File:** `tests/conftest.py` — add:
```python
EXPECTED_CONFIG_HASH = "<sha256_first_16_chars>"  # Update after validated config changes

# Per-track metric baselines (populated from golden recording analysis)
BASELINE_METRICS: dict[str, dict[str, float]] = {
    "s_loop": {
        "lateral_rmse": 0.203,
        "accel_p95": 1.32,
        "commanded_jerk_p95": 1.44,
        "steering_jerk_max": 18.4,
        "centered_pct": 97.1,
    },
    "highway_65": {
        "lateral_rmse": 0.044,
        # ... etc
    },
    # ... new tracks from Step 1
}
METRIC_REGRESSION_TOLERANCE = 0.20  # 20% — metric can worsen by up to 20%
```

#### Phase 2B: Local Pre-Push Gate Script

**New file:** `tools/ci/run_local_gate.sh`

A script developers run before pushing changes that touch config or control code. It:

1. Runs the full test suite (offline)
2. If config changed: runs live E2E on all registered tracks (requires Unity)
3. Produces a gate report JSON

```bash
#!/bin/bash
# Usage: tools/ci/run_local_gate.sh [--live]
#
# --live: also run live E2E on all tracks (requires Unity Player built)
#
# Exit code: 0 = pass, 1 = fail

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "=== Phase 1: Offline Tests ==="
cd "$REPO_ROOT"
source venv/bin/activate
PYTHONPATH="$REPO_ROOT" pytest tests/ -v --tb=short
OFFLINE_EXIT=$?

if [[ "$1" == "--live" ]]; then
    echo "=== Phase 2: Live E2E Gate ==="
    # For each track in golden_recordings.json:
    #   1. Run 60s
    #   2. Analyze
    #   3. Compare score to baseline
    #   4. Fail if regression > 2 pts
    PYTHONPATH="$REPO_ROOT" python tools/ci/run_live_gate.py
    LIVE_EXIT=$?
else
    LIVE_EXIT=0
fi

# Produce gate report
PYTHONPATH="$REPO_ROOT" python tools/ci/generate_gate_report.py \
    --offline-exit $OFFLINE_EXIT \
    --live-exit $LIVE_EXIT

exit $(( OFFLINE_EXIT || LIVE_EXIT ))
```

**New file:** `tools/ci/run_live_gate.py`

```python
"""
Run live E2E validation on all tracks in golden_recordings.json.
Compare results to BASELINE_SCORES. Exit 1 if any track regresses > 2 pts.

Requires Unity Player to be built (--skip-unity-build-if-clean handles this).
"""

# For each track_id in golden manifest:
#   1. Determine config overlay (track_id → config mapping)
#   2. Run: start_av_stack.sh --skip-unity-build-if-clean --config X --track-yaml Y --duration 60
#   3. Analyze: drive_summary_core.analyze_recording_summary(latest_recording)
#   4. Compare score to BASELINE_SCORES[track_id]
#   5. Report pass/fail per track

TRACK_CONFIG_MAP = {
    "s_loop":           "config/mpc_sloop.yaml",
    "highway_65":       "config/mpc_highway.yaml",
    "hairpin_15":       "config/mpc_sloop.yaml",       # PP only, s_loop overlay
    "mixed_radius":     "config/mpc_mixed.yaml",
    "sweeping_highway": "config/mpc_highway.yaml",
}
```

**New file:** `tools/ci/generate_gate_report.py`

Writes a JSON report to `data/reports/gates/<timestamp>_local_gate/report.json`:
```json
{
  "timestamp": "2026-03-10T12:00:00Z",
  "git_sha": "abc1234",
  "offline_tests": {"passed": 633, "failed": 0, "skipped": 9},
  "live_e2e": {
    "s_loop":           {"score": 95.7, "baseline": 95.6, "delta": 0.1, "pass": true},
    "highway_65":       {"score": 97.1, "baseline": 96.2, "delta": 0.9, "pass": true},
    "hairpin_15":       {"score": 88.2, "baseline": 88.0, "delta": 0.2, "pass": true},
    "mixed_radius":     {"score": 91.5, "baseline": 91.0, "delta": 0.5, "pass": true},
    "sweeping_highway": {"score": 96.8, "baseline": 96.5, "delta": 0.3, "pass": true}
  },
  "decision": "PASS"
}
```

#### Phase 2C: GitHub Actions Enhancement

**File:** `.github/workflows/tests.yml` — add a step after the existing test run:

```yaml
    - name: Verify golden recording baselines
      run: |
        pytest tests/test_comfort_gate_replay.py -v --tb=short
      # This runs automatically, but making it explicit ensures
      # the golden regression is visibly separate in CI output

    - name: Verify config hash
      run: |
        pytest tests/test_config_integration.py::test_base_config_hash_stable -v
      # Catches accidental config changes that weren't validated
```

No Unity-dependent steps added to GitHub Actions (no macOS runner available).

#### Phase 2D: Tests for CI Infrastructure

**New file:** `tests/test_ci_infrastructure.py`

```python
"""Tests for CI infrastructure itself — validates the gate system works."""

class TestGateReport:
    def test_gate_report_schema(self):
        """Gate report JSON matches expected schema."""

    def test_track_config_map_complete(self):
        """Every track in golden_recordings.json has a config in TRACK_CONFIG_MAP."""

    def test_baseline_scores_match_manifest(self):
        """BASELINE_SCORES keys match golden_recordings.json tracks."""

    def test_config_hash_is_current(self):
        """EXPECTED_CONFIG_HASH matches actual config file hash."""
        # This is the actual enforcement test — also in Phase 2C

    def test_metric_baselines_have_all_tracks(self):
        """BASELINE_METRICS has entries for all tracks in BASELINE_SCORES."""
```

#### Phase 2E: PhilViz Changes

**No PhilViz code changes.** The existing Gates tab already displays gate bundles from `data/reports/gates/`. The local gate script writes reports there, so they'll appear automatically.

**Verify:** After running `tools/ci/run_local_gate.sh --live`, check that PhilViz Gates tab shows the new report.

### Acceptance Criteria — Step 2

| Criteria | Gate |
|---|---|
| `pytest tests/test_comfort_gate_replay.py -v` passes for all 5 tracks | CI green |
| Config hash test catches unvalidated config changes | Manually test: change one YAML value, run test, confirm failure |
| `run_local_gate.sh` produces valid JSON gate report | Manual run |
| `run_local_gate.sh --live` runs E2E on all tracks, compares to baselines | Manual run |
| GitHub Actions workflow includes golden regression step | PR test |
| Per-track metric baselines are populated in conftest.py | Code review |

### Risks Summary — Step 2

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Golden recordings are too large for GitHub Actions cache | Low | CI can't run golden tests | Golden recordings are 8–20MB; GitHub caches up to 10GB. Use `actions/cache` if needed. But note: golden recordings must be committed or LFS'd. |
| Config hash test is too brittle (fails on benign whitespace changes) | Medium | Developer friction | Hash the parsed YAML dict (sorted keys), not raw bytes |
| Live gate takes too long (5 tracks × 60s = 5 min minimum) | Low | Developer skips it | Accept 5-min local gate; it's less than a coffee break |
| Baseline scores become stale as the system improves | Medium | Tests fail after good improvements | Process: after validated improvement, update BASELINE_SCORES and re-register golden recordings |
| False positives from Unity frame timing variance | Medium | Gate rejects good changes | Use ±2.0 point tolerance (already in place); increase to ±3.0 if variance is high |

### Failure Mode Management — Step 2

| Failure Mode | Detection | Recovery |
|---|---|---|
| Golden recording file missing or corrupted | `test_golden_recording_exists` fails | Re-run validation on that track, register new golden |
| Score regression is real (not false positive) | Score delta > 2.0 across 3 re-runs | Revert the change; investigate root cause |
| Score regression is false positive (variance) | Score delta > 2.0 on first run but passes on re-run | Document variance; consider increasing SCORE_TOLERANCE for that track |
| Config hash mismatch after intentional change | `test_config_hash_stable` fails | Run full live gate, update EXPECTED_CONFIG_HASH if all tracks pass |
| New track added but not in TRACK_CONFIG_MAP | `test_track_config_map_complete` fails | Add entry to TRACK_CONFIG_MAP |

---

## Step 3 — Grade and Banking

### Goal

Add elevation effects (hills, banked curves) to the simulation and control stack. Validate that PP and MPC handle grade-induced forces correctly.

### What We Have Today

- Track YAML schema already supports `grade` field on segments (rise/run slope, e.g., 0.05 = 5% grade)
- TrackBuilder.cs already generates elevation from grade: `y = startY + grade * distance` (line 182 for straights, 214 for arcs)
- Vehicle physics in Unity already respond to elevation (gravity, wheel forces)
- **BUT:** Vehicle state does NOT export pitch/roll/grade to Python. HDF5 has no elevation fields.
- **MPC dynamics model ignores grade** — assumes flat road: `a_eff = a` instead of `a_eff = a - g·sin(pitch)`

### Architecture

```
                    CURRENT                              AFTER STEP 3
                    ───────                              ────────────
Track YAML:         grade: 0.0 everywhere               grade: 0.0 to ±0.08
Unity mesh:         Flat road                            Elevation-correct road
Vehicle state:      No pitch/roll/grade                  + pitch_rad, roll_rad, road_grade
HDF5 recording:     No elevation fields                  + pitch_rad, roll_rad, road_grade
MPC dynamics:       a_eff = a                            a_eff = a - g·sin(pitch)
PP dynamics:        No grade compensation                No change (PP is geometric, not dynamic)
Speed governor:     Flat-road decel assumptions           Grade-adjusted decel limits
Analyzer:           No grade metrics                     + grade impact section
```

### Implementation Sequence

#### Phase 3A: Unity — Export Pitch/Roll/Grade to Vehicle State

**File:** `unity/AVSimulation/Assets/Scripts/CarController.cs`

Add to VehicleState class (around line 1686):
```csharp
// Grade and orientation (Step 3)
public float pitchRad;        // vehicle pitch angle in radians (positive = nose up)
public float rollRad;         // vehicle roll angle in radians (positive = right lean)
public float roadGrade;       // local road grade (rise/run) from track profile
```

**In the state update method** (where VehicleState is populated each frame):
```csharp
// Extract pitch and roll from vehicle transform rotation
Vector3 euler = transform.eulerAngles;
// Unity euler: x=pitch, y=yaw, z=roll (degrees, 0-360 range)
// Convert to signed radians: pitch positive = nose up
state.pitchRad = Mathf.DeltaAngle(0, euler.x) * Mathf.Deg2Rad;
state.rollRad  = Mathf.DeltaAngle(0, euler.z) * Mathf.Deg2Rad;
```

**Road grade from TrackPath:**
- `TrackPath.cs` needs a method `float GetGradeAtDistance(float distance)` that returns the grade of the segment containing that distance
- Call it with the vehicle's current track distance (already computed for ground truth)

**File:** `unity/AVSimulation/Assets/Scripts/TrackPath.cs` — add:
```csharp
public float GetGradeAtDistance(float distance) {
    // Walk segments to find which one contains 'distance'
    // Return that segment's grade value (default 0.0)
}
```

**Risks:**
- **Euler angle gimbal lock:** At extreme pitch (>80°), euler extraction is unreliable. But vehicle pitch on roads is <10° (8% grade = 4.6°), so this is not a real risk.
- **Unity euler convention:** Unity uses left-handed Y-up. `euler.x` is pitch (nose down is positive in Unity convention). **Must verify sign convention** by placing vehicle on a known 5% uphill grade and checking that `pitchRad > 0`.
- **TrackPath segment lookup:** If track is a loop and distance wraps around, the lookup must handle modular distance. Existing `GetPositionAtDistance` already handles this — reuse the same segment iteration logic.

#### Phase 3B: Python — Receive Grade Data

**File:** `data/formats/data_format.py` — add to VehicleState dataclass:
```python
pitch_rad: float = 0.0         # vehicle pitch (positive = nose up)
roll_rad: float = 0.0          # vehicle roll (positive = right lean)
road_grade: float = 0.0        # local road grade (rise/run from track)
```

**File:** `bridge/server.py` — parse new fields from Unity JSON:
```python
# In the vehicle state parsing section:
pitch_rad = data.get("pitchRad", 0.0)
roll_rad = data.get("rollRad", 0.0)
road_grade = data.get("roadGrade", 0.0)
```

**File:** `data/recorder.py` — register 3 new HDF5 fields. **Remember the 6-location rule:**

1. `create_dataset` — add `pitch_rad`, `roll_rad`, `road_grade` (dtype float32)
2. List init — add to the per-frame list initialization
3. Per-frame append — append from vehicle state
4. Resize loop — include in the resize key set
5. Write — include in the write loop
6. Orchestrator constructor — include in ControlCommandData construction

**File:** `av_stack/orchestrator.py` — pass grade data through to controller:
```python
# In _pf_compute_steering or wherever control is called:
# Pass road_grade to the controller for MPC grade compensation
```

#### Phase 3C: MPC Grade Compensation

**File:** `control/mpc_controller.py`

The kinematic bicycle model currently has:
```
v[k+1] = v[k] + a[k] * dt
```

With grade compensation:
```
v[k+1] = v[k] + (a[k] - g * sin(pitch)) * dt
```

Where `g = 9.81 m/s²` and `pitch` is the current vehicle pitch in radians.

**Implementation:**
1. Add `grade_rad: float = 0.0` to `MPCController.compute()` signature
2. In `MPCSolver._build_qp()`, modify the velocity dynamics row:
   ```python
   # Current: A_v_row: v[k+1] = v[k] + a[k]*dt
   # New:     A_v_row: v[k+1] = v[k] + a[k]*dt - g*sin(pitch)*dt
   # The -g*sin(pitch)*dt term is a constant offset (pitch doesn't change within horizon)
   # Add it to the linear term (q vector), not the quadratic term (P matrix)
   grade_accel = -9.81 * math.sin(grade_rad)  # negative on uphill (reduces effective accel)
   # Add grade_accel * dt to the velocity prediction at each step
   ```
3. **Important:** Grade is approximately constant over the MPC horizon (20 steps × 0.05s = 1s, vehicle travels ~15m). The grade over 15m of road changes negligibly. Therefore, use the **current** grade for all horizon steps — do NOT try to predict future grade.

**Risks:**
- **Small-angle assumption:** For grades up to 8% (4.6°), `sin(pitch) ≈ pitch` with <0.1% error. Safe to use `sin()` for correctness, but the approximation works too.
- **Grade sign convention mismatch:** If Unity reports pitch with opposite sign convention, MPC will accelerate on uphills instead of decelerating. **Must verify with a known-grade track before any tuning.**
- **QP matrix sparsity change:** Adding the grade offset to the linear term changes `q` but not `P`. OSQP should handle this without issue (it's just a constant shift in the objective).

#### Phase 3D: Speed Governor Grade Adjustment

**File:** `trajectory/speed_planner.py` (or wherever curve cap deceleration is computed)

On a downhill, the vehicle has more kinetic energy entering a curve. The speed governor's deceleration assumptions need adjustment:

```python
# Current: effective_decel = cap_tracking_decel_gain * speed_error
# With grade: effective_decel = cap_tracking_decel_gain * speed_error - g * sin(grade)
# On downhill (grade < 0): sin(grade) < 0, so -g*sin < 0 → reduces effective decel
# On uphill (grade > 0): sin(grade) > 0, so -g*sin > 0 → increases effective decel
```

This is a safety-critical change: on a downhill approach to a curve, the vehicle decelerates slower than expected. The speed governor must account for this.

**Implementation:**
1. Pass `road_grade` to the speed planner's `step()` method
2. Adjust the effective deceleration: `effective_decel = planned_decel + g * sin(grade)`
3. If on a steep downhill and deceleration is insufficient, increase brake authority

**Risks:**
- **Grade data arrives late.** The speed governor plans based on current grade, but the curve may be on a different grade. This is a second-order effect — for now, using current grade is sufficient.
- **Oscillation in grade-aware decel on rolling terrain.** If grade changes sign rapidly (crests/dips), the decel adjustment may oscillate. Add a 5-frame EMA on road_grade before using it.

#### Phase 3E: Create Graded Track Profiles

**File:** `tracks/hill_highway.yml` — Highway with grade

```yaml
name: hill_highway
road_width: 7.2
lane_line_width: 0.2
sample_spacing: 1.0
loop: true
start_distance: 3.0
speed_limit_mph: 55.0
segments:
  # Flat start
  - type: straight
    length: 200
    speed_limit_mph: 55.0
    grade: 0.0
  # Uphill 5%
  - type: straight
    length: 300
    speed_limit_mph: 55.0
    grade: 0.05
  # Crest — gentle curve at top of hill
  - type: arc
    radius: 300
    angle_deg: 45
    direction: left
    speed_limit_mph: 45.0
    grade: 0.0
  # Downhill 5% into curve — the dangerous scenario
  - type: straight
    length: 200
    speed_limit_mph: 55.0
    grade: -0.05
  # Curve at bottom of downhill
  - type: arc
    radius: 200
    angle_deg: 45
    direction: right
    speed_limit_mph: 40.0
    grade: -0.02
  # Flat recovery
  - type: straight
    length: 300
    speed_limit_mph: 55.0
    grade: 0.0
  # Close the loop with compensating arcs
  - type: arc
    radius: 400
    angle_deg: 90
    direction: left
    speed_limit_mph: 50.0
    grade: 0.0
  # Return straight (must compensate uphill grade with downhill to return to start elevation)
  - type: straight
    length: 300
    speed_limit_mph: 55.0
    grade: -0.033  # net elevation change must be zero for a loop
  - type: arc
    radius: 400
    angle_deg: 80
    direction: left
    speed_limit_mph: 50.0
    grade: 0.0
```

**File:** `tracks/banked_oval.yml` — Oval with banking (future extension if Unity supports road banking)

**Note on loop elevation:** For a looping track, the total elevation change must be zero. Sum of `grade × segment_length` across all segments must equal 0. **The implementing agent must verify this constraint.**

**Elevation balance formula:**
```
Σ (grade_i × length_i) = 0  (for loop tracks)
```
Where `length_i` is the physical length of each segment (straight: `length` field; arc: `radius × angle_deg × π/180`).

#### Phase 3F: Tests

**New file:** `tests/test_grade_compensation.py`

```python
"""Tests for grade compensation in MPC and speed governor."""

class TestMPCGradeCompensation:
    def test_uphill_reduces_effective_acceleration(self):
        """On 5% uphill, MPC should command more throttle for same speed."""
        # Run MPC solver with grade=0.05 vs grade=0.0
        # Verify acceleration command is higher with grade

    def test_downhill_increases_effective_deceleration(self):
        """On 5% downhill, MPC should command less throttle / more brake."""

    def test_flat_grade_unchanged(self):
        """grade=0.0 produces identical results to no-grade baseline."""

    def test_grade_does_not_affect_lateral_dynamics(self):
        """Grade compensation only affects longitudinal (velocity) dynamics."""
        # Same lateral error, heading error with grade=0 vs grade=0.05

    def test_extreme_grade_clamped(self):
        """Grade > 15% should be clamped to prevent unrealistic MPC commands."""

class TestSpeedGovernorGrade:
    def test_downhill_decel_is_weaker(self):
        """On downhill, effective deceleration is reduced (gravity assists speed)."""

    def test_uphill_decel_is_stronger(self):
        """On uphill, effective deceleration is increased (gravity assists braking)."""

    def test_grade_smoothing_prevents_oscillation(self):
        """Rapid grade changes are EMA-smoothed before use."""

class TestGradeHDF5Recording:
    def test_pitch_roll_grade_recorded(self):
        """pitch_rad, roll_rad, road_grade appear in HDF5 recording."""

    def test_grade_zero_on_flat_track(self):
        """On s_loop (flat), all grade fields are ~0.0."""

class TestGradedTrackGeometry:
    def test_hill_highway_loop_elevation_closes(self):
        """Total elevation change on hill_highway is ~0.0 (loop constraint)."""
        # Sum grade_i * length_i for all segments
        # Should be within ±0.1m of zero

    def test_all_grades_within_safe_range(self):
        """No segment has |grade| > 0.10 (10%)."""
```

**Update existing tests:**
- `tests/test_mpc_controller.py` — add `grade_rad` parameter to existing MPC solve tests (verify backward compatibility with grade=0.0)
- `tests/test_track_coverage.py` — add hill_highway to parametrized track list

#### Phase 3G: Analysis Pipeline Updates

**File:** `tools/drive_summary_core.py` — add Grade Impact section:

```python
# New section in analyze_recording_summary():
def _compute_grade_metrics(self, h5):
    """Compute grade-related metrics from HDF5 recording."""
    pitch = h5["vehicle/pitch_rad"][:]
    grade = h5["vehicle/road_grade"][:]

    return {
        "grade_max": float(np.max(np.abs(grade))),
        "grade_mean": float(np.mean(np.abs(grade))),
        "pitch_p95": float(np.percentile(np.abs(pitch), 95)),
        "speed_on_downhill_p95": ...,  # speed when grade < -0.02
        "decel_on_downhill_p95": ...,  # deceleration when grade < -0.02
        "overspeed_on_downhill_rate": ...,  # % of downhill frames above speed limit
    }
```

**File:** `tools/analyze/analyze_drive_overall.py` — add section 15: GRADE IMPACT

```
15. GRADE IMPACT
--------------------------------------------------------------------------------
   Max Grade: 5.0%
   Max Pitch: 0.050 rad (2.9°)
   Speed on Downhill (P95): 14.2 m/s
   Decel on Downhill (P95): 1.8 m/s²
   Overspeed on Downhill: 0.0%
   Grade Compensation Active: 85.3% (of graded frames)
```

#### Phase 3H: PhilViz Changes

**File:** `tools/debug_visualizer/backend/diagnostics.py` — add grade sub-panel:

```python
def compute_grade_diagnostics(h5_path):
    """Compute per-frame grade diagnostics for PhilViz."""
    return {
        "pitch_rad": [...],      # per-frame pitch
        "roll_rad": [...],       # per-frame roll
        "road_grade": [...],     # per-frame grade
        "grade_accel_offset": [...],  # -g*sin(pitch) per frame
    }
```

**File:** `tools/debug_visualizer/visualizer.js` — add:
1. Grade trace in the existing "Vehicle State" chart (overlay pitch/roll on existing position chart)
2. Grade compensation trace in the MPC diagnostics panel (show `grade_accel_offset` alongside `mpc_accel_cmd`)

**File:** `tools/debug_visualizer/server.py` — add endpoint:
```python
@app.route("/api/grade-diagnostics/<recording_id>")
def grade_diagnostics(recording_id):
    """Return grade diagnostics for a recording."""
```

**IMPORTANT:** After editing any PhilViz file, restart the server:
```bash
pkill -f "debug_visualizer/server.py" 2>/dev/null; sleep 1
source venv/bin/activate && PYTHONPATH=/Users/philiptullai/Documents/Coding/av \
  python tools/debug_visualizer/server.py > /tmp/philviz.log 2>&1 &
```

### Acceptance Criteria — Step 3

| Criteria | Gate |
|---|---|
| Unity exports pitch_rad, roll_rad, road_grade in vehicle state | Verify with flat track (all ~0.0) and graded track (non-zero) |
| HDF5 records all 3 new fields correctly | `test_grade_hdf5_recording` green |
| MPC grade compensation: uphill commands more throttle | `test_uphill_reduces_effective_acceleration` green |
| MPC grade compensation: flat track identical to no-grade | `test_flat_grade_unchanged` green |
| Speed governor adjusts decel for downhill | `test_downhill_decel_is_weaker` green |
| hill_highway track completes 3× 60s with 0 e-stops | Live validation |
| hill_highway score ≥ 85/100 | `analyze_drive_overall.py` |
| Existing flat tracks (s_loop, highway) not regressed | Grade=0.0 path is no-op; run comfort gates |
| PhilViz shows grade traces | Manual inspection |
| Grade section in analyze_drive_overall.py output | Manual inspection |
| Graded track elevation closes (loop constraint) | `test_hill_highway_loop_elevation_closes` green |

### Risks Summary — Step 3

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Unity pitch sign convention wrong | High | MPC accelerates on uphill | Test with known 5% grade track FIRST. Log pitch_rad and verify sign before writing any compensation code. |
| Grade compensation destabilizes MPC on flat tracks | Low | Regression on s_loop/highway | Grade=0.0 → sin(0)=0 → zero offset. Mathematically no-op. Verify with tests. |
| Downhill overspeed into curve | Medium | E-stop or OOL event | Speed governor grade adjustment is the fix; test with 5% downhill → R200 curve scenario explicitly |
| Loop elevation doesn't close | High | Vehicle drifts vertically over laps | Compute elevation sum before first run. Adjust last segment grade to close. |
| TrackPath.GetGradeAtDistance has off-by-one on loop wrap | Medium | Grade jumps to wrong value at loop boundary | Test with vehicle crossing loop boundary, verify grade continuity |
| Banking (road tilt perpendicular to travel) not yet supported | Low | Feature request | Explicitly out of scope for Step 3. Document as future enhancement. Banking requires road mesh normal rotation in TrackBuilder, which is more complex. |

### Failure Mode Management — Step 3

| Failure Mode | Detection | Recovery |
|---|---|---|
| Pitch_rad is always 0.0 even on graded track | Check HDF5 `vehicle/pitch_rad` — all zeros | Unity not exporting field. Check CarController state update code. Check bridge JSON serialization. |
| MPC diverges on graded track | E-stop within first 30s; `mpc_feasibility_rate < 99%` | Grade offset too large. Clamp `grade_accel` to ±2.0 m/s². Check sign. |
| Speed governor oscillates on rolling terrain | `commanded_jerk_p95 > 6.0 m/s³` on graded sections | Increase grade EMA smoothing (5→15 frames). Or use lookahead grade (grade at curve entry, not current). |
| Vehicle falls through road on graded sections | Chassis-ground penetration > 0 in HDF5 | TrackBuilder mesh has gaps at grade transitions. Check mesh generation for elevation continuity between segments. |
| Existing flat-track golden recordings fail after code changes | Comfort gate regression test fails | Grade compensation must be a strict no-op at grade=0.0. If it isn't, there's a code bug — don't tune around it. |
| Performance regression from 3 extra HDF5 fields | dt_p95 increases | 3 float32 fields add ~12 bytes/frame. Negligible. If it shows up, it's coincidental with something else. |

---

## Cross-Step Dependencies

```
Step 1 (Tracks)
  │
  ├─► Golden recordings for 5 tracks
  │
  └─► Step 2 (CI)
        │
        ├─► Automated regression catches grade-related regressions
        │
        └─► Step 3 (Grade)
              │
              └─► Graded track added to golden set
```

## Doc Updates Required After Each Step

After each step completes, update:
1. `docs/agent/current_state.md` — new milestone status
2. `docs/agent/tasks.md` — mark step complete, update next focus
3. `docs/ROADMAP.md` — update Status Snapshot and Capability Roadmap step status
4. `tests/fixtures/golden_recordings.json` — new golden recordings
5. `tests/conftest.py` — new BASELINE_SCORES entries
6. Memory files if any new patterns/conventions discovered
