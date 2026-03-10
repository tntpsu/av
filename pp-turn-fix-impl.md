# PP Turn-Entry Fix — Implementation Spec

**Status:** Ready for implementation
**Goal:** Reduce C1/C3 peak lateral error and PP floor rescue magnitude on s_loop
**Current:** C1 peak 0.67m, C3 peak 0.65m, floor rescue 1.7m on both
**Target:** C1/C3 peak < 0.50m, floor rescue < 1.0m

---

## Root Cause Summary

Traced from data → code → specific lines:

### Problem 1: COMMIT state persists across straights (C1 priority)

**Data evidence:** `curve_intent_state ~= COMMIT for 1132/1142 frames` (99%)

**Code path:**

1. `trajectory/utils.py` line 1040: `local_sustain_phase_raw = max(path, preview, time) * confidence`
2. On s_loop, `term_preview` is ALWAYS ~1.0 because the next curve is always visible within the preview window
3. So `local_sustain_phase_raw` never drops below ~0.6 even on straights
4. `trajectory/utils.py` line 1054: EMA filter `phase = 0.45 * sustain + 0.55 * previous` means phase decays very slowly
5. `trajectory/utils.py` line 1137: COMMIT floor `phase = max(phase, 0.30)` keeps phase above the OFF threshold (0.30)
6. `trajectory/utils.py` line 1096: COMMIT→REARM requires `phase < off (0.30)` — but the floor prevents this!

**Result:** Once COMMIT is entered, it can never exit on s_loop because the sustain term (driven by far preview) + COMMIT floor (0.30) ≥ OFF threshold (0.30). The state machine is trapped.

### Problem 2: Lookahead collapses too late and too fast (C2 priority)

**Data evidence:** Turn 1 lookahead `~6.6m → 3.0m`, turn 3 `~6.45m → 3.0m`. Floor rescue 1.7m.

**Code path:**

1. `trajectory/utils.py` line 1573-1602: `compute_reference_lookahead` blends straight/entry tables using `entry_weight = curve_local_phase`
2. Because phase is already ~0.5-0.6 on the "straight" (Problem 1), lookahead is ALREADY partially contracted before the real curve starts
3. When the real curve arrives, the remaining contraction happens in a few frames
4. `control/pid_controller.py` line 1267: `pp_curve_local_shorten_slew_m_per_frame = 0.10` limits rate to 0.10m/frame
5. But the TARGET lookahead drops to ~3.0m (commit table at 5 m/s ≈ 4.6m, minus entry severity scale 0.80 = 3.7m, further reduced by tight_curve_scale)
6. `control/pid_controller.py` line 1260: Floor catches it at 4.4-4.8m → rescue delta = 1.7m

**Result:** The lookahead contraction is "used up" by the always-on COMMIT state, so when the real curve comes, the remaining travel is a sudden drop that the floor must catch.

---

## Fix Design

Two changes, each independently testable. Do them in order.

### Fix 1: Gate COMMIT sustain on local relevance (trajectory/utils.py)

**What:** COMMIT can only be SUSTAINED when the vehicle has local relevance to the current curve (distance_ready OR time_ready OR in_curve_now). When local relevance is lost (exited curve, on straight), COMMIT transitions to REARM even if far preview is high.

**Why this works:** On s_loop, between curves the vehicle has ~5-10m of straight. `distance_ready` requires ≤ 3.0m to next curve, `time_ready` requires ≤ 0.60s. On the straight, neither is true. So COMMIT will drop to REARM, then to STRAIGHT, allowing the phase to decay and the lookahead to reset before the next curve.

**File:** `trajectory/utils.py`

**Change 1a:** In `compute_curve_phase_scheduler`, around line 1040-1054, gate the sustain term.

Currently (line ~1040):
```python
local_sustain_phase_raw = max(term_path, term_preview, term_time) * confidence_scale
```

Change to:
```python
# Sustain requires local relevance. Far preview alone cannot sustain phase.
# This prevents COMMIT from persisting across straights on looped tracks.
if local_commit_ready:
    # Full sustain when locally committed (distance_ready OR time_ready OR in_curve)
    local_sustain_phase_raw = max(term_path, term_preview, term_time) * confidence_scale
else:
    # Only path curvature can sustain when not locally committed.
    # Preview and time terms are suppressed — they represent FAR awareness, not LOCAL state.
    local_sustain_phase_raw = term_path * confidence_scale
```

**Why `term_path` stays:** When the vehicle is physically on a curve (path curvature is high), it should sustain even without distance_ready. `term_path` is computed from current lane curvature, not preview — it's genuinely local.

**Change 1b:** In the COMMIT transition logic (lines ~1096-1110), add a local-relevance exit.

Currently (line ~1096):
```python
# COMMIT state
if phase < off:
    new_state = "REARM"
```

Add an additional exit condition:
```python
# COMMIT state
if phase < off:
    new_state = "REARM"
elif not local_commit_ready and not local_path_sustain_active:
    # Lost local relevance — curve is behind us, drop to REARM.
    # Far preview is NOT sufficient to hold COMMIT.
    new_state = "REARM"
```

Where `local_path_sustain_active = term_path >= 0.3` (path curvature is genuinely present at the vehicle). This variable is already computed and returned in the dict (line ~1200: `curve_local_path_sustain_active`).

**Expected effect:**
- COMMIT duration drops from 99% of frames to ~40-60% (only during actual curves)
- `curve_local_active_on_straights` drops from 32% to < 10%
- Phase decays on straights, allowing lookahead to reset to full value before next curve entry

**Test (tests/test_reference_lookahead.py):**

```python
def test_commit_does_not_persist_across_straight():
    """COMMIT must exit when local relevance is lost on a straight."""
    config = make_default_config()

    # Drive through a curve (phase rises, enters COMMIT)
    state = {"curve_phase": 0.0, "curve_phase_state": "STRAIGHT",
             "curve_phase_entry_frames": 0, "curve_phase_rearm_hold_frames": 0}

    # 20 frames of curve approach + in-curve (distance_ready=True, high curvature)
    for i in range(20):
        result = compute_curve_phase_scheduler(
            preview_curvature_abs=0.025,
            path_curvature_abs=0.020,
            curvature_rise_abs=0.002,
            distance_to_curve_start_m=max(0.1, 3.0 - i * 0.15),
            time_to_curve_s=max(0.01, 0.6 - i * 0.03),
            previous_phase=state["curve_phase"],
            previous_state=state["curve_phase_state"],
            previous_entry_frames=state["curve_phase_entry_frames"],
            previous_rearm_hold_frames=state["curve_phase_rearm_hold_frames"],
            config=config,
        )
        state = {k: result[k] for k in state}

    assert result["curve_local_state"] == "COMMIT", "Should be in COMMIT during curve"

    # Now exit curve: straight segment with FAR preview still active
    for i in range(30):
        result = compute_curve_phase_scheduler(
            preview_curvature_abs=0.025,      # Next curve visible (far preview)
            path_curvature_abs=0.001,          # Straight road (low path curvature)
            curvature_rise_abs=0.0,
            distance_to_curve_start_m=8.0 + i * 0.5,  # Next curve is far
            time_to_curve_s=1.5 + i * 0.1,            # Next curve is far in time
            previous_phase=state["curve_phase"],
            previous_state=state["curve_phase_state"],
            previous_entry_frames=state["curve_phase_entry_frames"],
            previous_rearm_hold_frames=state["curve_phase_rearm_hold_frames"],
            config=config,
        )
        state = {k: result[k] for k in state}

    # After 30 frames of straight, COMMIT must NOT persist
    assert result["curve_local_state"] in ("STRAIGHT", "REARM"), \
        f"COMMIT persisted on straight: state={result['curve_local_state']}, " \
        f"phase={result['curve_local_phase']:.3f}"
```

```python
def test_commit_sustains_when_in_curve():
    """COMMIT must sustain when vehicle is physically on a curve (high path curvature)."""
    config = make_default_config()

    # Same setup: reach COMMIT
    state = _reach_commit_state(config)  # helper that runs 20 frames of curve approach

    # Continue IN the curve: distance is past curve start, but path curvature is high
    for i in range(20):
        result = compute_curve_phase_scheduler(
            preview_curvature_abs=0.025,
            path_curvature_abs=0.020,      # HIGH — vehicle is on the curve
            curvature_rise_abs=0.0,
            distance_to_curve_start_m=0.0,  # Past curve start
            time_to_curve_s=0.0,
            previous_phase=state["curve_phase"],
            previous_state=state["curve_phase_state"],
            previous_entry_frames=state["curve_phase_entry_frames"],
            previous_rearm_hold_frames=state["curve_phase_rearm_hold_frames"],
            config=config,
        )
        state = {k: result[k] for k in state}

    assert result["curve_local_state"] == "COMMIT", \
        "COMMIT should sustain while path curvature is high (vehicle on curve)"
```

---

### Fix 2: Start lookahead contraction earlier with a gentler slope (trajectory/utils.py + config)

**What:** Increase the distance/time windows so ENTRY arms earlier, giving more frames for the lookahead to contract smoothly before the curve arrives. Combined with Fix 1 (phase resets on straights), this means the lookahead starts from a FULL value and has more runway to reach the target.

**Why this works:** Currently, ENTRY arms at ~6.5m from curve start and the phase ramps up over ~10 frames via EMA. With Fix 1, the phase starts near 0 on straights (not 0.5). Starting contraction at ~10m instead of ~8m gives more frames to blend from straight→entry lookahead. The floor backstop intervenes less because the contraction is gradual.

**File:** `config/mpc_sloop.yaml`

**Change 2a:** Widen the local phase distance/time windows.

```yaml
# BEFORE:
curve_local_phase_distance_start_m: 8.0
curve_local_phase_distance_start_tight_m: 10.0
curve_local_phase_time_start_s: 1.2
curve_local_phase_time_start_tight_s: 1.6

# AFTER (start detecting earlier):
curve_local_phase_distance_start_m: 12.0
curve_local_phase_distance_start_tight_m: 14.0
curve_local_phase_time_start_s: 2.0
curve_local_phase_time_start_tight_s: 2.5
```

**Change 2b:** Raise the entry lookahead table so the entry→commit drop is smaller.

```yaml
# BEFORE — entry table drops too aggressively:
reference_lookahead_speed_table_entry:
  - speed_mps: 0.0
    lookahead_m: 6.0
  - speed_mps: 4.47
    lookahead_m: 7.0

# AFTER — entry lookahead stays closer to straight, less rescue needed:
reference_lookahead_speed_table_entry:
  - speed_mps: 0.0
    lookahead_m: 7.0
  - speed_mps: 4.47
    lookahead_m: 8.0
```

**Change 2c:** Increase the PP shorten-slew to allow slightly faster contraction (since we're starting earlier, we can afford a gentler slope AND arrive at the target in time).

```yaml
# BEFORE:
pp_curve_local_shorten_slew_m_per_frame: 0.10

# AFTER (slightly faster, but starting from higher baseline):
pp_curve_local_shorten_slew_m_per_frame: 0.15
```

**Rationale for numbers:**
- At 5 m/s, 12.0m distance = 2.4s lead time = ~72 frames at 30 Hz
- Contraction from 9.0m → 4.6m = 4.4m over 72 frames = 0.061 m/frame (well within 0.15 slew)
- Even at 7 m/s: 12.0m / 7.0 = 1.7s = ~51 frames, 4.4m / 51 = 0.086 m/frame (still within slew)
- Floor rescue: if contraction is gradual, floor only needs to catch ~0.5m instead of 1.7m

**Test (tests/test_reference_lookahead.py):**

```python
def test_lookahead_contraction_is_gradual():
    """Lookahead should contract smoothly over entry, not in a sudden drop."""
    config = make_default_config()
    config.update({
        "curve_local_phase_distance_start_m": 12.0,
        "pp_curve_local_shorten_slew_m_per_frame": 0.15,
    })

    lookaheads = []
    for distance in np.linspace(15.0, 0.0, 60):  # 60 frames of approach
        result = compute_reference_lookahead(
            base_lookahead=9.0,
            current_speed=5.5,
            path_curvature=0.0 if distance > 2.0 else 0.02,
            config=config,
            distance_to_curve_start_m=max(0, distance),
            curve_scheduler_mode="phase_active",
            curve_phase=min(1.0, max(0, (12.0 - distance) / 12.0)),  # linear ramp
            curve_local_state="ENTRY" if distance > 1.0 else "COMMIT",
            return_diagnostics=True,
        )
        lookaheads.append(result["lookahead"])

    # Check no frame-to-frame drop exceeds 0.3m (generous — 2x slew)
    deltas = [lookaheads[i] - lookaheads[i+1] for i in range(len(lookaheads)-1)]
    max_drop = max(deltas)
    assert max_drop < 0.3, f"Lookahead dropped {max_drop:.3f}m in one frame (limit 0.3m)"

    # Check contraction starts well before curve arrival
    first_contract_frame = next(i for i, d in enumerate(deltas) if d > 0.01)
    assert first_contract_frame < 20, \
        f"Contraction started at frame {first_contract_frame} (should be < 20)"
```

---

## Implementation Order

### Step 1: Fix 1a + 1b (gate sustain + COMMIT exit)

**File:** `trajectory/utils.py`

**Lines to modify:**
- ~1040: Change `local_sustain_phase_raw` computation
- ~1096-1110: Add COMMIT→REARM exit on lost local relevance

**Lines to read first:**
- 881-1228: Full `compute_curve_phase_scheduler` function
- 693-814: `_compute_local_curve_relevance_terms` and `_compute_local_curve_relevance_weight`

**Testing:**
```bash
pytest tests/test_reference_lookahead.py -v -k "commit"
```

**Verify no regression:**
```bash
pytest tests/ -v --timeout=30
```

### Step 2: Fix 2a + 2b + 2c (config tuning)

**File:** `config/mpc_sloop.yaml`

**Lines to modify:**
- Curve local phase distance/time windows (~line 370-385)
- Reference lookahead speed table entry (~line 395-410)
- PP shorten slew (~line 50)

**Testing:** Same test suite. Then live validation:
```bash
./start_av_stack.sh --build-unity-player --skip-unity-build-if-clean \
  --config config/mpc_sloop.yaml --track-yaml tracks/s_loop.yml --duration 60
python tools/analyze/analyze_drive_overall.py --latest
```

### Step 3: Validate

Run 3× s_loop, analyze each. Check:

| Metric | Current | Target |
|---|---|---|
| C1 peak lateral error | 0.67m | < 0.50m |
| C3 peak lateral error | 0.65m | < 0.50m |
| Floor rescue max C1 | 1.7m | < 1.0m |
| Floor rescue max C3 | 1.7m | < 1.0m |
| curve_local_active_on_straights | 32% | < 10% |
| Overall score | 95.2 | ≥ 94.0 (no regression) |
| E-stops | 0 | 0 |

### Step 4: Highway non-regression

```bash
./start_av_stack.sh --build-unity-player --skip-unity-build-if-clean \
  --config config/mpc_highway.yaml --track-yaml tracks/highway_65.yml --duration 120
```

2× runs, check no regression in MPC score (≥ 95).

---

## Files Modified (complete list)

| File | Change | Lines |
|---|---|---|
| `trajectory/utils.py` | Gate sustain on local relevance | ~1040 |
| `trajectory/utils.py` | COMMIT→REARM on lost local relevance | ~1096-1110 |
| `config/mpc_sloop.yaml` | Widen distance/time windows | ~370-385 |
| `config/mpc_sloop.yaml` | Raise entry lookahead table | ~395-410 |
| `config/mpc_sloop.yaml` | Increase shorten slew | ~50 |
| `tests/test_reference_lookahead.py` | 3 new tests | New |

**Files NOT modified:**
- `control/pid_controller.py` — PP floor logic is correct, just over-triggered. Fix the upstream cause.
- `av_stack/orchestrator.py` — Passes data through, no logic change needed.
- `data/recorder.py` — No new telemetry fields needed (existing `curve_local_*` fields cover it).

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Phase drops too fast on short straights → late re-arm for next curve | `term_path` sustain keeps phase up while physically on curve. REARM→ENTRY transition re-arms quickly when distance gate fires. |
| Widened detection windows false-trigger on highway | Highway curves are gentle (κ≈0.003). The `term_preview` normalization maps 0.003→0.015 so highway barely registers. Also highway uses MPC, not PP. |
| Raised entry table reduces PP's ability to track tight curves | The entry table is an intermediate blend target, not the final commit value. COMMIT table (4.0-4.6m) is unchanged. The floor (4.4-4.8m) is unchanged. |
| EMA alpha too slow to re-arm between tight curves on s_loop | At α=0.45, phase goes 0.5→0.25 in ~5 frames, 0.25→0.12 in ~5 more. That's 10 frames ≈ 0.33s. At 5 m/s over 5m straight, vehicle has ~1s. Plenty of time. |

---

## What NOT to Change

1. **Do not change the COMMIT floor value (0.30).** The problem isn't the floor — it's that COMMIT never exits. With the exit fix, the floor becomes irrelevant on straights.
2. **Do not change the EMA alpha (0.45).** It's correct for smooth transitions. The problem is the sustain input, not the filter.
3. **Do not touch the speed governor or cap-tracking.** No speed-feasibility issue exists.
4. **Do not change PP geometric formula or feedback gain.** PP tracks fine when given correct lookahead.
5. **Do not add new telemetry fields.** Existing `curve_local_*` fields already capture everything needed.
