# Phase 2.9 — Stanley Low-Speed Regime + Phase 2.7 — NMPC

**Created:** 2026-03-13
**Status:** Plan complete, ready for implementation
**Prerequisite:** Phase 2.8.6 complete (mixed_radius validated)

---

## Motivation

The current system has 2 active lateral control regimes:
- **Pure Pursuit (PP):** 0–9.5 m/s — geometric lookahead controller
- **Linear MPC:** 11+ m/s — kinematic bicycle model, OSQP solver

Two gaps remain:
1. **Low-speed tight curves (hairpin, R15):** PP's chord approximation causes 0.9m peak errors. Stanley controller (already implemented but gated off) tracks the nearest path point directly — no lookahead, no floor, no chord error. Score: 79/100 with gate at 75.
2. **High-speed large heading error:** Linear MPC uses small-angle sin(x)≈x approximation which breaks down when |e_heading| > 0.25 rad or speed > 20 m/s. Nonlinear MPC with CasADi removes this constraint.

**Target regime map after both phases:**
```
0 ─── 5 ─── 10 ──── 15 ──── 20 ──── 25+ m/s
│Stan│blend│ PP  │blend│ LMPC │blend│ NMPC │
```

**Implementation order:** Phase 2.9 first (Stanley is simpler, validates the multi-regime plumbing), then Phase 2.7 (NMPC reuses the same plumbing).

---

# PHASE 2.9 — Stanley Low-Speed Regime

## Architecture

**Regime encoding (backward compatible):**
```python
class ControlRegime(IntEnum):
    STANLEY = -1           # NEW: low-speed precise tracking
    PURE_PURSUIT = 0       # unchanged
    LINEAR_MPC = 1         # unchanged
    NONLINEAR_MPC = 2      # unchanged (future Phase 2.7)
```

HDF5 stores regime as float. Existing recordings have 0.0 or 1.0. Stanley writes -1.0. Backward compatible: the existing mask `regime >= 0.5` for MPC detection still works. New Stanley mask: `regime < -0.5`.

**Stanley activation:** speed-based with hysteresis, same mechanism as PP→MPC:
- Downshift PP → Stanley when `speed < stanley_max_speed_mps - hysteresis` (default: 5.0 - 0.5 = 4.5 m/s)
- Upshift Stanley → PP when `speed > stanley_max_speed_mps + hysteresis` (default: 5.0 + 0.5 = 5.5 m/s)
- Blends over `blend_frames` (default 15 frames = 0.5s)

**Stanley formula (already in pid_controller.py:1428–1435):**
```python
steering = stanley_heading_weight * heading_error + arctan2(stanley_k * lateral_error, max(soft_speed, speed))
```
No lookahead distance. No floor. Tracks nearest path point using heading + cross-track error.

---

## Sub-phase 2.9.1 — Extend ControlRegime enum + RegimeConfig + RegimeSelector

**Files:** `control/regime_selector.py`

### Step 1: Add STANLEY to enum (line 23)
```python
class ControlRegime(IntEnum):
    STANLEY = -1               # Low-speed precise tracking (heading + crosstrack)
    PURE_PURSUIT = 0
    LINEAR_MPC = 1
    NONLINEAR_MPC = 2          # Future — Phase 2.7
```

### Step 2: Add config fields to RegimeConfig (line 30)
Add these fields after `enabled`:
```python
    stanley_max_speed_mps: float = 5.0         # Below this → Stanley
    stanley_upshift_hysteresis_mps: float = 0.5
    stanley_downshift_hysteresis_mps: float = 0.5
    stanley_blend_frames: int = 10             # Shorter blend (low speed = less risk)
    stanley_min_hold_frames: int = 15          # Shorter hold (low speed = faster transitions)
    stanley_enabled: bool = True               # Can disable Stanley independently of regime.enabled
```

### Step 3: Update from_config (line 82)
Add the new keys to the `for key in [...]` list:
```python
for key in ['enabled', 'pp_max_speed_mps', 'lmpc_max_speed_mps',
            'lmpc_max_heading_error_rad', 'upshift_hysteresis_mps',
            'downshift_hysteresis_mps', 'blend_frames', 'min_hold_frames',
            'stanley_max_speed_mps', 'stanley_upshift_hysteresis_mps',
            'stanley_downshift_hysteresis_mps', 'stanley_blend_frames',
            'stanley_min_hold_frames', 'stanley_enabled']:
```

### Step 4: Extend _compute_desired (line 147)
Current logic checks high-speed first (NMPC > LMPC > PP). Add Stanley check at the end:
```python
def _compute_desired(self, speed: float, heading_error: float) -> ControlRegime:
    # Existing high-speed logic (unchanged)
    pp_up = self.config.pp_max_speed_mps + self.config.upshift_hysteresis_mps
    pp_down = self.config.pp_max_speed_mps - self.config.downshift_hysteresis_mps
    # ... existing NMPC and LMPC checks ...

    # NEW: Stanley downshift check
    if self.config.stanley_enabled:
        stanley_down = self.config.stanley_max_speed_mps - self.config.stanley_downshift_hysteresis_mps
        stanley_up = self.config.stanley_max_speed_mps + self.config.stanley_upshift_hysteresis_mps

        if self._active_regime == ControlRegime.STANLEY:
            # Currently in Stanley — upshift to PP only if speed exceeds upper threshold
            if speed > stanley_up:
                return ControlRegime.PURE_PURSUIT
            return ControlRegime.STANLEY
        else:
            # Currently in PP (or higher) — downshift to Stanley if speed drops below lower threshold
            if speed < stanley_down:
                return ControlRegime.STANLEY

    # Default: return PP or the existing desired regime
    # ... (integrate with existing if/elif chain)
```

**Important:** The hold counter and blend logic already exist and work generically. Stanley transitions reuse them. Use `stanley_blend_frames` and `stanley_min_hold_frames` when the transition involves Stanley (check `_active_regime` or `_target_regime` == STANLEY to select the right frame counts).

### Step 5: Update blend frame selection in update() method
Currently `blend_frames` is a single config value. For Stanley transitions, use `stanley_blend_frames` instead. In the `update()` method (around line 120), when computing blend progress:
```python
# Determine which blend_frames to use based on transition type
if (self._active_regime == ControlRegime.STANLEY or
    self._target_regime == ControlRegime.STANLEY):
    effective_blend_frames = max(1, self.config.stanley_blend_frames)
else:
    effective_blend_frames = max(1, self.config.blend_frames)
self._blend_progress = min(1.0, self._blend_progress + 1.0 / effective_blend_frames)
```

### Acceptance criteria (2.9.1)
- [ ] `ControlRegime.STANLEY` has integer value -1
- [ ] `RegimeConfig` loads all stanley_* fields from YAML
- [ ] `RegimeSelector.update()` returns STANLEY when speed < 4.5 m/s (with defaults)
- [ ] `RegimeSelector.update()` returns PP when speed > 5.5 m/s (upshift from Stanley)
- [ ] Hysteresis prevents chatter at 5.0 m/s
- [ ] Blend weight ramps correctly on Stanley transitions
- [ ] When `regime.enabled=false`, always returns PP (Stanley disabled too)
- [ ] When `regime.enabled=true` but `stanley_enabled=false`, never returns Stanley
- [ ] `reset()` returns to PP (not Stanley)

---

## Sub-phase 2.9.2 — Wire Stanley in pid_controller.py

**File:** `control/pid_controller.py`

### Step 1: Accept regime in compute_steering
The `compute_steering()` method in `LateralController` (around line 1170) currently uses `self.control_mode` (a static string from config). Instead of changing the static mode, add a `regime` parameter to `compute_steering()`:

```python
def compute_steering(self, reference_point, current_state, ..., regime=None):
```

### Step 2: Dynamic Stanley activation
In the steering dispatch section (around line 1275–1435), add Stanley path. Currently:
```python
if self.control_mode in ("pid", "stanley"):
    # PID/Stanley branch (lines 1303-1435)
    if self.control_mode == "stanley":
        # Stanley formula (lines 1428-1435)
```

Change to also activate Stanley when regime says so:
```python
use_stanley = (
    self.control_mode == "stanley"
    or (regime is not None and regime == ControlRegime.STANLEY)
)

if self.control_mode in ("pid", "stanley") or use_stanley:
    if use_stanley:
        # Stanley formula
        stanley_heading_term = self.stanley_heading_weight * heading_error
        denom = max(self.stanley_soft_speed, float(stanley_speed))
        stanley_crosstrack_term = np.arctan2(
            self.stanley_k * lateral_error_for_control, denom
        )
        feedback_steering = stanley_heading_term + stanley_crosstrack_term
    else:
        feedback_steering = self.pid.update(total_error, dt)
```

### Step 3: Wire regime through VehicleController
In `VehicleController.compute_control()` (around line 5264), pass the regime to the lateral controller:
```python
regime, blend_weight = self._regime_selector.update(speed=speed, ...)

# Pass regime to lateral controller
lateral_metadata = self.lateral_controller.compute_steering(
    reference_point, current_state, ..., regime=regime
)
```

### Step 4: Stanley↔PP blending
When blending Stanley↔PP, blend the steering output:
```python
if regime == ControlRegime.STANLEY and blend_weight < 1.0:
    # Compute BOTH Stanley and PP steering, blend
    stanley_steering = ...  # from Stanley formula
    pp_steering = ...       # from PP formula
    steering = (1.0 - blend_weight) * pp_steering + blend_weight * stanley_steering
```

**Note:** Currently PP steering is always computed first (line 5264). If regime is STANLEY, recompute Stanley steering and blend. The existing PP→MPC blend pattern (line 5362-5365) is the template.

### Step 5: Add Stanley metadata to return dict
In the metadata dict returned by `compute_steering()` (around line 3446), the fields `stanley_heading_term`, `stanley_crosstrack_term`, `stanley_speed` are already present. Add:
```python
'stanley_active': 1.0 if use_stanley else 0.0,
```

### Acceptance criteria (2.9.2)
- [ ] When regime=STANLEY, steering comes from Stanley formula (heading + crosstrack)
- [ ] When regime=PP, steering comes from PP formula (unchanged from current behavior)
- [ ] Stanley↔PP blend produces smooth steering transition
- [ ] Stanley uses the same `lateral_error_for_control` and `heading_error_for_control` as PP
- [ ] `stanley_heading_term`, `stanley_crosstrack_term` correctly populated in metadata
- [ ] No regression: when regime=PP or regime=LMPC, behavior is identical to pre-change

---

## Sub-phase 2.9.3 — HDF5 recording

**Files:** `data/recorder.py`, `data/formats/data_format.py`, `av_stack/orchestrator.py`

**Remember the 6-location rule for new HDF5 fields:**
1. `create_dataset` in recorder.py
2. List init in recorder.py
3. Per-frame append in recorder.py
4. Resize loop in recorder.py
5. Write in recorder.py
6. `ControlCommandData` construction in orchestrator.py

### New fields
| Field | Type | Source |
|---|---|---|
| `control/stanley_active` | float32 | 1.0 if Stanley formula used, 0.0 otherwise |
| `control/stanley_heading_term` | float32 | Heading error component of Stanley steering |
| `control/stanley_crosstrack_term` | float32 | Cross-track error component of Stanley steering |

**Note:** `control/regime` already exists and will naturally record -1.0 for Stanley frames. No new field needed for regime identification.

### Acceptance criteria (2.9.3)
- [ ] All 3 new fields written correctly to HDF5 in all 6 locations
- [ ] `control/regime` correctly records -1.0 when Stanley is active
- [ ] Existing fields unchanged (no silent zeros)
- [ ] PP-only recordings (e.g., highway) write 0.0 for stanley_active (not missing)

---

## Sub-phase 2.9.4 — Tests

**Files:** `tests/test_regime_selector.py`, `tests/test_control.py` (or new `tests/test_stanley_regime.py`)

### New tests for regime_selector.py (add to existing test file)
```python
def test_stanley_downshift_on_low_speed():
    """Speed drops below stanley_max → STANLEY regime."""
    config = RegimeConfig(enabled=True, stanley_enabled=True, stanley_max_speed_mps=5.0)
    sel = RegimeSelector(config)
    # Run at 3.0 m/s for min_hold_frames
    for _ in range(config.stanley_min_hold_frames + 5):
        regime, blend = sel.update(speed=3.0)
    assert regime == ControlRegime.STANLEY
    assert blend == 1.0  # fully settled

def test_stanley_upshift_to_pp():
    """Speed rises above stanley_max + hysteresis → PP regime."""
    config = RegimeConfig(enabled=True, stanley_enabled=True, stanley_max_speed_mps=5.0)
    sel = RegimeSelector(config)
    # Settle into Stanley
    for _ in range(50):
        sel.update(speed=3.0)
    # Now speed rises
    for _ in range(50):
        regime, blend = sel.update(speed=7.0)
    assert regime == ControlRegime.PURE_PURSUIT
    assert blend == 1.0

def test_stanley_pp_mpc_full_sweep():
    """Speed sweep 2→15 m/s hits all three regimes."""
    config = RegimeConfig(enabled=True, stanley_enabled=True,
                          stanley_max_speed_mps=5.0, pp_max_speed_mps=10.0)
    sel = RegimeSelector(config)
    regimes_seen = set()
    for speed in [2.0]*50 + [7.0]*50 + [14.0]*50:
        regime, _ = sel.update(speed=speed)
        regimes_seen.add(regime)
    assert ControlRegime.STANLEY in regimes_seen
    assert ControlRegime.PURE_PURSUIT in regimes_seen
    assert ControlRegime.LINEAR_MPC in regimes_seen

def test_stanley_hysteresis_no_chatter():
    """Speed oscillates around stanley_max → no chatter."""
    config = RegimeConfig(enabled=True, stanley_enabled=True, stanley_max_speed_mps=5.0,
                          stanley_upshift_hysteresis_mps=0.5, stanley_downshift_hysteresis_mps=0.5)
    sel = RegimeSelector(config)
    # Start in Stanley
    for _ in range(50):
        sel.update(speed=3.0)
    # Oscillate around 5.0
    transitions = 0
    prev_regime = ControlRegime.STANLEY
    for speed in [4.8, 5.2, 4.9, 5.1, 4.7, 5.3] * 10:
        regime, _ = sel.update(speed=speed)
        if regime != prev_regime:
            transitions += 1
            prev_regime = regime
    assert transitions <= 2  # at most one switch each way

def test_stanley_disabled():
    """stanley_enabled=false → never returns Stanley."""
    config = RegimeConfig(enabled=True, stanley_enabled=False)
    sel = RegimeSelector(config)
    for _ in range(50):
        regime, _ = sel.update(speed=2.0)
    assert regime == ControlRegime.PURE_PURSUIT

def test_stanley_blend_weight():
    """Blend ramps 0→1 over stanley_blend_frames."""
    config = RegimeConfig(enabled=True, stanley_enabled=True,
                          stanley_max_speed_mps=5.0, stanley_blend_frames=10,
                          stanley_min_hold_frames=1)
    sel = RegimeSelector(config)
    for _ in range(50):
        sel.update(speed=3.0)
    # Now settled in Stanley. Speed jumps to 8.0 → PP.
    blends = []
    for _ in range(20):
        _, blend = sel.update(speed=8.0)
        blends.append(blend)
    # Blend should ramp from < 1.0 to 1.0
    assert blends[0] < 1.0  # not instantly settled
    assert blends[-1] == 1.0  # settled after enough frames
```

### Integration test for Stanley steering output
```python
def test_stanley_steering_at_low_speed():
    """At low speed with regime=STANLEY, steering uses heading + crosstrack formula."""
    # Create lateral controller with known params
    # Call compute_steering with regime=ControlRegime.STANLEY
    # Assert steering = stanley_heading_weight * heading_error + arctan2(k * lat_error, speed)
    # Assert PP geometric component is NOT used
```

### Regression test
```python
def test_regime_disabled_no_stanley():
    """When regime.enabled=false, compute_control produces identical output to pre-change."""
    # Run compute_control with regime disabled
    # Compare output to expected PP-only result
    # This ensures the Stanley code path has zero impact when disabled
```

### Acceptance criteria (2.9.4)
- [ ] 6+ new unit tests for regime selector with Stanley
- [ ] 1+ integration test for Stanley steering formula
- [ ] 1+ regression test proving no change when Stanley disabled
- [ ] All existing 11 regime selector tests still pass
- [ ] All 32 comfort gate tests still pass

---

## Sub-phase 2.9.5 — Configuration

**Files:** `config/av_stack_config.yaml`, `config/mpc_sloop.yaml`, `config/mpc_mixed.yaml`

### Base config (av_stack_config.yaml)
Add to `control.regime` section (after existing keys, around line 292):
```yaml
  regime:
    enabled: false                       # Master switch
    pp_max_speed_mps: 10.0
    # ... existing keys ...
    stanley_enabled: true                # Enable Stanley low-speed regime
    stanley_max_speed_mps: 5.0           # Below this speed → Stanley (m/s)
    stanley_upshift_hysteresis_mps: 0.5  # Must exceed 5.5 m/s to leave Stanley
    stanley_downshift_hysteresis_mps: 0.5  # Must drop below 4.5 m/s to enter Stanley
    stanley_blend_frames: 10             # 0.33s blend at 30 Hz
    stanley_min_hold_frames: 15          # 0.5s hold before switching
```

Add Stanley tuning params to `control.lateral` section (near existing stanley_k, around line 13):
```yaml
    stanley_k: 1.5                       # Cross-track gain (existing)
    stanley_soft_speed: 2.0              # Denominator floor for low-speed stability (existing)
    stanley_heading_weight: 1.0          # Heading error weight (existing)
    stanley_curve_only: false            # If true, only activate Stanley during curve phases (not on straights)
```

### Track overlays
No overlay changes needed for Stanley — it activates automatically at low speed. However, for tracks that should NEVER use Stanley (e.g., highway where speed is always > 10 m/s), it's already inactive because speed never drops below threshold.

For hairpin specifically, consider adding to a future `config/hairpin_15.yaml` if Stanley thresholds need adjustment:
```yaml
control:
  regime:
    enabled: true
    stanley_max_speed_mps: 6.0     # Higher threshold — hairpin peak speed is ~4 m/s
```

### Acceptance criteria (2.9.5)
- [ ] Stanley config loads correctly in base config
- [ ] Default `stanley_enabled: true` means Stanley activates when `regime.enabled: true`
- [ ] Overlays can override stanley_max_speed_mps per track
- [ ] `regime.enabled: false` disables both MPC and Stanley (always PP)

---

## Sub-phase 2.9.6 — PhilViz updates

**Files:** `tools/debug_visualizer/visualizer.js`, `tools/debug_visualizer/backend/mpc_pipeline.py`, `tools/debug_visualizer/index.html`

### 2.9.6a — Regime color scheme
In `visualizer.js`, update the regime shade plugin (around line 9930–9950) to handle STANLEY = -1:
```javascript
// Regime color map
const REGIME_COLORS = {
    '-1': 'rgba(76, 175, 80, 0.15)',   // Stanley = green (geometric, precise)
    '0':  'rgba(33, 150, 243, 0.15)',   // PP = blue (existing)
    '1':  'rgba(156, 39, 176, 0.15)',   // LMPC = purple (existing)
    '2':  'rgba(255, 152, 0, 0.15)',    // NMPC = orange (existing/future)
};
```

### 2.9.6b — Regime labels
In `visualizer.js`, update control mode label mapping (around line 1949):
```javascript
function regimeLabel(value) {
    if (value < -0.5) return 'Stanley';
    if (value < 0.5) return 'Pure Pursuit';
    if (value < 1.5) return 'Linear MPC';
    return 'Nonlinear MPC';
}
```

### 2.9.6c — Summary tab regime breakdown
In `visualizer.js` `loadSummary()` (around line 1968), extend PP/MPC frame count to include Stanley:
```javascript
// Show: Stanley / PP / MPC frame counts
const stanleyFrames = summary.stanley_frames || 0;
const ppFrames = summary.pp_frames || 0;
const mpcFrames = summary.mpc_frames || 0;
```

### 2.9.6d — MPC Pipeline tab → rename to "Regime Pipeline"
The MPC Pipeline tab should be renamed to "Regime Pipeline" and extended to show all regimes. In `index.html`, rename the tab button. In `visualizer.js` and `backend/mpc_pipeline.py`:
- Add a "Regime Distribution" card showing Stanley/PP/MPC frame percentages
- Keep existing MPC cards (only show when MPC frames > 0)
- Add Stanley quality card when Stanley frames > 0:
  - Stanley heading term P50/P95
  - Stanley crosstrack term P50/P95
  - Stanley active rate
  - Transition quality: |delta_steering| at Stanley↔PP transitions

### 2.9.6e — Guard on field existence
All new rendering MUST check field existence. PP-only recordings (before this change) will not have `stanley_active` field. Guard pattern:
```javascript
if (data.stanley_active !== undefined) { /* render Stanley card */ }
```

### Acceptance criteria (2.9.6)
- [ ] Regime timeline shows green for Stanley, blue for PP, purple for MPC
- [ ] Summary tab shows 3-way frame breakdown (Stanley / PP / MPC)
- [ ] Regime Pipeline tab shows Stanley quality card when Stanley frames > 0
- [ ] Old recordings (pre-Stanley) render identically — no errors, no missing cards
- [ ] PhilViz restarts cleanly after changes (manual restart required — debug=False)

---

## Sub-phase 2.9.7 — drive_summary_core.py scoring updates

**File:** `tools/drive_summary_core.py`

### 2.9.7a — Control mode detection (line 1056)
Update the control mode detection to include Stanley:
```python
regime_arr = np.asarray(regime, dtype=float)
stanley_rate = float(np.mean(regime_arr < -0.5))
mpc_rate = float(np.mean(regime_arr >= 0.5))
if stanley_rate > 0.5:
    return 'stanley'
elif mpc_rate > 0.5:
    return 'mpc'
elif stanley_rate > 0.0 and mpc_rate > 0.0:
    return 'hybrid_stanley_pp_mpc'
elif stanley_rate > 0.0:
    return 'hybrid_stanley_pp'
elif mpc_rate > 0.0:
    return 'hybrid_pp_mpc'
else:
    return 'pure_pursuit'
```

### 2.9.7b — Regime frame counts (line 5719)
Add Stanley frame count to the score dict:
```python
stanley_frames = int(np.sum(regime_arr < -0.5)) if regime_arr is not None else 0
```

### 2.9.7c — Stanley Health section
Add a Stanley Health section in the analysis report (similar to MPC Health at line 657), displayed as section 13.5 (after MPC Health section 13):
```
13.5. STANLEY HEALTH
   Stanley Active Rate: X% (Y frames)
   Heading Term P50/P95: a / b rad
   Crosstrack Term P50/P95: c / d rad
   Transition Quality (Stanley↔PP): max |delta_steering| = e rad [PASS/FAIL] (gate: < 0.10)
```

### 2.9.7d — Oscillation exclusion mask
In the oscillation detection section (line 2180), exclude Stanley frames the same way MPC frames are excluded:
```python
stanley_exclude = regime_arr < -0.5  # Stanley uses different error model
mpc_exclude = regime_arr > 0.5
exclude_mask = stanley_exclude | mpc_exclude | blend_mask
```

### 2.9.7e — Lateral error source selection
In the lateral error analysis (line 2255), use `lateral_error_for_control` for Stanley frames (same as PP), not `mpc_e_lat`:
```python
# Stanley and PP both use perception-derived lateral error
# MPC uses GT-derived e_lat
pp_stanley_mask = regime_arr < 0.5   # regime -1 (Stanley) or 0 (PP)
mpc_mask = regime_arr >= 0.5
# Use mpc_e_lat for MPC frames, lateral_error for PP/Stanley
```

### Acceptance criteria (2.9.7)
- [ ] Control mode correctly identifies 'stanley', 'hybrid_stanley_pp', etc.
- [ ] Stanley Health section appears in analysis output when Stanley frames > 0
- [ ] Stanley frames excluded from oscillation analysis
- [ ] No scoring changes for PP-only or MPC recordings (backward compatible)
- [ ] Lateral error correctly sourced per regime

---

## Sub-phase 2.9.8 — mpc_pipeline_analysis.py update

**File:** `tools/analyze/mpc_pipeline_analysis.py`

Rename to `regime_pipeline_analysis.py` (keep `mpc_pipeline_analysis.py` as a symlink or alias for backward compatibility).

### Updates
- Add Stanley regime detection: `stanley_mask = regime < -0.5`
- Add "Regime Distribution" section at top: Stanley/PP/MPC/Blend frame counts and percentages
- Add Stanley transition quality: `|delta_steering|` at Stanley↔PP transition frames
- Keep all existing MPC analysis sections (only print when MPC frames > 0)
- Print Stanley quality metrics when Stanley frames > 0

### Acceptance criteria (2.9.8)
- [ ] `--latest` works on recordings with Stanley, PP-only, and MPC-active
- [ ] Regime distribution shows all active regimes
- [ ] Stanley transition quality reported
- [ ] Old recordings produce identical output (no new sections when Stanley absent)

---

## Sub-phase 2.9.9 — Hairpin validation

### Step 1: Run hairpin with Stanley enabled
```bash
./start_av_stack.sh --build-unity-player --skip-unity-build-if-clean \
  --config config/mpc_sloop.yaml --track-yaml tracks/hairpin_15.yml --duration 60
```
Note: using `mpc_sloop.yaml` to enable regime selector. Stanley activates automatically because hairpin speed (2–4 m/s) < 5.0 m/s threshold.

### Step 2: Analyze
```bash
python tools/analyze/analyze_drive_overall.py --latest
python tools/analyze/regime_pipeline_analysis.py --latest
```

### Step 3: Expected results
- Stanley Active Rate > 80% (most frames are < 5 m/s on hairpin)
- Lateral RMSE < 0.30m (down from 0.435m with PP-only)
- Lateral P95 < 0.60m (down from 0.889m)
- Score > 85 (breaking the 79 Trajectory-yellow cap)
- 0 e-stops, 0 out-of-lane events
- `lateTurnIn=NO` on all curves (Stanley doesn't have the lookahead delay problem)

### Step 4: Run 3 validation runs
Repeat 3× and record median score. If median > 85 and 0 e-stops:
- Update `tests/fixtures/golden_recordings.json` with new hairpin baseline
- Update `BASELINE_SCORES` in `tests/conftest.py`
- Update hairpin score gate from 75 to 80+ as appropriate

### Step 5: Non-regression on other tracks
Run comfort gate suite to verify no regression on existing golden recordings:
```bash
pytest tests/test_comfort_gate_replay.py -v
```

### Acceptance criteria (2.9.9)
- [ ] Hairpin score > 85/100 (breaking 79 cap)
- [ ] 0 e-stops across 3 runs
- [ ] Lateral RMSE improvement > 30% (from 0.435m)
- [ ] All 32 comfort gate tests pass (non-regression)

---

## Sub-phase 2.9.10 — Documentation and cleanup

### Files to update
1. `docs/agent/architecture.md` — add Stanley regime to architecture diagram and regime table
2. `docs/agent/current_state.md` — Phase 2.9 status
3. `docs/agent/tasks.md` — mark hairpin improvement complete
4. `CLAUDE.md` — update Active Control Configuration section
5. `docs/ROADMAP.md` — update regime architecture description

### Memory update
Update MEMORY.md with:
- Stanley regime architecture
- Hairpin validated score
- New workflow commands if any

---

# PHASE 2.7 — Nonlinear MPC (NMPC)

## Prerequisites
- Phase 2.9 complete (Stanley regime validates multi-regime plumbing)
- Linear MPC validated on all current tracks (done — Phase 2.8.6)
- Need: either (a) target speed > 20 m/s, or (b) track with |e_heading| > 0.25 rad

## Architecture

**Solver:** CasADi + IPOPT (nonlinear interior-point solver)

**Model (full nonlinear bicycle dynamics):**
```
e_lat[k+1]     = e_lat[k]  +  v[k] * sin(e_heading[k]) * dt  -  kappa_ref[k] * v[k]^2 * dt
e_heading[k+1] = e_heading[k]  +  (v[k]/L_wb) * tan(delta[k]) * dt  -  kappa_ref[k] * v[k] * dt
v[k+1]         = v[k]  +  a[k] * dt
```

**Key differences from Linear MPC (Phase 2.1):**
| Aspect | Linear MPC (current) | NMPC (Phase 2.7) |
|---|---|---|
| Dynamics | sin(x)≈x, tan(x)≈x | Full sin(x), tan(x) |
| Solver | OSQP (convex QP) | IPOPT (nonlinear interior-point) |
| Solve time | ~0.5 ms P50 | ~5–15 ms expected |
| Valid range | |e_heading| < 0.25 rad, v < 20 m/s | Full range |
| Dependency | osqp (already installed) | casadi (new: `pip install casadi`) |

---

## Sub-phase 2.7.1 — nmpc_controller.py

**New file:** `control/nmpc_controller.py`

### Class structure (mirrors mpc_controller.py pattern)
```python
@dataclass
class NMPCParams:
    """NMPC configuration — loaded from YAML trajectory.nmpc.nmpc_*"""
    horizon: int = 20
    dt: float = 0.033
    wheelbase_m: float = 2.5
    max_steer_rad: float = 0.5236      # 30 degrees
    q_lat: float = 10.0
    q_heading: float = 5.0
    q_speed: float = 1.0
    q_lat_terminal_scale: float = 3.0
    q_heading_terminal_scale: float = 3.0
    r_steer: float = 0.1
    r_accel: float = 0.05
    r_steer_rate: float = 1.0
    delta_rate_max: float = 0.5
    max_accel: float = 1.2
    max_decel: float = 2.4
    v_min: float = 1.0
    v_max: float = 25.0
    max_solve_time_ms: float = 20.0
    max_consecutive_failures: int = 3

    @classmethod
    def from_config(cls, config: dict) -> "NMPCParams":
        nmpc_cfg = config.get("trajectory", {}).get("nmpc", {})
        kwargs = {}
        for field in cls.__dataclass_fields__:
            key = f"nmpc_{field}"
            if key in nmpc_cfg:
                kwargs[field] = nmpc_cfg[key]
        return cls(**kwargs)


class NMPCSolver:
    """CasADi + IPOPT nonlinear MPC solver."""

    def __init__(self, params: NMPCParams):
        self.params = params
        self._build_nlp()

    def _build_nlp(self):
        """Build CasADi NLP problem (called once at init)."""
        import casadi as ca

        N = self.params.horizon
        dt = self.params.dt
        L = self.params.wheelbase_m

        # Decision variables
        # State: [e_lat, e_heading, v] at each step
        # Control: [delta, accel] at each step

        # CasADi symbolic variables
        x = ca.MX.sym('x', 3, N+1)   # states
        u = ca.MX.sym('u', 2, N)      # controls
        p = ca.MX.sym('p', N+4)       # parameters: [e_lat0, e_heading0, v0, delta_prev, kappa_ref[0:N]]

        # Build NLP
        cost = 0
        constraints = []
        lbg = []
        ubg = []

        # Initial state constraint
        constraints.append(x[:, 0] - p[0:3])
        lbg += [0, 0, 0]
        ubg += [0, 0, 0]

        for k in range(N):
            e_lat_k = x[0, k]
            e_heading_k = x[1, k]
            v_k = x[2, k]
            delta_k = u[0, k]
            accel_k = u[1, k]
            kappa_k = p[4 + k]

            # Full nonlinear dynamics
            e_lat_next = e_lat_k + v_k * ca.sin(e_heading_k) * dt - kappa_k * v_k**2 * dt
            e_heading_next = e_heading_k + (v_k / L) * ca.tan(delta_k) * dt - kappa_k * v_k * dt
            v_next = v_k + accel_k * dt

            # Dynamics constraint
            constraints.append(x[:, k+1] - ca.vertcat(e_lat_next, e_heading_next, v_next))
            lbg += [0, 0, 0]
            ubg += [0, 0, 0]

            # Stage cost
            cost += self.params.q_lat * e_lat_k**2
            cost += self.params.q_heading * e_heading_k**2
            cost += self.params.q_speed * (v_k - p[2])**2  # speed tracking
            cost += self.params.r_steer * delta_k**2
            cost += self.params.r_accel * accel_k**2

            # Steering rate penalty
            if k == 0:
                delta_prev = p[3]
            else:
                delta_prev = u[0, k-1]
            cost += self.params.r_steer_rate * (delta_k - delta_prev)**2

        # Terminal cost
        cost += self.params.q_lat * self.params.q_lat_terminal_scale * x[0, N]**2
        cost += self.params.q_heading * self.params.q_heading_terminal_scale * x[1, N]**2

        # Flatten decision variables
        opt_x = ca.vertcat(ca.reshape(x, -1, 1), ca.reshape(u, -1, 1))

        # Build NLP
        nlp = {'x': opt_x, 'f': cost, 'g': ca.vertcat(*constraints), 'p': p}

        # IPOPT options
        opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': 100,
            'ipopt.max_cpu_time': self.params.max_solve_time_ms / 1000.0,
            'print_time': 0,
        }

        self._solver = ca.nlpsol('nmpc', 'ipopt', nlp, opts)
        self._N = N
        # Store bounds for decision variables
        # ... (steering bounds, accel bounds, speed bounds)

    def solve(self, e_lat, e_heading, speed, delta_prev, kappa_horizon):
        """Solve NMPC and return optimal first steering command."""
        # Set parameter vector
        # Call self._solver(x0=warm_start, p=params, lbx=lb, ubx=ub, lbg=lbg, ubg=ubg)
        # Extract first control action
        # Return dict with steering, feasible, solve_time, etc.
        pass


class NMPCController:
    """High-level NMPC controller with fallback handling."""

    def __init__(self, params: NMPCParams):
        self.params = params
        self._solver = NMPCSolver(params)
        self._fallback_active = False
        self._consecutive_failures = 0
        self._last_solution = None  # warm-start

    def compute_steering(self, e_lat, e_heading, current_speed, delta_prev, kappa_horizon):
        """Compute NMPC steering command. Falls back to last good solution on failure."""
        result = self._solver.solve(e_lat, e_heading, current_speed, delta_prev, kappa_horizon)

        if result['feasible']:
            self._consecutive_failures = 0
            self._fallback_active = False
            self._last_solution = result
        else:
            self._consecutive_failures += 1
            self._fallback_active = self._consecutive_failures >= self.params.max_consecutive_failures
            if self._last_solution is not None:
                result = self._last_solution  # warm-start fallback

        return result
```

### Acceptance criteria (2.7.1)
- [ ] `NMPCParams.from_config()` loads from `trajectory.nmpc.nmpc_*` keys
- [ ] `NMPCSolver._build_nlp()` constructs CasADi NLP with full nonlinear dynamics
- [ ] `NMPCSolver.solve()` returns steering, feasible, solve_time
- [ ] `NMPCController` handles consecutive failures with fallback
- [ ] Solve time P95 < 20 ms on standard hardware
- [ ] Unit tests: known-state steering correctness, constraint satisfaction, fallback behavior

---

## Sub-phase 2.7.2 — Wire NMPC in regime selector + pid_controller.py

**Files:** `control/regime_selector.py`, `control/pid_controller.py`

### Step 1: RegimeSelector already handles NMPC (line 164–167)
The existing code already has NMPC upshift logic:
```python
if speed > self.config.lmpc_max_speed_mps + self.config.upshift_hysteresis_mps:
    return ControlRegime.NONLINEAR_MPC
```
**No changes needed in regime_selector.py for NMPC activation.**

### Step 2: Wire NMPCController in VehicleController (pid_controller.py)
In `VehicleController.__init__()` (around line 5222), add NMPC instantiation:
```python
# Existing:
self._mpc_controller = MPCController(...) if mpc_enabled else None

# New:
nmpc_params = NMPCParams.from_config(self._full_config)
nmpc_enabled = self._full_config.get('trajectory', {}).get('nmpc', {}).get('nmpc_enabled', False)
self._nmpc_controller = NMPCController(nmpc_params) if nmpc_enabled else None
```

### Step 3: Add NMPC dispatch in compute_control (around line 5293)
```python
if regime == ControlRegime.NONLINEAR_MPC and self._nmpc_controller is not None:
    # Same pattern as LINEAR_MPC dispatch (lines 5293-5369)
    # Use GT state (groundTruthLaneCenterX)
    # Apply delay compensation
    # Call nmpc_controller.compute_steering()
    # Blend with LMPC or PP if blend_weight < 1.0
    pass
elif regime == ControlRegime.LINEAR_MPC and self._mpc_controller is not None:
    # ... existing LMPC code ...
```

### Step 4: LMPC↔NMPC warm-start handoff
When transitioning from LMPC to NMPC, pass the last LMPC solution as NMPC warm-start:
```python
if prev_regime == ControlRegime.LINEAR_MPC and regime == ControlRegime.NONLINEAR_MPC:
    self._nmpc_controller.set_warm_start(self._mpc_controller.last_solution)
```

### Acceptance criteria (2.7.2)
- [ ] NMPC activates when speed > 21 m/s (with default hysteresis)
- [ ] NMPC steering output is applied correctly
- [ ] LMPC→NMPC blend is smooth (blend_weight ramp)
- [ ] NMPC→LMPC downshift works on speed reduction
- [ ] Warm-start handoff reduces NMPC solve time on first frame

---

## Sub-phase 2.7.3 — HDF5 recording

**New fields (same 6-location pattern):**
| Field | Type | Source |
|---|---|---|
| `control/nmpc_feasible` | int8 | NMPC solver feasibility |
| `control/nmpc_solve_time_ms` | float32 | NMPC solve time |
| `control/nmpc_fallback_active` | int8 | NMPC in fallback mode |
| `control/nmpc_e_lat` | float32 | Cross-track error fed to NMPC |
| `control/nmpc_e_heading` | float32 | Heading error fed to NMPC |

`control/regime` already handles value 2 (NONLINEAR_MPC). No new field needed.

---

## Sub-phase 2.7.4 — Tests

**New file:** `tests/test_nmpc_controller.py`

Mirror the structure of `tests/test_mpc_controller.py` (14 tests):
1. `test_nmpc_params_from_config()` — config loading
2. `test_nmpc_straight_line()` — zero error → near-zero steering
3. `test_nmpc_lateral_offset_correction()` — positive e_lat → negative steering
4. `test_nmpc_heading_error_correction()` — heading error → corrective steering
5. `test_nmpc_curvature_feedforward()` — nonzero kappa → proportional steering
6. `test_nmpc_steering_constraints()` — output within [-max_steer, max_steer]
7. `test_nmpc_rate_constraints()` — delta_rate within bounds
8. `test_nmpc_fallback_on_failure()` — consecutive failures → fallback
9. `test_nmpc_solve_time()` — P95 < 20 ms
10. `test_nmpc_vs_lmpc_small_angle()` — at small angles, NMPC ≈ LMPC (consistency)
11. `test_nmpc_large_heading()` — |heading| = 0.5 rad → stable (LMPC would diverge)
12. `test_nmpc_warm_start()` — warm-started solve is faster than cold
13. `test_nmpc_speed_adaptive_horizon()` — horizon extends at high speed

### Regime integration tests
Add to `tests/test_regime_selector.py`:
```python
def test_nmpc_upshift_at_high_speed():
    """Speed > lmpc_max + hysteresis → NMPC regime."""

def test_full_four_regime_sweep():
    """Speed sweep 2→25 m/s: Stanley → PP → LMPC → NMPC."""
```

---

## Sub-phase 2.7.5 — Configuration

**Base config (av_stack_config.yaml):**
Add `trajectory.nmpc` section:
```yaml
trajectory:
  nmpc:
    nmpc_enabled: false              # Master switch
    nmpc_horizon: 20
    nmpc_dt: 0.033
    nmpc_wheelbase_m: 2.5
    nmpc_max_steer_rad: 0.5236       # 30 degrees
    nmpc_q_lat: 10.0
    nmpc_q_heading: 5.0
    nmpc_q_speed: 1.0
    nmpc_q_lat_terminal_scale: 3.0
    nmpc_q_heading_terminal_scale: 3.0
    nmpc_r_steer: 0.1
    nmpc_r_accel: 0.05
    nmpc_r_steer_rate: 1.0
    nmpc_delta_rate_max: 0.5
    nmpc_max_accel: 1.2
    nmpc_max_decel: 2.4
    nmpc_v_min: 1.0
    nmpc_v_max: 25.0
    nmpc_max_solve_time_ms: 20.0
    nmpc_max_consecutive_failures: 3
```

**High-speed track overlay (future):**
```yaml
# config/highway_25mps.yaml (when needed)
control:
  regime:
    enabled: true
    lmpc_max_speed_mps: 20.0
trajectory:
  target_speed: 25.0
  nmpc:
    nmpc_enabled: true
```

---

## Sub-phase 2.7.6 — PhilViz + scoring

Same pattern as 2.9.6/2.9.7:
- Regime timeline includes NMPC = orange shading
- NMPC Health card in Regime Pipeline tab (solve time, feasibility, fallback rate)
- drive_summary_core.py: NMPC frames excluded from PP oscillation analysis
- NMPC Health section in analysis report (same structure as MPC Health)

---

## Sub-phase 2.7.7 — High-speed validation

### Entry criteria
- Need a track that requires > 20 m/s
- OR a track with tight curves at 15+ m/s (where LMPC linearization breaks)

### Validation plan
1. Create `tracks/highway_fast.yml` — long straight + moderate curve (R100) at 25 m/s
2. Create `config/highway_25mps.yaml` overlay with `nmpc_enabled: true`, `target_speed: 25.0`
3. Run 5 A/B comparison: LMPC-only vs LMPC+NMPC at 25 m/s
4. Score gate: ≥ 90/100 with 0 e-stops
5. NMPC should show improvement on the R100 curve where heading error > 0.25 rad

---

## Summary — Implementation Order

| Phase | Sub-phase | Effort | Dependencies |
|---|---|---|---|
| **2.9.1** | Regime selector + Stanley enum | ~2 hours | None |
| **2.9.2** | Wire Stanley in controller | ~2 hours | 2.9.1 |
| **2.9.3** | HDF5 recording | ~1 hour | 2.9.2 |
| **2.9.4** | Tests | ~2 hours | 2.9.1–2.9.3 |
| **2.9.5** | Configuration | ~30 min | 2.9.1 |
| **2.9.6** | PhilViz updates | ~2 hours | 2.9.3 |
| **2.9.7** | drive_summary_core.py | ~2 hours | 2.9.3 |
| **2.9.8** | regime_pipeline_analysis.py | ~1 hour | 2.9.3, 2.9.7 |
| **2.9.9** | Hairpin validation | ~2 hours (3 runs) | 2.9.1–2.9.8 |
| **2.9.10** | Documentation | ~1 hour | 2.9.9 |
| | **Phase 2.9 total** | **~15 hours** | |
| **2.7.1** | nmpc_controller.py | ~4 hours | 2.9 complete, `pip install casadi` |
| **2.7.2** | Wire NMPC in controller | ~2 hours | 2.7.1 |
| **2.7.3** | HDF5 recording | ~1 hour | 2.7.2 |
| **2.7.4** | Tests | ~3 hours | 2.7.1–2.7.3 |
| **2.7.5** | Configuration | ~30 min | 2.7.1 |
| **2.7.6** | PhilViz + scoring | ~2 hours | 2.7.3 |
| **2.7.7** | High-speed validation | ~3 hours | 2.7.1–2.7.6 |
| | **Phase 2.7 total** | **~16 hours** | |
