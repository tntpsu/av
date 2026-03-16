# Camera Optimization + Summary Tab Accuracy + Trajectory Score Plan

**Created:** 2026-03-03
**Baseline commit:** `67dabb4` (Phase 2.4b + metric fixes)
**Tracks:** s_loop, highway_65

---

## Part 1 — Unity Camera Pipeline Optimization

### Current State

| Technique | Status | File |
|---|---|---|
| AsyncGPUReadback | **YES** (primary path) | `CameraCapture.cs:408-416` |
| Shader prewarming | **NO** | — |
| Fixed timestep capture | **YES** (GT sync mode, off by default) | `CameraCapture.cs:295-313` |
| RenderTexture pipeline | **YES** | `CameraCapture.cs:243-250` |
| Double buffering | **NO** (single RenderTexture) | `CameraCapture.cs:69` |

**Problem observed:** 267ms cadence gaps concentrated in startup frames (shader compilation jank), plus 4 MPC solve spikes of 75–85ms in steady state.

### Fix A — Shader Prewarming (P0, ~1 hour)

**Goal:** Eliminate startup hitching by precompiling shaders before first camera frame.

**New file:** `unity/AVSimulation/Assets/Scripts/ShaderPrewarmer.cs`

```csharp
public class ShaderPrewarmer : MonoBehaviour
{
    public ShaderVariantCollection variantCollection;
    public bool useGenericWarmup = true;

    void Awake()
    {
        if (variantCollection != null)
        {
            variantCollection.WarmUp();
            Debug.Log($"ShaderPrewarmer: Warmed {variantCollection.shaderCount} shaders");
        }
        if (useGenericWarmup)
        {
            Shader.WarmupAllShaders();
            Debug.Log("ShaderPrewarmer: WarmupAllShaders complete");
        }
    }
}
```

**Steps:**
1. Create `ShaderPrewarmer.cs` with Script Execution Order = -100 (before CameraCapture)
2. Optionally create a ShaderVariantCollection asset (Window → Rendering → Shader Variant Collection)
3. Add startup grace period to `CameraCapture.cs`: skip first 10 `Update()` cycles after `Start()`
4. Attach to simulation manager GameObject

**Validation metric:** Count of frames with dt > 100ms in first 300 frames (before: ~110, target: < 10)

### Fix B — Double Buffering (P2, ~2 hours)

**Goal:** Overlap GPU render and CPU readback to reduce steady-state jitter.

**Modify:** `unity/AVSimulation/Assets/Scripts/CameraCapture.cs`

**Steps:**
1. Replace `private RenderTexture renderTexture` with `private RenderTexture[] renderTextures = new RenderTexture[2]`
2. Allocate both in `Start()`, alternate `activeBufferIndex` in `CaptureAndSend()`
3. Allow up to 2 in-flight async readbacks (one per buffer)
4. Update `OnDestroy()` cleanup to release both textures

**Validation metric:** Steady-state dt P95 (before: ~67ms, target: < 40ms)

### Fix C — Enable Async Readback in GT Sync Mode (P1, ~15 min)

**Goal:** GT sync mode currently forces synchronous ReadPixels. Enable async for better timing.

**Modify:** `CameraCapture.cs` line 57: `public bool disableAsyncReadbackInGtSync = true` → `false`

**Validation:** GT sync timestamps remain monotonic, no frame drops.

---

## Part 2 — Validation Runs

After implementing camera fixes, run both tracks and analyze:

### 2a. S-Loop Validation

```bash
./start_av_stack.sh \
  --build-unity-player \
  --skip-unity-build-if-clean \
  --duration 60

python tools/analyze/analyze_drive_overall.py --latest
```

**Metrics to check:**
- Cadence: frames with dt > 100ms (target: < 5 in first 300 frames)
- Cadence P95 dt (target: < 40ms)
- Overall score (baseline: 94.0, target: ≥ 94.0 — no regression)
- All comfort gates pass

### 2b. Highway Validation

```bash
./start_av_stack.sh \
  --build-unity-player \
  --skip-unity-build-if-clean \
  --config config/highway_15mps.yaml \
  --track-yaml tracks/highway_65.yml \
  --duration 120

python tools/analyze/analyze_drive_overall.py --latest
```

**Metrics to check:**
- Same cadence metrics as s-loop
- Overall score (baseline: 92.1, target: ≥ 92.1)
- Lateral Error RMSE (baseline: 0.3146m — see Part 4 for improvement plan)
- MPC health gates (feasibility, solve time P95)

---

## Part 3 — Summary Tab Red Metric Audit

Audit every metric that currently shows red/orange on the Summary tab for each track. For each one, determine if it's **real** (actual performance issue) or **fake** (metric artifact / wrong measurement).

### Highway Recording (`recording_20260302_231952.h5`)

#### 3a. Steering Jerk Max: 101.9 /s² (limit ≤ 18.0) — RED

**Diagnosis:** Investigate whether this is:
- Concentrated in PP startup frames (first ~100 frames before MPC takes over)
- Caused by cadence gaps (dt irregularity amplifies jerk calculation)
- A real MPC issue during steady-state

**Investigation steps:**
1. Extract per-frame steering jerk time series from HDF5
2. Identify frames where |steering_jerk| > 18.0
3. Check if those frames are in PP regime (frames 0–~300) or MPC regime
4. Check dt at those frames — are they cadence gap boundaries?
5. Check if filtering steering jerk (skip frames with dt > 50ms) brings it within limits

**If fake (startup/cadence artifact):**
- Add regime-aware and cadence-aware steering jerk computation
- Exclude PP startup frames (first N frames where regime=0) from the max
- Exclude frames adjacent to cadence gaps (dt > 2× median) from the max
- Display: "Steering Jerk Max: X.XX /s² (excl. startup)" with raw value in gray

**If real:**
- Investigate MPC steering rate limiting
- Check if `r_steer_rate` weight in MPC is aggressive enough
- May need to tighten rate penalty in `mpc_sloop.yaml` / `mpc_highway.yaml`

#### 3b. Steering Rate Max: high — RED/ORANGE

**Diagnosis:** Same investigation as 3a — likely correlated with same frames.

**Investigation steps:**
1. Extract steering rate time series
2. Cross-reference with steering jerk hotspot frames
3. If same frames → same root cause, same fix

#### 3c. Jerk P95 (Measured): 387.6 m/s³ — ALREADY FIXED ✅

**Status:** Already addressed in commit `a208e9c`. Commanded jerk (1.44 m/s³) is now the primary colored metric. Measured jerk is gray "diagnostic". No further action needed.

#### 3d. Curvature Undercall Rate: 100% — ALREADY FIXED ✅

**Status:** Already addressed in commit `a208e9c`. GT curvature floor guard (< 0.005 rad/m → N/A). No further action needed.

#### 3e. Cadence Health — RED

**Diagnosis:** Cadence gaps from Unity camera delivery.

**If camera fixes (Part 1) resolve:**
- Cadence health should go green automatically
- Verify: dt P95 < 40ms, max gap < 100ms

**If camera fixes don't fully resolve:**
- Add cadence-aware exclusion: "Cadence Health: X frames irregular (excl. from derivative metrics)"
- Ensure derivative metrics (jerk, steering rate) skip irregular-dt frames

#### 3f. Control Score: 80.0 (steering jerk cap -20) — RED

**Depends on:** 3a investigation. If steering jerk is a startup/cadence artifact, the fix is to exclude those frames from scoring. If real, needs MPC tuning.

### S-Loop Recording (`recording_20260302_150208.h5`)

#### 3g. Steering Jerk Max: 43.3 /s² (limit ≤ 18.0) — RED

**Same investigation as 3a.** S-loop runs mostly PP at low speed, so this is likely all PP frames. Check if spikes are at curve entry/exit transitions.

#### 3h. Oscillation Amplitude Runaway: YES — RED

**Diagnosis:** RMS grew from 0.052m to 0.382m over the run. Investigate:
1. Is this real lateral oscillation growth, or a measurement artifact?
2. Does it correlate with curve segments (expected wider oscillation)?
3. Plot lateral error RMS in 10-second windows to visualize growth

**If real:** PP damping issue at s-loop speeds. May need `pp_feedback_gain` tuning.
**If fake:** Window size or segment selection artifact — adjust the metric.

#### 3i. Heading Suppression Rate — ORANGE (SignalIntegrity -7.2)

**Diagnosis:** Heading suppression during curves is likely the correct behavior (EMA gating suppresses noisy heading on tight curves). If the metric penalizes correct behavior, it's a scoring issue.

**Investigation:** Check what heading_suppression_rate measures and whether it should be gated on curvature.

---

## Part 4 — Highway Trajectory Score Investigation

**Current:** 82.2/100 (target: ≥ 90)
**Primary deduction:** Lateral Error RMSE = 0.3146m (target ≤ 0.20m, costs 15.7 points)
**Secondary:** Lateral Error P95 = 0.4589m (target ≤ 0.40m, costs 2.1 points)

### Root Cause Hypotheses

1. **MPC steady-state offset:** Mean lateral error = 0.2608m suggests a constant bias, not oscillation. Check if MPC has a systematic cross-track offset.
   - Is the sign convention causing a persistent ~26cm bias?
   - Is the lane center reference (groundTruthLaneCenterX) accurate?
   - Does the offset correlate with speed (worse at higher speeds)?

2. **MPC weight tuning:** `q_lat=0.5` may be too low for highway. The MPC converges to ±0.005m in its own error frame but the *scoring* uses perception-based lateral error, which may have a different reference.

3. **Perception vs GT reference frame mismatch:** The scoring uses `control/lateral_error` (perception-based), but MPC tracks `mpc_gt_cross_track_m` (ground-truth-based). If perception has a systematic bias vs GT, the MPC will track the wrong reference.

### Investigation Steps

1. **Plot both error signals:**
   ```python
   # In a notebook or script:
   lateral_error = f['control/lateral_error'][:]      # perception-based (what scoring uses)
   mpc_gt_ct = f['control/mpc_gt_cross_track_m'][:]   # GT-based (what MPC tracks)
   regime = f['control/regime'][:]
   ```
   Compare during MPC-active frames. If `mpc_gt_ct ≈ 0` but `lateral_error ≈ 0.26m`, it's a reference frame mismatch.

2. **Check lateral_error source:** Where does `control/lateral_error` come from in the recorder? Is it perception lane center or GT lane center? If perception, that's the mismatch.

3. **Quantify the bias:**
   ```python
   mpc_frames = regime >= 1
   bias = np.mean(lateral_error[mpc_frames]) - np.mean(mpc_gt_ct[mpc_frames])
   ```
   If bias ≈ 0.26m → reference frame mismatch confirmed.

### Resolution Options

**Option A — Fix the reference (preferred):**
If MPC tracks GT correctly but scoring uses perception with systematic offset, either:
- Score using GT lateral error when MPC is active (regime ≥ 1)
- Or calibrate perception reference to match GT

**Option B — Tune MPC to compensate:**
Add a bias correction term to MPC's cross-track error: `e_lat_corrected = e_lat - bias`
Less clean but quick. Risk: bias may vary by track/speed.

**Option C — Dual tracking:**
MPC tracks GT for stability, but adds a slow correction term toward perception lane center.
Most robust but most complex.

### Target After Fix
- Lateral Error RMSE: ≤ 0.20m → Trajectory score ≥ 97
- Overall score: ≥ 96 (up from 92.1)

---

## Part 5 — Summary Tab Recommendations (Post-Investigation)

After completing the investigations in Part 3, update the Summary tab rendering for any metrics confirmed as "fake":

| Metric | If Fake → Action |
|---|---|
| Steering Jerk Max | Exclude startup + cadence-gap frames; show filtered value as primary, raw in gray |
| Steering Rate Max | Same exclusion logic as jerk |
| Measured Jerk P95 | Already fixed (gray diagnostic) |
| Curvature Undercall | Already fixed (N/A with reason) |
| Cadence Health | If camera fixes help, auto-resolves; if not, exclude from derivative penalties |
| Oscillation Runaway | If curve-correlated, gate on curvature; show "N/A on curved tracks" |
| Heading Suppression | If correct behavior, remove penalty or gate on curvature |

For metrics confirmed as **real**, implement the fixes described in Part 3 and re-validate.

---

## Part 6 — Diagnostic Consistency Alignment

The three PhilViz diagnostic surfaces (Summary, Triage, Layers) currently tell inconsistent
stories about the same recording. This section aligns them so a user can drill down from
Summary → Triage → Layers without encountering contradictions.

### Principle: `drive_summary_core.py` is the Single Source of Truth

All scoring thresholds and metric definitions live in `drive_summary_core.py`. Both the
triage engine and layer health derive from it — never define independent thresholds.

### 6a. Triage Engine Fixes

**File:** `tools/debug_visualizer/backend/triage_engine.py`

| # | Fix | Details |
|---|---|---|
| T1 | **Remove dead patterns** | `traj_curvature_error` and `ctrl_throttle_surge` check metrics never extracted. Either implement the metric extraction or remove the patterns. Recommend: remove for now, add back when metrics exist. |
| T2 | **Fix rate_key_map for perc_low_confidence** | Currently maps to `stale_rate`. Should map to a new metric: count frames where confidence < 0.4 |
| T3 | **Fix occurrence fallback** | 7 patterns default to `1/n` instead of actual counts. Add `out_of_lane_events`, `steering_jerk_max`, `commanded_jerk_p95` to rate_key_map with proper frame counts |
| T4 | **Align jerk threshold** | `CTRL_JERK_ALERT=4.5` is an early-warning threshold (75% of 6.0 gate). This is intentional — triage should warn before the gate trips. Keep as-is but add a note in the pattern description: "Alert at 75% of 6.0 m/s³ comfort gate" |
| T5 | **Update code pointers** | 3 patterns reference `av_stack.py` → update to `av_stack/lane_gating.py` or `av_stack/orchestrator.py` |
| T6 | **Add cadence-awareness** | `ctrl_steering_jerk` pattern should exclude frames with irregular dt (> 2× median). Currently it can fire on cadence gap artifacts |
| T7 | **Add regime-awareness** | `ctrl_lateral_oscillation` osc_slope should be computed separately for PP and MPC frames. Mixing both regimes produces meaningless slope values |

### 6b. Layer Health Alignment with Summary Scoring

**File:** `tools/debug_visualizer/backend/layer_health.py`

The layer health system scores 3 layers per-frame; the summary scores 6 layers aggregate.
The fix is NOT to make them identical (they serve different purposes) but to ensure they
tell the same directional story.

| # | Fix | Details |
|---|---|---|
| L1 | **Add Safety layer** | layer_health has no safety scoring. Add `_score_safety()` that checks `control/emergency_stop` and `control/lateral_error` > lane_half_width at each frame. Flags: `out_of_lane`, `emergency_stop` |
| L2 | **Tighten lateral error threshold** | Currently 2.0m max (everything below looks healthy). Summary targets 0.20m RMSE. Change to: penalty starts at 0.15m, full penalty at 0.60m (matches the range where summary deductions kick in) |
| L3 | **Add heading error to trajectory** | Summary penalizes heading error > 10°. Add to `_score_trajectory()`: if `heading_error_rad > 0.175` (10°), penalty -0.20; if > 0.35 (20°), penalty -0.40 |
| L4 | **Add oscillation to control** | Summary penalizes oscillation frequency > 1.0 Hz. This requires a windowed computation — add a rolling oscillation detector to `_score_control()` that checks zero-crossing frequency of lateral error derivative over a 60-frame window |
| L5 | **Add LongitudinalComfort layer** | New `_score_longitudinal()` that checks commanded jerk (from throttle/brake proxy). Use same 6.0 m/s³ gate as summary. Flags: `jerk_spike`, `accel_spike` |
| L6 | **Add SignalIntegrity layer** | New `_score_signal_integrity()` that checks heading gate active + ref rate limit active. Only penalize during curve frames (GT curvature > 0.003). Flags: `heading_suppressed`, `rate_limited` |
| L7 | **Import thresholds from drive_summary_core** | Create a shared `SCORING_THRESHOLDS` dict in `drive_summary_core.py` that both systems import. Prevents threshold drift. |

### 6c. Issues Tab Alignment

**File:** `tools/debug_visualizer/backend/issue_detector.py`

The issues tab is the best-aligned surface but has gaps:

| # | Fix | Details |
|---|---|---|
| I1 | **Centerline cross severity** | Flagged as "high" but summary doesn't penalize. Downgrade to "info" severity — it's a diagnostic event, not a scoring issue |
| I2 | **Lane jitter threshold** | Detector uses 0.8 m/frame (raw diff). Summary penalizes at 0.30m P95. Align: use P95-based detection with 0.30m threshold to match summary |
| I3 | **Heading jump — orphan** | Detected but no summary penalty. Either add summary penalty or downgrade to "info" with note "diagnostic only" |
| I4 | **Add missing detectors** | Summary penalizes these but issues tab is silent: steering jerk > 18 /s², oscillation > 1.0 Hz, lane detection < 90%, stale rate > 10% |
| I5 | **Straight sign mismatch** | Detector uses duration (≥10 frames). Summary uses rate (>5%). Align: add rate-based detection |

### 6d. Compare Tab Fixes

**Files:** `tools/debug_visualizer/server.py` + `tools/debug_visualizer/visualizer.js`

#### D1. Three-Column Layout (Current vs Pinned)

Currently the non-baseline column shows only deltas. Change to:

| Column | Content |
|---|---|
| **Metric** | Row label |
| **Baseline (pinned)** | Raw value |
| **Current** | Raw value |
| **Delta** | Difference with color (green=better, red=worse) |

Implementation: In `loadCompare()`, change the cell renderer so non-baseline rows show
raw value AND a delta badge. Add a third `<td>` for the delta column.

#### D2. Add Missing Metrics to Match Summary

The compare tab currently has 26 metrics but is missing key summary-scored fields.
Add these to close the gap:

| Group | Metric to Add | Summary Source |
|---|---|---|
| Path Tracking | **Lateral Error P95 (m)** | `path_tracking.lateral_error_p95` |
| Path Tracking | **Centered Frames %** | `path_tracking.time_in_lane_centered` |
| Path Tracking | **Heading Error RMSE (rad)** | `path_tracking.heading_error_rmse` |
| Control | **Steering Jerk Max (/s²)** | `control_smoothness.steering_jerk_max` |
| Control | **Oscillation Freq (Hz)** | `control_stability.straight_oscillation_frequency` |
| Control | **Cmd Jerk P95 (m/s³)** | `comfort.commanded_jerk_p95` |
| Perception | **Lane Detection Rate %** | `perception_quality.lane_detection_rate` |
| Perception | **Lane Jitter P95 (m)** | `perception_quality.lane_line_jitter_p95` |
| Longitudinal | **Lateral Accel P95 (g)** | `comfort.lateral_accel_p95_g` |

#### D3. Add Layer Scores Group

Add a new group at top (after Summary) showing all 6 layer scores so the compare
view directly mirrors the summary scorecard:

```javascript
['Layer Scores', [
    ['Safety', 'layer_safety_score', 'delta', false],
    ['Trajectory', 'layer_trajectory_score', 'delta', false],
    ['Control', 'layer_control_score', 'delta', false],
    ['Perception', 'layer_perception_score', 'delta', false],
    ['SignalIntegrity', 'layer_signal_integrity_score', 'delta', false],
    ['LongComfort', 'layer_longitudinal_comfort_score', 'delta', false],
]],
```

Backend: Extract these from `summary["executive_summary"]["layer_scores"]` in the
`/api/compare` row builder.

### 6e. Forward-Looking Alignment: Metric Registry

**Problem:** Today, thresholds and metric definitions are scattered across 5 files. When
someone adds a new metric or changes a threshold, they update one surface and forget the
others. This will keep happening.

**Solution:** Create a **Metric Registry** — a single declarative data structure that
all surfaces derive from.

**File:** `tools/scoring_registry.py` (new)

```python
"""
Single source of truth for all scoring metrics, thresholds, and display config.

Every PhilViz surface (Summary, Triage, Issues, Layers, Compare) imports from here.
Adding a new metric = adding one entry to this registry.
"""

METRICS = {
    # ── Safety ────────────────────────────────────────
    "out_of_lane_events": {
        "layer": "safety",
        "label": "Out-of-Lane Events",
        "unit": "count",
        "gate": 0,                    # comfort gate threshold
        "summary_max_penalty": 45,    # max points deducted in summary
        "issue_severity": "critical",
        "triage_pattern": "safety_ool_event",
        "compare_group": "Safety",
        "compare_format": "deltaInt",
        "direction": "lower_is_better",
        "hdf5_source": "control/emergency_stop",
    },
    "lateral_error_rmse": {
        "layer": "trajectory",
        "label": "Lateral Error RMSE",
        "unit": "m",
        "gate": 0.40,
        "target": 0.20,             # penalty starts here
        "summary_max_penalty": 30,
        "layer_health_penalty_range": [0.15, 0.60],  # per-frame scoring range
        "compare_group": "Path Tracking",
        "compare_format": "delta",
        "direction": "lower_is_better",
        "hdf5_source": "control/lateral_error",
    },
    "commanded_jerk_p95": {
        "layer": "longitudinal_comfort",
        "label": "Commanded Jerk P95",
        "unit": "m/s³",
        "gate": 6.0,
        "triage_alert": 4.5,        # early warning at 75% of gate
        "summary_max_penalty": 20,
        "issue_severity": None,      # no frame-level issue (aggregate only)
        "triage_pattern": "ctrl_jerk_spike",
        "compare_group": "Longitudinal",
        "compare_format": "delta",
        "direction": "lower_is_better",
    },
    # ... (all ~30 scored metrics)
}

LAYERS = {
    "safety":              {"weight": 0.2944, "critical": True},
    "trajectory":          {"weight": 0.276,  "critical": False},
    "control":             {"weight": 0.1472, "critical": False},
    "perception":          {"weight": 0.1288, "critical": False},
    "signal_integrity":    {"weight": 0.080,  "critical": False},
    "longitudinal_comfort":{"weight": 0.0736, "critical": False},
}

def get_metrics_for_layer(layer_name):
    return {k: v for k, v in METRICS.items() if v["layer"] == layer_name}

def get_compare_groups():
    """Return compare tab group structure derived from registry."""
    groups = {}
    for key, m in METRICS.items():
        group = m.get("compare_group")
        if group:
            groups.setdefault(group, []).append((m["label"], key, m["compare_format"]))
    return groups
```

**How each surface uses it:**

| Surface | What it imports |
|---|---|
| `drive_summary_core.py` | `METRICS[key]["gate"]`, `METRICS[key]["target"]`, `METRICS[key]["summary_max_penalty"]`, `LAYERS` weights |
| `triage_engine.py` | `METRICS[key]["triage_alert"]`, `METRICS[key]["triage_pattern"]` |
| `layer_health.py` | `METRICS[key]["layer_health_penalty_range"]`, `METRICS[key]["gate"]` |
| `issue_detector.py` | `METRICS[key]["gate"]`, `METRICS[key]["issue_severity"]` |
| `visualizer.js` (compare) | Backend exposes `get_compare_groups()` via API; frontend renders dynamically |

**Key property:** Adding a new metric to the registry automatically makes it available
in all surfaces. Forgetting a surface is no longer possible because the surfaces don't
have independent metric lists.

### 6f. Consistency Test

**File:** `tests/test_diagnostic_consistency.py` (new)

Automated test that prevents drift:

```python
def test_all_triage_patterns_have_metric_extraction():
    """Every triage pattern must reference a metric that _extract_aggregate_metrics computes."""

def test_all_issue_types_have_summary_scoring():
    """Every issue type with severity >= 'medium' must correspond to a summary penalty."""

def test_layer_health_layers_match_summary():
    """layer_health.py must score the same 6 layers as drive_summary_core.py."""

def test_compare_tab_covers_all_gate_metrics():
    """Every metric with a 'gate' threshold must appear in the compare tab."""

def test_thresholds_imported_from_registry():
    """No hardcoded threshold values in triage/layer_health/issues — all from registry."""
```

This test suite runs in CI and **fails if someone adds a metric to one surface without
adding it to the registry.**

### 6g. What NOT to Align

Some differences are intentional and correct:

- **Triage alerts at 75% of gate** — this is an early warning, not a contradiction. Registry encodes both `gate` and `triage_alert` explicitly.
- **layer_health is per-frame, summary is aggregate** — different granularity is fine if they agree directionally
- **Triage has code pointers** — these are diagnostic, not scoring. Don't need to match summary exactly
- **Issues tab detects frame-level events, summary scores aggregates** — an issue can exist without a penalty (e.g., a single jitter frame). The `issue_severity` field in the registry controls this.

---

## Execution Order

| Step | Description | Depends On | Effort |
|---|---|---|---|
| **1** | Camera Fix A: Shader prewarming | — | 1h |
| **2** | Camera Fix C: Async in GT sync mode | — | 15m |
| **3** | Camera Fix B: Double buffering | — | 2h |
| **4** | Validation runs: s-loop + highway | Steps 1-3 | 30m |
| **5** | Summary tab red metric investigation (Part 3) | Step 4 recordings | 2h |
| **6** | Highway trajectory investigation (Part 4) | Step 4 highway recording | 1h |
| **7** | Implement summary metric fixes (Part 5) | Steps 5-6 | 2-3h |
| **8** | Create scoring_registry.py (Part 6e) | — | 2h |
| **9** | Triage engine fixes (Part 6a: T1-T7) | Step 8 | 2h |
| **10** | Layer health alignment (Part 6b: L1-L7) | Step 8 | 3-4h |
| **11** | Issues tab alignment (Part 6c: I1-I5) | Step 8 | 1h |
| **12** | Compare tab fixes (Part 6d: D1-D3) | Step 8 | 2-3h |
| **13** | Consistency test suite (Part 6f) | Steps 8-12 | 1h |
| **14** | Final validation runs + regression check | Steps 7-13 | 30m |

Steps 1-3 (camera) and 8-12 (diagnostic consistency) can run in parallel since they
touch completely different parts of the stack.

---

## Success Criteria

### Camera Pipeline
- [ ] No cadence gaps > 100ms in first 300 frames
- [ ] Steady-state dt P95 < 40ms

### Summary Tab Accuracy
- [ ] All "fake red" metrics show N/A or corrected values
- [ ] All "real red" metrics either fixed or have documented action plan
- [ ] Highway trajectory score ≥ 90 (lateral error RMSE ≤ 0.20m)
- [ ] S-loop score ≥ 94 (no regression)

### Diagnostic Consistency
- [ ] `scoring_registry.py` exists with all ~30 scored metrics
- [ ] Triage: no dead patterns, all patterns have metric extraction
- [ ] Triage: rate_key_map correct for all patterns
- [ ] Issues: no "high" severity issues without corresponding summary penalty
- [ ] Issues: thresholds match summary (or registry documents intentional difference)
- [ ] Layers tab: 6 layers matching Summary tab layer names
- [ ] Layers tab: lateral error threshold aligned with Summary scoring
- [ ] Layers tab: directionally consistent (red summary → red frames in layers)
- [ ] Compare tab: 3-column layout (baseline raw / current raw / delta)
- [ ] Compare tab: all gate metrics visible, layer scores group present
- [ ] Consistency test suite passes (test_diagnostic_consistency.py)

### Regression
- [ ] All existing tests pass (25/25 PhilViz + comfort gates)
- [ ] No new regressions in full test suite
