# Plan: Config Unification — Track-Agnostic Base Config

**Date:** 2026-03-09
**Status:** Ready to plan
**Problem:** Per-track config overlays contain dozens of manually tuned parameters.
Adding a new track requires re-tuning all of them. The system is not robust.

---

## Root Cause Analysis

Examining the diffs between `av_stack_config.yaml`, `mpc_sloop.yaml`, and `mpc_highway.yaml`,
the per-track differences fall into four categories:

### Category A — Speed-dependent (should be unified tables in base)

These parameters have different values per track solely because the tracks run at different
speeds. s_loop cap-tracks to 4–7 m/s; highway runs at 12–15 m/s. A single table covering
0–20 m/s would serve both tracks with no per-track override.

| Parameter | Base | s_loop | Highway |
|---|---|---|---|
| `reference_lookahead_speed_table` | Up to 13.4 m/s | Low-speed values | Extends to 15+ m/s |
| `reference_lookahead_speed_table_entry` | Partial | Low-speed values | Not overridden |
| `reference_lookahead_speed_table_commit` | **Missing** | Defined | Not overridden |
| `local_curve_reference_target_distance_table` | **Missing** | Defined | Not overridden |
| `pp_curve_local_shorten_slew_m_per_frame` | 0.08 | 0.15 | Not overridden |
| `reference_lookahead_slew_rate_shorten_m_per_frame` | 0.15 | 0.35 | 0.35 |
| `mpc_horizon` | 20 steps | 30 steps | Not overridden |

### Category B — Curvature-dependent (should be derived from track geometry)

These threshold parameters differ because s_loop has tight curves (r=40–50m, κ=0.02–0.025) and
highway has gentle curves (r≈500m, κ=0.002). They should be computed at startup from the loaded
track YAML, not hand-tuned per config.

| Parameter | Base/s_loop | Highway | Physical meaning |
|---|---|---|---|
| `curve_phase_preview_curvature_min/max` | 0.003/0.015 | 0.0015/0.008 | Detection range for curve preview |
| `curve_phase_path_curvature_min/max` | 0.003/0.015 | 0.0015/0.008 | Path curvature gate |
| `curve_local_entry_severity_curvature_min/max` | 0.003/0.012 | 0.0015/0.008 | Severity ramp range |
| `curve_context_curve_start_curvature` | 0.006 | 0.0015 | What counts as "entering a curve" |
| `curve_local_commit_distance_ready_m` | 3.0 | 8.0 (s_loop) | COMMIT activation distance |
| `curve_local_phase_distance_start_m` | 8.0 | 10.0 (s_loop) | Gate open distance |
| `curve_phase_on/off/commit_on` thresholds | 0.45/0.3/0.55 | 0.35/0.22/0.45 | Phase transition thresholds |

### Category C — Architecture flags (should be promoted to base defaults)

These are "graduated capability" flags that were off-by-default while being tested, but are now
validated and should be the universal default.

| Parameter | Base | s_loop | Highway | Action |
|---|---|---|---|---|
| `curve_scheduler_mode` | `phase_shadow` | `phase_active` | Not overridden | Promote `phase_active` to base |
| `control.regime.enabled` | `false` | `true` | `true` | Promote `true` to base |
| `pp_curve_local_floor_state_min` | `ENTRY` | `COMMIT` | Not overridden | Promote `COMMIT` to base |
| `mpc_curvature_preview_enabled` | unset | `true` | `true` | Add `true` to base |
| `reference_lookahead_phase_active_*` | Missing | Defined | Not overridden | Add to base |
| `local_curve_reference_*` (full section) | Missing | Defined | Not overridden | Add to base |
| `curve_local_phase_time_start_s` | 1.2 | 0.0 | Not overridden | Base → 0.0 |

### Category D — Legitimately per-track (keep in overlays, nothing to do)

| Parameter | Why it's track-specific |
|---|---|
| `reference_lookahead_entry_track_name` | By definition |
| `target_speed` / `max_speed` | Track speed limit |
| `steering_smoothing_alpha` | Highway oscillation-specific physics fix |
| `pp_feedback_gain: 0.0` | Highway-specific PID suppression |
| `curve_feedforward_gain: 0.0` | Highway straight — disables noise amplification |
| `pp_speed_norm_enabled/min_scale` | Highway-specific gain normalization |
| `speed_drag_gain` / `accel_target_smoothing_alpha` | Highway longitudinal tuning |
| MPC weights (`q_lat`, `r_steer_rate`) | Genuinely speed/geometry dependent |

---

## Target State

After this work, a **new track config should require only:**

```yaml
# tracks/new_track.yaml — the only mandatory per-track override
trajectory:
  reference_lookahead_entry_track_name: new_track   # wire arc distance

control:
  trajectory:
    target_speed: X.X                                # track speed limit
```

All curvature thresholds, lookahead distances, slew rates, and phase parameters are derived
automatically from track geometry and operating speed.

---

## Implementation Plan

### Phase 1 — Promote architecture flags to base (config-only, low risk)

**What:** Move all Category C flags into `av_stack_config.yaml`.

**Changes to `av_stack_config.yaml`:**
- `curve_scheduler_mode: phase_shadow → phase_active`
- `control.regime.enabled: false → true`
- `pp_curve_local_floor_state_min: ENTRY → COMMIT`
- `curve_local_phase_time_start_s: 1.2 → 0.0` / `_tight_s: 1.6 → 0.0`
- Add `reference_lookahead_phase_active_*` params (copy from s_loop)
- Add `local_curve_reference_*` full section (copy from s_loop)
- Add `mpc_curvature_preview_enabled: true` / `mpc_curvature_preview_gain: 1.0`

**Remove from `mpc_sloop.yaml`:** all params now in base (they inherit automatically).

**Validation:** A/B batch on s_loop (≥5 runs) comparing current vs promoted base. Score must
stay ≥ 97. Then A/B on highway (≥3 runs), score ≥ 98.

**Risk:** Low. These flags are already validated on both tracks. The base config gains them;
overlays that don't set them start inheriting the correct value.

---

### Phase 2 — Unified speed tables in base (config-only, medium risk)

**What:** Merge s_loop and highway speed tables into single coherent 0–20 m/s tables in base.
Remove per-track speed table overrides.

**Design principle:** Each table entry should represent the physically correct lookahead for that
speed. At low speed (s_loop range), the values from the s_loop overlay apply. At high speed
(highway range), the highway values apply. Interpolation handles everything in between.

**Tables to unify:**

1. `reference_lookahead_speed_table` (primary lookahead)
   - s_loop values at 0–7 m/s → low end of merged table
   - highway values at 12–17 m/s → high end of merged table
   - Linear interpolation through 7–12 m/s gap

2. `reference_lookahead_speed_table_entry` (entry-mode lookahead)
   - Same approach

3. `reference_lookahead_speed_table_commit` (commit-mode lookahead)
   - Currently only in s_loop; add to base covering 0–20 m/s

4. `local_curve_reference_target_distance_table` (reference point distance)
   - Currently only in s_loop; extend to cover highway speeds

**Validation:** Run both tracks using ONLY the base config + track name override. No table
overrides in per-track configs. A/B vs current on each track.

**Risk:** Medium. Table merging requires care at the crossover region (7–12 m/s). May need 1–2
tuning iterations.

---

### Phase 3 — Auto-derive curvature thresholds from track geometry (code + config)

**What:** Compute curvature thresholds at startup from the loaded track YAML, rather than
specifying them as magic numbers in the config.

**How it works:**

The track YAML already has all the information needed:
```yaml
# s_loop: min_radius=40m → κ_max=0.025, highway: min_radius≈500m → κ_max=0.002
segments:
  - type: arc
    radius: 40
    ...
```

`_load_reference_entry_track_profile()` already iterates these segments. Extend it to also
compute:
- `track_curvature_min` = min κ across all arc segments
- `track_curvature_max` = max κ across all arc segments
- `track_radius_min` = minimum arc radius

Then derive the threshold parameters from these values at config-load time:

```python
# curve detection range: covers track's actual curvature range with margin
curvature_min = track_curvature_min * 0.3   # detect curves 30% gentler than track min
curvature_max = track_curvature_max * 1.2   # saturate at 20% above track max

# commit distance: should be ~20% of tightest curve arc length
# arc_len = r * θ. For 90° turn at r=40: arc_len=62.8m. 20% = 12.5m.
# For 90° turn at r=500: arc_len=785m. 20% = 157m → cap at 15m.
commit_distance = min(15.0, 0.2 * track_radius_min * math.pi / 2)

# gate open distance: ~40% of commit distance, minimum 5m
distance_start = max(5.0, commit_distance * 0.4)
```

**These derived values become runtime defaults** — they apply when the corresponding YAML key
is absent from the config. Explicit config values override them (escape hatch for edge cases).

**New config keys to add (all optional, auto-derived if absent):**
```yaml
curve_auto_derive_thresholds: true   # enable/disable auto-derivation (default: true)
```

**Impact:** `curve_phase_preview_curvature_min/max`, `curve_phase_path_curvature_min/max`,
`curve_local_entry_severity_curvature_min/max`, `curve_context_curve_start_curvature`,
`curve_local_commit_distance_ready_m`, `curve_local_phase_distance_start_m`,
`curve_phase_on/off/commit_on` thresholds can all be auto-derived or validated against the
derived values with a warning if they diverge significantly.

**Files to change:**
- `av_stack/orchestrator.py` — `_load_reference_entry_track_profile()` + new
  `_derive_curvature_thresholds()` method
- `trajectory/utils.py` — `compute_curve_phase_scheduler()` reads derived values if config
  keys are absent

**Validation:** Run on both tracks without any curvature threshold keys in per-track configs.
Compare scheduler gate timing (HDF5 `curve_local_gate_weight`) and phase onset against current
manually-tuned values.

**Risk:** Medium-high. Requires code changes. Safeguard: auto-derived values are only used when
explicit config keys are absent, so rollback = add the explicit keys back.

---

### Phase 4 — Slim down per-track overlays (cleanup)

After Phase 1–3, remove every parameter from per-track configs that is now correctly handled
by the base + auto-derivation. Each overlay should shrink to ~20 lines.

**`mpc_sloop.yaml` after cleanup:**
```yaml
# s_loop track config
trajectory:
  reference_lookahead_entry_track_name: s_loop

control:
  trajectory:
    target_speed: 8.0   # or whatever cap-tracked speed target
```

**`mpc_highway.yaml` after cleanup:**
```yaml
# highway_65 track config
trajectory:
  reference_lookahead_entry_track_name: highway_65

control:
  lateral:
    steering_smoothing_alpha: 0.3      # highway oscillation fix
    pp_feedback_gain: 0.0              # suppress PID wind-up at highway speed
    curve_feedforward_gain: 0.0        # disable noise amplification on straight
    pp_speed_norm_enabled: true
    pp_speed_norm_min_scale: 0.60
  trajectory:
    target_speed: 15.0
    max_speed: 18.0
  mpc:
    mpc_q_lat: 0.5                     # highway-speed MPC tuning
    mpc_r_steer_rate: 2.0
```

**Validation:** Full regression on both tracks. All comfort gates green.

---

## Sequencing and Dependencies

```
Phase 1 (flags)       → safe to do now, independent
     ↓
Phase 2 (tables)      → depends on Phase 1 being stable
     ↓
Phase 3 (auto-derive) → depends on Phase 2; requires code changes
     ↓
Phase 4 (cleanup)     → depends on Phase 3 validated
```

Each phase is independently committable and independently reversible.

---

## What a New Track Needs After This Work

1. Create `tracks/new_track.yml` (arc/straight segments with radii)
2. Create minimal config overlay (track name + target speed)
3. Run 3 laps, check gates — done

No curvature threshold tuning. No lookahead table adjustment. No slew rate tweaking.
The system reads the track geometry and configures itself.

---

## Files Changed

| Phase | Files |
|---|---|
| 1 | `config/av_stack_config.yaml`, `config/mpc_sloop.yaml`, `config/mpc_highway.yaml` |
| 2 | `config/av_stack_config.yaml`, `config/mpc_sloop.yaml`, `config/mpc_highway.yaml` |
| 3 | `av_stack/orchestrator.py`, `trajectory/utils.py`, `config/av_stack_config.yaml` |
| 4 | `config/mpc_sloop.yaml`, `config/mpc_highway.yaml` |
