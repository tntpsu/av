# Plan: Wire Map-Based Arc Distance into Curve Phase Scheduler

**Date:** 2026-03-09
**Status:** Ready to implement
**Motivation:** Replace the `time_start_s=1.2` workaround (Workstream C1 fix) with the correct
architecture — true track arc distance fed into the scheduler, so the gate stays closed on
straights by geometry, not by an empirically tuned time threshold.

---

## Background: Two Distance Systems

The scheduler's gate weight (`local_gate_weight`) controls when ENTRY state can fire. It is
driven by **two independent distance signals**:

| System | Source | s_loop S2 value | Used for |
|---|---|---|---|
| **A (active)** | Lane polynomial `get_curve_context()` | ~9m (sees s_loop geometry everywhere) | `distance_to_curve_start_m` → `local_distance_term`, `local_time_gate_term` |
| **B (bypassed)** | Track YAML arc arithmetic in `_compute_reference_entry_preview()` | 60m→0m (correct) | Same fields — but only runs when a track is loaded |

System B already exists and works. It is bypassed because:

```yaml
# mpc_sloop.yaml line 389:
reference_lookahead_entry_track_name: ''   # ← empty string, NOT absent key
```

`orchestrator.py:1095-1100`:
```python
self._reference_entry_track_name = str(
    trajectory_cfg.get(
        'reference_lookahead_entry_track_name',
        lateral_cfg.get('curve_phase_track_name', ''),  # fallback to 's_loop'
    )
).strip()
```

Because `''` is a **present** value, `dict.get(key, fallback)` returns `''` — the fallback to
`curve_phase_track_name: s_loop` never fires. `_load_reference_entry_track_profile('')` returns
immediately at the `if not track_name: return` guard.

---

## Why System A Causes S2 Latch

On s_loop, the lane polynomial has limited look-ahead. From anywhere on S2, it detects the
nearest lane curve at ~9m (constant). At cap-tracked speed ≈ 4.5 m/s:

```
time_to_curve ≈ 9m / 4.5 m/s = 2.0s
```

With `curve_local_phase_time_start_tight_s: 2.5s` (the Phase B default before C1 fix):
- 2.0s < 2.5s → time gate OPEN throughout all 160 frames of S2
- Result: 36.7% `curve_local_active_on_straights`

The C1 fix (`time_start_tight: 2.5 → 1.2s`) works because:
- 2.0s > 1.2s → time gate closed
- But this is a fragile assumption about constant polynomial distance on s_loop

---

## What System B Gives Us

With `reference_lookahead_entry_track_name: s_loop`, the loader parses `tracks/s_loop.yml`:

```
S1: straight 25m
C1: arc r=40, 90° right  → len=62.8m, κ=0.0250, sign=-1  (start=25.0,  end=87.8)
C2: arc r=50, 90° left   → len=78.5m, κ=0.0200, sign=+1  (start=87.8,  end=166.4)
S2: straight 60m
C3: arc r=40, 90° left   → len=62.8m, κ=0.0250, sign=+1  (start=226.4, end=289.2)
C4: arc r=40, 90° right  → len=62.8m, κ=0.0250, sign=-1  (start=289.2, end=352.0)
S3: straight 100m
Total: 452.0m
```

Arc-distance behavior on S2 (cursor 166.4m → 226.4m):

| Position on S2 | dist to C3 | time @4.5m/s | time gate (1.2s)? | distance gate (10m)? |
|---|---|---|---|---|
| Entry (166.4m) | 60.0m | 13.3s | CLOSED ✓ | CLOSED ✓ |
| Mid (196.4m) | 30.0m | 6.7s | CLOSED ✓ | CLOSED ✓ |
| 10m before C3 (216.4m) | 10.0m | 2.2s | CLOSED ✓ | OPENS (desired!) |
| 5m before C3 (221.4m) | 5.0m | 1.1s | OPENS | OPENS |

The gate stays closed for the full straight, then opens exactly as the car approaches C3.
This is the correct behavior — by geometry, not by empirical threshold.

---

## What Changes

### Side effects to understand

Once System B is active, three outputs change:

1. **`distance_to_curve_start_m`**: changes from ~9m (flat, polynomial) to a 60m→0m ramp on S2.
   This is the primary fix — it drives both `local_distance_term` and `local_time_gate_term`.

2. **`preview_curvature_abs/signed`** (orchestrator.py:1601): changes from lane-polynomial κ to
   YAML track κ (`0.0250` for C1/C3/C4, `0.0200` for C2). Track κ is above the
   `reference_lookahead_entry_preview_curvature_min: 0.003` threshold ✓

3. **`in_curve`** flag (orchestrator.py:1569): changes from polynomial-based curve detection to
   arc-segment membership. Inside any arc segment, `in_curve=True` and
   `distance_to_curve_start_m=0.0` is returned immediately — `local_gate_weight = 1.0`.

4. **`source` field in HDF5**: changes from `poly` to `track`.

5. **Odometry**: track position uses `_track_odometer_m` (dead-reckoning from
   `start_distance: 3.0m` in s_loop.yml). No Unity `roadCenterReferenceT` needed — already
   implemented.

### Disable the time gate explicitly

With arc distance, `time_to_curve_s` on S2 is ≥ 6.7s — the time gate is irrelevant. Rather than
leaving it at 1.2s as a "harmless workaround", disable it cleanly:

```yaml
curve_local_phase_time_start_s: 0.0        # disables: start(0.0) ≤ end(0.25) → time_weight=0
curve_local_phase_time_start_tight_s: 0.0  # same
```

`_compute_local_curve_relevance_terms()` has the guard `if start_s > end_s:` — with `start=0.0`
and `end=0.25` (default), the block never runs, producing `time_weight=0.0` unconditionally.

The distance gate (`curve_local_phase_distance_start_m: 10.0`, driven by arc distance) becomes the
sole mechanism. This is explicit and communicates intent clearly.

---

## Implementation — Config Changes Only

### Change 1 — `config/mpc_sloop.yaml`

```yaml
# Wire map arc distance (line 389):
reference_lookahead_entry_track_name: s_loop    # was ''

# Disable time gate — distance gate is now the sole trigger (lines ~408-409):
curve_local_phase_time_start_s: 0.0             # was 1.2
curve_local_phase_time_start_tight_s: 0.0       # was 1.2
```

### Change 2 — `config/mpc_highway.yaml`

```yaml
# Wire map arc distance (line 343):
reference_lookahead_entry_track_name: highway_65  # was ''

# Disable time gate:
curve_local_phase_time_start_s: 0.0
curve_local_phase_time_start_tight_s: 0.0
```

**No code changes required.** `_load_reference_entry_track_profile()` already parses the YAML
track files and computes arc lengths per segment. `_compute_reference_entry_preview()` track
branch is already complete. `trajectory/utils.py` scheduler is unchanged.

---

## Testing Protocol

### Step 1 — Unit tests (no Unity)

```bash
pytest tests/test_reference_lookahead.py -v
```

Confirm all pass. These exercise `_compute_reference_entry_preview()` directly.

### Step 2 — Single run diagnostic

Run one 60s s_loop run with the change and inspect HDF5:

```bash
python tools/analyze/analyze_drive_overall.py --latest
```

Check in HDF5:
- `reference_lookahead_source` field = `track` (not `poly`)
- `distance_to_next_curve_start_m` on S2 frames = ramp 60m→10m (not ~9m flat)
- `curve_local_gate_weight` on S2 = low (≤ 0.1) throughout middle of S2

### Step 3 — A/B batch (5 runs each)

```bash
python tools/analyze/run_ab_batch.py \
  --config-a config/mpc_sloop.yaml \
  --config-b config/mpc_sloop_mapwired.yaml \
  --track-yaml tracks/s_loop.yml \
  --runs 5
```

Key metrics to check:

| Metric | Current (C1 result) | Target |
|---|---|---|
| `curve_local_active_on_straights` | ~15.6% | ≤ 15% |
| S2 latch specifically | ~10.8-11.2% | ≤ 11% |
| C1/C3 peak lateral (P95) | 0.572-0.612m | ≤ 0.612m |
| Score | 97.1 | ≥ 97 |
| E-stops | 0 | 0 |

### Step 5 — Highway non-regression

```bash
python tools/analyze/run_ab_batch.py \
  --config-a config/mpc_highway.yaml \
  --config-b config/mpc_highway_mapwired.yaml \
  --track-yaml tracks/highway_65.yml \
  --runs 3
```

Verify score ≥ 98, 0 e-stops.

---

## Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Odometry drift shifts gate timing | Low-medium | `start_distance: 3.0m` pre-seeded. Drift is slow at 4.5 m/s. Check `track_distance_m` in HDF5 across a lap. |
| `preview_curvature_abs` change alters `term_preview` | Low | New κ (0.025) ≈ lane poly κ. Gate timing dominated by distance/time. |
| `in_curve=True` inside arcs changes sustain signal | Expected, desired | COMMIT mode should be active during arcs anyway. |

---

## Rollback

Revert by setting:
```yaml
reference_lookahead_entry_track_name: ''
curve_local_phase_time_start_s: 1.2
curve_local_phase_time_start_tight_s: 1.2
```

---

## Notes on Base Config

`config/av_stack_config.yaml` keeps `reference_lookahead_entry_track_name: ''` as default — this
is correct. The base config is track-agnostic. Only per-track overlays specify the track name.
