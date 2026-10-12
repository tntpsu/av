# Scenario Registry

Single source of truth mapping a scenario name to its **track YAML**, **config overlay**, and
**duration**. `/run`, `/e2e`, and `/diagnose` resolve `$ARGUMENTS` through this file. The nightly
`/acc-sweep` Step 2.5 re-seed (`/e2e tracks/scenarios/<name>.yml`) also depends on it.

**Why this file exists (2026-10-12, T-ACC-H8-OVERLAY-DROP).** It was missing for months; `/e2e` fell
back to "convention" and on Night-61 launched the fresh H8 and G1 re-seeds with **no `--config` overlay**.
H8 then ran at the base 12 m/s target instead of its scenario 15 m/s, could not close on a lead that
starts beyond radar range, never engaged ACC, and the composite collapsed to 80.9 (a FALSE regression that
looked like the curvature-sign promotion but was an overlay-application bug — the recording's
`runtime_config_json` showed `target_speed 12.0` and no overlay path). G1 survived only because grade
following does not need the closing speed. **An ACC scenario MUST be launched with both its track YAML and
its ACC overlay.** Verify after any re-seed: the recording's merged config must show the overlay's
`control.longitudinal.target_speed`, not the base 12.0.

---

## Launch command

```bash
./start_av_stack.sh --skip-unity-build-if-clean \
  --config config/<overlay>.yaml \
  --track-yaml <track-yaml> \
  --duration <seconds>
```

- For ACC scenarios the track YAML **is** the scenario file under `tracks/scenarios/` (it carries the
  base road geometry plus the `lead_vehicle` block); the overlay carries the ACC block and the scenario
  speed target. Both are required — omitting `--config` silently runs the base 12 m/s stack.
- For the lateral base tracks, `--config` is omitted on `hill_highway` (auto-derived q_lat); the others
  take no overlay either unless a specific experiment config is named.

---

## ACC scenarios

| Scenario | Aliases | Track YAML | Config overlay | Duration | Premise (target / lead) |
|---|---|---|---|---|---|
| `autobahn_a1_steady` | a1, autobahn steady | `tracks/scenarios/autobahn_a1_steady.yml` | `config/acc_autobahn.yaml` | 120 | ego 12 / lead 11.5 steady |
| `autobahn_a2_hard_brake` | a2, autobahn brake | `tracks/scenarios/autobahn_a2_hard_brake.yml` | `config/acc_autobahn.yaml` | 120 | ego 12 / lead 20→brake |
| `highway_h2_steady` | h2, steady | `tracks/scenarios/highway_h2_steady.yml` | `config/acc_highway.yaml` | 90 | lead 12 steady |
| `highway_h3_hard_brake` | h3, hard brake | `tracks/scenarios/highway_h3_hard_brake.yml` | `config/acc_highway.yaml` | 90 | lead 12→brake |
| `highway_h4_accel_away` | h4, accel away | `tracks/scenarios/highway_h4_accel_away.yml` | `config/acc_highway.yaml` | 120 | lead 8→accel |
| `highway_h5_stop_go` | h5, stop go | `tracks/scenarios/highway_h5_stop_go.yml` | `config/acc_highway.yaml` | 90 | lead 8 stop-and-go |
| `highway_h6_close_gap` | h6, close gap | `tracks/scenarios/highway_h6_close_gap.yml` | `config/acc_highway.yaml` | 120 | lead 5 close gap |
| `highway_h7_straight_catchup` | h7, straight catchup | `tracks/scenarios/highway_h7_straight_catchup.yml` | `config/acc_highway.yaml` | 120 | ego 15 / lead 10, straight |
| `highway_h8_curve_catchup` | h8, curve catchup | `tracks/scenarios/highway_h8_curve_catchup.yml` | `config/acc_highway.yaml` | 90 | **ego 15 / lead 10, engages near R500 S-curve** |
| `highway_h9_adjacent_lane_parallel_reject` | h9, adjacent reject | `tracks/scenarios/highway_h9_adjacent_lane_parallel_reject.yml` | `config/acc_highway.yaml` | 90 | adjacent-lane lead, must NOT engage |
| `highway_h10_oncoming_straight_reject` | h10, oncoming reject | `tracks/scenarios/highway_h10_oncoming_straight_reject.yml` | `config/acc_highway.yaml` | 90 | oncoming lead, must NOT engage |
| `highway_h11_freeflow_mild_curve` | h11, freeflow | `tracks/scenarios/highway_h11_freeflow_mild_curve.yml` | `config/acc_highway.yaml` | 90 | lateral-only, no lead in range |
| `hill_g1_grade_following` | g1, grade following | `tracks/scenarios/hill_g1_grade_following.yml` | `config/acc_hill_highway.yaml` | 90 | lead 5.5 up/down grade |
| `hill_g2_stop_on_grade` | g2, stop on grade | `tracks/scenarios/hill_g2_stop_on_grade.yml` | `config/acc_hill_highway.yaml` | 90 | lead 10→stop on grade |

`config/acc_highway.yaml` sets `control.longitudinal.target_speed: 15.0`; `config/acc_autobahn.yaml`
and `config/acc_hill_highway.yaml` set their own. The overlays are the minimal 2026-10-09 form (ACC block
+ scenario speed keys, no `_inherits`); the archived full stacks are under `config/legacy/`.

---

## Lateral base tracks

| Track | Aliases | Track YAML | Config overlay | Duration |
|---|---|---|---|---|
| `s_loop` | sloop, s loop | `tracks/s_loop.yml` | (none) | 60 |
| `highway_65` | highway, hwy, 65 | `tracks/highway_65.yml` | (none) | 60 |
| `hairpin_15` | hairpin, 15 | `tracks/hairpin_15.yml` | (none) | 60 |
| `sweeping_highway` | sweeping | `tracks/sweeping_highway.yml` | (none) | 60 |
| `mixed_radius` | mixed | `tracks/mixed_radius.yml` | (none) | 60 |
| `hill_highway` | hill, hill highway | `tracks/hill_highway.yml` | (none — auto-derived q_lat) | 60 |

Durations are the values that have been producing converging, gate-valid recordings. A scenario that needs
more convergence time (the catch-up / close-gap / accel-away / autobahn cases) is set to 120 s; the steady
and reject cases are 90 s. Change a duration only with a reason, and never compare a detection rate or a
composite across two different durations without stating both.
