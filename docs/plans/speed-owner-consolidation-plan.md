# Speed-owner consolidation — design note (2026-10-11, roadmap item 2)

**Status:** design note, no code. Each phase below lands behind a kill-switch and is promoted only
by A/B (≥ 5 runs) with the acceptance tests in §6. **Owner:** T-SPEED-OWNER-CONSOLIDATION.

## 1. Why

The longitudinal command has five owners that each hold their own copy of the same physics, and
the last month's defects all lived in the seams between them:

| defect | seam |
|---|---|
| Launch frozen at 3 m/s for ~12 s (T-LAUNCH-CURVATURE-SEED) | comfort governor fed by the curvature EMA; the March overlays "fixed" it by switching two owners off |
| Curve cap latched at target − 0.4 on R500 (T-GOVERNOR-CURVE-CAP-LATCH) | curve cap vs target precedence |
| G1 gap bias +1.8 → +6.5 m, G2 TTC 2.45 → 2.2 s, pedal alternations 2× (T-ACC-EQ-BIAS) | IDM accel vs accel-tracking PID: two owners of one acceleration |
| SpeedCompliance 31 (s_loop) / 40 (hill_highway) | velocity profile built with a 0.05 g lateral budget — a controller-capability proxy, not a comfort or tire limit |

The scorer now measures the consequence (SpeedCompliance, convergence time, engage edges, pedal
chatter), so this work has acceptance tests for the first time.

## 2. The owners today (measured, not from docs)

Order inside `SpeedGovernor.compute_target_speed()` (`control/speed_governor.py`), fed by
`AVStack._pf_run_speed_governor()`:

| # | owner | lateral-accel budget it encodes | longitudinal limits it encodes | notes |
|---|---|---|---|---|
| 0 | `trajectory.target_speed` 12 m/s + map speed limit via `_apply_speed_limits` (`max_lateral_accel` bins **3.0 m/s² = 0.31 g**, `min_curve_speed` 3.0) | 0.31 g | slew 2.5 / 1.5 m/s², preview decel 2.5 | the ceiling everyone min()s against |
| 1 | Comfort governor: √(0.2 g / (2.5·κ_smoothed)), 5-frame max-hold, floor 3.0 | **0.08 g effective** (0.2 g ÷ `curvature_calibration_scale` 2.5) | — | reacts to the curvature EMA (12 m window) |
| 1b | **Velocity profile** `trajectory/velocity_profiler.py`: v(s) = √(a_lat/κ) from the map, backward 1.8, forward 2.0 m/s², clamp [3.0, target] | **0.05 g** (`a_lat_tracking_budget_g`) | 2.0 / 1.8 | binds 97–100 % of under-utilised time on s_loop and hill_highway today |
| 2 | Curve preview: √(0.2 g/(2.5·κ_preview)) + 1.2 m/s² braking distance | 0.08 g | 1.2 | same physics as the profile's backward pass, recomputed reactively |
| 3 | Perception-horizon guardrail | — | — | legitimate: do not outrun what you can see |
| 3b | Curve cap: turn-feasibility estimator, peak **0.26 g**, margin 0.4, floor 3.0, 1.6 m/s² | 0.26 g (+ guardband) | 1.6 | now `only_when_binding`; also *fires curve-intent / lookahead-shortening / feedforward-timing signals into the lateral loop* |
| 3c | Feasibility backstop; curve-intent speed guardrail cap 7.0 m/s | — | — | |
| 4 | Speed planner `trajectory/speed_planner.py`: jerk-limited tracker, **max_accel 1.2**, max_decel 2.4–2.5, jerk 0.5–1.0, launch floor 2.0 | — | 1.2 / 2.5 | the launch ramp: 12 m/s at ~11 s on every track |
| 5 | ACC/IDM → `_pf_resolve_longitudinal_target`: `min(governor, acc_target)` as a **speed**, *and* IDM accel routed as `reference_accel` (decel only unless `acc_idm_accel_routing_positive`) | — | IDM 2.0 / 2.5 (1.5 on hill) | the accel-tracking PID then tracks whichever accel it was handed |
| 6 | Longitudinal controller: speed PID + accel feedforward/tracking, overspeed brake, grade FF, jerk cooldown; `safety.max_speed` clip | — | its own `max_accel`/`max_decel` | |

**Five lateral-accel budgets** (0.05, 0.08, 0.26, 0.31 g, plus the regime selector's 1.5 m/s² = 0.15 g)
and **four longitudinal-accel sets** (profile 2.0/1.8, planner 1.2/2.5, preview 1.2, curve cap 1.6,
IDM 2.0/2.5) for one car. The binding one on the slow tracks is the profile's 0.05 g: on R40 it
allows 4.4 m/s, on R100 7.0 m/s — exactly s_loop's 4.1 and hill_highway's 6.2 m/s medians.

**The 0.05 g is a proxy.** It was calibrated (2026-04-10) as "the controller's closed-loop tracking
capability", i.e. the speed at which Pure Pursuit stays inside its error budget. That encodes a
lateral-controller weakness as a speed policy, and it is why the car is scored "comfortable" while
driving at half the limit. The honest decomposition: the *profile* carries the passenger comfort
budget; the *lateral loop* must hold the lane at that speed, and where it cannot, that is roadmap
item 3's defect to fix, reported as lateral error — not hidden by slowing down.

**The lateral coupling (why April's "replace the chain" attempt failed, 97 → 79 with OOL):** curve
preview and curve cap do not only cap speed — they drive `curve_intent_state`, lookahead
shortening and feedforward timing. Turning them off removed lateral signals, not just speed caps.
Any consolidation must first move those signals onto the map (the `lookahead_profiler` already
computes Ld from map geometry; distance-to-curve and κ-ahead are available) so that speed owners
have **no lateral side-effects**. That decoupling is Phase 2 and the reason this is architecture,
not tuning.

## 3. Target architecture

```
map geometry ─┐
speed limits ─┼─► VelocityProfile v*(s), a*(s)   (one lateral budget, one a_lon pair; rebuilt on constraint)
perception horizon (runtime) ─────────┐
IDM (v_idm, a_idm) when lead owned ───┼─► Arbiter: (v_ref, a_ref) = the lower-speed owner, carrying ITS accel
                                        │   owner ∈ {profile, horizon, acc, safety}; one `active_owner` string
                                        ▼
                     Jerk-limited tracker (the speed planner, limits = the profile's a_lon pair)
                                        ▼
                     Longitudinal controller: tracks (v_ref, a_ref) — one accel path, both signs
                                        ▼
                     safety.max_speed clip, e-stop
lateral loop ◄── curve phase / lookahead / FF timing from MAP geometry (not from speed owners)
```

Deleted as owners (kept as *fallbacks* only while `curvature_primary_source != map_track`):
comfort governor, curve preview, curve cap, feasibility backstop, curve-intent speed guardrail.
Deleted outright: `curvature_calibration_scale` (a hidden second budget), `speed_drag_gain` (already 0),
the decel-only asymmetry of IDM routing, the planner's private `max_accel`.

One budget: `trajectory.lateral_accel_budget_g` (comfort), consumed by the profile, the fallback
governor and the curvature bins. One a_lon pair: `trajectory.a_lon_accel_mps2 / a_lon_brake_mps2`,
consumed by the profile's passes and the tracker. One lateral tracking budget stays where it
belongs: `lookahead_profiler.e_target_m` (shapes Ld, not speed).

## 4. Phases (each flagged, each A/B'd)

**P0 — instrumentation (done):** `speed_governor_active_limiter`, utilisation binding-cap attribution,
SpeedCompliance layer, ACC convergence time / edge terms.

**P1 — one lateral budget.** Introduce `trajectory.lateral_accel_budget_g`; profile, comfort fallback
and bins read it; delete `curvature_calibration_scale`. Promote in steps 0.08 → 0.12 → 0.16 → 0.20 g
(profile speed on R40: 5.6 → 6.9 → 7.9 → 8.9 m/s; R100: 8.9 → 10.9 → 12), each step a 6-track A/B.
Guard: out-of-lane 0, e-stops 0, at-car lateral RMSE reported per track (s_loop/hairpin **will** rise
toward the PP ceiling — report it as the item-3 defect, do not back off the budget to hide it unless
OOL/e-stop fire). Expected: hill_highway SpeedCompliance 40 → ≥ 85 by 0.16 g; s_loop 31 → 60–75.

**P2 — decouple lateral signals from speed owners.** Route `curve_intent_state`, lookahead shortening
and FF timing from map geometry (`lookahead_profiler`, distance-to-curve). Then switch preview /
curve cap / comfort to fallback-only when the map is primary. A/B: lateral RMSE and P95 unchanged
on all six tracks with the three owners off — the test April never passed.

**P3 — one acceleration path.** Arbiter emits (v_ref, a_ref); planner limits = profile a_lon; IDM
accel routed symmetrically and tracked by the same PID. Offline first: re-fit the ACC harness plant
to the production law (T-ACC-HARNESS-ACCEL-TRACKING-PLANT), reproduce G1 bias / G2 margin / H2 pedal
chatter in the harness, fix there, then Unity A/B on H2, H5, G1, G2, A2.

**P4 — delete dead config** (`T-METRIC`-style audit of the ~30 governor keys left without a consumer),
update ODD, ROADMAP, CLAUDE.md "Active Control Configuration".

## 5. Risks

- P1 raises curve speeds on tracks where PP already misses the lane by 0.34–0.58 m at the car. The
  guard is OOL/e-stop, not RMSE; a worse RMSE is the honest signal for item 3.
- P2 is the April failure mode. Do it signal by signal with the lateral metrics as the A/B target.
- P3 touches the brake-authority path that G2 depends on; the harness exists for exactly this.

## 6. Acceptance (numbers, from today's goldens / Night-60)

| metric | today | target |
|---|---|---|
| SpeedCompliance hill_highway / s_loop | 40.5 / 31.2 | ≥ 85 / ≥ 70 (or an explicit item-3 statement) |
| Lateral at-car RMSE highway_65 / mixed / sweeping | 0.036 / 0.06 / 0.06 m | within +0.02 m |
| Out-of-lane, e-stops (all tracks, all scenarios) | 0 | 0 |
| Launch: 12 m/s reached | ~11 s | ≤ 8 s (a_lon 2.0 + jerk) |
| H2 convergence | 15 s | ≤ 12 s |
| G1 post-conv bias / G2 TTC min | +6.5 m / 2.2 s | ≤ +3 m / ≥ 2.4 s |
| Pedal alternations H2 / G2 | 39 / 82 per min | ≤ 20 / ≤ 40 |
| Owners of the final speed command | 5 | 1 arbiter, one `active_owner` string |
