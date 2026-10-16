# Path to all-6-green, then new capabilities

**Written:** 2026-10-12 (after the margin fix + P1 budget promotion).
**Goal:** every lateral track passes the all-layers ≥ 95 bar and ACC is 14/14, then expand the ODD.

This is the honest plan. Two of the three remaining blockers are addressable with work we know how
to do; one may contain a genuine physics floor that we crack or formally accept, not tune away.

---

## Standing after Night-62 (expected)

Passing all-layers ≥ 95: **highway_65, sweeping_highway, and mixed_radius (marginal, Traj 95.1)** — ~3 of 6.
Blockers, by nature and effort:

| # | Blocker | Layer | Nature | Effort |
|---|---|---|---|---|
| 1 | hill_highway SpeedCompliance 86.6 | SpeedCompliance | control (budget cap) | 1–2 sessions |
| 2 | s_loop Traj 82 / SC 47, hairpin_15 Traj 50 | Trajectory | architecture (tight-curve control) | multi-session, real research |
| 3 | ACC H8 | ACC gate | confirm the registry fix | tonight |

---

## Blocker 1 — hill SpeedCompliance (nearest term, the budget ladder)

**Root cause, measured.** The velocity-profile curve cap holds hill's R100 curves at 6.7 m/s (budget off) /
8.6 m/s (0.08 g); the posted+target allowed is 11.2 m/s. `tools/speed_utilization.py` names the binding cap:
`velocity_profile 100 %`. So hill is slow by a *control cap*, not a metric artifact. And 11.2 m/s on R100 is
v²/R = 1.25 m/s² = **0.13 g lateral, inside the 0.18 g comfort limit** — physically drivable.

**Fix.** Continue the lateral-accel-budget ladder past 0.08 g (→ ~0.12–0.13 g) with the six-track regression
each step. The margin fix (gated actuator) is what makes the higher curve speed trackable; that is why 0.08 g
already lifted hill SC 43 → 86 with tracking *improving*. One or two more steps should carry hill SC over 95.

**Discipline.** Each step is an A/B + six-track sweep; promote only on no-regression. Stop raising g when any
track hits a real tracking or comfort limit — that boundary is a finding, not a failure. The budget is the
T-SPEED-OWNER-CONSOLIDATION path, so each step also retires one of the stacked legacy speed owners.

**Risk.** A track may hit its tracking limit before hill reaches SC 95. If so, hill SC-95 becomes a genuine
speed-vs-comfort trade for the user to set, not an engineering bug.

---

## Blocker 2 — s_loop + hairpin Trajectory (the hard one, tight-curve control)

**Root cause.** Pure Pursuit apex-cuts tight curves (single-point lookahead geometry). Curvature above
`mpc_max_map_curvature` = 0.015 routes to PP; hairpin R15 is κ 0.067 and s_loop R40 is κ 0.025, both well
above it, so both run PP. The boundary sits at 0.015 because the *kinematic* LMPC's model-plant mismatch was
too large at R < 67 m (the 0.111 m s_loop RMSE floor is that mismatch). These tracks are low speed (4–5 m/s),
so this is a *geometry/model* problem, not the high-speed margin we just fixed — the actuator gate is off
below 9 m/s and does not help here.

**Characterized 2026-10-12 (offline, today's fresh recordings + harness).** The tight-curve error is a
Pure-Pursuit CONTROL-LAW failure, not a physics or steering-authority floor:
- s_loop R40 @ 5.2 m/s: PP RMS 0.36, 97 % random (not apex-cut bias), error grows with curvature (corr +0.40),
  steering has headroom (max 0.50 of 0.70). PP cannot track the fast R40 reversals.
- hairpin R15 @ 3.3 m/s: PP RMS 0.98, runs WIDE (outside, not apex-cut), steering SATURATES (0.70) recovering —
  yet the geometric steady-state demand at R15 is only ~0.33 normalized, so saturation is PP falling behind,
  not a physical limit.
- Both run PP because `mpc_max_map_curvature` 0.015 routes κ > 0.015 away from the LMPC.
- In the harness the IMPROVED LMPC (sign + actuator + symmetric warmup) tracks R40 at 0.24 and R15 at 0.41 —
  far better than PP. **Caveat: the harness is kinematic and cannot prove this transfers to Unity** (the old
  `feedback_sloop_pp_ceiling` memory said the OLD LMPC couldn't help here — that premise may have changed, but
  only a Unity A/B settles it).

**Plan, in order (each gate the next) — REFINED by the above:**

**Unity A/B 2026-10-12 (mpc_max_map_curvature 0.015 → 0.08, s_loop + hairpin, 5 pairs) — boundary extension
REFUTED, but it reframes the problem.** The LMPC tracks the sustained tight ARC far better than PP, exactly as
the harness promised — hairpin tight-arc RMS 0.955 → **0.072** (Traj 50.6 → 85.5), s_loop 0.367 → 0.233. BUT
overall COLLAPSES: hairpin 49.8 → 29.0 (Control 97.6 → **65.0**, OOL 0 → 5), s_loop 91.0 → 38.5 (Control 98 →
72, OOL 0 → 4). So the LMPC holds the tight line but with **aggressive, jerky, occasionally-unstable steering**
(Control collapse + out-of-lane), not a tracking-capability limit. **This refutes the old `feedback_sloop_pp_
ceiling` premise** (the kinematic LMPC CAN track R15 at 0.072) and reframes Blocker 2: the problem is making
the LMPC track tight curves SMOOTHLY (comfort / stability / transition handling), not whether it can track
them. The kinematic-harness over-promised the OVERALL outcome (same lesson as the margin) while being right
about sustained-arc tracking.

**SIX-TRACK REGRESSION (2026-10-15 23:14, corrected scorer) — 5/6 at or above baseline.** hairpin_15 50.2→**97.8**
(all layers ≥95 except one at 93.8, see journal #57), s_loop 89.0→93.6 (Traj 94.4→95.9 with FF gain 0.78; blocked only by the
SpeedCompliance speed-owner floor 48.5), highway =, sweeping +0.8, hill +1.8. mixed_radius 98.3→84.0 is NOT Stanley: 3/5 runs
carry the known LMPC R200 weave and the R40 approach hands PP the car at 11.6 m/s (production routing) where it saturates
and diverges before the 9 m/s Stanley cap; fix = tight cap 20 (Stanley replaces PP on tight curves at all speeds) — final
candidate running with a same-night production control. Blocker 2 (tight-curve floor) is effectively RESOLVED by design.

**RESEARCH Phases 1b–1e (2026-10-15) — Stanley fully diagnosed; promotable.** True-heading source alone regressed
(lost the pursuit angle's anticipation — Smell 7); the real residual was the FF reading the +RIGHT `reference_point['curvature']`
with a +LEFT formula (FF steered OUT of the curve on every prior Stanley trial). Fixed FF source (map horizon) + plant gain
0.78: s_loop overall 94.2 / Traj 95.4 / Control 100 / RMSE 0.082; hairpin 92.4 / RMSE 0.089 / Control 100 (after the scorer
out-of-lane frame fix). Two scorer frame defects fixed (OOL lookahead lines; heading penalty = pursuit angle). Promotable
routing `stanley_tight_curve_routing` (Stanley in PP's κ≥0.015 slot) in six-track ×5 regression; s_loop 92.8 (bl 89.0),
hairpin 91.6 (bl 50.2) so far, all layers ≥95 except Trajectory (heading-penalty fix staged) and SpeedCompliance (design floor).

**RESEARCH Phase 1 (2026-10-15) — classical Stanley WINS on both tight tracks.** Three Stanley defects fixed behind
kill-switches (cross-track sign inverted for the road_frame_at_car source; radian command passed as normalized ≈0.37×;
wrapper never read control.lateral.stanley_* from full_config). Routed to the whole run below 9 m/s with FF on:
s_loop Traj 81.9→92.4 (RMSE 0.37→0.15), hairpin overall 49.3→87.2 / Traj 50.1→73.7 (RMSE 0.97→0.28), OOL 0 both,
Control the residual (heading term = PP pursuit angle, contains cross-track, fights the cross-track term → 0.4 Hz
weave). Phase 1b = gt_path_tangent heading source. Then routing design + 6-track regression before promotion.

**RESEARCH Phase 0 (2026-10-15) — dynamic bicycle model REFUTED.** Re-baselined with the measured 0.78 steering
gain (its documented blocker) on the clean stack: worse than PP everywhere (s_loop 91->74, hairpin 49->29,
mixed 98->34 CATASTROPHIC). The "95 on s_loop" was lookahead-frame inflation. The whole optimization-MPC family
(kinematic + dynamic) is refuted for tight curves — they track the arc better but fail the reversals/exits.
Research Phase 1 = Stanley extended to 3-5 m/s (untried, geometric + heading law, PP-like transition robustness)
and/or curvature-adaptive PP lookahead; if both lose to PP, formalize the floors.

**RESOLVED 2026-10-15 — Blocker 2 is a REAL PP ceiling; LMPC-on-tight is refuted on clean runs.** Fixing the
e-stop frame (it false-fired on tight-curve lookahead displacement) let the car run the full curve and exposed the
truth: the LMPC under-steers R15/R40, drifts off, and scores WORSE than PP (hairpin 29 vs 49, s_loop 80 vs 91; OOL
5-8 vs 0). Both earlier LMPC readings were e-stop-truncation artifacts. PP is the best controller we have for the
tight tracks and the production boundary is correct. To clear s_loop/hairpin Trajectory needs a genuinely better
tight-curve controller (large-angle, since the LMPC's small-angle QP breaks at heading error >0.3 rad) OR
formalizing the floors with this evidence. The regime-blend plan is dropped. See the ORIGINAL (now-superseded) pivot
note below for the e-stop-bug detail.

**PIVOT 2026-10-15 (superseded by the clean A/B above) — the e-stop was false-firing (T-SAFETY-ESTOP-FRAME).** The /plan-feature
mechanism-verification gate showed the LMPC never left the lane on the tight tracks (max at-car 0.73 m); every
"OOL" was one false GT lane-boundary e-stop per run, because the e-stop tested Unity's LOOKAHEAD lane-line
positions (displaced 1.2–1.9 m on R15/R40) against an at-car margin. Pure Pursuit escaped it only by running wide.
Fixed (at-car frame, kill-switched). The regime-blend architecture is WITHDRAWN; the remaining tight-curve issue
is the secondary steering chatter. Probe A's "refutation" below stands only for the chatter, not the OOL.

**Probe A (2026-10-15, damping tune) — refuted for the CHATTER; the OOL it could not touch was the false e-stop.** With the LMPC on the tight
tracks, `mpc_r_steer_rate_kappa_gain` 0 → 30 (saturate raised to reach R15): s_loop chatter 12.4 → 12.7 rev/s
(inert), OOL 5 → 5; hairpin chatter 15.1 → 12.7, Control 65 → 73, arc RMS 0.070 unchanged, but OOL 5 → 5 and
overall ~29 both arms. **Every run leaves the lane regardless of damping.** The chatter is a secondary, partly
tunable symptom on a sustained arc; the dominant failure is the out-of-lane at the regime handoff/transitions,
which no steering-rate penalty touches. This is design smell 1 (binary PP↔LMPC gate at `mpc_max_map_curvature`)
— the LMPC tracks the arc at 0.07 and the discontinuous handoff throws the car out at entry/exit. → Option B.

**Refined plan for Blocker 2 (2026-10-12 pm):**
0. (done) Boundary extension alone — refuted: LMPC tracks the arc, collapses Control + leaves the lane.
1. **Diagnose WHERE the LMPC fails on tight curves** (tomorrow): entry cold-start, the PP↔LMPC handoff/blend,
   exit, or steady-state jerk/aggressiveness. hairpin Control 65 says it is a comfort/stability problem; the
   lever is likely a tight-curve-specific r_steer_rate / jerk / q_lat schedule, or a smoother regime blend.
2. If smoothing the LMPC's tight-curve steering recovers Control + OOL while keeping the arc tracking (Traj
   85+), extend the boundary and promote — s_loop/hairpin Trajectory could clear the bar without the dynamic
   model. If it cannot be made smooth within comfort, THEN the dynamic model / a floor.

(superseded first-try note below)
0-OLD. **Try extending the LMPC boundary FIRST (cheaper than the dynamic model).** Unity A/B: raise
   `mpc_max_map_curvature` 0.015 → ~0.07 on s_loop and hairpin, LMPC vs PP, measure at-car Trajectory. The
   harness says the LMPC control law should track these far better; if it transfers, s_loop/hairpin improve
   without the dynamic model. If the kinematic LMPC falls short of the harness promise in Unity, THEN:

1. **Characterize the floor** with the same closed-loop sysid that cracked the margin (steering injection on
   s_loop/hairpin at 4–6 m/s). Decide: is the tight-curve error model-plant mismatch, a feedforward
   deficiency, or a real floor at the ~21 Hz camera cadence? This tells us which lever, before building.
2. **If model mismatch → the dynamic bicycle model.** It is built and unit-tested (`test_dynamic_bicycle_mpc`,
   `mpc_dynamic_model_enabled` / `mpc_tire_ekf_enabled`, both currently false) but never validated E2E. Validate
   it offline on recorded tight-curve frames first (the replay-before-E2E discipline), then one Unity A/B on
   hairpin/s_loop. The dynamic model with identified tire stiffness is what the kinematic model lacks at tight
   radius.
3. **Extend the LMPC regime boundary** (`mpc_max_map_curvature` 0.015 → toward 0.025–0.067) once the model
   tracks tight curves, so s_loop/hairpin run the capable controller instead of PP.

**Honest caveat.** The tightest geometry may have a real physical tracking floor at this sensor cadence. If the
characterization in step 1 shows that, the principled move is to formalize the floor via `GOLDEN_BELOW_FLOOR`
with the evidence, not to tune the score up. "All green" then means *cracked where crackable, floor-accepted
with proof where not*. s_loop and hairpin are the two tracks where this caveat is live.

---

## Blocker 3 — ACC H8 (confirm, then close)

The H8 FAIL was the missing scenario registry dropping the overlay (fixed, `b316587`). Confirm tonight that
the nightly re-seed read the registry and H8 ran at its 15 m/s target with ACC engaged. If it held, ACC is
14/14 and this is closed. If not, the fix is in the `/e2e` → registry path, not control.

---

## Sequencing

1. **Tomorrow:** read Night-62 — H8, any ACC scenario that moved under the stacked stack, and the lateral P1
   gains on fresh runs. Fix H8's path if the registry did not take.
2. **Budget ladder → hill green** (Blocker 1). Nearest, highest-confidence win. Likely also nudges s_loop SC.
3. **Tight-curve characterization → dynamic model → s_loop/hairpin** (Blocker 2). The real research; may end in
   a cracked floor or a proven-and-accepted one.
4. **All-green (or floor-accepted) → new capabilities.**

---

## New capabilities (once the base is green or floor-accepted)

The margin fix + actuator model were the *prerequisite* for most of these — they unlock speed the stack could
not previously hold.

- **Higher-speed ODD.** The 13.5+ m/s band and the autobahn scenarios: the sensitivity peak there (17–67×) is
  exactly what the actuator model and the rate-limiter scheduling address. This is the most direct new
  capability the margin work enables.
- **Tight-curve tracks.** Once the dynamic model tracks R15–R40, new tight geometries become in-ODD.
- **Vision-primary perception.** Retrain the segmentation checkpoint (pipeline ready, `train_pipeline.sh`); the
  current checkpoint mislabels the hood, so the CV fallback is the de-facto path. This is independent of the
  lateral work and can run in parallel.
- **Richer ACC.** Cut-in, multi-lead, and the high-speed catch-up scenarios the higher ODD enables.

The principle throughout: a score moves only from a named mechanism, every config step is A/B + six-track
regression, and a structural floor is formalized with evidence rather than tuned.
