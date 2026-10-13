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

**Plan, in order (each gate the next):**

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
