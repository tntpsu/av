# Step 3C Design Lessons — Grade Robustness Failure Analysis

**Date:** 2026-03-16
**Context:** Step 3C implemented 9 phases of grade compensation. Unit tests pass (905+), flat-track regression clean. E2E validation on `hill_highway` (5% grade, R100 arcs, ~943m oval) fails with lateral divergence → e-stop.

---

## Lesson 1: Control Subsystem Interactions Are Not Compositional

**What we assumed:** Each phase (effective max_accel, jerk relaxation, steering damping, preview fix, mesh smoothing) was designed to be independently correct and a mathematical no-op at grade=0. We verified each in isolation with unit tests.

**What actually happened:** The jerk cooldown mechanism (`jerk_cooldown_frames: 8, jerk_cooldown_scale: 0.4`) — a comfort feature designed for transient events on flat tracks — created a catastrophic feedback loop with grade compensation:

```
Grade FF adds sustained 0.88 m/s² offset
  → desired_accel jumps well above last_accel_cmd
  → jerk limiter clips every frame (jerk_capped = True)
  → cooldown resets to 8 frames every frame (never expires)
  → accel_cmd *= 0.4 permanently
  → throttle starved to 5-10% (needed: 42%)
  → car can't maintain speed on 5% grade
```

**Root cause:** The jerk cooldown was never designed for sustained acceleration offsets. Its implicit assumption is "if jerk fired, something unexpected happened — back off." On a grade, the jerk is *expected* and sustained.

**Fix:** Bypass jerk cooldown when `|gravity_accel| > 0.1` — the sustained grade FF means jerk events are expected, not anomalous.

**Lesson:** *Unit tests verify each subsystem's correctness, but control loops have emergent interactions that only appear under sustained operating-point shifts.* The jerk cooldown was 4 levels of indirection from the grade FF: grade_rad → gravity_accel → desired_accel → jerk_limiter → cooldown_trigger → throttle_attenuation. No unit test naturally exercises this chain.

---

## Lesson 2: Telemetry Fields Must Map to Root-Cause Observability

**What happened in debugging:** The first run's `longitudinal_accel_cmd_raw` field showed max 0.24 m/s² when the PID + grade FF should produce ~1.75 m/s². This was the key diagnostic signal.

But `accel_cmd_raw` is computed as `throttle × self.max_accel` — using the BASE max_accel (1.2), not the effective max_accel (2.08). This misled initial analysis because the scaling mismatch made it look like the accel command was being clipped much earlier in the pipeline.

**The actual diagnostic chain that found the root cause:**
1. `accel_cmd_raw` max = 0.24 → "accel is being suppressed"
2. `longitudinal_jerk_capped` = True on ~50% of frames → "jerk limiter is active"
3. Tracing code: `jerk_cooldown_remaining` resets to 8 every frame → "cooldown never expires"
4. `accel_cmd *= 0.4` applied every frame → "throttle permanently attenuated"

**Lesson:** *When adding new operating modes (grade compensation), also verify that existing telemetry fields still provide root-cause observability.* We should have recorded `gravity_accel`, `jerk_cooldown_remaining`, and `dynamic_max_jerk` to HDF5 for faster diagnosis.

---

## Lesson 3: Two-Stage Failure — Fixing Throttle Reveals Lateral Instability

**Run 1 (no jerk cooldown fix):** Max speed 9.0 m/s. E-stop at frame 379. Lateral oscillation ±0.5m with growing amplitude on the 5% uphill straight.

**Run 2 (jerk cooldown bypassed on grade):** Max speed 11.5 m/s. E-stop at frame 302. Lateral oscillation starts smaller (±0.2m) but diverges catastrophically at ~11.5 m/s near the uphill-to-arc transition.

**What this means:** The throttle starvation in Run 1 was masking a second failure mode. At 8-9 m/s the PP controller was marginally unstable on grade; at 11.5 m/s it becomes decisively unstable.

**Root cause hypothesis for lateral divergence:** At 11.5 m/s on a 5% grade:
- WheelCollider weight distribution shifts (rear axle loaded, front unloaded)
- Speed oscillation ±0.3 m/s from WheelCollider physics creates PID hunting
- PP proportional gain (no speed normalization in base config) doesn't reduce with speed
- Grade steering damping (alpha 0.9 → 0.65) is insufficient at this speed
- The uphill→flat→arc transition creates a compound perturbation (grade change + curvature onset)

**Lesson:** *Fixing one failure mode often reveals a deeper one. Always plan for iterative E2E validation, not single-shot success.* The plan correctly identified 3 design gaps but the interaction effects create a 4th gap (jerk cooldown) and the fix for gap 1 exposes gap 3 at higher speed.

---

## Lesson 4: "Grade-Zero No-Op" Is Necessary But Not Sufficient

The design principle "every change is a mathematical no-op when grade=0" is correct and valuable — it guarantees flat-track regression. All 125 flat-track tests pass.

But **grade-zero no-op only guarantees backward compatibility, not forward correctness.** The jerk cooldown interaction is an emergent behavior that only appears under sustained grade — no flat-track test can catch it.

**What would have caught this earlier:**
- An integration test that runs the full longitudinal controller for 100+ frames with grade=0.05 and checks that `throttle > 0.3` after stabilization
- A test that verifies `jerk_cooldown_remaining` doesn't stay permanently > 0 under sustained grade
- Recording `jerk_cooldown_remaining` to HDF5 for post-hoc analysis

---

## Lesson 5: Config Overlays vs Base System — The Right Boundary

The original Step 3B approach used per-track config overlays (`mpc_hill_highway.yaml`) with:
- `max_accel: 2.0` (hardcoded budget expansion)
- `max_jerk: 3.0` (hardcoded jerk limit increase)
- `steering_smoothing_alpha: 0.40` (hardcoded damping)

Step 3C correctly identified that these should be **auto-derived from grade** rather than per-track constants. The base-system approach (grade-aware effective max_accel, proportional jerk relaxation, proportional steering damping) is architecturally superior.

But the E2E failure shows that **promoting to base system requires testing the full interaction space**, not just verifying the math of each component. The per-track overlay "worked" (conceptually) because it bypassed the interaction by hardcoding the final values. The base-system approach is more general but also more fragile to interaction effects.

**Lesson:** *When promoting per-track tuning to auto-derived base behavior, budget time for iterative E2E debugging. The number of interaction effects scales with the number of control subsystems.*

---

## Current Status and Next Steps

### Fixed:
- [x] Jerk cooldown bypass when grade compensation active (prevents throttle starvation)

### Remaining:
- [ ] Lateral oscillation at 11.5 m/s on graded surface — PP controller instability
- [ ] Potential fixes to investigate:
  - Speed-dependent PP gain (already exists as `pp_speed_norm_enabled` — currently off in base)
  - More aggressive grade steering damping (current: 5.0 gain → alpha 0.9→0.65; may need 0.40 at 11+ m/s)
  - Grade × speed cross-term in steering alpha (grade alone insufficient at high speed)
  - Mesh smoothing verification — confirm TrackBuilder `SmoothGradeTransitions()` is applying correctly at hilltop
  - Check if curvature preview is seeing the R100 arc too early and causing premature steering
- [ ] Add integration test: full longitudinal controller, 100+ frames, grade=0.05, verify throttle > 0.3
- [ ] Record `jerk_cooldown_remaining` and `gravity_accel` to HDF5 for diagnostics

### Test results:
- 906 unit tests pass (130 key scoring/regression/comfort/grade tests clean)
- Flat-track regression: 0 regressions
- E2E hill_highway: FAILING (e-stop at frame 302, lateral divergence at 11.5 m/s)

---

## Lesson 4: Verify Hypotheses Before Implementing Fixes (Step 3D, 2026-03-17)

**What we assumed:** "Weight transfer from pitch reduces front tire grip on grade, causing lateral instability." We spent multiple iterations adding control-layer patches (jerk cooldown bypass, steering rate limits, EMA smoothing) based on this hypothesis.

**What we discovered:**
1. **Pitch is frozen** in Unity (`FreezeRotationX | FreezeRotationZ` in CarController.cs). The car body stays perfectly level on grade. Weight transfer from pitch CANNOT happen.
2. **Angular velocity was hardcoded to zeros** in orchestrator.py (line 6583), discarding real yaw dynamics data from Unity.
3. **No per-wheel telemetry** was being captured — zero visibility into what WheelColliders were actually doing on graded surfaces.

**What we should have done first:** Instrument the physics and diagnose with real data before changing control parameters. The PP grade-proportional steering rate limit was based on a wrong hypothesis and has been reverted.

**Step 3D approach:** Instrument first (angular velocity fix, per-wheel GetGroundHit telemetry, analysis script), then diagnose from data, then fix.

**Key principle:** When a complex system fails in an unexpected way, the first response should be to add observability, not to add compensating control logic. Compensating for a misunderstood failure mode can mask the real issue and create new problems.

---

## Lesson 5: Two-Layer Failure — MPC Hunting + GT Boundary Corruption (Step 3D, 2026-03-17)

**What we found by instrumenting instead of guessing:**

**Root cause 1 — MPC hunting (control):** The base config `mpc_q_lat=10.0` caused MPC steering oscillation whenever the regime engaged at 12 m/s on grade. The MPC regime dispatcher activates when speed > `pp_max_speed_mps + upshift_hysteresis` = 11.0 m/s. On grade, PP tracking error grows to ~0.3m, pushing the car above the upshift threshold. With `q_lat=10.0`, MPC over-corrects → oscillation → e-stop at frame 765. Fix: `q_lat=0.5, r_steer_rate=2.0` (already validated on highway and mixed_radius — should have been the base config from the start).

**Root cause 2 — GT boundary corruption (safety):** After fixing MPC hunting, e-stop moved to frame 2192. The actual lateral error at that frame was only 0.24m. The trigger was Unity's `GroundTruthReporter.GetLanePositionsAtLookahead()` producing corrupt boundary values (5000+ meters) at specific track positions — likely mesh seam or `FindClosestPointOnOval()` failure. The e-stop logic trusted these values blindly.

**Key diagnostic signals that identified the root causes:**
1. `control/regime` field showed MPC active (regime=1) starting at frame 255 — every frame after 12 m/s
2. `control/steering` vs `control/steering_post_hard_clip` divergence revealed MPC was overriding PP
3. MPC steering sign-change rate > 30% of frames = classic hunting pattern
4. GT boundary value of 5000+ m at frame 2192 = clearly corrupt, not a real lateral error

**Lesson:** *The existing telemetry had enough information to identify both root causes, but our diagnostic tools weren't looking for them.* Five diagnostic improvements were needed: MPC oscillation detection, regime-aware steering visualization, GT boundary sanity checking, e-stop root cause attribution, and regime transition logging. These are now implemented.
