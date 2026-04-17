"""Diagnose 2DOF feedforward decomposition: verify curvature cancellation at QP level.

Runs the MPC solver on synthetic curve-entry scenarios and prints the affine
term (c_head) before and after feedforward cancellation.  This validates that
the decomposition is actually canceling the curvature bias in the dynamics.

Usage:
    python tools/analyze/diagnose_ff_decomp.py
"""

import math
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from control.mpc_controller import MPCParams, MPCSolver


def _make_params(**overrides) -> MPCParams:
    defaults = dict(
        horizon=20,
        dt=0.1,
        q_lat=2.0,
        q_heading=5.0,
        q_speed=1.0,
        r_steer=1e-4,
        r_accel=0.05,
        r_steer_rate=2.0,
        delta_rate_max=0.5,
        first_step_rate_enabled=True,
        ff_alignment_enabled=True,
        wheelbase_m=2.5,
    )
    defaults.update(overrides)
    return MPCParams(**defaults)


def run_scenario(label: str, kappa_val: float, v: float, e_lat: float = 0.0):
    """Run decomp-on vs decomp-off and compare."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  κ={kappa_val:.4f} (R={1/kappa_val:.0f}m), v={v:.1f} m/s, e_lat={e_lat:.3f} m")
    print(f"{'='*70}")

    p_on = _make_params(ff_decomposition_enabled=True)
    p_off = _make_params(ff_decomposition_enabled=False)
    s_on = MPCSolver(p_on)
    s_off = MPCSolver(p_off)

    kappa = np.full(20, kappa_val)

    r_on = s_on.solve(e_lat=e_lat, e_heading=0.0, v=v,
                       last_delta_norm=0.0, kappa_ref_horizon=kappa,
                       v_target=12.0, v_max=15.0, dt=0.1)
    r_off = s_off.solve(e_lat=e_lat, e_heading=0.0, v=v,
                         last_delta_norm=0.0, kappa_ref_horizon=kappa,
                         v_target=12.0, v_max=15.0, dt=0.1)

    # Compute expected values
    max_steer = s_on._max_steer_at_speed(v)
    delta_ff = math.atan(p_on.wheelbase_m * kappa_val) / max_steer
    steer_gain = (v / p_on.wheelbase_m) * max_steer * p_on.dt
    c_head_raw = -kappa_val * v * p_on.dt
    c_head_ff_term = steer_gain * delta_ff
    c_head_after = c_head_raw + c_head_ff_term

    print(f"\n  Kinematic feedforward:")
    print(f"    δ_ff (normalized)     = {delta_ff:+.6f}")
    print(f"    δ_ff (deg)            = {delta_ff * math.degrees(max_steer):+.4f}°")

    print(f"\n  QP affine term c_head[0] (heading dynamics bias):")
    print(f"    raw (no decomp)       = {c_head_raw:+.8f}")
    print(f"    steer_gain × δ_ff     = {c_head_ff_term:+.8f}")
    print(f"    after cancellation    = {c_head_after:+.8f}")
    if abs(c_head_raw) > 1e-10:
        cancel_pct = (1.0 - abs(c_head_after) / abs(c_head_raw)) * 100
        print(f"    cancellation ratio    = {cancel_pct:.2f}%")
    else:
        print(f"    cancellation ratio    = N/A (straight road)")

    # Verify against actual solver diagnostics
    if 'ff_decomp_c_head_raw' in r_on:
        print(f"\n  Solver-reported diagnostics:")
        print(f"    c_head_raw            = {r_on['ff_decomp_c_head_raw']:+.8f}")
        print(f"    c_head_after          = {r_on['ff_decomp_c_head_after']:+.8f}")
        print(f"    cancel_ratio          = {r_on['ff_decomp_cancel_ratio']:.4f}")

    print(f"\n  Steering output comparison:")
    print(f"    decomp OFF: δ_cmd     = {r_off['steering_normalized']:+.6f}")
    print(f"    decomp ON:  δ_cmd     = {r_on['steering_normalized']:+.6f}  (ε={r_on.get('ff_decomp_epsilon', 0):+.6f} + δ_ff={r_on.get('ff_decomp_delta_ff', 0):+.6f})")
    delta_diff = r_on['steering_normalized'] - r_off['steering_normalized']
    print(f"    Δδ (on - off)         = {delta_diff:+.6f}")
    if abs(r_off['steering_normalized']) > 1e-6:
        pct = delta_diff / abs(r_off['steering_normalized']) * 100
        print(f"    relative change       = {pct:+.1f}%")

    # Multi-frame simulation: 10 frames entering a curve from straight
    print(f"\n  10-frame curve-entry simulation:")
    print(f"    {'Frame':>5}  {'δ_off':>8}  {'δ_on':>8}  {'ε':>8}  {'δ_ff':>8}  {'Δδ':>8}")
    print(f"    {'─'*5}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    last_off = 0.0
    last_on = 0.0
    s_on_multi = MPCSolver(_make_params(ff_decomposition_enabled=True))
    s_off_multi = MPCSolver(_make_params(ff_decomposition_enabled=False))

    for i in range(10):
        # Transition: first 3 frames straight, then curve
        k_frame = kappa_val if i >= 3 else 0.0
        kappa_frame = np.full(20, k_frame)
        # Simulate growing e_lat from late turn-in
        e_lat_frame = 0.0 if i < 3 else min(0.05 * (i - 3), 0.2)

        r_off_f = s_off_multi.solve(e_lat=e_lat_frame, e_heading=0.0, v=v,
                                      last_delta_norm=last_off, kappa_ref_horizon=kappa_frame,
                                      v_target=12.0, v_max=15.0, dt=0.1)
        r_on_f = s_on_multi.solve(e_lat=e_lat_frame, e_heading=0.0, v=v,
                                    last_delta_norm=last_on, kappa_ref_horizon=kappa_frame,
                                    v_target=12.0, v_max=15.0, dt=0.1)

        eps = r_on_f.get('ff_decomp_epsilon', 0.0)
        dff = r_on_f.get('ff_decomp_delta_ff', 0.0)
        dd = r_on_f['steering_normalized'] - r_off_f['steering_normalized']

        print(f"    {i:>5}  {r_off_f['steering_normalized']:+8.5f}  {r_on_f['steering_normalized']:+8.5f}  {eps:+8.5f}  {dff:+8.5f}  {dd:+8.5f}")

        last_off = r_off_f['steering_normalized']
        last_on = r_on_f['steering_normalized']


def main():
    print("2DOF Feedforward Decomposition — QP Cancellation Diagnostics")
    print("=" * 70)

    # Scenario 1: Gentle curve (R200) at speed
    run_scenario("Gentle curve (highway)", kappa_val=0.005, v=12.0)

    # Scenario 2: Medium curve (R50) — typical mixed_radius
    run_scenario("Medium curve (R50)", kappa_val=0.02, v=10.0)

    # Scenario 3: Tight curve (R25)
    run_scenario("Tight curve (R25)", kappa_val=0.04, v=7.0)

    # Scenario 4: Medium curve with existing lateral error
    run_scenario("Medium curve + 0.15m error", kappa_val=0.02, v=10.0, e_lat=0.15)

    # Summary
    print(f"\n{'='*70}")
    print("2DOF DECOMPOSITION SUMMARY")
    print(f"{'='*70}")
    print("""
If cancellation ratio is near 100%, the decomposition is working correctly
at the QP level — the curvature bias is being removed from the dynamics.

If δ_cmd is nearly identical between decomp-on and decomp-off, it means
the MPC was ALREADY finding the feedforward solution without explicit
decomposition. This would explain the weak E2E signal.
""")

    # --- Preview-weighted q_lat: stability boundary sweep ---
    print(f"\n{'='*70}")
    print("PREVIEW-WEIGHTED q_lat — STABILITY BOUNDARY SWEEP")
    print(f"{'='*70}")
    print("""
Sweep q_lat from 1.0 to 5.0 and compare oscillation behavior:
  - Uniform (step0_scale=1.0): current behavior
  - Ramped  (step0_scale=0.3): softened step 0

Metric: peak-to-peak steering amplitude and sign changes over 50 frames
on a straight road with e_lat=0.3m initial error.
""")
    stability_sweep()


def stability_sweep():
    """Sweep q_lat and compare uniform vs ramped oscillation behavior.

    Uses closed-loop kinematic bicycle simulation so e_lat evolves based on
    the MPC steering output — this is what creates real oscillation feedback.
    """
    n_frames = 50
    e_lat_init = 0.3
    v = 10.0
    dt = 0.1
    L = 2.5  # wheelbase

    print(f"  {'q_lat':>5}  {'─── Uniform (1.0) ───':^25}  {'─── Ramped (0.3) ───':^25}  {'Boundary?'}")
    print(f"  {'':>5}  {'PtP':>8}  {'SignΔ':>6}  {'eLatPk':>8}  {'PtP':>8}  {'SignΔ':>6}  {'eLatPk':>8}  {'':>10}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*10}")

    for q_lat_val in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        results = {}
        for label, s0s in [("uniform", 1.0), ("ramped", 0.3)]:
            p = _make_params(
                q_lat=q_lat_val,
                q_lat_preview_step0_scale=s0s,
                ff_decomposition_enabled=False,
            )
            s = MPCSolver(p)
            last_delta = 0.0
            e_lat = e_lat_init
            e_heading = 0.0
            deltas = []
            e_lats = []
            for _ in range(n_frames):
                r = s.solve(e_lat=e_lat, e_heading=e_heading, v=v,
                            last_delta_norm=last_delta,
                            kappa_ref_horizon=np.zeros(s._N),
                            v_target=12.0, v_max=15.0, dt=dt)
                if not r['feasible']:
                    delta_cmd = 0.0
                else:
                    delta_cmd = r['steering_normalized']

                # Kinematic bicycle plant update
                max_steer = s._max_steer_at_speed(v)
                delta_rad = delta_cmd * max_steer
                e_heading += (v / L) * math.tan(delta_rad) * dt
                e_lat += v * math.sin(e_heading) * dt

                last_delta = delta_cmd
                deltas.append(delta_cmd)
                e_lats.append(e_lat)

            ptp = max(deltas) - min(deltas)
            e_lat_peak = max(abs(e) for e in e_lats[n_frames // 2:])  # peak in 2nd half
            sign_changes = sum(1 for i in range(1, len(deltas))
                               if deltas[i] * deltas[i - 1] < 0)
            results[label] = (ptp, sign_changes, e_lat_peak)

        u_ptp, u_sc, u_ep = results["uniform"]
        r_ptp, r_sc, r_ep = results["ramped"]

        # Flag if uniform is oscillating (>5 sign changes) but ramped is not
        boundary = ""
        if u_sc > 5 and r_sc <= 5:
            boundary = "SHIFTED ↑"
        elif u_sc > 5 and r_sc > 5:
            boundary = "both osc"
        elif u_ep > 0.1 and r_ep < u_ep * 0.7:
            boundary = "dampened"

        print(f"  {q_lat_val:>5.1f}  {u_ptp:>8.4f}  {u_sc:>6}  {u_ep:>8.4f}  "
              f"{r_ptp:>8.4f}  {r_sc:>6}  {r_ep:>8.4f}  {boundary:>10}")

    print(f"""
If "SHIFTED ↑" appears, the ramped config is stable at a q_lat value where
uniform oscillates — preview weighting raised the effective stability boundary.
"dampened" means both oscillate but ramped has significantly lower amplitude.
""")


if __name__ == "__main__":
    main()
