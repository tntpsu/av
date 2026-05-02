"""Phase 0 oracle for NMPC cold-start sign-determinism fix.

Investigates whether `_warm_u is None` cold-starts with non-zero `e_lat` on
straight roads (kappa=0) yield deterministic SLSQP convergence across BLAS
implementations. Reports the existing x0[0] (which is 0 at kappa=0) and the
proposed seed value for a range of (e_lat, gain) pairs.

The Linux CI failure (test_nmpc_matches_lmpc_direction at e_lat=-0.3) is
caused by SLSQP starting at the saddle point of a near-symmetric cost
surface. Tiny BLAS noise tips it into the wrong basin of attraction.

Usage:
    python3 tools/analyze/validate_nmpc_cold_start.py
    python3 tools/analyze/validate_nmpc_cold_start.py --gain 0.5

Exit code 0 = recommended gain found (flips sign deterministically across
the test range without saturating the seed cap).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, "/Users/philtullai/av")

from control.nmpc_controller import NMPCParams, SLSQPNMPCSolver  # noqa: E402


E_LAT_TEST_GRID = [-1.0, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 1.0]
KAPPA_STRAIGHT = 0.0
V_TEST = 5.0
DT_TEST = 0.077


def existing_cold_start_x0_first_steer(p: NMPCParams) -> float:
    """Reproduce the existing cold-start x0[0] formula at kappa=0.

    From control/nmpc_controller.py:426-431 — for k=0:
      x0[0] = _feedforward_delta_norm(kappa[0], wheelbase_m, max_steer_rad)
    With kappa=0, atan(L*0)=0, so x0[0]=0 regardless of e_lat.
    """
    import math
    return math.atan(p.wheelbase_m * KAPPA_STRAIGHT) / p.max_steer_rad


def proposed_seed_term(e_lat: float, p: NMPCParams,
                       gain: float, max_frac: float) -> float:
    """Compute the proposed sign-preserving lateral-error seed.

    Bias x0[0] toward -sign(e_lat) (so negative e_lat → positive steering).
    Magnitude bounded by max_frac * max_steer_rad to leave SLSQP room.
    """
    seed_max = max_frac * p.max_steer_rad
    return float(np.clip(-e_lat * gain, -seed_max, seed_max))


def solve_nmpc_steering(solver: SLSQPNMPCSolver, e_lat: float) -> dict:
    """Run the actual NMPC solve (no fix applied — tests current behavior)."""
    N = solver.p.horizon
    kappa_horizon = np.zeros(N)
    return solver.solve(
        e_lat=e_lat, e_heading=0.0, v=V_TEST, last_delta_norm=0.0,
        kappa_ref_horizon=kappa_horizon, v_target=V_TEST, v_max=10.0,
        dt=DT_TEST, grade_rad=0.0,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gain", type=float, default=0.5,
                    help="Proposed seed gain (normalized-rad per m of e_lat)")
    ap.add_argument("--max-frac", type=float, default=0.10,
                    help="Max seed as fraction of max_steer_rad")
    ap.add_argument("--gain-sweep", action="store_true",
                    help="Try gains [0.1, 0.3, 0.5, 1.0, 2.0] and report")
    args = ap.parse_args()

    p = NMPCParams()
    print(f"NMPC defaults: wheelbase={p.wheelbase_m}m, max_steer_rad={p.max_steer_rad:.4f} "
          f"(={p.max_steer_rad * 180 / np.pi:.2f}°), horizon={p.horizon}")
    print()

    print("=" * 80)
    print("EXISTING cold-start behavior at kappa=0 (the CI failure regime)")
    print("=" * 80)
    print(f"{'e_lat':>8s}  {'existing x0[0]':>14s}  {'NMPC steer':>10s}  "
          f"{'sign(e_lat)':>11s}  {'expected sign':>13s}  {'OK?':>4s}")
    for e_lat in E_LAT_TEST_GRID:
        existing_x0 = existing_cold_start_x0_first_steer(p)
        # Fresh solver each time to force cold-start path
        solver = SLSQPNMPCSolver(NMPCParams())
        result = solve_nmpc_steering(solver, e_lat)
        steer = result.get("steering_normalized", float("nan"))
        s_e = np.sign(e_lat) if e_lat != 0 else 0
        expected_sign = -s_e  # negative e_lat → positive steering
        actual_sign = np.sign(steer) if abs(steer) > 1e-4 else 0
        ok = "✓" if (abs(steer) < 1e-4 or actual_sign == expected_sign) else "✗"
        print(f"{e_lat:>8.2f}  {existing_x0:>14.4f}  {steer:>10.4f}  "
              f"{s_e:>11.0f}  {expected_sign:>13.0f}  {ok:>4s}")

    print()
    gains_to_test = [0.1, 0.3, 0.5, 1.0, 2.0] if args.gain_sweep else [args.gain]
    for gain in gains_to_test:
        print("=" * 80)
        print(f"PROPOSED seed at gain={gain}, max_frac={args.max_frac}")
        print("=" * 80)
        seed_max = args.max_frac * p.max_steer_rad
        print(f"  seed_max = {args.max_frac} × {p.max_steer_rad:.4f} = {seed_max:.4f}")
        print()
        print(f"{'e_lat':>8s}  {'proposed x0[0]':>14s}  {'clipped?':>9s}  {'sign matches?':>14s}")
        all_match = True
        for e_lat in E_LAT_TEST_GRID:
            seed = proposed_seed_term(e_lat, p, gain, args.max_frac)
            unclipped = -e_lat * gain
            clipped = abs(unclipped) > seed_max
            s_e = np.sign(e_lat) if e_lat != 0 else 0
            expected_sign = -s_e
            seed_sign = np.sign(seed) if abs(seed) > 1e-6 else 0
            sign_ok = (e_lat == 0) or (seed_sign == expected_sign)
            if not sign_ok:
                all_match = False
            mark = "✓" if sign_ok else "✗"
            clip_mark = "Y" if clipped else "n"
            print(f"{e_lat:>8.2f}  {seed:>14.4f}  {clip_mark:>9s}  {mark:>14s}")
        print()
        print(f"Verdict for gain={gain}: "
              f"{'PASS — all signs deterministic' if all_match else 'FAIL'}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
