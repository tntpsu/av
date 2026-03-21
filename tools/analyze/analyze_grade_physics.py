#!/usr/bin/env python3
"""
Step 3D: Grade Physics Diagnostic Analyzer

Examines per-wheel telemetry, angular velocity, and steering actuation
on graded road sections to diagnose the root cause of lateral divergence
at 11+ m/s on hill_highway.

Usage:
    python tools/analyze/analyze_grade_physics.py --latest
    python tools/analyze/analyze_grade_physics.py --recording data/recordings/foo.h5
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))


def find_latest_recording() -> Path:
    rec_dir = REPO_ROOT / "data" / "recordings"
    h5_files = sorted(rec_dir.glob("*.h5"), key=lambda p: p.stat().st_mtime)
    if not h5_files:
        raise FileNotFoundError(f"No recordings in {rec_dir}")
    return h5_files[-1]


def safe_load(f: h5py.File, key: str) -> np.ndarray | None:
    if key in f:
        return np.array(f[key][:])
    return None


def analyze(recording_path: Path):
    print(f"\n{'='*70}")
    print(f"  Grade Physics Diagnostic — {recording_path.name}")
    print(f"{'='*70}\n")

    with h5py.File(recording_path, "r") as f:
        # Core signals
        speed = safe_load(f, "vehicle/speed")
        road_grade = safe_load(f, "vehicle/road_grade")
        ang_vel = safe_load(f, "vehicle/angular_velocity")
        lat_err = safe_load(f, "control/lateral_error")

        # Per-wheel telemetry (Step 3D)
        ws_slip = safe_load(f, "vehicle/wheel_sideways_slip")
        wf_slip = safe_load(f, "vehicle/wheel_forward_slip")
        wc_force = safe_load(f, "vehicle/wheel_contact_force")
        w_sprung = safe_load(f, "vehicle/wheel_sprung_mass")
        wcn_y = safe_load(f, "vehicle/wheel_contact_normal_y")
        w_steer = safe_load(f, "vehicle/wheel_steer_angle_actual")

        # Commanded steering
        cmd_steer = safe_load(f, "control/steering")
        e_stop = safe_load(f, "control/emergency_stop")

    n_frames = len(speed) if speed is not None else 0
    print(f"Frames: {n_frames}")

    if road_grade is None or n_frames == 0:
        print("ERROR: No road_grade data. Run with grade-enabled config.")
        return

    # Section masks
    graded = np.abs(road_grade) > 0.02
    flat = np.abs(road_grade) <= 0.02
    uphill = road_grade > 0.02
    downhill = road_grade < -0.02
    high_speed = speed > 10.0 if speed is not None else np.zeros(n_frames, dtype=bool)

    print(f"\nRoad grade: {np.sum(graded)}/{n_frames} graded frames "
          f"({np.sum(uphill)} up, {np.sum(downhill)} down), "
          f"{np.sum(flat)} flat")
    print(f"High-speed (>10 m/s): {np.sum(high_speed)} frames")
    if np.any(graded):
        print(f"Grade range: [{np.min(road_grade[graded]):.4f}, {np.max(road_grade[graded]):.4f}] rad "
              f"({np.min(road_grade[graded])*100:.1f}% to {np.max(road_grade[graded])*100:.1f}%)")

    # E-stop location
    if e_stop is not None:
        estop_frames = np.where(e_stop > 0.5)[0]
        if len(estop_frames) > 0:
            fr = estop_frames[0]
            print(f"\n*** E-STOP at frame {fr} — "
                  f"speed={speed[fr]:.1f} m/s, grade={road_grade[fr]:.4f} rad")
        else:
            print("\nNo e-stop triggered.")

    # ─── 1. Yaw Dynamics ──────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("1. YAW DYNAMICS (angular_velocity.y)")
    print(f"{'─'*50}")
    if ang_vel is not None and ang_vel.ndim == 2:
        yaw_rate = ang_vel[:, 1]  # Y component = yaw
        all_zero = np.all(np.abs(yaw_rate) < 1e-6)
        if all_zero:
            print("  WARNING: Yaw rate is ALL ZEROS — angular velocity still hardcoded!")
        else:
            print(f"  Global: mean={np.mean(yaw_rate):.4f}, std={np.std(yaw_rate):.4f}, "
                  f"P95={np.percentile(np.abs(yaw_rate), 95):.4f} rad/s")
            if np.any(graded & high_speed):
                mask = graded & high_speed
                yr = yaw_rate[:len(mask)][mask[:len(yaw_rate)]]
                print(f"  Graded+fast: mean={np.mean(yr):.4f}, std={np.std(yr):.4f}, "
                      f"P95={np.percentile(np.abs(yr), 95):.4f} rad/s")
                # Check for acceleration (diverging yaw)
                if len(yr) > 10:
                    yaw_accel = np.diff(yr)
                    print(f"  Yaw acceleration: mean={np.mean(yaw_accel):.6f}, "
                          f"max={np.max(np.abs(yaw_accel)):.6f} rad/s²")
    else:
        print("  No angular velocity data.")

    # ─── 2. Per-Wheel Forces ──────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("2. PER-WHEEL CONTACT FORCES")
    print(f"{'─'*50}")
    if wc_force is not None and wc_force.ndim == 2:
        labels = ["FL", "FR", "RL", "RR"]
        all_zero = np.all(np.abs(wc_force) < 1e-6)
        if all_zero:
            print("  WARNING: All wheel forces are ZERO — telemetry not populating!")
        else:
            for i, lbl in enumerate(labels):
                print(f"  {lbl}: mean={np.mean(wc_force[:, i]):.0f}N, "
                      f"std={np.std(wc_force[:, i]):.0f}N")

            front_avg = (wc_force[:, 0] + wc_force[:, 1]) / 2
            rear_avg = (wc_force[:, 2] + wc_force[:, 3]) / 2
            ratio = front_avg / np.maximum(rear_avg, 1.0)

            if np.any(flat):
                print(f"\n  Flat sections — F/R ratio: {np.mean(ratio[flat]):.3f}")
            if np.any(graded):
                print(f"  Graded sections — F/R ratio: {np.mean(ratio[graded]):.3f}")
            if np.any(uphill):
                print(f"  Uphill — F/R ratio: {np.mean(ratio[uphill]):.3f}")
            if np.any(downhill):
                print(f"  Downhill — F/R ratio: {np.mean(ratio[downhill]):.3f}")
    else:
        print("  No wheel contact force data.")

    # ─── 3. Lateral Slip ──────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("3. LATERAL SLIP (per wheel)")
    print(f"{'─'*50}")
    if ws_slip is not None and ws_slip.ndim == 2:
        labels = ["FL", "FR", "RL", "RR"]
        all_zero = np.all(np.abs(ws_slip) < 1e-6)
        if all_zero:
            print("  WARNING: All sideways slip is ZERO — telemetry not populating!")
        else:
            for i, lbl in enumerate(labels):
                vals = ws_slip[:, i]
                print(f"  {lbl}: mean={np.mean(vals):.4f}, P95={np.percentile(np.abs(vals), 95):.4f}")

            left_slip = (ws_slip[:, 0] + ws_slip[:, 2]) / 2
            right_slip = (ws_slip[:, 1] + ws_slip[:, 3]) / 2
            asymmetry = np.abs(left_slip - right_slip)
            print(f"\n  Slip asymmetry (L-R): P50={np.median(asymmetry):.4f}, "
                  f"P95={np.percentile(asymmetry, 95):.4f}")

            if np.any(graded & high_speed):
                mask = graded & high_speed
                mask = mask[:len(asymmetry)]
                print(f"  Graded+fast asymmetry: P95={np.percentile(asymmetry[mask], 95):.4f}")

            # Check for diverging slip
            if e_stop is not None:
                estop_frames = np.where(e_stop > 0.5)[0]
                if len(estop_frames) > 0:
                    fr = estop_frames[0]
                    pre = max(0, fr - 30)
                    print(f"\n  Pre-failure slip (frames {pre}–{fr}):")
                    for i, lbl in enumerate(labels):
                        print(f"    {lbl}: {ws_slip[pre:fr, i].tolist()[:10]}...")
    else:
        print("  No sideways slip data.")

    # ─── 4. Contact Normal ────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("4. CONTACT SURFACE NORMAL Y")
    print(f"{'─'*50}")
    if wcn_y is not None and wcn_y.ndim == 2:
        all_zero = np.all(np.abs(wcn_y) < 1e-6)
        if all_zero:
            print("  WARNING: Contact normal Y is ALL ZEROS!")
        else:
            avg_y = np.mean(wcn_y, axis=1)
            if np.any(flat):
                print(f"  Flat sections: mean normal Y = {np.mean(avg_y[flat]):.4f} "
                      f"(expected ~1.0)")
            if np.any(graded):
                print(f"  Graded sections: mean normal Y = {np.mean(avg_y[graded]):.4f} "
                      f"(< 1.0 confirms tilted surface)")
            # Per-wheel on grade
            if np.any(graded):
                for i, lbl in enumerate(["FL", "FR", "RL", "RR"]):
                    print(f"    {lbl} on grade: mean={np.mean(wcn_y[graded, i]):.4f}")
    else:
        print("  No contact normal data.")

    # ─── 5. Steering Actual vs Commanded ──────────────────────────────
    print(f"\n{'─'*50}")
    print("5. STEERING ACTUAL vs COMMANDED")
    print(f"{'─'*50}")
    if w_steer is not None and cmd_steer is not None:
        # w_steer is WheelCollider steer angle (degrees)
        # cmd_steer is normalized [-1, 1]
        n = min(len(w_steer), len(cmd_steer))
        steer_lag = w_steer[:n] - cmd_steer[:n] * 35.0  # approx max steer angle
        print(f"  Lag (actual - commanded*35): mean={np.mean(steer_lag):.2f}°, "
              f"P95={np.percentile(np.abs(steer_lag), 95):.2f}°")
        if np.any(high_speed[:n]):
            hs = high_speed[:n]
            print(f"  At high speed: mean lag={np.mean(steer_lag[hs]):.2f}°, "
                  f"P95={np.percentile(np.abs(steer_lag[hs]), 95):.2f}°")
    elif w_steer is not None:
        print(f"  WheelCollider steer: mean={np.mean(w_steer):.2f}°, "
              f"range=[{np.min(w_steer):.2f}, {np.max(w_steer):.2f}]°")
    else:
        print("  No wheel steer angle data.")

    # ─── 6. Force Comparison: Grade vs Flat ───────────────────────────
    print(f"\n{'─'*50}")
    print("6. FORCE COMPARISON: GRADE vs FLAT at SIMILAR SPEED")
    print(f"{'─'*50}")
    if wc_force is not None and wc_force.ndim == 2 and speed is not None:
        # Compare at similar speed ranges
        speed_bins = [(8, 10), (10, 12), (12, 14)]
        for lo, hi in speed_bins:
            flat_speed = flat & (speed >= lo) & (speed < hi)
            grade_speed = graded & (speed >= lo) & (speed < hi)
            if np.sum(flat_speed) > 5 and np.sum(grade_speed) > 5:
                flat_total = np.sum(wc_force[flat_speed], axis=1)
                grade_total = np.sum(wc_force[grade_speed], axis=1)
                print(f"\n  Speed {lo}-{hi} m/s:")
                print(f"    Flat: total force mean={np.mean(flat_total):.0f}N "
                      f"(front={np.mean(wc_force[flat_speed, 0] + wc_force[flat_speed, 1]):.0f}N, "
                      f"rear={np.mean(wc_force[flat_speed, 2] + wc_force[flat_speed, 3]):.0f}N)")
                print(f"    Grade: total force mean={np.mean(grade_total):.0f}N "
                      f"(front={np.mean(wc_force[grade_speed, 0] + wc_force[grade_speed, 1]):.0f}N, "
                      f"rear={np.mean(wc_force[grade_speed, 2] + wc_force[grade_speed, 3]):.0f}N)")
    else:
        print("  Insufficient data for force comparison.")

    # ─── 7. Sprung Mass Distribution ─────────────────────────────────
    print(f"\n{'─'*50}")
    print("7. SPRUNG MASS DISTRIBUTION")
    print(f"{'─'*50}")
    if w_sprung is not None and w_sprung.ndim == 2:
        all_zero = np.all(np.abs(w_sprung) < 1e-6)
        if all_zero:
            print("  WARNING: Sprung mass is ALL ZEROS!")
        else:
            for i, lbl in enumerate(["FL", "FR", "RL", "RR"]):
                print(f"  {lbl}: mean={np.mean(w_sprung[:, i]):.1f} kg")
            if np.any(graded):
                print(f"  On grade:")
                for i, lbl in enumerate(["FL", "FR", "RL", "RR"]):
                    print(f"    {lbl}: mean={np.mean(w_sprung[graded, i]):.1f} kg")
    else:
        print("  No sprung mass data.")

    # ─── Summary Hypothesis Check ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("HYPOTHESIS CHECK SUMMARY")
    print(f"{'='*70}")
    checks = [
        ("Contact normal tilted on grade",
         wcn_y is not None and np.any(graded) and np.mean(np.mean(wcn_y[graded], axis=1)) < 0.999),
        ("Front grip reduced on grade (F/R < 1.0)",
         wc_force is not None and np.any(graded) and
         np.mean((wc_force[graded, 0] + wc_force[graded, 1]) /
                 np.maximum(wc_force[graded, 2] + wc_force[graded, 3], 1.0)) < 0.95),
        ("Lateral slip asymmetry on grade",
         ws_slip is not None and np.any(graded & high_speed) and
         np.percentile(np.abs((ws_slip[graded & high_speed, 0] + ws_slip[graded & high_speed, 2]) / 2 -
                              (ws_slip[graded & high_speed, 1] + ws_slip[graded & high_speed, 3]) / 2), 95) > 0.05),
        ("Yaw rate accelerating before failure",
         ang_vel is not None and e_stop is not None and len(np.where(e_stop > 0.5)[0]) > 0),
        ("Angular velocity recording works (not zeros)",
         ang_vel is not None and not np.all(np.abs(ang_vel) < 1e-6)),
    ]
    for desc, result in checks:
        status = "YES" if result else "NO"
        marker = ">>>" if result else "   "
        print(f"  {marker} [{status}] {desc}")

    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Grade physics diagnostic analyzer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--latest", action="store_true", help="Analyze latest recording")
    group.add_argument("--recording", type=str, help="Path to HDF5 recording")
    args = parser.parse_args()

    if args.latest:
        path = find_latest_recording()
    else:
        path = Path(args.recording)

    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)

    analyze(path)


if __name__ == "__main__":
    main()
