"""
Analyze lateral error convergence and identify why error isn't dropping below 0.5m.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def analyze_lateral_error_convergence(recording_file: str):
    """Analyze why lateral error isn't converging below 0.5m."""
    
    print("=" * 80)
    print("LATERAL ERROR CONVERGENCE ANALYSIS")
    print("=" * 80)
    print()
    
    with h5py.File(recording_file, 'r') as f:
        # Check what data is available
        print("AVAILABLE DATA:")
        print("-" * 80)
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
        f.visititems(print_structure)
        print()
        
        # Load lateral error data
        if 'control/lateral_error' not in f:
            print("ERROR: control/lateral_error not found in recording!")
            print("Available control datasets:")
            for key in f.get('control', {}).keys():
                print(f"  - control/{key}")
            return
        
        lateral_error = f['control/lateral_error'][:]
        timestamps = f.get('control/timestamps', f.get('camera/timestamps', None))
        if timestamps is not None:
            timestamps = timestamps[:]
            time_seconds = (timestamps - timestamps[0]) if len(timestamps) > 0 else np.arange(len(lateral_error)) / 30.0
        else:
            time_seconds = np.arange(len(lateral_error)) / 30.0  # Assume 30 FPS
        
        # Load related data
        steering = f.get('control/steering', None)
        if steering is not None:
            steering = steering[:]
        else:
            steering = None
            print("WARNING: control/steering not found")
        
        heading_error = f.get('control/heading_error', None)
        if heading_error is not None:
            heading_error = heading_error[:]
        else:
            heading_error = None
            print("WARNING: control/heading_error not found")
        
        total_error = f.get('control/total_error', None)
        if total_error is not None:
            total_error = total_error[:]
        else:
            total_error = None
            print("WARNING: control/total_error not found")
        
        pid_integral = f.get('control/pid_integral', None)
        if pid_integral is not None:
            pid_integral = pid_integral[:]
        else:
            pid_integral = None
            print("WARNING: control/pid_integral not found")
        
        # Trajectory data
        ref_x = None
        ref_y = None
        ref_heading = None
        if 'trajectory/reference_point' in f:
            ref_data = f['trajectory/reference_point'][:]
            if ref_data.shape[1] >= 3:
                ref_x = ref_data[:, 0]
                ref_y = ref_data[:, 1]
                ref_heading = ref_data[:, 2]
        else:
            print("WARNING: trajectory/reference_point not found")
        
        # Lane data
        left_lane_x = f.get('perception/left_lane_x', None)
        right_lane_x = f.get('perception/right_lane_x', None)
        if left_lane_x is not None:
            left_lane_x = left_lane_x[:]
        else:
            print("WARNING: perception/left_lane_x not found")
        if right_lane_x is not None:
            right_lane_x = right_lane_x[:]
        else:
            print("WARNING: perception/right_lane_x not found")
        
        # Vehicle position
        vehicle_pos = None
        vehicle_heading = None
        if 'vehicle/position' in f:
            vehicle_pos = f['vehicle/position'][:]
            if 'vehicle/rotation' in f:
                rotations = f['vehicle/rotation'][:]
                # Extract yaw from quaternion
                vehicle_heading = []
                for rot in rotations:
                    x, y, z, w = rot[0], rot[1], rot[2], rot[3]
                    yaw = np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z))
                    vehicle_heading.append(yaw)
                vehicle_heading = np.array(vehicle_heading)
        else:
            print("WARNING: vehicle/position not found")
        
        print()
        print("=" * 80)
        print("LATERAL ERROR ANALYSIS")
        print("=" * 80)
        print()
        
        abs_lateral_error = np.abs(lateral_error)
        
        # Overall statistics
        print("OVERALL STATISTICS:")
        print("-" * 80)
        print(f"  Total frames: {len(lateral_error)}")
        print(f"  Duration: {time_seconds[-1]:.2f} seconds")
        print(f"  Mean lateral error: {np.mean(abs_lateral_error):.4f} m")
        print(f"  Median lateral error: {np.median(abs_lateral_error):.4f} m")
        print(f"  Std lateral error: {np.std(abs_lateral_error):.4f} m")
        print(f"  Min lateral error: {np.min(abs_lateral_error):.4f} m")
        print(f"  Max lateral error: {np.max(abs_lateral_error):.4f} m")
        print()
        
        # Convergence analysis
        print("CONVERGENCE ANALYSIS:")
        print("-" * 80)
        
        # Split into quarters
        quarter_len = len(lateral_error) // 4
        q1 = abs_lateral_error[:quarter_len]
        q2 = abs_lateral_error[quarter_len:2*quarter_len]
        q3 = abs_lateral_error[2*quarter_len:3*quarter_len]
        q4 = abs_lateral_error[3*quarter_len:]
        
        print(f"  Q1 (0-25%): mean={np.mean(q1):.4f}m, median={np.median(q1):.4f}m")
        print(f"  Q2 (25-50%): mean={np.mean(q2):.4f}m, median={np.median(q2):.4f}m")
        print(f"  Q3 (50-75%): mean={np.mean(q3):.4f}m, median={np.median(q3):.4f}m")
        print(f"  Q4 (75-100%): mean={np.mean(q4):.4f}m, median={np.median(q4):.4f}m")
        
        q1_to_q4 = np.mean(q4) / np.mean(q1) if np.mean(q1) > 0 else 0
        print(f"  Q1→Q4 change: {q1_to_q4:.2f}x ({'improving' if q1_to_q4 < 1.0 else 'worsening'})")
        print()
        
        # Time below threshold
        threshold = 0.5
        below_threshold = abs_lateral_error < threshold
        pct_below = 100 * np.sum(below_threshold) / len(below_threshold)
        print(f"TIME BELOW {threshold}m THRESHOLD:")
        print("-" * 80)
        print(f"  Frames below {threshold}m: {np.sum(below_threshold)}/{len(below_threshold)} ({pct_below:.1f}%)")
        print(f"  Frames above {threshold}m: {len(below_threshold) - np.sum(below_threshold)}/{len(below_threshold)} ({100-pct_below:.1f}%)")
        print()
        
        # Persistent bias analysis
        print("PERSISTENT BIAS ANALYSIS:")
        print("-" * 80)
        mean_error = np.mean(lateral_error)
        std_error = np.std(lateral_error)
        print(f"  Mean lateral error (signed): {mean_error:.4f} m")
        print(f"  Std lateral error: {std_error:.4f} m")
        print(f"  Bias ratio (mean/std): {abs(mean_error)/std_error if std_error > 0 else 0:.2f}")
        if abs(mean_error) > 0.1:
            direction = "LEFT" if mean_error > 0 else "RIGHT"
            print(f"  ⚠️  PERSISTENT BIAS: {abs(mean_error):.4f}m to the {direction}")
        else:
            print(f"  ✓ No significant persistent bias")
        print()
        
        # Oscillation analysis
        print("OSCILLATION ANALYSIS:")
        print("-" * 80)
        sign_changes = np.sum(np.diff(np.sign(lateral_error)) != 0)
        oscillation_freq = sign_changes / time_seconds[-1] if time_seconds[-1] > 0 else 0
        print(f"  Sign changes: {sign_changes}")
        print(f"  Oscillation frequency: {oscillation_freq:.2f} Hz")
        if oscillation_freq > 5.0:
            print(f"  ⚠️  HIGH OSCILLATION: {oscillation_freq:.2f} Hz (should be < 5 Hz)")
        else:
            print(f"  ✓ Oscillation frequency acceptable")
        print()
        
        # Steering correlation
        if steering is not None:
            print("STEERING CORRELATION:")
            print("-" * 80)
            # Check if steering is correcting error
            # lateral_error = ref_x (in vehicle frame: +x = RIGHT)
            # If lateral_error > 0: Reference is to the RIGHT of vehicle → should steer RIGHT (positive)
            # If lateral_error < 0: Reference is to the LEFT of vehicle → should steer LEFT (negative)
            # So they should have SAME SIGN (not opposite!)
            significant = (abs_lateral_error > 0.1) & (np.abs(steering) > 0.05)
            if np.sum(significant) > 0:
                correct_direction = ((lateral_error[significant] > 0) & (steering[significant] > 0)) | \
                                   ((lateral_error[significant] < 0) & (steering[significant] < 0))
                correct_pct = 100 * np.sum(correct_direction) / np.sum(significant)
                print(f"  Correct steering direction: {correct_pct:.1f}%")
                if correct_pct < 70:
                    print(f"  ⚠️  LOW CORRECT DIRECTION: {correct_pct:.1f}% (should be > 70%)")
                else:
                    print(f"  ✓ Steering direction mostly correct")
            else:
                print("  No significant errors to analyze")
            print()
        
        # PID integral analysis
        if pid_integral is not None:
            print("PID INTEGRAL ANALYSIS:")
            print("-" * 80)
            abs_integral = np.abs(pid_integral)
            print(f"  Mean integral: {np.mean(abs_integral):.4f}")
            print(f"  Max integral: {np.max(abs_integral):.4f}")
            
            # Check accumulation
            first_third = abs_integral[:len(abs_integral)//3]
            last_third = abs_integral[2*len(abs_integral)//3:]
            accumulation = np.mean(last_third) / np.mean(first_third) if np.mean(first_third) > 0 else 0
            print(f"  Accumulation ratio: {accumulation:.2f}x")
            if accumulation > 1.5:
                print(f"  ⚠️  INTEGRAL ACCUMULATING: {accumulation:.2f}x (should be < 1.5x)")
            else:
                print(f"  ✓ Integral not accumulating excessively")
            print()
        
        # Trajectory centering analysis
        if ref_x is not None and left_lane_x is not None and right_lane_x is not None:
            print("TRAJECTORY CENTERING ANALYSIS:")
            print("-" * 80)
            lane_center = (left_lane_x + right_lane_x) / 2.0
            trajectory_offset = ref_x - lane_center
            mean_offset = np.mean(np.abs(trajectory_offset))
            print(f"  Mean trajectory offset from lane center: {mean_offset:.4f} m")
            if mean_offset > 0.2:
                print(f"  ⚠️  TRAJECTORY NOT CENTERED: {mean_offset:.4f}m offset (should be < 0.2m)")
            else:
                print(f"  ✓ Trajectory well-centered")
            print()
        
        # Missing data recommendations
        print("=" * 80)
        print("MISSING DATA CHECK:")
        print("=" * 80)
        missing = []
        if steering is None:
            missing.append("control/steering - Critical for understanding control response")
        if heading_error is None:
            missing.append("control/heading_error - Important for understanding total error")
        if total_error is None:
            missing.append("control/total_error - Shows combined error before PID")
        if pid_integral is None:
            missing.append("control/pid_integral - Critical for detecting integral windup")
        if ref_x is None:
            missing.append("trajectory/reference_point - Needed for trajectory analysis")
        if left_lane_x is None or right_lane_x is None:
            missing.append("perception/left_lane_x, right_lane_x - Needed for trajectory centering verification")
        if vehicle_pos is None:
            missing.append("vehicle/position - Needed for position-based analysis")
        
        if missing:
            print("⚠️  MISSING DATA:")
            for item in missing:
                print(f"  - {item}")
        else:
            print("✓ All critical data present")
        print()
        
        # Root cause hypotheses
        print("=" * 80)
        print("ROOT CAUSE HYPOTHESES:")
        print("=" * 80)
        hypotheses = []
        
        if abs(mean_error) > 0.1:
            hypotheses.append(f"PERSISTENT BIAS: {abs(mean_error):.4f}m bias preventing convergence")
        
        if oscillation_freq > 5.0:
            hypotheses.append(f"HIGH OSCILLATION: {oscillation_freq:.2f} Hz preventing stable convergence")
        
        if pid_integral is not None and accumulation > 1.5:
            hypotheses.append(f"PID INTEGRAL WINDUP: {accumulation:.2f}x accumulation causing persistent error")
        
        if ref_x is not None and left_lane_x is not None:
            lane_center = (left_lane_x + right_lane_x) / 2.0
            trajectory_offset = ref_x - lane_center
            if np.mean(np.abs(trajectory_offset)) > 0.2:
                hypotheses.append(f"TRAJECTORY OFFSET: {np.mean(np.abs(trajectory_offset)):.4f}m offset from lane center")
        
        if steering is not None:
            significant = (abs_lateral_error > 0.1) & (np.abs(steering) > 0.05)
            if np.sum(significant) > 0:
                correct_direction = ((lateral_error[significant] > 0) & (steering[significant] > 0)) | \
                                   ((lateral_error[significant] < 0) & (steering[significant] < 0))
                correct_pct = 100 * np.sum(correct_direction) / np.sum(significant)
                if correct_pct < 70:
                    hypotheses.append(f"STEERING DIRECTION ERRORS: {correct_pct:.1f}% correct (should be > 70%)")
        
        if q1_to_q4 > 1.1:
            hypotheses.append(f"ERROR INCREASING OVER TIME: {q1_to_q4:.2f}x increase (system degrading)")
        elif q1_to_q4 < 0.9:
            hypotheses.append(f"ERROR DECREASING: {q1_to_q4:.2f}x decrease (system improving)")
        else:
            hypotheses.append(f"ERROR STABLE: {q1_to_q4:.2f}x change (not converging)")
        
        if hypotheses:
            for i, hyp in enumerate(hypotheses, 1):
                print(f"  {i}. {hyp}")
        else:
            print("  No clear issues identified - system may be operating as expected")
        print()
        
        # Recommendations
        print("=" * 80)
        print("RECOMMENDATIONS:")
        print("=" * 80)
        recommendations = []
        
        if pct_below < 50:
            recommendations.append(f"Increase controller aggressiveness or reduce threshold (only {pct_below:.1f}% below {threshold}m)")
        
        if abs(mean_error) > 0.1:
            recommendations.append("Address persistent bias - check trajectory centering, camera offset, or lane detection")
        
        if oscillation_freq > 5.0:
            recommendations.append("Reduce controller gains or add more damping to reduce oscillation")
        
        if pid_integral is not None and accumulation > 1.5:
            recommendations.append("Improve PID integral reset/decay mechanisms")
        
        if ref_x is not None and left_lane_x is not None:
            lane_center = (left_lane_x + right_lane_x) / 2.0
            trajectory_offset = ref_x - lane_center
            if np.mean(np.abs(trajectory_offset)) > 0.2:
                recommendations.append("Fix trajectory centering - ensure reference point is at lane center")
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        else:
            print("  System appears to be operating well - consider if 0.5m threshold is appropriate")
        print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Find latest recording
        recordings = sorted(Path('data/recordings').glob('*.h5'), key=lambda p: p.stat().st_mtime, reverse=True)
        if recordings:
            recording_file = str(recordings[0])
            print(f"Using latest recording: {recording_file}")
        else:
            print("No recordings found. Usage: python tools/analyze_lateral_error_convergence.py <recording.h5>")
            sys.exit(1)
    else:
        recording_file = sys.argv[1]
    
    analyze_lateral_error_convergence(recording_file)

