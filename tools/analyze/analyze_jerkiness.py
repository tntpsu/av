"""
Analyze jerkiness/oscillation in steering during curves.

This script analyzes the latest recording to identify:
1. Steering oscillation patterns (turn -> straight -> correcting)
2. Error oscillation patterns
3. Feedforward vs feedback contributions
4. Root cause of jerkiness
"""

import sys
import numpy as np
import h5py
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_jerkiness(recording_file: str):
    """Analyze jerkiness patterns in the recording."""
    print("=" * 80)
    print("JERKINESS ANALYSIS")
    print("=" * 80)
    print(f"Recording: {recording_file}")
    print()
    
    with h5py.File(recording_file, 'r') as f:
        # Load control data
        control_group = f['control']
        steering = control_group['steering'][:]
        timestamps = control_group['timestamps'][:]
        throttles = control_group['throttle'][:] if 'throttle' in control_group else None
        brakes = control_group['brake'][:] if 'brake' in control_group else None
        path_curvature_input = (
            control_group['path_curvature_input'][:]
            if 'path_curvature_input' in control_group
            else None
        )
        
        # Load vehicle data for longitudinal jerk analysis
        vehicle_group = f.get('vehicle', None)
        vehicle_timestamps = None
        speeds = None
        if vehicle_group is not None:
            if 'timestamps' in vehicle_group:
                vehicle_timestamps = vehicle_group['timestamps'][:]
            if 'speed' in vehicle_group:
                speeds = vehicle_group['speed'][:]

        # Load trajectory data
        trajectory_group = f.get('trajectory', None)
        if trajectory_group is not None:
            ref_x = trajectory_group['reference_point_x'][:] if 'reference_point_x' in trajectory_group else None
            ref_heading = trajectory_group['reference_point_heading'][:] if 'reference_point_heading' in trajectory_group else None
        else:
            ref_x = None
            ref_heading = None
        
        # Load error data if available
        lateral_error = control_group.get('lateral_error', None)
        if lateral_error is not None:
            lateral_error = lateral_error[:]
        heading_error = control_group.get('heading_error', None)
        if heading_error is not None:
            heading_error = heading_error[:]
        
        # Calculate time from start
        if len(timestamps) > 0:
            time_from_start = timestamps - timestamps[0]
        else:
            time_from_start = np.arange(len(steering)) * 0.033
        
        # Analyze first 15 seconds (where initial curve likely is)
        max_time = min(15.0, time_from_start[-1] if len(time_from_start) > 0 else 15.0)
        mask = time_from_start <= max_time
        
        steering_subset = steering[mask]
        time_subset = time_from_start[mask]
        
        print(f"Analyzing first {max_time:.1f} seconds ({np.sum(mask)} frames)")
        print()
        
        # 1. STEERING OSCILLATION ANALYSIS
        print("1. STEERING OSCILLATION:")
        print("-" * 80)
        
        # Count zero crossings (oscillation indicator)
        steering_zero_crossings = 0
        for i in range(1, len(steering_subset)):
            if (steering_subset[i-1] > 0 and steering_subset[i] <= 0) or \
               (steering_subset[i-1] < 0 and steering_subset[i] >= 0):
                steering_zero_crossings += 1
        
        print(f"   Zero crossings: {steering_zero_crossings}")
        print(f"   Steering range: [{np.min(steering_subset):.3f}, {np.max(steering_subset):.3f}]")
        print(f"   Steering std: {np.std(steering_subset):.3f}")
        
        # Calculate steering changes
        steering_changes = np.diff(steering_subset)
        steering_change_rate = np.abs(steering_changes) / np.diff(time_subset)
        print(f"   Max change rate: {np.max(steering_change_rate):.3f} per second")
        print(f"   Mean change rate: {np.mean(steering_change_rate):.3f} per second")
        print()
        
        # 2. PATH CURVATURE vs STEERING
        if ref_heading is not None:
            ref_heading_subset = ref_heading[mask]
            path_curvature = ref_heading_subset
            
            print("2. PATH CURVATURE vs STEERING:")
            print("-" * 80)
            print(f"   Curvature range: [{np.min(path_curvature):.4f}, {np.max(path_curvature):.4f}]")
            print(f"   Curvature std: {np.std(path_curvature):.4f}")
            
            # Check if steering matches curvature
            curvature_sign = np.sign(path_curvature)
            steering_sign = np.sign(steering_subset)
            sign_matches = np.sum(curvature_sign == steering_sign)
            print(f"   Sign matches: {sign_matches}/{len(steering_subset)} ({100*sign_matches/len(steering_subset):.1f}%)")
            
            # Find where curvature changes (curve transitions)
            curvature_changes = np.sum(np.diff(curvature_sign) != 0)
            print(f"   Curvature sign changes: {curvature_changes}")
            print()
            
            # Check if curvature is noisy (rapid changes)
            curvature_changes_magnitude = np.abs(np.diff(path_curvature))
            print(f"   Curvature change magnitude - max: {np.max(curvature_changes_magnitude):.4f}, mean: {np.mean(curvature_changes_magnitude):.4f}")
            if np.max(curvature_changes_magnitude) > 0.1:
                print("   ⚠️  WARNING: Large curvature changes detected - feedforward might be jerky!")
            print()
        
        # 3. ERROR OSCILLATION
        if lateral_error is not None:
            lateral_error_subset = lateral_error[mask]
            
            print("3. LATERAL ERROR OSCILLATION:")
            print("-" * 80)
            print(f"   Error range: [{np.min(lateral_error_subset):.3f}, {np.max(lateral_error_subset):.3f}] m")
            print(f"   Error std: {np.std(lateral_error_subset):.3f} m")
            
            # Count error zero crossings
            error_zero_crossings = 0
            for i in range(1, len(lateral_error_subset)):
                if (lateral_error_subset[i-1] > 0 and lateral_error_subset[i] <= 0) or \
                   (lateral_error_subset[i-1] < 0 and lateral_error_subset[i] >= 0):
                    error_zero_crossings += 1
            
            print(f"   Error zero crossings: {error_zero_crossings}")
            
            # Check correlation between error and steering
            error_steering_corr = np.corrcoef(lateral_error_subset, steering_subset)[0, 1]
            print(f"   Error-steering correlation: {error_steering_corr:.3f}")
            if error_steering_corr < 0:
                print("   ⚠️  WARNING: Negative correlation! Steering might be backwards!")
            print()
        
        # 4. PATTERN DETECTION: Turn -> Straight -> Correcting
        print("4. OSCILLATION PATTERN DETECTION:")
        print("-" * 80)
        
        # Detect pattern: steering goes from non-zero -> zero -> non-zero
        pattern_count = 0
        for i in range(2, len(steering_subset) - 2):
            # Check if we have: non-zero -> near zero -> non-zero (opposite sign)
            prev_steering = steering_subset[i-2]
            curr_steering = steering_subset[i]
            next_steering = steering_subset[i+2]
            
            # Pattern: significant steering -> near zero -> significant steering (opposite)
            if abs(prev_steering) > 0.1 and abs(curr_steering) < 0.05 and abs(next_steering) > 0.1:
                if np.sign(prev_steering) != np.sign(next_steering):
                    pattern_count += 1
        
        print(f"   'Turn -> Straight -> Correcting' patterns: {pattern_count}")
        if pattern_count > 5:
            print("   ⚠️  WARNING: Frequent oscillation pattern detected!")
        print()
        
        # 5. LONGITUDINAL JERK HOTSPOTS
        print("5. LONGITUDINAL JERK HOTSPOTS:")
        print("-" * 80)
        if vehicle_timestamps is not None and speeds is not None and len(speeds) > 2:
            dt = np.diff(vehicle_timestamps)
            dt[dt <= 0] = 1e-3
            alpha = 0.7
            filtered_speed = np.empty_like(speeds)
            filtered_speed[0] = speeds[0]
            for i in range(1, len(speeds)):
                filtered_speed[i] = alpha * filtered_speed[i - 1] + (1.0 - alpha) * speeds[i]
            accel = np.diff(filtered_speed) / dt
            jerk = np.diff(accel) / dt[1:]

            top_n = min(10, len(jerk))
            if top_n > 0:
                top_idx = np.argsort(np.abs(jerk))[-top_n:][::-1]
                print("   Top jerk events:")
                for idx in top_idx:
                    # Map jerk index to original frame index (offset by 2 due to diff)
                    frame_idx = idx + 2
                    t = vehicle_timestamps[frame_idx]
                    speed = speeds[frame_idx]
                    acc = accel[idx + 1] if idx + 1 < len(accel) else accel[-1]
                    j = jerk[idx]
                    steer = steering[frame_idx] if frame_idx < len(steering) else 0.0
                    thr = throttles[frame_idx] if throttles is not None and frame_idx < len(throttles) else 0.0
                    brk = brakes[frame_idx] if brakes is not None and frame_idx < len(brakes) else 0.0
                    curv = (
                        path_curvature_input[frame_idx]
                        if path_curvature_input is not None and frame_idx < len(path_curvature_input)
                        else 0.0
                    )
                    print(
                        f"   frame={frame_idx:4d} t={t:7.3f} "
                        f"speed={speed:6.2f} accel={acc:7.2f} jerk={j:8.2f} "
                        f"thr={thr:4.2f} brk={brk:4.2f} steer={steer:5.2f} "
                        f"curv={curv:7.4f}"
                    )
            else:
                print("   Not enough samples for jerk analysis.")
        else:
            print("   Vehicle speed/timestamps not available.")
        print()

        # 6. ROOT CAUSE ANALYSIS
        print("6. ROOT CAUSE ANALYSIS:")
        print("-" * 80)
        
        issues = []
        
        # Check for excessive zero crossings
        if steering_zero_crossings > len(steering_subset) * 0.1:
            issues.append(f"Excessive steering zero crossings ({steering_zero_crossings}) - steering oscillating")
        
        # Check for rapid steering changes
        if np.max(steering_change_rate) > 2.0:
            issues.append(f"Rapid steering changes (max {np.max(steering_change_rate):.2f}/s) - jerky control")
        
        # Check for noisy curvature
        if ref_heading is not None:
            if np.max(curvature_changes_magnitude) > 0.1:
                issues.append("Noisy path curvature - feedforward might be jerky")
        
        # Check for error oscillation
        if lateral_error is not None:
            if error_zero_crossings > len(lateral_error_subset) * 0.15:
                issues.append(f"Excessive error oscillation ({error_zero_crossings} zero crossings)")
        
        # Check for pattern
        if pattern_count > 5:
            issues.append(f"Frequent 'turn->straight->correcting' pattern ({pattern_count} occurrences)")
        
        if issues:
            print("   Issues detected:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("   No obvious issues detected")
        print()
        
        # 7. RECOMMENDATIONS
        print("7. RECOMMENDATIONS:")
        print("-" * 80)
        
        if steering_zero_crossings > 10:
            print("   - Disable integral resets during curves (already done)")
            print("   - Add feedforward smoothing (already done)")
            print("   - Consider reducing PID gains or increasing damping")
        
        if ref_heading is not None and np.max(curvature_changes_magnitude) > 0.1:
            print("   - Feedforward smoothing should help (already implemented)")
            print("   - Check trajectory planner for noisy curvature output")
        
        if pattern_count > 5:
            print("   - The 'turn->straight->correcting' pattern suggests:")
            print("     * Integral might be resetting too aggressively")
            print("     * Feedforward might be changing too rapidly")
            print("     * Error might be oscillating due to overcorrection")
        
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze jerkiness in steering")
    parser.add_argument("recording", nargs="?", default="data/recordings/recording_20260103_173840.h5",
                      help="Path to recording file")
    
    args = parser.parse_args()
    analyze_jerkiness(args.recording)

