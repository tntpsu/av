"""
Analyze recording to understand why car is going in circles.
"""

import sys
import numpy as np
from pathlib import Path

try:
    import h5py
except ImportError:
    print("⚠️  h5py not available - install with: pip install h5py")
    sys.exit(1)


def analyze_circular_drive(recording_file: str):
    """Analyze recording to understand circular driving issue."""
    
    print("=" * 70)
    print(f"ANALYZING CIRCULAR DRIVE ISSUE: {recording_file}")
    print("=" * 70)
    print()
    
    with h5py.File(recording_file, 'r') as f:
        # Get vehicle positions
        if "vehicle/position" not in f:
            print("⚠️  No vehicle position data found!")
            return
        
        positions = f["vehicle/position"][:]
        timestamps = f["vehicle/timestamps"][:]
        speeds = f["vehicle/speed"][:]
        steering_angles = f["vehicle/steering_angle"][:]
        
        print(f"Recording: {len(timestamps)} frames, {timestamps[-1] - timestamps[0]:.2f} seconds")
        print()
        
        # Analyze position trajectory
        print("POSITION TRAJECTORY ANALYSIS:")
        print("-" * 70)
        
        # Extract X and Z positions (Unity: X is right, Z is forward)
        x_positions = positions[:, 0]
        z_positions = positions[:, 2]
        
        # Calculate center of movement
        x_center = np.mean(x_positions)
        z_center = np.mean(z_positions)
        
        # Calculate distances from center
        distances_from_center = np.sqrt((x_positions - x_center)**2 + (z_positions - z_center)**2)
        avg_radius = np.mean(distances_from_center)
        max_radius = np.max(distances_from_center)
        min_radius = np.min(distances_from_center)
        
        print(f"  Center of movement: X={x_center:.2f}, Z={z_center:.2f}")
        print(f"  Average radius: {avg_radius:.2f} m")
        print(f"  Min radius: {min_radius:.2f} m")
        print(f"  Max radius: {max_radius:.2f} m")
        print(f"  Radius variation: {max_radius - min_radius:.2f} m")
        
        # Check if it's a circle
        if max_radius - min_radius < 5.0 and avg_radius < 20.0:
            print(f"  ⚠️  WARNING: Car appears to be going in a small circle!")
            print(f"     (radius variation < 5m, average radius < 20m)")
        
        print()
        
        # Analyze steering
        print("STEERING ANALYSIS:")
        print("-" * 70)
        
        avg_steering = np.mean(steering_angles)
        max_steering = np.max(np.abs(steering_angles))
        steering_std = np.std(steering_angles)
        
        print(f"  Average steering: {avg_steering:.2f} deg")
        print(f"  Max steering: {max_steering:.2f} deg")
        print(f"  Steering std dev: {steering_std:.2f} deg")
        
        # Check if steering is consistently in one direction
        positive_steering = np.sum(steering_angles > 1.0)  # > 1 degree right
        negative_steering = np.sum(steering_angles < -1.0)  # < -1 degree left
        neutral_steering = len(steering_angles) - positive_steering - negative_steering
        
        print(f"  Frames steering RIGHT (>1°): {positive_steering} ({100*positive_steering/len(steering_angles):.1f}%)")
        print(f"  Frames steering LEFT (<-1°): {negative_steering} ({100*negative_steering/len(steering_angles):.1f}%)")
        print(f"  Frames steering NEUTRAL: {neutral_steering} ({100*neutral_steering/len(steering_angles):.1f}%)")
        
        if positive_steering > len(steering_angles) * 0.7:
            print(f"  ⚠️  WARNING: Car is steering RIGHT most of the time!")
        elif negative_steering > len(steering_angles) * 0.7:
            print(f"  ⚠️  WARNING: Car is steering LEFT most of the time!")
        
        print()
        
        # Analyze ground truth data
        print("GROUND TRUTH DATA ANALYSIS:")
        print("-" * 70)
        
        if "ground_truth/lane_center_x" in f:
            lane_centers = f["ground_truth/lane_center_x"][:]
            print(f"  Lane center data: {len(lane_centers)} samples")
            print(f"  Mean lane center: {np.mean(lane_centers):.3f} m")
            print(f"  Std dev: {np.std(lane_centers):.3f} m")
            print(f"  Max error: {np.max(np.abs(lane_centers)):.3f} m")
            
            if np.all(np.abs(lane_centers) < 0.01):
                print(f"  ⚠️  WARNING: All lane center values are near zero!")
                print(f"     This suggests ground truth data may not be working.")
        else:
            print(f"  ⚠️  WARNING: No lane center data found!")
        
        if "ground_truth/path_curvature" in f:
            curvatures = f["ground_truth/path_curvature"][:]
            print(f"  Path curvature data: {len(curvatures)} samples")
            print(f"  Mean curvature: {np.mean(curvatures):.6f} 1/m")
            print(f"  Max curvature: {np.max(np.abs(curvatures)):.6f} 1/m")
            print(f"  Non-zero samples: {np.sum(np.abs(curvatures) > 1e-6)} / {len(curvatures)}")
            
            if np.all(np.abs(curvatures) < 1e-6):
                print(f"  ⚠️  WARNING: All curvature values are zero!")
                print(f"     This means path curvature is not being calculated.")
            else:
                # Check if curvature is consistently positive (right turn)
                positive_curvature = np.sum(curvatures > 1e-6)
                negative_curvature = np.sum(curvatures < -1e-6)
                print(f"  Positive curvature (right turn): {positive_curvature} samples")
                print(f"  Negative curvature (left turn): {negative_curvature} samples")
                
                if positive_curvature > len(curvatures) * 0.7:
                    print(f"  ⚠️  WARNING: Path curvature is mostly POSITIVE (right turn)!")
                    print(f"     This could explain why car is going in circles to the right.")
        else:
            print(f"  ⚠️  WARNING: No path curvature data found!")
        
        print()
        
        # Analyze control commands
        print("CONTROL COMMAND ANALYSIS:")
        print("-" * 70)
        
        if "control/steering" in f:
            control_steerings = f["control/steering"][:]
            print(f"  Control steering commands: {len(control_steerings)} samples")
            print(f"  Mean: {np.mean(control_steerings):.3f}")
            print(f"  Max: {np.max(np.abs(control_steerings)):.3f}")
            print(f"  Std dev: {np.std(control_steerings):.3f}")
            
            # Check if control steering matches actual steering
            if len(control_steerings) == len(steering_angles):
                # Convert control steering (-1 to 1) to degrees for comparison
                max_steer_angle_deg = 30.0  # Assuming 30 degrees max
                control_steering_deg = control_steerings * max_steer_angle_deg
                
                steering_diff = np.abs(steering_angles - control_steering_deg)
                print(f"  Average difference (actual vs control): {np.mean(steering_diff):.2f} deg")
                print(f"  Max difference: {np.max(steering_diff):.2f} deg")
                
                if np.mean(steering_diff) > 5.0:
                    print(f"  ⚠️  WARNING: Large difference between control and actual steering!")
        
        # Check calculated steering
        if "control/calculated_steering_angle_deg" in f:
            calc_steerings = f["control/calculated_steering_angle_deg"][:]
            print(f"  Calculated steering angles: {len(calc_steerings)} samples")
            print(f"  Mean: {np.mean(calc_steerings):.2f} deg")
            print(f"  Max: {np.max(np.abs(calc_steerings)):.2f} deg")
            
            if np.mean(calc_steerings) > 5.0:
                print(f"  ⚠️  WARNING: Calculated steering is consistently positive (right)!")
        
        # Check path curvature input
        if "control/path_curvature_input" in f:
            path_curvature_inputs = f["control/path_curvature_input"][:]
            print(f"  Path curvature inputs: {len(path_curvature_inputs)} samples")
            print(f"  Mean: {np.mean(path_curvature_inputs):.6f} 1/m")
            print(f"  Max: {np.max(np.abs(path_curvature_inputs)):.6f} 1/m")
            
            if np.mean(path_curvature_inputs) > 0.01:
                print(f"  ⚠️  WARNING: Path curvature input is consistently positive!")
                print(f"     This would cause constant right steering.")
        
        print()
        
        # Analyze Unity feedback
        print("UNITY FEEDBACK ANALYSIS:")
        print("-" * 70)
        
        if "unity_feedback/ground_truth_mode_active" in f:
            gt_mode_active = f["unity_feedback/ground_truth_mode_active"][:]
            print(f"  Ground truth mode active: {np.sum(gt_mode_active)} / {len(gt_mode_active)} samples")
            
            if np.sum(gt_mode_active) == 0:
                print(f"  ⚠️  WARNING: Ground truth mode was NEVER active!")
                print(f"     This means Unity was using physics mode, not direct velocity control.")
            elif np.sum(gt_mode_active) < len(gt_mode_active) * 0.9:
                print(f"  ⚠️  WARNING: Ground truth mode was not consistently active!")
        
        if "unity_feedback/path_curvature_calculated" in f:
            path_curvature_calc = f["unity_feedback/path_curvature_calculated"][:]
            print(f"  Path curvature calculated: {np.sum(path_curvature_calc)} / {len(path_curvature_calc)} samples")
            
            if np.sum(path_curvature_calc) == 0:
                print(f"  ⚠️  WARNING: Path curvature was NEVER calculated!")
        
        if "unity_feedback/actual_steering_applied" in f:
            actual_steerings = f["unity_feedback/actual_steering_applied"][:]
            print(f"  Actual steering applied (from Unity): {len(actual_steerings)} samples")
            print(f"  Mean: {np.mean(actual_steerings):.2f} deg")
            print(f"  Max: {np.max(np.abs(actual_steerings)):.2f} deg")
            
            if np.mean(actual_steerings) > 5.0:
                print(f"  ⚠️  WARNING: Unity is applying consistently positive steering!")
        
        print()
        print("=" * 70)
        print("ROOT CAUSE ANALYSIS:")
        print("=" * 70)
        print()
        
        # Try to identify root cause
        issues = []
        
        if "ground_truth/path_curvature" in f:
            curvatures = f["ground_truth/path_curvature"][:]
            if np.mean(curvatures) > 0.01:
                issues.append("Path curvature is consistently positive (right turn)")
        
        if "control/path_curvature_input" in f:
            path_curvature_inputs = f["control/path_curvature_input"][:]
            if np.mean(path_curvature_inputs) > 0.01:
                issues.append("Path curvature input to steering calculation is positive")
        
        if "unity_feedback/ground_truth_mode_active" in f:
            gt_mode_active = f["unity_feedback/ground_truth_mode_active"][:]
            if np.sum(gt_mode_active) == 0:
                issues.append("Ground truth mode was never active (using physics instead)")
        
        if len(issues) > 0:
            print("LIKELY ROOT CAUSES:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("Could not identify specific root cause from data.")
            print("Check steering calculation logic and coordinate system interpretation.")
        
        print()
        print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Find latest recording
        recordings_dir = Path("data/recordings")
        recordings = sorted(recordings_dir.glob("*.h5"), 
                          key=lambda p: p.stat().st_mtime, reverse=True)
        if recordings:
            recording_file = str(recordings[0])
            print(f"Using latest recording: {recording_file}")
        else:
            print("No recordings found!")
            sys.exit(1)
    else:
        recording_file = sys.argv[1]
    
    analyze_circular_drive(recording_file)

