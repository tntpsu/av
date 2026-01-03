"""
Analyze recording to understand why car is going in circles.
Uses data.replay module to avoid numpy dependency issues.
"""

import sys
from pathlib import Path

try:
    from data.replay import DataReplay
except ImportError:
    print("⚠️  Cannot import data.replay - need to run from project root")
    sys.exit(1)


def analyze_circular_drive(recording_file: str):
    """Analyze recording to understand circular driving issue."""
    
    print("=" * 70)
    print(f"ANALYZING CIRCULAR DRIVE ISSUE")
    print(f"Recording: {recording_file}")
    print("=" * 70)
    print()
    
    try:
        replay = DataReplay(recording_file)
    except Exception as e:
        print(f"⚠️  Error opening recording: {e}")
        return
    
    # Get all vehicle states
    vehicle_states = list(replay.get_vehicle_states())
    control_commands = list(replay.get_control_commands())
    
    if not vehicle_states:
        print("⚠️  No vehicle state data found!")
        return
    
    print(f"Recording: {len(vehicle_states)} frames")
    print()
    
    # Analyze positions
    print("POSITION TRAJECTORY:")
    print("-" * 70)
    
    positions = [vs['position'] for vs in vehicle_states]
    x_positions = [p[0] for p in positions]
    z_positions = [p[2] for p in positions]
    
    x_min, x_max = min(x_positions), max(x_positions)
    z_min, z_max = min(z_positions), max(z_positions)
    x_range = x_max - x_min
    z_range = z_max - z_min
    
    print(f"  X range: [{x_min:.2f}, {x_max:.2f}] (span: {x_range:.2f} m)")
    print(f"  Z range: [{z_min:.2f}, {z_max:.2f}] (span: {z_range:.2f} m)")
    
    # Calculate center
    x_center = (x_min + x_max) / 2
    z_center = (z_min + z_max) / 2
    print(f"  Center: X={x_center:.2f}, Z={z_center:.2f}")
    
    # Check if it's a small circle
    max_span = max(x_range, z_range)
    if max_span < 20.0:
        print(f"  ⚠️  WARNING: Movement span is only {max_span:.2f} m - car is going in a small circle!")
    
    print()
    
    # Analyze steering
    print("STEERING ANALYSIS:")
    print("-" * 70)
    
    steering_angles = [vs['steering_angle'] for vs in vehicle_states]
    avg_steering = sum(steering_angles) / len(steering_angles)
    max_steering = max(abs(s) for s in steering_angles)
    
    print(f"  Average steering: {avg_steering:.2f} deg")
    print(f"  Max steering: {max_steering:.2f} deg")
    
    # Count steering direction
    right_steering = sum(1 for s in steering_angles if s > 1.0)
    left_steering = sum(1 for s in steering_angles if s < -1.0)
    neutral = len(steering_angles) - right_steering - left_steering
    
    print(f"  Steering RIGHT (>1°): {right_steering} ({100*right_steering/len(steering_angles):.1f}%)")
    print(f"  Steering LEFT (<-1°): {left_steering} ({100*left_steering/len(steering_angles):.1f}%)")
    print(f"  Steering NEUTRAL: {neutral} ({100*neutral/len(steering_angles):.1f}%)")
    
    if right_steering > len(steering_angles) * 0.7:
        print(f"  ⚠️  WARNING: Car is steering RIGHT most of the time!")
    
    print()
    
    # Try to get ground truth data from HDF5 directly
    print("GROUND TRUTH DATA:")
    print("-" * 70)
    
    try:
        import h5py
        with h5py.File(recording_file, 'r') as f:
            if "ground_truth/path_curvature" in f:
                curvatures = f["ground_truth/path_curvature"][:]
                avg_curvature = sum(curvatures) / len(curvatures)
                max_curvature = max(abs(c) for c in curvatures)
                non_zero = sum(1 for c in curvatures if abs(c) > 1e-6)
                
                print(f"  Path curvature: {len(curvatures)} samples")
                print(f"  Average: {avg_curvature:.6f} 1/m")
                print(f"  Max: {max_curvature:.6f} 1/m")
                print(f"  Non-zero: {non_zero} / {len(curvatures)}")
                
                if avg_curvature > 0.01:
                    print(f"  ⚠️  WARNING: Average curvature is POSITIVE (right turn)!")
                    print(f"     This would cause constant right steering.")
                elif abs(avg_curvature) < 1e-6:
                    print(f"  ⚠️  WARNING: All curvature values are near zero!")
            else:
                print(f"  ⚠️  No path curvature data found!")
            
            if "control/path_curvature_input" in f:
                path_curvature_inputs = f["control/path_curvature_input"][:]
                avg_input = sum(path_curvature_inputs) / len(path_curvature_inputs)
                print(f"  Path curvature input to steering: {len(path_curvature_inputs)} samples")
                print(f"  Average: {avg_input:.6f} 1/m")
                
                if avg_input > 0.01:
                    print(f"  ⚠️  WARNING: Path curvature input is consistently positive!")
            
            if "unity_feedback/ground_truth_mode_active" in f:
                gt_mode_active = f["unity_feedback/ground_truth_mode_active"][:]
                active_count = sum(gt_mode_active)
                print(f"  Ground truth mode active: {active_count} / {len(gt_mode_active)}")
                
                if active_count == 0:
                    print(f"  ⚠️  WARNING: Ground truth mode was NEVER active!")
            
            if "unity_feedback/path_curvature_calculated" in f:
                path_curvature_calc = f["unity_feedback/path_curvature_calculated"][:]
                calc_count = sum(path_curvature_calc)
                print(f"  Path curvature calculated: {calc_count} / {len(path_curvature_calc)}")
                
                if calc_count == 0:
                    print(f"  ⚠️  WARNING: Path curvature was NEVER calculated!")
    except ImportError:
        print("  (h5py not available - cannot read detailed ground truth data)")
    except Exception as e:
        print(f"  (Error reading ground truth data: {e})")
    
    print()
    print("=" * 70)
    print("ROOT CAUSE ANALYSIS:")
    print("=" * 70)
    print()
    
    # Identify likely causes
    causes = []
    
    if avg_steering > 5.0:
        causes.append("Car is consistently steering RIGHT (avg={:.2f}°)".format(avg_steering))
    
    try:
        import h5py
        with h5py.File(recording_file, 'r') as f:
            if "ground_truth/path_curvature" in f:
                curvatures = f["ground_truth/path_curvature"][:]
                avg_curvature = sum(curvatures) / len(curvatures)
                if avg_curvature > 0.01:
                    causes.append("Path curvature is consistently POSITIVE (right turn, avg={:.6f} 1/m)".format(avg_curvature))
            
            if "unity_feedback/ground_truth_mode_active" in f:
                gt_mode_active = f["unity_feedback/ground_truth_mode_active"][:]
                if sum(gt_mode_active) == 0:
                    causes.append("Ground truth mode was NEVER active (using physics mode instead)")
    except:
        pass
    
    if causes:
        print("LIKELY ROOT CAUSES:")
        for i, cause in enumerate(causes, 1):
            print(f"  {i}. {cause}")
    else:
        print("Could not identify specific root cause.")
        print("Check steering calculation and coordinate system.")
    
    print()
    print("=" * 70)
    print("WHY TESTS DIDN'T CATCH THIS:")
    print("=" * 70)
    print()
    print("Our tests verify:")
    print("  ✓ Steering calculation is mathematically correct")
    print("  ✓ Coordinate system interpretation")
    print("  ✓ Path curvature to steering conversion")
    print()
    print("BUT they don't test:")
    print("  ✗ What happens when path curvature is consistently positive")
    print("  ✗ What happens when ground truth mode is not active")
    print("  ✗ Integration with actual Unity path geometry")
    print("  ✗ Whether Unity is sending correct path curvature")
    print()
    print("RECOMMENDATION:")
    print("  Add integration tests that use actual recording data")
    print("  to verify steering behavior matches expected path.")
    print()
    print("=" * 70)
    
    replay.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Find latest recording
        recordings_dir = Path("data/recordings")
        recordings = sorted(recordings_dir.glob("*.h5"), 
                          key=lambda p: p.stat().st_mtime, reverse=True)
        if recordings:
            recording_file = str(recordings[0])
            print(f"Using latest recording: {recording_file}\n")
        else:
            print("No recordings found!")
            sys.exit(1)
    else:
        recording_file = sys.argv[1]
    
    analyze_circular_drive(recording_file)

