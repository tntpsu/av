"""
Test steering calculation using data from actual Unity recordings.

This test can be updated with data from recordings to verify:
1. Steering calculation matches recorded values
2. Coordinate system is correct
3. Path curvature interpretation is correct

Usage:
    python3 tests/test_steering_from_recording.py [recording_file.h5]
    
If no file provided, uses latest recording.
"""

import numpy as np
import sys
from pathlib import Path
from typing import Optional

# Import the exact steering calculator
from tests.test_exact_steering_calculation import ExactSteeringCalculator


def load_recording_data(recording_file: str) -> Optional[dict]:
    """
    Load ground truth and control data from recording.
    
    Returns dict with:
    - path_curvatures: array of path curvatures
    - lane_centers: array of lane center positions
    - calculated_steerings: array of calculated steering commands
    - actual_steerings: array of actual steering angles from Unity
    """
    try:
        import h5py
        
        data = {
            'path_curvatures': [],
            'lane_centers': [],
            'calculated_steerings': [],
            'actual_steerings': [],
            'raw_steerings': [],
            'lateral_corrections': []
        }
        
        with h5py.File(recording_file, 'r') as f:
            # Load ground truth data
            if 'ground_truth/path_curvature' in f:
                data['path_curvatures'] = f['ground_truth/path_curvature'][:]
            if 'ground_truth/lane_center_x' in f:
                data['lane_centers'] = f['ground_truth/lane_center_x'][:]
            
            # Load control data
            if 'control/calculated_steering_angle_deg' in f:
                data['calculated_steerings'] = f['control/calculated_steering_angle_deg'][:]
            if 'control/raw_steering' in f:
                data['raw_steerings'] = f['control/raw_steering'][:]
            if 'control/lateral_correction' in f:
                data['lateral_corrections'] = f['control/lateral_correction'][:]
            
            # Load actual steering from vehicle state
            if 'vehicle/steering_angle' in f:
                data['actual_steerings'] = f['vehicle/steering_angle'][:]
        
        return data
    except ImportError:
        print("⚠️  h5py not available - cannot load recording data")
        return None
    except Exception as e:
        print(f"⚠️  Error loading recording: {e}")
        return None


def test_steering_from_recording(recording_file: Optional[str] = None):
    """
    Test steering calculation using data from recording.
    
    This verifies:
    1. Our calculation matches what was recorded
    2. Coordinate system is correct
    3. Path curvature interpretation is correct
    """
    if recording_file is None:
        # Find latest recording
        recordings_dir = Path("data/recordings")
        recordings = sorted(recordings_dir.glob("*.h5"), 
                          key=lambda p: p.stat().st_mtime, reverse=True)
        if not recordings:
            print("⚠️  No recordings found!")
            return
        recording_file = str(recordings[0])
        print(f"Using latest recording: {recording_file}")
    
    print("=" * 70)
    print("TESTING STEERING FROM RECORDING")
    print("=" * 70)
    print()
    
    # Load recording data
    data = load_recording_data(recording_file)
    if data is None:
        print("⚠️  Could not load recording data")
        return
    
    # Check if we have the required data
    if len(data['path_curvatures']) == 0:
        print("⚠️  No path curvature data in recording!")
        print("   This means ground truth path curvature was not recorded.")
        return
    
    if len(data['lane_centers']) == 0:
        print("⚠️  No lane center data in recording!")
        return
    
    # Create calculator
    calc = ExactSteeringCalculator()
    
    # Test: Recalculate steering from recorded inputs
    print("Test 1: Recalculate steering from recorded inputs")
    print("-" * 70)
    
    recalculated_steerings = []
    differences = []
    
    # Sample every 10th frame for testing
    sample_indices = range(0, min(len(data['path_curvatures']), 
                                  len(data['lane_centers'])), 10)
    
    for i in sample_indices:
        curvature = data['path_curvatures'][i]
        lane_center = data['lane_centers'][i]
        
        # Recalculate steering
        steering, metadata = calc.calculate_steering(curvature, lane_center)
        recalculated_steerings.append(steering)
        
        # Compare with recorded (if available)
        if len(data['calculated_steerings']) > i:
            recorded_steering = data['calculated_steerings'][i] / calc.max_steer_angle_deg
            diff = abs(steering - recorded_steering)
            differences.append(diff)
            
            if diff > 0.01:  # > 1% difference
                print(f"  Frame {i}: diff={diff:.4f}, "
                      f"recalc={steering:.4f}, recorded={recorded_steering:.4f}, "
                      f"curvature={curvature:.6f}, lane_center={lane_center:.3f}")
    
    if differences:
        max_diff = max(differences)
        mean_diff = np.mean(differences)
        print(f"  Max difference: {max_diff:.4f}")
        print(f"  Mean difference: {mean_diff:.4f}")
        
        if max_diff < 0.01:
            print("  ✓ PASS: Recalculated steering matches recorded")
        else:
            print(f"  ⚠️  WARNING: Differences found (max={max_diff:.4f})")
    else:
        print("  ⚠️  No recorded calculated steering to compare")
    
    print()
    
    # Test 2: Verify path curvature interpretation
    print("Test 2: Verify path curvature to steering conversion")
    print("-" * 70)
    
    # For non-zero curvatures, verify steering calculation
    non_zero_curvatures = []
    for i in sample_indices:
        curvature = data['path_curvatures'][i]
        if abs(curvature) > 1e-6:
            steering, metadata = calc.calculate_steering(curvature, 0.0)
            expected_angle_rad = np.arctan(calc.wheelbase * curvature)
            expected_steering = expected_angle_rad / calc.max_steer_angle_rad
            
            diff = abs(steering - expected_steering)
            non_zero_curvatures.append((curvature, steering, expected_steering, diff))
            
            if diff > 0.01:
                print(f"  Frame {i}: curvature={curvature:.6f}, "
                      f"steering={steering:.4f}, expected={expected_steering:.4f}, "
                      f"diff={diff:.4f}")
    
    if non_zero_curvatures:
        max_diff = max(diff for _, _, _, diff in non_zero_curvatures)
        if max_diff < 0.01:
            print(f"  ✓ PASS: Path curvature conversion correct (tested {len(non_zero_curvatures)} frames)")
        else:
            print(f"  ⚠️  WARNING: Path curvature conversion issues (max diff={max_diff:.4f})")
    else:
        print("  ⚠️  No non-zero curvatures found in recording")
    
    print()
    
    # Test 3: Verify coordinate system
    print("Test 3: Verify coordinate system interpretation")
    print("-" * 70)
    
    # Check: positive lane_center should give positive steering (for straight path)
    straight_path_samples = []
    for i in sample_indices:
        curvature = data['path_curvatures'][i]
        lane_center = data['lane_centers'][i]
        
        if abs(curvature) < 1e-6 and abs(lane_center) > 0.1:  # Straight path, off center
            steering, _ = calc.calculate_steering(curvature, lane_center)
            
            # Verify: positive lane_center -> positive steering
            if lane_center > 0 and steering <= 0:
                print(f"  ⚠️  Frame {i}: lane_center={lane_center:.3f} (positive) but steering={steering:.3f} (not positive)")
            elif lane_center < 0 and steering >= 0:
                print(f"  ⚠️  Frame {i}: lane_center={lane_center:.3f} (negative) but steering={steering:.3f} (not negative)")
            else:
                straight_path_samples.append((lane_center, steering))
    
    if straight_path_samples:
        print(f"  ✓ PASS: Coordinate system correct (tested {len(straight_path_samples)} frames)")
    else:
        print("  ⚠️  No straight path samples found for testing")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    recording_file = sys.argv[1] if len(sys.argv) > 1 else None
    test_steering_from_recording(recording_file)

