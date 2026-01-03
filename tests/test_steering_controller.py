"""
Test steering controller to verify fixes work before Unity testing.

This test reproduces the steering controller issues found in analysis:
1. Heading/lateral error conflicts causing wrong steering sign
2. Low effective gain (0.159 vs config 0.7)
3. Poor correlation between trajectory and steering

The test uses recorded data or synthetic scenarios to verify:
- Steering sign is correct
- Steering gain matches config
- Steering reduces lateral error over time
"""

import sys
from pathlib import Path
import numpy as np
import h5py
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from control.pid_controller import LateralController
from data.formats.data_format import VehicleState


class SteeringControllerTest:
    """Test steering controller with recorded or synthetic data."""
    
    def __init__(self, kp: float = 0.7, ki: float = 0.002, kd: float = 0.5,
                 heading_weight: float = 0.6, lateral_weight: float = 0.4):
        """Initialize test with controller parameters."""
        self.controller = LateralController(
            kp=kp,
            ki=ki,
            kd=kd,
            heading_weight=heading_weight,
            lateral_weight=lateral_weight,
            max_steering=0.5
        )
        self.kp = kp
        self.heading_weight = heading_weight
        self.lateral_weight = lateral_weight
        
    def test_with_recording(self, recording_path: str, max_frames: int = 100) -> Dict:
        """
        Test controller with recorded data.
        
        Args:
            recording_path: Path to HDF5 recording file
            max_frames: Maximum number of frames to test
            
        Returns:
            Dict with test results and metrics
        """
        print(f"Testing with recording: {recording_path}")
        print(f"Max frames: {max_frames}")
        print()
        
        with h5py.File(recording_path, 'r') as f:
            num_frames = min(max_frames, len(f['camera/timestamps']))
            
            # Load data
            ref_x = f['trajectory/reference_point_x'][:num_frames]
            ref_y = f['trajectory/reference_point_y'][:num_frames]
            ref_heading = f['trajectory/reference_point_heading'][:num_frames]
            ref_velocity = f['trajectory/reference_point_velocity'][:num_frames] if 'trajectory/reference_point_velocity' in f else np.full(num_frames, 8.0)
            
            # Load vehicle state (for current heading)
            vehicle_heading = f['vehicle/rotation'][:num_frames]
            # Convert quaternion to heading (simplified - just use yaw)
            # Unity: forward is +Y, so heading is rotation around Y axis
            current_headings = np.zeros(num_frames)
            for i in range(num_frames):
                # Extract yaw from quaternion (simplified)
                q = vehicle_heading[i]
                # Quaternion to euler (yaw only)
                yaw = np.arctan2(2*(q[3]*q[1] + q[0]*q[2]), 1 - 2*(q[1]**2 + q[2]**2))
                current_headings[i] = yaw
            
            # Run controller
            steering_commands = []
            lateral_errors = []
            heading_errors = []
            total_errors = []
            
            for i in range(num_frames):
                reference_point = {
                    'x': float(ref_x[i]),
                    'y': float(ref_y[i]),
                    'heading': float(ref_heading[i]),
                    'velocity': float(ref_velocity[i])
                }
                
                result = self.controller.compute_steering(
                    current_heading=current_headings[i],
                    reference_point=reference_point,
                    return_metadata=True
                )
                
                steering_commands.append(result['steering'])
                lateral_errors.append(result['lateral_error'])
                heading_errors.append(result['heading_error'])
                total_errors.append(result['total_error'])
            
            # Analyze results
            return self._analyze_results(
                ref_x, steering_commands, lateral_errors, 
                heading_errors, total_errors
            )
    
    def test_synthetic_scenario(self, scenario: str = "conflict") -> Dict:
        """
        Test controller with synthetic scenarios.
        
        Args:
            scenario: "conflict" (heading/lateral conflict), "straight", "curve"
            
        Returns:
            Dict with test results
        """
        print(f"Testing synthetic scenario: {scenario}")
        print()
        
        if scenario == "conflict":
            # Scenario: heading_error and lateral_error have opposite signs
            # This reproduces the 67% conflict issue
            num_frames = 50
            ref_x = np.linspace(0.5, 1.5, num_frames)  # Trajectory to the right
            current_headings = np.linspace(0.1, 0.2, num_frames)  # Car heading slightly right
            desired_heading = np.linspace(-0.1, -0.2, num_frames)  # Desired heading left (conflict!)
            
        elif scenario == "straight":
            # Scenario: straight road, no conflicts
            num_frames = 50
            ref_x = np.linspace(0.1, 0.2, num_frames)  # Small offset
            current_headings = np.zeros(num_frames)
            desired_heading = np.zeros(num_frames)
            
        elif scenario == "curve":
            # Scenario: curve to the right
            num_frames = 50
            ref_x = np.linspace(0.5, 2.0, num_frames)  # Increasing offset
            current_headings = np.zeros(num_frames)
            desired_heading = np.linspace(0.0, 0.3, num_frames)  # Increasing heading
        
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
        
        # Run controller
        steering_commands = []
        lateral_errors = []
        heading_errors = []
        total_errors = []
        
        for i in range(num_frames):
            reference_point = {
                'x': float(ref_x[i]),
                'y': 8.0,  # 8m lookahead
                'heading': float(desired_heading[i]),
                'velocity': 8.0
            }
            
            result = self.controller.compute_steering(
                current_heading=current_headings[i],
                reference_point=reference_point,
                return_metadata=True
            )
            
            steering_commands.append(result['steering'])
            lateral_errors.append(result['lateral_error'])
            heading_errors.append(result['heading_error'])
            total_errors.append(result['total_error'])
        
        # Analyze results
        return self._analyze_results(
            ref_x, steering_commands, lateral_errors,
            heading_errors, total_errors
        )
    
    def _analyze_results(self, ref_x: np.ndarray, steering: List[float],
                        lateral_errors: List[float], heading_errors: List[float],
                        total_errors: List[float]) -> Dict:
        """Analyze controller results and return metrics."""
        ref_x = np.array(ref_x)
        steering = np.array(steering)
        lateral_errors = np.array(lateral_errors)
        heading_errors = np.array(heading_errors)
        total_errors = np.array(total_errors)
        
        # 1. Check steering sign correctness
        right_of_car = ref_x > 0.1
        left_of_car = ref_x < -0.1
        
        sign_correct_right = 0
        sign_correct_left = 0
        
        if np.sum(right_of_car) > 0:
            sign_correct_right = np.sum(steering[right_of_car] > 0) / np.sum(right_of_car) * 100
        
        if np.sum(left_of_car) > 0:
            sign_correct_left = np.sum(steering[left_of_car] < 0) / np.sum(left_of_car) * 100
        
        overall_sign_correct = (sign_correct_right + sign_correct_left) / 2.0 if (np.sum(right_of_car) > 0 and np.sum(left_of_car) > 0) else (sign_correct_right if np.sum(right_of_car) > 0 else sign_correct_left)
        
        # 2. Check effective gain
        significant_errors = np.abs(lateral_errors) > 0.1
        if np.sum(significant_errors) > 0:
            effective_gains = steering[significant_errors] / lateral_errors[significant_errors]
            mean_gain = np.mean(effective_gains)
        else:
            mean_gain = 0.0
        
        # 3. Check correlation
        correlation = np.corrcoef(np.abs(ref_x), np.abs(steering))[0, 1] if len(ref_x) > 1 else 0.0
        total_error_correlation = np.corrcoef(total_errors, steering)[0, 1] if len(total_errors) > 1 else 0.0
        
        # 4. Check heading/lateral conflicts
        conflicts = np.sum((lateral_errors > 0) & (heading_errors < 0)) + \
                   np.sum((lateral_errors < 0) & (heading_errors > 0))
        conflict_rate = conflicts / len(lateral_errors) * 100 if len(lateral_errors) > 0 else 0.0
        
        # 5. Check if steering reduces error (simulate)
        # For this test, we can't actually simulate car movement, but we can check
        # if steering direction matches error direction
        error_reduction_frames = 0
        for i in range(len(lateral_errors) - 1):
            if abs(lateral_errors[i]) > 0.01:
                # Steering should be opposite to error
                if (lateral_errors[i] > 0 and steering[i] > 0) or \
                   (lateral_errors[i] < 0 and steering[i] < 0):
                    error_reduction_frames += 1
        
        correction_rate = error_reduction_frames / (len(lateral_errors) - 1) * 100 if len(lateral_errors) > 1 else 0.0
        
        results = {
            'sign_correct_right': sign_correct_right,
            'sign_correct_left': sign_correct_left,
            'overall_sign_correct': overall_sign_correct,
            'mean_effective_gain': mean_gain,
            'expected_gain': self.kp,
            'correlation_ref_steering': correlation,
            'correlation_total_error_steering': total_error_correlation,
            'conflict_rate': conflict_rate,
            'correction_rate': correction_rate,
            'mean_steering': np.mean(np.abs(steering)),
            'mean_lateral_error': np.mean(np.abs(lateral_errors)),
            'mean_heading_error': np.mean(np.abs(heading_errors)),
        }
        
        return results
    
    def print_results(self, results: Dict, test_name: str = "Test"):
        """Print test results in a readable format."""
        print("=" * 70)
        print(f"{test_name} RESULTS")
        print("=" * 70)
        print()
        
        print("1. STEERING SIGN CORRECTNESS:")
        print(f"   Overall: {results['overall_sign_correct']:.1f}%")
        print(f"   When RIGHT: {results['sign_correct_right']:.1f}%")
        print(f"   When LEFT: {results['sign_correct_left']:.1f}%")
        if results['overall_sign_correct'] > 90:
            print("   ✅ PASS")
        elif results['overall_sign_correct'] > 75:
            print("   ⚠️  WARN (should be >90%)")
        else:
            print("   ❌ FAIL (should be >90%)")
        print()
        
        print("2. STEERING GAIN:")
        print(f"   Effective gain: {results['mean_effective_gain']:.3f}")
        print(f"   Expected gain (kp): {results['expected_gain']:.3f}")
        gain_ratio = results['mean_effective_gain'] / results['expected_gain'] if results['expected_gain'] > 0 else 0.0
        print(f"   Ratio: {gain_ratio:.1%}")
        if gain_ratio > 0.7:
            print("   ✅ PASS (gain is reasonable)")
        elif gain_ratio > 0.4:
            print("   ⚠️  WARN (gain is low)")
        else:
            print("   ❌ FAIL (gain is too low)")
        print()
        
        print("3. CORRELATION:")
        print(f"   |ref_x| vs |steering|: {results['correlation_ref_steering']:.3f}")
        print(f"   total_error vs steering: {results['correlation_total_error_steering']:.3f}")
        if results['correlation_total_error_steering'] > 0.7:
            print("   ✅ PASS")
        elif results['correlation_total_error_steering'] > 0.5:
            print("   ⚠️  WARN")
        else:
            print("   ❌ FAIL")
        print()
        
        print("4. HEADING/LATERAL CONFLICTS:")
        print(f"   Conflict rate: {results['conflict_rate']:.1f}%")
        if results['conflict_rate'] < 30:
            print("   ✅ PASS (low conflicts)")
        elif results['conflict_rate'] < 50:
            print("   ⚠️  WARN (moderate conflicts)")
        else:
            print("   ❌ FAIL (high conflicts)")
        print()
        
        print("5. CORRECTION EFFECTIVENESS:")
        print(f"   Correction rate: {results['correction_rate']:.1f}%")
        if results['correction_rate'] > 60:
            print("   ✅ PASS")
        elif results['correction_rate'] > 45:
            print("   ⚠️  WARN")
        else:
            print("   ❌ FAIL")
        print()
        
        # Overall pass/fail
        passes = 0
        total = 5
        
        if results['overall_sign_correct'] > 90:
            passes += 1
        if gain_ratio > 0.4:
            passes += 1
        if results['correlation_total_error_steering'] > 0.5:
            passes += 1
        if results['conflict_rate'] < 50:
            passes += 1
        if results['correction_rate'] > 45:
            passes += 1
        
        print("=" * 70)
        print(f"OVERALL: {passes}/{total} tests passed")
        if passes == total:
            print("✅ ALL TESTS PASSED")
        elif passes >= 3:
            print("⚠️  SOME TESTS FAILED - needs improvement")
        else:
            print("❌ MOST TESTS FAILED - needs significant fixes")
        print()


def main():
    """Run steering controller tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test steering controller")
    parser.add_argument("--recording", type=str, help="Path to recording file (optional)")
    parser.add_argument("--scenario", type=str, default="conflict",
                        choices=["conflict", "straight", "curve"],
                        help="Synthetic scenario to test")
    parser.add_argument("--kp", type=float, default=0.7, help="Proportional gain")
    parser.add_argument("--heading-weight", type=float, default=0.6,
                        help="Heading error weight")
    parser.add_argument("--lateral-weight", type=float, default=0.4,
                        help="Lateral error weight")
    parser.add_argument("--max-frames", type=int, default=100,
                        help="Max frames to test from recording")
    
    args = parser.parse_args()
    
    # Create test
    test = SteeringControllerTest(
        kp=args.kp,
        heading_weight=args.heading_weight,
        lateral_weight=args.lateral_weight
    )
    
    # Run test
    if args.recording:
        results = test.test_with_recording(args.recording, args.max_frames)
        test.print_results(results, f"Recording Test ({Path(args.recording).name})")
    else:
        results = test.test_synthetic_scenario(args.scenario)
        test.print_results(results, f"Synthetic Scenario ({args.scenario})")
    
    # Return exit code based on results
    if results['overall_sign_correct'] > 90 and \
       results['mean_effective_gain'] / args.kp > 0.4 and \
       results['correlation_total_error_steering'] > 0.5:
        return 0  # Pass
    else:
        return 1  # Fail


if __name__ == "__main__":
    sys.exit(main())

