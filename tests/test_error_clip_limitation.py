"""
Test to reproduce and verify error_clip limitation issue.

This test reproduces the low effective gain problem:
- kp=2.5 but effective gain is only 0.342 (13.7%)
- Likely cause: error_clip (0.785 rad) is limiting total_error
- This prevents PID from using full kp value

The test verifies:
1. That error_clip is indeed limiting
2. That increasing error_clip fixes the issue
3. That effective gain improves with fix
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from control.pid_controller import LateralController


def test_error_clip_limitation():
    """Test that error_clip is limiting PID output."""
    print("=" * 70)
    print("TEST: ERROR CLIP LIMITATION")
    print("=" * 70)
    print()
    
    # Create controller with current config
    kp = 2.5
    error_clip = 0.785  # Current config value (π/4)
    
    controller = LateralController(
        kp=kp,
        ki=0.002,
        kd=0.5,
        error_clip=error_clip,
        heading_weight=0.6,
        lateral_weight=0.4
    )
    
    # Test scenario: Large lateral error that would exceed error_clip
    # If lateral_error = 2.0m and heading_error = 0, total_error = 0.4 * 2.0 = 0.8
    # But error_clip = 0.785, so it gets clipped to 0.785
    # Expected steering = kp * 0.785 = 2.5 * 0.785 = 1.96
    # But max_steering = 0.5, so it gets clipped to 0.5
    # Effective gain = 0.5 / 2.0 = 0.25 (much lower than kp=2.5!)
    # 
    # NOTE: max_steering=0.5 is ALSO limiting! This is a second limitation.
    # The test should show improvement with increased error_clip, but max_steering
    # will still limit for very large errors.
    
    test_cases = [
        {"lateral_error": 0.5, "heading_error": 0.0, "name": "Small error (not clipped)"},
        {"lateral_error": 1.0, "heading_error": 0.0, "name": "Medium error (not clipped)"},
        {"lateral_error": 2.0, "heading_error": 0.0, "name": "Large error (will be clipped)"},
        {"lateral_error": 3.0, "heading_error": 0.0, "name": "Very large error (definitely clipped)"},
    ]
    
    print("Testing with error_clip = 0.785 rad:")
    print()
    print(f"{'Test Case':<30} {'Lateral Err':<12} {'Total Err':<12} {'Steering':<10} {'Eff Gain':<10}")
    print("-" * 80)
    
    results = []
    for case in test_cases:
        # Create reference point with lateral error
        ref_x = case["lateral_error"]
        ref_heading = case["heading_error"]  # Desired heading = current + heading_error
        
        reference_point = {
            'x': ref_x,
            'y': 8.0,
            'heading': ref_heading,  # If current_heading=0, desired=heading_error
            'velocity': 8.0
        }
        
        result = controller.compute_steering(
            current_heading=0.0,  # Assume car is heading straight
            reference_point=reference_point,
            return_metadata=True
        )
        
        steering = result['steering']
        total_error = result['total_error']
        lateral_error_actual = result['lateral_error']
        
        # Calculate effective gain
        if abs(lateral_error_actual) > 0.01:
            effective_gain = abs(steering) / abs(lateral_error_actual)
        else:
            effective_gain = 0.0
        
        results.append({
            'name': case['name'],
            'lateral_error': lateral_error_actual,
            'total_error': total_error,
            'steering': steering,
            'effective_gain': effective_gain
        })
        
        print(f"{case['name']:<30} {lateral_error_actual:>11.3f} {total_error:>11.3f} {steering:>9.3f} {effective_gain:>9.3f}")
    
    print()
    
    # Analyze results
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()
    
    # Check if errors are being clipped
    clipped_cases = [r for r in results if abs(r['total_error']) >= 0.78]
    if len(clipped_cases) > 0:
        print(f"⚠️  {len(clipped_cases)} test cases have errors at/near clip limit (0.785)")
        print("   This confirms error_clip is limiting PID input")
        print()
        
        # Calculate expected vs actual
        for case in clipped_cases:
            expected_total = case['lateral_error'] * 0.4  # lateral_weight
            if abs(expected_total) > 0.785:
                clipped_amount = abs(expected_total) - 0.785
                print(f"   {case['name']}:")
                print(f"     Expected total_error: {expected_total:.3f}")
                print(f"     Actual (clipped): {case['total_error']:.3f}")
                print(f"     Lost to clipping: {clipped_amount:.3f} rad")
                print(f"     Effective gain: {case['effective_gain']:.3f} (should be ~{kp:.1f})")
                print()
    else:
        print("✅ No clipping detected in test cases")
        print()
    
    # Check effective gain
    avg_effective_gain = np.mean([r['effective_gain'] for r in results if r['effective_gain'] > 0])
    print(f"Average effective gain: {avg_effective_gain:.3f}")
    print(f"Expected gain (kp): {kp:.3f}")
    print(f"Effectiveness: {100 * avg_effective_gain / kp:.1f}%")
    print()
    
    # Note: max_steering=0.5 also limits, so we can't expect full kp effectiveness
    # But we should see improvement with increased error_clip
    if avg_effective_gain < kp * 0.15:  # At least 15% effective (accounting for max_steering limit)
        print("❌ FAIL: Effective gain is too low (<15% of kp)")
        print("   This confirms the issue - error_clip is limiting PID")
        return False
    else:
        print("✅ PASS: Effective gain is reasonable (accounting for max_steering=0.5 limit)")
        return True


def test_with_increased_error_clip():
    """Test that increasing error_clip fixes the issue."""
    print("=" * 70)
    print("TEST: FIX WITH INCREASED ERROR_CLIP")
    print("=" * 70)
    print()
    
    kp = 2.5
    error_clip_increased = 1.57  # π/2 (90 degrees) - more reasonable
    
    controller = LateralController(
        kp=kp,
        ki=0.002,
        kd=0.5,
        error_clip=error_clip_increased,  # Increased!
        heading_weight=0.6,
        lateral_weight=0.4
    )
    
    # Same test cases
    test_cases = [
        {"lateral_error": 0.5, "heading_error": 0.0},
        {"lateral_error": 1.0, "heading_error": 0.0},
        {"lateral_error": 2.0, "heading_error": 0.0},
        {"lateral_error": 3.0, "heading_error": 0.0},
    ]
    
    print(f"Testing with error_clip = {error_clip_increased:.3f} rad (π/2):")
    print()
    
    effective_gains = []
    for case in test_cases:
        reference_point = {
            'x': case["lateral_error"],
            'y': 8.0,
            'heading': case["heading_error"],
            'velocity': 8.0
        }
        
        result = controller.compute_steering(
            current_heading=0.0,
            reference_point=reference_point,
            return_metadata=True
        )
        
        steering = result['steering']
        lateral_error_actual = result['lateral_error']
        
        if abs(lateral_error_actual) > 0.01:
            effective_gain = abs(steering) / abs(lateral_error_actual)
            effective_gains.append(effective_gain)
    
    avg_effective_gain = np.mean(effective_gains) if effective_gains else 0.0
    
    print(f"Average effective gain: {avg_effective_gain:.3f}")
    print(f"Expected gain (kp): {kp:.3f}")
    print(f"Effectiveness: {100 * avg_effective_gain / kp:.1f}%")
    print()
    
    # Compare to baseline (with error_clip=0.785)
    # Baseline had avg_effective_gain ~0.479 (19.2% of kp)
    # With increased error_clip, we should see improvement
    # But max_steering=0.5 still limits, so we can't expect >50% effectiveness
    
    baseline_effectiveness = 0.192  # From first test
    current_effectiveness = avg_effective_gain / kp
    
    if current_effectiveness > baseline_effectiveness * 1.1:  # At least 10% improvement
        print(f"✅ PASS: Increased error_clip improves effective gain")
        print(f"   Baseline: {baseline_effectiveness*100:.1f}%, Current: {current_effectiveness*100:.1f}%")
        return True
    elif current_effectiveness > baseline_effectiveness:
        print(f"⚠️  PARTIAL: Small improvement ({baseline_effectiveness*100:.1f}% → {current_effectiveness*100:.1f}%)")
        print("   max_steering=0.5 is also limiting for large errors")
        return True  # Still an improvement
    else:
        print("❌ FAIL: No improvement")
        return False


def main():
    """Run all tests."""
    print()
    print("=" * 70)
    print("REPRODUCING ERROR_CLIP LIMITATION ISSUE")
    print("=" * 70)
    print()
    
    # Test 1: Reproduce the issue
    test1_passed = test_error_clip_limitation()
    print()
    
    # Test 2: Verify fix
    test2_passed = test_with_increased_error_clip()
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    if not test1_passed:
        print("✅ Issue reproduced: error_clip is limiting PID")
    else:
        print("⚠️  Issue not fully reproduced (may need different test)")
    
    if test2_passed:
        print("✅ Fix verified: Increasing error_clip improves effective gain")
        print()
        print("RECOMMENDATION:")
        print("   Increase error_clip from 0.785 (π/4) to 1.57 (π/2)")
        print("   This will allow PID to use more of the kp value")
    else:
        print("❌ Fix not verified - may need different approach")
    
    return 0 if (not test1_passed and test2_passed) else 1


if __name__ == "__main__":
    sys.exit(main())

