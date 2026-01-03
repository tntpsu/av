"""
Simple unit test for ground truth steering logic (no dependencies).

Tests the fundamental control law: steering = kp * lane_center
"""

def test_control_law():
    """Test the fundamental control law."""
    kp = 1.5
    
    print("=" * 70)
    print("TESTING FUNDAMENTAL CONTROL LAW: steering = kp * lane_center")
    print("=" * 70)
    print()
    
    test_cases = [
        (0.0, 0.0, "At center"),
        (1.0, 1.5, "1m right of center (center is 1m to right)"),
        (-1.0, -1.5, "1m left of center (center is 1m to left)"),
        (2.0, 3.0, "2m right of center"),
        (-2.0, -3.0, "2m left of center"),
        (4.33, 6.495, "4.33m right (from actual logs)"),
    ]
    
    all_passed = True
    for lane_center, expected_steering, description in test_cases:
        actual_steering = kp * lane_center
        # Clip to [-1, 1] range
        actual_steering_clipped = max(-1.0, min(1.0, actual_steering))
        
        if abs(actual_steering - expected_steering) < 0.001:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
            all_passed = False
        
        print(f"{status}: {description}")
        print(f"        lane_center={lane_center:6.2f}m → steering={actual_steering:7.3f} "
              f"(clipped: {actual_steering_clipped:6.3f})")
        print()
    
    print("=" * 70)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 70)
    print()
    
    # Test coordinate system interpretation
    print("COORDINATE SYSTEM INTERPRETATION:")
    print("-" * 70)
    print("From logs: groundTruthLaneCenterX = 4.33m")
    print("Interpretation:")
    print("  - Positive value = lane center is to the RIGHT of car")
    print("  - This means car is to the LEFT of lane center")
    print("  - To correct: steer RIGHT (positive steering)")
    print("  - Expected: steering = kp * 4.33 = 1.5 * 4.33 = 6.495 (clipped to 1.0)")
    print()
    print("If car is NOT steering right when lane_center=4.33m, then:")
    print("  1. Steering sign might be wrong (flip steering_sign to -1)")
    print("  2. Ground truth mode might not be active (check Unity console)")
    print("  3. Coordinate system might be misinterpreted")
    print()


if __name__ == "__main__":
    test_control_law()

