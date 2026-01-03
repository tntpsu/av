"""
Diagnostic script to verify ground truth following components.

This script helps diagnose why ground truth following isn't working by:
1. Checking if ground truth data is being received
2. Verifying steering calculations
3. Testing coordinate system interpretation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bridge.client import UnityBridgeClient
import time

def diagnose_ground_truth():
    """Diagnose ground truth following issues."""
    print("=" * 70)
    print("GROUND TRUTH FOLLOWING DIAGNOSTIC")
    print("=" * 70)
    print()
    
    bridge = UnityBridgeClient()
    
    # Check bridge connection
    print("1. Checking bridge connection...")
    try:
        response = bridge.session.get(f"{bridge.base_url}/api/health", timeout=1.0)
        if response.status_code == 200:
            print("   ✓ Bridge server is running")
        else:
            print(f"   ✗ Bridge server returned status {response.status_code}")
            return
    except Exception as e:
        print(f"   ✗ Cannot connect to bridge server: {e}")
        print("   → Make sure Unity is running and bridge server is started")
        return
    
    print()
    print("2. Checking ground truth data from Unity...")
    print("   (Waiting for vehicle state...)")
    
    # Try to get vehicle state
    max_attempts = 10
    vehicle_state = None
    for i in range(max_attempts):
        vehicle_state = bridge.get_latest_vehicle_state()
        if vehicle_state:
            break
        time.sleep(0.5)
        print(f"   Attempt {i+1}/{max_attempts}...")
    
    if not vehicle_state:
        print("   ✗ No vehicle state received from Unity")
        print("   → Make sure Unity is running and AVBridge is enabled")
        return
    
    print("   ✓ Vehicle state received")
    print()
    
    # Check for ground truth data
    print("3. Analyzing ground truth data...")
    lane_center = (
        vehicle_state.get('groundTruthLaneCenterX') or
        vehicle_state.get('ground_truth_lane_center_x') or
        vehicle_state.get('ground_truth', {}).get('lane_center_x') or
        None
    )
    
    if lane_center is None:
        print("   ✗ No ground truth lane center data found!")
        print("   → Check Unity console - is GroundTruthReporter enabled?")
        print(f"   → Available keys: {list(vehicle_state.keys())}")
        return
    
    print(f"   ✓ Ground truth lane center: {lane_center:.3f}m")
    
    # Interpret the value
    if abs(lane_center) < 0.1:
        print("   → Car is at lane center (good!)")
    elif lane_center > 0:
        print(f"   → Lane center is {lane_center:.2f}m to the RIGHT of car")
        print(f"   → Car is {lane_center:.2f}m to the LEFT of lane center")
        print(f"   → Should steer RIGHT (positive steering) to correct")
    else:
        print(f"   → Lane center is {abs(lane_center):.2f}m to the LEFT of car")
        print(f"   → Car is {abs(lane_center):.2f}m to the RIGHT of lane center")
        print(f"   → Should steer LEFT (negative steering) to correct")
    
    print()
    
    # Calculate expected steering
    print("4. Calculating expected steering...")
    kp = 1.5  # Current base gain
    expected_steering = kp * lane_center
    expected_steering_clipped = max(-1.0, min(1.0, expected_steering))
    
    print(f"   kp = {kp}")
    print(f"   Expected steering (raw) = {kp} * {lane_center:.3f} = {expected_steering:.3f}")
    print(f"   Expected steering (clipped) = {expected_steering_clipped:.3f}")
    
    if abs(lane_center) > 0.1:
        direction = "RIGHT" if lane_center > 0 else "LEFT"
        print(f"   → With lane_center={lane_center:.3f}m, should steer {direction}")
        if abs(expected_steering_clipped) > 0.5:
            print(f"   ⚠️  Steering is at maximum ({expected_steering_clipped:.3f}) - might cause overshoot")
    
    print()
    
    # Check current control command
    print("5. Checking current control command...")
    try:
        response = bridge.session.get(f"{bridge.base_url}/api/vehicle/control", timeout=1.0)
        if response.status_code == 200:
            control = response.json()
            print(f"   Current steering: {control.get('steering', 'N/A'):.3f}")
            print(f"   Current throttle: {control.get('throttle', 'N/A'):.3f}")
            print(f"   Current brake: {control.get('brake', 'N/A'):.3f}")
            print(f"   Ground truth mode: {control.get('ground_truth_mode', 'N/A')}")
            print(f"   Ground truth speed: {control.get('ground_truth_speed', 'N/A')}")
            
            if control.get('ground_truth_mode'):
                print("   ✓ Ground truth mode is enabled in control command")
            else:
                print("   ✗ Ground truth mode is NOT enabled in control command")
                print("   → This might be the problem! Check if ground truth follower is running")
        else:
            print(f"   ✗ Could not get control command (status {response.status_code})")
    except Exception as e:
        print(f"   ✗ Error getting control command: {e}")
    
    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print()
    print("NEXT STEPS:")
    print("1. Check Unity console for ground truth mode activation logs")
    print("2. Verify in Unity Inspector that CarController.groundTruthMode = true")
    print("3. Check if steering values match expected values")
    print("4. If ground truth mode is not activating, check JSON deserialization in Unity")


if __name__ == "__main__":
    diagnose_ground_truth()

