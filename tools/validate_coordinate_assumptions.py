#!/usr/bin/env python3
"""
Validate coordinate conversion assumptions.

This script checks if our assumptions about camera parameters are correct
by analyzing recorded data and comparing with expected values.
"""

import h5py
import numpy as np
from pathlib import Path
import sys


def analyze_coordinate_conversion():
    """Analyze coordinate conversion assumptions from recorded data."""
    
    print("="*80)
    print("COORDINATE CONVERSION ASSUMPTIONS VALIDATION")
    print("="*80)
    print()
    
    # Find latest recording
    recordings_dir = Path('data/recordings')
    if not recordings_dir.exists():
        print("ERROR: No recordings directory found")
        return False
    
    recordings = sorted(recordings_dir.glob('*.h5'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not recordings:
        print("ERROR: No recordings found")
        return False
    
    recording_file = recordings[0]
    print(f"Analyzing: {recording_file.name}")
    print()
    
    with h5py.File(recording_file, 'r') as f:
        # Check what data we have
        if 'perception/left_lane_x' not in f or 'perception/right_lane_x' not in f:
            print("ERROR: Recording doesn't have lane position data")
            return False
        
        left_lane_x = np.array(f['perception/left_lane_x'])
        right_lane_x = np.array(f['perception/right_lane_x'])
        num_lanes = np.array(f['perception/num_lanes_detected'])
        
        # Calculate lane widths
        lane_widths = []
        for i in range(len(num_lanes)):
            if num_lanes[i] == 2 and not np.isnan(left_lane_x[i]) and not np.isnan(right_lane_x[i]):
                width = abs(right_lane_x[i] - left_lane_x[i])
                lane_widths.append(width)
        
        if not lane_widths:
            print("ERROR: No dual-lane detections in recording")
            return False
        
        lane_widths = np.array(lane_widths)
        mean_width = np.mean(lane_widths)
        std_width = np.std(lane_widths)
        
        print("1. LANE WIDTH ANALYSIS")
        print("-" * 80)
        print(f"   Frames with dual lanes: {len(lane_widths)}")
        print(f"   Mean lane width: {mean_width:.3f}m")
        print(f"   Std lane width: {std_width:.3f}m")
        print(f"   Expected: ~3.5m (standard lane width)")
        print(f"   Error: {abs(mean_width - 3.5):.3f}m ({abs(mean_width - 3.5)/3.5*100:.1f}%)")
        print()
        
        # Check if error is systematic
        if mean_width > 3.5 * 1.5:
            print("   ⚠️  ERROR: Lane width is TOO LARGE (systematic error)")
            print(f"      Ratio: {mean_width / 3.5:.2f}x too large")
            print("      Possible causes:")
            print("        - Camera height wrong (affects distance estimation)")
            print("        - FOV interpretation wrong (vertical vs horizontal)")
            print("        - Pixel-to-meter conversion wrong")
        elif mean_width < 3.5 * 0.7:
            print("   ⚠️  ERROR: Lane width is TOO SMALL (systematic error)")
            print(f"      Ratio: {mean_width / 3.5:.2f}x too small")
        else:
            print("   ✓ Lane width is reasonable")
        
        if std_width > 1.0:
            print(f"   ⚠️  WARNING: High variance (std={std_width:.3f}m)")
            print("      Coordinate conversion is inconsistent")
        print()
        
        # Analyze assumptions
        print("2. ASSUMPTIONS CHECK")
        print("-" * 80)
        
        # Load config
        config_path = Path('config/av_stack_config.yaml')
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as cfg_file:
                config = yaml.safe_load(cfg_file)
                trajectory_cfg = config.get('trajectory', {})
                camera_fov = trajectory_cfg.get('camera_fov', 60.0)
                camera_height = trajectory_cfg.get('camera_height', 0.5)
                image_width = trajectory_cfg.get('image_width', 640)
                image_height = trajectory_cfg.get('image_height', 480)
        else:
            print("   WARNING: Config file not found, using defaults")
            camera_fov = 60.0
            camera_height = 0.5
            image_width = 640
            image_height = 480
        
        print(f"   Config camera_fov: {camera_fov}°")
        print(f"   Config camera_height: {camera_height}m")
        print(f"   Config image_size: {image_width}x{image_height}px")
        print()
        
        # Unity values (from prefab inspection)
        print("   Unity AVCamera (from CarPrefab.prefab):")
        print("     - FOV: 60° (field of view: 60)")
        print("     - Position relative to car: {x: 0, y: 1.2, z: 1.0}")
        print("     - Camera height above ground: UNKNOWN (depends on car height)")
        print()
        
        # Calculate expected error
        print("3. ERROR ANALYSIS")
        print("-" * 80)
        
        # If lane width is 2.2x too large, what could cause it?
        error_ratio = mean_width / 3.5
        
        print(f"   Error ratio: {error_ratio:.2f}x")
        print()
        print("   Possible causes:")
        print()
        
        # Cause 1: FOV interpretation
        print("   A. FOV Interpretation:")
        print("      - Unity uses VERTICAL FOV by default (m_FOVAxisMode: 0)")
        print("      - Our code assumes HORIZONTAL FOV")
        print("      - For 640x480 image:")
        print("        * If 60° is vertical: horizontal FOV ≈ 73.7°")
        print("        * If 60° is horizontal: vertical FOV ≈ 46.8°")
        print("      - If we use 60° as horizontal but Unity uses vertical:")
        print("        * We underestimate FOV → overestimate pixel-to-meter")
        print("        * This would make lane width TOO SMALL (not our issue)")
        print("      - If we use 60° as vertical but Unity uses horizontal:")
        print("        * We overestimate FOV → underestimate pixel-to-meter")
        print("        * This would make lane width TOO LARGE (matches our issue!)")
        print()
        
        # Cause 2: Camera height
        print("   B. Camera Height:")
        print(f"      - Config assumes: {camera_height}m")
        print("      - Unity camera y position: 1.2m relative to car")
        print("      - Estimated car height: ~0.7m")
        print("      - Actual camera height: ~1.9m (3.8x too large!)")
        print("      - BUT: Camera height mainly affects distance estimation")
        print("      - If we use lookahead_distance directly, height shouldn't matter")
        print("      - UNLESS: Initial pixel_to_meter calculation uses wrong height")
        print()
        
        # Cause 3: Distance calculation
        print("   C. Distance Calculation:")
        print("      - Our code uses lookahead_distance when provided")
        print("      - But initial pixel_to_meter uses typical_lookahead=15m")
        print("      - If actual lookahead is different, conversion is wrong")
        print()
        
        # Calculate what FOV would give correct lane width
        print("4. CALIBRATION CALCULATION")
        print("-" * 80)
        
        # Reverse engineer: what FOV would give correct lane width?
        # At 10m distance, for 3.5m lane:
        # pixel_separation = 3.5m / pixel_to_meter
        # pixel_to_meter = width_at_distance / image_width
        # width_at_distance = 2 * distance * tan(fov/2)
        # So: 3.5 = pixel_sep * (2 * 10 * tan(fov/2)) / 640
        
        # From recorded data, we know mean_width = 7.830m
        # This is 2.2x too large
        # So pixel_to_meter is 2.2x too large
        # Which means width_at_distance is 2.2x too large
        # Which means tan(fov/2) is 2.2x too large
        # Which means fov is too large
        
        # If correct FOV is F, and we're using F', then:
        # tan(F'/2) / tan(F/2) = 2.2
        # For F = 60°: tan(30°) = 0.577
        # tan(F'/2) = 0.577 * 2.2 = 1.27
        # F'/2 = arctan(1.27) = 51.8°
        # F' = 103.6°
        
        # But wait, if we're using 60° as horizontal but Unity uses vertical:
        # Unity vertical FOV = 60°
        # Unity horizontal FOV = 2 * atan(tan(30°) * 640/480) = 73.7°
        # If we use 60° as horizontal, we underestimate
        # But our error is overestimate...
        
        # Let's try: if Unity uses 60° horizontal, but we think it's vertical:
        # We calculate: horizontal FOV = 2 * atan(tan(30°) * 640/480) = 73.7°
        # But actual is 60°
        # So we overestimate FOV by 73.7/60 = 1.23x
        # This would make lane width 1.23x too large (not 2.2x)
        
        # What if Unity uses 60° vertical, but we use it as horizontal?
        # We use: 60° horizontal
        # Actual: 60° vertical = 73.7° horizontal
        # We underestimate FOV by 60/73.7 = 0.81x
        # This would make lane width 0.81x (too small, not our issue)
        
        # What if there's a 2x error somewhere else?
        # Maybe we're using distance wrong?
        
        print("   To get correct lane width (3.5m), we need:")
        print(f"   - Current mean: {mean_width:.3f}m")
        print(f"   - Correction factor: {3.5 / mean_width:.3f}x")
        print()
        print("   Possible fixes:")
        print("     1. Verify FOV interpretation (vertical vs horizontal)")
        print("     2. Measure actual camera height in Unity")
        print("     3. Calibrate with known lane width")
        print()
        
        return True


if __name__ == '__main__':
    success = analyze_coordinate_conversion()
    sys.exit(0 if success else 1)

