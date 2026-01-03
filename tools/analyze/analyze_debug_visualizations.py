#!/usr/bin/env python3
"""
Comprehensive analysis of debug visualizations correlated with recorded data.
Analyzes vehicle state, perception, control, and trajectory for each debug frame.
"""

import h5py
import numpy as np
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional

def extract_frame_number(filename: str) -> Optional[int]:
    """Extract frame number from filename."""
    match = re.search(r'frame_(\d+)', filename)
    return int(match.group(1)) if match else None

def analyze_debug_frames(recording_path: Optional[Path] = None, 
                        debug_dir: Path = Path('tmp/debug_visualizations')) -> None:
    """
    Analyze debug visualizations correlated with recorded data.
    
    Args:
        recording_path: Path to HDF5 recording file (auto-detects latest if None)
        debug_dir: Directory containing debug visualizations
    """
    # Find latest recording if not specified
    if recording_path is None:
        recordings = sorted(Path('data/recordings').glob('*.h5'), 
                          key=lambda p: p.stat().st_mtime, reverse=True)
        if not recordings:
            print("ERROR: No recordings found in data/recordings/")
            return
        recording_path = recordings[0]
    
    print('='*80)
    print('COMPREHENSIVE DEBUG VISUALIZATION ANALYSIS')
    print('='*80)
    print(f'Recording: {recording_path.name}')
    print()
    
    # Find debug visualization frames
    debug_frames = []
    for f in debug_dir.glob('frame_*.png'):
        if 'combined' not in f.name and 'edges' not in f.name and 'mask' not in f.name and 'histogram' not in f.name:
            frame_num = extract_frame_number(f.name)
            if frame_num is not None:
                debug_frames.append(frame_num)
    
    debug_frames = sorted(set(debug_frames))
    print(f'Found {len(debug_frames)} debug visualization frames: {debug_frames}')
    print()
    
    # Load data from recording
    with h5py.File(recording_path, 'r') as f:
        # Get frame count from any available dataset
        if 'frame_id' in f:
            num_frames = len(f['frame_id'])
        elif 'camera/frame_ids' in f:
            num_frames = len(f['camera/frame_ids'])
        elif 'control/lateral_error' in f:
            num_frames = len(f['control/lateral_error'])
        elif 'perception/num_lanes_detected' in f:
            num_frames = len(f['perception/num_lanes_detected'])
        else:
            num_frames = 0
        
        # Vehicle state
        # Note: vehicle/heading might not exist, use rotation to compute heading
        vehicle_heading = None
        if 'vehicle/rotation' in f:
            # Extract heading from quaternion rotation
            rotations = np.array(f['vehicle/rotation'])
            # Convert quaternion to heading (simplified - just use yaw from quaternion)
            # For Unity, quaternion is (x, y, z, w)
            # Heading is rotation around Y axis (yaw)
            try:
                from scipy.spatial.transform import Rotation
                rots = Rotation.from_quat(rotations[:, [1, 2, 0, 3]])  # Unity order: x,y,z,w -> scipy: x,y,z,w
                euler = rots.as_euler('xyz', degrees=False)
                vehicle_heading = euler[:, 1]  # Y axis rotation (yaw)
            except:
                # Fallback: estimate from quaternion manually
                # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
                w, x, y, z = rotations[:, 3], rotations[:, 0], rotations[:, 1], rotations[:, 2]
                vehicle_heading = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        
        vehicle_position = np.array(f['vehicle/position']) if 'vehicle/position' in f else None
        vehicle_speed = np.array(f['vehicle/speed']) if 'vehicle/speed' in f else None
        vehicle_steering_angle = np.array(f['vehicle/steering_angle']) if 'vehicle/steering_angle' in f else None
        
        # Perception
        num_lanes = np.array(f['perception/num_lanes_detected']) if 'perception/num_lanes_detected' in f else None
        detection_method = None
        if 'perception/detection_method' in f:
            methods = f['perception/detection_method']
            detection_method = []
            for i in range(len(methods)):
                method_bytes = methods[i]
                method = method_bytes.decode('utf-8') if isinstance(method_bytes, bytes) else str(method_bytes)
                detection_method.append(method)
        
        left_lane_x = np.array(f['perception/left_lane_x']) if 'perception/left_lane_x' in f else None
        right_lane_x = np.array(f['perception/right_lane_x']) if 'perception/right_lane_x' in f else None
        
        # Control
        lateral_error = np.array(f['control/lateral_error']) if 'control/lateral_error' in f else None
        heading_error = np.array(f['control/heading_error']) if 'control/heading_error' in f else None
        steering = np.array(f['control/steering']) if 'control/steering' in f else None
        
        # Trajectory
        ref_x = np.array(f['trajectory/reference_point_x']) if 'trajectory/reference_point_x' in f else None
        ref_heading = np.array(f['trajectory/reference_point_heading']) if 'trajectory/reference_point_heading' in f else None
        
        # Analyze each debug frame
        print('='*80)
        print('FRAME-BY-FRAME ANALYSIS')
        print('='*80)
        print(f'Total frames in recording: {num_frames}')
        print(f'Debug frames to analyze: {debug_frames}')
        print()
        
        for frame in debug_frames:
            if frame >= num_frames:
                print(f'Skipping frame {frame} (>= {num_frames})')
                continue
            
            print('-'*80)
            print(f'FRAME {frame}')
            print('-'*80)
            
            # Vehicle state
            if vehicle_heading is not None:
                heading_deg = np.degrees(vehicle_heading[frame])
                print(f'Vehicle Heading: {heading_deg:>7.2f}°', end='')
                if abs(heading_deg) > 5:
                    print(f' {"(TURNING)" if abs(heading_deg) > 10 else "(SLIGHT TURN)"}', end='')
                elif abs(heading_deg) < 1:
                    print(' (STRAIGHT)', end='')
                else:
                    print(' (DRIFTING)', end='')
                print()
            
            if vehicle_position is not None:
                pos_x = vehicle_position[frame, 0]  # Lateral position
                pos_y = vehicle_position[frame, 1]  # Forward position
                print(f'Vehicle Position: X={pos_x:>7.3f}m, Y={pos_y:>7.3f}m', end='')
                if abs(pos_x) > 0.1:
                    print(f' {"(LEFT)" if pos_x < 0 else "(RIGHT)"} of center', end='')
                print()
            
            if vehicle_speed is not None:
                print(f'Vehicle Speed: {vehicle_speed[frame]:>7.2f} m/s')
            
            if vehicle_steering_angle is not None:
                steer_deg = np.degrees(vehicle_steering_angle[frame])
                print(f'Vehicle Steering Angle: {steer_deg:>7.2f}°')
            
            print()
            
            # Perception
            if num_lanes is not None:
                lanes_detected = num_lanes[frame]
                print(f'Lanes Detected: {lanes_detected}', end='')
                if lanes_detected == 0:
                    print(' ⚠️  NO LANES!')
                elif lanes_detected == 1:
                    print(' ⚠️  ONLY ONE LANE')
                else:
                    print(' ✓')
            
            if detection_method is not None:
                print(f'Detection Method: {detection_method[frame]}')
            
            if left_lane_x is not None and right_lane_x is not None:
                left_x = left_lane_x[frame]
                right_x = right_lane_x[frame]
                if not np.isnan(left_x) and not np.isnan(right_x):
                    lane_width = abs(right_x - left_x)
                    lane_center = (left_x + right_x) / 2.0
                    print(f'Lane Positions: Left={left_x:>7.3f}m, Right={right_x:>7.3f}m')
                    print(f'Lane Width: {lane_width:>7.3f}m (expected ~3.5m)')
                    print(f'Lane Center: {lane_center:>7.3f}m from vehicle center')
            
            print()
            
            # Control
            if lateral_error is not None:
                lat_err = lateral_error[frame]
                print(f'Lateral Error: {lat_err:>7.3f}m', end='')
                if abs(lat_err) > 0.5:
                    print(f' ⚠️  LARGE ERROR', end='')
                elif abs(lat_err) > 0.2:
                    print(f' ⚠️  Moderate error', end='')
                else:
                    print(f' ✓', end='')
                if lat_err > 0:
                    print(' (car RIGHT of target)')
                elif lat_err < 0:
                    print(' (car LEFT of target)')
                else:
                    print(' (centered)')
            
            if heading_error is not None:
                head_err_deg = np.degrees(heading_error[frame])
                print(f'Heading Error: {head_err_deg:>7.2f}°')
            
            if steering is not None:
                print(f'Steering Command: {steering[frame]:>7.3f}', end='')
                if abs(steering[frame]) > 0.3:
                    print(f' {"(LEFT)" if steering[frame] < 0 else "(RIGHT)"}', end='')
                print()
            
            print()
            
            # Trajectory
            if ref_x is not None:
                print(f'Reference Point X: {ref_x[frame]:>7.3f}m', end='')
                if abs(ref_x[frame]) > 0.5:
                    print(f' ⚠️  LARGE OFFSET', end='')
                print()
            
            if ref_heading is not None:
                ref_head_deg = np.degrees(ref_heading[frame])
                print(f'Reference Point Heading: {ref_head_deg:>7.2f}°')
            
            print()
            
            # Issues summary
            issues = []
            if num_lanes is not None and num_lanes[frame] < 2:
                issues.append(f"Only {num_lanes[frame]} lane(s) detected (expected 2)")
            if lateral_error is not None and abs(lateral_error[frame]) > 0.5:
                issues.append(f"Large lateral error: {lateral_error[frame]:.3f}m")
            if ref_x is not None and abs(ref_x[frame]) > 0.5:
                issues.append(f"Large trajectory offset: {ref_x[frame]:.3f}m")
            
            if issues:
                print('⚠️  ISSUES:')
                for issue in issues:
                    print(f'   - {issue}')
                print()
        
        # Summary statistics
        print('='*80)
        print('SUMMARY STATISTICS')
        print('='*80)
        print()
        
        if num_lanes is not None:
            frames_0_lanes = np.sum(num_lanes == 0)
            frames_1_lane = np.sum(num_lanes == 1)
            frames_2_lanes = np.sum(num_lanes == 2)
            total = len(num_lanes)
            
            print(f'Lane Detection:')
            print(f'  {frames_2_lanes}/{total} frames ({frames_2_lanes/total*100:.1f}%) with 2 lanes ✓')
            print(f'  {frames_1_lane}/{total} frames ({frames_1_lane/total*100:.1f}%) with 1 lane ⚠️')
            print(f'  {frames_0_lanes}/{total} frames ({frames_0_lanes/total*100:.1f}%) with 0 lanes ✗')
            print()
        
        if lateral_error is not None:
            mean_lat_err = np.mean(np.abs(lateral_error))
            max_lat_err = np.max(np.abs(lateral_error))
            print(f'Lateral Error:')
            print(f'  Mean: {mean_lat_err:.3f}m')
            print(f'  Max: {max_lat_err:.3f}m')
            print()
        
        if vehicle_heading is not None:
            mean_heading = np.mean(np.abs(np.degrees(vehicle_heading)))
            max_heading = np.max(np.abs(np.degrees(vehicle_heading)))
            print(f'Vehicle Heading:')
            print(f'  Mean absolute: {mean_heading:.2f}°')
            print(f'  Max absolute: {max_heading:.2f}°')
            print()

if __name__ == '__main__':
    import sys
    recording_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    analyze_debug_frames(recording_path)

