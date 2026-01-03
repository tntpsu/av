"""
Python server for debug visualizer.
Converts HDF5 recordings to JSON and serves camera frames.
"""

import h5py
import numpy as np
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for local development

# Paths
RECORDINGS_DIR = Path(__file__).parent.parent.parent / "data" / "recordings"
DEBUG_VIS_DIR = Path(__file__).parent.parent.parent / "tmp" / "debug_visualizations"


def numpy_to_list(obj):
    """Convert numpy arrays to lists recursively."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_list(item) for item in obj]
    elif isinstance(obj, bytes):
        return obj.decode('utf-8')
    return obj


@app.route('/api/recordings')
def list_recordings():
    """List all available HDF5 recordings."""
    recordings = []
    if RECORDINGS_DIR.exists():
        for file in sorted(RECORDINGS_DIR.glob("*.h5"), reverse=True):
            recordings.append({
                "filename": file.name,
                "path": str(file.relative_to(RECORDINGS_DIR.parent.parent))
            })
    return jsonify(recordings)


@app.route('/api/recording/<filename>/frames')
def get_frame_count(filename):
    """Get total number of frames in recording."""
    filepath = RECORDINGS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "Recording not found"}), 404
    
    try:
        with h5py.File(filepath, 'r') as f:
            if 'camera/images' in f:
                frame_count = len(f['camera/images'])
            elif 'camera/frame_ids' in f:
                frame_count = len(f['camera/frame_ids'])
            else:
                frame_count = 0
            return jsonify({"frame_count": frame_count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/recording/<filename>/frame/<int:frame_index>')
def get_frame_data(filename, frame_index):
    """Get all data for a specific frame.
    
    Uses timestamp-based synchronization to match data from different datasets
    that may have different lengths or frame indices.
    """
    filepath = RECORDINGS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "Recording not found"}), 404
    
    try:
        with h5py.File(filepath, 'r') as f:
            frame_data = {}
            
            # Get camera frame timestamp (this is our reference)
            camera_timestamp = None
            if 'camera/images' in f and frame_index < len(f['camera/images']):
                camera_timestamp = float(f['camera/timestamps'][frame_index])
                frame_data['camera'] = {
                    'timestamp': camera_timestamp,
                    'frame_id': int(f['camera/frame_ids'][frame_index])
                }
            else:
                # No camera frame at this index - return empty
                return jsonify(frame_data)
            
            # Helper function to find closest timestamp match
            def find_closest_index(timestamps, target_ts, max_diff=0.1):
                """Find index with closest timestamp, within max_diff seconds."""
                if len(timestamps) == 0:
                    return None
                diffs = np.abs(timestamps - target_ts)
                closest_idx = np.argmin(diffs)
                if diffs[closest_idx] <= max_diff:
                    return int(closest_idx)
                return None
            
            # Initialize indices
            vehicle_idx = None
            
            # Vehicle state - find closest by timestamp
            if 'vehicle/timestamps' in f and len(f['vehicle/timestamps']) > 0:
                vehicle_timestamps = np.array(f['vehicle/timestamps'])
                vehicle_idx = find_closest_index(vehicle_timestamps, camera_timestamp)
                
                if vehicle_idx is not None and vehicle_idx < len(f['vehicle/position']):
                    frame_data['vehicle'] = {
                        # NEW: Camera calibration - actual y pixel where 8m appears
                        'camera_8m_screen_y': float(f['vehicle/camera_8m_screen_y'][vehicle_idx]) if 'vehicle/camera_8m_screen_y' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/camera_8m_screen_y']) else None,
                        # NEW: Camera FOV values from Unity
                        'camera_field_of_view': float(f['vehicle/camera_field_of_view'][vehicle_idx]) if 'vehicle/camera_field_of_view' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/camera_field_of_view']) else None,
                        'camera_horizontal_fov': float(f['vehicle/camera_horizontal_fov'][vehicle_idx]) if 'vehicle/camera_horizontal_fov' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/camera_horizontal_fov']) else None,
                        # NEW: Camera position and forward direction from Unity
                        'camera_pos_x': float(f['vehicle/camera_pos_x'][vehicle_idx]) if 'vehicle/camera_pos_x' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/camera_pos_x']) else None,
                        'camera_pos_y': float(f['vehicle/camera_pos_y'][vehicle_idx]) if 'vehicle/camera_pos_y' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/camera_pos_y']) else None,
                        'camera_pos_z': float(f['vehicle/camera_pos_z'][vehicle_idx]) if 'vehicle/camera_pos_z' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/camera_pos_z']) else None,
                        'camera_forward_x': float(f['vehicle/camera_forward_x'][vehicle_idx]) if 'vehicle/camera_forward_x' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/camera_forward_x']) else None,
                        'camera_forward_y': float(f['vehicle/camera_forward_y'][vehicle_idx]) if 'vehicle/camera_forward_y' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/camera_forward_y']) else None,
                        'camera_forward_z': float(f['vehicle/camera_forward_z'][vehicle_idx]) if 'vehicle/camera_forward_z' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/camera_forward_z']) else None,
                        # NEW: Debug fields for diagnosing ground truth offset issues
                        'road_center_at_car_x': float(f['vehicle/road_center_at_car_x'][vehicle_idx]) if 'vehicle/road_center_at_car_x' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/road_center_at_car_x']) else None,
                        'road_center_at_car_y': float(f['vehicle/road_center_at_car_y'][vehicle_idx]) if 'vehicle/road_center_at_car_y' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/road_center_at_car_y']) else None,
                        'road_center_at_car_z': float(f['vehicle/road_center_at_car_z'][vehicle_idx]) if 'vehicle/road_center_at_car_z' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/road_center_at_car_z']) else None,
                        'road_center_at_lookahead_x': float(f['vehicle/road_center_at_lookahead_x'][vehicle_idx]) if 'vehicle/road_center_at_lookahead_x' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/road_center_at_lookahead_x']) else None,
                        'road_center_at_lookahead_y': float(f['vehicle/road_center_at_lookahead_y'][vehicle_idx]) if 'vehicle/road_center_at_lookahead_y' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/road_center_at_lookahead_y']) else None,
                        'road_center_at_lookahead_z': float(f['vehicle/road_center_at_lookahead_z'][vehicle_idx]) if 'vehicle/road_center_at_lookahead_z' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/road_center_at_lookahead_z']) else None,
                        'road_center_reference_t': float(f['vehicle/road_center_reference_t'][vehicle_idx]) if 'vehicle/road_center_reference_t' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/road_center_reference_t']) else None,
                        # NEW: Calculate actual 2D distance from camera to road center at lookahead (XZ plane, ignoring Y)
                        'actual_distance_to_road_center': None,  # Will calculate below if we have the data
                        'timestamp': float(vehicle_timestamps[vehicle_idx]),
                        'position': numpy_to_list(f['vehicle/position'][vehicle_idx]),
                        'rotation': numpy_to_list(f['vehicle/rotation'][vehicle_idx]),
                        'velocity': numpy_to_list(f['vehicle/velocity'][vehicle_idx]),
                        'angular_velocity': numpy_to_list(f['vehicle/angular_velocity'][vehicle_idx]),
                        'speed': float(f['vehicle/speed'][vehicle_idx]),
                        'steering_angle': float(f['vehicle/steering_angle'][vehicle_idx])
                    }
                    
                    # Calculate actual distance from camera to road center at lookahead (2D, XZ plane)
                    if (frame_data['vehicle'].get('camera_pos_x') is not None and 
                        frame_data['vehicle'].get('camera_pos_z') is not None and
                        frame_data['vehicle'].get('road_center_at_lookahead_x') is not None and
                        frame_data['vehicle'].get('road_center_at_lookahead_z') is not None):
                        cam_x = frame_data['vehicle']['camera_pos_x']
                        cam_z = frame_data['vehicle']['camera_pos_z']
                        road_x = frame_data['vehicle']['road_center_at_lookahead_x']
                        road_z = frame_data['vehicle']['road_center_at_lookahead_z']
                        actual_dist = np.sqrt((road_x - cam_x)**2 + (road_z - cam_z)**2)
                        frame_data['vehicle']['actual_distance_to_road_center'] = float(actual_dist)
            
            # Perception data - find closest by timestamp
            if 'perception/timestamps' in f and len(f['perception/timestamps']) > 0:
                perception_timestamps = np.array(f['perception/timestamps'])
                perception_idx = find_closest_index(perception_timestamps, camera_timestamp)
                
                if perception_idx is not None and perception_idx < len(f['perception/confidence']):
                    frame_data['perception'] = {
                        'confidence': float(f['perception/confidence'][perception_idx]),
                        'detection_method': f['perception/detection_method'][perception_idx].decode('utf-8') if isinstance(f['perception/detection_method'][perception_idx], bytes) else str(f['perception/detection_method'][perception_idx]),
                        'num_lanes_detected': int(f['perception/num_lanes_detected'][perception_idx]) if 'perception/num_lanes_detected' in f and perception_idx < len(f['perception/num_lanes_detected']) else None
                    }
                    
                    # Try new name first, fall back to old name for backward compatibility
                    if 'perception/left_lane_line_x' in f and perception_idx < len(f['perception/left_lane_line_x']):
                        frame_data['perception']['left_lane_line_x'] = float(f['perception/left_lane_line_x'][perception_idx])
                    elif 'perception/left_lane_x' in f and perception_idx < len(f['perception/left_lane_x']):
                        frame_data['perception']['left_lane_line_x'] = float(f['perception/left_lane_x'][perception_idx])  # Backward compatibility
                    if 'perception/right_lane_line_x' in f and perception_idx < len(f['perception/right_lane_line_x']):
                        frame_data['perception']['right_lane_line_x'] = float(f['perception/right_lane_line_x'][perception_idx])
                    elif 'perception/right_lane_x' in f and perception_idx < len(f['perception/right_lane_x']):
                        frame_data['perception']['right_lane_line_x'] = float(f['perception/right_lane_x'][perception_idx])  # Backward compatibility
                    
                    # Read lane line coefficients if available
                    if 'perception/lane_line_coefficients' in f and perception_idx < len(f['perception/lane_line_coefficients']):
                        coeffs_array = f['perception/lane_line_coefficients'][perception_idx]
                        num_lanes = frame_data['perception'].get('num_lanes_detected', 0)
                        num_coeffs_per_lane = 3  # 2nd degree polynomial: ax^2 + bx + c
                        
                        # Handle both flattened array and nested array formats
                        if hasattr(coeffs_array, '__len__') and len(coeffs_array) > 0:
                            # Check if it's already a nested array (list of arrays)
                            if num_lanes > 0 and isinstance(coeffs_array[0], (list, np.ndarray)) and len(coeffs_array[0]) == num_coeffs_per_lane:
                                # Already nested - convert to list of lists
                                lane_coeffs_list = [list(c) if isinstance(c, np.ndarray) else c for c in coeffs_array]
                                frame_data['perception']['lane_line_coefficients'] = lane_coeffs_list
                            else:
                                # Flattened array: [coeff0_lane0, coeff1_lane0, coeff2_lane0, coeff0_lane1, ...]
                                expected_length = num_lanes * num_coeffs_per_lane
                                if len(coeffs_array) >= expected_length:
                                    lane_coeffs_list = []
                                    for i in range(num_lanes):
                                        start_idx = i * num_coeffs_per_lane
                                        end_idx = start_idx + num_coeffs_per_lane
                                        lane_coeffs = coeffs_array[start_idx:end_idx].tolist()
                                        lane_coeffs_list.append(lane_coeffs)
                                    frame_data['perception']['lane_line_coefficients'] = lane_coeffs_list
            
            # Trajectory data - find closest by timestamp
            if 'trajectory/timestamps' in f and len(f['trajectory/timestamps']) > 0:
                trajectory_timestamps = np.array(f['trajectory/timestamps'])
                trajectory_idx = find_closest_index(trajectory_timestamps, camera_timestamp)
                
                if trajectory_idx is not None and trajectory_idx < len(f['trajectory/reference_point_x']):
                    frame_data['trajectory'] = {
                        'reference_point': {
                            'x': float(f['trajectory/reference_point_x'][trajectory_idx]),
                            'y': float(f['trajectory/reference_point_y'][trajectory_idx]),
                            'heading': float(f['trajectory/reference_point_heading'][trajectory_idx]),
                            'velocity': float(f['trajectory/reference_point_velocity'][trajectory_idx]) if 'trajectory/reference_point_velocity' in f and trajectory_idx < len(f['trajectory/reference_point_velocity']) else None
                        }
                    }
                    
                    if 'trajectory/reference_point_raw_x' in f and trajectory_idx < len(f['trajectory/reference_point_raw_x']):
                        frame_data['trajectory']['reference_point_raw'] = {
                            'x': float(f['trajectory/reference_point_raw_x'][trajectory_idx]),
                            'y': float(f['trajectory/reference_point_raw_y'][trajectory_idx]),
                            'heading': float(f['trajectory/reference_point_raw_heading'][trajectory_idx])
                        }
                    
                    # NEW: Load debug fields (method and perception center)
                    if 'trajectory/reference_point_method' in f and trajectory_idx < len(f['trajectory/reference_point_method']):
                        method_val = f['trajectory/reference_point_method'][trajectory_idx]
                        if isinstance(method_val, bytes):
                            method_val = method_val.decode('utf-8')
                        frame_data['trajectory']['reference_point_method'] = method_val
                    if 'trajectory/perception_center_x' in f and trajectory_idx < len(f['trajectory/perception_center_x']):
                        frame_data['trajectory']['perception_center_x'] = float(f['trajectory/perception_center_x'][trajectory_idx])
                    
                    # NEW: Load trajectory points if available
                    if 'trajectory/trajectory_points' in f and trajectory_idx < len(f['trajectory/trajectory_points']):
                        traj_points = f['trajectory/trajectory_points'][trajectory_idx]
                        if traj_points is not None and len(traj_points) > 0:
                            # CRITICAL FIX: trajectory_points are stored as flattened arrays (N*3,)
                            # Need to reshape to (N, 3) before iterating
                            try:
                                # Check if it's already 2D
                                if traj_points.ndim == 2 and traj_points.shape[1] == 3:
                                    # Already 2D [N, 3] - use as is
                                    traj_points_2d = traj_points
                                elif traj_points.ndim == 1:
                                    # Flattened array - reshape to (N, 3)
                                    if len(traj_points) % 3 == 0:
                                        traj_points_2d = traj_points.reshape(-1, 3)
                                    else:
                                        # Invalid size - skip
                                        traj_points_2d = np.array([], dtype=np.float32).reshape(0, 3)
                                else:
                                    # Invalid shape - skip
                                    traj_points_2d = np.array([], dtype=np.float32).reshape(0, 3)
                                
                                # Convert to list of dicts for JSON serialization
                                frame_data['trajectory']['trajectory_points'] = [
                                    {'x': float(p[0]), 'y': float(p[1]), 'heading': float(p[2])}
                                    for p in traj_points_2d
                                ]
                            except Exception as e:
                                # If there's any error reading trajectory points, skip them
                                # Don't break the entire frame loading
                                print(f"Warning: Error reading trajectory_points for frame {trajectory_idx}: {e}")
                                frame_data['trajectory']['trajectory_points'] = []
            
            # Control data - find closest by timestamp
            if 'control/timestamps' in f and len(f['control/timestamps']) > 0:
                control_timestamps = np.array(f['control/timestamps'])
                control_idx = find_closest_index(control_timestamps, camera_timestamp)
                
                if control_idx is not None and control_idx < len(f['control/steering']):
                    frame_data['control'] = {
                        'timestamp': float(control_timestamps[control_idx]),
                        'steering': float(f['control/steering'][control_idx]),
                        'throttle': float(f['control/throttle'][control_idx]),
                        'brake': float(f['control/brake'][control_idx])
                    }
                    
                    # Add optional fields if they exist
                    if 'control/steering_before_limits' in f and control_idx < len(f['control/steering_before_limits']):
                        frame_data['control']['steering_before_limits'] = float(f['control/steering_before_limits'][control_idx])
                    if 'control/throttle_before_limits' in f and control_idx < len(f['control/throttle_before_limits']):
                        frame_data['control']['throttle_before_limits'] = float(f['control/throttle_before_limits'][control_idx])
                    if 'control/brake_before_limits' in f and control_idx < len(f['control/brake_before_limits']):
                        frame_data['control']['brake_before_limits'] = float(f['control/brake_before_limits'][control_idx])
                    if 'control/pid_integral' in f and control_idx < len(f['control/pid_integral']):
                        frame_data['control']['pid_integral'] = float(f['control/pid_integral'][control_idx])
                    if 'control/pid_derivative' in f and control_idx < len(f['control/pid_derivative']):
                        frame_data['control']['pid_derivative'] = float(f['control/pid_derivative'][control_idx])
                    if 'control/pid_error' in f and control_idx < len(f['control/pid_error']):
                        frame_data['control']['pid_error'] = float(f['control/pid_error'][control_idx])
                    if 'control/lateral_error' in f and control_idx < len(f['control/lateral_error']):
                        frame_data['control']['lateral_error'] = float(f['control/lateral_error'][control_idx])
                    if 'control/heading_error' in f and control_idx < len(f['control/heading_error']):
                        heading_err_rad = float(f['control/heading_error'][control_idx])
                        frame_data['control']['heading_error'] = heading_err_rad
                    if 'control/total_error' in f and control_idx < len(f['control/total_error']):
                        frame_data['control']['total_error'] = float(f['control/total_error'][control_idx])
            
            # Ground truth data - use same index as vehicle state (they should be synchronized)
            # Try new name first, fall back to old name for backward compatibility
            if 'ground_truth/left_lane_line_x' in f and len(f['ground_truth/left_lane_line_x']) > 0:
                # Use vehicle state index if available, otherwise use frame_index
                gt_idx = vehicle_idx if vehicle_idx is not None else frame_index
                
                if gt_idx is not None and gt_idx < len(f['ground_truth/left_lane_line_x']):
                    frame_data['ground_truth'] = {
                        'left_lane_line_x': float(f['ground_truth/left_lane_line_x'][gt_idx]),
                        'right_lane_line_x': float(f['ground_truth/right_lane_line_x'][gt_idx]),
                        'lane_center_x': float(f['ground_truth/lane_center_x'][gt_idx]) if 'ground_truth/lane_center_x' in f and gt_idx < len(f['ground_truth/lane_center_x']) else None,
                    }
                    # Add path curvature and desired heading if available
                    if 'ground_truth/path_curvature' in f and gt_idx < len(f['ground_truth/path_curvature']):
                        frame_data['ground_truth']['path_curvature'] = float(f['ground_truth/path_curvature'][gt_idx])
                    if 'ground_truth/desired_heading' in f and gt_idx < len(f['ground_truth/desired_heading']):
                        frame_data['ground_truth']['desired_heading'] = float(f['ground_truth/desired_heading'][gt_idx])
            elif 'ground_truth/left_lane_x' in f and len(f['ground_truth/left_lane_x']) > 0:  # Backward compatibility
                # Use vehicle state index if available, otherwise use frame_index
                gt_idx = vehicle_idx if vehicle_idx is not None else frame_index
                
                if gt_idx < len(f['ground_truth/left_lane_x']):
                    frame_data['ground_truth'] = {
                        'left_lane_line_x': float(f['ground_truth/left_lane_x'][gt_idx]),  # Map old name to new
                        'right_lane_line_x': float(f['ground_truth/right_lane_x'][gt_idx]),  # Map old name to new
                        'lane_center_x': float(f['ground_truth/lane_center_x'][gt_idx])
                    }
                    
                    # Add path curvature and desired heading if available
                    if 'ground_truth/path_curvature' in f and gt_idx < len(f['ground_truth/path_curvature']):
                        frame_data['ground_truth']['path_curvature'] = float(f['ground_truth/path_curvature'][gt_idx])
                    if 'ground_truth/desired_heading' in f and gt_idx < len(f['ground_truth/desired_heading']):
                        frame_data['ground_truth']['desired_heading'] = float(f['ground_truth/desired_heading'][gt_idx])
            
            # Unity feedback data (if available) - find closest by timestamp
            if 'unity_feedback/timestamps' in f and len(f['unity_feedback/timestamps']) > 0:
                unity_timestamps = np.array(f['unity_feedback/timestamps'])
                unity_idx = find_closest_index(unity_timestamps, camera_timestamp, max_diff=1.0)  # Larger tolerance (sent every 1s)
                
                if unity_idx is not None and unity_idx < len(f['unity_feedback/ground_truth_mode_active']):
                    frame_data['unity_feedback'] = {
                        'ground_truth_mode_active': bool(f['unity_feedback/ground_truth_mode_active'][unity_idx]),
                        'path_curvature_calculated': bool(f['unity_feedback/path_curvature_calculated'][unity_idx]) if 'unity_feedback/path_curvature_calculated' in f and unity_idx < len(f['unity_feedback/path_curvature_calculated']) else None
                    }
                    if 'unity_feedback/actual_steering_applied' in f and unity_idx < len(f['unity_feedback/actual_steering_applied']):
                        frame_data['unity_feedback']['actual_steering_applied'] = float(f['unity_feedback/actual_steering_applied'][unity_idx])
            
            return jsonify(frame_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/recording/<filename>/frame/<int:frame_index>/image')
def get_frame_image(filename, frame_index):
    """Get camera frame as base64-encoded image."""
    filepath = RECORDINGS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "Recording not found"}), 404
    
    try:
        with h5py.File(filepath, 'r') as f:
            if 'camera/images' not in f or frame_index >= len(f['camera/images']):
                return jsonify({"error": "Frame not found"}), 404
            
            image = f['camera/images'][frame_index]
            # Convert to PIL Image
            img = Image.fromarray(image)
            
            # Convert to base64
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return jsonify({"image": f"data:image/png;base64,{img_str}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/debug/<path:image_name>')
def get_debug_image(image_name):
    """Get debug visualization image by name."""
    # image_name format: "frame_000000_combined" or "frame_000000_edges.png"
    if not image_name.endswith('.png'):
        image_name += '.png'
    filepath = DEBUG_VIS_DIR / image_name
    if filepath.exists():
        return send_from_directory(str(DEBUG_VIS_DIR), image_name)
    return jsonify({"error": "Debug image not found"}), 404


@app.route('/')
def index():
    """Serve the main HTML file."""
    return send_from_directory(Path(__file__).parent, 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (JS, CSS)."""
    return send_from_directory(Path(__file__).parent, filename)


if __name__ == '__main__':
    print(f"Starting debug visualizer server...")
    print(f"Recordings directory: {RECORDINGS_DIR}")
    print(f"Debug visualizations directory: {DEBUG_VIS_DIR}")
    print(f"Server running at http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)

