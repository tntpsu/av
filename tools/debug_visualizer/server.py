"""
Python server for debug visualizer.
Converts HDF5 recordings to JSON and serves camera frames.
"""

import h5py
import numpy as np
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import json
import sys
import logging
import subprocess
import re

# Add backend modules to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))
# Add analysis modules to path
analysis_path = Path(__file__).parent.parent / "analyze"
sys.path.insert(0, str(analysis_path))
from summary_analyzer import analyze_recording_summary
from diagnostics import analyze_trajectory_vs_steering
from issue_detector import detect_issues
from analyze_trajectory_layer_localization import analyze_trajectory_layer_localization

app = Flask(__name__)
CORS(app)  # Enable CORS for local development
logging.getLogger("werkzeug").setLevel(logging.ERROR)
app.logger.setLevel(logging.ERROR)

# Paths
RECORDINGS_DIR = Path(__file__).parent.parent.parent / "data" / "recordings"
DEBUG_VIS_DIR = Path(__file__).parent.parent.parent / "tmp" / "debug_visualizations"
REPO_ROOT = Path(__file__).parent.parent.parent


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


def parse_flattened_xy_points(row, include_valid=False):
    """Parse flattened [x0,y0,x1,y1,...] to point objects."""
    if row is None:
        return []
    try:
        arr = np.asarray(row, dtype=np.float32).reshape(-1)
    except Exception:
        return []
    if arr.size < 2 or arr.size % 2 != 0:
        return []
    points = []
    for i in range(0, arr.size, 2):
        x = float(arr[i])
        y = float(arr[i + 1])
        if include_valid:
            valid = bool(np.isfinite(x) and np.isfinite(y) and x >= 0.0 and y >= 0.0)
            points.append({"x": x, "y": y, "valid": valid})
        else:
            if np.isfinite(x) and np.isfinite(y):
                points.append({"x": x, "y": y})
    return points


def parse_flattened_xyz_points(row):
    """Parse flattened [x0,y0,z0,x1,y1,z1,...] to point objects."""
    if row is None:
        return []
    try:
        arr = np.asarray(row, dtype=np.float32).reshape(-1)
    except Exception:
        return []
    if arr.size < 3 or arr.size % 3 != 0:
        return []
    points = []
    for i in range(0, arr.size, 3):
        x = float(arr[i])
        y = float(arr[i + 1])
        z = float(arr[i + 2])
        if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
            points.append({"x": x, "y": y, "z": z})
    return points


def _is_numeric_series(dataset: h5py.Dataset) -> bool:
    if not isinstance(dataset, h5py.Dataset):
        return False
    if dataset.ndim != 1:
        return False
    if dataset.dtype.kind not in ("i", "u", "f"):
        return False
    return True


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


@app.route('/api/recording/<path:filename>/frames')
def get_frame_count(filename):
    """Get total number of frames in recording.
    
    Uses <path:filename> to handle filenames with special characters.
    """
    # Decode URL-encoded filename if needed
    from urllib.parse import unquote
    filename = unquote(filename)
    
    filepath = RECORDINGS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": f"Recording not found: {filename}", "filepath": str(filepath)}), 404
    
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
        import traceback
        error_msg = f"Error reading recording {filename}: {str(e)}"
        print(f"[SERVER ERROR] {error_msg}")
        print(f"[SERVER ERROR] Filepath: {filepath}")
        print(f"[SERVER ERROR] File exists: {filepath.exists()}")
        print(f"[SERVER ERROR] Traceback:\n{traceback.format_exc()}")
        # Return error with more details for debugging
        return jsonify({
            "error": error_msg,
            "filepath": str(filepath),
            "file_exists": filepath.exists(),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/api/recording/<path:filename>/meta')
def get_recording_meta(filename):
    """Get recording metadata (type/source/top-down availability)."""
    from urllib.parse import unquote
    filename = unquote(filename)
    filepath = RECORDINGS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": f"Recording not found: {filename}"}), 404

    try:
        with h5py.File(filepath, 'r') as f:
            meta_raw = f.attrs.get('metadata')
            metadata = {}
            if meta_raw is not None:
                try:
                    meta_str = (
                        meta_raw.decode('utf-8', 'ignore')
                        if isinstance(meta_raw, (bytes, bytearray))
                        else str(meta_raw)
                    )
                    metadata = json.loads(meta_str)
                except Exception:
                    metadata = {}

            topdown_available = (
                "camera/topdown_images" in f and len(f["camera/topdown_images"]) > 0
            )
            return jsonify(
                {
                    "recording_type": metadata.get("recording_type", "unknown"),
                    "source_recording": metadata.get("source_recording"),
                    "lock_source_recording": metadata.get("lock_source_recording"),
                    "control_lock_source_recording": metadata.get("control_lock_source_recording"),
                    "perception_mode": metadata.get("perception_mode"),
                    "topdown_available": bool(topdown_available),
                    "metadata": metadata,
                }
            )
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/api/recording/<path:filename>/signals')
def list_signals(filename):
    """List numeric time-series signals available in the recording."""
    from urllib.parse import unquote
    filename = unquote(filename)
    filepath = RECORDINGS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": f"Recording not found: {filename}"}), 404

    signals = []
    try:
        with h5py.File(filepath, 'r') as f:
            def visit(name, obj):
                if not isinstance(obj, h5py.Dataset):
                    return
                if name in ("camera/images", "camera/topdown_images"):
                    return
                if _is_numeric_series(obj):
                    signals.append({
                        "name": name,
                        "length": int(obj.shape[0]),
                        "dtype": str(obj.dtype),
                    })

            f.visititems(visit)

            # Derived signals
            if "vehicle/speed" in f and "vehicle/timestamps" in f:
                signals.append({
                    "name": "derived/distance_m",
                    "length": int(min(len(f["vehicle/speed"]), len(f["vehicle/timestamps"]))),
                    "dtype": "float64",
                })
            # Expected right lane center (for debugging ref_x_raw)
            if "perception/right_lane_line_x" in f:
                # Read config to get target_lane_width_m
                import yaml
                config_path = Path(__file__).parent.parent.parent / "config" / "av_stack_config.yaml"
                target_lane_width = 3.6  # default
                if config_path.exists():
                    try:
                        with open(config_path, 'r') as cfg:
                            config = yaml.safe_load(cfg)
                            target_lane_width = config.get('trajectory', {}).get('target_lane_width_m', 3.6)
                    except:
                        pass
                signals.append({
                    "name": "derived/expected_right_center",
                    "length": int(len(f["perception/right_lane_line_x"])),
                    "dtype": "float64",
                })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    signals.sort(key=lambda s: s["name"])
    return jsonify({"signals": signals})


@app.route('/api/recording/<path:filename>/timeseries')
def get_timeseries(filename):
    """Return aligned time-series data for selected signals."""
    from urllib.parse import unquote
    filename = unquote(filename)
    filepath = RECORDINGS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": f"Recording not found: {filename}"}), 404

    signal_param = request.args.get("signals", "")
    time_key = request.args.get("time", "")
    signal_names = [s.strip() for s in signal_param.split(",") if s.strip()]
    if not signal_names:
        return jsonify({"error": "No signals provided"}), 400

    try:
        with h5py.File(filepath, 'r') as f:
            if not time_key:
                if "control/timestamps" in f:
                    time_key = "control/timestamps"
                elif "vehicle/timestamps" in f:
                    time_key = "vehicle/timestamps"
                elif "camera/timestamps" in f:
                    time_key = "camera/timestamps"

            time_series = None
            if time_key and time_key in f and _is_numeric_series(f[time_key]):
                time_series = f[time_key][:]

            series = {}
            min_len = len(time_series) if time_series is not None else None
            for name in signal_names:
                if name == "derived/distance_m":
                    if "vehicle/speed" not in f or "vehicle/timestamps" not in f:
                        continue
                    speed = f["vehicle/speed"][:]
                    base_time = f["vehicle/timestamps"][:]
                    base_len = min(len(speed), len(base_time))
                    speed = speed[:base_len]
                    base_time = base_time[:base_len]
                    dt = np.diff(base_time, prepend=base_time[0])
                    dt[0] = 0.0
                    distance = np.cumsum(speed * dt)
                    series[name] = distance
                    min_len = len(distance) if min_len is None else min(min_len, len(distance))
                elif name == "derived/expected_right_center":
                    if "perception/right_lane_line_x" not in f:
                        continue
                    # Read config to get target_lane_width_m
                    import yaml
                    config_path = Path(__file__).parent.parent.parent / "config" / "av_stack_config.yaml"
                    target_lane_width = 3.6  # default
                    if config_path.exists():
                        try:
                            with open(config_path, 'r') as cfg:
                                config = yaml.safe_load(cfg)
                                target_lane_width = config.get('trajectory', {}).get('target_lane_width_m', 3.6)
                        except:
                            pass
                    right_lane_line_x = f["perception/right_lane_line_x"][:]
                    expected_right_center = right_lane_line_x - (target_lane_width / 2.0)
                    series[name] = expected_right_center
                    min_len = len(expected_right_center) if min_len is None else min(min_len, len(expected_right_center))
                else:
                    if name not in f or not _is_numeric_series(f[name]):
                        continue
                    values = f[name][:]
                    series[name] = values
                    min_len = len(values) if min_len is None else min(min_len, len(values))

            if min_len is None or min_len == 0:
                return jsonify({"error": "No valid signals found"}), 400

            payload = {
                "time_key": time_key if time_series is not None else None,
                "time": time_series[:min_len].astype(float).tolist() if time_series is not None else None,
                "signals": {},
            }
            for name, values in series.items():
                payload["signals"][name] = values[:min_len].astype(float).tolist()

            return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/recording/<path:filename>/frame/<int:frame_index>')
def get_frame_data(filename, frame_index):
    """Get all data for a specific frame.
    
    Uses timestamp-based synchronization to match data from different datasets
    that may have different lengths or frame indices.
    """
    from urllib.parse import unquote
    filename = unquote(filename)
    filepath = RECORDINGS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": f"Recording not found: {filename}"}), 404
    
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
                if 'camera/topdown_images' in f and frame_index < len(f['camera/topdown_images']):
                    frame_data['camera_topdown'] = {
                        'timestamp': float(f['camera/topdown_timestamps'][frame_index]),
                        'frame_id': int(f['camera/topdown_frame_ids'][frame_index])
                    }
            else:
                # No camera frame at this index - return empty
                return jsonify(frame_data)
            
            # Nearest-valid bounded pairing policy (seconds).
            max_pair_diff_s = {
                "vehicle": 0.2,
                "perception": 0.2,
                "trajectory": 0.2,
                "control": 0.2,
            }

            # Helper function to find closest timestamp match.
            def find_nearest_with_diff(timestamps, target_ts, max_diff=0.1):
                """Find closest index and absolute diff (seconds), bounded by max_diff."""
                if len(timestamps) == 0 or target_ts is None or not np.isfinite(target_ts):
                    return None, None
                diffs = np.abs(timestamps - target_ts)
                closest_idx = int(np.argmin(diffs))
                closest_diff = float(diffs[closest_idx])
                if max_diff is not None and closest_diff > float(max_diff):
                    return None, closest_diff
                return closest_idx, closest_diff
            
            # Initialize indices
            vehicle_idx = None
            perception_idx = None
            trajectory_idx = None
            control_idx = None
            vehicle_dt_s = None
            perception_dt_s = None
            trajectory_dt_s = None
            control_dt_s = None
            
            # Vehicle state - find closest by timestamp
            if 'vehicle/timestamps' in f and len(f['vehicle/timestamps']) > 0:
                vehicle_timestamps = np.array(f['vehicle/timestamps'])
                vehicle_idx, vehicle_dt_s = find_nearest_with_diff(
                    vehicle_timestamps, camera_timestamp, max_diff=max_pair_diff_s["vehicle"]
                )
                
                if vehicle_idx is not None and vehicle_idx < len(f['vehicle/position']):
                    frame_data['vehicle'] = {
                        # NEW: Camera calibration - actual y pixel where 8m appears
                        'camera_8m_screen_y': float(f['vehicle/camera_8m_screen_y'][vehicle_idx]) if 'vehicle/camera_8m_screen_y' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/camera_8m_screen_y']) else None,
                        'camera_lookahead_screen_y': float(f['vehicle/camera_lookahead_screen_y'][vehicle_idx]) if 'vehicle/camera_lookahead_screen_y' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/camera_lookahead_screen_y']) else None,
                        'ground_truth_lookahead_distance': float(f['vehicle/ground_truth_lookahead_distance'][vehicle_idx]) if 'vehicle/ground_truth_lookahead_distance' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/ground_truth_lookahead_distance']) else None,
                        'right_lane_fiducials_point_count': int(f['vehicle/right_lane_fiducials_point_count'][vehicle_idx]) if 'vehicle/right_lane_fiducials_point_count' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/right_lane_fiducials_point_count']) else 0,
                        'right_lane_fiducials_horizon_meters': float(f['vehicle/right_lane_fiducials_horizon_meters'][vehicle_idx]) if 'vehicle/right_lane_fiducials_horizon_meters' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/right_lane_fiducials_horizon_meters']) else 0.0,
                        'right_lane_fiducials_spacing_meters': float(f['vehicle/right_lane_fiducials_spacing_meters'][vehicle_idx]) if 'vehicle/right_lane_fiducials_spacing_meters' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/right_lane_fiducials_spacing_meters']) else 0.0,
                        'right_lane_fiducials_enabled': bool(f['vehicle/right_lane_fiducials_enabled'][vehicle_idx]) if 'vehicle/right_lane_fiducials_enabled' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/right_lane_fiducials_enabled']) else False,
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
                        'topdown_camera_pos_x': float(f['vehicle/topdown_camera_pos_x'][vehicle_idx]) if 'vehicle/topdown_camera_pos_x' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/topdown_camera_pos_x']) else None,
                        'topdown_camera_pos_y': float(f['vehicle/topdown_camera_pos_y'][vehicle_idx]) if 'vehicle/topdown_camera_pos_y' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/topdown_camera_pos_y']) else None,
                        'topdown_camera_pos_z': float(f['vehicle/topdown_camera_pos_z'][vehicle_idx]) if 'vehicle/topdown_camera_pos_z' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/topdown_camera_pos_z']) else None,
                        'topdown_camera_forward_x': float(f['vehicle/topdown_camera_forward_x'][vehicle_idx]) if 'vehicle/topdown_camera_forward_x' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/topdown_camera_forward_x']) else None,
                        'topdown_camera_forward_y': float(f['vehicle/topdown_camera_forward_y'][vehicle_idx]) if 'vehicle/topdown_camera_forward_y' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/topdown_camera_forward_y']) else None,
                        'topdown_camera_forward_z': float(f['vehicle/topdown_camera_forward_z'][vehicle_idx]) if 'vehicle/topdown_camera_forward_z' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/topdown_camera_forward_z']) else None,
                        'topdown_camera_orthographic_size': float(f['vehicle/topdown_camera_orthographic_size'][vehicle_idx]) if 'vehicle/topdown_camera_orthographic_size' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/topdown_camera_orthographic_size']) else None,
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
                        'speed_limit': float(f['vehicle/speed_limit'][vehicle_idx]) if 'vehicle/speed_limit' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/speed_limit']) else None,
                        'speed_limit_preview': float(f['vehicle/speed_limit_preview'][vehicle_idx]) if 'vehicle/speed_limit_preview' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/speed_limit_preview']) else None,
                        'speed_limit_preview_distance': float(f['vehicle/speed_limit_preview_distance'][vehicle_idx]) if 'vehicle/speed_limit_preview_distance' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/speed_limit_preview_distance']) else None,
                        'speed_limit_preview_min_distance': float(f['vehicle/speed_limit_preview_min_distance'][vehicle_idx]) if 'vehicle/speed_limit_preview_min_distance' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/speed_limit_preview_min_distance']) else None,
                        'speed_limit_preview_mid': float(f['vehicle/speed_limit_preview_mid'][vehicle_idx]) if 'vehicle/speed_limit_preview_mid' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/speed_limit_preview_mid']) else None,
                        'speed_limit_preview_mid_distance': float(f['vehicle/speed_limit_preview_mid_distance'][vehicle_idx]) if 'vehicle/speed_limit_preview_mid_distance' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/speed_limit_preview_mid_distance']) else None,
                        'speed_limit_preview_mid_min_distance': float(f['vehicle/speed_limit_preview_mid_min_distance'][vehicle_idx]) if 'vehicle/speed_limit_preview_mid_min_distance' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/speed_limit_preview_mid_min_distance']) else None,
                        'speed_limit_preview_long': float(f['vehicle/speed_limit_preview_long'][vehicle_idx]) if 'vehicle/speed_limit_preview_long' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/speed_limit_preview_long']) else None,
                        'speed_limit_preview_long_distance': float(f['vehicle/speed_limit_preview_long_distance'][vehicle_idx]) if 'vehicle/speed_limit_preview_long_distance' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/speed_limit_preview_long_distance']) else None,
                        'speed_limit_preview_long_min_distance': float(f['vehicle/speed_limit_preview_long_min_distance'][vehicle_idx]) if 'vehicle/speed_limit_preview_long_min_distance' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/speed_limit_preview_long_min_distance']) else None,
                        'steering_angle': float(f['vehicle/steering_angle'][vehicle_idx]),
                        'steering_angle_actual': float(f['vehicle/steering_angle_actual'][vehicle_idx])
                        if 'vehicle/steering_angle_actual' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/steering_angle_actual']) else None
                        ,
                        'steering_input': float(f['vehicle/steering_input'][vehicle_idx])
                        if 'vehicle/steering_input' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/steering_input']) else None,
                        'desired_steer_angle': float(f['vehicle/desired_steer_angle'][vehicle_idx])
                        if 'vehicle/desired_steer_angle' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/desired_steer_angle']) else None,
                        'unity_time': float(f['vehicle/unity_time'][vehicle_idx]) if 'vehicle/unity_time' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/unity_time']) else None,
                        'unity_frame_count': int(f['vehicle/unity_frame_count'][vehicle_idx]) if 'vehicle/unity_frame_count' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/unity_frame_count']) else None,
                        'unity_delta_time': float(f['vehicle/unity_delta_time'][vehicle_idx]) if 'vehicle/unity_delta_time' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/unity_delta_time']) else None,
                        'unity_smooth_delta_time': float(f['vehicle/unity_smooth_delta_time'][vehicle_idx]) if 'vehicle/unity_smooth_delta_time' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/unity_smooth_delta_time']) else None,
                        'unity_unscaled_delta_time': float(f['vehicle/unity_unscaled_delta_time'][vehicle_idx]) if 'vehicle/unity_unscaled_delta_time' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/unity_unscaled_delta_time']) else None,
                        'unity_time_scale': float(f['vehicle/unity_time_scale'][vehicle_idx]) if 'vehicle/unity_time_scale' in f and vehicle_idx is not None and vehicle_idx < len(f['vehicle/unity_time_scale']) else None
                    }
                    for stream_key in [
                        'stream_front_source_timestamp',
                        'stream_topdown_source_timestamp',
                        'stream_front_timestamp_reused',
                        'stream_topdown_timestamp_reused',
                        'stream_front_timestamp_non_monotonic',
                        'stream_topdown_timestamp_non_monotonic',
                        'stream_front_negative_frame_delta',
                        'stream_topdown_negative_frame_delta',
                        'stream_front_frame_id_reused',
                        'stream_topdown_frame_id_reused',
                        'stream_front_clock_jump',
                        'stream_topdown_clock_jump',
                    ]:
                        ds = f"vehicle/{stream_key}"
                        if ds in f and vehicle_idx < len(f[ds]):
                            frame_data['vehicle'][stream_key] = float(f[ds][vehicle_idx])
                    
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

                    if 'vehicle/right_lane_fiducials_vehicle_xy' in f and vehicle_idx < len(f['vehicle/right_lane_fiducials_vehicle_xy']):
                        frame_data['vehicle']['right_lane_fiducials_vehicle_points'] = parse_flattened_xy_points(
                            f['vehicle/right_lane_fiducials_vehicle_xy'][vehicle_idx],
                            include_valid=False,
                        )
                    else:
                        frame_data['vehicle']['right_lane_fiducials_vehicle_points'] = []
                    if 'vehicle/right_lane_fiducials_vehicle_true_xy' in f and vehicle_idx < len(f['vehicle/right_lane_fiducials_vehicle_true_xy']):
                        frame_data['vehicle']['right_lane_fiducials_vehicle_true_points'] = parse_flattened_xy_points(
                            f['vehicle/right_lane_fiducials_vehicle_true_xy'][vehicle_idx],
                            include_valid=False,
                        )
                    else:
                        frame_data['vehicle']['right_lane_fiducials_vehicle_true_points'] = []
                    if 'vehicle/right_lane_fiducials_vehicle_monotonic_xy' in f and vehicle_idx < len(f['vehicle/right_lane_fiducials_vehicle_monotonic_xy']):
                        frame_data['vehicle']['right_lane_fiducials_vehicle_monotonic_points'] = parse_flattened_xy_points(
                            f['vehicle/right_lane_fiducials_vehicle_monotonic_xy'][vehicle_idx],
                            include_valid=False,
                        )
                    else:
                        frame_data['vehicle']['right_lane_fiducials_vehicle_monotonic_points'] = []
                    if 'vehicle/right_lane_fiducials_world_xyz' in f and vehicle_idx < len(f['vehicle/right_lane_fiducials_world_xyz']):
                        frame_data['vehicle']['right_lane_fiducials_world_points'] = parse_flattened_xyz_points(
                            f['vehicle/right_lane_fiducials_world_xyz'][vehicle_idx],
                        )
                    else:
                        frame_data['vehicle']['right_lane_fiducials_world_points'] = []
                    if 'vehicle/oracle_trajectory_world_xyz' in f and vehicle_idx < len(f['vehicle/oracle_trajectory_world_xyz']):
                        frame_data['vehicle']['oracle_trajectory_world_points'] = parse_flattened_xyz_points(
                            f['vehicle/oracle_trajectory_world_xyz'][vehicle_idx],
                        )
                    else:
                        frame_data['vehicle']['oracle_trajectory_world_points'] = []
                    if 'vehicle/oracle_trajectory_screen_xy' in f and vehicle_idx < len(f['vehicle/oracle_trajectory_screen_xy']):
                        frame_data['vehicle']['oracle_trajectory_screen_points'] = parse_flattened_xy_points(
                            f['vehicle/oracle_trajectory_screen_xy'][vehicle_idx],
                            include_valid=True,
                        )
                    else:
                        frame_data['vehicle']['oracle_trajectory_screen_points'] = []

                    if 'vehicle/right_lane_fiducials_screen_xy' in f and vehicle_idx < len(f['vehicle/right_lane_fiducials_screen_xy']):
                        frame_data['vehicle']['right_lane_fiducials_screen_points'] = parse_flattened_xy_points(
                            f['vehicle/right_lane_fiducials_screen_xy'][vehicle_idx],
                            include_valid=True,
                        )
                    else:
                        frame_data['vehicle']['right_lane_fiducials_screen_points'] = []
            
            # Perception data - find closest by timestamp
            if 'perception/timestamps' in f and len(f['perception/timestamps']) > 0:
                perception_timestamps = np.array(f['perception/timestamps'])
                perception_idx, perception_dt_s = find_nearest_with_diff(
                    perception_timestamps, camera_timestamp, max_diff=max_pair_diff_s["perception"]
                )
                
                if perception_idx is not None and perception_idx < len(f['perception/confidence']):
                    frame_data['perception'] = {
                        'timestamp': float(perception_timestamps[perception_idx]),
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
                    
                    # NEW: Read perception health metrics
                    if 'perception/consecutive_bad_detection_frames' in f and perception_idx < len(f['perception/consecutive_bad_detection_frames']):
                        frame_data['perception']['consecutive_bad_detection_frames'] = int(f['perception/consecutive_bad_detection_frames'][perception_idx])
                    if 'perception/perception_health_score' in f and perception_idx < len(f['perception/perception_health_score']):
                        frame_data['perception']['perception_health_score'] = float(f['perception/perception_health_score'][perception_idx])
                    if 'perception/perception_health_status' in f and perception_idx < len(f['perception/perception_health_status']):
                        health_status = f['perception/perception_health_status'][perception_idx]
                        if isinstance(health_status, bytes):
                            health_status = health_status.decode('utf-8')
                        frame_data['perception']['perception_health_status'] = health_status
                    if 'perception/perception_bad_events' in f and perception_idx < len(f['perception/perception_bad_events']):
                        bad_events = f['perception/perception_bad_events'][perception_idx]
                        if isinstance(bad_events, bytes):
                            bad_events = bad_events.decode('utf-8')
                        frame_data['perception']['perception_bad_events'] = bad_events if bad_events else None
                    if 'perception/perception_bad_events_recent' in f and perception_idx < len(f['perception/perception_bad_events_recent']):
                        bad_events_recent = f['perception/perception_bad_events_recent'][perception_idx]
                        if isinstance(bad_events_recent, bytes):
                            bad_events_recent = bad_events_recent.decode('utf-8')
                        frame_data['perception']['perception_bad_events_recent'] = (
                            bad_events_recent if bad_events_recent else None
                        )
                    if 'perception/perception_clamp_events' in f and perception_idx < len(f['perception/perception_clamp_events']):
                        clamp_events = f['perception/perception_clamp_events'][perception_idx]
                        if isinstance(clamp_events, bytes):
                            clamp_events = clamp_events.decode('utf-8')
                        frame_data['perception']['perception_clamp_events'] = clamp_events if clamp_events else None
                    if 'perception/perception_timestamp_frozen' in f and perception_idx < len(f['perception/perception_timestamp_frozen']):
                        frame_data['perception']['perception_timestamp_frozen'] = bool(
                            f['perception/perception_timestamp_frozen'][perception_idx]
                        )
                    
                    # NEW: Read stale data fields
                    if 'perception/using_stale_data' in f and perception_idx < len(f['perception/using_stale_data']):
                        frame_data['perception']['using_stale_data'] = bool(f['perception/using_stale_data'][perception_idx])
                    if 'perception/stale_reason' in f and perception_idx < len(f['perception/stale_reason']):
                        stale_reason = f['perception/stale_reason'][perception_idx]
                        if isinstance(stale_reason, bytes):
                            stale_reason = stale_reason.decode('utf-8')
                        frame_data['perception']['stale_data_reason'] = stale_reason if stale_reason else None
                    if 'perception/reject_reason' in f and perception_idx < len(f['perception/reject_reason']):
                        reject_reason = f['perception/reject_reason'][perception_idx]
                        if isinstance(reject_reason, bytes):
                            reject_reason = reject_reason.decode('utf-8')
                        frame_data['perception']['reject_reason'] = reject_reason if reject_reason else None
                    
                    # NEW: Read jump detection fields (help understand why stale data is used)
                    if 'perception/left_jump_magnitude' in f and perception_idx < len(f['perception/left_jump_magnitude']):
                        left_jump = float(f['perception/left_jump_magnitude'][perception_idx])
                        frame_data['perception']['left_jump_magnitude'] = left_jump if left_jump > 0.0 else None
                    if 'perception/right_jump_magnitude' in f and perception_idx < len(f['perception/right_jump_magnitude']):
                        right_jump = float(f['perception/right_jump_magnitude'][perception_idx])
                        frame_data['perception']['right_jump_magnitude'] = right_jump if right_jump > 0.0 else None
                    if 'perception/jump_threshold' in f and perception_idx < len(f['perception/jump_threshold']):
                        jump_threshold = float(f['perception/jump_threshold'][perception_idx])
                        frame_data['perception']['jump_threshold'] = jump_threshold if jump_threshold > 0.0 else None
                    
                    # NEW: Read instability diagnostic fields
                    if 'perception/actual_detected_left_lane_x' in f and perception_idx < len(f['perception/actual_detected_left_lane_x']):
                        actual_left = float(f['perception/actual_detected_left_lane_x'][perception_idx])
                        frame_data['perception']['actual_detected_left_lane_x'] = actual_left if actual_left != 0.0 else None
                    if 'perception/actual_detected_right_lane_x' in f and perception_idx < len(f['perception/actual_detected_right_lane_x']):
                        actual_right = float(f['perception/actual_detected_right_lane_x'][perception_idx])
                        frame_data['perception']['actual_detected_right_lane_x'] = actual_right if actual_right != 0.0 else None
                    if 'perception/instability_width_change' in f and perception_idx < len(f['perception/instability_width_change']):
                        width_change = float(f['perception/instability_width_change'][perception_idx])
                        frame_data['perception']['instability_width_change'] = width_change if width_change > 0.0 else None
                    if 'perception/instability_center_shift' in f and perception_idx < len(f['perception/instability_center_shift']):
                        center_shift = float(f['perception/instability_center_shift'][perception_idx])
                        frame_data['perception']['instability_center_shift'] = center_shift if center_shift > 0.0 else None
                    
                    # NEW: Read fit_points (points used for polynomial fitting)
                    if 'perception/fit_points_left' in f and perception_idx < len(f['perception/fit_points_left']):
                        try:
                            fit_points_left_json = f['perception/fit_points_left'][perception_idx]
                            if isinstance(fit_points_left_json, bytes):
                                fit_points_left_json = fit_points_left_json.decode('utf-8')
                            if fit_points_left_json:
                                frame_data['perception']['fit_points_left'] = json.loads(fit_points_left_json)
                        except Exception:
                            pass
                    if 'perception/fit_points_right' in f and perception_idx < len(f['perception/fit_points_right']):
                        try:
                            fit_points_right_json = f['perception/fit_points_right'][perception_idx]
                            if isinstance(fit_points_right_json, bytes):
                                fit_points_right_json = fit_points_right_json.decode('utf-8')
                            if fit_points_right_json:
                                frame_data['perception']['fit_points_right'] = json.loads(fit_points_right_json)
                        except Exception:
                            pass
                    if 'perception/segmentation_mask_png' in f and perception_idx < len(f['perception/segmentation_mask_png']):
                        try:
                            mask_bytes = f['perception/segmentation_mask_png'][perception_idx]
                            has_data = False
                            if isinstance(mask_bytes, np.ndarray):
                                has_data = len(mask_bytes) > 0
                                mask_bytes = mask_bytes.tobytes() if has_data else b""
                            elif isinstance(mask_bytes, (bytes, bytearray)):
                                has_data = len(mask_bytes) > 0
                            if has_data:
                                mask_b64 = base64.b64encode(mask_bytes).decode('utf-8')
                                frame_data['perception']['segmentation_mask_png'] = (
                                    f"data:image/png;base64,{mask_b64}"
                                )
                        except Exception:
                            pass
                    
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
                trajectory_idx, trajectory_dt_s = find_nearest_with_diff(
                    trajectory_timestamps, camera_timestamp, max_diff=max_pair_diff_s["trajectory"]
                )
                
                if trajectory_idx is not None and trajectory_idx < len(f['trajectory/reference_point_x']):
                    frame_data['trajectory'] = {
                        'timestamp': float(trajectory_timestamps[trajectory_idx]),
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
                    if 'trajectory/oracle_points' in f and trajectory_idx < len(f['trajectory/oracle_points']):
                        oracle_points = f['trajectory/oracle_points'][trajectory_idx]
                        if oracle_points is not None and len(oracle_points) > 0:
                            try:
                                if oracle_points.ndim == 2 and oracle_points.shape[1] == 2:
                                    oracle_points_2d = oracle_points
                                elif oracle_points.ndim == 1:
                                    if len(oracle_points) % 2 == 0:
                                        oracle_points_2d = oracle_points.reshape(-1, 2)
                                    else:
                                        oracle_points_2d = np.array([], dtype=np.float32).reshape(0, 2)
                                else:
                                    oracle_points_2d = np.array([], dtype=np.float32).reshape(0, 2)
                                frame_data['trajectory']['oracle_points'] = [
                                    {'x': float(p[0]), 'y': float(p[1])}
                                    for p in oracle_points_2d
                                ]
                            except Exception as e:
                                print(f"Warning: Error reading oracle_points for frame {trajectory_idx}: {e}")
                                frame_data['trajectory']['oracle_points'] = []
                        else:
                            frame_data['trajectory']['oracle_points'] = []
                    if 'trajectory/oracle_point_count' in f and trajectory_idx < len(f['trajectory/oracle_point_count']):
                        frame_data['trajectory']['oracle_point_count'] = int(
                            f['trajectory/oracle_point_count'][trajectory_idx]
                        )
                    if 'trajectory/oracle_horizon_meters' in f and trajectory_idx < len(f['trajectory/oracle_horizon_meters']):
                        frame_data['trajectory']['oracle_horizon_meters'] = float(
                            f['trajectory/oracle_horizon_meters'][trajectory_idx]
                        )
                    if 'trajectory/oracle_point_spacing_meters' in f and trajectory_idx < len(f['trajectory/oracle_point_spacing_meters']):
                        frame_data['trajectory']['oracle_point_spacing_meters'] = float(
                            f['trajectory/oracle_point_spacing_meters'][trajectory_idx]
                        )
                    if 'trajectory/oracle_samples_enabled' in f and trajectory_idx < len(f['trajectory/oracle_samples_enabled']):
                        frame_data['trajectory']['oracle_samples_enabled'] = bool(
                            f['trajectory/oracle_samples_enabled'][trajectory_idx]
                        )
                    traj_diag_fields = [
                        "diag_available",
                        "diag_generated_by_fallback",
                        "diag_points_generated",
                        "diag_x_clip_count",
                        "diag_pre_y0",
                        "diag_pre_y1",
                        "diag_pre_y2",
                        "diag_post_y0",
                        "diag_post_y1",
                        "diag_post_y2",
                        "diag_used_provided_distance0",
                        "diag_used_provided_distance1",
                        "diag_used_provided_distance2",
                        "diag_post_minus_pre_y0",
                        "diag_post_minus_pre_y1",
                        "diag_post_minus_pre_y2",
                        "diag_preclip_x0",
                        "diag_preclip_x1",
                        "diag_preclip_x2",
                        "diag_preclip_x_abs_max",
                        "diag_preclip_x_abs_p95",
                        "diag_preclip_abs_mean_0_8m",
                        "diag_preclip_abs_mean_8_12m",
                        "diag_preclip_abs_mean_12_20m",
                        "diag_postclip_x0",
                        "diag_postclip_x1",
                        "diag_postclip_x2",
                        "diag_postclip_abs_mean_0_8m",
                        "diag_postclip_abs_mean_8_12m",
                        "diag_postclip_abs_mean_12_20m",
                        "diag_postclip_near_clip_frac_12_20m",
                        "diag_first_segment_y0_gt_y1_pre",
                        "diag_first_segment_y0_gt_y1_post",
                        "diag_inversion_introduced_after_conversion",
                    ]
                    for diag_key in traj_diag_fields:
                        ds = f"trajectory/{diag_key}"
                        if ds in f and trajectory_idx < len(f[ds]):
                            frame_data['trajectory'][diag_key] = float(f[ds][trajectory_idx])
            
            # Control data - find closest by timestamp
            if 'control/timestamps' in f and len(f['control/timestamps']) > 0:
                control_timestamps = np.array(f['control/timestamps'])
                control_idx, control_dt_s = find_nearest_with_diff(
                    control_timestamps, camera_timestamp, max_diff=max_pair_diff_s["control"]
                )
                
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
                    if 'control/total_error_scaled' in f and control_idx < len(f['control/total_error_scaled']):
                        frame_data['control']['total_error_scaled'] = float(f['control/total_error_scaled'][control_idx])
                    if 'control/feedforward_steering' in f and control_idx < len(f['control/feedforward_steering']):
                        frame_data['control']['feedforward_steering'] = float(f['control/feedforward_steering'][control_idx])
                    if 'control/feedback_steering' in f and control_idx < len(f['control/feedback_steering']):
                        frame_data['control']['feedback_steering'] = float(f['control/feedback_steering'][control_idx])
                    if 'control/straight_sign_flip_override_active' in f and control_idx < len(
                        f['control/straight_sign_flip_override_active']
                    ):
                        frame_data['control']['straight_sign_flip_override_active'] = (
                            int(f['control/straight_sign_flip_override_active'][control_idx]) == 1
                        )
                    if 'control/target_speed_raw' in f and control_idx < len(f['control/target_speed_raw']):
                        frame_data['control']['target_speed_raw'] = float(f['control/target_speed_raw'][control_idx])
                    if 'control/target_speed_post_limits' in f and control_idx < len(f['control/target_speed_post_limits']):
                        frame_data['control']['target_speed_post_limits'] = float(f['control/target_speed_post_limits'][control_idx])
                    if 'control/target_speed_planned' in f and control_idx < len(f['control/target_speed_planned']):
                        frame_data['control']['target_speed_planned'] = float(f['control/target_speed_planned'][control_idx])
                    if 'control/target_speed_final' in f and control_idx < len(f['control/target_speed_final']):
                        frame_data['control']['target_speed_final'] = float(f['control/target_speed_final'][control_idx])
                    if 'control/target_speed_slew_active' in f and control_idx < len(f['control/target_speed_slew_active']):
                        frame_data['control']['target_speed_slew_active'] = int(f['control/target_speed_slew_active'][control_idx]) == 1
                    if 'control/target_speed_ramp_active' in f and control_idx < len(f['control/target_speed_ramp_active']):
                        frame_data['control']['target_speed_ramp_active'] = int(f['control/target_speed_ramp_active'][control_idx]) == 1
                    if 'control/launch_throttle_cap' in f and control_idx < len(f['control/launch_throttle_cap']):
                        frame_data['control']['launch_throttle_cap'] = float(f['control/launch_throttle_cap'][control_idx])
                    if 'control/launch_throttle_cap_active' in f and control_idx < len(f['control/launch_throttle_cap_active']):
                        frame_data['control']['launch_throttle_cap_active'] = int(f['control/launch_throttle_cap_active'][control_idx]) == 1
                    if 'control/is_straight' in f and control_idx < len(f['control/is_straight']):
                        frame_data['control']['is_straight'] = int(f['control/is_straight'][control_idx]) == 1
                    if 'control/straight_oscillation_rate' in f and control_idx < len(f['control/straight_oscillation_rate']):
                        frame_data['control']['straight_oscillation_rate'] = float(f['control/straight_oscillation_rate'][control_idx])
                    if 'control/tuned_deadband' in f and control_idx < len(f['control/tuned_deadband']):
                        frame_data['control']['tuned_deadband'] = float(f['control/tuned_deadband'][control_idx])
                    if 'control/tuned_error_smoothing_alpha' in f and control_idx < len(f['control/tuned_error_smoothing_alpha']):
                        frame_data['control']['tuned_error_smoothing_alpha'] = float(f['control/tuned_error_smoothing_alpha'][control_idx])
                    if 'control/steering_pre_rate_limit' in f and control_idx < len(f['control/steering_pre_rate_limit']):
                        frame_data['control']['steering_pre_rate_limit'] = float(f['control/steering_pre_rate_limit'][control_idx])
                    if 'control/steering_post_rate_limit' in f and control_idx < len(f['control/steering_post_rate_limit']):
                        frame_data['control']['steering_post_rate_limit'] = float(f['control/steering_post_rate_limit'][control_idx])
                    if 'control/steering_post_jerk_limit' in f and control_idx < len(f['control/steering_post_jerk_limit']):
                        frame_data['control']['steering_post_jerk_limit'] = float(f['control/steering_post_jerk_limit'][control_idx])
                    if 'control/steering_post_sign_flip' in f and control_idx < len(f['control/steering_post_sign_flip']):
                        frame_data['control']['steering_post_sign_flip'] = float(f['control/steering_post_sign_flip'][control_idx])
                    if 'control/steering_post_hard_clip' in f and control_idx < len(f['control/steering_post_hard_clip']):
                        frame_data['control']['steering_post_hard_clip'] = float(f['control/steering_post_hard_clip'][control_idx])
                    if 'control/steering_post_smoothing' in f and control_idx < len(f['control/steering_post_smoothing']):
                        frame_data['control']['steering_post_smoothing'] = float(f['control/steering_post_smoothing'][control_idx])
                    if 'control/steering_rate_limited_active' in f and control_idx < len(f['control/steering_rate_limited_active']):
                        frame_data['control']['steering_rate_limited_active'] = int(f['control/steering_rate_limited_active'][control_idx]) == 1
                    if 'control/steering_jerk_limited_active' in f and control_idx < len(f['control/steering_jerk_limited_active']):
                        frame_data['control']['steering_jerk_limited_active'] = int(f['control/steering_jerk_limited_active'][control_idx]) == 1
                    if 'control/steering_hard_clip_active' in f and control_idx < len(f['control/steering_hard_clip_active']):
                        frame_data['control']['steering_hard_clip_active'] = int(f['control/steering_hard_clip_active'][control_idx]) == 1
                    if 'control/steering_smoothing_active' in f and control_idx < len(f['control/steering_smoothing_active']):
                        frame_data['control']['steering_smoothing_active'] = int(f['control/steering_smoothing_active'][control_idx]) == 1
                    if 'control/steering_rate_limited_delta' in f and control_idx < len(f['control/steering_rate_limited_delta']):
                        frame_data['control']['steering_rate_limited_delta'] = float(f['control/steering_rate_limited_delta'][control_idx])
                    if 'control/steering_jerk_limited_delta' in f and control_idx < len(f['control/steering_jerk_limited_delta']):
                        frame_data['control']['steering_jerk_limited_delta'] = float(f['control/steering_jerk_limited_delta'][control_idx])
                    if 'control/steering_hard_clip_delta' in f and control_idx < len(f['control/steering_hard_clip_delta']):
                        frame_data['control']['steering_hard_clip_delta'] = float(f['control/steering_hard_clip_delta'][control_idx])
                    if 'control/steering_smoothing_delta' in f and control_idx < len(f['control/steering_smoothing_delta']):
                        frame_data['control']['steering_smoothing_delta'] = float(f['control/steering_smoothing_delta'][control_idx])
            
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
                unity_idx, _unity_dt_s = find_nearest_with_diff(
                    unity_timestamps, camera_timestamp, max_diff=1.0
                )  # Larger tolerance (sent every 1s)
                
                if unity_idx is not None and unity_idx < len(f['unity_feedback/ground_truth_mode_active']):
                    frame_data['unity_feedback'] = {
                        'ground_truth_mode_active': bool(f['unity_feedback/ground_truth_mode_active'][unity_idx]),
                        'path_curvature_calculated': bool(f['unity_feedback/path_curvature_calculated'][unity_idx]) if 'unity_feedback/path_curvature_calculated' in f and unity_idx < len(f['unity_feedback/path_curvature_calculated']) else None
                    }
                    if 'unity_feedback/actual_steering_applied' in f and unity_idx < len(f['unity_feedback/actual_steering_applied']):
                        frame_data['unity_feedback']['actual_steering_applied'] = float(f['unity_feedback/actual_steering_applied'][unity_idx])

            alignment_window_ms = 20.0

            def _alignment_for(dt_s, idx):
                if camera_timestamp is None or not np.isfinite(camera_timestamp):
                    return "missing", "camera_timestamp_invalid"
                if idx is None:
                    return "missing", "missing_sample_in_window"
                if dt_s is None or not np.isfinite(dt_s):
                    return "missing", "invalid_dt"
                dt_ms = float(dt_s) * 1000.0
                if abs(dt_ms) <= alignment_window_ms:
                    return "aligned", "within_window"
                return "misaligned", "out_of_window"

            traj_status, traj_reason = _alignment_for(trajectory_dt_s, trajectory_idx)
            control_status, control_reason = _alignment_for(control_dt_s, control_idx)
            vehicle_status, vehicle_reason = _alignment_for(vehicle_dt_s, vehicle_idx)
            overall = "aligned" if (traj_status == "aligned" and control_status == "aligned") else "degraded"

            frame_data['sync'] = {
                'alignment_window_ms': alignment_window_ms,
                'camera_timestamp': float(camera_timestamp) if camera_timestamp is not None else None,
                'vehicle_timestamp': frame_data.get('vehicle', {}).get('timestamp'),
                'trajectory_timestamp': frame_data.get('trajectory', {}).get('timestamp'),
                'control_timestamp': frame_data.get('control', {}).get('timestamp'),
                'perception_timestamp': frame_data.get('perception', {}).get('timestamp'),
                'dt_cam_vehicle_ms': float(vehicle_dt_s) * 1000.0 if vehicle_dt_s is not None else None,
                'dt_cam_traj_ms': float(trajectory_dt_s) * 1000.0 if trajectory_dt_s is not None else None,
                'dt_cam_control_ms': float(control_dt_s) * 1000.0 if control_dt_s is not None else None,
                'dt_cam_perception_ms': float(perception_dt_s) * 1000.0 if perception_dt_s is not None else None,
                'vehicle_alignment_status': vehicle_status,
                'vehicle_alignment_reason': vehicle_reason,
                'trajectory_alignment_status': traj_status,
                'trajectory_alignment_reason': traj_reason,
                'control_alignment_status': control_status,
                'control_alignment_reason': control_reason,
                'overall_alignment_status': overall,
                'source_indices': {
                    'camera_index': int(frame_index),
                    'vehicle_index': int(vehicle_idx) if vehicle_idx is not None else None,
                    'trajectory_index': int(trajectory_idx) if trajectory_idx is not None else None,
                    'control_index': int(control_idx) if control_idx is not None else None,
                    'perception_index': int(perception_idx) if perception_idx is not None else None,
                },
                'pairing_max_diff_ms': {
                    'vehicle': float(max_pair_diff_s["vehicle"] * 1000.0),
                    'trajectory': float(max_pair_diff_s["trajectory"] * 1000.0),
                    'control': float(max_pair_diff_s["control"] * 1000.0),
                    'perception': float(max_pair_diff_s["perception"] * 1000.0),
                },
            }
            
            return jsonify(frame_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/recording/<path:filename>/frame/<int:frame_index>/image')
def get_frame_image(filename, frame_index):
    """Get camera frame as base64-encoded image."""
    from urllib.parse import unquote
    filename = unquote(filename)
    filepath = RECORDINGS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": f"Recording not found: {filename}"}), 404
    
    camera_id = request.args.get("camera_id", "front_center")
    use_topdown = camera_id in ("top_down", "topdown")

    try:
        with h5py.File(filepath, 'r') as f:
            if use_topdown:
                if 'camera/topdown_images' not in f or frame_index >= len(f['camera/topdown_images']):
                    return jsonify({"error": "Frame not found"}), 404
                image = f['camera/topdown_images'][frame_index]
            else:
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


@app.route('/api/recording/<path:filename>/frame/<int:frame_index>/generate-debug')
def generate_debug_overlays(filename, frame_index):
    """Generate debug overlays (edges, yellow_mask, combined) for a specific frame on-demand.
    
    This allows viewing debug overlays for any frame, not just every 30th frame.
    """
    from urllib.parse import unquote
    filename = unquote(filename)
    filepath = RECORDINGS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": f"Recording not found: {filename}"}), 404
    
    try:
        # Import here to avoid circular dependencies
        import sys
        from pathlib import Path
        import base64
        from io import BytesIO
        from PIL import Image
        
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        
        from perception.models.lane_detection import SimpleLaneDetector
        import cv2
        
        with h5py.File(filepath, 'r') as f:
            if 'camera/images' not in f or frame_index >= len(f['camera/images']):
                return jsonify({"error": "Frame not found"}), 404
            
            # Load image
            image = f['camera/images'][frame_index]
            
            # Re-run perception to get debug images AND fit_points
            detector = SimpleLaneDetector()
            # Call detect with return_debug=True to get fit_points
            lane_coeffs, debug_info = detector.detect(image, return_debug=True)
            
            # Extract fit_points from debug_info
            fit_points_left = None
            fit_points_right = None
            if debug_info and 'fit_points' in debug_info:
                fit_points = debug_info['fit_points']
                if isinstance(fit_points, dict):
                    fit_points_left = fit_points.get('left')
                    fit_points_right = fit_points.get('right')
                    # Convert numpy arrays to lists for JSON serialization
                    if fit_points_left is not None:
                        fit_points_left = fit_points_left.tolist() if hasattr(fit_points_left, 'tolist') else fit_points_left
                    if fit_points_right is not None:
                        fit_points_right = fit_points_right.tolist() if hasattr(fit_points_right, 'tolist') else fit_points_right
            
            # Extract lane_line_coefficients from lane_coeffs (for orange curves)
            # Format: [[left_a, left_b, left_c], [right_a, right_b, right_c]] or [None, ...] if not detected
            lane_line_coefficients = []
            if lane_coeffs[0] is not None:
                lane_line_coefficients.append(lane_coeffs[0].tolist() if hasattr(lane_coeffs[0], 'tolist') else list(lane_coeffs[0]))
            else:
                lane_line_coefficients.append(None)
            if lane_coeffs[1] is not None:
                lane_line_coefficients.append(lane_coeffs[1].tolist() if hasattr(lane_coeffs[1], 'tolist') else list(lane_coeffs[1]))
            else:
                lane_line_coefficients.append(None)
            
            # Also generate the debug images manually (for consistency with existing code)
            h, w = image.shape[:2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Generate yellow mask
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            yellow_mask = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))
            
            # Generate edges
            mean_brightness = np.mean(gray)
            brightness_factor = max(0.5, min(2.0, mean_brightness / 128.0))
            median_val = np.median(blurred)
            lower_threshold = int(max(10, median_val * 0.3))
            upper_threshold = int(min(200, median_val * 2.0))
            edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
            
            # Generate combined
            # Create ROI mask - MATCH ACTUAL DETECTION SETTINGS
            # Detection uses: roi_margin=0 (0% margin, 100% width), start at 18% from top, exclude bottom 20%
            roi_margin = 0  # No horizontal margin - use full image width (matches detection)
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            roi_vertices = np.array([[
                (roi_margin, int(h * 0.80)),  # Stop 20% from bottom (matches detection)
                (w - roi_margin, int(h * 0.80)),
                (w - roi_margin, int(h * 0.18)),  # Start at 18% from top (matches detection)
                (roi_margin, int(h * 0.18))
            ]], dtype=np.int32)
            cv2.fillPoly(roi_mask, roi_vertices, 255)
            
            # Apply ROI to edges
            edges_roi = cv2.bitwise_and(edges, roi_mask)
            
            # Combined (yellow mask + edges)
            # NOTE: yellow_mask is created on full image, but we apply ROI mask here
            # This matches the detection logic: yellow_mask is full image, then ROI is applied
            yellow_mask_roi = cv2.bitwise_and(yellow_mask, roi_mask)
            combined = cv2.bitwise_or(yellow_mask_roi, edges_roi)
            
            # Convert to base64 for JSON response
            def image_to_base64(img_array, is_binary=False):
                """Convert numpy array to base64 PNG."""
                if is_binary:
                    # Binary mask - convert to 3-channel for display
                    img_3channel = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                    img_pil = Image.fromarray(img_3channel)
                else:
                    img_pil = Image.fromarray(img_array)
                
                buffer = BytesIO()
                img_pil.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return f"data:image/png;base64,{img_str}"
            
            result = {
                "frame_index": frame_index,
                "edges": image_to_base64(edges_roi, is_binary=True),
                "yellow_mask": image_to_base64(yellow_mask_roi, is_binary=True),
                "combined": image_to_base64(combined, is_binary=True)
            }
            
            # Add fit_points if available
            if fit_points_left is not None:
                result["fit_points_left"] = fit_points_left
            if fit_points_right is not None:
                result["fit_points_right"] = fit_points_right
            
            # Add lane_line_coefficients (for orange curves) - these are the newly generated coefficients
            result["lane_line_coefficients"] = lane_line_coefficients
            
            return jsonify(result)
            
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/api/recording/<path:filename>/frame/<int:frame_index>/polynomial-analysis')
def get_polynomial_analysis(filename, frame_index):
    """Get polynomial fitting analysis for a specific frame.
    
    Re-runs perception on the frame and returns detailed debug information
    about detected line segments, points used for fitting, and polynomial evaluation.
    """
    from urllib.parse import unquote
    filename = unquote(filename)
    filepath = RECORDINGS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": f"Recording not found: {filename}"}), 404
    
    try:
        # Import here to avoid circular dependencies
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        
        from perception.models.lane_detection import SimpleLaneDetector
        
        with h5py.File(filepath, 'r') as f:
            if 'camera/images' not in f or frame_index >= len(f['camera/images']):
                return jsonify({"error": "Frame not found"}), 404
            
            # Load image
            image = f['camera/images'][frame_index]
            
            # Load what was actually recorded in the original run
            recorded_data = {}
            if 'perception/timestamps' in f and len(f['perception/timestamps']) > 0:
                # Find matching perception frame by timestamp
                camera_timestamp = float(f['camera/timestamps'][frame_index])
                perception_timestamps = np.array(f['perception/timestamps'])
                diffs = np.abs(perception_timestamps - camera_timestamp)
                perception_idx = int(np.argmin(diffs)) if len(diffs) > 0 else None
                
                if perception_idx is not None and perception_idx < len(f['perception/timestamps']):
                    recorded_data = {
                        "num_lanes_detected": int(f['perception/num_lanes_detected'][perception_idx]) if 'perception/num_lanes_detected' in f and perception_idx < len(f['perception/num_lanes_detected']) else 0,
                        "lane_coefficients": None,
                        "using_stale_data": bool(f['perception/using_stale_data'][perception_idx]) if 'perception/using_stale_data' in f and perception_idx < len(f['perception/using_stale_data']) else False,
                        "stale_data_reason": None,
                        "reject_reason": None,
                    }
                    
                    # Load recorded coefficients
                    if 'perception/lane_line_coefficients' in f and perception_idx < len(f['perception/lane_line_coefficients']):
                        recorded_coeffs = f['perception/lane_line_coefficients'][perception_idx]
                        if hasattr(recorded_coeffs, '__len__') and len(recorded_coeffs) >= 6:
                            # Flattened format: [left_a, left_b, left_c, right_a, right_b, right_c]
                            recorded_data["lane_coefficients"] = [
                                numpy_to_list(recorded_coeffs[0:3]) if not np.all(np.isnan(recorded_coeffs[0:3])) else None,
                                numpy_to_list(recorded_coeffs[3:6]) if not np.all(np.isnan(recorded_coeffs[3:6])) else None
                            ]
                    
                    if 'perception/stale_reason' in f and perception_idx < len(f['perception/stale_reason']):
                        reason = f['perception/stale_reason'][perception_idx]
                        if isinstance(reason, bytes):
                            reason = reason.decode('utf-8')
                        recorded_data["stale_data_reason"] = reason
                    if 'perception/reject_reason' in f and perception_idx < len(f['perception/reject_reason']):
                        reason = f['perception/reject_reason'][perception_idx]
                        if isinstance(reason, bytes):
                            reason = reason.decode('utf-8')
                        recorded_data["reject_reason"] = reason
            
            # Re-run perception with debug info (current code)
            detector = SimpleLaneDetector()
            result = detector.detect(image, return_debug=True)
            
            if isinstance(result, tuple) and len(result) == 2:
                lane_coeffs, debug_info = result
                
                analysis = {
                    "frame_index": frame_index,
                    "recorded": recorded_data,  # What was actually recorded
                    "rerun": {  # What current code detects
                        "num_lanes_detected": sum(1 for c in lane_coeffs if c is not None),
                        "lane_coefficients": [
                            numpy_to_list(c) if c is not None else None 
                            for c in lane_coeffs
                        ]
                    },
                    "num_lanes_detected": sum(1 for c in lane_coeffs if c is not None),  # Keep for backward compatibility
                    "lane_coefficients": [
                        numpy_to_list(c) if c is not None else None 
                        for c in lane_coeffs
                    ],
                    "debug_info": {}
                }
                
                if debug_info:
                    # Extract relevant debug information
                    analysis["debug_info"] = {
                        "num_lines_detected": debug_info.get('num_lines_detected', 0),
                        "left_lines_count": debug_info.get('left_lines_count', 0),
                        "right_lines_count": debug_info.get('right_lines_count', 0),
                        "all_lines": numpy_to_list(debug_info.get('all_lines', [])),
                        "validation_failures": debug_info.get('validation_failures', {})
                    }
                    
                    # Add point information if available
                    for lane_name in ['left', 'right']:
                        points_key = f'{lane_name}_points_for_fit'
                        if points_key in debug_info and debug_info[points_key] is not None:
                            points = debug_info[points_key]
                            analysis["debug_info"][f'{lane_name}_points'] = numpy_to_list(points)
                            analysis["debug_info"][f'{lane_name}_points_count'] = len(points)
                            if len(points) > 0:
                                analysis["debug_info"][f'{lane_name}_y_range'] = [
                                    float(np.min(points[:, 1])),
                                    float(np.max(points[:, 1]))
                                ]
                                analysis["debug_info"][f'{lane_name}_x_range'] = [
                                    float(np.min(points[:, 0])),
                                    float(np.max(points[:, 0]))
                                ]
                
                # Evaluate polynomials at various y positions
                image_height = image.shape[0]
                image_width = image.shape[1]
                y_eval_points = [0, image_height // 4, image_height // 2, image_height * 3 // 4, image_height - 1]
                
                for lane_idx, lane_name in enumerate(['left', 'right']):
                    if lane_coeffs[lane_idx] is not None:
                        coeffs = lane_coeffs[lane_idx]
                        evaluations = []
                        for y_val in y_eval_points:
                            x_eval = coeffs[0] * y_val * y_val + coeffs[1] * y_val + coeffs[2]
                            evaluations.append({
                                "y": int(y_val),
                                "x": float(x_eval),
                                "in_bounds": bool(0 <= x_eval <= image_width)  # Convert numpy bool to Python bool
                            })
                        analysis["debug_info"][f'{lane_name}_polynomial_evaluation'] = evaluations
                
                # NEW: Run full system validation (same as av_stack.py)
                # This shows what the real system would reject
                from trajectory.models.trajectory_planner import RuleBasedTrajectoryPlanner
                
                # Initialize planner (needed for coordinate conversion)
                planner = RuleBasedTrajectoryPlanner(
                    lookahead_distance=20.0,
                    image_width=float(image_width),
                    image_height=float(image_height)
                )
                
                full_system_validation = {
                    "polynomial_x_validation": {},
                    "lane_width_validation": None,
                    "would_reject": False,  # Python bool, should be fine
                    "rejection_reasons": []
                }
                
                # 1. Polynomial x-value validation (same as av_stack.py)
                # CRITICAL: Only validate where we actually use the polynomial (lookahead distance)
                # Don't check top of image (y=0) - extrapolation there can be extreme on curves
                # Get lookahead y position from recording if available, otherwise use middle
                try:
                    # Try to get camera_8m_screen_y from recording (most accurate)
                    y_image_at_lookahead = None
                    if 'vehicle/camera_8m_screen_y' in h5_file and frame_idx < len(h5_file['vehicle/camera_8m_screen_y']):
                        camera_8m_screen_y = float(h5_file['vehicle/camera_8m_screen_y'][frame_idx])
                        if camera_8m_screen_y > 0:
                            y_image_at_lookahead = int(camera_8m_screen_y)
                    
                    if y_image_at_lookahead is None:
                        # Fallback: Use middle of image (typical lookahead position)
                        y_image_at_lookahead = int(image_height * 0.7)  # Bottom 30% of image
                except:
                    # If we can't get it, use middle
                    y_image_at_lookahead = int(image_height * 0.7)
                
                # CRITICAL: Only check at lookahead distance (where we actually use the polynomial)
                # Don't check bottom - missing dashed lines can cause incorrect extrapolation there
                # The polynomial is correct at the lookahead distance, which is what matters for control
                y_check_positions = [y_image_at_lookahead]  # Only check where we actually use it
                max_reasonable_x = image_width * 2.5  # Allow up to 2.5x image width (for curves)
                min_reasonable_x = -image_width * 1.5  # Allow some negative (for curves)
                
                for lane_idx, lane_name in enumerate(['left', 'right']):
                    if lane_coeffs[lane_idx] is not None and len(lane_coeffs[lane_idx]) >= 3:
                        coeffs = lane_coeffs[lane_idx]
                        extreme_detected = False
                        extreme_positions = []
                        
                        for y_check in y_check_positions:
                            x_eval = coeffs[0] * y_check * y_check + coeffs[1] * y_check + coeffs[2]
                            if x_eval < min_reasonable_x or x_eval > max_reasonable_x:
                                extreme_detected = True
                                extreme_positions.append({
                                    "y": int(y_check),
                                    "x": float(x_eval),
                                    "reason": f"x={x_eval:.1f}px outside range [{min_reasonable_x:.0f}, {max_reasonable_x:.0f}]px"
                                })
                        
                        full_system_validation["polynomial_x_validation"][lane_name] = {
                            "passed": bool(not extreme_detected),  # Convert to Python bool
                            "extreme_positions": extreme_positions if extreme_detected else []
                        }
                        
                        if extreme_detected:
                            full_system_validation["would_reject"] = True
                            full_system_validation["rejection_reasons"].append(f"{lane_name}_extreme_coefficients")
                
                # 2. Lane width validation (convert to vehicle coords and check width)
                if (lane_coeffs[0] is not None and lane_coeffs[1] is not None and 
                    len(lane_coeffs[0]) >= 3 and len(lane_coeffs[1]) >= 3):
                    try:
                        # Get lookahead distance and y position (simplified - use middle of image)
                        lookahead_distance = 8.0  # meters (typical lookahead)
                        y_image_at_lookahead = image_height * 0.7  # Bottom 30% of image
                        
                        # Evaluate polynomials at lookahead
                        left_x_image = lane_coeffs[0][0] * y_image_at_lookahead * y_image_at_lookahead + \
                                      lane_coeffs[0][1] * y_image_at_lookahead + lane_coeffs[0][2]
                        right_x_image = lane_coeffs[1][0] * y_image_at_lookahead * y_image_at_lookahead + \
                                       lane_coeffs[1][1] * y_image_at_lookahead + lane_coeffs[1][2]
                        
                        # Convert to vehicle coordinates (simplified - use planner's method)
                        # Note: This is approximate - full conversion needs camera FOV from recording
                        left_x_vehicle, _ = planner._convert_image_to_vehicle_coords(
                            left_x_image, y_image_at_lookahead, lookahead_distance=lookahead_distance
                        )
                        right_x_vehicle, _ = planner._convert_image_to_vehicle_coords(
                            right_x_image, y_image_at_lookahead, lookahead_distance=lookahead_distance
                        )
                        
                        calculated_lane_width = right_x_vehicle - left_x_vehicle
                        min_lane_width = 2.0  # meters
                        max_lane_width = 10.0  # meters
                        
                        width_valid = min_lane_width <= calculated_lane_width <= max_lane_width
                        
                        full_system_validation["lane_width_validation"] = {
                            "passed": bool(width_valid),  # Convert to Python bool
                            "width_meters": float(calculated_lane_width),
                            "left_x_vehicle": float(left_x_vehicle),
                            "right_x_vehicle": float(right_x_vehicle),
                            "expected_range": [float(min_lane_width), float(max_lane_width)]
                        }
                        
                        if not width_valid:
                            full_system_validation["would_reject"] = True
                            full_system_validation["rejection_reasons"].append("invalid_width")
                    except Exception as e:
                        # If conversion fails, skip width validation
                        full_system_validation["lane_width_validation"] = {
                            "passed": None,
                            "error": str(e)
                        }
                
                analysis["full_system_validation"] = full_system_validation
                
                return jsonify(analysis)
            else:
                return jsonify({"error": "Failed to get debug info from detector"}), 500
                
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/')
def index():
    """Serve the main HTML file."""
    return send_from_directory(Path(__file__).parent, 'index.html')


@app.route('/api/recording/<path:filename>/summary')
def get_recording_summary(filename):
    """Get recording summary metrics."""
    from flask import request
    from urllib.parse import unquote
    
    filename = unquote(filename)
    filepath = RECORDINGS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": f"Recording not found: {filename}"}), 404
    
    # Get query parameter for analyze_to_failure
    analyze_to_failure = request.args.get('analyze_to_failure', 'false').lower() == 'true'
    
    try:
        summary = analyze_recording_summary(filepath, analyze_to_failure=analyze_to_failure)
        return jsonify(numpy_to_list(summary))
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/api/recording/<path:filename>/issues')
def get_recording_issues(filename):
    """Get detected issues in recording."""
    from flask import request
    from urllib.parse import unquote
    
    filename = unquote(filename)
    filepath = RECORDINGS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": f"Recording not found: {filename}"}), 404
    
    # Get query parameter for analyze_to_failure
    analyze_to_failure = request.args.get('analyze_to_failure', 'false').lower() == 'true'
    
    try:
        issues_data = detect_issues(filepath, analyze_to_failure=analyze_to_failure)
        return jsonify(numpy_to_list(issues_data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/recording/<path:filename>/diagnostics')
def get_recording_diagnostics(filename):
    """Get trajectory vs steering diagnostic analysis."""
    from flask import request
    from urllib.parse import unquote
    
    filename = unquote(filename)
    filepath = RECORDINGS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": f"Recording not found: {filename}"}), 404
    
    # Get query parameter for analyze_to_failure
    analyze_to_failure = request.args.get('analyze_to_failure', 'false').lower() == 'true'
    curve_entry_start_distance_m = request.args.get('curve_entry_start_distance_m', None)
    curve_entry_window_distance_m = request.args.get('curve_entry_window_distance_m', None)
    try:
        curve_entry_start_distance_m = (
            float(curve_entry_start_distance_m)
            if curve_entry_start_distance_m is not None and curve_entry_start_distance_m != ''
            else None
        )
    except (TypeError, ValueError):
        curve_entry_start_distance_m = None
    try:
        curve_entry_window_distance_m = (
            float(curve_entry_window_distance_m)
            if curve_entry_window_distance_m is not None and curve_entry_window_distance_m != ''
            else None
        )
    except (TypeError, ValueError):
        curve_entry_window_distance_m = None
    
    try:
        diagnostics = analyze_trajectory_vs_steering(
            filepath,
            analyze_to_failure=analyze_to_failure,
            curve_entry_start_distance_m=curve_entry_start_distance_m,
            curve_entry_window_distance_m=curve_entry_window_distance_m,
        )
        return jsonify(numpy_to_list(diagnostics))
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/api/recording/<path:filename>/trajectory-layer-localization')
def get_trajectory_layer_localization(filename):
    """Get run-level trajectory layer/location localization summary."""
    from urllib.parse import unquote

    filename = unquote(filename)
    filepath = RECORDINGS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": f"Recording not found: {filename}"}), 404

    clip_limit_m_raw = request.args.get('clip_limit_m', '15.0')
    try:
        clip_limit_m = float(clip_limit_m_raw)
    except (TypeError, ValueError):
        clip_limit_m = 15.0

    try:
        summary = analyze_trajectory_layer_localization(filepath, clip_limit_m=clip_limit_m)
        return jsonify(numpy_to_list(summary))
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/api/recording/<path:filename>/topdown-diagnostics')
def get_topdown_diagnostics(filename):
    """Return timing/projection diagnostics for top-down trajectory overlay trust."""
    from urllib.parse import unquote

    def _nearest_abs_deltas(reference_ts: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
        if reference_ts.size == 0 or target_ts.size == 0:
            return np.array([], dtype=float)
        idx = np.searchsorted(target_ts, reference_ts, side='left')
        idx = np.clip(idx, 0, target_ts.size - 1)
        prev_idx = np.clip(idx - 1, 0, target_ts.size - 1)
        d1 = np.abs(reference_ts - target_ts[idx])
        d0 = np.abs(reference_ts - target_ts[prev_idx])
        return np.minimum(d0, d1)

    def _stats_ms(seconds: np.ndarray) -> dict:
        if seconds.size == 0:
            return {"count": 0, "mean_ms": None, "p95_ms": None, "max_ms": None}
        abs_ms = np.abs(seconds) * 1000.0
        return {
            "count": int(abs_ms.size),
            "mean_ms": float(np.mean(abs_ms)),
            "p95_ms": float(np.percentile(abs_ms, 95)),
            "max_ms": float(np.max(abs_ms)),
        }

    def _quality_from_p95(p95_ms: float | None) -> str:
        if p95_ms is None:
            return "unknown"
        if p95_ms <= 33.5:
            return "good"
        if p95_ms <= 66.5:
            return "warn"
        return "poor"

    def _nearest_indices(reference_ts: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
        if reference_ts.size == 0 or target_ts.size == 0:
            return np.array([], dtype=int)
        idx = np.searchsorted(target_ts, reference_ts, side='left')
        idx = np.clip(idx, 0, target_ts.size - 1)
        prev_idx = np.clip(idx - 1, 0, target_ts.size - 1)
        d1 = np.abs(reference_ts - target_ts[idx])
        d0 = np.abs(reference_ts - target_ts[prev_idx])
        choose_prev = d0 <= d1
        return np.where(choose_prev, prev_idx, idx).astype(int)

    def _stats_signed(arr: np.ndarray) -> dict:
        if arr.size == 0:
            return {
                "count": 0,
                "mean": None,
                "p95_abs": None,
                "max_abs": None,
            }
        abs_arr = np.abs(arr)
        return {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "p95_abs": float(np.percentile(abs_arr, 95)),
            "max_abs": float(np.max(abs_arr)),
        }

    def _finite_stats(arr: np.ndarray) -> dict:
        if arr.size == 0:
            return {"count": 0, "valid_ratio": 0.0, "mean": None, "p95": None, "max": None}
        finite = np.isfinite(arr)
        valid = arr[finite]
        if valid.size == 0:
            return {"count": int(arr.size), "valid_ratio": 0.0, "mean": None, "p95": None, "max": None}
        abs_valid = np.abs(valid)
        return {
            "count": int(arr.size),
            "valid_ratio": float(np.mean(finite)),
            "mean": float(np.mean(valid)),
            "p95": float(np.percentile(valid, 95)),
            "max": float(np.max(abs_valid)),
        }

    def _parse_traj_xy(row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        arr = np.array(row, dtype=float).reshape(-1)
        if arr.size < 3 or arr.size % 3 != 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        arr = arr.reshape(-1, 3)
        x = arr[:, 0]
        y = arr[:, 1]
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            return np.array([], dtype=float), np.array([], dtype=float)
        return x[finite], y[finite]

    def _parse_oracle_xy(row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        arr = np.array(row, dtype=float).reshape(-1)
        if arr.size < 2 or arr.size % 2 != 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        arr = arr.reshape(-1, 2)
        x = arr[:, 0]
        y = arr[:, 1]
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            return np.array([], dtype=float), np.array([], dtype=float)
        return x[finite], y[finite]

    def _prepare_curve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if x.size == 0 or y.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            return np.array([], dtype=float), np.array([], dtype=float)
        x = x[finite]
        y = y[finite]
        order = np.argsort(y)
        y_sorted = y[order]
        x_sorted = x[order]
        y_unique, unique_idx = np.unique(y_sorted, return_index=True)
        x_unique = x_sorted[unique_idx]
        return y_unique, x_unique

    filename = unquote(filename)
    filepath = RECORDINGS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": f"Recording not found: {filename}"}), 404

    try:
        with h5py.File(filepath, 'r') as f:
            topdown_ts = (
                np.array(f['camera/topdown_timestamps'][:], dtype=float)
                if 'camera/topdown_timestamps' in f
                else np.array([], dtype=float)
            )
            topdown_frame_ids = (
                np.array(f['camera/topdown_frame_ids'][:], dtype=float)
                if 'camera/topdown_frame_ids' in f
                else np.array([], dtype=float)
            )
            trajectory_ts = (
                np.array(f['trajectory/timestamps'][:], dtype=float)
                if 'trajectory/timestamps' in f
                else np.array([], dtype=float)
            )
            unity_ts = (
                np.array(f['vehicle/unity_time'][:], dtype=float)
                if 'vehicle/unity_time' in f
                else np.array([], dtype=float)
            )

            dt_topdown_traj = _nearest_abs_deltas(topdown_ts, trajectory_ts)
            dt_topdown_unity = _nearest_abs_deltas(topdown_ts, unity_ts)
            stats_traj = _stats_ms(dt_topdown_traj)
            stats_unity = _stats_ms(dt_topdown_unity)
            nearest_traj_idx = _nearest_indices(topdown_ts, trajectory_ts)
            nearest_unity_idx = _nearest_indices(topdown_ts, unity_ts)
            td_idx = np.arange(topdown_ts.size, dtype=float)
            idx_delta_traj = (
                nearest_traj_idx.astype(float) - td_idx
                if nearest_traj_idx.size == topdown_ts.size
                else np.array([], dtype=float)
            )
            idx_delta_unity = (
                nearest_unity_idx.astype(float) - td_idx
                if nearest_unity_idx.size == topdown_ts.size
                else np.array([], dtype=float)
            )
            traj_period_ms = (
                float(np.median(np.diff(trajectory_ts)) * 1000.0)
                if trajectory_ts.size > 1
                else None
            )
            unity_period_ms = (
                float(np.median(np.diff(unity_ts)) * 1000.0)
                if unity_ts.size > 1
                else None
            )
            dt_topdown_traj_frames = (
                (np.abs(dt_topdown_traj) * 1000.0) / traj_period_ms
                if traj_period_ms and traj_period_ms > 1e-6
                else np.array([], dtype=float)
            )
            dt_topdown_unity_frames = (
                (np.abs(dt_topdown_unity) * 1000.0) / unity_period_ms
                if unity_period_ms and unity_period_ms > 1e-6
                else np.array([], dtype=float)
            )
            topdown_frame_id_step = (
                np.diff(topdown_frame_ids) if topdown_frame_ids.size > 1 else np.array([], dtype=float)
            )
            topdown_frame_id_monotonic_ratio = (
                float(np.mean(topdown_frame_id_step >= 0.0))
                if topdown_frame_id_step.size > 0
                else None
            )
            timestamp_domain_mismatch_suspected = (
                stats_traj["p95_ms"] is not None
                and stats_traj["p95_ms"] > 120.0
                and _stats_signed(idx_delta_traj)["p95_abs"] is not None
                and _stats_signed(idx_delta_traj)["p95_abs"] <= 3.0
            )

            # Consume-point lag metrics recorded by AV stack (if present).
            stream_front_unity_dt_ms = (
                np.array(f['vehicle/stream_front_unity_dt_ms'][:], dtype=float)
                if 'vehicle/stream_front_unity_dt_ms' in f
                else np.array([], dtype=float)
            )
            stream_topdown_unity_dt_ms = (
                np.array(f['vehicle/stream_topdown_unity_dt_ms'][:], dtype=float)
                if 'vehicle/stream_topdown_unity_dt_ms' in f
                else np.array([], dtype=float)
            )
            stream_topdown_front_dt_ms = (
                np.array(f['vehicle/stream_topdown_front_dt_ms'][:], dtype=float)
                if 'vehicle/stream_topdown_front_dt_ms' in f
                else np.array([], dtype=float)
            )
            stream_front_frame_id_delta = (
                np.array(f['vehicle/stream_front_frame_id_delta'][:], dtype=float)
                if 'vehicle/stream_front_frame_id_delta' in f
                else np.array([], dtype=float)
            )
            stream_topdown_frame_id_delta = (
                np.array(f['vehicle/stream_topdown_frame_id_delta'][:], dtype=float)
                if 'vehicle/stream_topdown_frame_id_delta' in f
                else np.array([], dtype=float)
            )
            stream_topdown_front_frame_id_delta = (
                np.array(f['vehicle/stream_topdown_front_frame_id_delta'][:], dtype=float)
                if 'vehicle/stream_topdown_front_frame_id_delta' in f
                else np.array([], dtype=float)
            )
            stream_front_latest_age_ms = (
                np.array(f['vehicle/stream_front_latest_age_ms'][:], dtype=float)
                if 'vehicle/stream_front_latest_age_ms' in f
                else np.array([], dtype=float)
            )
            stream_front_queue_depth = (
                np.array(f['vehicle/stream_front_queue_depth'][:], dtype=float)
                if 'vehicle/stream_front_queue_depth' in f
                else np.array([], dtype=float)
            )
            stream_front_drop_count = (
                np.array(f['vehicle/stream_front_drop_count'][:], dtype=float)
                if 'vehicle/stream_front_drop_count' in f
                else np.array([], dtype=float)
            )
            stream_front_decode_in_flight = (
                np.array(f['vehicle/stream_front_decode_in_flight'][:], dtype=float)
                if 'vehicle/stream_front_decode_in_flight' in f
                else np.array([], dtype=float)
            )
            stream_topdown_latest_age_ms = (
                np.array(f['vehicle/stream_topdown_latest_age_ms'][:], dtype=float)
                if 'vehicle/stream_topdown_latest_age_ms' in f
                else np.array([], dtype=float)
            )
            stream_topdown_queue_depth = (
                np.array(f['vehicle/stream_topdown_queue_depth'][:], dtype=float)
                if 'vehicle/stream_topdown_queue_depth' in f
                else np.array([], dtype=float)
            )
            stream_topdown_drop_count = (
                np.array(f['vehicle/stream_topdown_drop_count'][:], dtype=float)
                if 'vehicle/stream_topdown_drop_count' in f
                else np.array([], dtype=float)
            )
            stream_topdown_decode_in_flight = (
                np.array(f['vehicle/stream_topdown_decode_in_flight'][:], dtype=float)
                if 'vehicle/stream_topdown_decode_in_flight' in f
                else np.array([], dtype=float)
            )
            stream_front_timestamp_minus_realtime_ms = (
                np.array(f['vehicle/stream_front_timestamp_minus_realtime_ms'][:], dtype=float)
                if 'vehicle/stream_front_timestamp_minus_realtime_ms' in f
                else np.array([], dtype=float)
            )
            stream_topdown_timestamp_minus_realtime_ms = (
                np.array(f['vehicle/stream_topdown_timestamp_minus_realtime_ms'][:], dtype=float)
                if 'vehicle/stream_topdown_timestamp_minus_realtime_ms' in f
                else np.array([], dtype=float)
            )

            # Projection input availability (for calibrated render feasibility).
            projection_fields = [
                'vehicle/camera_pos_x',
                'vehicle/camera_pos_y',
                'vehicle/camera_pos_z',
                'vehicle/camera_forward_x',
                'vehicle/camera_forward_y',
                'vehicle/camera_forward_z',
                'vehicle/camera_field_of_view',
                'vehicle/camera_horizontal_fov',
            ]
            projection_availability = {}
            for key in projection_fields:
                if key not in f:
                    projection_availability[key] = 0.0
                    continue
                arr = np.array(f[key][:], dtype=float)
                projection_availability[key] = float(np.mean(np.isfinite(arr))) if arr.size > 0 else 0.0

            # Explicit top-down camera calibration fields (currently not recorded).
            topdown_projection_fields = [
                'vehicle/topdown_camera_pos_x',
                'vehicle/topdown_camera_pos_y',
                'vehicle/topdown_camera_pos_z',
                'vehicle/topdown_camera_forward_x',
                'vehicle/topdown_camera_forward_y',
                'vehicle/topdown_camera_forward_z',
                'vehicle/topdown_camera_orthographic_size',
                'vehicle/topdown_camera_field_of_view',
            ]
            missing_topdown_projection_fields = [k for k in topdown_projection_fields if k not in f]

            # Compute top-down projection sanity metrics when calibration exists.
            topdown_projection_availability = {}
            for key in topdown_projection_fields:
                if key not in f:
                    topdown_projection_availability[key] = 0.0
                    continue
                arr = np.array(f[key][:], dtype=float)
                finite = np.isfinite(arr)
                if arr.size == 0:
                    topdown_projection_availability[key] = 0.0
                elif key.endswith("orthographic_size") or key.endswith("field_of_view"):
                    topdown_projection_availability[key] = float(np.mean(finite & (arr > 0.0)))
                else:
                    topdown_projection_availability[key] = float(np.mean(finite))

            topdown_forward_y = (
                np.array(f['vehicle/topdown_camera_forward_y'][:], dtype=float)
                if 'vehicle/topdown_camera_forward_y' in f
                else np.array([], dtype=float)
            )
            topdown_ortho = (
                np.array(f['vehicle/topdown_camera_orthographic_size'][:], dtype=float)
                if 'vehicle/topdown_camera_orthographic_size' in f
                else np.array([], dtype=float)
            )
            # Recorder currently stores top-down images at 640x480.
            topdown_image_height_px = 480.0
            topdown_meters_per_pixel = (
                (2.0 * topdown_ortho) / topdown_image_height_px
                if topdown_ortho.size > 0
                else np.array([], dtype=float)
            )
            topdown_calibrated_projection_ready = (
                len(missing_topdown_projection_fields) == 0
                and topdown_ortho.size > 0
                and float(np.mean(np.isfinite(topdown_ortho) & (topdown_ortho > 0.0))) > 0.95
            )

            # Step-1 trajectory geometry instrumentation (display vs planner separation).
            trajectory_geometry = {
                "available": False,
                "frames_analyzed": 0,
                "frames_with_non_monotonic_y": 0,
                "frames_with_x_saturation": 0,
                "sat_point_ratio": None,
                "y_monotonic_breaks_stats": {"count": 0, "mean": None, "p95_abs": None, "max_abs": None},
                "sat_points_per_frame_stats": {"count": 0, "mean": None, "p95_abs": None, "max_abs": None},
                "first_point_ref_dx_stats": {"count": 0, "mean": None, "p95_abs": None, "max_abs": None},
                "first_point_ref_dy_stats": {"count": 0, "mean": None, "p95_abs": None, "max_abs": None},
                "first_point_ref_dist_stats": {"count": 0, "mean": None, "p95_abs": None, "max_abs": None},
                "top_monotonic_break_frames": [],
                "top_saturation_frames": [],
            }
            if (
                "trajectory/trajectory_points" in f
                and "trajectory/reference_point_x" in f
                and "trajectory/reference_point_y" in f
            ):
                tp = f["trajectory/trajectory_points"]
                ref_x = np.array(f["trajectory/reference_point_x"][:], dtype=float)
                ref_y = np.array(f["trajectory/reference_point_y"][:], dtype=float)
                n = min(len(tp), ref_x.size, ref_y.size)
                breaks_arr = np.zeros(n, dtype=float)
                sat_arr = np.zeros(n, dtype=float)
                first_dx_arr = np.full(n, np.nan, dtype=float)
                first_dy_arr = np.full(n, np.nan, dtype=float)
                first_dist_arr = np.full(n, np.nan, dtype=float)
                total_points = 0
                total_sat_points = 0
                for i in range(n):
                    x, y = _parse_traj_xy(tp[i])
                    if x.size == 0:
                        continue
                    breaks = float(np.sum(np.diff(y) < -1e-6)) if y.size > 1 else 0.0
                    sat = float(np.sum(np.isclose(np.abs(x), 10.0, atol=1e-6)))
                    breaks_arr[i] = breaks
                    sat_arr[i] = sat
                    total_points += int(x.size)
                    total_sat_points += int(sat)
                    if np.isfinite(ref_x[i]) and np.isfinite(ref_y[i]):
                        dx = float(x[0] - ref_x[i])
                        dy = float(y[0] - ref_y[i])
                        first_dx_arr[i] = dx
                        first_dy_arr[i] = dy
                        first_dist_arr[i] = float(np.hypot(dx, dy))
                monotonic_rank = np.argsort(-breaks_arr)
                sat_rank = np.argsort(-sat_arr)
                top_breaks = []
                top_sat = []
                for idx in monotonic_rank[:5]:
                    if breaks_arr[idx] <= 0:
                        continue
                    top_breaks.append({"frame_idx": int(idx), "breaks": int(breaks_arr[idx])})
                for idx in sat_rank[:5]:
                    if sat_arr[idx] <= 0:
                        continue
                    top_sat.append({"frame_idx": int(idx), "sat_points": int(sat_arr[idx])})
                valid_first = np.isfinite(first_dx_arr) & np.isfinite(first_dy_arr) & np.isfinite(first_dist_arr)
                trajectory_geometry = {
                    "available": True,
                    "frames_analyzed": int(n),
                    "frames_with_non_monotonic_y": int(np.sum(breaks_arr > 0)),
                    "frames_with_x_saturation": int(np.sum(sat_arr > 0)),
                    "sat_point_ratio": (float(total_sat_points) / float(total_points)) if total_points > 0 else None,
                    "y_monotonic_breaks_stats": _stats_signed(breaks_arr),
                    "sat_points_per_frame_stats": _stats_signed(sat_arr),
                    "first_point_ref_dx_stats": _stats_signed(first_dx_arr[valid_first]),
                    "first_point_ref_dy_stats": _stats_signed(first_dy_arr[valid_first]),
                    "first_point_ref_dist_stats": _stats_signed(first_dist_arr[valid_first]),
                    "top_monotonic_break_frames": top_breaks,
                    "top_saturation_frames": top_sat,
                }

            # Source-isolation diagnostics: planner path vs serialized/augmented path.
            trajectory_source_isolation = {
                "available": False,
                "frames_analyzed": 0,
                "full_path_monotonic_breaks_stats": {"count": 0, "mean": None, "p95_abs": None, "max_abs": None},
                "without_first_point_monotonic_breaks_stats": {"count": 0, "mean": None, "p95_abs": None, "max_abs": None},
                "breaks_removed_by_dropping_first_point_ratio": None,
                "first_point_equals_reference_ratio": None,
                "first_point_y_greater_than_second_ratio": None,
                "x_saturation_ratio_without_first_point": None,
                "x_saturation_ratio_full_path": None,
                "reference_vs_lane_center_error_stats": {"count": 0, "mean": None, "p95_abs": None, "max_abs": None},
                "planner_vs_oracle_gap_metrics": {
                    "frames_with_oracle_overlap": 0,
                    "gap_at_lookahead_8m_stats": {"count": 0, "mean": None, "p95_abs": None, "max_abs": None},
                    "mean_lateral_gap_0_20m_stats": {"count": 0, "mean": None, "p95_abs": None, "max_abs": None},
                    "mean_lateral_gap_20_50m_stats": {"count": 0, "mean": None, "p95_abs": None, "max_abs": None},
                    "max_lateral_gap_0_h_stats": {"count": 0, "mean": None, "p95_abs": None, "max_abs": None},
                    "integrated_abs_gap_0_h_stats": {"count": 0, "mean": None, "p95_abs": None, "max_abs": None},
                },
                "generation_stage_diagnostics": {
                    "available_ratio": None,
                    "generated_by_fallback_ratio": None,
                    "x_clip_count_stats": {"count": 0, "mean": None, "p95_abs": None, "max_abs": None},
                    "first_segment_y0_gt_y1_pre_ratio": None,
                    "first_segment_y0_gt_y1_post_ratio": None,
                    "inversion_introduced_after_conversion_ratio": None,
                    "used_provided_distance0_ratio": None,
                    "used_provided_distance1_ratio": None,
                    "used_provided_distance2_ratio": None,
                    "post_minus_pre_y0_stats": {"count": 0, "mean": None, "p95_abs": None, "max_abs": None},
                    "post_minus_pre_y1_stats": {"count": 0, "mean": None, "p95_abs": None, "max_abs": None},
                    "post_minus_pre_y2_stats": {"count": 0, "mean": None, "p95_abs": None, "max_abs": None},
                    "preclip_x0_stats": {"count": 0, "mean": None, "p95_abs": None, "max_abs": None},
                    "postclip_x0_stats": {"count": 0, "mean": None, "p95_abs": None, "max_abs": None},
                },
            }
            if (
                "trajectory/trajectory_points" in f
                and "trajectory/reference_point_x" in f
                and "trajectory/reference_point_y" in f
            ):
                tp = f["trajectory/trajectory_points"]
                ref_x = np.array(f["trajectory/reference_point_x"][:], dtype=float)
                ref_y = np.array(f["trajectory/reference_point_y"][:], dtype=float)
                # Optional centerline intent check when GT lane center is available.
                gt_center = (
                    np.array(f["ground_truth/lane_center_x"][:], dtype=float)
                    if "ground_truth/lane_center_x" in f
                    else np.array([], dtype=float)
                )
                n = min(len(tp), ref_x.size, ref_y.size)
                breaks_full = np.zeros(n, dtype=float)
                breaks_wo_first = np.zeros(n, dtype=float)
                first_eq_ref = np.zeros(n, dtype=float)
                first_y_gt_second = np.zeros(n, dtype=float)
                sat_full_points = 0
                sat_wo_first_points = 0
                full_points = 0
                wo_first_points = 0
                ref_center_err = np.full(n, np.nan, dtype=float)
                for i in range(n):
                    x, y = _parse_traj_xy(tp[i])
                    if x.size == 0:
                        continue
                    full_points += int(x.size)
                    sat_full_points += int(np.sum(np.isclose(np.abs(x), 10.0, atol=1e-6)))
                    if y.size > 1:
                        breaks_full[i] = float(np.sum(np.diff(y) < -1e-6))
                    if np.isfinite(ref_x[i]) and np.isfinite(ref_y[i]):
                        first_eq_ref[i] = 1.0 if (abs(x[0] - ref_x[i]) < 1e-6 and abs(y[0] - ref_y[i]) < 1e-6) else 0.0
                    if y.size > 1 and (y[0] > y[1]):
                        first_y_gt_second[i] = 1.0
                    if x.size > 1:
                        x2 = x[1:]
                        y2 = y[1:]
                        wo_first_points += int(x2.size)
                        sat_wo_first_points += int(np.sum(np.isclose(np.abs(x2), 10.0, atol=1e-6)))
                        if y2.size > 1:
                            breaks_wo_first[i] = float(np.sum(np.diff(y2) < -1e-6))
                    if gt_center.size > i and np.isfinite(gt_center[i]) and np.isfinite(ref_x[i]):
                        ref_center_err[i] = float(ref_x[i] - gt_center[i])
                removed_ratio = None
                full_break_sum = float(np.sum(breaks_full))
                if full_break_sum > 0:
                    removed_ratio = float((np.sum(breaks_full) - np.sum(breaks_wo_first)) / full_break_sum)
                valid_ref_center = np.isfinite(ref_center_err)
                gen_diag = trajectory_source_isolation["generation_stage_diagnostics"]
                if "trajectory/diag_available" in f:
                    diag_available = np.array(f["trajectory/diag_available"][:], dtype=float)
                    m = min(int(n), int(diag_available.size))
                    if m > 0:
                        valid_diag = np.isfinite(diag_available[:m])
                        if np.any(valid_diag):
                            gen_diag["available_ratio"] = float(np.mean(diag_available[:m][valid_diag] > 0.5))
                if "trajectory/diag_generated_by_fallback" in f:
                    diag_fb = np.array(f["trajectory/diag_generated_by_fallback"][:], dtype=float)
                    m = min(int(n), int(diag_fb.size))
                    if m > 0:
                        valid_fb = np.isfinite(diag_fb[:m])
                        if np.any(valid_fb):
                            gen_diag["generated_by_fallback_ratio"] = float(np.mean(diag_fb[:m][valid_fb] > 0.5))
                if "trajectory/diag_x_clip_count" in f:
                    clip = np.array(f["trajectory/diag_x_clip_count"][:], dtype=float)
                    m = min(int(n), int(clip.size))
                    if m > 0:
                        valid_clip = np.isfinite(clip[:m])
                        gen_diag["x_clip_count_stats"] = _stats_signed(clip[:m][valid_clip])
                if "trajectory/diag_first_segment_y0_gt_y1_pre" in f:
                    pre_inv = np.array(f["trajectory/diag_first_segment_y0_gt_y1_pre"][:], dtype=float)
                    m = min(int(n), int(pre_inv.size))
                    if m > 0:
                        valid_pre = np.isfinite(pre_inv[:m])
                        if np.any(valid_pre):
                            gen_diag["first_segment_y0_gt_y1_pre_ratio"] = float(np.mean(pre_inv[:m][valid_pre] > 0.5))
                if "trajectory/diag_first_segment_y0_gt_y1_post" in f:
                    post_inv = np.array(f["trajectory/diag_first_segment_y0_gt_y1_post"][:], dtype=float)
                    m = min(int(n), int(post_inv.size))
                    if m > 0:
                        valid_post = np.isfinite(post_inv[:m])
                        if np.any(valid_post):
                            gen_diag["first_segment_y0_gt_y1_post_ratio"] = float(np.mean(post_inv[:m][valid_post] > 0.5))
                if "trajectory/diag_inversion_introduced_after_conversion" in f:
                    intro = np.array(f["trajectory/diag_inversion_introduced_after_conversion"][:], dtype=float)
                    m = min(int(n), int(intro.size))
                    if m > 0:
                        valid_intro = np.isfinite(intro[:m])
                        if np.any(valid_intro):
                            gen_diag["inversion_introduced_after_conversion_ratio"] = float(np.mean(intro[:m][valid_intro] > 0.5))
                if "trajectory/diag_used_provided_distance0" in f:
                    used0 = np.array(f["trajectory/diag_used_provided_distance0"][:], dtype=float)
                    m = min(int(n), int(used0.size))
                    if m > 0:
                        valid0 = np.isfinite(used0[:m])
                        if np.any(valid0):
                            gen_diag["used_provided_distance0_ratio"] = float(np.mean(used0[:m][valid0] > 0.5))
                if "trajectory/diag_used_provided_distance1" in f:
                    used1 = np.array(f["trajectory/diag_used_provided_distance1"][:], dtype=float)
                    m = min(int(n), int(used1.size))
                    if m > 0:
                        valid1 = np.isfinite(used1[:m])
                        if np.any(valid1):
                            gen_diag["used_provided_distance1_ratio"] = float(np.mean(used1[:m][valid1] > 0.5))
                if "trajectory/diag_used_provided_distance2" in f:
                    used2 = np.array(f["trajectory/diag_used_provided_distance2"][:], dtype=float)
                    m = min(int(n), int(used2.size))
                    if m > 0:
                        valid2 = np.isfinite(used2[:m])
                        if np.any(valid2):
                            gen_diag["used_provided_distance2_ratio"] = float(np.mean(used2[:m][valid2] > 0.5))
                if "trajectory/diag_post_minus_pre_y0" in f:
                    delta0 = np.array(f["trajectory/diag_post_minus_pre_y0"][:], dtype=float)
                    m = min(int(n), int(delta0.size))
                    if m > 0:
                        valid_delta0 = np.isfinite(delta0[:m])
                        gen_diag["post_minus_pre_y0_stats"] = _stats_signed(delta0[:m][valid_delta0])
                if "trajectory/diag_post_minus_pre_y1" in f:
                    delta1 = np.array(f["trajectory/diag_post_minus_pre_y1"][:], dtype=float)
                    m = min(int(n), int(delta1.size))
                    if m > 0:
                        valid_delta1 = np.isfinite(delta1[:m])
                        gen_diag["post_minus_pre_y1_stats"] = _stats_signed(delta1[:m][valid_delta1])
                if "trajectory/diag_post_minus_pre_y2" in f:
                    delta2 = np.array(f["trajectory/diag_post_minus_pre_y2"][:], dtype=float)
                    m = min(int(n), int(delta2.size))
                    if m > 0:
                        valid_delta2 = np.isfinite(delta2[:m])
                        gen_diag["post_minus_pre_y2_stats"] = _stats_signed(delta2[:m][valid_delta2])
                if "trajectory/diag_preclip_x0" in f:
                    pre_x0 = np.array(f["trajectory/diag_preclip_x0"][:], dtype=float)
                    m = min(int(n), int(pre_x0.size))
                    if m > 0:
                        valid_pre_x0 = np.isfinite(pre_x0[:m])
                        gen_diag["preclip_x0_stats"] = _stats_signed(pre_x0[:m][valid_pre_x0])
                if "trajectory/diag_postclip_x0" in f:
                    post_x0 = np.array(f["trajectory/diag_postclip_x0"][:], dtype=float)
                    m = min(int(n), int(post_x0.size))
                    if m > 0:
                        valid_post_x0 = np.isfinite(post_x0[:m])
                        gen_diag["postclip_x0_stats"] = _stats_signed(post_x0[:m][valid_post_x0])

                # Planner-vs-oracle gap metrics across near/far/full horizons.
                oracle_gap_diag = trajectory_source_isolation["planner_vs_oracle_gap_metrics"]
                if "trajectory/oracle_points" in f:
                    oracle_ds = f["trajectory/oracle_points"]
                    n_oracle = min(int(n), int(len(oracle_ds)))
                    gap_8m = []
                    gap_0_20 = []
                    gap_20_50 = []
                    gap_max_0_h = []
                    gap_int_0_h = []
                    for i in range(n_oracle):
                        px, py = _parse_traj_xy(tp[i])
                        ox, oy = _parse_oracle_xy(oracle_ds[i])
                        if px.size < 2 or ox.size < 2:
                            continue
                        # Compare planner-only path (drop prepended reference point when present).
                        if (
                            np.isfinite(ref_x[i]) and np.isfinite(ref_y[i]) and
                            abs(px[0] - ref_x[i]) < 1e-6 and abs(py[0] - ref_y[i]) < 1e-6 and
                            px.size > 1
                        ):
                            px = px[1:]
                            py = py[1:]
                        p_y, p_x = _prepare_curve(px, py)
                        o_y, o_x = _prepare_curve(ox, oy)
                        if p_y.size < 2 or o_y.size < 2:
                            continue
                        y_min = max(float(p_y[0]), float(o_y[0]), 0.0)
                        y_max = min(float(p_y[-1]), float(o_y[-1]), 50.0)
                        if y_max <= y_min + 1e-6:
                            continue
                        ys = np.arange(y_min, y_max + 1e-6, 1.0, dtype=float)
                        if ys.size < 2:
                            ys = np.array([y_min, y_max], dtype=float)
                        p_interp = np.interp(ys, p_y, p_x)
                        o_interp = np.interp(ys, o_y, o_x)
                        g = p_interp - o_interp
                        abs_g = np.abs(g)
                        if y_min <= 8.0 <= y_max:
                            g8 = float(np.interp(8.0, ys, g))
                            gap_8m.append(g8)
                        near = (ys >= 0.0) & (ys <= 20.0)
                        if np.any(near):
                            gap_0_20.append(float(np.mean(abs_g[near])))
                        far = (ys >= 20.0) & (ys <= 50.0)
                        if np.any(far):
                            gap_20_50.append(float(np.mean(abs_g[far])))
                        gap_max_0_h.append(float(np.max(abs_g)))
                        # Trapezoidal integral of absolute lateral gap across overlap horizon.
                        gap_int_0_h.append(float(np.trapz(abs_g, ys)))
                    oracle_gap_diag = {
                        "frames_with_oracle_overlap": int(len(gap_max_0_h)),
                        "gap_at_lookahead_8m_stats": _stats_signed(np.array(gap_8m, dtype=float)),
                        "mean_lateral_gap_0_20m_stats": _stats_signed(np.array(gap_0_20, dtype=float)),
                        "mean_lateral_gap_20_50m_stats": _stats_signed(np.array(gap_20_50, dtype=float)),
                        "max_lateral_gap_0_h_stats": _stats_signed(np.array(gap_max_0_h, dtype=float)),
                        "integrated_abs_gap_0_h_stats": _stats_signed(np.array(gap_int_0_h, dtype=float)),
                    }
                trajectory_source_isolation = {
                    "available": True,
                    "frames_analyzed": int(n),
                    "full_path_monotonic_breaks_stats": _stats_signed(breaks_full),
                    "without_first_point_monotonic_breaks_stats": _stats_signed(breaks_wo_first),
                    "breaks_removed_by_dropping_first_point_ratio": removed_ratio,
                    "first_point_equals_reference_ratio": float(np.mean(first_eq_ref)) if n > 0 else None,
                    "first_point_y_greater_than_second_ratio": float(np.mean(first_y_gt_second)) if n > 0 else None,
                    "x_saturation_ratio_without_first_point": (float(sat_wo_first_points) / float(wo_first_points)) if wo_first_points > 0 else None,
                    "x_saturation_ratio_full_path": (float(sat_full_points) / float(full_points)) if full_points > 0 else None,
                    "reference_vs_lane_center_error_stats": _stats_signed(ref_center_err[valid_ref_center]),
                    "planner_vs_oracle_gap_metrics": oracle_gap_diag,
                    "generation_stage_diagnostics": gen_diag,
                }

            return jsonify(
                {
                    "topdown_frames": int(topdown_ts.size),
                    "trajectory_frames": int(trajectory_ts.size),
                    "unity_frames": int(unity_ts.size),
                    "dt_topdown_traj": stats_traj,
                    "dt_topdown_unity": stats_unity,
                    "dt_topdown_traj_frames": _finite_stats(dt_topdown_traj_frames),
                    "dt_topdown_unity_frames": _finite_stats(dt_topdown_unity_frames),
                    "topdown_traj_index_delta": _stats_signed(idx_delta_traj),
                    "topdown_unity_index_delta": _stats_signed(idx_delta_unity),
                    "topdown_frame_id_step_stats": _finite_stats(topdown_frame_id_step),
                    "topdown_frame_id_monotonic_ratio": topdown_frame_id_monotonic_ratio,
                    "trajectory_period_ms_median": traj_period_ms,
                    "unity_period_ms_median": unity_period_ms,
                    "timestamp_domain_mismatch_suspected": bool(timestamp_domain_mismatch_suspected),
                    "stream_front_unity_dt_ms_stats": _stats_signed(stream_front_unity_dt_ms),
                    "stream_topdown_unity_dt_ms_stats": _stats_signed(stream_topdown_unity_dt_ms),
                    "stream_topdown_front_dt_ms_stats": _stats_signed(stream_topdown_front_dt_ms),
                    "stream_front_frame_id_delta_stats": _stats_signed(stream_front_frame_id_delta),
                    "stream_topdown_frame_id_delta_stats": _stats_signed(stream_topdown_frame_id_delta),
                    "stream_topdown_front_frame_id_delta_stats": _stats_signed(stream_topdown_front_frame_id_delta),
                    "stream_front_latest_age_ms_stats": _stats_signed(stream_front_latest_age_ms),
                    "stream_front_queue_depth_stats": _stats_signed(stream_front_queue_depth),
                    "stream_front_drop_count_stats": _stats_signed(stream_front_drop_count),
                    "stream_front_decode_in_flight_stats": _stats_signed(stream_front_decode_in_flight),
                    "stream_topdown_latest_age_ms_stats": _stats_signed(stream_topdown_latest_age_ms),
                    "stream_topdown_queue_depth_stats": _stats_signed(stream_topdown_queue_depth),
                    "stream_topdown_drop_count_stats": _stats_signed(stream_topdown_drop_count),
                    "stream_topdown_decode_in_flight_stats": _stats_signed(stream_topdown_decode_in_flight),
                    "stream_front_timestamp_minus_realtime_ms_stats": _stats_signed(
                        stream_front_timestamp_minus_realtime_ms
                    ),
                    "stream_topdown_timestamp_minus_realtime_ms_stats": _stats_signed(
                        stream_topdown_timestamp_minus_realtime_ms
                    ),
                    "sync_quality": _quality_from_p95(stats_traj["p95_ms"]),
                    "projection_inputs_available_ratio": projection_availability,
                    "topdown_projection_fields_missing": missing_topdown_projection_fields,
                    "topdown_projection_inputs_available_ratio": topdown_projection_availability,
                    "topdown_forward_y_stats": _finite_stats(topdown_forward_y),
                    "topdown_orthographic_size_stats": _finite_stats(topdown_ortho),
                    "topdown_meters_per_pixel_stats": _finite_stats(topdown_meters_per_pixel),
                    "topdown_calibrated_projection_ready": bool(topdown_calibrated_projection_ready),
                    "trajectory_geometry_diagnostics": trajectory_geometry,
                    "trajectory_source_isolation": trajectory_source_isolation,
                }
            )
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/api/recording/<path:filename>/run-perception-questions', methods=['POST'])
def run_perception_questions(filename):
    """Run tools/analyze/analyze_perception_questions.py for a recording and return parsed results."""
    from urllib.parse import unquote

    filename = unquote(filename)
    filepath = RECORDINGS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": f"Recording not found: {filename}"}), 404

    script_path = REPO_ROOT / "tools" / "analyze" / "analyze_perception_questions.py"
    if not script_path.exists():
        return jsonify({"error": f"Analyzer script not found: {script_path}"}), 500

    try:
        proc = subprocess.run(
            [sys.executable, str(script_path), str(filepath)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        combined_output = stdout if not stderr else f"{stdout}\n\n[stderr]\n{stderr}"

        question_results = {}
        for match in re.finditer(r"(?:✓|⚠️|❌)\s+Question\s+(\d+):\s+([A-Z]+)", stdout):
            question_results[f"q{match.group(1)}"] = match.group(2)
        # Q8 summary line format: "• Question 8 (diagnostic-only): Residual MAE ..."
        q8_match = re.search(r"Question 8 \(diagnostic-only\):\s*(.+)", stdout)
        if q8_match:
            question_results["q8"] = q8_match.group(1).strip()

        return jsonify(
            {
                "ok": proc.returncode == 0,
                "return_code": proc.returncode,
                "questions": question_results,
                "output": combined_output,
            }
        )
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Perception questions analysis timed out after 240s"}), 504
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (JS, CSS)."""
    return send_from_directory(Path(__file__).parent, filename)


if __name__ == '__main__':
    print(f"Starting debug visualizer server...")
    print(f"Recordings directory: {RECORDINGS_DIR}")
    print(f"Debug visualizations directory: {DEBUG_VIS_DIR}")
    print(f"Server running at http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=False)

