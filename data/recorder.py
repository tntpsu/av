"""
Data recorder for AV stack.
Records camera frames, vehicle state, control commands, and model outputs.
"""

import h5py
import numpy as np
import json
import time
import threading
import queue
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import cv2

logger = logging.getLogger(__name__)

UNITY_TIME_GAP_WARN_SECONDS = 0.2

from .formats.data_format import (
    CameraFrame, VehicleState, ControlCommand,
    PerceptionOutput, TrajectoryOutput, RecordingFrame
)


class DataRecorder:
    """Records AV stack data to HDF5 format."""
    
    def __init__(self, output_dir: str, recording_name: Optional[str] = None,
                 recording_type: Optional[str] = None, debug_timing: bool = False):
        """
        Initialize data recorder.
        
        Args:
            output_dir: Directory to save recordings
            recording_name: Name for this recording (default: timestamp)
            recording_type: Type of recording ("manual", "ground_truth_follower", 
                          "av_stack", "reprocessed", or None for auto-detect)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if recording_name is None:
            recording_name = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.recording_name = recording_name
        self.output_file = self.output_dir / f"{recording_name}.h5"
        
        # Auto-detect recording type from name if not provided
        if recording_type is None:
            name_lower = recording_name.lower()
            if "manual" in name_lower or "human" in name_lower:
                recording_type = "manual"
            elif "gt" in name_lower or "ground_truth" in name_lower or "follower" in name_lower or "gt_drive" in name_lower:
                recording_type = "gt_drive"  # Ground truth drive (for creating test recordings)
            elif "replay" in name_lower or "perception_replay" in name_lower:
                recording_type = "perception_replay"  # Perception replay test
            else:
                recording_type = "av_stack"  # Default (normal AV stack operation)
        
        self.recording_type = recording_type
        
        # Initialize HDF5 file
        self.h5_file = h5py.File(self.output_file, 'w')
        
        # Create datasets
        self._create_datasets()
        
        # Buffer for frames
        self.frame_buffer: List[RecordingFrame] = []
        self.frame_buffer_lock = threading.Lock()
        self.flush_queue: "queue.Queue[List[RecordingFrame]]" = queue.Queue()
        self.flush_stop_event = threading.Event()
        self.frame_count = 0
        self.last_unity_time: Optional[float] = None
        self.last_unity_frame_count: Optional[int] = None
        self.last_record_wall_time: Optional[float] = None
        self.last_record_frame_id: Optional[int] = None
        self.debug_timing = debug_timing
        self.flush_every = 30
        self.flush_queue_warn_threshold = 5
        self.flush_thread = threading.Thread(
            target=self._flush_worker,
            name="DataRecorderFlushWorker",
            daemon=True,
        )
        self.flush_thread.start()
        
        # Metadata
        self.metadata = {
            "recording_start_time": datetime.now().isoformat(),
            "recording_name": recording_name,
            "recording_type": recording_type
        }

    def _debug_print(self, message: str):
        if self.debug_timing:
            print(message)
    
    def _create_datasets(self):
        """Create HDF5 datasets for data storage."""
        # Use extensible datasets (maxshape allows resizing)
        max_shape = (None,)  # Unlimited length
        
        # Camera frames
        self.h5_file.create_dataset(
            "camera/images",
            shape=(0, 480, 640, 3),
            maxshape=(None, 480, 640, 3),
            dtype=np.uint8,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            chunks=(30, 480, 640, 3)
        )
        self.h5_file.create_dataset(
            "camera/timestamps",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float64
        )
        self.h5_file.create_dataset(
            "camera/frame_ids",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.int32
        )
        self.h5_file.create_dataset(
            "camera/topdown_images",
            shape=(0, 480, 640, 3),
            maxshape=(None, 480, 640, 3),
            dtype=np.uint8,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            chunks=(30, 480, 640, 3)
        )
        self.h5_file.create_dataset(
            "camera/topdown_timestamps",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float64
        )
        self.h5_file.create_dataset(
            "camera/topdown_frame_ids",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.int32
        )
        
        # Vehicle state
        self.h5_file.create_dataset(
            "vehicle/timestamps",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float64
        )
        self.h5_file.create_dataset(
            "vehicle/position",
            shape=(0, 3),
            maxshape=(None, 3),
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/rotation",
            shape=(0, 4),
            maxshape=(None, 4),
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/velocity",
            shape=(0, 3),
            maxshape=(None, 3),
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/angular_velocity",
            shape=(0, 3),
            maxshape=(None, 3),
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/speed",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/speed_limit",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/speed_limit_preview",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/speed_limit_preview_distance",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/speed_limit_preview_min_distance",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/speed_limit_preview_mid",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/speed_limit_preview_mid_distance",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/speed_limit_preview_mid_min_distance",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/speed_limit_preview_long",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/speed_limit_preview_long_distance",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/speed_limit_preview_long_min_distance",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/steering_angle",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/steering_angle_actual",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/steering_input",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/desired_steer_angle",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/motor_torque",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/brake_torque",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/unity_time",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float64
        )
        self.h5_file.create_dataset(
            "vehicle/unity_frame_count",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.int32
        )
        self.h5_file.create_dataset(
            "vehicle/unity_delta_time",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/unity_smooth_delta_time",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/unity_unscaled_delta_time",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/unity_time_scale",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        # NEW: Camera calibration - actual screen y pixel where 8m appears
        self.h5_file.create_dataset(
            "vehicle/camera_8m_screen_y",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        # NEW: Camera calibration - screen y for ground truth lookahead
        self.h5_file.create_dataset(
            "vehicle/camera_lookahead_screen_y",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        # NEW: Ground truth lookahead distance used for calibration
        self.h5_file.create_dataset(
            "vehicle/ground_truth_lookahead_distance",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/oracle_trajectory_world_xyz",
            shape=(0,),
            maxshape=max_shape,
            dtype=h5py.vlen_dtype(np.float32)
        )
        self.h5_file.create_dataset(
            "vehicle/oracle_trajectory_screen_xy",
            shape=(0,),
            maxshape=max_shape,
            dtype=h5py.vlen_dtype(np.float32)
        )
        self.h5_file.create_dataset(
            "vehicle/right_lane_fiducials_vehicle_xy",
            shape=(0,),
            maxshape=max_shape,
            dtype=h5py.vlen_dtype(np.float32)
        )
        self.h5_file.create_dataset(
            "vehicle/right_lane_fiducials_vehicle_true_xy",
            shape=(0,),
            maxshape=max_shape,
            dtype=h5py.vlen_dtype(np.float32)
        )
        self.h5_file.create_dataset(
            "vehicle/right_lane_fiducials_vehicle_monotonic_xy",
            shape=(0,),
            maxshape=max_shape,
            dtype=h5py.vlen_dtype(np.float32)
        )
        self.h5_file.create_dataset(
            "vehicle/right_lane_fiducials_world_xyz",
            shape=(0,),
            maxshape=max_shape,
            dtype=h5py.vlen_dtype(np.float32)
        )
        self.h5_file.create_dataset(
            "vehicle/right_lane_fiducials_screen_xy",
            shape=(0,),
            maxshape=max_shape,
            dtype=h5py.vlen_dtype(np.float32)
        )
        self.h5_file.create_dataset(
            "vehicle/right_lane_fiducials_point_count",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.int32
        )
        self.h5_file.create_dataset(
            "vehicle/right_lane_fiducials_horizon_meters",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/right_lane_fiducials_spacing_meters",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/right_lane_fiducials_enabled",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.int8
        )
        # NEW: Camera FOV values from Unity
        self.h5_file.create_dataset(
            "vehicle/camera_field_of_view",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/camera_horizontal_fov",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        # NEW: Camera position and forward direction from Unity
        self.h5_file.create_dataset(
            "vehicle/camera_pos_x",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/camera_pos_y",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/camera_pos_z",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/camera_forward_x",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/camera_forward_y",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/camera_forward_z",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        # Top-down camera calibration/projection (for top-down overlay diagnostics)
        self.h5_file.create_dataset(
            "vehicle/topdown_camera_pos_x",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/topdown_camera_pos_y",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/topdown_camera_pos_z",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/topdown_camera_forward_x",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/topdown_camera_forward_y",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/topdown_camera_forward_z",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/topdown_camera_orthographic_size",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/topdown_camera_field_of_view",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        # Stream consume-point lag diagnostics (instrumentation-only)
        self.h5_file.create_dataset(
            "vehicle/stream_front_unity_dt_ms",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/stream_topdown_unity_dt_ms",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/stream_topdown_front_dt_ms",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/stream_topdown_front_frame_id_delta",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/stream_front_frame_id_delta",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/stream_topdown_frame_id_delta",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/stream_front_latest_age_ms",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/stream_front_queue_depth",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/stream_front_drop_count",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/stream_front_decode_in_flight",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/stream_topdown_latest_age_ms",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/stream_topdown_queue_depth",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/stream_topdown_drop_count",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/stream_topdown_decode_in_flight",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/stream_front_last_realtime_s",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/stream_topdown_last_realtime_s",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/stream_front_timestamp_minus_realtime_ms",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/stream_topdown_timestamp_minus_realtime_ms",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/stream_front_source_timestamp",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float64
        )
        self.h5_file.create_dataset(
            "vehicle/stream_topdown_source_timestamp",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float64
        )
        self.h5_file.create_dataset(
            "vehicle/stream_front_timestamp_reused",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/stream_topdown_timestamp_reused",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/stream_front_timestamp_non_monotonic",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/stream_topdown_timestamp_non_monotonic",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/stream_front_negative_frame_delta",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/stream_topdown_negative_frame_delta",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/stream_front_frame_id_reused",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/stream_topdown_frame_id_reused",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/stream_front_clock_jump",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/stream_topdown_clock_jump",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        
        # NEW: Debug fields for diagnosing ground truth offset issues
        self.h5_file.create_dataset(
            "vehicle/road_center_at_car_x",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/road_center_at_car_y",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/road_center_at_car_z",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/road_center_at_lookahead_x",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/road_center_at_lookahead_y",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/road_center_at_lookahead_z",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/road_center_reference_t",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/road_frame_lateral_offset",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/road_heading_deg",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/car_heading_deg",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/heading_delta_deg",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/road_frame_lane_center_offset",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/road_frame_lane_center_error",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/vehicle_frame_lookahead_offset",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        # GT rotation input telemetry (exact Unity inputs used for target rotation).
        self.h5_file.create_dataset(
            "vehicle/gt_rotation_debug_valid",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.int8
        )
        self.h5_file.create_dataset(
            "vehicle/gt_rotation_used_road_frame",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.int8
        )
        self.h5_file.create_dataset(
            "vehicle/gt_rotation_rejected_road_frame_hop",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.int8
        )
        self.h5_file.create_dataset(
            "vehicle/gt_rotation_reference_heading_deg",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/gt_rotation_road_frame_heading_deg",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/gt_rotation_input_heading_deg",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/gt_rotation_road_vs_ref_delta_deg",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "vehicle/gt_rotation_applied_delta_deg",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        # Control commands
        self.h5_file.create_dataset(
            "control/timestamps",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float64
        )
        self.h5_file.create_dataset(
            "control/steering",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/throttle",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/brake",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/accel_feedforward",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/brake_feedforward",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/longitudinal_accel_capped",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.int8
        )
        self.h5_file.create_dataset(
            "control/longitudinal_jerk_capped",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.int8
        )
        self.h5_file.create_dataset(
            "control/emergency_stop",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.int8
        )
        self.h5_file.create_dataset(
            "control/target_speed_raw",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/target_speed_post_limits",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/target_speed_planned",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/target_speed_final",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/target_speed_slew_active",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.int8
        )
        self.h5_file.create_dataset(
            "control/target_speed_ramp_active",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.int8
        )
        self.h5_file.create_dataset(
            "control/launch_throttle_cap",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/launch_throttle_cap_active",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.int8
        )
        self.h5_file.create_dataset(
            "control/steering_pre_rate_limit",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/steering_post_rate_limit",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/steering_post_jerk_limit",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/steering_post_sign_flip",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/steering_post_hard_clip",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/steering_post_smoothing",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/steering_rate_limited_active",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.int8
        )
        self.h5_file.create_dataset(
            "control/steering_jerk_limited_active",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.int8
        )
        self.h5_file.create_dataset(
            "control/steering_hard_clip_active",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.int8
        )
        self.h5_file.create_dataset(
            "control/steering_smoothing_active",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.int8
        )
        self.h5_file.create_dataset(
            "control/steering_rate_limited_delta",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/steering_jerk_limited_delta",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/steering_hard_clip_delta",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/steering_smoothing_delta",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        
        # Perception outputs (optional)
        self.h5_file.create_dataset(
            "perception/timestamps",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float64
        )
        self.h5_file.create_group("perception/lane_lines")
        # Lane line coefficients (polynomial coefficients for detected lanes)
        # Stored as variable-length arrays: each frame can have different number of lanes
        # We'll use a compound dtype to store arrays of coefficients
        self.h5_file.create_dataset(
            "perception/lane_line_coefficients",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.vlen_dtype(np.float32)
        )
        
        # Trajectory outputs (optional)
        self.h5_file.create_dataset(
            "trajectory/timestamps",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float64
        )
        self.h5_file.create_group("trajectory/paths")
        # NEW: Store trajectory points (variable-length array since each frame can have different number of points)
        # Each element is a 2D array [N, 3] where N varies per frame
        self.h5_file.create_dataset(
            "trajectory/trajectory_points",
            shape=(0,),
            maxshape=max_shape,
            dtype=h5py.vlen_dtype(np.float32)  # Variable-length array of float32 arrays
        )
        self.h5_file.create_dataset(
            "trajectory/oracle_points",
            shape=(0,),
            maxshape=max_shape,
            dtype=h5py.vlen_dtype(np.float32)
        )
        self.h5_file.create_dataset(
            "trajectory/oracle_point_count",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.int32
        )
        self.h5_file.create_dataset(
            "trajectory/oracle_horizon_meters",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "trajectory/oracle_point_spacing_meters",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "trajectory/oracle_samples_enabled",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.int8
        )
        # Reference point data (for drift analysis)
        self.h5_file.create_dataset(
            "trajectory/reference_point_x",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "trajectory/reference_point_y",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "trajectory/reference_point_heading",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "trajectory/reference_point_velocity",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        # Reference point before smoothing (raw values)
        self.h5_file.create_dataset(
            "trajectory/reference_point_raw_x",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "trajectory/reference_point_raw_y",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "trajectory/reference_point_raw_heading",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        # NEW: Debug fields for tracking reference point calculation method
        self.h5_file.create_dataset(
            "trajectory/reference_point_method",
            shape=(0,),
            maxshape=max_shape,
            dtype=h5py.string_dtype(encoding='utf-8', length=20)  # Store method name as string
        )
        self.h5_file.create_dataset(
            "trajectory/perception_center_x",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset("trajectory/diag_available", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_generated_by_fallback", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_points_generated", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_x_clip_count", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_pre_y0", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_pre_y1", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_pre_y2", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_post_y0", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_post_y1", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_post_y2", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_used_provided_distance0", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_used_provided_distance1", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_used_provided_distance2", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_post_minus_pre_y0", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_post_minus_pre_y1", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_post_minus_pre_y2", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_preclip_x0", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_preclip_x1", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_preclip_x2", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_preclip_x_abs_max", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_preclip_x_abs_p95", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_preclip_mean_12_20m_lane_source_x", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_preclip_mean_12_20m_distance_scale_delta_x", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_preclip_mean_12_20m_camera_offset_delta_x", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_preclip_abs_mean_12_20m_lane_source_x", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_preclip_abs_mean_12_20m_distance_scale_delta_x", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_preclip_abs_mean_12_20m_camera_offset_delta_x", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_heading_zero_gate_active", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_small_heading_gate_active", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_multi_lookahead_active", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_smoothing_jump_reject", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_ref_x_rate_limit_active", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_raw_ref_x", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_smoothed_ref_x", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_ref_x_suppression_abs", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_raw_ref_heading", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_smoothed_ref_heading", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_heading_suppression_abs", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_smoothing_alpha", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_smoothing_alpha_x", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_multi_lookahead_heading_base", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_multi_lookahead_heading_far", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_multi_lookahead_heading_blended", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_multi_lookahead_blend_alpha", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_dynamic_effective_horizon_m", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_dynamic_effective_horizon_base_m", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_dynamic_effective_horizon_min_m", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_dynamic_effective_horizon_max_m", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_dynamic_effective_horizon_speed_scale", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_dynamic_effective_horizon_curvature_scale", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_dynamic_effective_horizon_confidence_scale", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_dynamic_effective_horizon_final_scale", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_dynamic_effective_horizon_speed_mps", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_dynamic_effective_horizon_curvature_abs", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_dynamic_effective_horizon_confidence_used", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_dynamic_effective_horizon_limiter_code", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_dynamic_effective_horizon_applied", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_preclip_abs_mean_0_8m", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_preclip_abs_mean_8_12m", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_preclip_abs_mean_12_20m", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_postclip_x0", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_postclip_x1", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_postclip_x2", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_postclip_abs_mean_0_8m", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_postclip_abs_mean_8_12m", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_postclip_abs_mean_12_20m", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_postclip_near_clip_frac_12_20m", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_first_segment_y0_gt_y1_pre", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_first_segment_y0_gt_y1_post", shape=(0,), maxshape=max_shape, dtype=np.float32)
        self.h5_file.create_dataset("trajectory/diag_inversion_introduced_after_conversion", shape=(0,), maxshape=max_shape, dtype=np.float32)
        
        # Control command details
        self.h5_file.create_dataset(
            "control/steering_before_limits",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/throttle_before_limits",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/brake_before_limits",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/feedforward_steering",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/feedback_steering",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        
        # PID internal state
        self.h5_file.create_dataset(
            "control/pid_integral",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/pid_derivative",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/pid_error",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/total_error_scaled",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/straight_sign_flip_override_active",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.int8
        )
        self.h5_file.create_dataset(
            "control/straight_sign_flip_triggered",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.int8
        )
        self.h5_file.create_dataset(
            "control/straight_sign_flip_trigger_error",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/straight_sign_flip_frames_remaining",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.int16
        )
        
        # Error breakdown
        self.h5_file.create_dataset(
            "control/lateral_error",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/heading_error",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/total_error",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        # Exact steering calculation breakdown (for ground truth verification)
        self.h5_file.create_dataset(
            "control/calculated_steering_angle_deg",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/raw_steering",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/lateral_correction",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/path_curvature_input",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/is_straight",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.int8
        )
        self.h5_file.create_dataset(
            "control/straight_oscillation_rate",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/tuned_deadband",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "control/tuned_error_smoothing_alpha",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        # NEW: Control stale data tracking
        self.h5_file.create_dataset(
            "control/using_stale_perception",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.bool_
        )
        self.h5_file.create_dataset(
            "control/stale_perception_reason",
            shape=(0,),
            maxshape=max_shape,
            dtype=h5py.string_dtype(encoding='utf-8', length=50)
        )
        
        # Perception details
        self.h5_file.create_dataset(
            "perception/confidence",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "perception/detection_method",
            shape=(0,),
            maxshape=max_shape,
            dtype=h5py.string_dtype(encoding='utf-8')
        )
        self.h5_file.create_dataset(
            "perception/num_lanes_detected",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.int32
        )
        # Lane positions at lookahead distance (for trajectory centering verification)
        self.h5_file.create_dataset(
            "perception/left_lane_line_x",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "perception/right_lane_line_x",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        # NEW: Stale data tracking fields
        self.h5_file.create_dataset(
            "perception/using_stale_data",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.bool_
        )
        self.h5_file.create_dataset(
            "perception/stale_reason",
            shape=(0,),
            maxshape=max_shape,
            dtype=h5py.string_dtype(encoding='utf-8', length=50)
        )
        self.h5_file.create_dataset(
            "perception/reject_reason",
            shape=(0,),
            maxshape=max_shape,
            dtype=h5py.string_dtype(encoding='utf-8', length=100)
        )
        self.h5_file.create_dataset(
            "perception/left_jump_magnitude",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "perception/right_jump_magnitude",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "perception/jump_threshold",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        # NEW: Perception health monitoring
        self.h5_file.create_dataset(
            "perception/consecutive_bad_detection_frames",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.int32
        )
        self.h5_file.create_dataset(
            "perception/perception_health_score",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "perception/perception_health_status",
            shape=(0,),
            maxshape=max_shape,
            dtype=h5py.string_dtype(encoding='utf-8', length=20)
        )
        self.h5_file.create_dataset(
            "perception/perception_bad_events",
            shape=(0,),
            maxshape=max_shape,
            dtype=h5py.string_dtype(encoding='utf-8', length=100)
        )
        self.h5_file.create_dataset(
            "perception/perception_bad_events_recent",
            shape=(0,),
            maxshape=max_shape,
            dtype=h5py.string_dtype(encoding='utf-8', length=100)
        )
        self.h5_file.create_dataset(
            "perception/perception_clamp_events",
            shape=(0,),
            maxshape=max_shape,
            dtype=h5py.string_dtype(encoding='utf-8', length=100)
        )
        self.h5_file.create_dataset(
            "perception/perception_timestamp_frozen",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.bool_
        )
        # NEW: Diagnostic fields for perception instability
        self.h5_file.create_dataset(
            "perception/actual_detected_left_lane_x",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "perception/actual_detected_right_lane_x",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "perception/instability_width_change",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "perception/instability_center_shift",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        # NEW: Points used for polynomial fitting (for debug visualization)
        # Store as JSON strings since HDF5 doesn't handle variable-length arrays well
        self.h5_file.create_dataset(
            "perception/fit_points_left",
            shape=(0,),
            maxshape=max_shape,
            dtype=h5py.string_dtype(encoding='utf-8', length=10000)  # Max 10KB per frame
        )
        self.h5_file.create_dataset(
            "perception/fit_points_right",
            shape=(0,),
            maxshape=max_shape,
            dtype=h5py.string_dtype(encoding='utf-8', length=10000)
        )
        self.h5_file.create_dataset(
            "perception/segmentation_mask_png",
            shape=(0,),
            maxshape=max_shape,
            dtype=h5py.vlen_dtype(np.uint8)
        )
        
        # Ground truth lane positions (from Unity)
        self.h5_file.create_group("ground_truth")
        self.h5_file.create_dataset(
            "ground_truth/left_lane_line_x",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "ground_truth/right_lane_line_x",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "ground_truth/lane_center_x",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        # Ground truth path information (for exact steering calculation verification)
        self.h5_file.create_dataset(
            "ground_truth/path_curvature",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "ground_truth/desired_heading",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        
        # Unity feedback data (NEW: eliminates need to check Unity console)
        self.h5_file.create_group("unity_feedback")
        self.h5_file.create_dataset(
            "unity_feedback/timestamps",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float64
        )
        self.h5_file.create_dataset(
            "unity_feedback/ground_truth_mode_active",
            shape=(0,),
            maxshape=max_shape,
            dtype=bool
        )
        self.h5_file.create_dataset(
            "unity_feedback/control_command_received",
            shape=(0,),
            maxshape=max_shape,
            dtype=bool
        )
        self.h5_file.create_dataset(
            "unity_feedback/actual_steering_applied",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "unity_feedback/actual_throttle_applied",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "unity_feedback/actual_brake_applied",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
        )
        self.h5_file.create_dataset(
            "unity_feedback/ground_truth_data_available",
            shape=(0,),
            maxshape=max_shape,
            dtype=bool
        )
        self.h5_file.create_dataset(
            "unity_feedback/ground_truth_reporter_enabled",
            shape=(0,),
            maxshape=max_shape,
            dtype=bool
        )
        self.h5_file.create_dataset(
            "unity_feedback/path_curvature_calculated",
            shape=(0,),
            maxshape=max_shape,
            dtype=bool
        )
        self.h5_file.create_dataset(
            "unity_feedback/car_controller_mode",
            shape=(0,),
            maxshape=max_shape,
            dtype=h5py.string_dtype(encoding='utf-8')
        )
        self.h5_file.create_dataset(
            "unity_feedback/av_control_enabled",
            shape=(0,),
            maxshape=max_shape,
            dtype=bool
        )
        self.h5_file.create_dataset(
            "unity_feedback/unity_frame_count",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.int32
        )
    
    def record_frame(self, frame: RecordingFrame):
        """
        Record a complete frame of data.
        
        Args:
            frame: RecordingFrame containing all data
        """
        now = time.time()
        if self.last_record_wall_time is not None:
            gap = now - self.last_record_wall_time
            if gap > UNITY_TIME_GAP_WARN_SECONDS:
                unity_time = None
                unity_frame = None
                if frame.vehicle_state is not None:
                    unity_time = getattr(frame.vehicle_state, "unity_time", None)
                    unity_frame = getattr(frame.vehicle_state, "unity_frame_count", None)
                logger.warning(
                    "[RECORDER_ARRIVAL_GAP] gap=%.3fs frame_id=%s prev_frame_id=%s "
                    "unity_frame=%s unity_time=%s types=camera:%s vehicle:%s control:%s",
                    gap,
                    frame.frame_id,
                    self.last_record_frame_id,
                    unity_frame if unity_frame is not None else "n/a",
                    f"{unity_time:.3f}" if unity_time is not None else "n/a",
                    frame.camera_frame is not None,
                    frame.vehicle_state is not None,
                    frame.control_command is not None,
                )
                self._debug_print(
                    "[RECORDER_ARRIVAL_GAP] "
                    f"gap={gap:.3f}s frame_id={frame.frame_id} "
                    f"unity_frame={unity_frame if unity_frame is not None else 'n/a'}"
                )
        self.last_record_wall_time = now
        self.last_record_frame_id = frame.frame_id

        with self.frame_buffer_lock:
            self.frame_buffer.append(frame)
            self.frame_count += 1

            # Flush buffer periodically (async)
            if len(self.frame_buffer) >= self.flush_every:
                frames = self.frame_buffer
                self.frame_buffer = []
                self._debug_print(f"[RECORDER_FLUSH_START] buffer={len(frames)}")
                self.flush_queue.put(frames)
    
    def record_camera_frame(self, camera_frame: CameraFrame):
        """Record camera frame."""
        frame = RecordingFrame(
            timestamp=camera_frame.timestamp,
            frame_id=camera_frame.frame_id,
            camera_frame=camera_frame
        )
        self.record_frame(frame)
    
    def record_vehicle_state(self, vehicle_state: VehicleState):
        """Record vehicle state."""
        frame = RecordingFrame(
            timestamp=vehicle_state.timestamp,
            frame_id=self.frame_count,
            vehicle_state=vehicle_state
        )
        self.record_frame(frame)
    
    def record_control_command(self, control_command: ControlCommand):
        """Record control command."""
        frame = RecordingFrame(
            timestamp=control_command.timestamp,
            frame_id=self.frame_count,
            control_command=control_command
        )
        self.record_frame(frame)
    
    def flush(self):
        """Flush buffered frames to disk."""
        with self.frame_buffer_lock:
            if not self.frame_buffer:
                return
            frames = self.frame_buffer
            self.frame_buffer = []
        self._debug_print(f"[RECORDER_FLUSH_START] buffer={len(frames)}")
        self.flush_queue.put(frames)
    
    def _flush_worker(self):
        while not self.flush_stop_event.is_set() or not self.flush_queue.empty():
            try:
                frames = self.flush_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if self.flush_queue.qsize() > self.flush_queue_warn_threshold:
                logger.warning(
                    "[RECORDER_QUEUE_BACKLOG] size=%d threshold=%d",
                    self.flush_queue.qsize(),
                    self.flush_queue_warn_threshold,
                )
            try:
                self._flush_frames(frames)
            finally:
                self.flush_queue.task_done()

    def _flush_frames(self, frames: List[RecordingFrame]):
        if not frames:
            return

        flush_start = time.time()

        # Group frames by type and write to HDF5
        camera_frames = []
        topdown_frames = []
        vehicle_states = []
        control_commands = []
        perception_outputs = []
        trajectory_outputs = []
        unity_feedbacks = []

        for frame in frames:
            if frame.camera_frame:
                camera_frames.append(frame)
            if frame.camera_topdown_frame:
                topdown_frames.append(frame)
            if frame.vehicle_state:
                vehicle_states.append(frame)
            if frame.control_command:
                control_commands.append(frame)
            if frame.perception_output:
                perception_outputs.append(frame)
            if frame.trajectory_output:
                trajectory_outputs.append(frame)
            if frame.unity_feedback:
                unity_feedbacks.append(frame)

        # Debug: Log what we're collecting
        sample_frame = frames[0]
        logger.info(f"[RECORDER] Flushing {len(frames)} frames: "
                   f"camera={len(camera_frames)}, topdown={len(topdown_frames)}, "
                   f"vehicle={len(vehicle_states)}, "
                   f"control={len(control_commands)}, perception={len(perception_outputs)}, "
                   f"trajectory={len(trajectory_outputs)}, unity={len(unity_feedbacks)}")
        if len(vehicle_states) == 0:
            logger.error(f"[RECORDER] ❌ CRITICAL: No vehicle states! Sample frame has vehicle_state={sample_frame.vehicle_state is not None}")
            if sample_frame.vehicle_state is None:
                logger.error(f"[RECORDER] ❌ Sample frame vehicle_state is None - frames are being collected without vehicle data!")
                logger.error(f"[RECORDER] This will cause ground truth and camera data to not be recorded!")
        if len(perception_outputs) == 0:
            logger.warning(f"[RECORDER] ⚠️  No perception outputs! Sample frame has perception_output={sample_frame.perception_output is not None}")
            if sample_frame.perception_output is None:
                logger.warning(f"[RECORDER] ⚠️  Sample frame perception_output is None - frames are being collected without perception data!")
        if len(control_commands) == 0:
            logger.warning(f"[RECORDER] ⚠️  No control commands! Sample frame has control_command={sample_frame.control_command is not None}")
            if sample_frame.control_command is None:
                logger.warning(f"[RECORDER] ⚠️  Sample frame control_command is None - frames are being collected without control data!")
        if len(trajectory_outputs) == 0:
            logger.warning(f"[RECORDER] ⚠️  No trajectory outputs! Sample frame has trajectory_output={sample_frame.trajectory_output is not None}")
            if sample_frame.trajectory_output is None:
                logger.warning(f"[RECORDER] ⚠️  Sample frame trajectory_output is None - frames are being collected without trajectory data!")

        # Write camera frames
        if camera_frames:
            try:
                section_start = time.time()
                self._write_camera_frames(camera_frames)
                self._debug_print(
                    f"[RECORDER_WRITE_CAMERA] duration={time.time() - section_start:.3f}s "
                    f"count={len(camera_frames)}"
                )
            except Exception as e:
                logger.error(f"Error writing camera frames: {e}", exc_info=True)
        if topdown_frames:
            try:
                section_start = time.time()
                self._write_topdown_camera_frames(topdown_frames)
                self._debug_print(
                    f"[RECORDER_WRITE_TOPDOWN] duration={time.time() - section_start:.3f}s "
                    f"count={len(topdown_frames)}"
                )
            except Exception as e:
                logger.error(f"Error writing top-down camera frames: {e}", exc_info=True)

        # Write vehicle states
        if vehicle_states:
            try:
                section_start = time.time()
                self._write_vehicle_states(vehicle_states)
                self._debug_print(
                    f"[RECORDER_WRITE_VEHICLE] duration={time.time() - section_start:.3f}s "
                    f"count={len(vehicle_states)}"
                )
            except Exception as e:
                logger.error(f"Error writing vehicle states: {e}", exc_info=True)
        else:
            logger.error(f"[RECORDER] ❌ No vehicle states to write! "
                       f"Total frames: {len(frames)}, "
                       f"Sample frame vehicle_state is None: {sample_frame.vehicle_state is None}")
            if sample_frame.vehicle_state is None:
                logger.error(f"[RECORDER] ❌ CRITICAL: vehicle_state is None in RecordingFrame! "
                           f"This means VehicleState creation failed or was skipped.")

        # Write control commands
        if control_commands:
            try:
                section_start = time.time()
                self._write_control_commands(control_commands)
                self._debug_print(
                    f"[RECORDER_WRITE_CONTROL] duration={time.time() - section_start:.3f}s "
                    f"count={len(control_commands)}"
                )
            except Exception as e:
                logger.error(f"Error writing control commands: {e}", exc_info=True)

        # Write perception outputs
        if perception_outputs:
            try:
                section_start = time.time()
                self._write_perception_outputs(perception_outputs)
                self._debug_print(
                    f"[RECORDER_WRITE_PERCEPTION] duration={time.time() - section_start:.3f}s "
                    f"count={len(perception_outputs)}"
                )
            except Exception as e:
                logger.error(f"Error writing perception outputs: {e}", exc_info=True)

        # Write trajectory outputs
        if trajectory_outputs:
            try:
                section_start = time.time()
                self._write_trajectory_outputs(trajectory_outputs)
                self._debug_print(
                    f"[RECORDER_WRITE_TRAJECTORY] duration={time.time() - section_start:.3f}s "
                    f"count={len(trajectory_outputs)}"
                )
            except Exception as e:
                logger.error(f"Error writing trajectory outputs: {e}", exc_info=True)

        # Write Unity feedback
        if unity_feedbacks:
            try:
                section_start = time.time()
                self._write_unity_feedback(unity_feedbacks)
                self._debug_print(
                    f"[RECORDER_WRITE_UNITY] duration={time.time() - section_start:.3f}s "
                    f"count={len(unity_feedbacks)}"
                )
            except Exception as e:
                logger.error(f"Error writing Unity feedback: {e}", exc_info=True)

        flush_start_time = time.time()
        self.h5_file.flush()
        self._debug_print(f"[RECORDER_H5_FLUSH] duration={time.time() - flush_start_time:.3f}s")

        flush_duration = time.time() - flush_start
        if flush_duration > UNITY_TIME_GAP_WARN_SECONDS:
            logger.warning("[RECORDER_FLUSH_SLOW] duration=%.3fs", flush_duration)
            self._debug_print(f"[RECORDER_FLUSH_SLOW] duration={flush_duration:.3f}s")
        else:
            self._debug_print(f"[RECORDER_FLUSH_DONE] duration={flush_duration:.3f}s")
    
    def _write_camera_frames(self, frames: List[RecordingFrame]):
        """Write camera frames to HDF5."""
        section_start = time.time()
        images = []
        timestamps = []
        frame_ids = []
        resized_count = 0
        
        for frame in frames:
            img = frame.camera_frame.image
            # Resize if needed
            if img.shape[:2] != (480, 640):
                resize_start = time.time()
                img = cv2.resize(img, (640, 480))
                resized_count += 1
                resize_duration = time.time() - resize_start
                if resize_duration > UNITY_TIME_GAP_WARN_SECONDS:
                    self._debug_print("[RECORDER_CAMERA_RESIZE_SLOW] duration=%.3fs" % resize_duration)
            images.append(img)
            timestamps.append(frame.camera_frame.timestamp)
            frame_ids.append(frame.camera_frame.frame_id)

        self._debug_print(
            f"[RECORDER_CAMERA_PREP] duration={time.time() - section_start:.3f}s "
            f"resized={resized_count}"
        )

        if len(timestamps) > 1:
            diffs = np.diff(np.array(timestamps, dtype=np.float64))
            gap_indices = np.where(diffs > UNITY_TIME_GAP_WARN_SECONDS)[0]
            if gap_indices.size > 0:
                first_idx = int(gap_indices[0])
                logger.warning(
                    "[CAMERA_TIMESTAMP_GAP_RECORDER] gaps=%d first_gap=%.3fs frame_id=%s->%s",
                    int(gap_indices.size),
                    float(diffs[first_idx]),
                    frame_ids[first_idx],
                    frame_ids[first_idx + 1],
                )
        
        if images:
            convert_start = time.time()
            images = np.array(images)
            timestamps = np.array(timestamps)
            frame_ids = np.array(frame_ids)
            self._debug_print(f"[RECORDER_CAMERA_CONVERT] duration={time.time() - convert_start:.3f}s")
            
            # Resize datasets if needed
            resize_start = time.time()
            current_size = self.h5_file["camera/images"].shape[0]
            new_size = current_size + len(images)
            
            self.h5_file["camera/images"].resize((new_size, 480, 640, 3))
            self.h5_file["camera/timestamps"].resize((new_size,))
            self.h5_file["camera/frame_ids"].resize((new_size,))
            self._debug_print(f"[RECORDER_CAMERA_RESIZE_DS] duration={time.time() - resize_start:.3f}s")
            
            # Write data
            write_start = time.time()
            total_bytes = images.nbytes + timestamps.nbytes + frame_ids.nbytes
            self._debug_print(
                f"[RECORDER_CAMERA_WRITE_SIZE] bytes={total_bytes} current_size={current_size} "
                f"new_size={new_size}"
            )
            self.h5_file["camera/images"][current_size:] = images
            self.h5_file["camera/timestamps"][current_size:] = timestamps
            self.h5_file["camera/frame_ids"][current_size:] = frame_ids
            self._debug_print(f"[RECORDER_CAMERA_WRITE] duration={time.time() - write_start:.3f}s")

    def _write_topdown_camera_frames(self, frames: List[RecordingFrame]):
        """Write top-down camera frames to HDF5."""
        section_start = time.time()
        images = []
        timestamps = []
        frame_ids = []
        resized_count = 0

        for frame in frames:
            img = frame.camera_topdown_frame.image
            if img.shape[:2] != (480, 640):
                resize_start = time.time()
                img = cv2.resize(img, (640, 480))
                resized_count += 1
                resize_duration = time.time() - resize_start
                if resize_duration > UNITY_TIME_GAP_WARN_SECONDS:
                    self._debug_print("[RECORDER_TOPDOWN_RESIZE_SLOW] duration=%.3fs" % resize_duration)
            images.append(img)
            timestamps.append(frame.camera_topdown_frame.timestamp)
            frame_ids.append(frame.camera_topdown_frame.frame_id)

        self._debug_print(
            f"[RECORDER_TOPDOWN_PREP] duration={time.time() - section_start:.3f}s "
            f"resized={resized_count}"
        )

        if images:
            convert_start = time.time()
            images = np.array(images)
            timestamps = np.array(timestamps)
            frame_ids = np.array(frame_ids)
            self._debug_print(f"[RECORDER_TOPDOWN_CONVERT] duration={time.time() - convert_start:.3f}s")

            resize_start = time.time()
            current_size = self.h5_file["camera/topdown_images"].shape[0]
            new_size = current_size + len(images)

            self.h5_file["camera/topdown_images"].resize((new_size, 480, 640, 3))
            self.h5_file["camera/topdown_timestamps"].resize((new_size,))
            self.h5_file["camera/topdown_frame_ids"].resize((new_size,))
            self._debug_print(f"[RECORDER_TOPDOWN_RESIZE_DS] duration={time.time() - resize_start:.3f}s")

            write_start = time.time()
            total_bytes = images.nbytes + timestamps.nbytes + frame_ids.nbytes
            self._debug_print(
                f"[RECORDER_TOPDOWN_WRITE_SIZE] bytes={total_bytes} current_size={current_size} "
                f"new_size={new_size}"
            )
            self.h5_file["camera/topdown_images"][current_size:] = images
            self.h5_file["camera/topdown_timestamps"][current_size:] = timestamps
            self.h5_file["camera/topdown_frame_ids"][current_size:] = frame_ids
            self._debug_print(f"[RECORDER_TOPDOWN_WRITE] duration={time.time() - write_start:.3f}s")
    
    def _write_vehicle_states(self, frames: List[RecordingFrame]):
        """Write vehicle states to HDF5."""
        positions = []
        rotations = []
        velocities = []
        angular_velocities = []
        speeds = []
        speed_limits = []
        speed_limit_previews = []
        speed_limit_preview_distances = []
        speed_limit_preview_min_distances = []
        speed_limit_preview_mid = []
        speed_limit_preview_mid_distances = []
        speed_limit_preview_mid_min_distances = []
        speed_limit_preview_long = []
        speed_limit_preview_long_distances = []
        speed_limit_preview_long_min_distances = []
        steering_angles = []
        steering_angles_actual = []
        steering_inputs = []
        desired_steer_angles = []
        motor_torques = []
        brake_torques = []
        timestamps = []
        unity_times = []
        unity_frame_counts = []
        unity_delta_times = []
        unity_smooth_delta_times = []
        unity_unscaled_delta_times = []
        unity_time_scales = []
        
        # Ground truth data
        gt_left_lane_line_x = []
        gt_right_lane_line_x = []
        gt_lane_center_x = []
        gt_path_curvature = []
        gt_desired_heading = []
        camera_8m_screen_y = []  # NEW: Camera calibration data
        camera_lookahead_screen_y = []  # NEW: Camera lookahead calibration data
        ground_truth_lookahead_distance = []  # NEW: Lookahead distance used for ground truth
        oracle_trajectory_world_xyz = []
        oracle_trajectory_screen_xy = []
        right_lane_fiducials_vehicle_xy = []
        right_lane_fiducials_vehicle_true_xy = []
        right_lane_fiducials_vehicle_monotonic_xy = []
        right_lane_fiducials_world_xyz = []
        right_lane_fiducials_screen_xy = []
        right_lane_fiducials_point_count = []
        right_lane_fiducials_horizon_meters = []
        right_lane_fiducials_spacing_meters = []
        right_lane_fiducials_enabled = []
        camera_field_of_view = []  # NEW: Camera FOV data
        camera_horizontal_fov = []  # NEW: Camera horizontal FOV data
        camera_pos_x = []  # NEW: Camera position data
        camera_pos_y = []
        camera_pos_z = []
        camera_forward_x = []  # NEW: Camera forward direction data
        camera_forward_y = []
        camera_forward_z = []
        topdown_camera_pos_x = []
        topdown_camera_pos_y = []
        topdown_camera_pos_z = []
        topdown_camera_forward_x = []
        topdown_camera_forward_y = []
        topdown_camera_forward_z = []
        topdown_camera_orthographic_size = []
        topdown_camera_field_of_view = []
        stream_front_unity_dt_ms = []
        stream_topdown_unity_dt_ms = []
        stream_topdown_front_dt_ms = []
        stream_topdown_front_frame_id_delta = []
        stream_front_frame_id_delta = []
        stream_topdown_frame_id_delta = []
        stream_front_latest_age_ms = []
        stream_front_queue_depth = []
        stream_front_drop_count = []
        stream_front_decode_in_flight = []
        stream_topdown_latest_age_ms = []
        stream_topdown_queue_depth = []
        stream_topdown_drop_count = []
        stream_topdown_decode_in_flight = []
        stream_front_last_realtime_s = []
        stream_topdown_last_realtime_s = []
        stream_front_timestamp_minus_realtime_ms = []
        stream_topdown_timestamp_minus_realtime_ms = []
        stream_front_source_timestamp = []
        stream_topdown_source_timestamp = []
        stream_front_timestamp_reused = []
        stream_topdown_timestamp_reused = []
        stream_front_timestamp_non_monotonic = []
        stream_topdown_timestamp_non_monotonic = []
        stream_front_negative_frame_delta = []
        stream_topdown_negative_frame_delta = []
        stream_front_frame_id_reused = []
        stream_topdown_frame_id_reused = []
        stream_front_clock_jump = []
        stream_topdown_clock_jump = []
        # NEW: Debug fields for diagnosing ground truth offset issues
        road_center_at_car_x = []
        road_center_at_car_y = []
        road_center_at_car_z = []
        road_center_at_lookahead_x = []
        road_center_at_lookahead_y = []
        road_center_at_lookahead_z = []
        road_center_reference_t = []
        road_frame_lateral_offset = []
        road_heading_deg = []
        car_heading_deg = []
        heading_delta_deg = []
        road_frame_lane_center_offset = []
        road_frame_lane_center_error = []
        vehicle_frame_lookahead_offset = []
        gt_rotation_debug_valid = []
        gt_rotation_used_road_frame = []
        gt_rotation_rejected_road_frame_hop = []
        gt_rotation_reference_heading_deg = []
        gt_rotation_road_frame_heading_deg = []
        gt_rotation_input_heading_deg = []
        gt_rotation_road_vs_ref_delta_deg = []
        gt_rotation_applied_delta_deg = []
        
        for frame in frames:
            vs = frame.vehicle_state
            unity_time = getattr(vs, 'unity_time', 0.0)
            unity_frame_count = getattr(vs, 'unity_frame_count', 0)
            if unity_time and self.last_unity_time:
                unity_time_gap = unity_time - self.last_unity_time
                frame_gap = unity_frame_count - (self.last_unity_frame_count or 0)
                if unity_time_gap > UNITY_TIME_GAP_WARN_SECONDS:
                    logger.warning(
                        "[RECORDER_TIME_GAP] unity_time_gap=%.3fs frame_gap=%s unity_frame=%s",
                        unity_time_gap,
                        frame_gap,
                        unity_frame_count
                    )
                    print(
                        "[RECORDER_TIME_GAP] "
                        f"unity_time_gap={unity_time_gap:.3f}s frame_gap={frame_gap} "
                        f"unity_frame={unity_frame_count}"
                    )
            if unity_time:
                self.last_unity_time = unity_time
                self.last_unity_frame_count = unity_frame_count
            positions.append(vs.position)
            rotations.append(vs.rotation)
            velocities.append(vs.velocity)
            angular_velocities.append(vs.angular_velocity)
            speeds.append(vs.speed)
            speed_limits.append(getattr(vs, 'speed_limit', 0.0))
            speed_limit_previews.append(getattr(vs, 'speed_limit_preview', 0.0))
            speed_limit_preview_distances.append(getattr(vs, 'speed_limit_preview_distance', 0.0))
            speed_limit_preview_min_distances.append(
                getattr(vs, 'speed_limit_preview_min_distance', 0.0)
            )
            speed_limit_preview_mid.append(getattr(vs, 'speed_limit_preview_mid', 0.0))
            speed_limit_preview_mid_distances.append(
                getattr(vs, 'speed_limit_preview_mid_distance', 0.0)
            )
            speed_limit_preview_mid_min_distances.append(
                getattr(vs, 'speed_limit_preview_mid_min_distance', 0.0)
            )
            speed_limit_preview_long.append(getattr(vs, 'speed_limit_preview_long', 0.0))
            speed_limit_preview_long_distances.append(
                getattr(vs, 'speed_limit_preview_long_distance', 0.0)
            )
            speed_limit_preview_long_min_distances.append(
                getattr(vs, 'speed_limit_preview_long_min_distance', 0.0)
            )
            steering_angles.append(vs.steering_angle)
            steering_angles_actual.append(
                vs.steering_angle_actual if vs.steering_angle_actual is not None else 0.0
            )
            steering_inputs.append(getattr(vs, 'steering_input', 0.0))
            desired_steer_angles.append(getattr(vs, 'desired_steer_angle', 0.0))
            motor_torques.append(vs.motor_torque)
            brake_torques.append(vs.brake_torque)
            timestamps.append(vs.timestamp)
            unity_times.append(unity_time)
            unity_frame_counts.append(unity_frame_count)
            unity_delta_times.append(getattr(vs, 'unity_delta_time', 0.0))
            unity_smooth_delta_times.append(getattr(vs, 'unity_smooth_delta_time', 0.0))
            unity_unscaled_delta_times.append(getattr(vs, 'unity_unscaled_delta_time', 0.0))
            unity_time_scales.append(getattr(vs, 'unity_time_scale', 1.0))
            
            # Extract ground truth if available (from vehicle state)
            # CRITICAL: Check for new names first, then old names for backward compatibility
            # Use getattr with default to handle both cases
            gt_left = getattr(vs, 'ground_truth_left_lane_line_x', None)
            if gt_left is None:
                gt_left = getattr(vs, 'ground_truth_left_lane_x', 0.0)  # Backward compatibility
            
            gt_right = getattr(vs, 'ground_truth_right_lane_line_x', None)
            if gt_right is None:
                gt_right = getattr(vs, 'ground_truth_right_lane_x', 0.0)  # Backward compatibility
            
            # Only append if we have valid ground truth data (not None and not both 0.0)
            if gt_left is not None and gt_right is not None and (gt_left != 0.0 or gt_right != 0.0):
                gt_left_lane_line_x.append(gt_left)
                gt_right_lane_line_x.append(gt_right)
                gt_lane_center_x.append(getattr(vs, 'ground_truth_lane_center_x', (gt_left + gt_right) / 2.0))
                gt_path_curvature.append(getattr(vs, 'ground_truth_path_curvature', 0.0))
                gt_desired_heading.append(getattr(vs, 'ground_truth_desired_heading', 0.0))
            else:
                # No ground truth data - append 0.0 to maintain array length
                gt_left_lane_line_x.append(0.0)
                gt_right_lane_line_x.append(0.0)
                gt_lane_center_x.append(0.0)
                gt_path_curvature.append(0.0)
                gt_desired_heading.append(0.0)
            
            # Extract camera calibration data
            cam_value = getattr(vs, 'camera_8m_screen_y', -1.0)
            camera_8m_screen_y.append(cam_value)
            camera_lookahead_screen_y.append(getattr(vs, 'camera_lookahead_screen_y', -1.0))
            ground_truth_lookahead_distance.append(getattr(vs, 'ground_truth_lookahead_distance', 8.0))
            oracle_trajectory_world_xyz.append(
                np.asarray(
                    getattr(vs, 'oracle_trajectory_world_xyz', np.array([], dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(-1)
            )
            oracle_trajectory_screen_xy.append(
                np.asarray(
                    getattr(vs, 'oracle_trajectory_screen_xy', np.array([], dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(-1)
            )
            right_lane_fiducials_vehicle_xy.append(
                np.asarray(
                    getattr(vs, 'right_lane_fiducials_vehicle_xy', np.array([], dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(-1)
            )
            right_lane_fiducials_vehicle_true_xy.append(
                np.asarray(
                    getattr(vs, 'right_lane_fiducials_vehicle_true_xy', np.array([], dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(-1)
            )
            right_lane_fiducials_vehicle_monotonic_xy.append(
                np.asarray(
                    getattr(vs, 'right_lane_fiducials_vehicle_monotonic_xy', np.array([], dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(-1)
            )
            right_lane_fiducials_world_xyz.append(
                np.asarray(
                    getattr(vs, 'right_lane_fiducials_world_xyz', np.array([], dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(-1)
            )
            right_lane_fiducials_screen_xy.append(
                np.asarray(
                    getattr(vs, 'right_lane_fiducials_screen_xy', np.array([], dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(-1)
            )
            right_lane_fiducials_point_count.append(
                int(getattr(vs, 'right_lane_fiducials_point_count', 0))
            )
            right_lane_fiducials_horizon_meters.append(
                float(getattr(vs, 'right_lane_fiducials_horizon_meters', 0.0))
            )
            right_lane_fiducials_spacing_meters.append(
                float(getattr(vs, 'right_lane_fiducials_spacing_meters', 0.0))
            )
            right_lane_fiducials_enabled.append(
                1 if getattr(vs, 'right_lane_fiducials_enabled', False) else 0
            )
            
            # NEW: Extract camera FOV and position data
            camera_field_of_view.append(getattr(vs, 'camera_field_of_view', -1.0))
            camera_horizontal_fov.append(getattr(vs, 'camera_horizontal_fov', -1.0))
            camera_pos_x.append(getattr(vs, 'camera_pos_x', 0.0))
            camera_pos_y.append(getattr(vs, 'camera_pos_y', 0.0))
            camera_pos_z.append(getattr(vs, 'camera_pos_z', 0.0))
            camera_forward_x.append(getattr(vs, 'camera_forward_x', 0.0))
            camera_forward_y.append(getattr(vs, 'camera_forward_y', 0.0))
            camera_forward_z.append(getattr(vs, 'camera_forward_z', 0.0))
            topdown_camera_pos_x.append(getattr(vs, 'topdown_camera_pos_x', 0.0))
            topdown_camera_pos_y.append(getattr(vs, 'topdown_camera_pos_y', 0.0))
            topdown_camera_pos_z.append(getattr(vs, 'topdown_camera_pos_z', 0.0))
            topdown_camera_forward_x.append(getattr(vs, 'topdown_camera_forward_x', 0.0))
            topdown_camera_forward_y.append(getattr(vs, 'topdown_camera_forward_y', 0.0))
            topdown_camera_forward_z.append(getattr(vs, 'topdown_camera_forward_z', 0.0))
            topdown_camera_orthographic_size.append(
                getattr(vs, 'topdown_camera_orthographic_size', 0.0)
            )
            topdown_camera_field_of_view.append(
                getattr(vs, 'topdown_camera_field_of_view', 0.0)
            )
            stream_front_unity_dt_ms.append(getattr(vs, 'stream_front_unity_dt_ms', 0.0))
            stream_topdown_unity_dt_ms.append(getattr(vs, 'stream_topdown_unity_dt_ms', 0.0))
            stream_topdown_front_dt_ms.append(getattr(vs, 'stream_topdown_front_dt_ms', 0.0))
            stream_topdown_front_frame_id_delta.append(
                getattr(vs, 'stream_topdown_front_frame_id_delta', 0.0)
            )
            stream_front_frame_id_delta.append(getattr(vs, 'stream_front_frame_id_delta', 0.0))
            stream_topdown_frame_id_delta.append(
                getattr(vs, 'stream_topdown_frame_id_delta', 0.0)
            )
            stream_front_latest_age_ms.append(getattr(vs, 'stream_front_latest_age_ms', 0.0))
            stream_front_queue_depth.append(getattr(vs, 'stream_front_queue_depth', 0.0))
            stream_front_drop_count.append(getattr(vs, 'stream_front_drop_count', 0.0))
            stream_front_decode_in_flight.append(
                getattr(vs, 'stream_front_decode_in_flight', 0.0)
            )
            stream_topdown_latest_age_ms.append(getattr(vs, 'stream_topdown_latest_age_ms', 0.0))
            stream_topdown_queue_depth.append(getattr(vs, 'stream_topdown_queue_depth', 0.0))
            stream_topdown_drop_count.append(getattr(vs, 'stream_topdown_drop_count', 0.0))
            stream_topdown_decode_in_flight.append(
                getattr(vs, 'stream_topdown_decode_in_flight', 0.0)
            )
            stream_front_last_realtime_s.append(getattr(vs, 'stream_front_last_realtime_s', 0.0))
            stream_topdown_last_realtime_s.append(
                getattr(vs, 'stream_topdown_last_realtime_s', 0.0)
            )
            stream_front_timestamp_minus_realtime_ms.append(
                getattr(vs, 'stream_front_timestamp_minus_realtime_ms', 0.0)
            )
            stream_topdown_timestamp_minus_realtime_ms.append(
                getattr(vs, 'stream_topdown_timestamp_minus_realtime_ms', 0.0)
            )
            stream_front_source_timestamp.append(
                getattr(vs, 'stream_front_source_timestamp', vs.timestamp)
            )
            stream_topdown_source_timestamp.append(
                getattr(vs, 'stream_topdown_source_timestamp', 0.0)
            )
            stream_front_timestamp_reused.append(
                getattr(vs, 'stream_front_timestamp_reused', 0.0)
            )
            stream_topdown_timestamp_reused.append(
                getattr(vs, 'stream_topdown_timestamp_reused', 0.0)
            )
            stream_front_timestamp_non_monotonic.append(
                getattr(vs, 'stream_front_timestamp_non_monotonic', 0.0)
            )
            stream_topdown_timestamp_non_monotonic.append(
                getattr(vs, 'stream_topdown_timestamp_non_monotonic', 0.0)
            )
            stream_front_negative_frame_delta.append(
                getattr(vs, 'stream_front_negative_frame_delta', 0.0)
            )
            stream_topdown_negative_frame_delta.append(
                getattr(vs, 'stream_topdown_negative_frame_delta', 0.0)
            )
            stream_front_frame_id_reused.append(
                getattr(vs, 'stream_front_frame_id_reused', 0.0)
            )
            stream_topdown_frame_id_reused.append(
                getattr(vs, 'stream_topdown_frame_id_reused', 0.0)
            )
            stream_front_clock_jump.append(
                getattr(vs, 'stream_front_clock_jump', 0.0)
            )
            stream_topdown_clock_jump.append(
                getattr(vs, 'stream_topdown_clock_jump', 0.0)
            )
            # NEW: Debug fields for diagnosing ground truth offset issues
            road_center_at_car_x.append(getattr(vs, 'road_center_at_car_x', 0.0))
            road_center_at_car_y.append(getattr(vs, 'road_center_at_car_y', 0.0))
            road_center_at_car_z.append(getattr(vs, 'road_center_at_car_z', 0.0))
            road_center_at_lookahead_x.append(getattr(vs, 'road_center_at_lookahead_x', 0.0))
            road_center_at_lookahead_y.append(getattr(vs, 'road_center_at_lookahead_y', 0.0))
            road_center_at_lookahead_z.append(getattr(vs, 'road_center_at_lookahead_z', 0.0))
            road_center_reference_t.append(getattr(vs, 'road_center_reference_t', 0.0))
            road_frame_lateral_offset.append(getattr(vs, 'road_frame_lateral_offset', 0.0))
            road_heading_deg.append(getattr(vs, 'road_heading_deg', 0.0))
            car_heading_deg.append(getattr(vs, 'car_heading_deg', 0.0))
            heading_delta_deg.append(getattr(vs, 'heading_delta_deg', 0.0))
            road_frame_lane_center_offset.append(getattr(vs, 'road_frame_lane_center_offset', 0.0))
            road_frame_lane_center_error.append(getattr(vs, 'road_frame_lane_center_error', 0.0))
            vehicle_frame_lookahead_offset.append(getattr(vs, 'vehicle_frame_lookahead_offset', 0.0))
            gt_rotation_debug_valid.append(
                1 if getattr(vs, 'gt_rotation_debug_valid', getattr(vs, 'gtRotationDebugValid', False)) else 0
            )
            gt_rotation_used_road_frame.append(
                1 if getattr(vs, 'gt_rotation_used_road_frame', getattr(vs, 'gtRotationUsedRoadFrame', False)) else 0
            )
            gt_rotation_rejected_road_frame_hop.append(
                1 if getattr(vs, 'gt_rotation_rejected_road_frame_hop', getattr(vs, 'gtRotationRejectedRoadFrameHop', False)) else 0
            )
            gt_rotation_reference_heading_deg.append(
                getattr(vs, 'gt_rotation_reference_heading_deg', getattr(vs, 'gtRotationReferenceHeadingDeg', 0.0))
            )
            gt_rotation_road_frame_heading_deg.append(
                getattr(vs, 'gt_rotation_road_frame_heading_deg', getattr(vs, 'gtRotationRoadFrameHeadingDeg', 0.0))
            )
            gt_rotation_input_heading_deg.append(
                getattr(vs, 'gt_rotation_input_heading_deg', getattr(vs, 'gtRotationInputHeadingDeg', 0.0))
            )
            gt_rotation_road_vs_ref_delta_deg.append(
                getattr(vs, 'gt_rotation_road_vs_ref_delta_deg', getattr(vs, 'gtRotationRoadVsRefDeltaDeg', 0.0))
            )
            gt_rotation_applied_delta_deg.append(
                getattr(vs, 'gt_rotation_applied_delta_deg', getattr(vs, 'gtRotationAppliedDeltaDeg', 0.0))
            )
            # Debug: Log first few frames to see what we're getting
            if len(camera_8m_screen_y) <= 3:
                logger.debug(f"[RECORDER] Frame {len(camera_8m_screen_y)-1}: camera_8m_screen_y = {cam_value} (from VehicleState)")
        
        if positions:
            current_size = self.h5_file["vehicle/timestamps"].shape[0]
            new_size = current_size + len(positions)
            
            # Resize all vehicle datasets
            for key in ["timestamps", "position", "rotation", "velocity",
                       "angular_velocity", "speed", "speed_limit", "speed_limit_preview",
                       "speed_limit_preview_distance", "speed_limit_preview_min_distance",
                       "speed_limit_preview_mid", "speed_limit_preview_mid_distance",
                       "speed_limit_preview_mid_min_distance", "speed_limit_preview_long",
                       "speed_limit_preview_long_distance",
                       "speed_limit_preview_long_min_distance",
                      "steering_angle", "steering_angle_actual",
                      "steering_input", "desired_steer_angle",
                       "motor_torque", "brake_torque", "camera_8m_screen_y",
                       "camera_lookahead_screen_y", "ground_truth_lookahead_distance",
                       "right_lane_fiducials_point_count", "right_lane_fiducials_horizon_meters",
                       "right_lane_fiducials_spacing_meters", "right_lane_fiducials_enabled",
                       "unity_time", "unity_frame_count", "unity_delta_time",
                       "unity_smooth_delta_time", "unity_unscaled_delta_time",
                       "unity_time_scale"]:
                if key == "timestamps" or key in ["speed", "speed_limit", "speed_limit_preview",
                                                "speed_limit_preview_distance", "speed_limit_preview_min_distance",
                                                "speed_limit_preview_mid", "speed_limit_preview_mid_distance",
                                                "speed_limit_preview_mid_min_distance", "speed_limit_preview_long",
                                                "speed_limit_preview_long_distance",
                                                "speed_limit_preview_long_min_distance",
                                                "steering_angle", "steering_angle_actual",
                                                "steering_input", "desired_steer_angle",
                                                 "motor_torque", "brake_torque", "camera_8m_screen_y",
                                                 "camera_lookahead_screen_y", "ground_truth_lookahead_distance",
                                                 "right_lane_fiducials_point_count",
                                                 "right_lane_fiducials_horizon_meters",
                                                 "right_lane_fiducials_spacing_meters",
                                                 "right_lane_fiducials_enabled",
                                                 "unity_time", "unity_frame_count", "unity_delta_time",
                                                 "unity_smooth_delta_time", "unity_unscaled_delta_time",
                                                 "unity_time_scale"]:
                    self.h5_file[f"vehicle/{key}"].resize((new_size,))
                else:
                    dim = 3 if key != "rotation" else 4
                    self.h5_file[f"vehicle/{key}"].resize((new_size, dim))
            
            # Write data
            self.h5_file["vehicle/timestamps"][current_size:] = timestamps
            self.h5_file["vehicle/position"][current_size:] = positions
            self.h5_file["vehicle/rotation"][current_size:] = rotations
            self.h5_file["vehicle/velocity"][current_size:] = velocities
            self.h5_file["vehicle/angular_velocity"][current_size:] = angular_velocities
            self.h5_file["vehicle/speed"][current_size:] = speeds
            self.h5_file["vehicle/speed_limit"][current_size:] = speed_limits
            self.h5_file["vehicle/speed_limit_preview"][current_size:] = speed_limit_previews
            self.h5_file["vehicle/speed_limit_preview_distance"][current_size:] = speed_limit_preview_distances
            self.h5_file["vehicle/speed_limit_preview_min_distance"][current_size:] = (
                speed_limit_preview_min_distances
            )
            self.h5_file["vehicle/speed_limit_preview_mid"][current_size:] = speed_limit_preview_mid
            self.h5_file["vehicle/speed_limit_preview_mid_distance"][current_size:] = (
                speed_limit_preview_mid_distances
            )
            self.h5_file["vehicle/speed_limit_preview_mid_min_distance"][current_size:] = (
                speed_limit_preview_mid_min_distances
            )
            self.h5_file["vehicle/speed_limit_preview_long"][current_size:] = speed_limit_preview_long
            self.h5_file["vehicle/speed_limit_preview_long_distance"][current_size:] = (
                speed_limit_preview_long_distances
            )
            self.h5_file["vehicle/speed_limit_preview_long_min_distance"][current_size:] = (
                speed_limit_preview_long_min_distances
            )
            self.h5_file["vehicle/steering_angle"][current_size:] = steering_angles
            self.h5_file["vehicle/steering_angle_actual"][current_size:] = steering_angles_actual
            self.h5_file["vehicle/steering_input"][current_size:] = steering_inputs
            self.h5_file["vehicle/desired_steer_angle"][current_size:] = desired_steer_angles
            self.h5_file["vehicle/motor_torque"][current_size:] = motor_torques
            self.h5_file["vehicle/brake_torque"][current_size:] = brake_torques
            self.h5_file["vehicle/unity_time"][current_size:] = np.array(unity_times, dtype=np.float64)
            self.h5_file["vehicle/unity_frame_count"][current_size:] = np.array(unity_frame_counts, dtype=np.int32)
            self.h5_file["vehicle/unity_delta_time"][current_size:] = np.array(unity_delta_times, dtype=np.float32)
            self.h5_file["vehicle/unity_smooth_delta_time"][current_size:] = np.array(unity_smooth_delta_times, dtype=np.float32)
            self.h5_file["vehicle/unity_unscaled_delta_time"][current_size:] = np.array(unity_unscaled_delta_times, dtype=np.float32)
            self.h5_file["vehicle/unity_time_scale"][current_size:] = np.array(unity_time_scales, dtype=np.float32)
            self.h5_file["vehicle/right_lane_fiducials_point_count"][current_size:] = np.array(
                right_lane_fiducials_point_count, dtype=np.int32
            )
            self.h5_file["vehicle/right_lane_fiducials_horizon_meters"][current_size:] = np.array(
                right_lane_fiducials_horizon_meters, dtype=np.float32
            )
            self.h5_file["vehicle/right_lane_fiducials_spacing_meters"][current_size:] = np.array(
                right_lane_fiducials_spacing_meters, dtype=np.float32
            )
            self.h5_file["vehicle/right_lane_fiducials_enabled"][current_size:] = np.array(
                right_lane_fiducials_enabled, dtype=np.int8
            )
            self.h5_file["vehicle/oracle_trajectory_world_xyz"].resize((new_size,))
            self.h5_file["vehicle/oracle_trajectory_screen_xy"].resize((new_size,))
            self.h5_file["vehicle/right_lane_fiducials_vehicle_xy"].resize((new_size,))
            self.h5_file["vehicle/right_lane_fiducials_vehicle_true_xy"].resize((new_size,))
            self.h5_file["vehicle/right_lane_fiducials_vehicle_monotonic_xy"].resize((new_size,))
            self.h5_file["vehicle/right_lane_fiducials_world_xyz"].resize((new_size,))
            self.h5_file["vehicle/right_lane_fiducials_screen_xy"].resize((new_size,))
            for i in range(len(positions)):
                self.h5_file["vehicle/oracle_trajectory_world_xyz"][current_size + i] = (
                    oracle_trajectory_world_xyz[i]
                )
                self.h5_file["vehicle/oracle_trajectory_screen_xy"][current_size + i] = (
                    oracle_trajectory_screen_xy[i]
                )
                self.h5_file["vehicle/right_lane_fiducials_vehicle_xy"][current_size + i] = (
                    right_lane_fiducials_vehicle_xy[i]
                )
                self.h5_file["vehicle/right_lane_fiducials_vehicle_true_xy"][current_size + i] = (
                    right_lane_fiducials_vehicle_true_xy[i]
                )
                self.h5_file["vehicle/right_lane_fiducials_vehicle_monotonic_xy"][current_size + i] = (
                    right_lane_fiducials_vehicle_monotonic_xy[i]
                )
                self.h5_file["vehicle/right_lane_fiducials_world_xyz"][current_size + i] = (
                    right_lane_fiducials_world_xyz[i]
                )
                self.h5_file["vehicle/right_lane_fiducials_screen_xy"][current_size + i] = (
                    right_lane_fiducials_screen_xy[i]
                )
            
            # Write ground truth data
            # CRITICAL: Wrap in try-except to prevent errors from blocking camera_8m_screen_y write
            try:
                if gt_left_lane_line_x and len(gt_left_lane_line_x) > 0:
                    # CRITICAL: Ensure all ground truth arrays have the same length
                    expected_len = len(gt_left_lane_line_x)
                    if (len(gt_right_lane_line_x) != expected_len or 
                        len(gt_lane_center_x) != expected_len or
                        len(gt_path_curvature) != expected_len or
                        len(gt_desired_heading) != expected_len):
                        logger.warning(f"[RECORDER] Ground truth array length mismatch: "
                                     f"left={len(gt_left_lane_line_x)}, right={len(gt_right_lane_line_x)}, "
                                     f"center={len(gt_lane_center_x)}, curvature={len(gt_path_curvature)}, "
                                     f"heading={len(gt_desired_heading)}. Skipping ground truth write.")
                    else:
                        gt_current_size = self.h5_file["ground_truth/left_lane_line_x"].shape[0]
                        gt_new_size = gt_current_size + len(gt_left_lane_line_x)
                        
                        self.h5_file["ground_truth/left_lane_line_x"].resize((gt_new_size,))
                        self.h5_file["ground_truth/right_lane_line_x"].resize((gt_new_size,))
                        self.h5_file["ground_truth/lane_center_x"].resize((gt_new_size,))
                        self.h5_file["ground_truth/path_curvature"].resize((gt_new_size,))
                        self.h5_file["ground_truth/desired_heading"].resize((gt_new_size,))
                        
                        # Convert to numpy arrays to ensure correct types
                        self.h5_file["ground_truth/left_lane_line_x"][gt_current_size:] = np.array(gt_left_lane_line_x, dtype=np.float32)
                        self.h5_file["ground_truth/right_lane_line_x"][gt_current_size:] = np.array(gt_right_lane_line_x, dtype=np.float32)
                        self.h5_file["ground_truth/lane_center_x"][gt_current_size:] = np.array(gt_lane_center_x, dtype=np.float32)
                        self.h5_file["ground_truth/path_curvature"][gt_current_size:] = np.array(gt_path_curvature, dtype=np.float32)
                        self.h5_file["ground_truth/desired_heading"][gt_current_size:] = np.array(gt_desired_heading, dtype=np.float32)
            except Exception as e:
                logger.error(f"[RECORDER] Error writing ground truth data: {e}", exc_info=True)
                # Continue to write camera_8m_screen_y even if ground truth fails
            
            # Write camera calibration data (already resized above, just write values)
            # CRITICAL: Always write if we have data (even if some values are 0.0 or -1.0)
            # This is written AFTER ground truth to ensure it's written even if ground truth fails
            try:
                if len(camera_8m_screen_y) > 0:
                    # Convert to numpy array to ensure correct shape
                    camera_8m_screen_y_array = np.array(camera_8m_screen_y, dtype=np.float32)
                    camera_lookahead_screen_y_array = np.array(camera_lookahead_screen_y, dtype=np.float32)
                    ground_truth_lookahead_distance_array = np.array(ground_truth_lookahead_distance, dtype=np.float32)
                    # CRITICAL: Ensure the array has the correct shape (should be 1D with length matching positions)
                    if camera_8m_screen_y_array.shape[0] == len(positions):
                        self.h5_file["vehicle/camera_8m_screen_y"][current_size:] = camera_8m_screen_y_array
                        self.h5_file["vehicle/camera_lookahead_screen_y"][current_size:] = camera_lookahead_screen_y_array
                        self.h5_file["vehicle/ground_truth_lookahead_distance"][current_size:] = ground_truth_lookahead_distance_array
                        logger.debug(f"[RECORDER] Wrote {len(camera_8m_screen_y)} camera_8m_screen_y values (min={camera_8m_screen_y_array.min():.1f}, max={camera_8m_screen_y_array.max():.1f})")
                    else:
                        logger.warning(f"[RECORDER] camera_8m_screen_y length mismatch: {len(camera_8m_screen_y)} vs {len(positions)}. Skipping write.")
                else:
                    logger.warning(f"[RECORDER] camera_8m_screen_y is empty! Expected {len(positions)} values. Filling with -1.0.")
                    # Fill with -1.0 to indicate no data
                    self.h5_file["vehicle/camera_8m_screen_y"][current_size:] = np.full(len(positions), -1.0, dtype=np.float32)
                    self.h5_file["vehicle/camera_lookahead_screen_y"][current_size:] = np.full(len(positions), -1.0, dtype=np.float32)
                    self.h5_file["vehicle/ground_truth_lookahead_distance"][current_size:] = np.full(len(positions), 8.0, dtype=np.float32)
            except Exception as e:
                logger.error(f"[RECORDER] Error writing camera_8m_screen_y: {e}", exc_info=True)
            
            # Write camera FOV and position data
            try:
                if len(camera_field_of_view) > 0:
                    self.h5_file["vehicle/camera_field_of_view"].resize((current_size + len(camera_field_of_view),))
                    self.h5_file["vehicle/camera_field_of_view"][current_size:] = np.array(camera_field_of_view, dtype=np.float32)
                    self.h5_file["vehicle/camera_horizontal_fov"].resize((current_size + len(camera_horizontal_fov),))
                    self.h5_file["vehicle/camera_horizontal_fov"][current_size:] = np.array(camera_horizontal_fov, dtype=np.float32)
                    self.h5_file["vehicle/camera_pos_x"].resize((current_size + len(camera_pos_x),))
                    self.h5_file["vehicle/camera_pos_x"][current_size:] = np.array(camera_pos_x, dtype=np.float32)
                    self.h5_file["vehicle/camera_pos_y"].resize((current_size + len(camera_pos_y),))
                    self.h5_file["vehicle/camera_pos_y"][current_size:] = np.array(camera_pos_y, dtype=np.float32)
                    self.h5_file["vehicle/camera_pos_z"].resize((current_size + len(camera_pos_z),))
                    self.h5_file["vehicle/camera_pos_z"][current_size:] = np.array(camera_pos_z, dtype=np.float32)
                    self.h5_file["vehicle/camera_forward_x"].resize((current_size + len(camera_forward_x),))
                    self.h5_file["vehicle/camera_forward_x"][current_size:] = np.array(camera_forward_x, dtype=np.float32)
                    self.h5_file["vehicle/camera_forward_y"].resize((current_size + len(camera_forward_y),))
                    self.h5_file["vehicle/camera_forward_y"][current_size:] = np.array(camera_forward_y, dtype=np.float32)
                    self.h5_file["vehicle/camera_forward_z"].resize((current_size + len(camera_forward_z),))
                    self.h5_file["vehicle/camera_forward_z"][current_size:] = np.array(camera_forward_z, dtype=np.float32)
                    self.h5_file["vehicle/topdown_camera_pos_x"].resize((current_size + len(topdown_camera_pos_x),))
                    self.h5_file["vehicle/topdown_camera_pos_x"][current_size:] = np.array(topdown_camera_pos_x, dtype=np.float32)
                    self.h5_file["vehicle/topdown_camera_pos_y"].resize((current_size + len(topdown_camera_pos_y),))
                    self.h5_file["vehicle/topdown_camera_pos_y"][current_size:] = np.array(topdown_camera_pos_y, dtype=np.float32)
                    self.h5_file["vehicle/topdown_camera_pos_z"].resize((current_size + len(topdown_camera_pos_z),))
                    self.h5_file["vehicle/topdown_camera_pos_z"][current_size:] = np.array(topdown_camera_pos_z, dtype=np.float32)
                    self.h5_file["vehicle/topdown_camera_forward_x"].resize((current_size + len(topdown_camera_forward_x),))
                    self.h5_file["vehicle/topdown_camera_forward_x"][current_size:] = np.array(topdown_camera_forward_x, dtype=np.float32)
                    self.h5_file["vehicle/topdown_camera_forward_y"].resize((current_size + len(topdown_camera_forward_y),))
                    self.h5_file["vehicle/topdown_camera_forward_y"][current_size:] = np.array(topdown_camera_forward_y, dtype=np.float32)
                    self.h5_file["vehicle/topdown_camera_forward_z"].resize((current_size + len(topdown_camera_forward_z),))
                    self.h5_file["vehicle/topdown_camera_forward_z"][current_size:] = np.array(topdown_camera_forward_z, dtype=np.float32)
                    self.h5_file["vehicle/topdown_camera_orthographic_size"].resize((current_size + len(topdown_camera_orthographic_size),))
                    self.h5_file["vehicle/topdown_camera_orthographic_size"][current_size:] = np.array(topdown_camera_orthographic_size, dtype=np.float32)
                    self.h5_file["vehicle/topdown_camera_field_of_view"].resize((current_size + len(topdown_camera_field_of_view),))
                    self.h5_file["vehicle/topdown_camera_field_of_view"][current_size:] = np.array(topdown_camera_field_of_view, dtype=np.float32)
                    self.h5_file["vehicle/stream_front_unity_dt_ms"].resize((current_size + len(stream_front_unity_dt_ms),))
                    self.h5_file["vehicle/stream_front_unity_dt_ms"][current_size:] = np.array(stream_front_unity_dt_ms, dtype=np.float32)
                    self.h5_file["vehicle/stream_topdown_unity_dt_ms"].resize((current_size + len(stream_topdown_unity_dt_ms),))
                    self.h5_file["vehicle/stream_topdown_unity_dt_ms"][current_size:] = np.array(stream_topdown_unity_dt_ms, dtype=np.float32)
                    self.h5_file["vehicle/stream_topdown_front_dt_ms"].resize((current_size + len(stream_topdown_front_dt_ms),))
                    self.h5_file["vehicle/stream_topdown_front_dt_ms"][current_size:] = np.array(stream_topdown_front_dt_ms, dtype=np.float32)
                    self.h5_file["vehicle/stream_topdown_front_frame_id_delta"].resize((current_size + len(stream_topdown_front_frame_id_delta),))
                    self.h5_file["vehicle/stream_topdown_front_frame_id_delta"][current_size:] = np.array(stream_topdown_front_frame_id_delta, dtype=np.float32)
                    self.h5_file["vehicle/stream_front_frame_id_delta"].resize((current_size + len(stream_front_frame_id_delta),))
                    self.h5_file["vehicle/stream_front_frame_id_delta"][current_size:] = np.array(stream_front_frame_id_delta, dtype=np.float32)
                    self.h5_file["vehicle/stream_topdown_frame_id_delta"].resize((current_size + len(stream_topdown_frame_id_delta),))
                    self.h5_file["vehicle/stream_topdown_frame_id_delta"][current_size:] = np.array(stream_topdown_frame_id_delta, dtype=np.float32)
                    self.h5_file["vehicle/stream_front_latest_age_ms"].resize((current_size + len(stream_front_latest_age_ms),))
                    self.h5_file["vehicle/stream_front_latest_age_ms"][current_size:] = np.array(stream_front_latest_age_ms, dtype=np.float32)
                    self.h5_file["vehicle/stream_front_queue_depth"].resize((current_size + len(stream_front_queue_depth),))
                    self.h5_file["vehicle/stream_front_queue_depth"][current_size:] = np.array(stream_front_queue_depth, dtype=np.float32)
                    self.h5_file["vehicle/stream_front_drop_count"].resize((current_size + len(stream_front_drop_count),))
                    self.h5_file["vehicle/stream_front_drop_count"][current_size:] = np.array(stream_front_drop_count, dtype=np.float32)
                    self.h5_file["vehicle/stream_front_decode_in_flight"].resize((current_size + len(stream_front_decode_in_flight),))
                    self.h5_file["vehicle/stream_front_decode_in_flight"][current_size:] = np.array(stream_front_decode_in_flight, dtype=np.float32)
                    self.h5_file["vehicle/stream_topdown_latest_age_ms"].resize((current_size + len(stream_topdown_latest_age_ms),))
                    self.h5_file["vehicle/stream_topdown_latest_age_ms"][current_size:] = np.array(stream_topdown_latest_age_ms, dtype=np.float32)
                    self.h5_file["vehicle/stream_topdown_queue_depth"].resize((current_size + len(stream_topdown_queue_depth),))
                    self.h5_file["vehicle/stream_topdown_queue_depth"][current_size:] = np.array(stream_topdown_queue_depth, dtype=np.float32)
                    self.h5_file["vehicle/stream_topdown_drop_count"].resize((current_size + len(stream_topdown_drop_count),))
                    self.h5_file["vehicle/stream_topdown_drop_count"][current_size:] = np.array(stream_topdown_drop_count, dtype=np.float32)
                    self.h5_file["vehicle/stream_topdown_decode_in_flight"].resize((current_size + len(stream_topdown_decode_in_flight),))
                    self.h5_file["vehicle/stream_topdown_decode_in_flight"][current_size:] = np.array(stream_topdown_decode_in_flight, dtype=np.float32)
                    self.h5_file["vehicle/stream_front_last_realtime_s"].resize((current_size + len(stream_front_last_realtime_s),))
                    self.h5_file["vehicle/stream_front_last_realtime_s"][current_size:] = np.array(stream_front_last_realtime_s, dtype=np.float32)
                    self.h5_file["vehicle/stream_topdown_last_realtime_s"].resize((current_size + len(stream_topdown_last_realtime_s),))
                    self.h5_file["vehicle/stream_topdown_last_realtime_s"][current_size:] = np.array(stream_topdown_last_realtime_s, dtype=np.float32)
                    self.h5_file["vehicle/stream_front_timestamp_minus_realtime_ms"].resize((current_size + len(stream_front_timestamp_minus_realtime_ms),))
                    self.h5_file["vehicle/stream_front_timestamp_minus_realtime_ms"][current_size:] = np.array(stream_front_timestamp_minus_realtime_ms, dtype=np.float32)
                    self.h5_file["vehicle/stream_topdown_timestamp_minus_realtime_ms"].resize((current_size + len(stream_topdown_timestamp_minus_realtime_ms),))
                    self.h5_file["vehicle/stream_topdown_timestamp_minus_realtime_ms"][current_size:] = np.array(stream_topdown_timestamp_minus_realtime_ms, dtype=np.float32)
                    self.h5_file["vehicle/stream_front_source_timestamp"].resize((current_size + len(stream_front_source_timestamp),))
                    self.h5_file["vehicle/stream_front_source_timestamp"][current_size:] = np.array(stream_front_source_timestamp, dtype=np.float64)
                    self.h5_file["vehicle/stream_topdown_source_timestamp"].resize((current_size + len(stream_topdown_source_timestamp),))
                    self.h5_file["vehicle/stream_topdown_source_timestamp"][current_size:] = np.array(stream_topdown_source_timestamp, dtype=np.float64)
                    self.h5_file["vehicle/stream_front_timestamp_reused"].resize((current_size + len(stream_front_timestamp_reused),))
                    self.h5_file["vehicle/stream_front_timestamp_reused"][current_size:] = np.array(stream_front_timestamp_reused, dtype=np.float32)
                    self.h5_file["vehicle/stream_topdown_timestamp_reused"].resize((current_size + len(stream_topdown_timestamp_reused),))
                    self.h5_file["vehicle/stream_topdown_timestamp_reused"][current_size:] = np.array(stream_topdown_timestamp_reused, dtype=np.float32)
                    self.h5_file["vehicle/stream_front_timestamp_non_monotonic"].resize((current_size + len(stream_front_timestamp_non_monotonic),))
                    self.h5_file["vehicle/stream_front_timestamp_non_monotonic"][current_size:] = np.array(stream_front_timestamp_non_monotonic, dtype=np.float32)
                    self.h5_file["vehicle/stream_topdown_timestamp_non_monotonic"].resize((current_size + len(stream_topdown_timestamp_non_monotonic),))
                    self.h5_file["vehicle/stream_topdown_timestamp_non_monotonic"][current_size:] = np.array(stream_topdown_timestamp_non_monotonic, dtype=np.float32)
                    self.h5_file["vehicle/stream_front_negative_frame_delta"].resize((current_size + len(stream_front_negative_frame_delta),))
                    self.h5_file["vehicle/stream_front_negative_frame_delta"][current_size:] = np.array(stream_front_negative_frame_delta, dtype=np.float32)
                    self.h5_file["vehicle/stream_topdown_negative_frame_delta"].resize((current_size + len(stream_topdown_negative_frame_delta),))
                    self.h5_file["vehicle/stream_topdown_negative_frame_delta"][current_size:] = np.array(stream_topdown_negative_frame_delta, dtype=np.float32)
                    self.h5_file["vehicle/stream_front_frame_id_reused"].resize((current_size + len(stream_front_frame_id_reused),))
                    self.h5_file["vehicle/stream_front_frame_id_reused"][current_size:] = np.array(stream_front_frame_id_reused, dtype=np.float32)
                    self.h5_file["vehicle/stream_topdown_frame_id_reused"].resize((current_size + len(stream_topdown_frame_id_reused),))
                    self.h5_file["vehicle/stream_topdown_frame_id_reused"][current_size:] = np.array(stream_topdown_frame_id_reused, dtype=np.float32)
                    self.h5_file["vehicle/stream_front_clock_jump"].resize((current_size + len(stream_front_clock_jump),))
                    self.h5_file["vehicle/stream_front_clock_jump"][current_size:] = np.array(stream_front_clock_jump, dtype=np.float32)
                    self.h5_file["vehicle/stream_topdown_clock_jump"].resize((current_size + len(stream_topdown_clock_jump),))
                    self.h5_file["vehicle/stream_topdown_clock_jump"][current_size:] = np.array(stream_topdown_clock_jump, dtype=np.float32)
                    
                    # NEW: Write debug fields for diagnosing ground truth offset issues
                    if len(road_center_at_car_x) > 0:
                        self.h5_file["vehicle/road_center_at_car_x"].resize((current_size + len(road_center_at_car_x),))
                        self.h5_file["vehicle/road_center_at_car_x"][current_size:] = np.array(road_center_at_car_x, dtype=np.float32)
                        self.h5_file["vehicle/road_center_at_car_y"].resize((current_size + len(road_center_at_car_y),))
                        self.h5_file["vehicle/road_center_at_car_y"][current_size:] = np.array(road_center_at_car_y, dtype=np.float32)
                        self.h5_file["vehicle/road_center_at_car_z"].resize((current_size + len(road_center_at_car_z),))
                        self.h5_file["vehicle/road_center_at_car_z"][current_size:] = np.array(road_center_at_car_z, dtype=np.float32)
                        self.h5_file["vehicle/road_center_at_lookahead_x"].resize((current_size + len(road_center_at_lookahead_x),))
                        self.h5_file["vehicle/road_center_at_lookahead_x"][current_size:] = np.array(road_center_at_lookahead_x, dtype=np.float32)
                        self.h5_file["vehicle/road_center_at_lookahead_y"].resize((current_size + len(road_center_at_lookahead_y),))
                        self.h5_file["vehicle/road_center_at_lookahead_y"][current_size:] = np.array(road_center_at_lookahead_y, dtype=np.float32)
                        self.h5_file["vehicle/road_center_at_lookahead_z"].resize((current_size + len(road_center_at_lookahead_z),))
                        self.h5_file["vehicle/road_center_at_lookahead_z"][current_size:] = np.array(road_center_at_lookahead_z, dtype=np.float32)
                        self.h5_file["vehicle/road_center_reference_t"].resize((current_size + len(road_center_reference_t),))
                        self.h5_file["vehicle/road_center_reference_t"][current_size:] = np.array(road_center_reference_t, dtype=np.float32)
                        self.h5_file["vehicle/road_frame_lateral_offset"].resize((current_size + len(road_frame_lateral_offset),))
                        self.h5_file["vehicle/road_frame_lateral_offset"][current_size:] = np.array(
                            road_frame_lateral_offset, dtype=np.float32
                        )
                        self.h5_file["vehicle/road_heading_deg"].resize((current_size + len(road_heading_deg),))
                        self.h5_file["vehicle/road_heading_deg"][current_size:] = np.array(
                            road_heading_deg, dtype=np.float32
                        )
                        self.h5_file["vehicle/car_heading_deg"].resize((current_size + len(car_heading_deg),))
                        self.h5_file["vehicle/car_heading_deg"][current_size:] = np.array(
                            car_heading_deg, dtype=np.float32
                        )
                        self.h5_file["vehicle/heading_delta_deg"].resize((current_size + len(heading_delta_deg),))
                        self.h5_file["vehicle/heading_delta_deg"][current_size:] = np.array(
                            heading_delta_deg, dtype=np.float32
                        )
                        self.h5_file["vehicle/road_frame_lane_center_offset"].resize(
                            (current_size + len(road_frame_lane_center_offset),)
                        )
                        self.h5_file["vehicle/road_frame_lane_center_offset"][current_size:] = np.array(
                            road_frame_lane_center_offset, dtype=np.float32
                        )
                        self.h5_file["vehicle/road_frame_lane_center_error"].resize(
                            (current_size + len(road_frame_lane_center_error),)
                        )
                        self.h5_file["vehicle/road_frame_lane_center_error"][current_size:] = np.array(
                            road_frame_lane_center_error, dtype=np.float32
                        )
                        self.h5_file["vehicle/vehicle_frame_lookahead_offset"].resize(
                            (current_size + len(vehicle_frame_lookahead_offset),)
                        )
                        self.h5_file["vehicle/vehicle_frame_lookahead_offset"][current_size:] = np.array(
                            vehicle_frame_lookahead_offset, dtype=np.float32
                        )
                        self.h5_file["vehicle/gt_rotation_debug_valid"].resize(
                            (current_size + len(gt_rotation_debug_valid),)
                        )
                        self.h5_file["vehicle/gt_rotation_debug_valid"][current_size:] = np.array(
                            gt_rotation_debug_valid, dtype=np.int8
                        )
                        self.h5_file["vehicle/gt_rotation_used_road_frame"].resize(
                            (current_size + len(gt_rotation_used_road_frame),)
                        )
                        self.h5_file["vehicle/gt_rotation_used_road_frame"][current_size:] = np.array(
                            gt_rotation_used_road_frame, dtype=np.int8
                        )
                        self.h5_file["vehicle/gt_rotation_rejected_road_frame_hop"].resize(
                            (current_size + len(gt_rotation_rejected_road_frame_hop),)
                        )
                        self.h5_file["vehicle/gt_rotation_rejected_road_frame_hop"][current_size:] = np.array(
                            gt_rotation_rejected_road_frame_hop, dtype=np.int8
                        )
                        self.h5_file["vehicle/gt_rotation_reference_heading_deg"].resize(
                            (current_size + len(gt_rotation_reference_heading_deg),)
                        )
                        self.h5_file["vehicle/gt_rotation_reference_heading_deg"][current_size:] = np.array(
                            gt_rotation_reference_heading_deg, dtype=np.float32
                        )
                        self.h5_file["vehicle/gt_rotation_road_frame_heading_deg"].resize(
                            (current_size + len(gt_rotation_road_frame_heading_deg),)
                        )
                        self.h5_file["vehicle/gt_rotation_road_frame_heading_deg"][current_size:] = np.array(
                            gt_rotation_road_frame_heading_deg, dtype=np.float32
                        )
                        self.h5_file["vehicle/gt_rotation_input_heading_deg"].resize(
                            (current_size + len(gt_rotation_input_heading_deg),)
                        )
                        self.h5_file["vehicle/gt_rotation_input_heading_deg"][current_size:] = np.array(
                            gt_rotation_input_heading_deg, dtype=np.float32
                        )
                        self.h5_file["vehicle/gt_rotation_road_vs_ref_delta_deg"].resize(
                            (current_size + len(gt_rotation_road_vs_ref_delta_deg),)
                        )
                        self.h5_file["vehicle/gt_rotation_road_vs_ref_delta_deg"][current_size:] = np.array(
                            gt_rotation_road_vs_ref_delta_deg, dtype=np.float32
                        )
                        self.h5_file["vehicle/gt_rotation_applied_delta_deg"].resize(
                            (current_size + len(gt_rotation_applied_delta_deg),)
                        )
                        self.h5_file["vehicle/gt_rotation_applied_delta_deg"][current_size:] = np.array(
                            gt_rotation_applied_delta_deg, dtype=np.float32
                        )
            except Exception as e:
                logger.error(f"[RECORDER] Error writing camera FOV/position data: {e}", exc_info=True)
    
    def _write_control_commands(self, frames: List[RecordingFrame]):
        """Write control commands to HDF5."""
        timestamps = []
        steerings = []
        throttles = []
        brakes = []
        steering_before = []
        throttle_before = []
        brake_before = []
        feedforward_steering_list = []
        feedback_steering_list = []
        pid_integrals = []
        pid_derivatives = []
        pid_errors = []
        lateral_errors = []
        heading_errors = []
        total_errors = []
        total_error_scaled_list = []
        calculated_steering_angles = []
        raw_steerings = []
        lateral_corrections = []
        path_curvature_inputs = []
        straight_sign_flip_override_list = []
        straight_sign_flip_triggered_list = []
        straight_sign_flip_trigger_error_list = []
        straight_sign_flip_frames_remaining_list = []
        is_straight_list = []
        straight_oscillation_rate_list = []
        tuned_deadband_list = []
        tuned_error_smoothing_alpha_list = []
        # NEW: Control stale data tracking
        using_stale_perception_list = []
        stale_perception_reason_list = []
        emergency_stop_list = []
        target_speed_raw_list = []
        target_speed_post_limits_list = []
        target_speed_planned_list = []
        target_speed_final_list = []
        target_speed_slew_active_list = []
        target_speed_ramp_active_list = []
        launch_throttle_cap_list = []
        launch_throttle_cap_active_list = []
        steering_pre_rate_limit_list = []
        steering_post_rate_limit_list = []
        steering_post_jerk_limit_list = []
        steering_post_sign_flip_list = []
        steering_post_hard_clip_list = []
        steering_post_smoothing_list = []
        steering_rate_limited_active_list = []
        steering_jerk_limited_active_list = []
        steering_hard_clip_active_list = []
        steering_smoothing_active_list = []
        steering_rate_limited_delta_list = []
        steering_jerk_limited_delta_list = []
        steering_hard_clip_delta_list = []
        steering_smoothing_delta_list = []
        accel_feedforward_list = []
        brake_feedforward_list = []
        accel_capped_list = []
        jerk_capped_list = []
        
        for frame in frames:
            cc = frame.control_command
            timestamps.append(cc.timestamp)
            steerings.append(cc.steering)
            throttles.append(cc.throttle)
            brakes.append(cc.brake)
            steering_before.append(cc.steering_before_limits if cc.steering_before_limits is not None else cc.steering)
            throttle_before.append(cc.throttle_before_limits if cc.throttle_before_limits is not None else cc.throttle)
            brake_before.append(cc.brake_before_limits if cc.brake_before_limits is not None else cc.brake)
            feedforward_steering_list.append(getattr(cc, 'feedforward_steering', 0.0) or 0.0)
            feedback_steering_list.append(getattr(cc, 'feedback_steering', 0.0) or 0.0)
            accel_feedforward_list.append(getattr(cc, 'accel_feedforward', 0.0) or 0.0)
            brake_feedforward_list.append(getattr(cc, 'brake_feedforward', 0.0) or 0.0)
            accel_capped_list.append(1 if getattr(cc, 'longitudinal_accel_capped', False) else 0)
            jerk_capped_list.append(1 if getattr(cc, 'longitudinal_jerk_capped', False) else 0)
            pid_integrals.append(cc.pid_integral if cc.pid_integral is not None else 0.0)
            pid_derivatives.append(cc.pid_derivative if cc.pid_derivative is not None else 0.0)
            pid_errors.append(cc.pid_error if cc.pid_error is not None else 0.0)
            lateral_errors.append(cc.lateral_error if cc.lateral_error is not None else 0.0)
            heading_errors.append(cc.heading_error if cc.heading_error is not None else 0.0)
            total_errors.append(cc.total_error if cc.total_error is not None else 0.0)
            total_error_scaled_list.append(getattr(cc, 'total_error_scaled', 0.0) or 0.0)
            calculated_steering_angles.append(cc.calculated_steering_angle_deg if cc.calculated_steering_angle_deg is not None else 0.0)
            raw_steerings.append(cc.raw_steering if cc.raw_steering is not None else 0.0)
            lateral_corrections.append(cc.lateral_correction if cc.lateral_correction is not None else 0.0)
            path_curvature_inputs.append(cc.path_curvature_input if cc.path_curvature_input is not None else 0.0)
            straight_sign_flip_override_list.append(
                1 if getattr(cc, 'straight_sign_flip_override_active', False) else 0
            )
            straight_sign_flip_triggered_list.append(
                1 if getattr(cc, 'straight_sign_flip_triggered', False) else 0
            )
            straight_sign_flip_trigger_error_list.append(
                getattr(cc, 'straight_sign_flip_trigger_error', 0.0) or 0.0
            )
            straight_sign_flip_frames_remaining_list.append(
                int(getattr(cc, 'straight_sign_flip_frames_remaining', 0) or 0)
            )
            is_straight_list.append(1 if getattr(cc, 'is_straight', False) else 0)
            straight_oscillation_rate_list.append(getattr(cc, 'straight_oscillation_rate', 0.0) or 0.0)
            tuned_deadband_list.append(getattr(cc, 'tuned_deadband', 0.0) or 0.0)
            tuned_error_smoothing_alpha_list.append(getattr(cc, 'tuned_error_smoothing_alpha', 0.0) or 0.0)
            # NEW: Control stale data tracking
            using_stale_perception_list.append(cc.using_stale_perception if hasattr(cc, 'using_stale_perception') else False)
            stale_perception_reason_list.append(cc.stale_perception_reason if hasattr(cc, 'stale_perception_reason') and cc.stale_perception_reason else "")
            emergency_stop_list.append(1 if getattr(cc, 'emergency_stop', False) else 0)
            target_speed_raw_list.append(getattr(cc, 'target_speed_raw', 0.0) or 0.0)
            target_speed_post_limits_list.append(getattr(cc, 'target_speed_post_limits', 0.0) or 0.0)
            target_speed_planned_list.append(getattr(cc, 'target_speed_planned', 0.0) or 0.0)
            target_speed_final_list.append(getattr(cc, 'target_speed_final', 0.0) or 0.0)
            target_speed_slew_active_list.append(1 if getattr(cc, 'target_speed_slew_active', False) else 0)
            target_speed_ramp_active_list.append(1 if getattr(cc, 'target_speed_ramp_active', False) else 0)
            launch_throttle_cap_list.append(getattr(cc, 'launch_throttle_cap', 0.0) or 0.0)
            launch_throttle_cap_active_list.append(1 if getattr(cc, 'launch_throttle_cap_active', False) else 0)
            steering_pre_rate_limit_list.append(getattr(cc, 'steering_pre_rate_limit', 0.0) or 0.0)
            steering_post_rate_limit_list.append(getattr(cc, 'steering_post_rate_limit', 0.0) or 0.0)
            steering_post_jerk_limit_list.append(getattr(cc, 'steering_post_jerk_limit', 0.0) or 0.0)
            steering_post_sign_flip_list.append(getattr(cc, 'steering_post_sign_flip', 0.0) or 0.0)
            steering_post_hard_clip_list.append(getattr(cc, 'steering_post_hard_clip', 0.0) or 0.0)
            steering_post_smoothing_list.append(getattr(cc, 'steering_post_smoothing', 0.0) or 0.0)
            steering_rate_limited_active_list.append(1 if getattr(cc, 'steering_rate_limited_active', False) else 0)
            steering_jerk_limited_active_list.append(1 if getattr(cc, 'steering_jerk_limited_active', False) else 0)
            steering_hard_clip_active_list.append(1 if getattr(cc, 'steering_hard_clip_active', False) else 0)
            steering_smoothing_active_list.append(1 if getattr(cc, 'steering_smoothing_active', False) else 0)
            steering_rate_limited_delta_list.append(getattr(cc, 'steering_rate_limited_delta', 0.0) or 0.0)
            steering_jerk_limited_delta_list.append(getattr(cc, 'steering_jerk_limited_delta', 0.0) or 0.0)
            steering_hard_clip_delta_list.append(getattr(cc, 'steering_hard_clip_delta', 0.0) or 0.0)
            steering_smoothing_delta_list.append(getattr(cc, 'steering_smoothing_delta', 0.0) or 0.0)
        
        if timestamps:
            current_size = self.h5_file["control/timestamps"].shape[0]
            new_size = current_size + len(timestamps)
            
            # Resize all control datasets
            for key in ["timestamps", "steering", "throttle", "brake",
                       "accel_feedforward", "brake_feedforward",
                       "longitudinal_accel_capped", "longitudinal_jerk_capped",
                       "steering_before_limits", "throttle_before_limits", "brake_before_limits",
                       "feedforward_steering", "feedback_steering",
                       "pid_integral", "pid_derivative", "pid_error",
                       "lateral_error", "heading_error", "total_error", "total_error_scaled",
                       "straight_sign_flip_override_active",
                       "straight_sign_flip_triggered", "straight_sign_flip_trigger_error",
                       "straight_sign_flip_frames_remaining",
                       "calculated_steering_angle_deg", "raw_steering", 
                       "lateral_correction", "path_curvature_input",
                       "is_straight", "straight_oscillation_rate",
                       "tuned_deadband", "tuned_error_smoothing_alpha",
                       "using_stale_perception", "stale_perception_reason",
                       "emergency_stop", "target_speed_raw",
                       "target_speed_post_limits", "target_speed_planned", "target_speed_final",
                       "target_speed_slew_active", "target_speed_ramp_active",
                       "launch_throttle_cap", "launch_throttle_cap_active",
                       "steering_pre_rate_limit", "steering_post_rate_limit",
                       "steering_post_jerk_limit", "steering_post_sign_flip",
                       "steering_post_hard_clip", "steering_post_smoothing",
                       "steering_rate_limited_active", "steering_jerk_limited_active",
                       "steering_hard_clip_active", "steering_smoothing_active",
                       "steering_rate_limited_delta", "steering_jerk_limited_delta",
                       "steering_hard_clip_delta", "steering_smoothing_delta"]:
                self.h5_file[f"control/{key}"].resize((new_size,))
            
            # Write data
            self.h5_file["control/timestamps"][current_size:] = timestamps
            self.h5_file["control/steering"][current_size:] = steerings
            self.h5_file["control/throttle"][current_size:] = throttles
            self.h5_file["control/brake"][current_size:] = brakes
            self.h5_file["control/accel_feedforward"][current_size:] = accel_feedforward_list
            self.h5_file["control/brake_feedforward"][current_size:] = brake_feedforward_list
            self.h5_file["control/longitudinal_accel_capped"][current_size:] = np.array(accel_capped_list, dtype=np.int8)
            self.h5_file["control/longitudinal_jerk_capped"][current_size:] = np.array(jerk_capped_list, dtype=np.int8)
            self.h5_file["control/steering_before_limits"][current_size:] = steering_before
            self.h5_file["control/throttle_before_limits"][current_size:] = throttle_before
            self.h5_file["control/brake_before_limits"][current_size:] = brake_before
            self.h5_file["control/feedforward_steering"][current_size:] = feedforward_steering_list
            self.h5_file["control/feedback_steering"][current_size:] = feedback_steering_list
            self.h5_file["control/pid_integral"][current_size:] = pid_integrals
            self.h5_file["control/pid_derivative"][current_size:] = pid_derivatives
            self.h5_file["control/pid_error"][current_size:] = pid_errors
            self.h5_file["control/lateral_error"][current_size:] = lateral_errors
            self.h5_file["control/heading_error"][current_size:] = heading_errors
            self.h5_file["control/total_error"][current_size:] = total_errors
            self.h5_file["control/total_error_scaled"][current_size:] = total_error_scaled_list
            self.h5_file["control/straight_sign_flip_override_active"][current_size:] = np.array(
                straight_sign_flip_override_list, dtype=np.int8
            )
            self.h5_file["control/straight_sign_flip_triggered"][current_size:] = np.array(
                straight_sign_flip_triggered_list, dtype=np.int8
            )
            self.h5_file["control/straight_sign_flip_trigger_error"][current_size:] = (
                straight_sign_flip_trigger_error_list
            )
            self.h5_file["control/straight_sign_flip_frames_remaining"][current_size:] = np.array(
                straight_sign_flip_frames_remaining_list, dtype=np.int16
            )
            self.h5_file["control/calculated_steering_angle_deg"][current_size:] = calculated_steering_angles
            self.h5_file["control/raw_steering"][current_size:] = raw_steerings
            self.h5_file["control/lateral_correction"][current_size:] = lateral_corrections
            self.h5_file["control/path_curvature_input"][current_size:] = path_curvature_inputs
            self.h5_file["control/is_straight"][current_size:] = np.array(is_straight_list, dtype=np.int8)
            self.h5_file["control/straight_oscillation_rate"][current_size:] = straight_oscillation_rate_list
            self.h5_file["control/tuned_deadband"][current_size:] = tuned_deadband_list
            self.h5_file["control/tuned_error_smoothing_alpha"][current_size:] = tuned_error_smoothing_alpha_list
            # NEW: Write control stale data
            self.h5_file["control/using_stale_perception"][current_size:] = using_stale_perception_list
            stale_perception_reason_array = np.array(stale_perception_reason_list, dtype=h5py.string_dtype(encoding='utf-8', length=50))
            self.h5_file["control/stale_perception_reason"][current_size:] = stale_perception_reason_array
            self.h5_file["control/emergency_stop"][current_size:] = emergency_stop_list
            self.h5_file["control/target_speed_raw"][current_size:] = target_speed_raw_list
            self.h5_file["control/target_speed_post_limits"][current_size:] = target_speed_post_limits_list
            self.h5_file["control/target_speed_planned"][current_size:] = target_speed_planned_list
            self.h5_file["control/target_speed_final"][current_size:] = target_speed_final_list
            self.h5_file["control/target_speed_slew_active"][current_size:] = np.array(target_speed_slew_active_list, dtype=np.int8)
            self.h5_file["control/target_speed_ramp_active"][current_size:] = np.array(target_speed_ramp_active_list, dtype=np.int8)
            self.h5_file["control/launch_throttle_cap"][current_size:] = launch_throttle_cap_list
            self.h5_file["control/launch_throttle_cap_active"][current_size:] = np.array(launch_throttle_cap_active_list, dtype=np.int8)
            self.h5_file["control/steering_pre_rate_limit"][current_size:] = steering_pre_rate_limit_list
            self.h5_file["control/steering_post_rate_limit"][current_size:] = steering_post_rate_limit_list
            self.h5_file["control/steering_post_jerk_limit"][current_size:] = steering_post_jerk_limit_list
            self.h5_file["control/steering_post_sign_flip"][current_size:] = steering_post_sign_flip_list
            self.h5_file["control/steering_post_hard_clip"][current_size:] = steering_post_hard_clip_list
            self.h5_file["control/steering_post_smoothing"][current_size:] = steering_post_smoothing_list
            self.h5_file["control/steering_rate_limited_active"][current_size:] = np.array(steering_rate_limited_active_list, dtype=np.int8)
            self.h5_file["control/steering_jerk_limited_active"][current_size:] = np.array(steering_jerk_limited_active_list, dtype=np.int8)
            self.h5_file["control/steering_hard_clip_active"][current_size:] = np.array(steering_hard_clip_active_list, dtype=np.int8)
            self.h5_file["control/steering_smoothing_active"][current_size:] = np.array(steering_smoothing_active_list, dtype=np.int8)
            self.h5_file["control/steering_rate_limited_delta"][current_size:] = steering_rate_limited_delta_list
            self.h5_file["control/steering_jerk_limited_delta"][current_size:] = steering_jerk_limited_delta_list
            self.h5_file["control/steering_hard_clip_delta"][current_size:] = steering_hard_clip_delta_list
            self.h5_file["control/steering_smoothing_delta"][current_size:] = steering_smoothing_delta_list
    
    def _write_perception_outputs(self, frames: List[RecordingFrame]):
        """Write perception outputs to HDF5."""
        timestamps = []
        confidences = []
        detection_methods = []
        num_lanes = []
        left_lane_line_x = []
        right_lane_line_x = []
        lane_coefficients_list = []
        # NEW: Stale data tracking
        using_stale_data_list = []
        stale_reason_list = []
        reject_reason_list = []
        left_jump_magnitude_list = []
        right_jump_magnitude_list = []
        jump_threshold_list = []
        # NEW: Diagnostic fields for perception instability
        actual_detected_left_lane_x_list = []
        actual_detected_right_lane_x_list = []
        instability_width_change_list = []
        instability_center_shift_list = []
        # NEW: Perception health tracking
        consecutive_bad_frames_list = []
        health_score_list = []
        health_status_list = []
        bad_events_list = []
        bad_events_recent_list = []
        clamp_events_list = []
        timestamp_frozen_list = []
        # NEW: Points used for polynomial fitting
        fit_points_left_list = []
        fit_points_right_list = []
        segmentation_mask_png_list = []
        
        for frame in frames:
            po = frame.perception_output
            timestamps.append(po.timestamp)
            confidences.append(po.confidence if po.confidence is not None else 0.0)
            detection_methods.append(po.detection_method if po.detection_method else "unknown")
            num_lanes.append(po.num_lanes_detected if po.num_lanes_detected is not None else 0)
            left_lane_line_x.append(po.left_lane_line_x if po.left_lane_line_x is not None else (po.left_lane_x if hasattr(po, 'left_lane_x') and po.left_lane_x is not None else 0.0))  # Backward compatibility
            right_lane_line_x.append(po.right_lane_line_x if po.right_lane_line_x is not None else (po.right_lane_x if hasattr(po, 'right_lane_x') and po.right_lane_x is not None else 0.0))  # Backward compatibility
            # NEW: Stale data tracking
            using_stale_data_list.append(po.using_stale_data if hasattr(po, 'using_stale_data') else False)
            stale_reason_list.append(po.stale_data_reason if hasattr(po, 'stale_data_reason') and po.stale_data_reason else "")
            reject_reason_list.append(
                po.reject_reason if hasattr(po, 'reject_reason') and po.reject_reason else ""
            )
            left_jump_magnitude_list.append(po.left_jump_magnitude if hasattr(po, 'left_jump_magnitude') and po.left_jump_magnitude is not None else 0.0)
            right_jump_magnitude_list.append(po.right_jump_magnitude if hasattr(po, 'right_jump_magnitude') and po.right_jump_magnitude is not None else 0.0)
            jump_threshold_list.append(po.jump_threshold if hasattr(po, 'jump_threshold') and po.jump_threshold is not None else 0.0)
            # NEW: Diagnostic fields for perception instability
            actual_detected_left_lane_x_list.append(po.actual_detected_left_lane_x if hasattr(po, 'actual_detected_left_lane_x') and po.actual_detected_left_lane_x is not None else 0.0)
            actual_detected_right_lane_x_list.append(po.actual_detected_right_lane_x if hasattr(po, 'actual_detected_right_lane_x') and po.actual_detected_right_lane_x is not None else 0.0)
            instability_width_change_list.append(po.instability_width_change if hasattr(po, 'instability_width_change') and po.instability_width_change is not None else 0.0)
            instability_center_shift_list.append(po.instability_center_shift if hasattr(po, 'instability_center_shift') and po.instability_center_shift is not None else 0.0)
            # NEW: Perception health tracking
            consecutive_bad_frames_list.append(po.consecutive_bad_detection_frames if hasattr(po, 'consecutive_bad_detection_frames') else 0)
            health_score_list.append(po.perception_health_score if hasattr(po, 'perception_health_score') else 1.0)
            health_status_list.append(po.perception_health_status if hasattr(po, 'perception_health_status') else "healthy")
            bad_events_list.append(
                ",".join(po.perception_bad_events)
                if hasattr(po, 'perception_bad_events') and po.perception_bad_events
                else ""
            )
            bad_events_recent_list.append(
                ",".join(po.perception_bad_events_recent)
                if hasattr(po, 'perception_bad_events_recent') and po.perception_bad_events_recent
                else ""
            )
            clamp_events_list.append(
                ",".join(po.perception_clamp_events)
                if hasattr(po, 'perception_clamp_events') and po.perception_clamp_events
                else ""
            )
            timestamp_frozen_list.append(
                bool(getattr(po, 'perception_timestamp_frozen', False))
            )
            # NEW: Points used for polynomial fitting
            # Store as JSON strings since HDF5 doesn't handle variable-length arrays well
            fit_points_left_json = ""
            fit_points_right_json = ""
            if hasattr(po, 'fit_points_left') and po.fit_points_left is not None:
                try:
                    import json
                    fit_points_left_value = (
                        po.fit_points_left.tolist()
                        if hasattr(po.fit_points_left, 'tolist')
                        else po.fit_points_left
                    )
                    fit_points_left_json = json.dumps(fit_points_left_value)
                except Exception:
                    pass
            if hasattr(po, 'fit_points_right') and po.fit_points_right is not None:
                try:
                    import json
                    fit_points_right_value = (
                        po.fit_points_right.tolist()
                        if hasattr(po.fit_points_right, 'tolist')
                        else po.fit_points_right
                    )
                    fit_points_right_json = json.dumps(fit_points_right_value)
                except Exception:
                    pass
            fit_points_left_list.append(fit_points_left_json)
            fit_points_right_list.append(fit_points_right_json)
            if hasattr(po, 'segmentation_mask_png') and po.segmentation_mask_png is not None:
                segmentation_mask_png_list.append(
                    np.frombuffer(po.segmentation_mask_png, dtype=np.uint8)
                )
            else:
                segmentation_mask_png_list.append(np.array([], dtype=np.uint8))
            
            # Store lane coefficients (flattened array of all coefficients)
            # Format: [coeff0_lane0, coeff1_lane0, coeff2_lane0, coeff0_lane1, ...]
            if po.lane_line_coefficients is not None and len(po.lane_line_coefficients) > 0:
                # Flatten all coefficients into a single array
                coeffs_flat = np.concatenate([c.flatten() for c in po.lane_line_coefficients if c is not None])
                lane_coefficients_list.append(coeffs_flat.astype(np.float32))
            else:
                # Use empty array with proper dtype for variable-length dataset
                lane_coefficients_list.append(np.array([], dtype=np.float32))
        
        if timestamps:
            current_size = self.h5_file["perception/timestamps"].shape[0]
            new_size = current_size + len(timestamps)
            
            # Resize all perception datasets
            self.h5_file["perception/timestamps"].resize((new_size,))
            self.h5_file["perception/confidence"].resize((new_size,))
            self.h5_file["perception/detection_method"].resize((new_size,))
            self.h5_file["perception/num_lanes_detected"].resize((new_size,))
            self.h5_file["perception/left_lane_line_x"].resize((new_size,))
            self.h5_file["perception/right_lane_line_x"].resize((new_size,))
            # NEW: Resize stale data datasets
            self.h5_file["perception/using_stale_data"].resize((new_size,))
            self.h5_file["perception/stale_reason"].resize((new_size,))
            self.h5_file["perception/reject_reason"].resize((new_size,))
            self.h5_file["perception/left_jump_magnitude"].resize((new_size,))
            self.h5_file["perception/right_jump_magnitude"].resize((new_size,))
            self.h5_file["perception/jump_threshold"].resize((new_size,))
            # NEW: Resize health monitoring datasets
            self.h5_file["perception/consecutive_bad_detection_frames"].resize((new_size,))
            self.h5_file["perception/perception_health_score"].resize((new_size,))
            self.h5_file["perception/perception_health_status"].resize((new_size,))
            self.h5_file["perception/perception_bad_events"].resize((new_size,))
            self.h5_file["perception/perception_bad_events_recent"].resize((new_size,))
            self.h5_file["perception/perception_clamp_events"].resize((new_size,))
            self.h5_file["perception/perception_timestamp_frozen"].resize((new_size,))
            # NEW: Resize instability diagnostic datasets
            self.h5_file["perception/actual_detected_left_lane_x"].resize((new_size,))
            self.h5_file["perception/actual_detected_right_lane_x"].resize((new_size,))
            self.h5_file["perception/instability_width_change"].resize((new_size,))
            self.h5_file["perception/instability_center_shift"].resize((new_size,))
            # NEW: Resize fit_points datasets
            self.h5_file["perception/fit_points_left"].resize((new_size,))
            self.h5_file["perception/fit_points_right"].resize((new_size,))
            # Write data
            self.h5_file["perception/timestamps"][current_size:] = timestamps
            self.h5_file["perception/confidence"][current_size:] = confidences
            # Convert string list to numpy array for string dtype
            detection_methods_array = np.array(detection_methods, dtype=h5py.string_dtype(encoding='utf-8'))
            self.h5_file["perception/detection_method"][current_size:] = detection_methods_array
            self.h5_file["perception/num_lanes_detected"][current_size:] = num_lanes
            self.h5_file["perception/left_lane_line_x"][current_size:] = left_lane_line_x
            self.h5_file["perception/right_lane_line_x"][current_size:] = right_lane_line_x
            # NEW: Write stale data
            self.h5_file["perception/using_stale_data"][current_size:] = using_stale_data_list
            stale_reason_array = np.array(stale_reason_list, dtype=h5py.string_dtype(encoding='utf-8', length=50))
            self.h5_file["perception/stale_reason"][current_size:] = stale_reason_array
            reject_reason_array = np.array(
                reject_reason_list, dtype=h5py.string_dtype(encoding='utf-8', length=100)
            )
            self.h5_file["perception/reject_reason"][current_size:] = reject_reason_array
            self.h5_file["perception/left_jump_magnitude"][current_size:] = left_jump_magnitude_list
            self.h5_file["perception/right_jump_magnitude"][current_size:] = right_jump_magnitude_list
            self.h5_file["perception/jump_threshold"][current_size:] = jump_threshold_list
            # NEW: Write instability diagnostic data
            self.h5_file["perception/actual_detected_left_lane_x"][current_size:] = actual_detected_left_lane_x_list
            self.h5_file["perception/actual_detected_right_lane_x"][current_size:] = actual_detected_right_lane_x_list
            self.h5_file["perception/instability_width_change"][current_size:] = instability_width_change_list
            self.h5_file["perception/instability_center_shift"][current_size:] = instability_center_shift_list
            # NEW: Write health monitoring data
            self.h5_file["perception/consecutive_bad_detection_frames"][current_size:] = consecutive_bad_frames_list
            self.h5_file["perception/perception_health_score"][current_size:] = health_score_list
            health_status_array = np.array(health_status_list, dtype=h5py.string_dtype(encoding='utf-8', length=20))
            self.h5_file["perception/perception_health_status"][current_size:] = health_status_array
            bad_events_array = np.array(bad_events_list, dtype=h5py.string_dtype(encoding='utf-8', length=100))
            self.h5_file["perception/perception_bad_events"][current_size:] = bad_events_array
            bad_events_recent_array = np.array(
                bad_events_recent_list, dtype=h5py.string_dtype(encoding='utf-8', length=100)
            )
            self.h5_file["perception/perception_bad_events_recent"][current_size:] = bad_events_recent_array
            clamp_events_array = np.array(
                clamp_events_list, dtype=h5py.string_dtype(encoding='utf-8', length=100)
            )
            self.h5_file["perception/perception_clamp_events"][current_size:] = clamp_events_array
            self.h5_file["perception/perception_timestamp_frozen"][current_size:] = np.array(
                timestamp_frozen_list, dtype=np.bool_
            )
            # NEW: Write fit_points (as JSON strings)
            fit_points_left_array = np.array(fit_points_left_list, dtype=h5py.string_dtype(encoding='utf-8', length=10000))
            fit_points_right_array = np.array(fit_points_right_list, dtype=h5py.string_dtype(encoding='utf-8', length=10000))
            self.h5_file["perception/fit_points_left"][current_size:] = fit_points_left_array
            self.h5_file["perception/fit_points_right"][current_size:] = fit_points_right_array
            self.h5_file["perception/segmentation_mask_png"].resize((new_size,))
            for i, mask_arr in enumerate(segmentation_mask_png_list):
                idx = current_size + i
                self.h5_file["perception/segmentation_mask_png"][idx] = mask_arr
            
            # Write variable-length arrays (handle empty arrays properly)
            # CRITICAL: Always resize and write if we have data, even if some arrays are empty
            # vlen dtype can handle empty arrays, but we must resize the dataset first
            if lane_coefficients_list:
                # Resize dataset to accommodate all arrays (including empty ones)
                self.h5_file["perception/lane_line_coefficients"].resize((new_size,))
                try:
                    # Write each array individually (vlen dtype handles variable-length arrays, including empty ones)
                    for i, coeffs in enumerate(lane_coefficients_list):
                        idx = current_size + i
                        # Ensure it's a numpy array
                        if isinstance(coeffs, np.ndarray):
                            # vlen dtype can handle empty arrays - just write them directly
                            self.h5_file["perception/lane_line_coefficients"][idx] = coeffs
                        else:
                            coeffs_array = np.array(coeffs, dtype=np.float32)
                            self.h5_file["perception/lane_line_coefficients"][idx] = coeffs_array
                except Exception as e:
                    # If writing fails, log warning but continue (non-critical)
                    logger.warning(f"Failed to write lane_line_coefficients (non-critical): {e}. "
                                 f"Attempted to write {len(lane_coefficients_list)} arrays. "
                                 f"Error type: {type(e).__name__}")
                    # Skip writing coefficients if it fails (non-critical data)
    
    def _write_trajectory_outputs(self, frames: List[RecordingFrame]):
        """Write trajectory outputs to HDF5."""
        # Write timestamps
        timestamps = []
        for frame in frames:
            if frame.trajectory_output:
                timestamps.append(frame.trajectory_output.timestamp)
        
        if timestamps:
            current_size = self.h5_file["trajectory/timestamps"].shape[0]
            new_size = current_size + len(timestamps)
            self.h5_file["trajectory/timestamps"].resize((new_size,))
            self.h5_file["trajectory/timestamps"][current_size:] = timestamps
        
        # Write reference points (smoothed and raw)
        ref_points = []
        ref_points_raw = []
        ref_methods = []  # NEW: Track which method was used
        perception_centers = []  # NEW: Track perception center
        diag_available = []
        diag_generated_by_fallback = []
        diag_points_generated = []
        diag_x_clip_count = []
        diag_pre_y0 = []
        diag_pre_y1 = []
        diag_pre_y2 = []
        diag_post_y0 = []
        diag_post_y1 = []
        diag_post_y2 = []
        diag_used_provided_distance0 = []
        diag_used_provided_distance1 = []
        diag_used_provided_distance2 = []
        diag_post_minus_pre_y0 = []
        diag_post_minus_pre_y1 = []
        diag_post_minus_pre_y2 = []
        diag_preclip_x0 = []
        diag_preclip_x1 = []
        diag_preclip_x2 = []
        diag_preclip_x_abs_max = []
        diag_preclip_x_abs_p95 = []
        diag_preclip_mean_12_20m_lane_source_x = []
        diag_preclip_mean_12_20m_distance_scale_delta_x = []
        diag_preclip_mean_12_20m_camera_offset_delta_x = []
        diag_preclip_abs_mean_12_20m_lane_source_x = []
        diag_preclip_abs_mean_12_20m_distance_scale_delta_x = []
        diag_preclip_abs_mean_12_20m_camera_offset_delta_x = []
        diag_heading_zero_gate_active = []
        diag_small_heading_gate_active = []
        diag_multi_lookahead_active = []
        diag_smoothing_jump_reject = []
        diag_ref_x_rate_limit_active = []
        diag_raw_ref_x = []
        diag_smoothed_ref_x = []
        diag_ref_x_suppression_abs = []
        diag_raw_ref_heading = []
        diag_smoothed_ref_heading = []
        diag_heading_suppression_abs = []
        diag_smoothing_alpha = []
        diag_smoothing_alpha_x = []
        diag_multi_lookahead_heading_base = []
        diag_multi_lookahead_heading_far = []
        diag_multi_lookahead_heading_blended = []
        diag_multi_lookahead_blend_alpha = []
        diag_dynamic_effective_horizon_m = []
        diag_dynamic_effective_horizon_base_m = []
        diag_dynamic_effective_horizon_min_m = []
        diag_dynamic_effective_horizon_max_m = []
        diag_dynamic_effective_horizon_speed_scale = []
        diag_dynamic_effective_horizon_curvature_scale = []
        diag_dynamic_effective_horizon_confidence_scale = []
        diag_dynamic_effective_horizon_final_scale = []
        diag_dynamic_effective_horizon_speed_mps = []
        diag_dynamic_effective_horizon_curvature_abs = []
        diag_dynamic_effective_horizon_confidence_used = []
        diag_dynamic_effective_horizon_limiter_code = []
        diag_dynamic_effective_horizon_applied = []
        diag_preclip_abs_mean_0_8m = []
        diag_preclip_abs_mean_8_12m = []
        diag_preclip_abs_mean_12_20m = []
        diag_postclip_x0 = []
        diag_postclip_x1 = []
        diag_postclip_x2 = []
        diag_postclip_abs_mean_0_8m = []
        diag_postclip_abs_mean_8_12m = []
        diag_postclip_abs_mean_12_20m = []
        diag_postclip_near_clip_frac_12_20m = []
        diag_first_segment_y0_gt_y1_pre = []
        diag_first_segment_y0_gt_y1_post = []
        diag_inversion_introduced_after_conversion = []
        for frame in frames:
            if frame.trajectory_output and frame.trajectory_output.reference_point:
                rp = frame.trajectory_output.reference_point
                ref_points.append([
                    rp.get('x', 0.0),
                    rp.get('y', 0.0),
                    rp.get('heading', 0.0),
                    rp.get('velocity', 0.0)
                ])
                # Raw (before smoothing) values
                ref_points_raw.append([
                    rp.get('raw_x', rp.get('x', 0.0)),
                    rp.get('raw_y', rp.get('y', 0.0)),
                    rp.get('raw_heading', rp.get('heading', 0.0))
                ])
                # NEW: Extract method and perception center
                method = frame.trajectory_output.reference_point_method
                if method is None:
                    method = rp.get('method', 'unknown')  # Fallback to dict value
                ref_methods.append(method if method else 'unknown')
                perception_center = frame.trajectory_output.perception_center_x
                if perception_center is None:
                    perception_center = rp.get('perception_center_x', 0.0)  # Fallback to dict value
                perception_centers.append(perception_center if perception_center is not None else 0.0)
                to = frame.trajectory_output
                diag_available.append(float(to.diag_available if to.diag_available is not None else np.nan))
                diag_generated_by_fallback.append(float(to.diag_generated_by_fallback if to.diag_generated_by_fallback is not None else np.nan))
                diag_points_generated.append(float(to.diag_points_generated if to.diag_points_generated is not None else np.nan))
                diag_x_clip_count.append(float(to.diag_x_clip_count if to.diag_x_clip_count is not None else np.nan))
                diag_pre_y0.append(float(to.diag_pre_y0 if to.diag_pre_y0 is not None else np.nan))
                diag_pre_y1.append(float(to.diag_pre_y1 if to.diag_pre_y1 is not None else np.nan))
                diag_pre_y2.append(float(to.diag_pre_y2 if to.diag_pre_y2 is not None else np.nan))
                diag_post_y0.append(float(to.diag_post_y0 if to.diag_post_y0 is not None else np.nan))
                diag_post_y1.append(float(to.diag_post_y1 if to.diag_post_y1 is not None else np.nan))
                diag_post_y2.append(float(to.diag_post_y2 if to.diag_post_y2 is not None else np.nan))
                diag_used_provided_distance0.append(float(to.diag_used_provided_distance0 if to.diag_used_provided_distance0 is not None else np.nan))
                diag_used_provided_distance1.append(float(to.diag_used_provided_distance1 if to.diag_used_provided_distance1 is not None else np.nan))
                diag_used_provided_distance2.append(float(to.diag_used_provided_distance2 if to.diag_used_provided_distance2 is not None else np.nan))
                diag_post_minus_pre_y0.append(float(to.diag_post_minus_pre_y0 if to.diag_post_minus_pre_y0 is not None else np.nan))
                diag_post_minus_pre_y1.append(float(to.diag_post_minus_pre_y1 if to.diag_post_minus_pre_y1 is not None else np.nan))
                diag_post_minus_pre_y2.append(float(to.diag_post_minus_pre_y2 if to.diag_post_minus_pre_y2 is not None else np.nan))
                diag_preclip_x0.append(float(to.diag_preclip_x0 if to.diag_preclip_x0 is not None else np.nan))
                diag_preclip_x1.append(float(to.diag_preclip_x1 if to.diag_preclip_x1 is not None else np.nan))
                diag_preclip_x2.append(float(to.diag_preclip_x2 if to.diag_preclip_x2 is not None else np.nan))
                diag_preclip_x_abs_max.append(float(to.diag_preclip_x_abs_max if to.diag_preclip_x_abs_max is not None else np.nan))
                diag_preclip_x_abs_p95.append(float(to.diag_preclip_x_abs_p95 if to.diag_preclip_x_abs_p95 is not None else np.nan))
                diag_preclip_mean_12_20m_lane_source_x.append(float(to.diag_preclip_mean_12_20m_lane_source_x if to.diag_preclip_mean_12_20m_lane_source_x is not None else np.nan))
                diag_preclip_mean_12_20m_distance_scale_delta_x.append(float(to.diag_preclip_mean_12_20m_distance_scale_delta_x if to.diag_preclip_mean_12_20m_distance_scale_delta_x is not None else np.nan))
                diag_preclip_mean_12_20m_camera_offset_delta_x.append(float(to.diag_preclip_mean_12_20m_camera_offset_delta_x if to.diag_preclip_mean_12_20m_camera_offset_delta_x is not None else np.nan))
                diag_preclip_abs_mean_12_20m_lane_source_x.append(float(to.diag_preclip_abs_mean_12_20m_lane_source_x if to.diag_preclip_abs_mean_12_20m_lane_source_x is not None else np.nan))
                diag_preclip_abs_mean_12_20m_distance_scale_delta_x.append(float(to.diag_preclip_abs_mean_12_20m_distance_scale_delta_x if to.diag_preclip_abs_mean_12_20m_distance_scale_delta_x is not None else np.nan))
                diag_preclip_abs_mean_12_20m_camera_offset_delta_x.append(float(to.diag_preclip_abs_mean_12_20m_camera_offset_delta_x if to.diag_preclip_abs_mean_12_20m_camera_offset_delta_x is not None else np.nan))
                diag_heading_zero_gate_active.append(float(to.diag_heading_zero_gate_active if to.diag_heading_zero_gate_active is not None else np.nan))
                diag_small_heading_gate_active.append(float(to.diag_small_heading_gate_active if to.diag_small_heading_gate_active is not None else np.nan))
                diag_multi_lookahead_active.append(float(to.diag_multi_lookahead_active if to.diag_multi_lookahead_active is not None else np.nan))
                diag_smoothing_jump_reject.append(float(to.diag_smoothing_jump_reject if to.diag_smoothing_jump_reject is not None else np.nan))
                diag_ref_x_rate_limit_active.append(float(to.diag_ref_x_rate_limit_active if to.diag_ref_x_rate_limit_active is not None else np.nan))
                diag_raw_ref_x.append(float(to.diag_raw_ref_x if to.diag_raw_ref_x is not None else np.nan))
                diag_smoothed_ref_x.append(float(to.diag_smoothed_ref_x if to.diag_smoothed_ref_x is not None else np.nan))
                diag_ref_x_suppression_abs.append(float(to.diag_ref_x_suppression_abs if to.diag_ref_x_suppression_abs is not None else np.nan))
                diag_raw_ref_heading.append(float(to.diag_raw_ref_heading if to.diag_raw_ref_heading is not None else np.nan))
                diag_smoothed_ref_heading.append(float(to.diag_smoothed_ref_heading if to.diag_smoothed_ref_heading is not None else np.nan))
                diag_heading_suppression_abs.append(float(to.diag_heading_suppression_abs if to.diag_heading_suppression_abs is not None else np.nan))
                diag_smoothing_alpha.append(float(to.diag_smoothing_alpha if to.diag_smoothing_alpha is not None else np.nan))
                diag_smoothing_alpha_x.append(float(to.diag_smoothing_alpha_x if to.diag_smoothing_alpha_x is not None else np.nan))
                diag_multi_lookahead_heading_base.append(float(to.diag_multi_lookahead_heading_base if to.diag_multi_lookahead_heading_base is not None else np.nan))
                diag_multi_lookahead_heading_far.append(float(to.diag_multi_lookahead_heading_far if to.diag_multi_lookahead_heading_far is not None else np.nan))
                diag_multi_lookahead_heading_blended.append(float(to.diag_multi_lookahead_heading_blended if to.diag_multi_lookahead_heading_blended is not None else np.nan))
                diag_multi_lookahead_blend_alpha.append(float(to.diag_multi_lookahead_blend_alpha if to.diag_multi_lookahead_blend_alpha is not None else np.nan))
                diag_dynamic_effective_horizon_m.append(float(to.diag_dynamic_effective_horizon_m if to.diag_dynamic_effective_horizon_m is not None else np.nan))
                diag_dynamic_effective_horizon_base_m.append(float(to.diag_dynamic_effective_horizon_base_m if to.diag_dynamic_effective_horizon_base_m is not None else np.nan))
                diag_dynamic_effective_horizon_min_m.append(float(to.diag_dynamic_effective_horizon_min_m if to.diag_dynamic_effective_horizon_min_m is not None else np.nan))
                diag_dynamic_effective_horizon_max_m.append(float(to.diag_dynamic_effective_horizon_max_m if to.diag_dynamic_effective_horizon_max_m is not None else np.nan))
                diag_dynamic_effective_horizon_speed_scale.append(float(to.diag_dynamic_effective_horizon_speed_scale if to.diag_dynamic_effective_horizon_speed_scale is not None else np.nan))
                diag_dynamic_effective_horizon_curvature_scale.append(float(to.diag_dynamic_effective_horizon_curvature_scale if to.diag_dynamic_effective_horizon_curvature_scale is not None else np.nan))
                diag_dynamic_effective_horizon_confidence_scale.append(float(to.diag_dynamic_effective_horizon_confidence_scale if to.diag_dynamic_effective_horizon_confidence_scale is not None else np.nan))
                diag_dynamic_effective_horizon_final_scale.append(float(to.diag_dynamic_effective_horizon_final_scale if to.diag_dynamic_effective_horizon_final_scale is not None else np.nan))
                diag_dynamic_effective_horizon_speed_mps.append(float(to.diag_dynamic_effective_horizon_speed_mps if to.diag_dynamic_effective_horizon_speed_mps is not None else np.nan))
                diag_dynamic_effective_horizon_curvature_abs.append(float(to.diag_dynamic_effective_horizon_curvature_abs if to.diag_dynamic_effective_horizon_curvature_abs is not None else np.nan))
                diag_dynamic_effective_horizon_confidence_used.append(float(to.diag_dynamic_effective_horizon_confidence_used if to.diag_dynamic_effective_horizon_confidence_used is not None else np.nan))
                diag_dynamic_effective_horizon_limiter_code.append(float(to.diag_dynamic_effective_horizon_limiter_code if to.diag_dynamic_effective_horizon_limiter_code is not None else np.nan))
                diag_dynamic_effective_horizon_applied.append(float(to.diag_dynamic_effective_horizon_applied if to.diag_dynamic_effective_horizon_applied is not None else np.nan))
                diag_preclip_abs_mean_0_8m.append(float(to.diag_preclip_abs_mean_0_8m if to.diag_preclip_abs_mean_0_8m is not None else np.nan))
                diag_preclip_abs_mean_8_12m.append(float(to.diag_preclip_abs_mean_8_12m if to.diag_preclip_abs_mean_8_12m is not None else np.nan))
                diag_preclip_abs_mean_12_20m.append(float(to.diag_preclip_abs_mean_12_20m if to.diag_preclip_abs_mean_12_20m is not None else np.nan))
                diag_postclip_x0.append(float(to.diag_postclip_x0 if to.diag_postclip_x0 is not None else np.nan))
                diag_postclip_x1.append(float(to.diag_postclip_x1 if to.diag_postclip_x1 is not None else np.nan))
                diag_postclip_x2.append(float(to.diag_postclip_x2 if to.diag_postclip_x2 is not None else np.nan))
                diag_postclip_abs_mean_0_8m.append(float(to.diag_postclip_abs_mean_0_8m if to.diag_postclip_abs_mean_0_8m is not None else np.nan))
                diag_postclip_abs_mean_8_12m.append(float(to.diag_postclip_abs_mean_8_12m if to.diag_postclip_abs_mean_8_12m is not None else np.nan))
                diag_postclip_abs_mean_12_20m.append(float(to.diag_postclip_abs_mean_12_20m if to.diag_postclip_abs_mean_12_20m is not None else np.nan))
                diag_postclip_near_clip_frac_12_20m.append(float(to.diag_postclip_near_clip_frac_12_20m if to.diag_postclip_near_clip_frac_12_20m is not None else np.nan))
                diag_first_segment_y0_gt_y1_pre.append(float(to.diag_first_segment_y0_gt_y1_pre if to.diag_first_segment_y0_gt_y1_pre is not None else np.nan))
                diag_first_segment_y0_gt_y1_post.append(float(to.diag_first_segment_y0_gt_y1_post if to.diag_first_segment_y0_gt_y1_post is not None else np.nan))
                diag_inversion_introduced_after_conversion.append(float(to.diag_inversion_introduced_after_conversion if to.diag_inversion_introduced_after_conversion is not None else np.nan))
            else:
                # No reference point for this frame - append zeros
                ref_points.append([0.0, 0.0, 0.0, 0.0])
                ref_points_raw.append([0.0, 0.0, 0.0])
                ref_methods.append('none')
                perception_centers.append(0.0)
                to = frame.trajectory_output
                diag_available.append(float(to.diag_available if (to and to.diag_available is not None) else np.nan))
                diag_generated_by_fallback.append(float(to.diag_generated_by_fallback if (to and to.diag_generated_by_fallback is not None) else np.nan))
                diag_points_generated.append(float(to.diag_points_generated if (to and to.diag_points_generated is not None) else np.nan))
                diag_x_clip_count.append(float(to.diag_x_clip_count if (to and to.diag_x_clip_count is not None) else np.nan))
                diag_pre_y0.append(float(to.diag_pre_y0 if (to and to.diag_pre_y0 is not None) else np.nan))
                diag_pre_y1.append(float(to.diag_pre_y1 if (to and to.diag_pre_y1 is not None) else np.nan))
                diag_pre_y2.append(float(to.diag_pre_y2 if (to and to.diag_pre_y2 is not None) else np.nan))
                diag_post_y0.append(float(to.diag_post_y0 if (to and to.diag_post_y0 is not None) else np.nan))
                diag_post_y1.append(float(to.diag_post_y1 if (to and to.diag_post_y1 is not None) else np.nan))
                diag_post_y2.append(float(to.diag_post_y2 if (to and to.diag_post_y2 is not None) else np.nan))
                diag_used_provided_distance0.append(float(to.diag_used_provided_distance0 if (to and to.diag_used_provided_distance0 is not None) else np.nan))
                diag_used_provided_distance1.append(float(to.diag_used_provided_distance1 if (to and to.diag_used_provided_distance1 is not None) else np.nan))
                diag_used_provided_distance2.append(float(to.diag_used_provided_distance2 if (to and to.diag_used_provided_distance2 is not None) else np.nan))
                diag_post_minus_pre_y0.append(float(to.diag_post_minus_pre_y0 if (to and to.diag_post_minus_pre_y0 is not None) else np.nan))
                diag_post_minus_pre_y1.append(float(to.diag_post_minus_pre_y1 if (to and to.diag_post_minus_pre_y1 is not None) else np.nan))
                diag_post_minus_pre_y2.append(float(to.diag_post_minus_pre_y2 if (to and to.diag_post_minus_pre_y2 is not None) else np.nan))
                diag_preclip_x0.append(float(to.diag_preclip_x0 if (to and to.diag_preclip_x0 is not None) else np.nan))
                diag_preclip_x1.append(float(to.diag_preclip_x1 if (to and to.diag_preclip_x1 is not None) else np.nan))
                diag_preclip_x2.append(float(to.diag_preclip_x2 if (to and to.diag_preclip_x2 is not None) else np.nan))
                diag_preclip_x_abs_max.append(float(to.diag_preclip_x_abs_max if (to and to.diag_preclip_x_abs_max is not None) else np.nan))
                diag_preclip_x_abs_p95.append(float(to.diag_preclip_x_abs_p95 if (to and to.diag_preclip_x_abs_p95 is not None) else np.nan))
                diag_preclip_mean_12_20m_lane_source_x.append(float(to.diag_preclip_mean_12_20m_lane_source_x if (to and to.diag_preclip_mean_12_20m_lane_source_x is not None) else np.nan))
                diag_preclip_mean_12_20m_distance_scale_delta_x.append(float(to.diag_preclip_mean_12_20m_distance_scale_delta_x if (to and to.diag_preclip_mean_12_20m_distance_scale_delta_x is not None) else np.nan))
                diag_preclip_mean_12_20m_camera_offset_delta_x.append(float(to.diag_preclip_mean_12_20m_camera_offset_delta_x if (to and to.diag_preclip_mean_12_20m_camera_offset_delta_x is not None) else np.nan))
                diag_preclip_abs_mean_12_20m_lane_source_x.append(float(to.diag_preclip_abs_mean_12_20m_lane_source_x if (to and to.diag_preclip_abs_mean_12_20m_lane_source_x is not None) else np.nan))
                diag_preclip_abs_mean_12_20m_distance_scale_delta_x.append(float(to.diag_preclip_abs_mean_12_20m_distance_scale_delta_x if (to and to.diag_preclip_abs_mean_12_20m_distance_scale_delta_x is not None) else np.nan))
                diag_preclip_abs_mean_12_20m_camera_offset_delta_x.append(float(to.diag_preclip_abs_mean_12_20m_camera_offset_delta_x if (to and to.diag_preclip_abs_mean_12_20m_camera_offset_delta_x is not None) else np.nan))
                diag_heading_zero_gate_active.append(float(to.diag_heading_zero_gate_active if (to and to.diag_heading_zero_gate_active is not None) else np.nan))
                diag_small_heading_gate_active.append(float(to.diag_small_heading_gate_active if (to and to.diag_small_heading_gate_active is not None) else np.nan))
                diag_multi_lookahead_active.append(float(to.diag_multi_lookahead_active if (to and to.diag_multi_lookahead_active is not None) else np.nan))
                diag_smoothing_jump_reject.append(float(to.diag_smoothing_jump_reject if (to and to.diag_smoothing_jump_reject is not None) else np.nan))
                diag_ref_x_rate_limit_active.append(float(to.diag_ref_x_rate_limit_active if (to and to.diag_ref_x_rate_limit_active is not None) else np.nan))
                diag_raw_ref_x.append(float(to.diag_raw_ref_x if (to and to.diag_raw_ref_x is not None) else np.nan))
                diag_smoothed_ref_x.append(float(to.diag_smoothed_ref_x if (to and to.diag_smoothed_ref_x is not None) else np.nan))
                diag_ref_x_suppression_abs.append(float(to.diag_ref_x_suppression_abs if (to and to.diag_ref_x_suppression_abs is not None) else np.nan))
                diag_raw_ref_heading.append(float(to.diag_raw_ref_heading if (to and to.diag_raw_ref_heading is not None) else np.nan))
                diag_smoothed_ref_heading.append(float(to.diag_smoothed_ref_heading if (to and to.diag_smoothed_ref_heading is not None) else np.nan))
                diag_heading_suppression_abs.append(float(to.diag_heading_suppression_abs if (to and to.diag_heading_suppression_abs is not None) else np.nan))
                diag_smoothing_alpha.append(float(to.diag_smoothing_alpha if (to and to.diag_smoothing_alpha is not None) else np.nan))
                diag_smoothing_alpha_x.append(float(to.diag_smoothing_alpha_x if (to and to.diag_smoothing_alpha_x is not None) else np.nan))
                diag_multi_lookahead_heading_base.append(float(to.diag_multi_lookahead_heading_base if (to and to.diag_multi_lookahead_heading_base is not None) else np.nan))
                diag_multi_lookahead_heading_far.append(float(to.diag_multi_lookahead_heading_far if (to and to.diag_multi_lookahead_heading_far is not None) else np.nan))
                diag_multi_lookahead_heading_blended.append(float(to.diag_multi_lookahead_heading_blended if (to and to.diag_multi_lookahead_heading_blended is not None) else np.nan))
                diag_multi_lookahead_blend_alpha.append(float(to.diag_multi_lookahead_blend_alpha if (to and to.diag_multi_lookahead_blend_alpha is not None) else np.nan))
                diag_dynamic_effective_horizon_m.append(float(to.diag_dynamic_effective_horizon_m if (to and to.diag_dynamic_effective_horizon_m is not None) else np.nan))
                diag_dynamic_effective_horizon_base_m.append(float(to.diag_dynamic_effective_horizon_base_m if (to and to.diag_dynamic_effective_horizon_base_m is not None) else np.nan))
                diag_dynamic_effective_horizon_min_m.append(float(to.diag_dynamic_effective_horizon_min_m if (to and to.diag_dynamic_effective_horizon_min_m is not None) else np.nan))
                diag_dynamic_effective_horizon_max_m.append(float(to.diag_dynamic_effective_horizon_max_m if (to and to.diag_dynamic_effective_horizon_max_m is not None) else np.nan))
                diag_dynamic_effective_horizon_speed_scale.append(float(to.diag_dynamic_effective_horizon_speed_scale if (to and to.diag_dynamic_effective_horizon_speed_scale is not None) else np.nan))
                diag_dynamic_effective_horizon_curvature_scale.append(float(to.diag_dynamic_effective_horizon_curvature_scale if (to and to.diag_dynamic_effective_horizon_curvature_scale is not None) else np.nan))
                diag_dynamic_effective_horizon_confidence_scale.append(float(to.diag_dynamic_effective_horizon_confidence_scale if (to and to.diag_dynamic_effective_horizon_confidence_scale is not None) else np.nan))
                diag_dynamic_effective_horizon_final_scale.append(float(to.diag_dynamic_effective_horizon_final_scale if (to and to.diag_dynamic_effective_horizon_final_scale is not None) else np.nan))
                diag_dynamic_effective_horizon_speed_mps.append(float(to.diag_dynamic_effective_horizon_speed_mps if (to and to.diag_dynamic_effective_horizon_speed_mps is not None) else np.nan))
                diag_dynamic_effective_horizon_curvature_abs.append(float(to.diag_dynamic_effective_horizon_curvature_abs if (to and to.diag_dynamic_effective_horizon_curvature_abs is not None) else np.nan))
                diag_dynamic_effective_horizon_confidence_used.append(float(to.diag_dynamic_effective_horizon_confidence_used if (to and to.diag_dynamic_effective_horizon_confidence_used is not None) else np.nan))
                diag_dynamic_effective_horizon_limiter_code.append(float(to.diag_dynamic_effective_horizon_limiter_code if (to and to.diag_dynamic_effective_horizon_limiter_code is not None) else np.nan))
                diag_dynamic_effective_horizon_applied.append(float(to.diag_dynamic_effective_horizon_applied if (to and to.diag_dynamic_effective_horizon_applied is not None) else np.nan))
                diag_preclip_abs_mean_0_8m.append(float(to.diag_preclip_abs_mean_0_8m if (to and to.diag_preclip_abs_mean_0_8m is not None) else np.nan))
                diag_preclip_abs_mean_8_12m.append(float(to.diag_preclip_abs_mean_8_12m if (to and to.diag_preclip_abs_mean_8_12m is not None) else np.nan))
                diag_preclip_abs_mean_12_20m.append(float(to.diag_preclip_abs_mean_12_20m if (to and to.diag_preclip_abs_mean_12_20m is not None) else np.nan))
                diag_postclip_x0.append(float(to.diag_postclip_x0 if (to and to.diag_postclip_x0 is not None) else np.nan))
                diag_postclip_x1.append(float(to.diag_postclip_x1 if (to and to.diag_postclip_x1 is not None) else np.nan))
                diag_postclip_x2.append(float(to.diag_postclip_x2 if (to and to.diag_postclip_x2 is not None) else np.nan))
                diag_postclip_abs_mean_0_8m.append(float(to.diag_postclip_abs_mean_0_8m if (to and to.diag_postclip_abs_mean_0_8m is not None) else np.nan))
                diag_postclip_abs_mean_8_12m.append(float(to.diag_postclip_abs_mean_8_12m if (to and to.diag_postclip_abs_mean_8_12m is not None) else np.nan))
                diag_postclip_abs_mean_12_20m.append(float(to.diag_postclip_abs_mean_12_20m if (to and to.diag_postclip_abs_mean_12_20m is not None) else np.nan))
                diag_postclip_near_clip_frac_12_20m.append(float(to.diag_postclip_near_clip_frac_12_20m if (to and to.diag_postclip_near_clip_frac_12_20m is not None) else np.nan))
                diag_first_segment_y0_gt_y1_pre.append(float(to.diag_first_segment_y0_gt_y1_pre if (to and to.diag_first_segment_y0_gt_y1_pre is not None) else np.nan))
                diag_first_segment_y0_gt_y1_post.append(float(to.diag_first_segment_y0_gt_y1_post if (to and to.diag_first_segment_y0_gt_y1_post is not None) else np.nan))
                diag_inversion_introduced_after_conversion.append(float(to.diag_inversion_introduced_after_conversion if (to and to.diag_inversion_introduced_after_conversion is not None) else np.nan))
        
        if ref_points:
            ref_points_array = np.array(ref_points, dtype=np.float32)
            ref_points_raw_array = np.array(ref_points_raw, dtype=np.float32)
            current_size_rp = self.h5_file["trajectory/reference_point_x"].shape[0]
            new_size_rp = current_size_rp + len(ref_points)
            
            # Resize all reference point datasets (smoothed and raw)
            self.h5_file["trajectory/reference_point_x"].resize((new_size_rp,))
            self.h5_file["trajectory/reference_point_y"].resize((new_size_rp,))
            self.h5_file["trajectory/reference_point_heading"].resize((new_size_rp,))
            self.h5_file["trajectory/reference_point_velocity"].resize((new_size_rp,))
            self.h5_file["trajectory/reference_point_raw_x"].resize((new_size_rp,))
            self.h5_file["trajectory/reference_point_raw_y"].resize((new_size_rp,))
            self.h5_file["trajectory/reference_point_raw_heading"].resize((new_size_rp,))
            # NEW: Resize debug datasets
            self.h5_file["trajectory/reference_point_method"].resize((new_size_rp,))
            self.h5_file["trajectory/perception_center_x"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_available"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_generated_by_fallback"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_points_generated"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_x_clip_count"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_pre_y0"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_pre_y1"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_pre_y2"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_post_y0"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_post_y1"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_post_y2"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_used_provided_distance0"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_used_provided_distance1"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_used_provided_distance2"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_post_minus_pre_y0"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_post_minus_pre_y1"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_post_minus_pre_y2"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_preclip_x0"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_preclip_x1"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_preclip_x2"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_preclip_x_abs_max"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_preclip_x_abs_p95"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_preclip_mean_12_20m_lane_source_x"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_preclip_mean_12_20m_distance_scale_delta_x"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_preclip_mean_12_20m_camera_offset_delta_x"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_preclip_abs_mean_12_20m_lane_source_x"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_preclip_abs_mean_12_20m_distance_scale_delta_x"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_preclip_abs_mean_12_20m_camera_offset_delta_x"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_heading_zero_gate_active"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_small_heading_gate_active"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_multi_lookahead_active"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_smoothing_jump_reject"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_ref_x_rate_limit_active"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_raw_ref_x"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_smoothed_ref_x"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_ref_x_suppression_abs"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_raw_ref_heading"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_smoothed_ref_heading"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_heading_suppression_abs"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_smoothing_alpha"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_smoothing_alpha_x"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_multi_lookahead_heading_base"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_multi_lookahead_heading_far"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_multi_lookahead_heading_blended"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_multi_lookahead_blend_alpha"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_dynamic_effective_horizon_m"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_dynamic_effective_horizon_base_m"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_dynamic_effective_horizon_min_m"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_dynamic_effective_horizon_max_m"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_dynamic_effective_horizon_speed_scale"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_dynamic_effective_horizon_curvature_scale"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_dynamic_effective_horizon_confidence_scale"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_dynamic_effective_horizon_final_scale"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_dynamic_effective_horizon_speed_mps"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_dynamic_effective_horizon_curvature_abs"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_dynamic_effective_horizon_confidence_used"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_dynamic_effective_horizon_limiter_code"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_dynamic_effective_horizon_applied"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_preclip_abs_mean_0_8m"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_preclip_abs_mean_8_12m"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_preclip_abs_mean_12_20m"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_postclip_x0"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_postclip_x1"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_postclip_x2"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_postclip_abs_mean_0_8m"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_postclip_abs_mean_8_12m"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_postclip_abs_mean_12_20m"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_postclip_near_clip_frac_12_20m"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_first_segment_y0_gt_y1_pre"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_first_segment_y0_gt_y1_post"].resize((new_size_rp,))
            self.h5_file["trajectory/diag_inversion_introduced_after_conversion"].resize((new_size_rp,))
            
            # Write smoothed data
            self.h5_file["trajectory/reference_point_x"][current_size_rp:] = ref_points_array[:, 0]
            self.h5_file["trajectory/reference_point_y"][current_size_rp:] = ref_points_array[:, 1]
            self.h5_file["trajectory/reference_point_heading"][current_size_rp:] = ref_points_array[:, 2]
            self.h5_file["trajectory/reference_point_velocity"][current_size_rp:] = ref_points_array[:, 3]
            # Write raw data
            self.h5_file["trajectory/reference_point_raw_x"][current_size_rp:] = ref_points_raw_array[:, 0]
            self.h5_file["trajectory/reference_point_raw_y"][current_size_rp:] = ref_points_raw_array[:, 1]
            self.h5_file["trajectory/reference_point_raw_heading"][current_size_rp:] = ref_points_raw_array[:, 2]
            # NEW: Write debug data
            # Convert method strings to bytes for HDF5
            method_bytes = [m.encode('utf-8') if isinstance(m, str) else b'unknown' for m in ref_methods]
            self.h5_file["trajectory/reference_point_method"][current_size_rp:] = method_bytes
            self.h5_file["trajectory/perception_center_x"][current_size_rp:] = np.array(perception_centers, dtype=np.float32)
            self.h5_file["trajectory/diag_available"][current_size_rp:] = np.array(diag_available, dtype=np.float32)
            self.h5_file["trajectory/diag_generated_by_fallback"][current_size_rp:] = np.array(diag_generated_by_fallback, dtype=np.float32)
            self.h5_file["trajectory/diag_points_generated"][current_size_rp:] = np.array(diag_points_generated, dtype=np.float32)
            self.h5_file["trajectory/diag_x_clip_count"][current_size_rp:] = np.array(diag_x_clip_count, dtype=np.float32)
            self.h5_file["trajectory/diag_pre_y0"][current_size_rp:] = np.array(diag_pre_y0, dtype=np.float32)
            self.h5_file["trajectory/diag_pre_y1"][current_size_rp:] = np.array(diag_pre_y1, dtype=np.float32)
            self.h5_file["trajectory/diag_pre_y2"][current_size_rp:] = np.array(diag_pre_y2, dtype=np.float32)
            self.h5_file["trajectory/diag_post_y0"][current_size_rp:] = np.array(diag_post_y0, dtype=np.float32)
            self.h5_file["trajectory/diag_post_y1"][current_size_rp:] = np.array(diag_post_y1, dtype=np.float32)
            self.h5_file["trajectory/diag_post_y2"][current_size_rp:] = np.array(diag_post_y2, dtype=np.float32)
            self.h5_file["trajectory/diag_used_provided_distance0"][current_size_rp:] = np.array(diag_used_provided_distance0, dtype=np.float32)
            self.h5_file["trajectory/diag_used_provided_distance1"][current_size_rp:] = np.array(diag_used_provided_distance1, dtype=np.float32)
            self.h5_file["trajectory/diag_used_provided_distance2"][current_size_rp:] = np.array(diag_used_provided_distance2, dtype=np.float32)
            self.h5_file["trajectory/diag_post_minus_pre_y0"][current_size_rp:] = np.array(diag_post_minus_pre_y0, dtype=np.float32)
            self.h5_file["trajectory/diag_post_minus_pre_y1"][current_size_rp:] = np.array(diag_post_minus_pre_y1, dtype=np.float32)
            self.h5_file["trajectory/diag_post_minus_pre_y2"][current_size_rp:] = np.array(diag_post_minus_pre_y2, dtype=np.float32)
            self.h5_file["trajectory/diag_preclip_x0"][current_size_rp:] = np.array(diag_preclip_x0, dtype=np.float32)
            self.h5_file["trajectory/diag_preclip_x1"][current_size_rp:] = np.array(diag_preclip_x1, dtype=np.float32)
            self.h5_file["trajectory/diag_preclip_x2"][current_size_rp:] = np.array(diag_preclip_x2, dtype=np.float32)
            self.h5_file["trajectory/diag_preclip_x_abs_max"][current_size_rp:] = np.array(diag_preclip_x_abs_max, dtype=np.float32)
            self.h5_file["trajectory/diag_preclip_x_abs_p95"][current_size_rp:] = np.array(diag_preclip_x_abs_p95, dtype=np.float32)
            self.h5_file["trajectory/diag_preclip_mean_12_20m_lane_source_x"][current_size_rp:] = np.array(diag_preclip_mean_12_20m_lane_source_x, dtype=np.float32)
            self.h5_file["trajectory/diag_preclip_mean_12_20m_distance_scale_delta_x"][current_size_rp:] = np.array(diag_preclip_mean_12_20m_distance_scale_delta_x, dtype=np.float32)
            self.h5_file["trajectory/diag_preclip_mean_12_20m_camera_offset_delta_x"][current_size_rp:] = np.array(diag_preclip_mean_12_20m_camera_offset_delta_x, dtype=np.float32)
            self.h5_file["trajectory/diag_preclip_abs_mean_12_20m_lane_source_x"][current_size_rp:] = np.array(diag_preclip_abs_mean_12_20m_lane_source_x, dtype=np.float32)
            self.h5_file["trajectory/diag_preclip_abs_mean_12_20m_distance_scale_delta_x"][current_size_rp:] = np.array(diag_preclip_abs_mean_12_20m_distance_scale_delta_x, dtype=np.float32)
            self.h5_file["trajectory/diag_preclip_abs_mean_12_20m_camera_offset_delta_x"][current_size_rp:] = np.array(diag_preclip_abs_mean_12_20m_camera_offset_delta_x, dtype=np.float32)
            self.h5_file["trajectory/diag_heading_zero_gate_active"][current_size_rp:] = np.array(diag_heading_zero_gate_active, dtype=np.float32)
            self.h5_file["trajectory/diag_small_heading_gate_active"][current_size_rp:] = np.array(diag_small_heading_gate_active, dtype=np.float32)
            self.h5_file["trajectory/diag_multi_lookahead_active"][current_size_rp:] = np.array(diag_multi_lookahead_active, dtype=np.float32)
            self.h5_file["trajectory/diag_smoothing_jump_reject"][current_size_rp:] = np.array(diag_smoothing_jump_reject, dtype=np.float32)
            self.h5_file["trajectory/diag_ref_x_rate_limit_active"][current_size_rp:] = np.array(diag_ref_x_rate_limit_active, dtype=np.float32)
            self.h5_file["trajectory/diag_raw_ref_x"][current_size_rp:] = np.array(diag_raw_ref_x, dtype=np.float32)
            self.h5_file["trajectory/diag_smoothed_ref_x"][current_size_rp:] = np.array(diag_smoothed_ref_x, dtype=np.float32)
            self.h5_file["trajectory/diag_ref_x_suppression_abs"][current_size_rp:] = np.array(diag_ref_x_suppression_abs, dtype=np.float32)
            self.h5_file["trajectory/diag_raw_ref_heading"][current_size_rp:] = np.array(diag_raw_ref_heading, dtype=np.float32)
            self.h5_file["trajectory/diag_smoothed_ref_heading"][current_size_rp:] = np.array(diag_smoothed_ref_heading, dtype=np.float32)
            self.h5_file["trajectory/diag_heading_suppression_abs"][current_size_rp:] = np.array(diag_heading_suppression_abs, dtype=np.float32)
            self.h5_file["trajectory/diag_smoothing_alpha"][current_size_rp:] = np.array(diag_smoothing_alpha, dtype=np.float32)
            self.h5_file["trajectory/diag_smoothing_alpha_x"][current_size_rp:] = np.array(diag_smoothing_alpha_x, dtype=np.float32)
            self.h5_file["trajectory/diag_multi_lookahead_heading_base"][current_size_rp:] = np.array(diag_multi_lookahead_heading_base, dtype=np.float32)
            self.h5_file["trajectory/diag_multi_lookahead_heading_far"][current_size_rp:] = np.array(diag_multi_lookahead_heading_far, dtype=np.float32)
            self.h5_file["trajectory/diag_multi_lookahead_heading_blended"][current_size_rp:] = np.array(diag_multi_lookahead_heading_blended, dtype=np.float32)
            self.h5_file["trajectory/diag_multi_lookahead_blend_alpha"][current_size_rp:] = np.array(diag_multi_lookahead_blend_alpha, dtype=np.float32)
            self.h5_file["trajectory/diag_dynamic_effective_horizon_m"][current_size_rp:] = np.array(diag_dynamic_effective_horizon_m, dtype=np.float32)
            self.h5_file["trajectory/diag_dynamic_effective_horizon_base_m"][current_size_rp:] = np.array(diag_dynamic_effective_horizon_base_m, dtype=np.float32)
            self.h5_file["trajectory/diag_dynamic_effective_horizon_min_m"][current_size_rp:] = np.array(diag_dynamic_effective_horizon_min_m, dtype=np.float32)
            self.h5_file["trajectory/diag_dynamic_effective_horizon_max_m"][current_size_rp:] = np.array(diag_dynamic_effective_horizon_max_m, dtype=np.float32)
            self.h5_file["trajectory/diag_dynamic_effective_horizon_speed_scale"][current_size_rp:] = np.array(diag_dynamic_effective_horizon_speed_scale, dtype=np.float32)
            self.h5_file["trajectory/diag_dynamic_effective_horizon_curvature_scale"][current_size_rp:] = np.array(diag_dynamic_effective_horizon_curvature_scale, dtype=np.float32)
            self.h5_file["trajectory/diag_dynamic_effective_horizon_confidence_scale"][current_size_rp:] = np.array(diag_dynamic_effective_horizon_confidence_scale, dtype=np.float32)
            self.h5_file["trajectory/diag_dynamic_effective_horizon_final_scale"][current_size_rp:] = np.array(diag_dynamic_effective_horizon_final_scale, dtype=np.float32)
            self.h5_file["trajectory/diag_dynamic_effective_horizon_speed_mps"][current_size_rp:] = np.array(diag_dynamic_effective_horizon_speed_mps, dtype=np.float32)
            self.h5_file["trajectory/diag_dynamic_effective_horizon_curvature_abs"][current_size_rp:] = np.array(diag_dynamic_effective_horizon_curvature_abs, dtype=np.float32)
            self.h5_file["trajectory/diag_dynamic_effective_horizon_confidence_used"][current_size_rp:] = np.array(diag_dynamic_effective_horizon_confidence_used, dtype=np.float32)
            self.h5_file["trajectory/diag_dynamic_effective_horizon_limiter_code"][current_size_rp:] = np.array(diag_dynamic_effective_horizon_limiter_code, dtype=np.float32)
            self.h5_file["trajectory/diag_dynamic_effective_horizon_applied"][current_size_rp:] = np.array(diag_dynamic_effective_horizon_applied, dtype=np.float32)
            self.h5_file["trajectory/diag_preclip_abs_mean_0_8m"][current_size_rp:] = np.array(diag_preclip_abs_mean_0_8m, dtype=np.float32)
            self.h5_file["trajectory/diag_preclip_abs_mean_8_12m"][current_size_rp:] = np.array(diag_preclip_abs_mean_8_12m, dtype=np.float32)
            self.h5_file["trajectory/diag_preclip_abs_mean_12_20m"][current_size_rp:] = np.array(diag_preclip_abs_mean_12_20m, dtype=np.float32)
            self.h5_file["trajectory/diag_postclip_x0"][current_size_rp:] = np.array(diag_postclip_x0, dtype=np.float32)
            self.h5_file["trajectory/diag_postclip_x1"][current_size_rp:] = np.array(diag_postclip_x1, dtype=np.float32)
            self.h5_file["trajectory/diag_postclip_x2"][current_size_rp:] = np.array(diag_postclip_x2, dtype=np.float32)
            self.h5_file["trajectory/diag_postclip_abs_mean_0_8m"][current_size_rp:] = np.array(diag_postclip_abs_mean_0_8m, dtype=np.float32)
            self.h5_file["trajectory/diag_postclip_abs_mean_8_12m"][current_size_rp:] = np.array(diag_postclip_abs_mean_8_12m, dtype=np.float32)
            self.h5_file["trajectory/diag_postclip_abs_mean_12_20m"][current_size_rp:] = np.array(diag_postclip_abs_mean_12_20m, dtype=np.float32)
            self.h5_file["trajectory/diag_postclip_near_clip_frac_12_20m"][current_size_rp:] = np.array(diag_postclip_near_clip_frac_12_20m, dtype=np.float32)
            self.h5_file["trajectory/diag_first_segment_y0_gt_y1_pre"][current_size_rp:] = np.array(diag_first_segment_y0_gt_y1_pre, dtype=np.float32)
            self.h5_file["trajectory/diag_first_segment_y0_gt_y1_post"][current_size_rp:] = np.array(diag_first_segment_y0_gt_y1_post, dtype=np.float32)
            self.h5_file["trajectory/diag_inversion_introduced_after_conversion"][current_size_rp:] = np.array(diag_inversion_introduced_after_conversion, dtype=np.float32)
        
        # NEW: Write trajectory points (full path, not just reference point)
        trajectory_points_list = []
        oracle_points_list = []
        oracle_point_counts = []
        oracle_horizons = []
        oracle_spacings = []
        oracle_enabled = []
        for frame in frames:
            if frame.trajectory_output and frame.trajectory_output.trajectory_points is not None:
                # trajectory_points is [N, 3] array (x, y, heading)
                # Convert to array for variable-length storage
                traj_points = frame.trajectory_output.trajectory_points
                
                # CRITICAL FIX: Validate and fix shape BEFORE adding to list
                # Handle 1D arrays (flattened) and ensure 2D [N, 3]
                if traj_points.ndim == 1:
                    # Reshape 1D array to 2D: (N*3,) -> (N, 3)
                    if traj_points.size % 3 == 0:
                        traj_points = traj_points.reshape(-1, 3)
                    else:
                        # Invalid size - can't reshape, use empty array
                        logger.warning(f"[RECORDER] Trajectory points has invalid 1D shape {traj_points.shape}, "
                                     f"size {traj_points.size} not divisible by 3. Using empty array.")
                        traj_points = np.array([], dtype=np.float32).reshape(0, 3)
                elif traj_points.ndim == 2:
                    # Already 2D - ensure it has shape [N, 3]
                    if traj_points.shape[1] != 3:
                        logger.warning(f"[RECORDER] Trajectory points has shape {traj_points.shape}, "
                                     f"expected [N, 3]. Using empty array.")
                        traj_points = np.array([], dtype=np.float32).reshape(0, 3)
                else:
                    # Invalid dimensions - use empty array
                    logger.warning(f"[RECORDER] Trajectory points has invalid shape {traj_points.shape}, "
                                 f"expected 1D or 2D. Using empty array.")
                    traj_points = np.array([], dtype=np.float32).reshape(0, 3)
                
                # Ensure it's float32 and add to list
                trajectory_points_list.append(traj_points.astype(np.float32))
            else:
                # No trajectory points for this frame - store empty array
                trajectory_points_list.append(np.array([], dtype=np.float32).reshape(0, 3))

            # Oracle points are tracked per frame alongside trajectory points.
            if frame.trajectory_output and frame.trajectory_output.oracle_points is not None:
                oracle_pts = frame.trajectory_output.oracle_points
                if oracle_pts.ndim == 1:
                    if oracle_pts.size % 2 == 0:
                        oracle_pts = oracle_pts.reshape(-1, 2)
                    else:
                        logger.warning(
                            f"[RECORDER] Oracle points has invalid 1D shape {oracle_pts.shape}, "
                            f"size {oracle_pts.size} not divisible by 2. Using empty array."
                        )
                        oracle_pts = np.array([], dtype=np.float32).reshape(0, 2)
                elif oracle_pts.ndim == 2:
                    if oracle_pts.shape[1] != 2:
                        logger.warning(
                            f"[RECORDER] Oracle points has shape {oracle_pts.shape}, expected [N, 2]. Using empty array."
                        )
                        oracle_pts = np.array([], dtype=np.float32).reshape(0, 2)
                else:
                    logger.warning(
                        f"[RECORDER] Oracle points has invalid shape {oracle_pts.shape}, expected 1D or 2D. Using empty array."
                    )
                    oracle_pts = np.array([], dtype=np.float32).reshape(0, 2)
                oracle_points_list.append(oracle_pts.astype(np.float32))
            else:
                oracle_points_list.append(np.array([], dtype=np.float32).reshape(0, 2))
            if frame.trajectory_output:
                oracle_point_counts.append(int(frame.trajectory_output.oracle_point_count or 0))
                oracle_horizons.append(float(frame.trajectory_output.oracle_horizon_meters or 0.0))
                oracle_spacings.append(float(frame.trajectory_output.oracle_point_spacing_meters or 0.0))
                oracle_enabled.append(1 if frame.trajectory_output.oracle_samples_enabled else 0)
            else:
                oracle_point_counts.append(0)
                oracle_horizons.append(0.0)
                oracle_spacings.append(0.0)
                oracle_enabled.append(0)
        
        if trajectory_points_list:
            # Resize trajectory_points dataset
            # CRITICAL FIX: Handle empty dataset case (shape might be () instead of (0,))
            try:
                dataset_shape = self.h5_file["trajectory/trajectory_points"].shape
                if len(dataset_shape) == 0:
                    # Dataset is scalar/empty - initialize to 1D array
                    current_size_traj = 0
                else:
                    current_size_traj = dataset_shape[0]
            except (AttributeError, IndexError, TypeError):
                # Fallback if shape access fails
                current_size_traj = 0
            
            new_size_traj = current_size_traj + len(trajectory_points_list)
            self.h5_file["trajectory/trajectory_points"].resize((new_size_traj,))
            self.h5_file["trajectory/oracle_points"].resize((new_size_traj,))
            self.h5_file["trajectory/oracle_point_count"].resize((new_size_traj,))
            self.h5_file["trajectory/oracle_horizon_meters"].resize((new_size_traj,))
            self.h5_file["trajectory/oracle_point_spacing_meters"].resize((new_size_traj,))
            self.h5_file["trajectory/oracle_samples_enabled"].resize((new_size_traj,))
            
            # Write trajectory points (variable-length arrays)
            for i, traj_points in enumerate(trajectory_points_list):
                # CRITICAL FIX: Ensure traj_points is a 2D array [N, 3]
                # If it's 1D, reshape it (assuming it's flattened [x1, y1, h1, x2, y2, h2, ...])
                if traj_points.ndim == 1:
                    # Reshape 1D array to 2D: (N*3,) -> (N, 3)
                    if traj_points.size % 3 == 0:
                        traj_points = traj_points.reshape(-1, 3)
                    else:
                        # Invalid size - can't reshape, use empty array
                        logger.warning(f"[RECORDER] Trajectory points has invalid 1D shape {traj_points.shape}, "
                                     f"size {traj_points.size} not divisible by 3. Using empty array.")
                        traj_points = np.array([], dtype=np.float32).reshape(0, 3)
                elif traj_points.ndim == 2:
                    # Already 2D - ensure it has shape [N, 3]
                    if traj_points.shape[1] != 3:
                        logger.warning(f"[RECORDER] Trajectory points has shape {traj_points.shape}, "
                                     f"expected [N, 3]. Using empty array.")
                        traj_points = np.array([], dtype=np.float32).reshape(0, 3)
                else:
                    # Invalid dimensions - use empty array
                    logger.warning(f"[RECORDER] Trajectory points has invalid shape {traj_points.shape}, "
                                 f"expected 1D or 2D. Using empty array.")
                    traj_points = np.array([], dtype=np.float32).reshape(0, 3)
                
                # Ensure it's float32 for HDF5
                traj_points = traj_points.astype(np.float32)
                
                # CRITICAL FIX: For variable-length arrays in HDF5, we need to flatten 2D arrays
                # h5py vlen_dtype expects 1D arrays, not 2D arrays
                # So we flatten (N, 3) to (N*3,) before writing
                if traj_points.ndim == 2:
                    traj_points_flat = traj_points.flatten()  # (N, 3) -> (N*3,)
                else:
                    traj_points_flat = traj_points  # Already 1D
                
                # Write to dataset with error handling
                try:
                    self.h5_file["trajectory/trajectory_points"][current_size_traj + i] = traj_points_flat
                except (TypeError, ValueError) as e:
                    logger.error(f"[RECORDER] Error writing trajectory points at index {current_size_traj + i}: {e}")
                    logger.error(f"  traj_points original shape: {traj_points.shape}, dtype: {traj_points.dtype}")
                    logger.error(f"  traj_points_flat shape: {traj_points_flat.shape if 'traj_points_flat' in locals() else 'N/A'}")
                    logger.error(f"  dataset shape: {self.h5_file['trajectory/trajectory_points'].shape}")
                    # Use empty array as fallback
                    self.h5_file["trajectory/trajectory_points"][current_size_traj + i] = np.array([], dtype=np.float32)

                # Oracle points are vlen float arrays flattened from [N, 2].
                oracle_pts = oracle_points_list[i]
                if oracle_pts.ndim == 2:
                    oracle_pts_flat = oracle_pts.flatten()
                else:
                    oracle_pts_flat = oracle_pts
                try:
                    self.h5_file["trajectory/oracle_points"][current_size_traj + i] = oracle_pts_flat
                except (TypeError, ValueError) as e:
                    logger.error(f"[RECORDER] Error writing oracle points at index {current_size_traj + i}: {e}")
                    self.h5_file["trajectory/oracle_points"][current_size_traj + i] = np.array([], dtype=np.float32)

            self.h5_file["trajectory/oracle_point_count"][current_size_traj:new_size_traj] = np.array(
                oracle_point_counts, dtype=np.int32
            )
            self.h5_file["trajectory/oracle_horizon_meters"][current_size_traj:new_size_traj] = np.array(
                oracle_horizons, dtype=np.float32
            )
            self.h5_file["trajectory/oracle_point_spacing_meters"][current_size_traj:new_size_traj] = np.array(
                oracle_spacings, dtype=np.float32
            )
            self.h5_file["trajectory/oracle_samples_enabled"][current_size_traj:new_size_traj] = np.array(
                oracle_enabled, dtype=np.int8
            )
    
    def _write_unity_feedback(self, frames: List[RecordingFrame]):
        """Write Unity feedback data to HDF5."""
        timestamps = []
        gt_mode_active = []
        control_received = []
        actual_steerings = []
        actual_throttles = []
        actual_brakes = []
        gt_data_available = []
        gt_reporter_enabled = []
        path_curvature_calc = []
        car_controller_modes = []
        av_control_enabled = []
        unity_frame_counts = []
        
        for frame in frames:
            uf = frame.unity_feedback
            if uf is None:
                continue
            timestamps.append(uf.timestamp)
            gt_mode_active.append(uf.ground_truth_mode_active)
            control_received.append(uf.control_command_received)
            actual_steerings.append(uf.actual_steering_applied if uf.actual_steering_applied is not None else 0.0)
            actual_throttles.append(uf.actual_throttle_applied if uf.actual_throttle_applied is not None else 0.0)
            actual_brakes.append(uf.actual_brake_applied if uf.actual_brake_applied is not None else 0.0)
            gt_data_available.append(uf.ground_truth_data_available)
            gt_reporter_enabled.append(uf.ground_truth_reporter_enabled)
            path_curvature_calc.append(uf.path_curvature_calculated)
            car_controller_modes.append(uf.car_controller_mode if uf.car_controller_mode else "unknown")
            av_control_enabled.append(uf.av_control_enabled)
            unity_frame_counts.append(uf.unity_frame_count if uf.unity_frame_count is not None else 0)
        
        if timestamps:
            current_size = self.h5_file["unity_feedback/timestamps"].shape[0]
            new_size = current_size + len(timestamps)
            
            # Resize all datasets
            for key in ["timestamps", "ground_truth_mode_active", "control_command_received",
                       "actual_steering_applied", "actual_throttle_applied", "actual_brake_applied",
                       "ground_truth_data_available", "ground_truth_reporter_enabled",
                       "path_curvature_calculated", "car_controller_mode", "av_control_enabled",
                       "unity_frame_count"]:
                self.h5_file[f"unity_feedback/{key}"].resize((new_size,))
            
            # Write data
            self.h5_file["unity_feedback/timestamps"][current_size:] = timestamps
            self.h5_file["unity_feedback/ground_truth_mode_active"][current_size:] = gt_mode_active
            self.h5_file["unity_feedback/control_command_received"][current_size:] = control_received
            self.h5_file["unity_feedback/actual_steering_applied"][current_size:] = actual_steerings
            self.h5_file["unity_feedback/actual_throttle_applied"][current_size:] = actual_throttles
            self.h5_file["unity_feedback/actual_brake_applied"][current_size:] = actual_brakes
            self.h5_file["unity_feedback/ground_truth_data_available"][current_size:] = gt_data_available
            self.h5_file["unity_feedback/ground_truth_reporter_enabled"][current_size:] = gt_reporter_enabled
            self.h5_file["unity_feedback/path_curvature_calculated"][current_size:] = path_curvature_calc
            self.h5_file["unity_feedback/car_controller_mode"][current_size:] = car_controller_modes
            self.h5_file["unity_feedback/av_control_enabled"][current_size:] = av_control_enabled
            self.h5_file["unity_feedback/unity_frame_count"][current_size:] = unity_frame_counts
    
    def close(self):
        """Close the recording file."""
        try:
            with self.frame_buffer_lock:
                if self.frame_buffer:
                    frames = self.frame_buffer
                    self.frame_buffer = []
                    self.flush_queue.put(frames)
            self.flush_stop_event.set()
            self.flush_thread.join(timeout=5.0)
        except Exception as e:
            logger.error(f"Error during final flush: {e}", exc_info=True)
            # Continue to close the file even if flush fails
        
        # Save metadata
        self.metadata["recording_end_time"] = datetime.now().isoformat()
        self.metadata["total_frames"] = self.frame_count
        
        try:
            metadata_str = json.dumps(self.metadata, indent=2)
            self.h5_file.attrs["metadata"] = metadata_str
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")
        
        try:
            self.h5_file.close()
            print(f"Recording saved to: {self.output_file}")
        except Exception as e:
            logger.error(f"Error closing HDF5 file: {e}", exc_info=True)
            raise
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

