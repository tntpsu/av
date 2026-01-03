"""
Data recorder for AV stack.
Records camera frames, vehicle state, control commands, and model outputs.
"""

import h5py
import numpy as np
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import cv2

logger = logging.getLogger(__name__)

from .formats.data_format import (
    CameraFrame, VehicleState, ControlCommand,
    PerceptionOutput, TrajectoryOutput, RecordingFrame
)


class DataRecorder:
    """Records AV stack data to HDF5 format."""
    
    def __init__(self, output_dir: str, recording_name: Optional[str] = None,
                 recording_type: Optional[str] = None):
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
        self.frame_count = 0
        
        # Metadata
        self.metadata = {
            "recording_start_time": datetime.now().isoformat(),
            "recording_name": recording_name,
            "recording_type": recording_type
        }
    
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
            compression="gzip"
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
            "vehicle/steering_angle",
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
        # NEW: Camera calibration - actual screen y pixel where 8m appears
        self.h5_file.create_dataset(
            "vehicle/camera_8m_screen_y",
            shape=(0,),
            maxshape=max_shape,
            dtype=np.float32
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
        self.frame_buffer.append(frame)
        self.frame_count += 1
        
        # Flush buffer periodically
        if len(self.frame_buffer) >= 100:
            self.flush()
    
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
        if not self.frame_buffer:
            return
        
        # Group frames by type and write to HDF5
        camera_frames = []
        vehicle_states = []
        control_commands = []
        perception_outputs = []
        trajectory_outputs = []
        unity_feedbacks = []
        
        for frame in self.frame_buffer:
            if frame.camera_frame:
                camera_frames.append(frame)
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
        if len(self.frame_buffer) > 0:
            sample_frame = self.frame_buffer[0]
            logger.info(f"[RECORDER] Flushing {len(self.frame_buffer)} frames: "
                       f"camera={len(camera_frames)}, vehicle={len(vehicle_states)}, "
                       f"control={len(control_commands)}, perception={len(perception_outputs)}, "
                       f"trajectory={len(trajectory_outputs)}, unity={len(unity_feedbacks)}")
            if len(vehicle_states) == 0 and len(self.frame_buffer) > 0:
                logger.error(f"[RECORDER] ❌ CRITICAL: No vehicle states! Sample frame has vehicle_state={sample_frame.vehicle_state is not None}")
                if sample_frame.vehicle_state is None:
                    logger.error(f"[RECORDER] ❌ Sample frame vehicle_state is None - frames are being collected without vehicle data!")
                    logger.error(f"[RECORDER] This will cause ground truth and camera data to not be recorded!")
            if len(perception_outputs) == 0 and len(self.frame_buffer) > 0:
                logger.warning(f"[RECORDER] ⚠️  No perception outputs! Sample frame has perception_output={sample_frame.perception_output is not None}")
                if sample_frame.perception_output is None:
                    logger.warning(f"[RECORDER] ⚠️  Sample frame perception_output is None - frames are being collected without perception data!")
            if len(control_commands) == 0 and len(self.frame_buffer) > 0:
                logger.warning(f"[RECORDER] ⚠️  No control commands! Sample frame has control_command={sample_frame.control_command is not None}")
                if sample_frame.control_command is None:
                    logger.warning(f"[RECORDER] ⚠️  Sample frame control_command is None - frames are being collected without control data!")
            if len(trajectory_outputs) == 0 and len(self.frame_buffer) > 0:
                logger.warning(f"[RECORDER] ⚠️  No trajectory outputs! Sample frame has trajectory_output={sample_frame.trajectory_output is not None}")
                if sample_frame.trajectory_output is None:
                    logger.warning(f"[RECORDER] ⚠️  Sample frame trajectory_output is None - frames are being collected without trajectory data!")
        
        # Write camera frames
        if camera_frames:
            try:
                self._write_camera_frames(camera_frames)
            except Exception as e:
                logger.error(f"Error writing camera frames: {e}", exc_info=True)
        
        # Write vehicle states
        if vehicle_states:
            try:
                self._write_vehicle_states(vehicle_states)
            except Exception as e:
                logger.error(f"Error writing vehicle states: {e}", exc_info=True)
        else:
            # CRITICAL: Log why vehicle states are empty
            if len(self.frame_buffer) > 0:
                sample_frame = self.frame_buffer[0]
                logger.error(f"[RECORDER] ❌ No vehicle states to write! "
                           f"Total frames: {len(self.frame_buffer)}, "
                           f"Sample frame vehicle_state is None: {sample_frame.vehicle_state is None}")
                if sample_frame.vehicle_state is None:
                    logger.error(f"[RECORDER] ❌ CRITICAL: vehicle_state is None in RecordingFrame! "
                               f"This means VehicleState creation failed or was skipped.")
        
        # Write control commands
        if control_commands:
            try:
                self._write_control_commands(control_commands)
            except Exception as e:
                logger.error(f"Error writing control commands: {e}", exc_info=True)
        
        # Write perception outputs
        if perception_outputs:
            try:
                self._write_perception_outputs(perception_outputs)
            except Exception as e:
                logger.error(f"Error writing perception outputs: {e}", exc_info=True)
        
        # Write trajectory outputs
        if trajectory_outputs:
            try:
                self._write_trajectory_outputs(trajectory_outputs)
            except Exception as e:
                logger.error(f"Error writing trajectory outputs: {e}", exc_info=True)
        
        # Write Unity feedback
        if unity_feedbacks:
            try:
                self._write_unity_feedback(unity_feedbacks)
            except Exception as e:
                logger.error(f"Error writing Unity feedback: {e}", exc_info=True)
        
        self.frame_buffer.clear()
        self.h5_file.flush()
    
    def _write_camera_frames(self, frames: List[RecordingFrame]):
        """Write camera frames to HDF5."""
        images = []
        timestamps = []
        frame_ids = []
        
        for frame in frames:
            img = frame.camera_frame.image
            # Resize if needed
            if img.shape[:2] != (480, 640):
                img = cv2.resize(img, (640, 480))
            images.append(img)
            timestamps.append(frame.camera_frame.timestamp)
            frame_ids.append(frame.camera_frame.frame_id)
        
        if images:
            images = np.array(images)
            timestamps = np.array(timestamps)
            frame_ids = np.array(frame_ids)
            
            # Resize datasets if needed
            current_size = self.h5_file["camera/images"].shape[0]
            new_size = current_size + len(images)
            
            self.h5_file["camera/images"].resize((new_size, 480, 640, 3))
            self.h5_file["camera/timestamps"].resize((new_size,))
            self.h5_file["camera/frame_ids"].resize((new_size,))
            
            # Write data
            self.h5_file["camera/images"][current_size:] = images
            self.h5_file["camera/timestamps"][current_size:] = timestamps
            self.h5_file["camera/frame_ids"][current_size:] = frame_ids
    
    def _write_vehicle_states(self, frames: List[RecordingFrame]):
        """Write vehicle states to HDF5."""
        positions = []
        rotations = []
        velocities = []
        angular_velocities = []
        speeds = []
        steering_angles = []
        motor_torques = []
        brake_torques = []
        timestamps = []
        
        # Ground truth data
        gt_left_lane_line_x = []
        gt_right_lane_line_x = []
        gt_lane_center_x = []
        gt_path_curvature = []
        gt_desired_heading = []
        camera_8m_screen_y = []  # NEW: Camera calibration data
        camera_field_of_view = []  # NEW: Camera FOV data
        camera_horizontal_fov = []  # NEW: Camera horizontal FOV data
        camera_pos_x = []  # NEW: Camera position data
        camera_pos_y = []
        camera_pos_z = []
        camera_forward_x = []  # NEW: Camera forward direction data
        camera_forward_y = []
        camera_forward_z = []
        # NEW: Debug fields for diagnosing ground truth offset issues
        road_center_at_car_x = []
        road_center_at_car_y = []
        road_center_at_car_z = []
        road_center_at_lookahead_x = []
        road_center_at_lookahead_y = []
        road_center_at_lookahead_z = []
        road_center_reference_t = []
        
        for frame in frames:
            vs = frame.vehicle_state
            positions.append(vs.position)
            rotations.append(vs.rotation)
            velocities.append(vs.velocity)
            angular_velocities.append(vs.angular_velocity)
            speeds.append(vs.speed)
            steering_angles.append(vs.steering_angle)
            motor_torques.append(vs.motor_torque)
            brake_torques.append(vs.brake_torque)
            timestamps.append(vs.timestamp)
            
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
            
            # NEW: Extract camera FOV and position data
            camera_field_of_view.append(getattr(vs, 'camera_field_of_view', -1.0))
            camera_horizontal_fov.append(getattr(vs, 'camera_horizontal_fov', -1.0))
            camera_pos_x.append(getattr(vs, 'camera_pos_x', 0.0))
            camera_pos_y.append(getattr(vs, 'camera_pos_y', 0.0))
            camera_pos_z.append(getattr(vs, 'camera_pos_z', 0.0))
            camera_forward_x.append(getattr(vs, 'camera_forward_x', 0.0))
            camera_forward_y.append(getattr(vs, 'camera_forward_y', 0.0))
            camera_forward_z.append(getattr(vs, 'camera_forward_z', 0.0))
            # NEW: Debug fields for diagnosing ground truth offset issues
            road_center_at_car_x.append(getattr(vs, 'road_center_at_car_x', 0.0))
            road_center_at_car_y.append(getattr(vs, 'road_center_at_car_y', 0.0))
            road_center_at_car_z.append(getattr(vs, 'road_center_at_car_z', 0.0))
            road_center_at_lookahead_x.append(getattr(vs, 'road_center_at_lookahead_x', 0.0))
            road_center_at_lookahead_y.append(getattr(vs, 'road_center_at_lookahead_y', 0.0))
            road_center_at_lookahead_z.append(getattr(vs, 'road_center_at_lookahead_z', 0.0))
            road_center_reference_t.append(getattr(vs, 'road_center_reference_t', 0.0))
            # Debug: Log first few frames to see what we're getting
            if len(camera_8m_screen_y) <= 3:
                logger.debug(f"[RECORDER] Frame {len(camera_8m_screen_y)-1}: camera_8m_screen_y = {cam_value} (from VehicleState)")
        
        if positions:
            current_size = self.h5_file["vehicle/timestamps"].shape[0]
            new_size = current_size + len(positions)
            
            # Resize all vehicle datasets
            for key in ["timestamps", "position", "rotation", "velocity", 
                       "angular_velocity", "speed", "steering_angle", 
                       "motor_torque", "brake_torque", "camera_8m_screen_y"]:
                if key == "timestamps" or key in ["speed", "steering_angle", 
                                                  "motor_torque", "brake_torque", "camera_8m_screen_y"]:
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
            self.h5_file["vehicle/steering_angle"][current_size:] = steering_angles
            self.h5_file["vehicle/motor_torque"][current_size:] = motor_torques
            self.h5_file["vehicle/brake_torque"][current_size:] = brake_torques
            
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
                    # CRITICAL: Ensure the array has the correct shape (should be 1D with length matching positions)
                    if camera_8m_screen_y_array.shape[0] == len(positions):
                        self.h5_file["vehicle/camera_8m_screen_y"][current_size:] = camera_8m_screen_y_array
                        logger.debug(f"[RECORDER] Wrote {len(camera_8m_screen_y)} camera_8m_screen_y values (min={camera_8m_screen_y_array.min():.1f}, max={camera_8m_screen_y_array.max():.1f})")
                    else:
                        logger.warning(f"[RECORDER] camera_8m_screen_y length mismatch: {len(camera_8m_screen_y)} vs {len(positions)}. Skipping write.")
                else:
                    logger.warning(f"[RECORDER] camera_8m_screen_y is empty! Expected {len(positions)} values. Filling with -1.0.")
                    # Fill with -1.0 to indicate no data
                    self.h5_file["vehicle/camera_8m_screen_y"][current_size:] = np.full(len(positions), -1.0, dtype=np.float32)
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
        pid_integrals = []
        pid_derivatives = []
        pid_errors = []
        lateral_errors = []
        heading_errors = []
        total_errors = []
        calculated_steering_angles = []
        raw_steerings = []
        lateral_corrections = []
        path_curvature_inputs = []
        # NEW: Control stale data tracking
        using_stale_perception_list = []
        stale_perception_reason_list = []
        
        for frame in frames:
            cc = frame.control_command
            timestamps.append(cc.timestamp)
            steerings.append(cc.steering)
            throttles.append(cc.throttle)
            brakes.append(cc.brake)
            steering_before.append(cc.steering_before_limits if cc.steering_before_limits is not None else cc.steering)
            throttle_before.append(cc.throttle_before_limits if cc.throttle_before_limits is not None else cc.throttle)
            brake_before.append(cc.brake_before_limits if cc.brake_before_limits is not None else cc.brake)
            pid_integrals.append(cc.pid_integral if cc.pid_integral is not None else 0.0)
            pid_derivatives.append(cc.pid_derivative if cc.pid_derivative is not None else 0.0)
            pid_errors.append(cc.pid_error if cc.pid_error is not None else 0.0)
            lateral_errors.append(cc.lateral_error if cc.lateral_error is not None else 0.0)
            heading_errors.append(cc.heading_error if cc.heading_error is not None else 0.0)
            total_errors.append(cc.total_error if cc.total_error is not None else 0.0)
            calculated_steering_angles.append(cc.calculated_steering_angle_deg if cc.calculated_steering_angle_deg is not None else 0.0)
            raw_steerings.append(cc.raw_steering if cc.raw_steering is not None else 0.0)
            lateral_corrections.append(cc.lateral_correction if cc.lateral_correction is not None else 0.0)
            path_curvature_inputs.append(cc.path_curvature_input if cc.path_curvature_input is not None else 0.0)
            # NEW: Control stale data tracking
            using_stale_perception_list.append(cc.using_stale_perception if hasattr(cc, 'using_stale_perception') else False)
            stale_perception_reason_list.append(cc.stale_perception_reason if hasattr(cc, 'stale_perception_reason') and cc.stale_perception_reason else "")
        
        if timestamps:
            current_size = self.h5_file["control/timestamps"].shape[0]
            new_size = current_size + len(timestamps)
            
            # Resize all control datasets
            for key in ["timestamps", "steering", "throttle", "brake", 
                       "steering_before_limits", "throttle_before_limits", "brake_before_limits",
                       "pid_integral", "pid_derivative", "pid_error",
                       "lateral_error", "heading_error", "total_error",
                       "calculated_steering_angle_deg", "raw_steering", 
                       "lateral_correction", "path_curvature_input",
                       "using_stale_perception", "stale_perception_reason"]:
                self.h5_file[f"control/{key}"].resize((new_size,))
            
            # Write data
            self.h5_file["control/timestamps"][current_size:] = timestamps
            self.h5_file["control/steering"][current_size:] = steerings
            self.h5_file["control/throttle"][current_size:] = throttles
            self.h5_file["control/brake"][current_size:] = brakes
            self.h5_file["control/steering_before_limits"][current_size:] = steering_before
            self.h5_file["control/throttle_before_limits"][current_size:] = throttle_before
            self.h5_file["control/brake_before_limits"][current_size:] = brake_before
            self.h5_file["control/pid_integral"][current_size:] = pid_integrals
            self.h5_file["control/pid_derivative"][current_size:] = pid_derivatives
            self.h5_file["control/pid_error"][current_size:] = pid_errors
            self.h5_file["control/lateral_error"][current_size:] = lateral_errors
            self.h5_file["control/heading_error"][current_size:] = heading_errors
            self.h5_file["control/total_error"][current_size:] = total_errors
            self.h5_file["control/calculated_steering_angle_deg"][current_size:] = calculated_steering_angles
            self.h5_file["control/raw_steering"][current_size:] = raw_steerings
            self.h5_file["control/lateral_correction"][current_size:] = lateral_corrections
            self.h5_file["control/path_curvature_input"][current_size:] = path_curvature_inputs
            # NEW: Write control stale data
            self.h5_file["control/using_stale_perception"][current_size:] = using_stale_perception_list
            stale_perception_reason_array = np.array(stale_perception_reason_list, dtype=h5py.string_dtype(encoding='utf-8', length=50))
            self.h5_file["control/stale_perception_reason"][current_size:] = stale_perception_reason_array
    
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
        left_jump_magnitude_list = []
        right_jump_magnitude_list = []
        jump_threshold_list = []
        
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
            left_jump_magnitude_list.append(po.left_jump_magnitude if hasattr(po, 'left_jump_magnitude') and po.left_jump_magnitude is not None else 0.0)
            right_jump_magnitude_list.append(po.right_jump_magnitude if hasattr(po, 'right_jump_magnitude') and po.right_jump_magnitude is not None else 0.0)
            jump_threshold_list.append(po.jump_threshold if hasattr(po, 'jump_threshold') and po.jump_threshold is not None else 0.0)
            
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
            self.h5_file["perception/left_jump_magnitude"].resize((new_size,))
            self.h5_file["perception/right_jump_magnitude"].resize((new_size,))
            self.h5_file["perception/jump_threshold"].resize((new_size,))
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
            self.h5_file["perception/left_jump_magnitude"][current_size:] = left_jump_magnitude_list
            self.h5_file["perception/right_jump_magnitude"][current_size:] = right_jump_magnitude_list
            self.h5_file["perception/jump_threshold"][current_size:] = jump_threshold_list
            
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
            else:
                # No reference point for this frame - append zeros
                ref_points.append([0.0, 0.0, 0.0, 0.0])
                ref_points_raw.append([0.0, 0.0, 0.0])
                ref_methods.append('none')
                perception_centers.append(0.0)
        
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
        
        # NEW: Write trajectory points (full path, not just reference point)
        trajectory_points_list = []
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
            self.flush()
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

