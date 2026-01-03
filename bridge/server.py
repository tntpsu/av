"""
FastAPI server for Unity-Python communication bridge.
Handles camera frames, vehicle state, and control commands.
"""

import asyncio
import base64
import io
import time
from typing import Optional
from datetime import datetime

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from PIL import Image

app = FastAPI(title="AV Stack Bridge Server")

# Enable CORS for Unity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
latest_camera_frame: Optional[np.ndarray] = None
latest_frame_timestamp: Optional[float] = None
latest_vehicle_state: Optional[dict] = None
latest_control_command: Optional[dict] = None
latest_trajectory_data: Optional[dict] = None  # Trajectory data for visualization
latest_unity_feedback: Optional[dict] = None  # Unity feedback/status data
shutdown_requested: bool = False  # Track if AV stack is shutting down
play_requested: bool = False  # Track if Unity should start playing


class VehicleState(BaseModel):
    """Vehicle state from Unity."""
    position: dict  # {"x": float, "y": float, "z": float}
    rotation: dict  # {"x": float, "y": float, "z": float, "w": float}
    velocity: dict  # {"x": float, "y": float, "z": float}
    angularVelocity: dict
    speed: float
    steeringAngle: float
    motorTorque: float
    brakeTorque: float
    # Ground truth lane line positions (optional)
    # These represent the painted lane line markings, not the drivable lanes
    groundTruthLeftLaneLineX: float = 0.0  # Left lane line (painted marking) position
    groundTruthRightLaneLineX: float = 0.0  # Right lane line (painted marking) position
    groundTruthLaneCenterX: float = 0.0  # Lane center (midpoint between lane lines)
    # NEW: Path-based steering data
    groundTruthDesiredHeading: float = 0.0  # Desired heading from path (degrees)
    groundTruthPathCurvature: float = 0.0  # Path curvature (1/meters)
    # NEW: Camera calibration - actual screen y pixel where 8m appears (from Unity's WorldToScreenPoint)
    camera8mScreenY: float = -1.0  # -1.0 means not calculated yet
    # NEW: Camera FOV information - what Unity actually uses
    cameraFieldOfView: float = 0.0  # Unity's Camera.fieldOfView value (always vertical FOV)
    cameraHorizontalFOV: float = 0.0  # Calculated horizontal FOV
    # NEW: Camera position and forward for debugging alignment
    cameraPosX: float = 0.0  # Camera position X (world coords)
    cameraPosY: float = 0.0  # Camera position Y (world coords)
    cameraPosZ: float = 0.0  # Camera position Z (world coords)
    cameraForwardX: float = 0.0  # Camera forward X (normalized)
    cameraForwardY: float = 0.0  # Camera forward Y (normalized)
    cameraForwardZ: float = 0.0  # Camera forward Z (normalized)
    # NEW: Debug fields for diagnosing ground truth offset issues
    roadCenterAtCarX: float = 0.0  # Road center X at car's location (world coords)
    roadCenterAtCarY: float = 0.0  # Road center Y at car's location (world coords)
    roadCenterAtCarZ: float = 0.0  # Road center Z at car's location (world coords)
    roadCenterAtLookaheadX: float = 0.0  # Road center X at 8m lookahead (world coords)
    roadCenterAtLookaheadY: float = 0.0  # Road center Y at 8m lookahead (world coords)
    roadCenterAtLookaheadZ: float = 0.0  # Road center Z at 8m lookahead (world coords)
    roadCenterReferenceT: float = 0.0  # Parameter t on road path for reference point


class ControlCommand(BaseModel):
    """Control command to Unity."""
    steering: float  # -1.0 to 1.0
    throttle: float  # -1.0 to 1.0
    brake: float     # 0.0 to 1.0
    ground_truth_mode: bool = False  # Enable direct velocity control
    ground_truth_speed: float = 5.0  # Speed for ground truth mode (m/s)


class GroundTruthMode(BaseModel):
    """Ground truth mode configuration."""
    enabled: bool
    speed: float = 5.0  # Speed in m/s for ground truth mode


class UnityFeedback(BaseModel):
    """Unity feedback/status data."""
    timestamp: float
    # Control status
    ground_truth_mode_active: bool = False
    control_command_received: bool = False
    actual_steering_applied: Optional[float] = None
    actual_throttle_applied: Optional[float] = None
    actual_brake_applied: Optional[float] = None
    # Ground truth data status
    ground_truth_data_available: bool = False
    ground_truth_reporter_enabled: bool = False
    path_curvature_calculated: bool = False
    # Errors/warnings
    unity_errors: Optional[str] = None
    unity_warnings: Optional[str] = None
    # Internal state
    car_controller_mode: Optional[str] = None
    av_control_enabled: bool = False
    # Frame info
    unity_frame_count: Optional[int] = None
    unity_time: Optional[float] = None


@app.post("/api/camera")
async def receive_camera_frame(
    image: UploadFile = File(...),
    timestamp: str = Form(...),
    frame_id: str = Form(...)
):
    """
    Receive camera frame from Unity.
    
    Args:
        image: JPEG image file
        timestamp: Frame timestamp
        frame_id: Frame ID
    """
    global latest_camera_frame, latest_frame_timestamp
    
    try:
        # Read image data
        image_data = await image.read()
        
        # Convert to numpy array
        img = Image.open(io.BytesIO(image_data))
        img_array = np.array(img)
        
        # Store latest frame
        latest_camera_frame = img_array
        latest_frame_timestamp = float(timestamp)
        
        return {
            "status": "received",
            "frame_id": frame_id,
            "timestamp": timestamp,
            "shape": list(img_array.shape)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/api/vehicle/state")
async def receive_vehicle_state(state: VehicleState):
    """
    Receive vehicle state from Unity.
    
    Args:
        state: Vehicle state data
    """
    global latest_vehicle_state
    
    latest_vehicle_state = state.model_dump()
    
    return {"status": "received"}


@app.get("/api/vehicle/control")
async def get_control_command():
    """
    Get latest control command for Unity.
    
    Returns:
        Control command (steering, throttle, brake)
    """
    global latest_control_command
    
    if latest_control_command is None:
        # Return neutral command if no command available
        return {
            "steering": 0.0,
            "throttle": 0.0,
            "brake": 0.0,
            "ground_truth_mode": False,
            "ground_truth_speed": 5.0
        }
    
    # Ensure ground truth fields are present (for backward compatibility)
    result = latest_control_command.copy()
    if "ground_truth_mode" not in result:
        result["ground_truth_mode"] = False
    if "ground_truth_speed" not in result:
        result["ground_truth_speed"] = 5.0
    
    return result


@app.post("/api/vehicle/control")
async def set_control_command(command: ControlCommand):
    """
    Set control command (called by AV stack).
    
    Args:
        command: Control command
    """
    global latest_control_command
    
    latest_control_command = command.model_dump()
    
    return {"status": "set"}


@app.get("/api/camera/latest")
async def get_latest_camera_frame():
    """
    Get latest camera frame (for AV stack processing).
    
    Returns:
        Base64 encoded image
    """
    global latest_camera_frame
    
    if latest_camera_frame is None:
        raise HTTPException(status_code=404, detail="No camera frame available")
    
    # Convert to JPEG
    img = Image.fromarray(latest_camera_frame)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()
    
    # Encode to base64
    img_base64 = base64.b64encode(img_bytes).decode()
    
    return {
        "image": img_base64,
        "timestamp": latest_frame_timestamp,
        "shape": list(latest_camera_frame.shape)
    }


@app.get("/api/vehicle/state/latest")
async def get_latest_vehicle_state():
    """
    Get latest vehicle state (for AV stack processing).
    
    Returns:
        Vehicle state data
    """
    global latest_vehicle_state
    
    if latest_vehicle_state is None:
        raise HTTPException(status_code=404, detail="No vehicle state available")
    
    return latest_vehicle_state


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "has_camera_frame": latest_camera_frame is not None,
        "has_vehicle_state": latest_vehicle_state is not None
    }


@app.post("/api/shutdown")
async def shutdown_signal():
    """
    Signal that AV stack is shutting down.
    Unity can poll this to detect when to exit play mode.
    """
    global shutdown_requested
    shutdown_requested = True
    return {
        "status": "shutdown",
        "message": "AV stack is shutting down",
        "timestamp": time.time()
    }


@app.get("/api/shutdown")
async def check_shutdown():
    """
    Check if AV stack is shutting down.
    Unity polls this to detect when to exit play mode.
    Returns shutdown status only if shutdown was requested.
    """
    global shutdown_requested
    if shutdown_requested:
        return {
            "status": "shutdown",
            "message": "AV stack is shutting down",
            "timestamp": time.time()
        }
    else:
        return {
            "status": "running",
            "message": "AV stack is running",
            "timestamp": time.time()
        }


@app.post("/api/unity/play")
async def request_play():
    """
    Request Unity to start playing (enter play mode).
    Unity can poll this to detect when to enter play mode.
    """
    global play_requested
    play_requested = True
    return {
        "status": "play_requested",
        "message": "Unity play mode requested",
        "timestamp": time.time()
    }


@app.get("/api/unity/play")
async def check_play_request():
    """
    Check if Unity should start playing.
    Unity polls this to detect when to enter play mode.
    Returns play status only if play was requested.
    """
    global play_requested
    if play_requested:
        # Reset flag after reading (one-time request)
        play_requested = False
        return {
            "status": "play",
            "message": "Unity should enter play mode",
            "timestamp": time.time()
        }
    else:
        return {
            "status": "no_action",
            "message": "No play request",
            "timestamp": time.time()
        }


@app.post("/api/trajectory")
async def set_trajectory_data(trajectory: dict):
    """
    Set trajectory data (called by AV stack for visualization).
    
    Args:
        trajectory: Trajectory data with points, reference_point, lateral_error
    """
    global latest_trajectory_data
    latest_trajectory_data = trajectory
    return {"status": "received"}


@app.get("/api/trajectory")
async def get_trajectory_data():
    """
    Get latest trajectory data for Unity visualization.
    
    Returns:
        Trajectory data with points, reference_point, lateral_error
    """
    global latest_trajectory_data
    
    if latest_trajectory_data is None:
        # Return empty trajectory if no data available
        return {
            "trajectory_points": [],
            "reference_point": [0.0, 0.0, 0.0, 0.0],
            "lateral_error": 0.0,
            "timestamp": time.time()
        }
    
    return latest_trajectory_data


@app.post("/api/groundtruth/mode")
async def set_ground_truth_mode(mode: GroundTruthMode):
    """
    Enable or disable ground truth mode in Unity CarController.
    
    Ground truth mode uses direct velocity control for precise path following,
    bypassing physics forces. This is ideal for collecting ground truth data.
    
    Args:
        mode: Ground truth mode configuration (enabled, speed)
    """
    # This will be handled by Unity's AVBridge when it polls for control commands
    # We'll store it in the control command so Unity can read it
    global latest_control_command
    
    # Store ground truth mode in control command metadata
    if latest_control_command is None:
        latest_control_command = {
            "steering": 0.0,
            "throttle": 0.0,
            "brake": 0.0
        }
    
    latest_control_command["ground_truth_mode"] = mode.enabled
    latest_control_command["ground_truth_speed"] = mode.speed
    
    return {
        "status": "set",
        "ground_truth_mode": mode.enabled,
        "speed": mode.speed
    }


@app.post("/api/unity/feedback")
async def receive_unity_feedback(feedback: UnityFeedback):
    """
    Receive Unity feedback/status data.
    
    Args:
        feedback: Unity feedback data
    """
    global latest_unity_feedback
    
    latest_unity_feedback = feedback.model_dump()
    
    return {"status": "received"}


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the bridge server."""
    print(f"Starting AV Stack Bridge Server on {host}:{port}")
    print("Endpoints:")
    print("  POST /api/camera - Receive camera frame from Unity")
    print("  POST /api/vehicle/state - Receive vehicle state from Unity")
    print("  GET  /api/vehicle/control - Get control command for Unity")
    print("  POST /api/vehicle/control - Set control command")
    print("  GET  /api/camera/latest - Get latest camera frame")
    print("  GET  /api/vehicle/state/latest - Get latest vehicle state")
    print("  GET  /api/health - Health check")
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()

