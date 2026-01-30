"""
FastAPI server for Unity-Python communication bridge.
Handles camera frames, vehicle state, and control commands.
"""

import asyncio
import base64
import io
import time
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from PIL import Image

app = FastAPI(title="AV Stack Bridge Server")

# Log slow requests to identify Unity↔Python stalls.
SLOW_REQUEST_SECONDS = 0.05
UNITY_TIME_GAP_SECONDS = 0.2


def _get_bridge_logger() -> logging.Logger:
    log_path = Path(__file__).resolve().parents[1] / "tmp" / "logs" / "av_bridge.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    bridge_logger = logging.getLogger("av_bridge")
    bridge_logger.setLevel(logging.INFO)

    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path)
               for h in bridge_logger.handlers):
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        bridge_logger.addHandler(handler)
        bridge_logger.propagate = False

    return bridge_logger


logger = _get_bridge_logger()

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
latest_frame_id: Optional[str] = None
latest_vehicle_state: Optional[dict] = None
latest_control_command: Optional[dict] = None
latest_trajectory_data: Optional[dict] = None  # Trajectory data for visualization
latest_unity_feedback: Optional[dict] = None  # Unity feedback/status data
shutdown_requested: bool = False  # Track if AV stack is shutting down
play_requested: bool = False  # Track if Unity should start playing
camera_decode_in_flight: bool = False
camera_drop_count: int = 0
last_camera_arrival_time: Optional[float] = None
last_camera_frame_id: Optional[str] = None
last_camera_timestamp: Optional[float] = None
last_camera_realtime: Optional[float] = None
last_camera_unscaled: Optional[float] = None
last_state_arrival_time: Optional[float] = None
last_unity_send_realtime: Optional[float] = None
last_unity_time: Optional[float] = None
last_control_arrival_time: Optional[float] = None
speed_limit_zero_streak: int = 0


async def _decode_and_store_camera_frame(
    image_data: bytes,
    timestamp: str,
    frame_id: str,
) -> None:
    global latest_camera_frame, latest_frame_timestamp, latest_frame_id, camera_decode_in_flight

    start_time = time.time()
    try:
        img = Image.open(io.BytesIO(image_data))
        img_array = np.array(img)
        latest_camera_frame = img_array
        latest_frame_timestamp = float(timestamp)
        latest_frame_id = frame_id
    finally:
        duration = time.time() - start_time
        if duration > SLOW_REQUEST_SECONDS:
            logger.warning(
                "[SLOW] /api/camera decode "
                "duration=%.3fs bytes=%d frame_id=%s ts=%s",
                duration,
                len(image_data),
                frame_id,
                timestamp,
            )
        camera_decode_in_flight = False


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
    # Vehicle parameters from Unity
    maxSteerAngle: float = 30.0
    wheelbaseMeters: float = 2.5
    fixedDeltaTime: float = 0.02
    unityTime: float = 0.0
    unityFrameCount: int = 0
    unityDeltaTime: float = 0.0
    unitySmoothDeltaTime: float = 0.0
    unityUnscaledDeltaTime: float = 0.0
    unityTimeScale: float = 1.0
    requestId: int = 0
    unitySendRealtime: float = 0.0
    unitySendUtcMs: int = 0
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
    # NEW: Camera calibration - screen y pixel for ground truth lookahead
    cameraLookaheadScreenY: float = -1.0
    # NEW: Ground truth lookahead distance used for calibration
    groundTruthLookaheadDistance: float = 8.0
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
    speedLimit: float = 0.0  # Speed limit at current reference point (m/s)
    speedLimitPreview: float = 0.0  # Speed limit at preview distance ahead (m/s)
    speedLimitPreviewDistance: float = 0.0  # Preview distance used for speed limit (m)


class ControlCommand(BaseModel):
    """Control command to Unity."""
    steering: float  # -1.0 to 1.0
    throttle: float  # -1.0 to 1.0
    brake: float     # 0.0 to 1.0
    ground_truth_mode: bool = False  # Enable direct velocity control
    ground_truth_speed: float = 5.0  # Speed for ground truth mode (m/s)
    randomize_start: bool = False  # Randomize car start on oval
    randomize_request_id: int = 0  # Deduplicate randomization requests
    randomize_seed: Optional[int] = None  # Optional seed for repeatability
    emergency_stop: bool = False  # Emergency stop flag from AV stack


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
    frame_id: str = Form(...),
    realtime_since_startup: Optional[str] = Form(None),
    unscaled_time: Optional[str] = Form(None),
    time_scale: Optional[str] = Form(None)
):
    """
    Receive camera frame from Unity.
    
    Args:
        image: JPEG image file
        timestamp: Frame timestamp
        frame_id: Frame ID
    """
    global latest_camera_frame, latest_frame_timestamp
    
    start_time = time.time()
    try:
        # Read image data
        image_data = await image.read()

        global last_camera_arrival_time, last_camera_frame_id, last_camera_timestamp
        global last_camera_realtime, last_camera_unscaled
        now = time.time()
        if last_camera_arrival_time is not None:
            gap = now - last_camera_arrival_time
            if gap > UNITY_TIME_GAP_SECONDS:
                logger.warning(
                    "[ARRIVAL_GAP] /api/camera gap=%.3fs frame_id=%s prev_frame_id=%s",
                    gap,
                    frame_id,
                    last_camera_frame_id,
                )
                print(
                    "[ARRIVAL_GAP] /api/camera "
                    f"gap={gap:.3f}s frame_id={frame_id} prev_frame_id={last_camera_frame_id}"
                )
        last_camera_arrival_time = now
        last_camera_frame_id = frame_id
        try:
            ts_value = float(timestamp)
        except ValueError:
            ts_value = None
        if ts_value is not None:
            if last_camera_timestamp is not None:
                ts_gap = ts_value - last_camera_timestamp
                if ts_gap > UNITY_TIME_GAP_SECONDS:
                    logger.warning(
                        "[CAMERA_TIMESTAMP_GAP] ts_gap=%.3fs frame_id=%s",
                        ts_gap,
                        frame_id,
                    )
                    print(
                        "[CAMERA_TIMESTAMP_GAP] "
                        f"ts_gap={ts_gap:.3f}s frame_id={frame_id}"
                    )
            last_camera_timestamp = ts_value

        def _parse_optional(value: Optional[str]) -> Optional[float]:
            if value is None:
                return None
            try:
                return float(value)
            except ValueError:
                return None

        realtime_value = _parse_optional(realtime_since_startup)
        unscaled_value = _parse_optional(unscaled_time)
        if realtime_value is not None and last_camera_realtime is not None:
            rt_gap = realtime_value - last_camera_realtime
            if rt_gap > UNITY_TIME_GAP_SECONDS:
                logger.warning(
                    "[CAMERA_REALTIME_GAP] rt_gap=%.3fs frame_id=%s",
                    rt_gap,
                    frame_id,
                )
        if unscaled_value is not None and last_camera_unscaled is not None:
            us_gap = unscaled_value - last_camera_unscaled
            if us_gap > UNITY_TIME_GAP_SECONDS:
                logger.warning(
                    "[CAMERA_UNSCALED_GAP] us_gap=%.3fs frame_id=%s",
                    us_gap,
                    frame_id,
                )
        if ts_value is not None and realtime_value is not None:
            drift = ts_value - realtime_value
            if abs(drift) > 0.2:
                logger.warning(
                    "[CAMERA_TIME_DRIFT] ts=%.3f realtime=%.3f drift=%.3f frame_id=%s scale=%s",
                    ts_value,
                    realtime_value,
                    drift,
                    frame_id,
                    time_scale or "n/a",
                )
        last_camera_realtime = realtime_value or last_camera_realtime
        last_camera_unscaled = unscaled_value or last_camera_unscaled

        try:
            frame_id_int = int(frame_id)
        except ValueError:
            frame_id_int = None
        if frame_id_int is not None and frame_id_int % 100 == 0:
            logger.info(
                "[CAMERA_SAMPLE] frame_id=%s ts=%s realtime=%s unscaled=%s scale=%s arrival=%.3f",
                frame_id,
                f"{ts_value:.3f}" if ts_value is not None else "n/a",
                f"{realtime_value:.3f}" if realtime_value is not None else "n/a",
                f"{unscaled_value:.3f}" if unscaled_value is not None else "n/a",
                time_scale or "n/a",
                now,
            )

        # Avoid blocking the event loop on JPEG decode.
        # If a decode is already in-flight, drop this frame.
        global camera_decode_in_flight, camera_drop_count
        if camera_decode_in_flight:
            camera_drop_count += 1
            response = {
                "status": "received",
                "frame_id": frame_id,
                "timestamp": timestamp,
                "dropped": True,
                "dropped_count": camera_drop_count,
            }
        else:
            camera_decode_in_flight = True
            asyncio.create_task(_decode_and_store_camera_frame(image_data, timestamp, frame_id))
            response = {
                "status": "received",
                "frame_id": frame_id,
                "timestamp": timestamp,
                "dropped": False,
            }
        duration = time.time() - start_time
        if duration > SLOW_REQUEST_SECONDS:
            logger.warning(
                "[SLOW] /api/camera duration=%.3fs bytes=%d frame_id=%s ts=%s",
                duration,
                len(image_data),
                frame_id,
                timestamp,
            )
        return response
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
    start_time = time.time()

    latest_vehicle_state = state.model_dump()

    global last_state_arrival_time, last_unity_send_realtime, last_unity_time, speed_limit_zero_streak
    duration = time.time() - start_time
    now = time.time()
    if last_state_arrival_time is not None:
        gap = now - last_state_arrival_time
        if gap > SLOW_REQUEST_SECONDS:
            logger.warning(
                "[ARRIVAL_GAP] /api/vehicle/state gap=%.3fs request_id=%s unity_frame=%s unity_time=%.3f",
                gap,
                state.requestId,
                state.unityFrameCount,
                state.unityTime,
            )
            print(
                "[ARRIVAL_GAP] /api/vehicle/state "
                f"gap={gap:.3f}s request_id={state.requestId} "
                f"unity_frame={state.unityFrameCount} unity_time={state.unityTime:.3f}"
            )
    last_state_arrival_time = now
    if last_unity_time is not None:
        unity_time_gap = state.unityTime - last_unity_time
        if unity_time_gap > UNITY_TIME_GAP_SECONDS:
            logger.warning(
                "[UNITY_TIME_GAP] unity_time_gap=%.3fs request_id=%s unity_frame=%s",
                unity_time_gap,
                state.requestId,
                state.unityFrameCount,
            )
            print(
                "[UNITY_TIME_GAP] "
                f"unity_time_gap={unity_time_gap:.3f}s request_id={state.requestId} "
                f"unity_frame={state.unityFrameCount}"
            )

    if state.unitySendRealtime > 0.0 and last_unity_send_realtime is not None:
        send_gap = state.unitySendRealtime - last_unity_send_realtime
        unity_time_gap = state.unityTime - (last_unity_time or 0.0)
        drift = unity_time_gap - send_gap
        if abs(drift) > 0.2:
            logger.warning(
                "[TIME_DRIFT] unity_time_gap=%.3fs send_gap=%.3fs drift=%.3fs "
                "request_id=%s unity_frame=%s",
                unity_time_gap,
                send_gap,
                drift,
                state.requestId,
                state.unityFrameCount,
            )
            print(
                "[TIME_DRIFT] "
                f"unity_time_gap={unity_time_gap:.3f}s send_gap={send_gap:.3f}s "
                f"drift={drift:.3f}s request_id={state.requestId} "
                f"unity_frame={state.unityFrameCount}"
            )
    if state.unitySendRealtime > 0.0:
        last_unity_send_realtime = state.unitySendRealtime
        last_unity_time = state.unityTime
    if state.unitySendUtcMs:
        now_ms = int(time.time() * 1000)
        arrival_delay_ms = now_ms - int(state.unitySendUtcMs)
        if arrival_delay_ms > 100:
            logger.warning(
                "[ARRIVAL] /api/vehicle/state delay=%dms request_id=%s unity_frame=%s unity_time=%.3f",
                arrival_delay_ms,
                state.requestId,
                state.unityFrameCount,
                state.unityTime,
            )
            print(
                "[ARRIVAL] /api/vehicle/state "
                f"delay={arrival_delay_ms}ms request_id={state.requestId} "
                f"unity_frame={state.unityFrameCount} unity_time={state.unityTime:.3f}"
            )
    if duration > SLOW_REQUEST_SECONDS:
        logger.warning(
            "[SLOW] /api/vehicle/state duration=%.3fs unity_frame=%s unity_time=%.3f",
            duration,
            state.unityFrameCount,
            state.unityTime,
        )

    if state.speedLimit <= 0.01:
        speed_limit_zero_streak += 1
        if speed_limit_zero_streak % 60 == 0:
            logger.warning(
                "[SPEED_LIMIT_MISSING] streak=%d unity_frame=%s unity_time=%.3f",
                speed_limit_zero_streak,
                state.unityFrameCount,
                state.unityTime,
            )
            print(
                "[SPEED_LIMIT_MISSING] "
                f"streak={speed_limit_zero_streak} unity_frame={state.unityFrameCount} "
                f"unity_time={state.unityTime:.3f}"
            )
    else:
        speed_limit_zero_streak = 0

    return {"status": "received"}


@app.get("/api/vehicle/control")
async def get_control_command(request_id: int = 0, unity_send_utc_ms: int = 0):
    """
    Get latest control command for Unity.
    
    Returns:
        Control command (steering, throttle, brake)
    """
    global latest_control_command, last_control_arrival_time
    start_time = time.time()
    now = time.time()
    if last_control_arrival_time is not None:
        gap = now - last_control_arrival_time
        if gap > SLOW_REQUEST_SECONDS:
            logger.warning(
                "[ARRIVAL_GAP] /api/vehicle/control gap=%.3fs request_id=%s",
                gap,
                request_id,
            )
            print(
                "[ARRIVAL_GAP] /api/vehicle/control "
                f"gap={gap:.3f}s request_id={request_id}"
            )
    last_control_arrival_time = now

    if unity_send_utc_ms:
        now_ms = int(time.time() * 1000)
        arrival_delay_ms = now_ms - int(unity_send_utc_ms)
        if arrival_delay_ms > 100:
            logger.warning(
                "[ARRIVAL] /api/vehicle/control delay=%dms request_id=%s",
                arrival_delay_ms,
                request_id,
            )
            print(
                "[ARRIVAL] /api/vehicle/control "
                f"delay={arrival_delay_ms}ms request_id={request_id}"
            )
    
    if latest_control_command is None:
        # Return neutral command if no command available
        response = {
            "steering": 0.0,
            "throttle": 0.0,
            "brake": 0.0,
            "ground_truth_mode": False,
            "ground_truth_speed": 5.0,
            "randomize_start": False,
            "randomize_request_id": 0,
            "randomize_seed": None,
        }
        duration = time.time() - start_time
        if duration > SLOW_REQUEST_SECONDS:
            logger.warning(
                "[SLOW] /api/vehicle/control duration=%.3fs (default)",
                duration,
            )
        return response
    
    # Ensure ground truth fields are present (for backward compatibility)
    result = latest_control_command.copy()
    if "ground_truth_mode" not in result:
        result["ground_truth_mode"] = False
    if "ground_truth_speed" not in result:
        result["ground_truth_speed"] = 5.0
    if "randomize_start" not in result:
        result["randomize_start"] = False
    if "randomize_request_id" not in result:
        result["randomize_request_id"] = 0
    if "randomize_seed" not in result:
        result["randomize_seed"] = None
    
    duration = time.time() - start_time
    if duration > SLOW_REQUEST_SECONDS:
        logger.warning(
            "[SLOW] /api/vehicle/control duration=%.3fs",
            duration,
        )
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

