#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

UNITY_BUILD_PATH="$SCRIPT_DIR/unity/AVSimulation/mybuild.app"
TRACK_YAML_PATH="$SCRIPT_DIR/tracks/oval.yml"
UNITY_CLI_PATH=""
BUILD_UNITY_PLAYER=false
SKIP_UNITY_BUILD_IF_CLEAN=false

DURATION=20
SPEED=8.0
ARC_RADIUS=""
LOG_LEVEL="error"
RANDOM_START=false
RANDOM_SEED=""
CONSTANT_SPEED=false
TARGET_LANE="right"
USE_CV=false
SEG_CHECKPOINT="$SCRIPT_DIR/data/segmentation_dataset/checkpoints/segnet_best.pt"
SINGLE_LANE_WIDTH_THRESHOLD="5.0"
LATERAL_CORRECTION_GAIN=""
GT_CENTERLINE_AS_LEFT_LANE="true"
STRICT_GT_POSE=false
STRICT_GT_FEEDBACK=false
FAST_RECORD=false
GT_SYNC_CAPTURE=true
GT_SYNC_FIXED_DELTA="0.033333333"
GT_DISABLE_TOPDOWN=false
GT_RECORD_STATE_LAG_FRAMES="0"
STREAM_SYNC_POLICY="aligned"
GT_JPEG_QUALITY="85"
GT_CAMERA_SEND_ASYNC="true"
GT_REDUCE_TOPDOWN_RATE="true"
GT_ROTATION_FROM_ROAD_FRAME="false"
GT_DISABLE_MOVE_ROTATION="false"

TMP_TRACK_PATH=""
UNITY_PID=""

print_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --track-yaml PATH           Track YAML to run (default: tracks/oval.yml)"
    echo "  --duration SECONDS          Run duration (default: 20)"
    echo "  --speed MPS                 Ground truth target speed (default: 8.0)"
    echo "  --random-start              Randomize start position"
    echo "  --random-seed SEED          Seed for random start"
    echo "  --log-level LEVEL           Ground-truth logging: error/info/debug (default: error)"
    echo "  --diagnostic-logging        Shortcut for --log-level info"
    echo "  --verbose                   Compatibility alias for --log-level info"
    echo "  --constant-speed            Disable GT PID and run constant speed"
    echo "  --target-lane LANE          Target lane center: right/left/center (default: right)"
    echo "  --use-cv                    Force CV-based perception (override default segmentation)"
    echo "  --segmentation-checkpoint P Segmentation checkpoint path"
    echo "  --single-lane-width-threshold M  Treat lane width below this as single lane"
    echo "  --lateral-correction-gain G  Optional lateral correction gain override"
    echo "  --gt-centerline-as-left-lane BOOL  Use center line as left lane (default: true)"
    echo "  --strict-gt-pose            Send zero manual controls; Unity GT reference path owns pose"
    echo "  --strict-gt-feedback        Fail if Unity GT feedback integrity is not continuously valid"
    echo "  --fast-record               Skip full stack compute; record GT/control/camera fast path"
    echo "  --gt-sync-capture BOOL      Enable deterministic GT sensor capture mode (default: true)"
    echo "  --gt-sync-fixed-delta SEC   Fixed timestep for GT sync mode (default: 0.033333333)"
    echo "  --gt-disable-topdown BOOL   Disable top-down camera capture for GT runs (default: false)"
    echo "  --gt-record-state-lag-frames N  Fast-record: use N-frame lagged vehicle state (default: 0)"
    echo "  --stream-sync-policy MODE  Stream sync policy: aligned/queued/latest (default: aligned)"
    echo "  --gt-jpeg-quality Q         JPEG quality for Unity camera upload (10-100, default: 85)"
    echo "  --gt-camera-send-async BOOL Queue camera uploads through single worker (default: true)"
    echo "  --gt-reduce-topdown-rate BOOL Reduce top-down camera capture cadence (default: true)"
    echo "  --gt-rotation-from-road-frame BOOL  Use road-frame tangent for GT rotation/velocity direction (default: false)"
    echo "  --gt-disable-move-rotation BOOL  Skip MoveRotation in GT mode for one-run heading experiment (default: false)"
    echo "  --arc-radius METERS         Override all arc radii and use a temp track"
    echo "  --build-unity-player        Build Unity player before running"
    echo "  --skip-unity-build-if-clean Skip Unity build if project is clean"
    echo "  --unity-path PATH           Unity CLI path for builds"
    echo "  --build-path PATH           Unity player output path"
    echo "  -h, --help                  Show this help"
}

cleanup() {
    if [ -n "$UNITY_PID" ] && kill -0 "$UNITY_PID" >/dev/null 2>&1; then
        kill "$UNITY_PID" >/dev/null 2>&1 || true
    fi
    if [ -n "$TMP_TRACK_PATH" ] && [ -f "$TMP_TRACK_PATH" ]; then
        rm -f "$TMP_TRACK_PATH"
    fi
}

trap cleanup EXIT

while [[ $# -gt 0 ]]; do
    case $1 in
        --track-yaml)
            TRACK_YAML_PATH="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --speed)
            SPEED="$2"
            shift 2
            ;;
        --arc-radius)
            ARC_RADIUS="$2"
            shift 2
            ;;
        --random-start)
            RANDOM_START=true
            shift
            ;;
        --random-seed)
            RANDOM_SEED="$2"
            shift 2
            ;;
        --constant-speed)
            CONSTANT_SPEED=true
            shift
            ;;
        --target-lane)
            TARGET_LANE="$2"
            shift 2
            ;;
        --use-cv)
            USE_CV=true
            shift
            ;;
        --segmentation-checkpoint)
            SEG_CHECKPOINT="$2"
            shift 2
            ;;
        --single-lane-width-threshold)
            SINGLE_LANE_WIDTH_THRESHOLD="$2"
            shift 2
            ;;
        --lateral-correction-gain)
            LATERAL_CORRECTION_GAIN="$2"
            shift 2
            ;;
        --gt-centerline-as-left-lane)
            GT_CENTERLINE_AS_LEFT_LANE="$2"
            shift 2
            ;;
        --strict-gt-pose)
            STRICT_GT_POSE=true
            shift
            ;;
        --strict-gt-feedback)
            STRICT_GT_FEEDBACK=true
            shift
            ;;
        --fast-record)
            FAST_RECORD=true
            shift
            ;;
        --gt-sync-capture)
            GT_SYNC_CAPTURE="$2"
            shift 2
            ;;
        --gt-sync-fixed-delta)
            GT_SYNC_FIXED_DELTA="$2"
            shift 2
            ;;
        --gt-disable-topdown)
            GT_DISABLE_TOPDOWN="$2"
            shift 2
            ;;
        --gt-record-state-lag-frames)
            GT_RECORD_STATE_LAG_FRAMES="$2"
            shift 2
            ;;
        --stream-sync-policy)
            STREAM_SYNC_POLICY="$2"
            shift 2
            ;;
        --gt-jpeg-quality)
            GT_JPEG_QUALITY="$2"
            shift 2
            ;;
        --gt-camera-send-async)
            GT_CAMERA_SEND_ASYNC="$2"
            shift 2
            ;;
        --gt-reduce-topdown-rate)
            GT_REDUCE_TOPDOWN_RATE="$2"
            shift 2
            ;;
        --gt-rotation-from-road-frame)
            GT_ROTATION_FROM_ROAD_FRAME="$2"
            shift 2
            ;;
        --gt-disable-move-rotation)
            GT_DISABLE_MOVE_ROTATION="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --diagnostic-logging)
            LOG_LEVEL="info"
            shift
            ;;
        --verbose)
            LOG_LEVEL="info"
            shift
            ;;
        --build-unity-player)
            BUILD_UNITY_PLAYER=true
            shift
            ;;
        --skip-unity-build-if-clean)
            SKIP_UNITY_BUILD_IF_CLEAN=true
            shift
            ;;
        --unity-path)
            UNITY_CLI_PATH="$2"
            shift 2
            ;;
        --build-path)
            UNITY_BUILD_PATH="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

if [[ "$TRACK_YAML_PATH" != /* ]]; then
    TRACK_YAML_PATH="$SCRIPT_DIR/$TRACK_YAML_PATH"
fi

case "$LOG_LEVEL" in
    error|info|debug) ;;
    *)
        echo "Invalid --log-level: $LOG_LEVEL (expected: error, info, or debug)"
        exit 1
        ;;
esac

case "$STREAM_SYNC_POLICY" in
    aligned|queued|latest) ;;
    *)
        echo "Invalid --stream-sync-policy: $STREAM_SYNC_POLICY (expected: aligned, queued, or latest)"
        exit 1
        ;;
esac

PYTHON_BIN="$SCRIPT_DIR/venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

echo "=== Ground Truth Runner ==="
echo "Track: $TRACK_YAML_PATH"
echo "Duration: ${DURATION}s | Speed: ${SPEED} m/s"

if [ -n "$ARC_RADIUS" ]; then
    mkdir -p "$SCRIPT_DIR/tracks/generated"
    TMP_TRACK_PATH="$SCRIPT_DIR/tracks/generated/ground_truth_r${ARC_RADIUS}.yml"
    "$PYTHON_BIN" - <<PY
from pathlib import Path
import yaml

track_path = Path(r"$TRACK_YAML_PATH")
data = yaml.safe_load(track_path.read_text())
for segment in data.get("segments", []):
    if segment.get("type") == "arc":
        segment["radius"] = float("$ARC_RADIUS")
data["name"] = f"{data.get('name', 'track')}_r{int(float('$ARC_RADIUS'))}"
out = Path(r"$TMP_TRACK_PATH")
out.write_text(yaml.safe_dump(data, sort_keys=False))
print(out)
PY
    TRACK_YAML_PATH="$TMP_TRACK_PATH"
    echo "Arc radius override: ${ARC_RADIUS}m -> $TRACK_YAML_PATH"
fi

if [ "$BUILD_UNITY_PLAYER" = true ]; then
    BUILD_ARGS=()
    if [ -n "$UNITY_CLI_PATH" ]; then
        BUILD_ARGS+=(--unity-path "$UNITY_CLI_PATH")
    fi
    if [ -n "$UNITY_BUILD_PATH" ]; then
        BUILD_ARGS+=(--build-path "$UNITY_BUILD_PATH")
    fi
    if [ "$SKIP_UNITY_BUILD_IF_CLEAN" = true ]; then
        BUILD_ARGS+=(--skip-if-clean)
    fi
    "$SCRIPT_DIR/build_unity_player.sh" "${BUILD_ARGS[@]}"
fi

if [ ! -d "$UNITY_BUILD_PATH" ]; then
    echo "Unity player not found at: $UNITY_BUILD_PATH"
    exit 1
fi

if command -v lsof >/dev/null 2>&1; then
    lsof -ti tcp:8000 | xargs -r kill -9 || true
fi

echo "Launching Unity player..."
"$UNITY_BUILD_PATH/Contents/MacOS/AVSimulation" \
    --track-yaml "$TRACK_YAML_PATH" \
    --gt-centerline-as-left-lane "$GT_CENTERLINE_AS_LEFT_LANE" \
    --gt-sync-capture "$GT_SYNC_CAPTURE" \
    --gt-sync-fixed-delta "$GT_SYNC_FIXED_DELTA" \
    --gt-disable-topdown "$GT_DISABLE_TOPDOWN" \
    --gt-jpeg-quality "$GT_JPEG_QUALITY" \
    --gt-camera-send-async "$GT_CAMERA_SEND_ASYNC" \
    --gt-reduce-topdown-rate "$GT_REDUCE_TOPDOWN_RATE" \
    --gt-rotation-from-road-frame "$GT_ROTATION_FROM_ROAD_FRAME" \
    --gt-disable-move-rotation "$GT_DISABLE_MOVE_ROTATION" &
UNITY_PID=$!

sleep 2

cd "$SCRIPT_DIR"
GT_ARGS=(--duration "$DURATION" --speed "$SPEED")
if [ "$RANDOM_START" = true ]; then
    GT_ARGS+=(--random-start)
fi
if [ -n "$RANDOM_SEED" ]; then
    GT_ARGS+=(--random-seed "$RANDOM_SEED")
fi
if [ "$CONSTANT_SPEED" = true ]; then
    GT_ARGS+=(--constant-speed)
fi
if [ -n "$TARGET_LANE" ]; then
    GT_ARGS+=(--target-lane "$TARGET_LANE")
fi
if [ "$USE_CV" = true ]; then
    GT_ARGS+=(--use-cv)
fi
if [ "$USE_CV" = false ] && [ -n "$SEG_CHECKPOINT" ]; then
    GT_ARGS+=(--segmentation-checkpoint "$SEG_CHECKPOINT")
fi
if [ -n "$SINGLE_LANE_WIDTH_THRESHOLD" ]; then
    GT_ARGS+=(--single-lane-width-threshold "$SINGLE_LANE_WIDTH_THRESHOLD")
fi
if [ -n "$LATERAL_CORRECTION_GAIN" ]; then
    GT_ARGS+=(--lateral-correction-gain "$LATERAL_CORRECTION_GAIN")
fi
if [ "$STRICT_GT_POSE" = true ]; then
    GT_ARGS+=(--strict-gt-pose)
fi
if [ "$STRICT_GT_FEEDBACK" = true ]; then
    GT_ARGS+=(--strict-gt-feedback)
fi
if [ "$FAST_RECORD" = true ]; then
    GT_ARGS+=(--fast-record)
fi
if [ -n "$GT_RECORD_STATE_LAG_FRAMES" ]; then
    GT_ARGS+=(--record-state-lag-frames "$GT_RECORD_STATE_LAG_FRAMES")
fi
if [ -n "$STREAM_SYNC_POLICY" ]; then
    GT_ARGS+=(--stream-sync-policy "$STREAM_SYNC_POLICY")
fi

echo "Starting ground truth follower (log level: $LOG_LEVEL)..."
"$PYTHON_BIN" tools/ground_truth_follower.py "${GT_ARGS[@]}" --log-level "$LOG_LEVEL"
echo "Ground truth run finished."
