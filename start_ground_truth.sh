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
VERBOSE=false
RANDOM_START=false
RANDOM_SEED=""
CONSTANT_SPEED=false

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
    echo "  --verbose                   Enable verbose ground truth logging"
    echo "  --constant-speed            Disable GT PID and run constant speed"
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
        --verbose)
            VERBOSE=true
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
"$UNITY_BUILD_PATH/Contents/MacOS/AVSimulation" --track-yaml "$TRACK_YAML_PATH" &
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

if [ "$VERBOSE" = true ]; then
    echo "Starting ground truth follower (verbose)..."
    "$PYTHON_BIN" -u tools/ground_truth_follower.py "${GT_ARGS[@]}" --verbose
else
    echo "Starting ground truth follower (quiet unless errors)..."
    "$PYTHON_BIN" tools/ground_truth_follower.py "${GT_ARGS[@]}"
fi
echo "Ground truth run finished."
