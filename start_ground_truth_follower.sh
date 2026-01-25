#!/bin/bash

# Ground Truth Follower Startup Script
# Starts the bridge server and runs tools/ground_truth_follower.py

set -e

BRIDGE_URL="http://localhost:8000"
BRIDGE_PORT=8000
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/venv"
FORCE_KILL=false

usage() {
    echo "Usage: ./start_ground_truth_follower.sh [options] [ground_truth_follower.py args]"
    echo ""
    echo "Script options:"
    echo "  --force, -f            Auto-kill existing process on port 8000"
    echo "  --help, -h             Show this help message"
    echo ""
    echo "ground_truth_follower.py args are passed through, e.g.:"
    echo "  --duration 60 --output path_follower_run --auto-play"
    echo "  --duration 60 --speed 10 --random-start --random-seed 42"
}

# Parse command line arguments
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --force|-f)
            FORCE_KILL=true
            shift
            ;;
        --no-prompt)
            echo "Error: --no-prompt is only supported by start_av_stack_compare.sh"
            echo "Use: ./start_av_stack_compare.sh --no-prompt --duration 20"
            exit 1
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 0
    else
        return 1
    fi
}

kill_port() {
    local port=$1
    local pid=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$pid" ]; then
        echo "Found existing process on port $port (PID: $pid)"
        kill $pid 2>/dev/null || true
        sleep 1
    fi
}

# Activate virtual environment
if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment not found at $VENV_PATH"
    exit 1
fi
source "$VENV_PATH/bin/activate"

# Check if bridge server is already running
if check_port $BRIDGE_PORT; then
    if [ "$FORCE_KILL" = true ]; then
        echo "Auto-killing existing process (--force flag set)..."
        kill_port $BRIDGE_PORT
    else
        echo "Port $BRIDGE_PORT is already in use."
        echo "Use --force to kill existing process."
        exit 1
    fi
fi

mkdir -p "$SCRIPT_DIR/tmp/logs"
BRIDGE_LOG="$SCRIPT_DIR/tmp/logs/av_bridge.log"

echo "Starting bridge server on port $BRIDGE_PORT..."
cd "$SCRIPT_DIR"
python -m bridge.server > "$BRIDGE_LOG" 2>&1 &
BRIDGE_PID=$!
echo "Bridge server started (PID: $BRIDGE_PID)"

echo "Waiting for bridge server to be ready..."
for i in {1..30}; do
    if curl -s "$BRIDGE_URL/api/health" > /dev/null 2>&1; then
        echo "Bridge server is ready."
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Bridge server failed to start."
        kill $BRIDGE_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

cleanup() {
    echo ""
    echo "Shutting down..."
    curl -s -X POST "$BRIDGE_URL/api/shutdown" > /dev/null 2>&1 || true
    sleep 0.5
    kill $BRIDGE_PID 2>/dev/null || true
    echo "Bridge server stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "Starting ground truth follower..."
python "$SCRIPT_DIR/tools/ground_truth_follower.py" "${PASSTHROUGH_ARGS[@]}"

cleanup
