#!/bin/bash

# AV Stack Startup Script
# Starts the Python bridge server and AV stack for testing
#
# Usage:
#   ./start_av_stack.sh                    # Interactive mode (prompts to kill existing processes)
#   ./start_av_stack.sh --force            # Auto-kill existing processes and start
#   ./start_av_stack.sh --launch-unity      # Launch Unity automatically
#   ./start_av_stack.sh --unity-auto-play   # Launch Unity and auto-enter play mode
#   ./start_av_stack.sh --record            # Start with data recording enabled
#   ./start_av_stack.sh --duration 60       # Run for 60 seconds (similar to ground_truth_follower.py)
#   ./start_av_stack.sh --duration 30 --launch-unity  # Run for 30 seconds and launch Unity

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BRIDGE_URL="http://localhost:8000"
BRIDGE_PORT=8000
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/venv"
FORCE_KILL=false
LAUNCH_UNITY=false
UNITY_AUTO_PLAY=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force|-f)
            FORCE_KILL=true
            shift
            ;;
        --launch-unity|-u)
            LAUNCH_UNITY=true
            shift
            ;;
        --unity-auto-play|-p)
            LAUNCH_UNITY=true
            UNITY_AUTO_PLAY=true
            shift
            ;;
        *)
            # Pass through to av_stack.py
            break
            ;;
    esac
done

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          AV Stack Startup Script                        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${YELLOW}⚠ Virtual environment not found. Creating one...${NC}"
    python3 -m venv "$VENV_PATH"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source "$VENV_PATH/bin/activate"

# Install/upgrade dependencies if needed
if [ ! -f "$VENV_PATH/.deps_installed" ]; then
    echo -e "${YELLOW}Installing dependencies (this may take a minute)...${NC}"
    pip install -q --upgrade pip
    pip install -q -r "$SCRIPT_DIR/requirements.txt"
    touch "$VENV_PATH/.deps_installed"
    echo -e "${GREEN}✓ Dependencies installed${NC}"
fi

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to kill process on port
# CRITICAL: Signal Unity to exit play mode BEFORE killing bridge server
# This prevents Unity from detecting connection loss and exiting play mode unexpectedly
kill_port() {
    local port=$1
    local pid=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$pid" ]; then
        echo -e "${YELLOW}Found existing process on port $port (PID: $pid)${NC}"
        
        # CRITICAL: Signal Unity to exit play mode gracefully BEFORE killing bridge server
        # This prevents Unity from detecting connection loss and exiting play mode unexpectedly
        echo -e "${BLUE}Signaling Unity to exit play mode gracefully (if Unity is running)...${NC}"
        curl -s -X POST "http://localhost:$port/api/shutdown" > /dev/null 2>&1 || true
        sleep 1  # Give Unity time to exit play mode (1 second should be enough)
        echo -e "${GREEN}✓ Unity signaled (if running, it will exit play mode gracefully)${NC}"
        
        echo -e "${YELLOW}Killing existing process on port $port (PID: $pid)...${NC}"
        kill $pid 2>/dev/null || true
        sleep 1  # Give process time to terminate
    fi
}

# Check if bridge server is already running
if check_port $BRIDGE_PORT; then
    echo -e "${YELLOW}⚠ Port $BRIDGE_PORT is already in use${NC}"
    if [ "$FORCE_KILL" = true ]; then
        echo -e "${BLUE}Auto-killing existing process (--force flag set)...${NC}"
        kill_port $BRIDGE_PORT
    else
        read -p "Kill existing process and restart? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            kill_port $BRIDGE_PORT
        else
            echo -e "${RED}Exiting. Please stop the existing service first.${NC}"
            echo -e "${BLUE}Tip: Use --force flag to auto-kill: ./start_av_stack.sh --force${NC}"
            exit 1
        fi
    fi
fi

# Create necessary directories
mkdir -p "$SCRIPT_DIR/data/recordings"
mkdir -p "$SCRIPT_DIR/captures"
mkdir -p "$SCRIPT_DIR/tmp/logs"

# Log file paths
AV_STACK_LOG="$SCRIPT_DIR/tmp/logs/av_stack.log"
AV_BRIDGE_LOG="$SCRIPT_DIR/tmp/logs/av_bridge.log"

# Start bridge server in background
echo -e "${BLUE}Starting bridge server on port $BRIDGE_PORT...${NC}"
cd "$SCRIPT_DIR"
python -m bridge.server > "$AV_BRIDGE_LOG" 2>&1 &
BRIDGE_PID=$!
echo -e "${GREEN}✓ Bridge server started (PID: $BRIDGE_PID)${NC}"
echo -e "  Logs: $AV_BRIDGE_LOG"

# Wait for bridge server to be ready
echo -e "${BLUE}Waiting for bridge server to be ready...${NC}"
for i in {1..30}; do
    if curl -s "$BRIDGE_URL/api/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Bridge server is ready!${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗ Bridge server failed to start${NC}"
        kill $BRIDGE_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# Display bridge server info
echo ""
echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Bridge Server Status:${NC}"
echo -e "  URL: $BRIDGE_URL"
echo -e "  Health: $(curl -s "$BRIDGE_URL/api/health" | python3 -m json.tool 2>/dev/null | grep -o '"status":"[^"]*"' || echo 'checking...')"
echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
echo ""

# Launch Unity if requested
if [ "$LAUNCH_UNITY" = true ]; then
    echo -e "${BLUE}Launching Unity...${NC}"
    UNITY_ARGS=""
    if [ "$UNITY_AUTO_PLAY" = true ]; then
        UNITY_ARGS="--auto-play"
    fi
    "$SCRIPT_DIR/launch_unity.sh" $UNITY_ARGS &
    UNITY_SCRIPT_PID=$!
    echo -e "${GREEN}✓ Unity launched (Script PID: $UNITY_SCRIPT_PID)${NC}"
    echo ""
    
    # Try to find Unity process PID (more reliable for shutdown)
    sleep 2  # Give Unity time to start
    UNITY_PID=$(pgrep -f "Unity.*AVSimulation" | head -1 || echo "")
    if [ ! -z "$UNITY_PID" ]; then
        echo -e "${GREEN}✓ Unity process found (PID: $UNITY_PID)${NC}"
    fi
else
    # Instructions for Unity
    echo -e "${YELLOW}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║              NEXT STEPS - UNITY EDITOR                  ║${NC}"
    echo -e "${YELLOW}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}1. Open Unity Editor${NC}"
    echo -e "   Project: $SCRIPT_DIR/unity/AVSimulation"
    echo ""
    echo -e "${BLUE}2. Open the scene${NC}"
    echo -e "   Assets/Scenes/SampleScene.unity"
    echo ""
    echo -e "${BLUE}3. In the Inspector, select your Car GameObject:${NC}"
    echo -e "   • AV Bridge component → Check 'Enable AV Control'"
    echo -e "   • Make sure Car Controller and Camera Capture are assigned"
    echo ""
    echo -e "${BLUE}4. Press ${GREEN}▶ PLAY${BLUE} in Unity${NC}"
    echo ""
    echo -e "${YELLOW}Or use: ${GREEN}./start_av_stack.sh --launch-unity${YELLOW} to auto-launch Unity${NC}"
    echo ""
    echo -e "${YELLOW}══════════════════════════════════════════════════════════${NC}"
    echo ""
fi

# Start AV stack
echo -e "${BLUE}Starting AV Stack...${NC}"
echo -e "${BLUE}Press Ctrl+C to stop everything${NC}"
echo ""

# Trap to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"
    
    # Signal bridge server that we're shutting down (before killing it)
    echo -e "${BLUE}Signaling bridge server of shutdown...${NC}"
    curl -s -X POST "$BRIDGE_URL/api/shutdown" > /dev/null 2>&1 || true
    sleep 0.5  # Give Unity time to detect shutdown signal
    
    # Stop bridge server
    kill $BRIDGE_PID 2>/dev/null || true
    
    # Handle Unity shutdown based on how it was started
    if [ "$LAUNCH_UNITY" = true ]; then
        # Unity was launched by this script - quit it entirely
        if [ -z "$UNITY_PID" ]; then
            UNITY_PID=$(pgrep -f "Unity.*AVSimulation" | head -1 || echo "")
        fi
        
        if [ ! -z "$UNITY_PID" ]; then
            echo -e "${BLUE}Shutting down Unity (PID: $UNITY_PID)...${NC}"
            echo -e "${BLUE}  (Unity will exit play mode automatically before quitting)${NC}"
            # Send TERM signal - Unity Editor script will exit play mode then quit
            kill -TERM $UNITY_PID 2>/dev/null || true
            sleep 3  # Give Unity time to exit play mode and quit gracefully
            # Force kill if still running
            if kill -0 $UNITY_PID 2>/dev/null; then
                echo -e "${YELLOW}Unity still running, forcing shutdown...${NC}"
                kill -9 $UNITY_PID 2>/dev/null || true
            else
                echo -e "${GREEN}✓ Unity closed gracefully${NC}"
            fi
        fi
        
        # Also kill the launch script if still running
        if [ ! -z "$UNITY_SCRIPT_PID" ]; then
            kill $UNITY_SCRIPT_PID 2>/dev/null || true
        fi
    else
        # Unity was manually started - just signal it to exit play mode (don't quit Unity)
        UNITY_PID=$(pgrep -f "Unity.*AVSimulation" | head -1 || echo "")
        if [ ! -z "$UNITY_PID" ]; then
            echo -e "${BLUE}Unity detected (manually started)${NC}"
            echo -e "${BLUE}  Signaling Unity to exit play mode (Unity will stay open)${NC}"
            echo -e "${YELLOW}  Note: Unity will exit play mode via shutdown endpoint polling${NC}"
            echo -e "${YELLOW}  Unity Editor will remain open - you can close it manually if needed${NC}"
            # Don't send TERM signal - that would quit Unity entirely
            # Unity will detect shutdown via /api/shutdown endpoint polling
        fi
    fi
    
    # Also kill the launch script if still running (only if we launched Unity)
    if [ "$LAUNCH_UNITY" = true ] && [ ! -z "$UNITY_SCRIPT_PID" ]; then
        kill $UNITY_SCRIPT_PID 2>/dev/null || true
    fi
    
    echo -e "${GREEN}✓ Services stopped${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Run AV stack (this will block)
# Enable recording by default for comprehensive logging
# Redirect output to log file for monitoring
# Pass through any additional arguments (e.g., --duration 60, --max_frames 1000)
python "$SCRIPT_DIR/av_stack.py" --bridge_url "$BRIDGE_URL" --record "$@" 2>&1 | tee "$AV_STACK_LOG"
EXIT_CODE=${PIPESTATUS[0]}  # Capture exit code of Python script (not tee)

# CRITICAL: Call cleanup when script ends normally (not just on signals)
# This ensures Unity is signaled and bridge server is stopped even when duration expires
if [ $EXIT_CODE -eq 0 ] || [ $EXIT_CODE -eq 130 ] || [ $EXIT_CODE -eq 143 ]; then
    # Exit code 0 = normal exit (duration/frame limit reached)
    # Exit code 130 = SIGINT (Ctrl+C) - cleanup already handled by trap
    # Exit code 143 = SIGTERM - cleanup already handled by trap
    # Only call cleanup if it wasn't already called by trap
    if [ $EXIT_CODE -eq 0 ]; then
        cleanup
    fi
else
    # Non-zero exit code (error) - still cleanup
    cleanup
fi


