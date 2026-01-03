#!/bin/bash

# Launch Unity and run the AV simulation scene
# This script finds Unity installation and launches it with the project

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNITY_PROJECT_PATH="$SCRIPT_DIR/unity/AVSimulation"
SCENE_PATH="Assets/Scenes/SampleScene.unity"

# Parse arguments
BATCH_MODE=false
AUTO_PLAY=false
QUIT_AFTER=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --batch|-b)
            BATCH_MODE=true
            QUIT_AFTER=true
            shift
            ;;
        --auto-play|-p)
            AUTO_PLAY=true
            shift
            ;;
        --quit|-q)
            QUIT_AFTER=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [--batch] [--auto-play] [--quit]"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          Unity Launcher for AV Simulation               ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if Unity project exists
if [ ! -d "$UNITY_PROJECT_PATH" ]; then
    echo -e "${RED}✗ Unity project not found at: $UNITY_PROJECT_PATH${NC}"
    exit 1
fi

echo -e "${BLUE}Project path: $UNITY_PROJECT_PATH${NC}"

# Find Unity installation
UNITY_PATH=""

# Try common Unity Hub locations on macOS
if [ "$(uname)" == "Darwin" ]; then
    # Check Unity Hub for installed editors
    if [ -d "/Applications/Unity Hub.app" ]; then
        # Try to find latest Unity version
        for version_dir in /Applications/Unity/Hub/Editor/*/; do
            if [ -d "$version_dir" ] && [ -f "${version_dir}Unity.app/Contents/MacOS/Unity" ]; then
                UNITY_PATH="${version_dir}Unity.app/Contents/MacOS/Unity"
                echo -e "${GREEN}✓ Found Unity at: $UNITY_PATH${NC}"
                break
            fi
        done
    fi
    
    # Fallback: try direct Unity installation
    if [ -z "$UNITY_PATH" ] && [ -f "/Applications/Unity/Unity.app/Contents/MacOS/Unity" ]; then
        UNITY_PATH="/Applications/Unity/Unity.app/Contents/MacOS/Unity"
        echo -e "${GREEN}✓ Found Unity at: $UNITY_PATH${NC}"
    fi
fi

# Try Linux Unity installation
if [ "$(uname)" == "Linux" ] && [ -z "$UNITY_PATH" ]; then
    if [ -f "/opt/unity/Editor/Unity" ]; then
        UNITY_PATH="/opt/unity/Editor/Unity"
        echo -e "${GREEN}✓ Found Unity at: $UNITY_PATH${NC}"
    fi
fi

# Try Windows Unity installation (if running in WSL or similar)
if [ -z "$UNITY_PATH" ] && command -v powershell.exe &> /dev/null; then
    # Windows paths would go here
    echo -e "${YELLOW}⚠ Windows Unity detection not implemented${NC}"
fi

if [ -z "$UNITY_PATH" ]; then
    echo -e "${RED}✗ Unity installation not found!${NC}"
    echo ""
    echo -e "${YELLOW}Please install Unity Hub and at least one Unity Editor version.${NC}"
    echo -e "${YELLOW}Or specify Unity path manually by setting UNITY_PATH environment variable.${NC}"
    echo ""
    echo "Example:"
    echo "  export UNITY_PATH=\"/Applications/Unity/Hub/Editor/2022.3.0f1/Unity.app/Contents/MacOS/Unity\""
    echo "  $0"
    exit 1
fi

# Build Unity command
UNITY_CMD="$UNITY_PATH -projectPath \"$UNITY_PROJECT_PATH\""

if [ "$BATCH_MODE" = true ]; then
    UNITY_CMD="$UNITY_CMD -batchmode -nographics"
    echo -e "${BLUE}Mode: Batch (headless)${NC}"
else
    echo -e "${BLUE}Mode: GUI${NC}"
fi

if [ "$AUTO_PLAY" = true ]; then
    # Create flag file for Unity Editor script to detect (more reliable than env vars)
    FLAG_FILE="$UNITY_PROJECT_PATH/.unity_autoplay"
    touch "$FLAG_FILE"
    echo -e "${BLUE}Auto-play: Enabled (via flag file: $FLAG_FILE)${NC}"
    echo -e "${YELLOW}Note: Unity Editor script will auto-enter play mode when project loads${NC}"
fi

if [ "$QUIT_AFTER" = true ]; then
    UNITY_CMD="$UNITY_CMD -quit"
    echo -e "${BLUE}Quit after execution: Enabled${NC}"
fi

# Add logging
mkdir -p "$SCRIPT_DIR/tmp/logs"
LOG_FILE="$SCRIPT_DIR/tmp/logs/unity_launch.log"
UNITY_CMD="$UNITY_CMD -logFile \"$LOG_FILE\""

echo ""
echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Launching Unity...${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Command:${NC}"
echo "$UNITY_CMD"
echo ""
echo -e "${BLUE}Log file: $LOG_FILE${NC}"
echo ""

# Launch Unity (flag file already created if auto-play enabled)
eval $UNITY_CMD

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Unity launched successfully${NC}"
else
    echo -e "${RED}✗ Unity exited with code: $EXIT_CODE${NC}"
    echo -e "${YELLOW}Check log file: $LOG_FILE${NC}"
    exit $EXIT_CODE
fi

