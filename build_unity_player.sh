#!/bin/bash

# Build Unity player from CLI (macOS)
# Uses BuildPlayerCLI.BuildMacPlayer in the Unity project.

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNITY_PROJECT_PATH="$SCRIPT_DIR/unity/AVSimulation"

UNITY_PATH="${UNITY_PATH:-}"
BUILD_OUTPUT="$UNITY_PROJECT_PATH/mybuild.app"

SKIP_IF_CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --unity-path)
            UNITY_PATH="$2"
            shift 2
            ;;
        --build-path)
            BUILD_OUTPUT="$2"
            shift 2
            ;;
        --skip-if-clean)
            SKIP_IF_CLEAN=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [--unity-path /path/to/Unity] [--build-path /path/to/output.app] [--skip-if-clean]"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         Unity Player Build (macOS)                      ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

if [ ! -d "$UNITY_PROJECT_PATH" ]; then
    echo -e "${RED}✗ Unity project not found at: $UNITY_PROJECT_PATH${NC}"
    exit 1
fi

if [ -z "$UNITY_PATH" ]; then
    if [ "$(uname)" == "Darwin" ]; then
        if [ -d "/Applications/Unity Hub.app" ]; then
            for version_dir in /Applications/Unity/Hub/Editor/*/; do
                if [ -d "$version_dir" ] && [ -f "${version_dir}Unity.app/Contents/MacOS/Unity" ]; then
                    UNITY_PATH="${version_dir}Unity.app/Contents/MacOS/Unity"
                    echo -e "${GREEN}✓ Found Unity at: $UNITY_PATH${NC}"
                    break
                fi
            done
        fi
        if [ -z "$UNITY_PATH" ] && [ -f "/Applications/Unity/Unity.app/Contents/MacOS/Unity" ]; then
            UNITY_PATH="/Applications/Unity/Unity.app/Contents/MacOS/Unity"
            echo -e "${GREEN}✓ Found Unity at: $UNITY_PATH${NC}"
        fi
    fi
fi

if [ -z "$UNITY_PATH" ]; then
    echo -e "${RED}✗ Unity installation not found!${NC}"
    echo -e "${YELLOW}Set UNITY_PATH or pass --unity-path to this script.${NC}"
    exit 1
fi

if pgrep -f "Unity.*AVSimulation" >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠ Unity Editor appears to be running for AVSimulation.${NC}"
    echo -e "${YELLOW}  Batch builds can fail if the editor is compiling; close it if needed.${NC}"
    echo ""
fi

if [ "$SKIP_IF_CLEAN" = true ]; then
    if command -v git >/dev/null 2>&1 && git -C "$SCRIPT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        if git -C "$SCRIPT_DIR" status --porcelain --untracked-files=no -- \
            unity/AVSimulation/Assets unity/AVSimulation/Packages unity/AVSimulation/ProjectSettings | \
            grep -q .; then
            echo -e "${YELLOW}⚠ Unity project has changes; build will run.${NC}"
        else
            if [ -d "$BUILD_OUTPUT" ]; then
                latest_source=$(find "$UNITY_PROJECT_PATH/Assets" \
                    "$UNITY_PROJECT_PATH/Packages" "$UNITY_PROJECT_PATH/ProjectSettings" \
                    -type f -print0 | xargs -0 stat -f "%m" | sort -n | tail -1)
                build_time=$(stat -f "%m" "$BUILD_OUTPUT")
                if [ "$build_time" -ge "$latest_source" ]; then
                    echo -e "${GREEN}✓ Unity player is up to date. Skipping build (--skip-if-clean).${NC}"
                    exit 0
                fi
            fi
        fi
    else
        echo -e "${YELLOW}⚠ Git not available; falling back to timestamp check only.${NC}"
        if [ -d "$BUILD_OUTPUT" ]; then
            latest_source=$(find "$UNITY_PROJECT_PATH/Assets" \
                "$UNITY_PROJECT_PATH/Packages" "$UNITY_PROJECT_PATH/ProjectSettings" \
                -type f -print0 | xargs -0 stat -f "%m" | sort -n | tail -1)
            build_time=$(stat -f "%m" "$BUILD_OUTPUT")
            if [ "$build_time" -ge "$latest_source" ]; then
                echo -e "${GREEN}✓ Unity player is up to date. Skipping build (--skip-if-clean).${NC}"
                exit 0
            fi
        fi
    fi
fi

mkdir -p "$SCRIPT_DIR/tmp/logs"
LOG_FILE="$SCRIPT_DIR/tmp/logs/unity_build.log"

UNITY_CMD=(
    "$UNITY_PATH"
    -batchmode
    -nographics
    -quit
    -projectPath "$UNITY_PROJECT_PATH"
    -executeMethod BuildPlayerCLI.BuildMacPlayer
    -buildOutput "$BUILD_OUTPUT"
    -logFile "$LOG_FILE"
)

echo -e "${BLUE}Build output:${NC} $BUILD_OUTPUT"
echo -e "${BLUE}Log file:${NC} $LOG_FILE"
echo ""
echo -e "${BLUE}Command:${NC}"
printf '%q ' "${UNITY_CMD[@]}"
echo ""
echo ""

"${UNITY_CMD[@]}"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Unity player build succeeded${NC}"
else
    echo -e "${RED}✗ Unity build failed (exit $EXIT_CODE)${NC}"
    echo -e "${YELLOW}Check log: $LOG_FILE${NC}"
    exit $EXIT_CODE
fi
