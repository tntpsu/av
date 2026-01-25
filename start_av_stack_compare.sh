#!/bin/bash

# Run back-to-back AV stack tests with CV and segmentation.
#
# Usage:
#   ./start_av_stack_compare.sh --duration 20
#   ./start_av_stack_compare.sh --launch-unity --unity-auto-play --duration 20
#   ./start_av_stack_compare.sh --segmentation-checkpoint data/segmentation_dataset/checkpoints/segnet_best.pt
#
# Notes:
# - Unity may exit Play mode between runs (AV stack sends shutdown signal).
#   You'll be prompted to re-enter Play before the second run.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DURATION=20
SEG_CHECKPOINT="$SCRIPT_DIR/data/segmentation_dataset/checkpoints/segnet_best.pt"
LAUNCH_UNITY=false
UNITY_AUTO_PLAY=false
KEEP_UNITY_OPEN=true
NO_PROMPT=false
PASSTHROUGH_ARGS=()

usage() {
    echo "Usage: ./start_av_stack_compare.sh [options] [av_stack.py args]"
    echo ""
    echo "Options:"
    echo "  --duration <sec>               Duration for each run (default: 20)"
    echo "  --launch-unity                 Launch Unity automatically"
    echo "  --unity-auto-play              Launch Unity and auto-enter Play mode"
    echo "  --segmentation-checkpoint <p>  Segmentation checkpoint path"
    echo "  --no-prompt                    Auto-relaunch Unity between runs (no prompt)"
    echo "  --help, -h                     Show this help"
    echo ""
    echo "Example:"
    echo "  ./start_av_stack_compare.sh --duration 20 --launch-unity --unity-auto-play"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            usage
            exit 0
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --launch-unity)
            LAUNCH_UNITY=true
            shift
            ;;
        --unity-auto-play)
            LAUNCH_UNITY=true
            UNITY_AUTO_PLAY=true
            shift
            ;;
        --segmentation-checkpoint)
            SEG_CHECKPOINT="$2"
            shift 2
            ;;
        --no-prompt)
            NO_PROMPT=true
            shift
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ "$NO_PROMPT" = true ]; then
    LAUNCH_UNITY=true
    UNITY_AUTO_PLAY=true
    KEEP_UNITY_OPEN=false
fi

COMMON_ARGS=("--duration" "$DURATION")
if [ "$KEEP_UNITY_OPEN" = true ]; then
    COMMON_ARGS+=("--keep-unity-open")
fi
if [ "$LAUNCH_UNITY" = true ]; then
    if [ "$UNITY_AUTO_PLAY" = true ]; then
        COMMON_ARGS+=("--unity-auto-play")
    else
        COMMON_ARGS+=("--launch-unity")
    fi
fi

echo ""
echo "=== Run 1/2: CV/ML perception ==="
"$SCRIPT_DIR/start_av_stack.sh" "${COMMON_ARGS[@]}" "${PASSTHROUGH_ARGS[@]}"

echo ""
echo "=== Run 2/2: Segmentation perception ==="
if [ "$NO_PROMPT" = true ]; then
    echo "Auto mode: relaunching Unity with auto-play..."
    sleep 3
else
    echo "Unity likely exited Play mode after run 1."
    read -p "Re-enter Play mode in Unity, then press Enter to continue... " -r
fi

"$SCRIPT_DIR/start_av_stack.sh" "${COMMON_ARGS[@]}" \
    --use-segmentation \
    --segmentation-checkpoint "$SEG_CHECKPOINT" \
    "${PASSTHROUGH_ARGS[@]}"
