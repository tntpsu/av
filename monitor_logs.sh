#!/bin/bash

# Monitor Unity and Python server logs together
# Shows both Unity console output and Python server logs in real-time

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNITY_LOG="$SCRIPT_DIR/tmp/logs/unity_launch.log"
BRIDGE_LOG="$SCRIPT_DIR/tmp/logs/av_bridge.log"
AV_STACK_LOG="$SCRIPT_DIR/tmp/logs/av_stack.log"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          AV Stack Log Monitor                           ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Monitoring logs...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

# Function to tail and colorize logs
tail_logs() {
    # Use multitail if available, otherwise use simple tail
    if command -v multitail &> /dev/null; then
        multitail \
            -s 2 \
            -cT ansi -ci green "$BRIDGE_LOG" \
            -cT ansi -ci yellow "$UNITY_LOG" \
            -cT ansi -ci cyan "$AV_STACK_LOG" 2>/dev/null || {
            echo -e "${YELLOW}multitail not available, using simple tail...${NC}"
            tail_logs_simple
        }
    else
        tail_logs_simple
    fi
}

tail_logs_simple() {
    # Simple tail with color coding
    (
        tail -f "$BRIDGE_LOG" 2>/dev/null | while read line; do
            echo -e "${GREEN}[BRIDGE]${NC} $line"
        done &
        PID1=$!
        
        tail -f "$UNITY_LOG" 2>/dev/null | while read line; do
            echo -e "${YELLOW}[UNITY]${NC} $line"
        done &
        PID2=$!
        
        tail -f "$AV_STACK_LOG" 2>/dev/null | while read line; do
            echo -e "${CYAN}[AV_STACK]${NC} $line"
        done &
        PID3=$!
        
        wait $PID1 $PID2 $PID3
    )
}

# Check if log files exist
if [ ! -f "$BRIDGE_LOG" ] && [ ! -f "$UNITY_LOG" ]; then
    echo -e "${YELLOW}⚠ No log files found yet. Waiting for services to start...${NC}"
    echo -e "${BLUE}Log locations:${NC}"
    echo -e "  Bridge: $BRIDGE_LOG"
    echo -e "  Unity: $UNITY_LOG"
    echo -e "  AV Stack: $AV_STACK_LOG"
    echo ""
    
    # Wait for at least one log file
    timeout=30
    elapsed=0
    while [ $elapsed -lt $timeout ]; do
        if [ -f "$BRIDGE_LOG" ] || [ -f "$UNITY_LOG" ]; then
            break
        fi
        sleep 1
        elapsed=$((elapsed + 1))
        echo -n "."
    done
    echo ""
fi

# Start monitoring
tail_logs

