#!/bin/bash

# AV Stack Stop Script
# Stops all running AV stack services

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Stopping AV Stack services...${NC}"

# Kill bridge server
BRIDGE_PID=$(lsof -ti:8000 2>/dev/null)
if [ ! -z "$BRIDGE_PID" ]; then
    echo -e "Stopping bridge server (PID: $BRIDGE_PID)"
    kill $BRIDGE_PID 2>/dev/null || true
    sleep 1
    if ! kill -0 $BRIDGE_PID 2>/dev/null; then
        echo -e "${GREEN}✓ Bridge server stopped${NC}"
    else
        echo -e "${RED}✗ Force killing bridge server...${NC}"
        kill -9 $BRIDGE_PID 2>/dev/null || true
    fi
else
    echo -e "No bridge server running on port 8000"
fi

# Kill any Python processes related to AV stack
pkill -f "bridge.server" 2>/dev/null || true
pkill -f "av_stack.py" 2>/dev/null || true

echo -e "${GREEN}✓ All services stopped${NC}"




