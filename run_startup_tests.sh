#!/bin/bash

# Quick startup tests to catch common initialization issues
# Run this before starting the AV stack to catch problems early

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          AV Stack Startup Tests                         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "av_stack.py" ]; then
    echo -e "${RED}✗ Error: Must run from project root directory${NC}"
    exit 1
fi

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo -e "${BLUE}Running startup tests...${NC}"
echo ""

# Run the startup tests (including new coverage tests)
if python3 -m pytest tests/test_av_stack_startup.py \
                      tests/test_config_integration.py \
                      tests/test_trajectory_state_reuse.py \
                      tests/test_smoothing_reset_on_failures.py \
                      tests/test_trajectory_edge_cases.py \
                      -v --tb=short; then
    echo ""
    echo -e "${GREEN}✅ All startup tests passed!${NC}"
    echo -e "${GREEN}   AV Stack should initialize correctly${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}✗ Startup tests failed!${NC}"
    echo -e "${YELLOW}   Please fix the issues before running the AV stack${NC}"
    exit 1
fi

