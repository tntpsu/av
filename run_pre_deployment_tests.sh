#!/bin/bash
# Pre-deployment test suite
# Runs system-level tests before deployment to ensure system is stable

set -e  # Exit on error

echo "🚀 Pre-Deployment Test Suite"
echo "================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
PASSED=0
FAILED=0
SKIPPED=0

# Function to run test and capture result
run_test() {
    local test_name=$1
    local test_command=$2
    
    echo -n "Running $test_name... "
    
    if eval "$test_command" > /tmp/test_output.log 2>&1; then
        echo -e "${GREEN}PASSED${NC}"
        ((PASSED++))
        return 0
    else
        local exit_code=$?
        if grep -q "SKIPPED" /tmp/test_output.log; then
            echo -e "${YELLOW}SKIPPED${NC}"
            ((SKIPPED++))
            return 0
        else
            echo -e "${RED}FAILED${NC}"
            cat /tmp/test_output.log | tail -10
            ((FAILED++))
            return 1
        fi
    fi
}

echo "1. Running unit tests..."
run_test "Unit Tests" "python3 -m pytest tests/test_control.py tests/test_coordinate_system.py -v --tb=no -q"

echo ""
echo "2. Running heading effect tests..."
run_test "Heading Effect Tests" "python3 -m pytest tests/test_heading_effect.py -v --tb=no -q"

echo ""
echo "3. Running system stability tests..."
run_test "System Stability (60s)" "python3 -m pytest tests/test_system_stability.py::TestSystemStabilityExtendedPeriod::test_system_stability_60_seconds -v --tb=no -q"

echo ""
echo "4. Running error correction effectiveness test..."
run_test "Error Correction" "python3 -m pytest tests/test_system_stability.py::TestSystemStabilityExtendedPeriod::test_error_correction_effectiveness -v --tb=no -q"

echo ""
echo "5. Running pre-deployment stability test..."
run_test "Pre-Deployment (30s)" "python3 -m pytest tests/test_system_stability.py::TestPreDeploymentStability::test_pre_deployment_30_seconds -v --tb=no -q"

echo ""
echo "================================"
echo "Test Results:"
echo "  ${GREEN}Passed: $PASSED${NC}"
echo "  ${YELLOW}Skipped: $SKIPPED${NC}"
echo "  ${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ PRE-DEPLOYMENT PASS: All tests passed!${NC}"
    echo "System is ready for deployment."
    exit 0
else
    echo -e "${RED}❌ PRE-DEPLOYMENT FAIL: $FAILED test(s) failed!${NC}"
    echo "System is NOT ready for deployment."
    echo ""
    echo "Please fix the failing tests before deploying."
    exit 1
fi

