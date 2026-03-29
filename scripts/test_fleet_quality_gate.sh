#!/usr/bin/env bash
# Test script for Fleet Dispatcher Quality Gate
# Usage: ./scripts/test_fleet_quality_gate.sh [file_path]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
AURA_SKILLS_URL="${AURA_SKILLS_URL:-http://localhost:8002}"
MCP_API_TOKEN="${MCP_API_TOKEN:-}"
TEST_FILE="${1:-${ROOT_DIR}/aura_cli/cli_main.py}"

echo "=== Fleet Dispatcher Quality Gate Test ==="
echo "AURA_SKILLS_URL: ${AURA_SKILLS_URL}"
echo "Test file: ${TEST_FILE}"
echo ""

if [[ ! -f "${TEST_FILE}" ]]; then
    echo "ERROR: Test file not found: ${TEST_FILE}"
    exit 1
fi

if [[ -z "${MCP_API_TOKEN}" ]]; then
    echo "WARNING: MCP_API_TOKEN not set. Some tests may fail."
fi

echo -n "Testing generation_quality_checker... "

# Call the generation_quality_checker skill
response=$(curl -s -X POST "${AURA_SKILLS_URL}/call" \
    -H "Authorization: Bearer ${MCP_API_TOKEN}" \
    -H "Content-Type: application/json" \
    -d "{
        \"tool_name\": \"generation_quality_checker\",
        \"args\": {
            \"file_path\": \"${TEST_FILE}\",
            \"goal_context\": \"Test quality gate for fleet dispatcher\"
        }
    }" 2>/dev/null || echo '{"error": "connection_failed"}')

# Validate response contains score
if [[ "${response}" == *"score"* ]]; then
    # Extract score using basic parsing
    score=$(echo "${response}" | grep -o '"score":[0-9.]*' | cut -d: -f2)
    
    if [[ -n "${score}" ]]; then
        # Validate score is numeric and in range 0-100
        if awk "BEGIN {exit !(${score} >= 0 && ${score} <= 100)}" 2>/dev/null; then
            echo "PASS"
            echo "  Score: ${score}"
            
            # Check against threshold
            threshold=70
            if awk "BEGIN {exit !(${score} >= ${threshold})}"; then
                echo "  Status: PASSED threshold (${threshold})"
            else
                echo "  Status: BELOW threshold (${threshold})"
            fi
            
            # Show additional metrics if available
            if [[ "${response}" == *"coverage"* ]]; then
                coverage=$(echo "${response}" | grep -o '"coverage":[0-9.]*' | cut -d: -f2)
                echo "  Coverage: ${coverage}%"
            fi
            
            if [[ "${response}" == *"complexity"* ]]; then
                complexity=$(echo "${response}" | grep -o '"complexity":[0-9.]*' | cut -d: -f2)
                echo "  Complexity: ${complexity}"
            fi
            
            echo ""
            echo "Quality gate test PASSED"
            exit 0
        else
            echo "FAIL"
            echo "  Score out of range: ${score}"
            echo "  Expected: 0-100"
            exit 1
        fi
    else
        echo "FAIL"
        echo "  Could not extract score from response"
        echo "  Response: ${response}"
        exit 1
    fi
else
    echo "FAIL"
    echo "  Response does not contain score field"
    echo "  Response: ${response}"
    exit 1
fi
