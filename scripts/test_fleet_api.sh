#!/usr/bin/env bash
# Test script for Fleet Dispatcher API endpoints
# Usage: ./scripts/test_fleet_api.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
AURA_API_URL="${AURA_API_URL:-http://localhost:8001}"
AURA_SKILLS_URL="${AURA_SKILLS_URL:-http://localhost:8002}"
MCP_API_TOKEN="${MCP_API_TOKEN:-}"

echo "=== Fleet Dispatcher API Tests ==="
echo "AURA_API_URL: ${AURA_API_URL}"
echo "AURA_SKILLS_URL: ${AURA_SKILLS_URL}"
echo ""

PASSED=0
FAILED=0

# Test helper function
test_endpoint() {
    local name="$1"
    local method="$2"
    local url="$3"
    local expected_status="${4:-200}"
    local body="${5:-}"
    
    echo -n "Testing ${name}... "
    
    local curl_args=(-s -o /dev/null -w "%{http_code}")
    if [[ -n "${MCP_API_TOKEN}" ]]; then
        curl_args+=(-H "Authorization: Bearer ${MCP_API_TOKEN}")
    fi
    
    if [[ "${method}" == "POST" && -n "${body}" ]]; then
        curl_args+=(-H "Content-Type: application/json" -d "${body}")
    fi
    
    local status
    status=$(curl "${curl_args[@]}" -X "${method}" "${url}")
    
    if [[ "${status}" == "${expected_status}" ]]; then
        echo "PASS (HTTP ${status})"
        ((PASSED++))
    else
        echo "FAIL (expected ${expected_status}, got ${status})"
        ((FAILED++))
    fi
}

# Test 1: Health check
test_endpoint "Health Check" "GET" "${AURA_API_URL}/health"

# Test 2: Metrics endpoint
test_endpoint "Metrics" "GET" "${AURA_API_URL}/metrics"

# Test 3: Discovery endpoint
test_endpoint "Discovery" "GET" "${AURA_API_URL}/discovery"

# Test 4: Skills API - Tools list
test_endpoint "Skills Tools" "GET" "${AURA_SKILLS_URL}/tools"

# Test 5: Webhook goal (POST with sample payload)
TEST_GOAL='{"goal": "Test fleet dispatcher endpoint", "priority": 1, "metadata": {"test": true}}'
test_endpoint "Webhook Goal" "POST" "${AURA_API_URL}/webhook/goal" "202" "${TEST_GOAL}"

# Test 6: Skills API - Call symbol_indexer (if token available)
if [[ -n "${MCP_API_TOKEN}" ]]; then
    echo -n "Testing Skills Call (symbol_indexer)... "
    response=$(curl -s -X POST "${AURA_SKILLS_URL}/call" \
        -H "Authorization: Bearer ${MCP_API_TOKEN}" \
        -H "Content-Type: application/json" \
        -d '{"tool_name": "symbol_indexer", "args": {"project_root": ".", "limit": 10}}')
    
    if [[ "${response}" == *"symbols"* || "${response}" == *"status"* ]]; then
        echo "PASS"
        ((PASSED++))
    else
        echo "FAIL (unexpected response)"
        ((FAILED++))
    fi
else
    echo "SKIP: Skills Call (no MCP_API_TOKEN)"
fi

echo ""
echo "=== Results ==="
echo "PASSED: ${PASSED}"
echo "FAILED: ${FAILED}"

if [[ ${FAILED} -gt 0 ]]; then
    exit 1
fi

echo ""
echo "All tests passed!"
