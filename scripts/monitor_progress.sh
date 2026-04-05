#!/bin/bash
# AURA CLI Progress Monitor Dashboard
# Usage: ./scripts/monitor_progress.sh [interval_seconds]

INTERVAL=${1:-30}

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

clear
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           🎯 AURA CLI PROGRESS MONITOR DASHBOARD              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to print section header
print_section() {
    echo ""
    echo -e "${YELLOW}$1${NC}"
    echo "────────────────────────────────────────────────────────────────"
}

# Function to check if file exists
file_status() {
    if [ -f "$1" ]; then
        size=$(ls -lh "$1" | awk '{ print $5 }')
        modified=$(stat -c %y "$1" 2>/dev/null | cut -d'.' -f1 || stat -f %Sm "$1" 2>/dev/null)
        echo -e "${GREEN}✅ EXISTS${NC} ($size, $modified)"
    else
        echo -e "${YELLOW}⏳ PENDING${NC}"
    fi
}

# Main monitoring loop
while true; do
    # Move cursor to top
    tput cup 4 0
    
    # Print timestamp
    echo "Last Updated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Refresh Interval: ${INTERVAL}s (Press Ctrl+C to exit)"
    
    # Section 1: Innovation Artifacts
    print_section "🔬 INNOVATION ARTIFACTS"
    printf "   %-40s %s\n" "tools/coverage_gap_analyzer.py:" "$(file_status tools/coverage_gap_analyzer.py)"
    printf "   %-40s %s\n" "tools/auto_test_generator.py:" "$(file_status tools/auto_test_generator.py)"
    printf "   %-40s %s\n" "tools/mutation_runner.py:" "$(file_status tools/mutation_runner.py)"
    printf "   %-40s %s\n" "web-ui/src/pages/Coverage.tsx:" "$(file_status web-ui/src/pages/Coverage.tsx)"
    printf "   %-40s %s\n" "n8n-workflows/WF-coverage-auto-remediation.json:" "$(file_status n8n-workflows/WF-coverage-auto-remediation.json)"
    printf "   %-40s %s\n" ".github/workflows/coverage-check.yml:" "$(file_status .github/workflows/coverage-check.yml)"
    
    # Section 2: Test Count
    print_section "🧪 TEST COVERAGE PROGRESS"
    test_count=$(ls tests/test_*.py 2>/dev/null | wc -l)
    echo "   Total Test Files: $test_count (target: 254+)"
    new_tests=$((test_count - 234))
    if [ $new_tests -gt 0 ]; then
        echo -e "   ${GREEN}✅ New tests created: $new_tests${NC}"
    else
        echo -e "   ${YELLOW}⏳ No new tests yet${NC}"
    fi
    
    # Section 3: Active Processes
    print_section "🔄 ACTIVE PROCESSES"
    if pgrep -f "sadd.*batch_unit_tests" > /dev/null; then
        echo -e "   ${GREEN}✅ Batch Unit Tests RUNNING${NC} (PID: $(pgrep -f "sadd.*batch_unit_tests" | head -1))"
    else
        echo -e "   ${YELLOW}⏳ Batch Unit Tests NOT RUNNING${NC}"
    fi
    
    if pgrep -f "sadd.*batch_annotation" > /dev/null; then
        echo -e "   ${GREEN}✅ Batch Annotation Fixes RUNNING${NC} (PID: $(pgrep -f "sadd.*batch_annotation" | head -1))"
    else
        echo -e "   ${YELLOW}⏳ Batch Annotation Fixes NOT RUNNING${NC}"
    fi
    
    if pgrep -f "main.py.*goal run" > /dev/null; then
        echo -e "   ${GREEN}✅ Goal Runner RUNNING${NC} (PID: $(pgrep -f "main.py.*goal run" | head -1))"
    else
        echo -e "   ${YELLOW}⏳ Goal Runner NOT RUNNING${NC}"
    fi
    
    # Section 4: Recent Logs
    print_section "📊 RECENT ACTIVITY"
    if [ -f logs/batch_unit_tests.log ]; then
        echo "   Batch Unit Tests Log:"
        tail -3 logs/batch_unit_tests.log | sed 's/^/     /'
    fi
    if [ -f logs/batch_annotations.log ]; then
        echo "   Annotation Fixes Log:"
        tail -3 logs/batch_annotations.log | sed 's/^/     /'
    fi
    
    # Section 5: Git Status
    print_section "📦 GIT STATUS"
    modified=$(git status --short 2>/dev/null | wc -l)
    echo "   Uncommitted Changes: $modified files"
    
    # Sleep
    sleep $INTERVAL
done
