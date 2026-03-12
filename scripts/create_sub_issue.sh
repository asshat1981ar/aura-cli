#!/bin/bash
# Wrapper script to create sub-issues for project-chimera issue #22
# This script provides a simple interface for common use cases

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

usage() {
    cat <<EOF
Usage: $0 [command] [options]

Commands:
    preview         Preview the default sub-issue (dry-run)
    create          Create the default sub-issue
    create-all      Create all planned sub-issues for issue #22
    custom          Create a custom sub-issue
    help            Show this help message

Options:
    --use-api       Use GitHub API instead of gh CLI (requires GITHUB_PAT)
    --repo REPO     Target repository (default: asshat1981ar/project-chimera)

Examples:
    # Preview what will be created
    $0 preview

    # Create using gh CLI (default)
    $0 create

    # Create using API token
    GITHUB_PAT=ghp_xxx $0 create --use-api

    # Create all sub-issues
    $0 create-all

Environment Variables:
    GITHUB_PAT      Personal Access Token (for --use-api)
    GITHUB_TOKEN    Alternative token variable

Notes:
    - gh CLI method requires: gh auth login
    - API method requires GITHUB_PAT with 'repo' scope
    - Token/user must have write access to target repository

EOF
}

preview() {
    echo -e "${YELLOW}Previewing sub-issue...${NC}"
    python3 "$SCRIPT_DIR/create_sub_issue.py" --dry-run "$@"
}

create_single() {
    echo -e "${YELLOW}Creating sub-issue...${NC}"
    if [[ " $* " =~ " --use-api " ]]; then
        python3 "$SCRIPT_DIR/create_sub_issue.py" "$@"
    else
        python3 "$SCRIPT_DIR/create_sub_issue.py" --use-gh-cli "$@"
    fi
}

create_all() {
    echo -e "${YELLOW}Creating all sub-issues for issue #22...${NC}"

    local method_flag=""
    if [[ " $* " =~ " --use-api " ]]; then
        method_flag=""
    else
        method_flag="--use-gh-cli"
    fi

    # Sub-issue 1: Metrics Query Interface
    echo -e "\n${GREEN}[1/4] Creating metrics query interface sub-issue...${NC}"
    python3 "$SCRIPT_DIR/create_sub_issue.py" \
        $method_flag \
        "$@" || echo -e "${RED}Failed to create sub-issue 1${NC}"

    # Sub-issue 2: Trend Computation
    echo -e "\n${GREEN}[2/4] Creating trend computation sub-issue...${NC}"
    python3 "$SCRIPT_DIR/create_sub_issue.py" \
        $method_flag \
        --title "Implement trend computation for self_opt_metrics" \
        --body "**Parent Issue:** #22

## Description
Implement trend analysis to compute meaningful trends from self_opt_metrics data.

## Tasks
- [ ] Design trend computation algorithms
- [ ] Implement fetch error trend tracking
- [ ] Implement chunk-size variance analysis
- [ ] Add statistical aggregation functions
- [ ] Create trend visualization data structures
- [ ] Add tests for trend computation
- [ ] Document trend analysis API

## Acceptance Criteria
- Can compute trends for configurable time windows
- Trend computation is efficient (< 200ms for typical datasets)
- Supports multiple metric types (errors, sizes, timing)
- Unit tests achieve >80% coverage

## Context
This is the second component of issue #22. Depends on the metrics query interface.

/cc @Copilot" \
        "$@" || echo -e "${RED}Failed to create sub-issue 2${NC}"

    # Sub-issue 3: Anomaly Detection
    echo -e "\n${GREEN}[3/4] Creating anomaly detection sub-issue...${NC}"
    python3 "$SCRIPT_DIR/create_sub_issue.py" \
        $method_flag \
        --title "Implement anomaly detection for metrics analysis" \
        --body "**Parent Issue:** #22

## Description
Implement anomaly detection to identify unusual patterns and bottlenecks in metrics data.

## Tasks
- [ ] Design anomaly detection algorithms
- [ ] Implement statistical outlier detection
- [ ] Add bottleneck identification logic
- [ ] Create configurable threshold system
- [ ] Implement alert/notification system
- [ ] Add tests for anomaly detection
- [ ] Document anomaly detection API

## Acceptance Criteria
- Can detect anomalies with configurable sensitivity
- Minimizes false positives (< 5%)
- Identifies bottlenecks accurately
- Unit tests achieve >80% coverage

## Context
This is the third component of issue #22. Depends on metrics query and trend computation.

/cc @Copilot" \
        "$@" || echo -e "${RED}Failed to create sub-issue 3${NC}"

    # Sub-issue 4: Summary Generation
    echo -e "\n${GREEN}[4/4] Creating summary generation sub-issue...${NC}"
    python3 "$SCRIPT_DIR/create_sub_issue.py" \
        $method_flag \
        --title "Implement findings summarization for dynamic adjustment" \
        --body "**Parent Issue:** #22

## Description
Implement summary generation to consolidate analysis findings for dynamic system adjustment.

## Tasks
- [ ] Design summary data structure
- [ ] Implement findings aggregation
- [ ] Create actionable recommendations engine
- [ ] Add summary formatting (text, JSON, etc.)
- [ ] Implement dynamic adjustment interface
- [ ] Add tests for summary generation
- [ ] Document summary API

## Acceptance Criteria
- Generates concise, actionable summaries
- Summaries include specific recommendations
- Supports multiple output formats
- Unit tests achieve >80% coverage

## Context
This is the final component of issue #22. Integrates all previous sub-issues.

/cc @Copilot" \
        "$@" || echo -e "${RED}Failed to create sub-issue 4${NC}"

    echo -e "\n${GREEN}✓ All sub-issues creation attempted${NC}"
    echo -e "Check the output above for any failures"
}

# Main script
case "${1:-help}" in
    preview)
        shift
        preview "$@"
        ;;
    create)
        shift
        create_single "$@"
        ;;
    create-all)
        shift
        create_all "$@"
        ;;
    custom)
        shift
        echo -e "${YELLOW}Creating custom sub-issue...${NC}"
        echo "Pass custom --title and --body or --body-file"
        python3 "$SCRIPT_DIR/create_sub_issue.py" "$@"
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        usage
        exit 1
        ;;
esac
