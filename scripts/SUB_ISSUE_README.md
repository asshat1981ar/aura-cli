# Sub-Issue Creation for Project Chimera Issue #22

This directory contains tools to create sub-issues for breaking down large features into manageable tasks.

## Background

Project Chimera issue #22 ("Automated Analysis & Insight Generation") is a large feature that requires:
1. Querying recent self_opt_metrics
2. Computing trends
3. Identifying anomalies or bottlenecks
4. Summarizing findings for dynamic adjustment

This script helps create well-structured sub-issues for each component.

## Quick Start

### Method 1: Using gh CLI (Recommended for Local Development)

```bash
# Install gh CLI if not already installed
# https://cli.github.com/

# Authenticate
gh auth login

# Preview the sub-issue
python3 scripts/create_sub_issue.py --dry-run

# Create the sub-issue
python3 scripts/create_sub_issue.py --use-gh-cli
```

### Method 2: Using GitHub Personal Access Token

```bash
# Create a PAT with 'repo' scope at https://github.com/settings/tokens
export GITHUB_PAT=ghp_your_token_here

# Preview the sub-issue
python3 scripts/create_sub_issue.py --dry-run

# Create the sub-issue
python3 scripts/create_sub_issue.py
```

## Usage Examples

### Preview Before Creating (Dry Run)

```bash
python3 scripts/create_sub_issue.py --dry-run
```

This shows you exactly what will be created without making any changes.

### Create with Custom Title and Body

```bash
python3 scripts/create_sub_issue.py \
  --title "Implement trend computation for metrics" \
  --body "This sub-issue focuses on computing trends from the metrics data..."
```

### Load Body from File

```bash
# Create a markdown file with your issue body
cat > /tmp/issue-body.md <<EOF
## Description
This is a custom sub-issue body.

## Tasks
- [ ] Task 1
- [ ] Task 2
EOF

python3 scripts/create_sub_issue.py \
  --title "Custom sub-issue" \
  --body-file /tmp/issue-body.md \
  --use-gh-cli
```

### Target a Different Repository

```bash
python3 scripts/create_sub_issue.py \
  --repo "owner/different-repo" \
  --use-gh-cli
```

## Default Sub-Issue Content

By default, the script creates a sub-issue for implementing the metrics query interface, which is the first logical step in the feature. The sub-issue includes:

- **Title**: "Implement metrics query interface for self_opt_metrics"
- **Tasks**: Data schema design, query interface implementation, filtering, testing, documentation
- **Acceptance Criteria**: Performance targets, documentation requirements, test coverage goals
- **Context**: Links back to parent issue #22 and identifies future sub-issues

## Permissions

### For API Method (GITHUB_PAT/GITHUB_TOKEN)
- Token must have `repo` scope
- Token must have write access to `asshat1981ar/project-chimera`

### For gh CLI Method
- User must have write access to `asshat1981ar/project-chimera`
- Authenticated via `gh auth login`

## Troubleshooting

### 403 Forbidden Error
If you get a 403 error:
1. Verify your token has the `repo` scope
2. Verify you have write access to the target repository
3. Try using `gh CLI` method instead: `--use-gh-cli`

### gh CLI Not Found
If you get "gh CLI not found":
1. Install gh CLI from https://cli.github.com/
2. On macOS: `brew install gh`
3. On Linux: Follow instructions at https://github.com/cli/cli/blob/trunk/docs/install_linux.md

### Module Import Error
If you get "Error importing GitHubTools":
1. Ensure you're running from the project root
2. Check that `tools/github_tools.py` exists
3. Install dependencies: `pip install -r requirements.txt`

## Creating Additional Sub-Issues

The parent issue #22 has multiple components. You can create additional sub-issues for:

1. **Trend Computation** (after metrics query)
   ```bash
   python3 scripts/create_sub_issue.py \
     --title "Implement trend computation for self_opt_metrics" \
     --body "Compute trends (fetch errors, chunk-size variance) from queried metrics..."
   ```

2. **Anomaly Detection**
   ```bash
   python3 scripts/create_sub_issue.py \
     --title "Implement anomaly detection for metrics analysis" \
     --body "Identify anomalies and bottlenecks in self_opt_metrics data..."
   ```

3. **Summary Generation**
   ```bash
   python3 scripts/create_sub_issue.py \
     --title "Implement findings summarization for dynamic adjustment" \
     --body "Summarize analysis findings for system adjustment..."
   ```

## Integration with AURA CLI

This script uses the same `GitHubTools` class that AURA CLI uses internally, ensuring consistency with the rest of the codebase. See `tools/github_tools.py` for the full API.

## See Also

- Parent issue: https://github.com/asshat1981ar/project-chimera/issues/22
- GitHubTools documentation: `tools/github_tools.py`
- gh CLI documentation: https://cli.github.com/manual/
