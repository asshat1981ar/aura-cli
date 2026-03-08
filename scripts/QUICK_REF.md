# Quick Reference: Sub-Issue Creation

## TL;DR

```bash
# Preview
./scripts/create_sub_issue.sh preview

# Create one sub-issue
./scripts/create_sub_issue.sh create

# Create all sub-issues
./scripts/create_sub_issue.sh create-all
```

## One-Liners

### Using gh CLI (Recommended)
```bash
# Must be authenticated: gh auth login
./scripts/create_sub_issue.sh create
```

### Using GitHub PAT
```bash
GITHUB_PAT=ghp_xxx ./scripts/create_sub_issue.sh create --use-api
```

## What Gets Created

By default, creates a sub-issue for "Implement metrics query interface" which is the first component needed for issue #22.

To create all 4 sub-issues at once:
```bash
./scripts/create_sub_issue.sh create-all
```

This creates:
1. Metrics Query Interface
2. Trend Computation
3. Anomaly Detection
4. Summary Generation

## Customization

### Custom Title & Body
```bash
python3 scripts/create_sub_issue.py \
  --title "My custom sub-issue" \
  --body "My custom body text"
```

### Load Body from File
```bash
python3 scripts/create_sub_issue.py \
  --title "From file" \
  --body-file /path/to/body.md
```

### Different Repository
```bash
./scripts/create_sub_issue.sh create --repo "owner/repo"
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| 403 Forbidden | Token lacks 'repo' scope or write access |
| gh not found | Install from https://cli.github.com/ |
| Import error | Run from project root, check requirements.txt |

## Authentication Setup

### gh CLI
```bash
gh auth login
# Follow prompts
```

### GitHub PAT
1. Go to https://github.com/settings/tokens
2. Generate new token (classic)
3. Select `repo` scope
4. Export: `export GITHUB_PAT=ghp_xxx`

## Files

- `scripts/create_sub_issue.py` - Main Python script
- `scripts/create_sub_issue.sh` - Bash wrapper
- `scripts/SUB_ISSUE_README.md` - Full documentation

## Help

```bash
./scripts/create_sub_issue.sh help
python3 scripts/create_sub_issue.py --help
```
