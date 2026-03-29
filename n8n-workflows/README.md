# Fleet Dispatcher n8n Workflows

This directory contains the n8n workflow definitions for the AURA Fleet Prompt Dispatcher - an autonomous GitHub Issue → code → PR → merge pipeline.

## Overview

The Fleet Dispatcher consists of 7 interconnected n8n workflows that form a complete automation pipeline:

| Workflow | File | Purpose |
|----------|------|---------|
| **WF-0** | `WF-0-master-dispatcher.json` | Master dispatcher - receives webhook, classifies issues, routes to handlers |
| **WF-1** | `WF-1-bug-fix-handler.json` | Bug fix handler - runs complexity_scorer, error_pattern_matcher |
| **WF-2** | `WF-2-feature-handler.json` | Feature handler - runs architecture_validator, dependency_analyzer |
| **WF-3** | `WF-3-refactor-handler.json` | Refactor handler - runs code_clone_detector, refactoring_advisor, tech_debt_quantifier |
| **WF-4** | `WF-4-security-handler.json` | Security handler - runs security_scanner, type_checker |
| **WF-5** | `WF-5-docs-handler.json` | Docs handler - runs doc_generator, linter_enforcer |
| **WF-6** | `WF-6-code-gen-pr-push.json` | Code generation, quality gate, PR creation, and auto-merge |

## Architecture

```
GitHub Issue (fleet:trigger label)
    ↓
[GitHub Actions] → POST to n8n webhook
    ↓
WF-0: Master Dispatcher
    ↓ (classifies via LLM)
WF-1/2/3/4/5: Handler Workflows (parallel skills analysis)
    ↓ (calls via ExecuteWorkflow)
WF-6: Code Gen & PR Push
    ↓
GitHub PR → CI Checks → Auto-merge
```

## Prerequisites

### Required Environment Variables

Create a `.env.n8n` file with the following:

```bash
# GitHub Configuration
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
GITHUB_REPO=owner/repo

# AURA API Configuration
AURA_API_URL=http://host.docker.internal:8001
AURA_SKILLS_URL=http://host.docker.internal:8002
AURA_REPO_PATH=/path/to/repo
MCP_API_TOKEN=your-mcp-token

# n8n Configuration
N8N_FLEET_WEBHOOK_URL=https://your-n8n-instance/webhook/fleet
```

### Required GitHub Labels

Run the setup script to create labels:

```bash
./docs/fleet-labels-setup.sh
```

This creates:
- `fleet:trigger` - Triggers the dispatcher
- `fleet:in-progress` - Applied while processing
- `fleet:done` - Applied on successful completion
- `fleet:blocked` - Applied when automation fails

## Import Instructions

### Option 1: n8n UI Import

1. Open n8n at `http://localhost:5678`
2. Go to **Workflows** → **Import from File**
3. Import in order: WF-0, WF-1, WF-2, WF-3, WF-4, WF-5, WF-6
4. Update workflow IDs in ExecuteWorkflow nodes if needed

### Option 2: n8n CLI

```bash
# Requires n8n CLI setup
n8n import:workflow --input=./n8n-workflows/WF-0-master-dispatcher.json
n8n import:workflow --input=./n8n-workflows/WF-1-bug-fix-handler.json
# ... etc
```

## Required Credentials

Configure these credentials in n8n:

| Credential | Type | Used In |
|------------|------|---------|
| GitHub API | OAuth2 or Personal Access Token | WF-0, WF-6 |
| AURA API | Header Auth (Bearer token) | WF-1-6 |
| AURA Skills | Header Auth (Bearer token) | WF-1-5 |
| OpenAI/LLM | API Key | WF-0 (classifier) |

## Workflow Details

### WF-0: Master Dispatcher

**Trigger:** Webhook at `/fleet`

**Nodes:**
1. Webhook - receives GitHub event
2. Filter - skips if issue has `fleet:done` or `fleet:blocked`
3. Set - extracts issue fields (number, title, body, type)
4. LLM Classifier - classifies to bug/feature/refactor/security/docs
5. Switch - routes to appropriate handler workflow
6. Label - applies `fleet:in-progress`

**Output:** `analyst_context` object passed to handler workflows

### WF-1 through WF-5: Handlers

Each handler follows the same pattern:

1. **Input** - receives `analyst_context` from WF-0
2. **Skill Nodes** (parallel) - calls AURA Skills API
   - POST to `{{ $env.AURA_SKILLS_URL }}/call`
   - Bearer token from `{{ $env.MCP_API_TOKEN }}`
3. **Merge** - combines skill outputs
4. **Build Context** - enriches analyst_context with skill results
5. **Execute WF-6** - calls code generation workflow

### WF-6: Code Gen & PR Push

**Nodes:**
1. Create Branch - `fleet/{type}/{issue}` from main
2. Render Goal - builds goal prompt from context
3. Post Goal to AURA - `POST /webhook/goal` with priority:5
4. Poll Goal Status - `GET /webhook/status/{goal_id}` every 30s
5. Git Commit Push - checkout, add, commit, push
6. Extract Applied Files - from cycle_summary
7. Quality Gate - `generation_quality_checker` skill
8. Create PR - `fleet(#{issue}): {title}` format
9. Poll CI Checks - waits for all green
10. Merge PR - squash merge
11. Label Done + Close Issue
12. Escalation - on failure, applies `fleet:blocked`

## Testing

### API Endpoints Test

```bash
export AURA_API_URL=http://localhost:8001
export AURA_SKILLS_URL=http://localhost:8002
export MCP_API_TOKEN=your-token

./scripts/test_fleet_api.sh
```

### Quality Gate Test

```bash
export AURA_SKILLS_URL=http://localhost:8002
export MCP_API_TOKEN=your-token

./scripts/test_fleet_quality_gate.sh [path/to/file.py]
```

## Troubleshooting

### Webhook Not Triggering

1. Verify GitHub Actions workflow is in `.github/workflows/fleet-trigger.yml`
2. Check `N8N_FLEET_WEBHOOK_URL` secret is set in GitHub repo
3. Ensure n8n webhook is publicly accessible (use ngrok for local dev)

### Skills API Errors

1. Verify AURA MCP Skills server is running on port 8002
2. Check `MCP_API_TOKEN` is valid
3. Review skill-specific error logs in n8n execution history

### Quality Gate Failing

- Default threshold is 70
- Adjust in WF-6 "Check Quality Score" node if needed
- Review `generation_quality_checker` output for details

### Workflow Loop Issues

- WF-6 has retry logic (2 retries for quality, 3 for CI)
- Check loop configurations in Wait nodes
- Review escalation path for blocked issues

## Security Considerations

- Store all tokens in n8n Credentials, never in workflow JSON
- Use GitHub fine-grained tokens with minimal permissions
- Enable n8n webhook verification if exposed to internet
- Review and approve WF-6 PRs manually until confident

## References

- [Fleet Dispatcher Spec](../docs/superpowers/specs/sadd-fleet-dispatcher.md)
- [Fleet Dispatcher Design](../docs/superpowers/specs/2026-03-29-fleet-prompt-dispatcher-design.md)
- [n8n Integration Docs](../docs/integrations/n8n.md)
