# Comprehensive MCP Ecosystem Specialization Plan (Android / Termux)

## 1. Gemini CLI (`~/.gemini/`)
Gemini CLI serves as a robust, plugin-heavy orchestrator.
**Skills/Agents:**
- **`gemini-kit`**: Comprehensive toolkit with agents for planning, scaffolding, and coding.
- **`pickle-rick`**: Aggressive "God-Mode" refactoring agent.
- **`gemini-beads`**: Git-backed memory and task tracking integration.
- **`listen`**: Audio/webhook inputs for hands-free workflow.
**MCP Servers:**
- **`mcp-toolbox-for-databases`**: Connect to local/cloud databases (MySQL, PostgreSQL, SQLite).
- **`exa-mcp-server`**: For deep contextual web and code search.
- **`open-aware`**: Repository codebase context indexing.
- **`filesystem`**: Read/write access to Android local storage `/data/data/com.termux/files/home`.

## 2. Claude Code (`~/.claude/`)
Claude Code natively supports an interactive terminal workflow.
**Skills/Agents:**
- **`system-agents`**: Personas and agentic behavior switching.
- **`ComputerUse`**: Allows Playwright-based testing inside Termux proot.
**MCP Servers:**
- **`github`**: Core GitHub integration for PR reviews and repository management.
- **`sqlite`**: Fast local database lookup (often used in Termux).
- **`brave-search`**: Search the web directly during coding.
- **`puppeteer`**: Visual end-to-end testing.

## 3. GitHub Copilot CLI (`~/.copilot/`)
Copilot CLI is best tailored for repository health and GitHub ecosystem tie-ins.
**Skills/Agents:**
- **PR Reviewer**: Auto-generates PR reviews based on context.
- **Sequential Thinking**: Complex problem breakdown.
**MCP Servers:**
- **`github-mcp`**: Enhanced interaction with GH APIs.
- **`memory`**: Knowledge-graph based persistent memory server across sessions.
- **`context7`**: Documentation and framework context.

## 4. Codex CLI (`~/.codex/`)
Codex serves as the core parallel multi-agent orchestrator.
**Skills/Agents:**
- **Worker Subagents**: Parallel git worktree execution agents (already built).
- **Critic Agent**: Reviews generated code for merging.
**MCP Servers:**
- **`sentry`**: Reading error logs and finding buggy code points.
- **`neo4j`**: For codebase dependency graphs.
- **`openai-docs`**: API integration reference.