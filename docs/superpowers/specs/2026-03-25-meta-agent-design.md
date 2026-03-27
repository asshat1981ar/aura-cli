# Spec: Agent-Architect Meta-Agent

## 1. Overview
The `agent-architect` is a specialized Gemini-cli agent designed to autonomously scaffold, document, install, and verify other agents. It bridges the gap between raw agent definitions and the AURA Forge engineering standards.

## 2. Goals
- **Autonomous Creation**: Scaffold new agents based on high-level role descriptions.
- **SDLC Compliance**: Ensure every agent is registered in the Forge backlog and follows the 10 SDLC lenses.
- **Dynamic Discovery**: Use `mcp_semantic_discovery` to wire agents to the correct MCP servers (ports 8001-8050).
- **Instant Activation**: Automate the `/agents reload` sequence.

## 3. Architecture
### 3.1 Metadata
- **Name**: `agent-architect`
- **Location**: `~/.gemini/agents/agent-architect.md`
- **Kind**: `local`
- **Model**: `inherit`
- **Tools**: `["read_file", "write_file", "bash", "google_search", "mcp_semantic_discovery"]`

### 3.2 State Machine
1. **INITIALIZING**: Gathers role and tool requirements.
2. **DOCUMENTING**: Writes a Forge story to `.aura_forge/refined/`.
3. **SCAFFOLDING**: Generates the `.md` file with validated YAML frontmatter.
4. **INSTALLING**: Writes the file to the global agents directory.
5. **ACTIVATING**: Executes `/agents reload`.
6. **VERIFYING**: Runs a smoke test against the new agent.
7. **COMPLETING**: Marks the Forge story as done.

## 4. Safety & Standards
- **YAML Validation**: Must use the `|` block scalar for the `description` field.
- **Tool Mapping**: Must verify tools exist via semantic search before adding them to an agent's frontmatter.
- **Isolation**: Agents are written to `~/.gemini/agents/` to ensure global availability while keeping the project repository clean.

## 5. Acceptance Criteria
- [ ] `agent-architect.md` exists and is loaded by Gemini-cli.
- [ ] Agent correctly identifies required tools for a new role.
- [ ] Agent creates a Forge story for the task.
- [ ] Agent successfully reloads the registry after installation.
- [ ] Verification task passes.
