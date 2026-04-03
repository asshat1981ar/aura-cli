# AURA Glossary

## Core Concepts

| Term | Meaning | Context |
|------|---------|---------|
| **SADD** | Sub-Agent Driven Development | Parallel workstream execution with multiple agents |
| **MCP** | Model Context Protocol | External tool integration protocol (Smithery) |
| **n8n** | Workflow automation platform | 7 fleet dispatcher workflows (WF-0 to WF-6) |
| **Swarm** | Multi-agent coordination | AURA_ENABLE_SWARM=1 feature flag |
| **Fleet** | Group of parallel workstreams | Wave-based execution |

## Acronyms

| Acronym | Full Name | Usage |
|---------|-----------|-------|
| **WF-X** | Workflow Number | n8n workflow identifiers (WF-0 through WF-6) |
| **LoC** | Lines of Code | Codebase metrics (76K+ LoC) |
| **E2E** | End-to-End | Testing terminology |
| **CI/CD** | Continuous Integration/Deployment | Pipeline automation |
| **SSE** | Server-Sent Events | MCP transport type |
| **RE** | Reverse Engineering | Skill domain |
| **PRD** | Product Requirements Document | Planning documents |
| **ADK** | Agent Development Kit | Botpress framework |

## AURA-Specific Terms

| Term | Definition |
|------|------------|
| **Goal Queue** | SQLite-backed task queue (206+ goals) |
| **Workstream** | Parallel execution unit in SADD |
| **Wave** | Dependency-ordered execution phase |
| **Adaptive Pipeline** | 10-phase loop (ingest → reflect) |
| **Agent Registry** | Auto-registration of 20+ agent types |
| **Orchestrator** | Central coordination component |
| **Sandbox** | Isolated execution environment |
| **Skill** | Modular capability extension |
| **Session** | Unique SADD execution instance |
| **CoT** | Chain-of-Thought | Step-by-step reasoning in LLM prompts |
| **Structured Output** | Pydantic-enforced LLM responses | Type-safe agent outputs |

## Agent Schemas (New)

Structured output schemas in `agents/schemas.py`:

| Schema | Agent | Output |
|--------|-------|--------|
| `PlannerOutput` | PlannerAgent | Plan with CoT reasoning, confidence, complexity |
| `CriticOutput` | CriticAgent | Critique with severity levels, recommendations |
| `CoderOutput` | CoderAgent | Code with approach explanation, edge cases |
| `MutationValidationOutput` | CriticAgent | Mutation approval with impact analysis |
| `InnovationOutput` | InnovationSwarm | Ideas with novelty, feasibility, diversity scores |
| `InnovationSessionState` | MetaConductor | Full 5-phase session tracking |

**CoT Sections:**
- Planner: Analysis → Gap Assessment → Approach → Risk Assessment
- Critic: Initial Assessment → Completeness → Feasibility → Risks
- Coder: Problem Analysis → Approach Selection → Design → Testing

## Prompt Manager (New)

Role-based system prompts with caching in `agents/prompt_manager.py`:

| Role | Persona | Expertise |
|------|---------|-----------|
| `planner` | Senior Software Architect | System design, risk assessment, task decomposition |
| `critic` | Principal Engineer | Code review, security analysis, quality standards |
| `coder` | Expert Python Developer | Clean code, TDD, edge case handling |
| `synthesizer` | Technical Integration Specialist | Merge perspectives, reconcile conflicts |
| `reflector` | Systems Analyst | Learning extraction, improvement identification |

**Features:**
- **Prompt Caching**: LRU cache with TTL (default 1 hour, max 100 entries)
- **Cache Stats**: Track hits, misses, hit rate
- **Token Efficiency**: Optimized templates reduce prompt size by ~30%
- **Backward Compatible**: Legacy templates still available

## Innovation Catalyst (New)

Multi-agent brainstorming system in `agents/innovation_swarm.py` and `agents/meta_conductor.py`:

### 5-Phase Innovation Process

| Phase | Description | Agent |
|-------|-------------|-------|
| **Immersion** | Deep understanding of problem | MetaConductor |
| **Divergence** | Generate many ideas via 8 techniques | InnovationSwarm |
| **Convergence** | Evaluate and select best ideas | InnovationSwarm |
| **Incubation** | Let ideas develop | MetaConductor |
| **Transformation** | Prepare actionable tasks | MetaConductor |

### 8 Brainstorming Techniques

| Technique | Bot | Description |
|-----------|-----|-------------|
| **SCAMPER** | `SCAMPERBot` | Substitute, Combine, Adapt, Modify, Put, Eliminate, Reverse |
| **Six Thinking Hats** | `SixThinkingHatsBot` | White, Red, Black, Yellow, Green, Blue perspectives |
| **Mind Mapping** | `MindMappingBot` | Visual branching from central concepts |
| **Reverse Brainstorming** | `ReverseBrainstormingBot` | Invert problem to find solutions |
| **Worst Idea** | `WorstIdeaBot` | Transform bad ideas into good ones |
| **Lotus Blossom** | `LotusBlossomBot` | Systematic expansion (center → 8 petals → sub-petals) |
| **Star Brainstorming** | `StarBrainstormingBot` | Radiating ideas from central star |
| **Bisociative Association** | `BIABot` | Connect disparate domains (biology, music, architecture...) |

### Usage

```python
from agents.meta_conductor import MetaConductor

conductor = MetaConductor()
session = conductor.start_session(
    problem_statement="How to improve code review?",
    techniques=['scamper', 'six_hats', 'mind_map']
)

# Execute phases
for phase in [InnovationPhase.DIVERGENCE, InnovationPhase.CONVERGENCE]:
    result = conductor.execute_phase(session.session_id, phase)
```

### Innovation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **Diversity Score** | >0.7 | Variety across techniques (0-1) |
| **Novelty Score** | >0.7 | Average novelty of selected ideas (0-1) |
| **Feasibility Score** | >0.6 | Average feasibility of selected ideas (0-1) |
| **Convergence Rate** | 10-20% | Percentage of ideas selected |

## File Patterns

| Pattern | Meaning |
|---------|---------|
| `*.sadd.md` | SADD specification documents |
| `*workflow*.json` | n8n workflow definitions |
| `SKILL.md` | Skill documentation files |
| `CLAUDE.md` | Hot cache memory file |
| `.local.md` | Project-specific settings |

## External Services

| Service | Purpose | Status |
|---------|---------|--------|
| **OpenRouter** | LLM model routing | Active |
| **Smithery** | MCP server discovery | 4 servers pending auth |
| **GitHub** | Code hosting | Connected |
| **Slack** | Notifications | Pending auth |
| **Sentry** | Error tracking | Pending auth |
| **Supabase** | Database | Pending auth |

## Metrics & Targets

| Metric | Current | Target |
|--------|---------|--------|
| Test Coverage | 40% | 70% |
| Bare Exceptions | 364+ | <20 |
| Code Duplication | 100+ lines | 0 |
| Memory Usage | Variable | <100MB |
| Active Agents | 20+ | 5+ concurrent |
