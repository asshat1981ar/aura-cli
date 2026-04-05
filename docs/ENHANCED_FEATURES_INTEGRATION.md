# Enhanced Features Integration Guide

## Overview

This document describes the integration of four advanced features into AURA's LoopOrchestrator:

1. **Simulation Engine** - Run "what-if" scenarios before committing changes
2. **Knowledge Base** - Share insights and lessons learned across sessions  
3. **LLM Voting** - Multi-model consensus for critical decisions
4. **Adversarial Agent** - Red-team critique and stress testing

## Quick Start

### Option 1: Attach to Existing Orchestrator (Recommended)

```python
from core.orchestrator import LoopOrchestrator
from core.enhanced_orchestrator import attach_enhanced_features_to_orchestrator

# Create your orchestrator as usual
orchestrator = LoopOrchestrator(
    agents=default_agents(brain, model),
    memory_store=MemoryStore()
)

# Attach enhanced features
orchestrator = attach_enhanced_features_to_orchestrator(
    orchestrator,
    enable_simulation=True,
    enable_knowledge=True,
    enable_voting=True,
    enable_adversarial=True
)

# Use normally - enhanced features integrate automatically
result = orchestrator.run_cycle("Implement feature X")
```

### Option 2: Use EnhancedOrchestrator Wrapper

```python
from core.enhanced_orchestrator import EnhancedOrchestrator

# Create enhanced orchestrator directly
orchestrator = EnhancedOrchestrator(
    base_orchestrator=existing_orchestrator,
    enable_simulation=True,
    enable_knowledge=True,
    enable_voting=True,
    enable_adversarial=True
)

# Process with enhancements
result = await orchestrator.process_with_enhancements(
    goal="Implement feature X",
    use_simulation=True,
    use_knowledge=True,
    use_adversarial=True
)
```

## Feature Details

### 1. Simulation Engine

**Purpose:** Test different approaches before committing to actual changes.

**When Activated:**
- Goal type is "refactoring", "feature", or "optimization"
- Simulation engine is attached and enabled

**What It Does:**
- Runs parallel simulations with different approaches (conservative, balanced, aggressive)
- Analyzes outcomes and recommends best approach
- Provides insights into trade-offs

**Integration Point:** Before planning phase

```python
# Context injected into planning
context["enhanced_analysis"]["recommended_approach"] = "best_scenario_name"
context["enhanced_analysis"]["simulation_insights"] = [...]
```

### 2. Knowledge Base

**Purpose:** Retrieve relevant past lessons and store new learnings.

**When Activated:**
- Always active when knowledge_base is attached

**What It Does:**
- **Before cycle:** Queries knowledge base for relevant past lessons based on goal
- **After cycle:** Stores successful outcomes as new knowledge entries

**Integration Points:**
- Start of run_cycle (retrieval)
- End of run_cycle (storage)

```python
# Retrieved knowledge added to phase_outputs
phase_outputs["relevant_knowledge"] = [
    {"content": "...", "score": 0.95},
    ...
]
```

### 3. LLM Voting

**Purpose:** Multi-model consensus for critical decisions.

**When Activated:**
- Manual use via `voting_engine.vote()`
- Available in enhanced context for decision-making

**Usage:**
```python
if orchestrator.voting_engine:
    result = await orchestrator.voting_engine.vote(
        prompt="Select best architecture approach",
        options=["microservices", "monolith", "hybrid"],
        config=VoteConfig(models=["model_a", "model_b", "model_c"])
    )
    winner = result.winner
    confidence = result.consensus_level
```

### 4. Adversarial Agent

**Purpose:** Red-team critique to identify risks and edge cases.

**When Activated:**
- Goal length > 30 characters
- Adversarial agent is attached

**What It Does:**
- Analyzes goal/plan for potential issues
- Identifies edge cases, security risks, scalability concerns
- Provides risk score and severity-ranked findings

**Integration Point:** After context gathering, before planning

```python
# Findings added to context
context["enhanced_analysis"]["adversarial_findings"] = [
    {"severity": "high", "description": "..."},
    ...
]
context["enhanced_analysis"]["risk_score"] = 0.75
```

## Configuration

### Environment Variables

```bash
# Enable/disable features globally
AURA_ENABLE_SIMULATION=true
AURA_ENABLE_KNOWLEDGE=true
AURA_ENABLE_VOTING=true
AURA_ENABLE_ADVERSARIAL=true
```

### Per-Feature Configuration

```python
# Knowledge Base
knowledge_base = KnowledgeBase(
    store=KnowledgeStore(),
    embedding_provider=LocalEmbeddingProvider()
)

# Simulation Engine
simulation_engine = SimulationEngine(
    orchestrator=base_orchestrator,
    metrics_collector=MetricsCollector()
)

# Voting Engine
voting_engine = VotingEngine(
    model_adapter=model_adapter
)

# Adversarial Agent
adversarial_agent = AdversarialAgent(
    brain=brain,
    model=model
)

# Attach to orchestrator
orchestrator.attach_enhanced_features(
    simulation_engine=simulation_engine,
    knowledge_base=knowledge_base,
    voting_engine=voting_engine,
    adversarial_agent=adversarial_agent
)
```

## Output in Cycle Results

Enhanced features add the following to phase_outputs:

```json
{
  "relevant_knowledge": [...],
  "simulation_result": {
    "winner": "scenario_id",
    "insights_count": 3
  },
  "adversarial_critique": {
    "critique_id": "...",
    "risk_score": 0.75,
    "findings_count": 5
  }
}
```

## Performance Considerations

| Feature | Overhead | When to Disable |
|---------|----------|-----------------|
| Simulation | ~30-60s per cycle | Time-critical tasks |
| Knowledge | ~100ms query | Memory-constrained |
| Voting | ~10-30s per vote | Single-model setups |
| Adversarial | ~5-15s per cycle | Simple, low-risk tasks |

## Troubleshooting

### Feature Not Activating

Check logs for:
```
WARN: simulation_engine_attach_failed
WARN: knowledge_base_attach_failed
```

Ensure dependencies are installed:
```bash
pip install numpy scipy  # For simulation analysis
```

### Cycle Takes Too Long

Disable time-intensive features:
```python
orchestrator = attach_enhanced_features_to_orchestrator(
    orchestrator,
    enable_simulation=False,  # Skip simulations
    enable_adversarial=False   # Skip red-team critique
)
```

## Testing

Run tests for all enhanced features:

```bash
python3 -m pytest tests/test_simulation_engine.py \
                  tests/test_knowledge_base.py \
                  tests/test_voting_system.py \
                  tests/test_adversarial_agent.py -v
```

## Files Added/Modified

### New Files
- `core/simulation/` - Simulation Engine (6 files)
- `core/knowledge/` - Knowledge Base (4 files)
- `core/voting/` - Voting System (4 files)
- `agents/adversarial/` - Adversarial Agent (5 files)
- `memory/knowledge_store.py` - Persistent storage
- `core/enhanced_orchestrator.py` - Integration wrapper

### Modified Files
- `core/orchestrator.py` - Added integration points
- `agents/schemas.py` - Added new schemas

## Migration Guide

### From Standard Orchestrator

```python
# Before
orchestrator = LoopOrchestrator(agents=agents)

# After
orchestrator = LoopOrchestrator(agents=agents)
orchestrator = attach_enhanced_features_to_orchestrator(orchestrator)
```

No other code changes needed - features integrate transparently.

## Future Enhancements

Planned improvements:
- [ ] Automatic feature selection based on goal type
- [ ] Cross-feature learning (simulation → knowledge)
- [ ] Real-time voting during planning phase
- [ ] Collaborative adversarial critique with multiple strategies
