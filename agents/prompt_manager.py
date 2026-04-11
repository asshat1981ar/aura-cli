"""
Prompt Manager with Role-Based System Prompts and Caching.

Implements Pattern 6 (Role-Based System Prompts) and prompt caching
for efficient token usage and consistent agent behavior.
"""

import hashlib
import time
from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class PromptCacheEntry:
    """Cached prompt with metadata."""

    prompt: str
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def touch(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()


class PromptCache:
    """
    LRU cache for rendered prompts with TTL support.

    Benefits:
    - Reduces repeated prompt rendering cost
    - Enables prompt caching hints for LLM APIs (Anthropic, OpenAI)
    - Tracks prompt usage statistics
    """

    def __init__(self, max_size: int = 100, ttl_seconds: float = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, PromptCacheEntry] = {}
        self._hits = 0
        self._misses = 0

    def _make_key(self, template_name: str, params: tuple) -> str:
        """Create cache key from template name and parameters."""
        key_data = f"{template_name}:{hash(params)}"
        return hashlib.md5(key_data.encode(), usedforsecurity=False).hexdigest()

    def get(self, template_name: str, params: dict) -> Optional[str]:
        """Get cached prompt if available and not expired."""
        key = self._make_key(template_name, tuple(sorted(params.items())))
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        # Check TTL
        if time.time() - entry.created_at > self.ttl_seconds:
            del self._cache[key]
            self._misses += 1
            return None

        entry.touch()
        self._hits += 1
        return entry.prompt

    def set(self, template_name: str, params: dict, prompt: str):
        """Cache a rendered prompt."""
        key = self._make_key(template_name, tuple(sorted(params.items())))

        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._cache, key=lambda k: self._cache[k].last_accessed)
            del self._cache[oldest_key]

        self._cache[key] = PromptCacheEntry(prompt=prompt)

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {"hits": self._hits, "misses": self._misses, "hit_rate": hit_rate, "size": len(self._cache), "max_size": self.max_size}

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0


# Global prompt cache instance
_prompt_cache = PromptCache()


def get_prompt_cache() -> PromptCache:
    """Get the global prompt cache instance."""
    return _prompt_cache


# ============================================================================
# ROLE-BASED SYSTEM PROMPTS (Pattern 6)
# ============================================================================

SYSTEM_PROMPTS = {
    "planner": """You are a Senior Software Architect and Technical Lead with 20+ years of experience.

Your expertise includes:
- System design and architecture patterns
- Codebase analysis and structural assessment
- Risk identification and mitigation planning
- Task decomposition and parallelization strategies
- Technical debt evaluation and refactoring roadmaps

Core Principles:
1. Think holistically about system impact before proposing changes
2. Identify hidden dependencies and edge cases early
3. Prioritize maintainability and testability
4. Balance ideal solutions with pragmatic constraints
5. Always include verification steps for each action

Communication Style:
- Be methodical and thorough in your analysis
- Quantify risks with confidence scores
- Structure plans with clear dependencies and checkpoints
- Consider both short-term fixes and long-term architecture""",
    "critic": """You are a Principal Engineer and Code Reviewer with exacting standards.

Your responsibilities:
- Rigorous evaluation of technical proposals and implementations
- Security vulnerability assessment
- Performance and scalability analysis
- Code quality and maintainability review
- Adherence to best practices and design patterns

Review Criteria (in order of importance):
1. CORRECTNESS: Does it work? Are there logical errors?
2. SECURITY: Are there vulnerabilities or unsafe practices?
3. COMPLETENESS: Does it address all requirements?
4. FEASIBILITY: Can this be implemented as described?
5. MAINTAINABILITY: Will future developers understand this?

Communication Style:
- Be direct and specific in your critiques
- Categorize issues by severity (critical/major/minor/suggestion)
- Always provide actionable recommendations
- Acknowledge what is done well, not just what needs fixing
- Flag any assumptions or ambiguities""",
    "coder": """You are an Expert Python Developer specializing in clean, production-ready code.

Your expertise:
- Python 3.10+ with type hints and modern patterns
- Test-driven development and pytest
- Error handling and edge case management
- Performance optimization and profiling
- Code that is self-documenting and maintainable

Code Quality Standards:
1. Type hints on all function signatures and returns
2. Docstrings for all public functions and classes
3. Comprehensive error handling with specific exceptions
4. Unit tests for all non-trivial logic
5. Follow PEP 8 and project-specific conventions

Constraints:
- Never use bare `except:` clauses
- Always validate inputs before processing
- Prefer pathlib over string manipulation for paths
- Use dataclasses or Pydantic for data structures
- Handle resource cleanup (context managers, try/finally)

Communication Style:
- Explain your design decisions briefly
- Call out edge cases you've handled
- Note any dependencies you're introducing
- Be confident but acknowledge uncertainty when appropriate""",
    "synthesizer": """You are a Technical Integration Specialist.

Your role is to merge multiple perspectives (plan, critique, context) into cohesive execution instructions.

Key Abilities:
- Reconcile conflicting recommendations
- Identify the critical path through feedback
- Preserve important constraints while simplifying
- Translate high-level plans into specific actions

Integration Principles:
1. Address all critical issues raised in critique
2. Maintain the original goal's intent
3. Simplify without losing important detail
4. Ensure clear ownership of each action item""",
    "reflector": """You are a Systems Analyst focused on learning and improvement.

Your job is to analyze execution outcomes and extract actionable insights.

Analysis Framework:
1. What worked well? (preserve these patterns)
2. What failed or underperformed? (identify root causes)
3. What was unexpected? (update mental models)
4. What should change next time? (concrete improvements)

Output Requirements:
- Be specific about what happened vs. what was expected
- Quantify results when possible (timing, accuracy, etc.)
- Distinguish between process issues and implementation issues
- Suggest specific adjustments to prompts, tools, or workflows""",
}


def get_system_prompt(role: str) -> str:
    """Get the system prompt for a specific agent role."""
    return SYSTEM_PROMPTS.get(role, SYSTEM_PROMPTS["coder"])


# ============================================================================
# OPTIMIZED PROMPT TEMPLATES
# ============================================================================

# Token-efficient planner prompt with role context
PLANNER_PROMPT_TEMPLATE = """{system_prompt}

## Current Task
Create an execution plan for: {goal}

## Context
- Memory: {memory}
- Similar past problems: {similar}
- Known weaknesses: {weakness}
{backfill_instr}

## Output Format
Respond with JSON:
{{
    "analysis": "codebase and goal analysis",
    "gap_assessment": "structural gaps identified",
    "approach": "overall strategy",
    "risk_assessment": "failure modes and mitigations",
    "plan": [
        {{
            "step_number": 1,
            "description": "actionable step",
            "target_file": "optional/path.py",
            "verification": "how to verify"
        }}
    ],
    "confidence": 0.0-1.0,
    "total_steps": N,
    "estimated_complexity": "low|medium|high"
}}"""


# Token-efficient critic prompt with role context
CRITIC_PROMPT_TEMPLATE = """{system_prompt}

## Task
Evaluate this {target_type} against the goal.

## Goal
{task}

## Item to Review
{target_content}

## Context
{memory}

## Output Format
{{
    "initial_assessment": "first impression",
    "completeness_check": "requirements coverage",
    "feasibility_analysis": "implementation feasibility",
    "risk_identification": "risks and edge cases",
    "overall_assessment": "approve|approve_with_changes|request_changes|reject",
    "confidence": 0.0-1.0,
    "issues": [
        {{
            "severity": "critical|major|minor|suggestion",
            "category": "completeness|clarity|feasibility|alignment|safety|other",
            "description": "specific issue",
            "recommendation": "how to fix"
        }}
    ],
    "positive_aspects": ["what's done well"],
    "summary": "executive summary"
}}"""


# Token-efficient coder prompt with role context
CODER_PROMPT_TEMPLATE = """{system_prompt}

## Task
{task}

## Context
{memory}

{code_section}
{tests_section}
{feedback_section}

## Output Format
{{
    "problem_analysis": "understanding of requirements",
    "approach_selection": "chosen implementation strategy",
    "design_considerations": "edge cases and patterns",
    "testing_strategy": "verification plan",
    "aura_target": "path/to/file.py",
    "code": "complete Python implementation",
    "explanation": "high-level explanation",
    "dependencies": ["required imports"],
    "edge_cases_handled": ["handled scenarios"],
    "confidence": 0.0-1.0
}}"""


# ============================================================================
# PROMPT RENDERING WITH CACHING
# ============================================================================


def render_prompt(template_name: str, role: str, params: dict, use_cache: bool = True) -> str:
    """
    Render a prompt with role-based system context and caching.

    Args:
        template_name: Name of the template (planner, critic, coder)
        role: Agent role for system prompt
        params: Template parameters
        use_cache: Whether to use caching

    Returns:
        Rendered prompt string
    """
    cache = get_prompt_cache()

    # Check cache first
    if use_cache:
        cached = cache.get(template_name, params)
        if cached:
            return cached

    # Get template and system prompt
    templates = {
        "planner": PLANNER_PROMPT_TEMPLATE,
        "critic": CRITIC_PROMPT_TEMPLATE,
        "coder": CODER_PROMPT_TEMPLATE,
    }

    template = templates.get(template_name)
    if not template:
        raise ValueError(f"Unknown template: {template_name}")

    system_prompt = get_system_prompt(role)

    # Add system prompt to params
    render_params = {**params, "system_prompt": system_prompt}

    # Render prompt
    prompt = template.format(**render_params)

    # Cache the result
    if use_cache:
        cache.set(template_name, params, prompt)

    return prompt


def get_cached_prompt_stats() -> dict:
    """Get statistics about prompt cache usage."""
    return get_prompt_cache().get_stats()


def clear_prompt_cache():
    """Clear the prompt cache."""
    get_prompt_cache().clear()


# ============================================================================
# LEGACY TEMPLATES (for backward compatibility)
# ============================================================================

# Keep old templates available but mark as deprecated
PLANNER_COT_PROMPT_TEMPLATE = PLANNER_PROMPT_TEMPLATE
CRITIC_COT_PROMPT_TEMPLATE = CRITIC_PROMPT_TEMPLATE
CODER_COT_PROMPT_TEMPLATE = CODER_PROMPT_TEMPLATE
