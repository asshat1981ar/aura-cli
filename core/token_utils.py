"""
Utilities for token estimation and context management.
"""

def estimate_context_budget(goal: str, goal_type: str = "default") -> int:
    """Returns a token budget estimate based on goal_type and goal length."""
    base_budgets = {
        "docs": 2000,
        "bug_fix": 4000,
        "feature": 6000,
        "refactor": 4000,
        "security": 5000,
        "default": 4000,
    }
    budget = base_budgets.get(goal_type, base_budgets["default"])
    # Longer goals suggest more context is needed; add up to 2000 extra tokens
    extra = min(len(goal) // 10, 2000)
    return budget + extra

def compress_context(text: str, max_tokens: int) -> str:
    """Truncates text to fit within max_tokens (rough estimate: 4 chars per token)."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars]
