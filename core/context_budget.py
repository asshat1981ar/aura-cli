"""Token-aware context assembly for ASCM v2."""

from __future__ import annotations

import json
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.memory_types import SearchHit


class ContextBudgetManager:
    """Assembles context from search hits within a token budget."""

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def assemble(
        self,
        hits: "List[SearchHit]",
        budget_tokens: int,
        format: str = "markdown",
        per_source_cap: int = 0,
        mandatory_ids: Optional[List[str]] = None,
    ) -> str:
        """
        Greedy fill: sort hits by score * importance, add until budget exhausted.

        Args:
            hits: Retrieved search hits to assemble.
            budget_tokens: Maximum token budget for the assembled context.
            format: Output format — "markdown", "plain", or "json".
            per_source_cap: If > 0, limit hits per unique source_ref to this count.
            mandatory_ids: record_ids that must be included regardless of budget.

        Formats:
          - "markdown": includes source attribution lines
          - "plain": content joined by newlines
          - "json": JSON list of {content, source_ref, score}
        """
        try:
            if not hits:
                return "" if format != "json" else "[]"

            mandatory_set = set(mandatory_ids or [])

            # Separate mandatory hits — included first, regardless of budget
            mandatory_hits = [h for h in hits if h.record_id in mandatory_set]
            remaining_hits = [h for h in hits if h.record_id not in mandatory_set]

            # Sort remaining by score * importance descending
            remaining_hits.sort(
                key=lambda h: h.score * h.metadata.get("importance", 1.0),
                reverse=True,
            )

            selected: List[tuple] = []  # (hit, content)
            remaining_budget = budget_tokens
            source_counts: dict[str, int] = {}

            # Include mandatory hits first (consume budget but never skip)
            for hit in mandatory_hits:
                content = hit.content or ""
                token_cost = self._estimate_tokens(content)
                remaining_budget -= token_cost
                selected.append((hit, content))

            # Greedy fill remaining
            for hit in remaining_hits:
                if per_source_cap > 0:
                    src = hit.source_ref or ""
                    if source_counts.get(src, 0) >= per_source_cap:
                        continue

                content = hit.content or ""
                token_cost = self._estimate_tokens(content)

                # Always include at least one non-mandatory item
                if selected and remaining_budget <= 0:
                    break
                if token_cost > remaining_budget and selected:
                    # Try to fit a truncated version
                    max_chars = remaining_budget * 4
                    if max_chars < 20:
                        break
                    content = content[:max_chars] + "…"
                    token_cost = self._estimate_tokens(content)

                remaining_budget -= token_cost
                selected.append((hit, content))

                if per_source_cap > 0:
                    src = hit.source_ref or ""
                    source_counts[src] = source_counts.get(src, 0) + 1

            if not selected:
                return "" if format != "json" else "[]"

            if format == "markdown":
                parts = []
                for hit, content in selected:
                    src = hit.source_ref or hit.metadata.get("source_type", "unknown")
                    score_str = f"{hit.score:.2f}"
                    parts.append(f"> [{src}] score={score_str}\n\n{content}")
                return "\n\n---\n\n".join(parts)

            elif format == "json":
                items = []
                for hit, content in selected:
                    items.append(
                        {
                            "content": content,
                            "source_ref": hit.source_ref,
                            "score": hit.score,
                        }
                    )
                return json.dumps(items)

            else:  # plain
                return "\n".join(content for _, content in selected)

        except (OSError, IOError, ValueError):
            return "" if format != "json" else "[]"
