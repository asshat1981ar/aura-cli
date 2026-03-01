"""Token-aware context assembly for ASCM v2."""
from __future__ import annotations

import json
from typing import List, TYPE_CHECKING

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
    ) -> str:
        """
        Greedy fill: sort hits by score * importance, add until budget exhausted.

        Formats:
          - "markdown": includes source attribution lines
          - "plain": content joined by newlines
          - "json": JSON list of {content, source_ref, score}
        """
        try:
            if not hits:
                return "" if format != "json" else "[]"

            # Sort by score * importance descending
            sorted_hits = sorted(
                hits,
                key=lambda h: h.score * getattr(h.record, "importance", 1.0),
                reverse=True,
            )

            selected = []
            remaining = budget_tokens

            for hit in sorted_hits:
                content = hit.record.content or ""
                token_cost = self._estimate_tokens(content)
                # Always include at least one item
                if selected and remaining <= 0:
                    break
                if token_cost > remaining and selected:
                    # Try to fit a truncated version
                    max_chars = remaining * 4
                    if max_chars < 20:
                        break
                    content = content[:max_chars] + "…"
                    token_cost = self._estimate_tokens(content)
                remaining -= token_cost
                selected.append((hit, content))

            if not selected:
                return "" if format != "json" else "[]"

            if format == "markdown":
                parts = []
                for hit, content in selected:
                    src = hit.record.source_ref or hit.record.source_type or "unknown"
                    score_str = f"{hit.score:.2f}"
                    parts.append(f"> [{src}] score={score_str}\n\n{content}")
                return "\n\n---\n\n".join(parts)

            elif format == "json":
                items = []
                for hit, content in selected:
                    items.append({
                        "content": content,
                        "source_ref": hit.record.source_ref,
                        "score": hit.score,
                    })
                return json.dumps(items)

            else:  # plain
                return "\n".join(content for _, content in selected)

        except Exception:
            return "" if format != "json" else "[]"
