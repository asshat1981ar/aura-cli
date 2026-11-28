"""Tests for core/context_budget.py — ContextBudgetManager."""

import json
import pytest

from core.context_budget import ContextBudgetManager
from core.memory_types import SearchHit


def make_hit(
    record_id: str,
    content: str,
    score: float = 0.9,
    source_ref: str = "test.py:1",
    source_type: str = "file",
    importance: float = 1.0,
) -> SearchHit:
    return SearchHit(
        record_id=record_id,
        content=content,
        score=score,
        source_ref=source_ref,
        metadata={"source_type": source_type, "importance": importance},
        explanation=f"score={score}",
    )


class TestAssembleEmptyInput:
    def test_empty_hits_markdown(self):
        result = ContextBudgetManager().assemble([], budget_tokens=100, format="markdown")
        assert result == ""

    def test_empty_hits_plain(self):
        result = ContextBudgetManager().assemble([], budget_tokens=100, format="plain")
        assert result == ""

    def test_empty_hits_json(self):
        result = ContextBudgetManager().assemble([], budget_tokens=100, format="json")
        assert result == "[]"


class TestAssembleBudget:
    def test_respects_budget(self):
        # Each word ~4 chars → ~1 token. 20 token budget.
        hits = [make_hit(f"r{i}", "word " * 10, score=0.9 - i * 0.1) for i in range(10)]
        result = ContextBudgetManager().assemble(hits, budget_tokens=20, format="plain")
        # Rough check: result fits within ~80 chars + markdown overhead
        assert len(result) <= 20 * 4 + 50  # allow for minimal overhead

    def test_always_includes_first_hit(self):
        # Budget of 0 should still include the first non-mandatory hit
        hit = make_hit("r1", "hello world")
        result = ContextBudgetManager().assemble([hit], budget_tokens=0, format="plain")
        assert "hello world" in result

    def test_truncates_oversized_item(self):
        # First hit fits. Second hit is far too large for the remaining budget.
        first = make_hit("r1", "a" * 40, score=0.95)  # ~10 tokens
        second = make_hit("r2", "b" * 2000, score=0.90)  # ~500 tokens — over budget
        result = ContextBudgetManager().assemble([first, second], budget_tokens=20, format="plain")
        # Second item should be present but truncated with ellipsis
        assert "…" in result
        assert "b" in result
        # Truncated content should be much shorter than the raw 2000 chars
        assert len(result) < 300


class TestAssembleFormats:
    def test_format_markdown(self):
        hit = make_hit("r1", "hello world", score=0.85, source_ref="core/foo.py:10")
        result = ContextBudgetManager().assemble([hit], budget_tokens=500, format="markdown")
        assert "> [core/foo.py:10] score=0.85" in result
        assert "hello world" in result

    def test_format_plain(self):
        hits = [make_hit("r1", "first"), make_hit("r2", "second", score=0.8)]
        result = ContextBudgetManager().assemble(hits, budget_tokens=500, format="plain")
        assert "first" in result
        assert "second" in result
        # No markdown attribution lines
        assert "> [" not in result

    def test_format_json(self):
        hit = make_hit("r1", "hello", score=0.9, source_ref="foo.py:1")
        result = ContextBudgetManager().assemble([hit], budget_tokens=500, format="json")
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["content"] == "hello"
        assert parsed[0]["source_ref"] == "foo.py:1"
        assert parsed[0]["score"] == 0.9

    def test_format_json_multiple(self):
        hits = [make_hit(f"r{i}", f"content{i}", score=0.9 - i * 0.05) for i in range(3)]
        result = ContextBudgetManager().assemble(hits, budget_tokens=500, format="json")
        parsed = json.loads(result)
        assert len(parsed) == 3


class TestAssembleSorting:
    def test_sorts_by_score_descending(self):
        low = make_hit("r1", "low score", score=0.3)
        high = make_hit("r2", "high score", score=0.9)
        result = ContextBudgetManager().assemble([low, high], budget_tokens=500, format="plain")
        # High score should appear first
        assert result.index("high score") < result.index("low score")

    def test_importance_affects_sort(self):
        low_imp = make_hit("r1", "low importance", score=0.8, importance=0.1)
        high_imp = make_hit("r2", "high importance", score=0.8, importance=2.0)
        result = ContextBudgetManager().assemble([low_imp, high_imp], budget_tokens=500, format="plain")
        assert result.index("high importance") < result.index("low importance")


class TestAssemblePerSourceCap:
    def test_per_source_cap_limits_duplicates(self):
        hits = [make_hit(f"r{i}", f"content{i}", score=0.9 - i * 0.01, source_ref="same_source.py:1") for i in range(5)]
        result = ContextBudgetManager().assemble(hits, budget_tokens=1000, format="plain", per_source_cap=2)
        # Only 2 items from same source should be included
        lines = [l for l in result.split("\n") if l.strip()]
        assert len(lines) <= 2

    def test_per_source_cap_zero_means_unlimited(self):
        hits = [make_hit(f"r{i}", f"content {i}", score=0.9 - i * 0.01, source_ref="same.py:1") for i in range(4)]
        result = ContextBudgetManager().assemble(hits, budget_tokens=1000, format="plain", per_source_cap=0)
        # All 4 should be included (cap=0 means no limit)
        for i in range(4):
            assert f"content {i}" in result

    def test_per_source_cap_mixed_sources(self):
        hits = [
            make_hit("r1", "from_a_1", score=0.95, source_ref="a.py:1"),
            make_hit("r2", "from_a_2", score=0.90, source_ref="a.py:1"),
            make_hit("r3", "from_b_1", score=0.85, source_ref="b.py:1"),
        ]
        result = ContextBudgetManager().assemble(hits, budget_tokens=1000, format="plain", per_source_cap=1)
        assert "from_a_1" in result
        assert "from_a_2" not in result  # capped
        assert "from_b_1" in result


class TestAssembleMandatoryIds:
    def test_mandatory_ids_included_first(self):
        # mandatory hit has a very low score — would normally be last
        mandatory = make_hit("must", "mandatory content", score=0.1)
        optional = make_hit("opt", "optional content", score=0.99)
        result = ContextBudgetManager().assemble([optional, mandatory], budget_tokens=500, format="plain", mandatory_ids=["must"])
        # Both present
        assert "mandatory content" in result
        assert "optional content" in result
        # Mandatory appears before optional
        assert result.index("mandatory content") < result.index("optional content")

    def test_mandatory_ids_included_even_with_tiny_budget(self):
        mandatory = make_hit("must", "I must appear", score=0.5)
        result = ContextBudgetManager().assemble([mandatory], budget_tokens=1, format="plain", mandatory_ids=["must"])
        assert "I must appear" in result

    def test_mandatory_ids_none_is_safe(self):
        hit = make_hit("r1", "hello")
        result = ContextBudgetManager().assemble([hit], budget_tokens=500, mandatory_ids=None)
        assert "hello" in result

    def test_mandatory_ids_empty_list_is_safe(self):
        hit = make_hit("r1", "hello")
        result = ContextBudgetManager().assemble([hit], budget_tokens=500, mandatory_ids=[])
        assert "hello" in result


class TestAssembleMarkdownFallbackSource:
    def test_falls_back_to_source_type_when_no_source_ref(self):
        hit = SearchHit(
            record_id="r1",
            content="test content",
            score=0.8,
            source_ref="",
            metadata={"source_type": "memory", "importance": 1.0},
            explanation="test",
        )
        result = ContextBudgetManager().assemble([hit], budget_tokens=500, format="markdown")
        assert "> [memory] score=0.80" in result
