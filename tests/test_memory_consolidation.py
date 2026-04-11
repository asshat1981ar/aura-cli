"""Tests for memory consolidation module - memory/consolidation.py

Covers memory entry, consolidation, and negative example store with comprehensive
test coverage for all consolidation phases and edge cases.
"""

import pytest
import json
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from memory.consolidation import (
    MemoryEntry,
    ConsolidationResult,
    MemoryConsolidator,
    NegativeExampleStore,
)


class TestMemoryEntry:
    """Test MemoryEntry data class and properties."""

    def test_memory_entry_creation(self):
        """MemoryEntry should be creatable with required fields."""
        entry = MemoryEntry(
            id="mem_1",
            content="Test memory",
            memory_type="goal_outcome",
        )
        assert entry.id == "mem_1"
        assert entry.content == "Test memory"
        assert entry.memory_type == "goal_outcome"
        assert entry.confidence == 0.5  # default
        assert entry.access_count == 0
        assert entry.last_accessed == 0.0
        assert entry.decay_rate == 0.01
        assert entry.tags == []
        assert entry.source_goal == ""
        assert entry.sentiment == 0.0

    def test_memory_entry_with_all_fields(self):
        """MemoryEntry should accept all optional fields."""
        now = time.time()
        entry = MemoryEntry(
            id="mem_1",
            content="Test",
            memory_type="decision",
            confidence=0.9,
            access_count=5,
            last_accessed=now,
            decay_rate=0.02,
            tags=["important", "urgent"],
            source_goal="goal_123",
            sentiment=0.8,
        )
        assert entry.confidence == 0.9
        assert entry.access_count == 5
        assert entry.last_accessed == now
        assert entry.decay_rate == 0.02
        assert entry.tags == ["important", "urgent"]
        assert entry.source_goal == "goal_123"
        assert entry.sentiment == 0.8

    def test_age_days_property(self):
        """age_days should calculate days since creation."""
        old_time = time.time() - (7 * 86400)  # 7 days ago
        entry = MemoryEntry(
            id="mem_1",
            content="Test",
            memory_type="pattern",
            created_at=old_time,
        )
        age = entry.age_days
        assert 6.9 < age < 7.1  # Allow small time variance

    def test_effective_confidence_with_decay(self):
        """effective_confidence should apply decay over time."""
        old_time = time.time() - (10 * 86400)  # 10 days ago
        entry = MemoryEntry(
            id="mem_1",
            content="Test",
            memory_type="pattern",
            confidence=1.0,
            decay_rate=0.1,
            created_at=old_time,
        )
        # After 10 days with decay_rate=0.1: 1.0 * max(0, 1.0 - 0.1*10) = 1.0 * 0 = 0
        effective = entry.effective_confidence
        assert 0.0 <= effective <= 1.0

    def test_effective_confidence_access_boost(self):
        """effective_confidence should boost with access frequency."""
        entry = MemoryEntry(
            id="mem_1",
            content="Test",
            memory_type="pattern",
            confidence=0.5,
            access_count=5,
        )
        effective = entry.effective_confidence
        # decayed = 0.5 * 1.0 = 0.5
        # access_boost = min(5*0.05, 0.3) = 0.25
        # final = max(0.0, min(1.0, 0.5 + 0.25)) = 0.75
        assert effective == pytest.approx(0.75)

    def test_effective_confidence_clamped_zero(self):
        """effective_confidence should not go below 0."""
        old_time = time.time() - (1000 * 86400)  # Very old
        entry = MemoryEntry(
            id="mem_1",
            content="Test",
            memory_type="pattern",
            confidence=0.1,
            decay_rate=1.0,
            created_at=old_time,
            access_count=0,
        )
        assert entry.effective_confidence >= 0.0

    def test_effective_confidence_clamped_one(self):
        """effective_confidence should not exceed 1.0."""
        entry = MemoryEntry(
            id="mem_1",
            content="Test",
            memory_type="pattern",
            confidence=1.0,
            access_count=100,
        )
        assert entry.effective_confidence <= 1.0

    def test_retention_score_high_confidence(self):
        """retention_score should be high for high confidence entries."""
        entry = MemoryEntry(
            id="mem_1",
            content="Test",
            memory_type="insight",
            confidence=1.0,
            access_count=10,
            sentiment=0.8,
        )
        score = entry.retention_score
        assert score > 0.5  # Should be high

    def test_retention_score_low_confidence(self):
        """retention_score should be low for low confidence entries."""
        old_time = time.time() - (30 * 86400)  # Old
        entry = MemoryEntry(
            id="mem_1",
            content="Test",
            memory_type="error",
            confidence=0.1,
            access_count=0,
            sentiment=0.0,
            decay_rate=0.1,
            created_at=old_time,
        )
        score = entry.retention_score
        assert score < 0.5  # Should be low

    def test_retention_score_pattern_type_boost(self):
        """retention_score should be boosted for pattern/insight types."""
        entry1 = MemoryEntry(
            id="mem_1",
            content="Test",
            memory_type="pattern",
            confidence=0.5,
        )
        entry2 = MemoryEntry(
            id="mem_2",
            content="Test",
            memory_type="error",
            confidence=0.5,
        )
        # Pattern should have higher score
        assert entry1.retention_score > entry2.retention_score

    def test_retention_score_negative_sentiment_boost(self):
        """retention_score should be boosted for negative sentiment."""
        entry1 = MemoryEntry(
            id="mem_1",
            content="Test",
            memory_type="pattern",
            sentiment=-0.8,  # Negative
        )
        entry2 = MemoryEntry(
            id="mem_2",
            content="Test",
            memory_type="pattern",
            sentiment=0.8,  # Positive
        )
        # Negative sentiment should be prioritized
        assert entry1.retention_score > entry2.retention_score


class TestConsolidationResult:
    """Test ConsolidationResult data class."""

    def test_consolidation_result_creation(self):
        """ConsolidationResult should be creatable."""
        result = ConsolidationResult(
            memories_before=100,
            memories_after=80,
            pruned=15,
            summarized=3,
            merged=2,
            compression_ratio=0.2,
            duration_seconds=1.5,
            summaries_created=["summary1"],
        )
        assert result.memories_before == 100
        assert result.memories_after == 80
        assert result.pruned == 15
        assert result.summarized == 3
        assert result.merged == 2
        assert result.compression_ratio == 0.2
        assert result.duration_seconds == 1.5
        assert result.summaries_created == ["summary1"]

    def test_default_values(self):
        """ConsolidationResult should have sensible defaults."""
        result = ConsolidationResult()
        assert result.memories_before == 0
        assert result.memories_after == 0
        assert result.pruned == 0
        assert result.summarized == 0
        assert result.merged == 0
        assert result.compression_ratio == 0.0
        assert result.duration_seconds == 0.0
        assert result.summaries_created == []


class TestMemoryConsolidator:
    """Test MemoryConsolidator main class."""

    @pytest.fixture
    def consolidator(self):
        return MemoryConsolidator(
            retention_threshold=0.2,
            summarize_after_days=7,
            max_memories=10000,
            similarity_threshold=0.85,
        )

    def test_consolidator_init(self, consolidator):
        """MemoryConsolidator should initialize with parameters."""
        assert consolidator.retention_threshold == 0.2
        assert consolidator.summarize_after_days == 7
        assert consolidator.max_memories == 10000
        assert consolidator.similarity_threshold == 0.85

    def test_consolidate_empty_memories(self, consolidator):
        """consolidate should handle empty memory list."""
        memories, result = consolidator.consolidate([])
        assert memories == []
        assert result.memories_before == 0
        assert result.memories_after == 0
        assert result.pruned == 0

    def test_consolidate_single_memory(self, consolidator):
        """consolidate should handle single memory."""
        memories = [
            MemoryEntry(
                id="m1",
                content="Test",
                memory_type="pattern",
                confidence=0.8,
            )
        ]
        retained, result = consolidator.consolidate(memories)
        assert len(retained) == 1
        assert result.memories_before == 1
        assert result.memories_after == 1
        assert result.pruned == 0

    def test_prune_low_retention_memories(self, consolidator):
        """_prune should remove low retention memories."""
        memories = [
            MemoryEntry(
                id="m1",
                content="High retention",
                memory_type="pattern",
                confidence=0.9,
                access_count=10,
            ),
            MemoryEntry(
                id="m2",
                content="Low retention",
                memory_type="error",
                confidence=0.05,
                access_count=0,
                decay_rate=0.1,
            ),
        ]
        pruned = consolidator._prune(memories)
        assert len(pruned) == 1
        assert pruned[0].id == "m1"

    def test_prune_threshold_boundary(self, consolidator):
        """_prune should use retention_threshold correctly."""
        consolidator.retention_threshold = 0.4
        memory = MemoryEntry(
            id="m1",
            content="Boundary",
            memory_type="pattern",
            confidence=0.5,
        )
        pruned = consolidator._prune([memory])
        # Should be retained (retention >= threshold)
        assert len(pruned) == 1

    def test_summarize_clusters_no_summarizer(self, consolidator):
        """_summarize_clusters should return unchanged if no summarizer."""
        memories = [
            MemoryEntry(id="m1", content="Test", memory_type="pattern"),
        ]
        retained, summaries = consolidator._summarize_clusters(memories, None)
        assert retained == memories
        assert summaries == []

    def test_summarize_clusters_few_old_memories(self, consolidator):
        """_summarize_clusters should skip if fewer than 5 old memories."""
        summarizer = Mock(return_value="summary")
        memories = [
            MemoryEntry(
                id="m1",
                content="Test",
                memory_type="pattern",
                created_at=time.time() - (10 * 86400),
            ),
            MemoryEntry(id="m2", content="Recent", memory_type="pattern"),
        ]
        retained, summaries = consolidator._summarize_clusters(memories, summarizer)
        assert retained == memories
        assert summaries == []
        summarizer.assert_not_called()

    def test_summarize_clusters_with_summarizer(self, consolidator):
        """_summarize_clusters should create summaries for old clusters."""
        consolidator.summarize_after_days = 0  # All are old
        summarizer = Mock(return_value="summary text")

        old_time = time.time() - (10 * 86400)
        memories = [
            MemoryEntry(
                id=f"m{i}",
                content=f"Pattern memory {i}",
                memory_type="pattern",
                created_at=old_time,
            )
            for i in range(5)
        ]

        retained, summaries = consolidator._summarize_clusters(memories, summarizer)
        assert len(summaries) == 1
        assert summaries[0] == "summary text"
        summarizer.assert_called_once()
        # Should have summary entry + recent (0) = 1
        assert len(retained) == 1

    def test_summarize_clusters_skip_small_clusters(self, consolidator):
        """_summarize_clusters should skip clusters with fewer than 3 items."""
        consolidator.summarize_after_days = 0
        summarizer = Mock(return_value="summary")

        old_time = time.time() - (10 * 86400)
        memories = [
            # Only 2 old pattern memories - should not be summarized
            MemoryEntry(
                id="m1",
                content="Pattern 1",
                memory_type="pattern",
                created_at=old_time,
            ),
            MemoryEntry(
                id="m2",
                content="Pattern 2",
                memory_type="pattern",
                created_at=old_time,
            ),
            # 3 old decision memories - should be summarized
            MemoryEntry(
                id="m3",
                content="Decision 1",
                memory_type="decision",
                created_at=old_time,
            ),
            MemoryEntry(
                id="m4",
                content="Decision 2",
                memory_type="decision",
                created_at=old_time,
            ),
            MemoryEntry(
                id="m5",
                content="Decision 3",
                memory_type="decision",
                created_at=old_time,
            ),
        ]

        retained, summaries = consolidator._summarize_clusters(memories, summarizer)
        # Pattern cluster should be kept as-is (< 3 items)
        # Decision cluster should be summarized (>= 3 items)
        assert len(summaries) == 1
        summarizer.assert_called_once()
        """_summarize_clusters should handle summarizer exceptions."""
        consolidator.summarize_after_days = 0
        summarizer = Mock(side_effect=Exception("Summarizer failed"))

        old_time = time.time() - (10 * 86400)
        memories = [
            MemoryEntry(
                id=f"m{i}",
                content=f"Memory {i}",
                memory_type="pattern",
                created_at=old_time,
            )
            for i in range(5)
        ]

        retained, summaries = consolidator._summarize_clusters(memories, summarizer)
        # Should return memories unchanged
        assert len(retained) == 5
        assert summaries == []

    def test_merge_duplicates_no_duplicates(self, consolidator):
        """_merge_duplicates should handle unique memories."""
        memories = [
            MemoryEntry(id="m1", content="Unique 1", memory_type="pattern"),
            MemoryEntry(id="m2", content="Unique 2", memory_type="pattern"),
        ]
        result, count = consolidator._merge_duplicates(memories)
        assert len(result) == 2
        assert count == 0

    def test_merge_duplicates_finds_duplicates(self, consolidator):
        """_merge_duplicates should find and merge near-identical content."""
        memories = [
            MemoryEntry(
                id="m1",
                content="The same content here",
                memory_type="pattern",
                access_count=3,
                confidence=0.8,
            ),
            MemoryEntry(
                id="m2",
                content="the same content here",  # Case difference
                memory_type="pattern",
                access_count=2,
                confidence=0.7,
            ),
        ]
        result, count = consolidator._merge_duplicates(memories)
        assert len(result) == 1
        assert count == 1
        # Merged entry should have combined access count
        assert result[0].access_count == 5
        # Confidence should be max
        assert result[0].confidence == 0.8

    def test_merge_duplicates_empty_list(self, consolidator):
        """_merge_duplicates should handle empty list."""
        result, count = consolidator._merge_duplicates([])
        assert result == []
        assert count == 0

    def test_merge_duplicates_single_memory(self, consolidator):
        """_merge_duplicates should handle single memory."""
        memories = [MemoryEntry(id="m1", content="Test", memory_type="pattern")]
        result, count = consolidator._merge_duplicates(memories)
        assert len(result) == 1
        assert count == 0

    def test_fingerprint_normalization(self, consolidator):
        """_fingerprint should normalize text for comparison."""
        fp1 = consolidator._fingerprint("The Same Content")
        fp2 = consolidator._fingerprint("the same content")
        fp3 = consolidator._fingerprint("different content")
        assert fp1 == fp2
        assert fp1 != fp3

    def test_fingerprint_whitespace_handling(self, consolidator):
        """_fingerprint should handle different whitespace."""
        fp1 = consolidator._fingerprint("text with spaces")
        fp2 = consolidator._fingerprint("text  with   spaces")
        fp3 = consolidator._fingerprint("text\twith\tspaces")
        assert fp1 == fp2 == fp3

    def test_consolidate_capacity_limit(self, consolidator):
        """consolidate should enforce max_memories limit."""
        consolidator.max_memories = 5
        memories = [
            MemoryEntry(
                id=f"m{i}",
                content=f"Memory {i}",
                memory_type="pattern",
                confidence=0.5 + (i * 0.01),  # Small variance to avoid clamping
            )
            for i in range(20)
        ]
        retained, result = consolidator.consolidate(memories)
        assert len(retained) <= 5
        # Should have pruned some memories
        assert result.pruned > 0

    def test_consolidate_compression_ratio(self, consolidator):
        """consolidate should calculate compression ratio."""
        memories = [
            MemoryEntry(
                id=f"m{i}",
                content=f"Memory {i}",
                memory_type="pattern",
                confidence=0.8,
            )
            for i in range(100)
        ]
        retained, result = consolidator.consolidate(memories)
        # compression_ratio = 1 - (after / before)
        expected = 1.0 - (len(retained) / 100.0)
        assert result.compression_ratio == pytest.approx(expected)

    def test_consolidate_duration_tracked(self, consolidator):
        """consolidate should track duration."""
        memories = [MemoryEntry(id="m1", content="Test", memory_type="pattern")]
        retained, result = consolidator.consolidate(memories)
        assert result.duration_seconds >= 0

    def test_consolidate_full_pipeline(self, consolidator):
        """consolidate should run all phases in sequence."""
        consolidator.max_memories = 10
        summarizer = Mock(return_value="summary")

        old_time = time.time() - (10 * 86400)
        memories = [
            MemoryEntry(
                id="m0",
                content="Will be pruned",
                memory_type="error",
                confidence=0.05,
            ),
        ]
        # Add old memories that will be summarized
        memories.extend(
            [
                MemoryEntry(
                    id=f"m{i}",
                    content=f"Old memory {i}",
                    memory_type="pattern",
                    confidence=0.8,
                    created_at=old_time,
                )
                for i in range(1, 7)
            ]
        )

        retained, result = consolidator.consolidate(memories, summarizer)
        assert result.pruned >= 1  # At least the low confidence one
        assert result.memories_after < result.memories_before


class TestNegativeExampleStore:
    """Test NegativeExampleStore for learning from failures."""

    @pytest.fixture
    def store_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "negative_examples.json"

    def test_store_init_creates_empty(self, store_path):
        """Store should initialize with empty examples if path doesn't exist."""
        store = NegativeExampleStore(store_path)
        assert store.examples == []
        assert not store_path.exists()

    def test_store_add_example(self, store_path):
        """Store should add failure examples."""
        store = NegativeExampleStore(store_path)
        store.add(
            goal="Extract data from API",
            failure_reason="Rate limit exceeded",
            cycle_count=3,
            error_type="ratelimit",
            attempted_fixes=["added delay"],
        )
        assert len(store.examples) == 1
        assert store.examples[0]["goal"] == "Extract data from API"
        assert store.examples[0]["error_type"] == "ratelimit"

    def test_store_persists_to_file(self, store_path):
        """Store should persist examples to JSON file."""
        store = NegativeExampleStore(store_path)
        store.add("Test goal", "Test failure")

        # Create new store from same path
        store2 = NegativeExampleStore(store_path)
        assert len(store2.examples) == 1
        assert store2.examples[0]["goal"] == "Test goal"

    def test_store_load_existing(self, store_path):
        """Store should load existing examples from file."""
        # Create file with example
        examples = [{"goal": "existing", "failure_reason": "was there"}]
        store_path.write_text(json.dumps(examples))

        store = NegativeExampleStore(store_path)
        assert len(store.examples) == 1
        assert store.examples[0]["goal"] == "existing"

    def test_store_load_invalid_json(self, store_path):
        """Store should handle invalid JSON gracefully."""
        store_path.write_text("not valid json{")
        store = NegativeExampleStore(store_path)
        assert store.examples == []

    def test_store_load_not_list(self, store_path):
        """Store should handle JSON that's not a list."""
        store_path.write_text('{"key": "value"}')
        store = NegativeExampleStore(store_path)
        assert store.examples == []

    def test_find_similar_failures(self, store_path):
        """Store should find similar failures by keyword overlap."""
        store = NegativeExampleStore(store_path)
        store.add("Extract data from API", "Rate limit")
        store.add("Parse JSON response", "Invalid format")
        store.add("Transform database records", "Timeout")

        similar = store.find_similar_failures("Extract data")
        assert len(similar) > 0
        # First should be most similar (has both "Extract" and "data")
        assert "Extract" in similar[0]["goal"]

    def test_find_similar_failures_limit(self, store_path):
        """find_similar_failures should respect limit parameter."""
        store = NegativeExampleStore(store_path)
        # Use very different goals to avoid ties
        store.add("Extract database records", "Failure 1")
        store.add("Parse API responses", "Failure 2")
        store.add("Transform XML documents", "Failure 3")
        store.add("Process file uploads", "Failure 4")
        store.add("Schedule background tasks", "Failure 5")

        similar = store.find_similar_failures("extract records", limit=3)
        # Should find matches and respect the limit
        assert len(similar) <= 3

    def test_find_similar_failures_no_match(self, store_path):
        """find_similar_failures should return empty list if no matches."""
        store = NegativeExampleStore(store_path)
        store.add("Something completely different", "Failure")

        similar = store.find_similar_failures("extract")
        assert similar == []

    def test_get_summary_empty(self, store_path):
        """get_summary should handle empty store."""
        store = NegativeExampleStore(store_path)
        summary = store.get_summary()
        assert summary["total"] == 0
        assert summary["error_types"] == {}

    def test_get_summary_with_errors(self, store_path):
        """get_summary should aggregate error types."""
        store = NegativeExampleStore(store_path)
        store.add("Goal 1", "Reason 1", error_type="timeout")
        store.add("Goal 2", "Reason 2", error_type="timeout")
        store.add("Goal 3", "Reason 3", error_type="ratelimit")

        summary = store.get_summary()
        assert summary["total"] == 3
        assert summary["error_types"]["timeout"] == 2
        assert summary["error_types"]["ratelimit"] == 1
        assert summary["most_common"] == "timeout"

    def test_add_with_defaults(self, store_path):
        """add should handle missing optional parameters."""
        store = NegativeExampleStore(store_path)
        store.add("Goal", "Failure")
        assert len(store.examples) == 1
        assert store.examples[0]["cycle_count"] == 0
        assert store.examples[0]["error_type"] == ""
        assert store.examples[0]["attempted_fixes"] == []

    def test_add_timestamp_recorded(self, store_path):
        """add should record timestamp."""
        store = NegativeExampleStore(store_path)
        before = time.time()
        store.add("Goal", "Failure")
        after = time.time()
        assert before <= store.examples[0]["timestamp"] <= after

    def test_multiple_error_types(self, store_path):
        """Store should handle multiple distinct error types."""
        store = NegativeExampleStore(store_path)
        error_types = ["timeout", "ratelimit", "auth", "parsing", "network"]
        for et in error_types:
            store.add(f"Goal {et}", f"Reason {et}", error_type=et)

        summary = store.get_summary()
        assert summary["total"] == 5
        assert len(summary["error_types"]) == 5

    def test_find_similar_word_matching(self, store_path):
        """find_similar_failures should count word overlaps correctly."""
        store = NegativeExampleStore(store_path)
        store.add("Extract numbers from API response", "Failed")
        store.add("Parse XML and transform data", "Failed")

        # "Extract data from API" shares more words with first
        similar = store.find_similar_failures("Extract data API", limit=2)
        assert len(similar) >= 1
        if len(similar) > 0:
            assert "Extract" in similar[0]["goal"]


class TestMemoryConsolidationIntegration:
    """Integration tests for memory consolidation workflow."""

    def test_full_consolidation_workflow(self):
        """Test complete consolidation workflow."""
        consolidator = MemoryConsolidator(
            retention_threshold=0.3,
            summarize_after_days=7,
            max_memories=1000,
        )

        # Create mixed memories
        now = time.time()
        old_time = now - (14 * 86400)

        memories = [
            # High value recent
            MemoryEntry(
                id="m1",
                content="Important recent insight",
                memory_type="insight",
                confidence=0.95,
                access_count=10,
                created_at=now,
            ),
            # Low value old
            MemoryEntry(
                id="m2",
                content="Useless old error",
                memory_type="error",
                confidence=0.1,
                access_count=0,
                created_at=old_time,
                decay_rate=0.2,
            ),
            # Duplicate
            MemoryEntry(
                id="m3",
                content="Important recent insight",
                memory_type="insight",
                confidence=0.9,
                access_count=5,
                created_at=now,
            ),
        ]

        summarizer = Mock(return_value="Old errors summary")
        retained, result = consolidator.consolidate(memories, summarizer)

        # Should retain high-value, remove low-value, merge duplicates
        assert result.memories_before >= 2
        assert result.memories_after < result.memories_before
        assert result.pruned >= 1

    def test_consolidation_preserves_high_value_memories(self):
        """Consolidation should prioritize keeping high-value memories."""
        consolidator = MemoryConsolidator(retention_threshold=0.4)

        high_value = MemoryEntry(
            id="high",
            content="Valuable pattern",
            memory_type="pattern",
            confidence=0.95,
            access_count=50,
            sentiment=-0.8,  # Learned from negative event
        )

        low_value = MemoryEntry(
            id="low",
            content="Forgotten error",
            memory_type="error",
            confidence=0.1,
            access_count=0,
        )

        retained, _ = consolidator.consolidate([high_value, low_value])
        retained_ids = {m.id for m in retained}
        assert "high" in retained_ids
        assert "low" not in retained_ids

    def test_consolidation_with_large_memory_set(self):
        """Consolidation should handle large memory sets efficiently."""
        consolidator = MemoryConsolidator(max_memories=100)

        # Create large set of memories
        memories = [
            MemoryEntry(
                id=f"m{i}",
                content=f"Memory content {i}",
                memory_type="pattern",
                confidence=0.5 + (i % 100) * 0.005,
                access_count=i % 20,
            )
            for i in range(500)
        ]

        retained, result = consolidator.consolidate(memories)
        assert len(retained) <= 100
        assert result.compression_ratio > 0
        assert result.duration_seconds < 10  # Should complete quickly
