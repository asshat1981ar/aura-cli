"""Tests for creative_memory module."""
import pytest
from unittest.mock import Mock
from core.creative_memory import (
    CreativePattern,
    CreativePatternMemory,
    RecallError,
    StorageError,
)


class TestCreativePattern:
    """Test CreativePattern dataclass."""

    def test_to_memory_text(self):
        pattern = CreativePattern(
            id="test-001",
            content="Test pattern",
            domain="api_design",
            technique="SCAMPER",
            success_rate=0.85,
            usage_count=10,
        )
        text = pattern.to_memory_text()
        assert "test-001" in text
        assert "api_design" in text
        assert "0.85" in text

    def test_to_dict_roundtrip(self):
        pattern = CreativePattern(
            id="test-002",
            content="Content",
            domain="testing",
            technique="RPE",
        )
        d = pattern.to_dict()
        restored = CreativePattern.from_dict(d)
        assert restored.id == pattern.id
        assert restored.content == pattern.content


class TestCreativePatternMemory:
    """Test CreativePatternMemory."""

    @pytest.fixture
    def mock_brain(self):
        brain = Mock()
        brain.remember = Mock()
        brain.recall_with_budget = Mock(return_value=[])
        return brain

    def test_record_pattern(self, mock_brain):
        memory = CreativePatternMemory(mock_brain)
        pattern = CreativePattern(
            id="pattern-001",
            content="Use caching",
            domain="performance",
            technique="SCAMPER",
        )
        
        memory.record(pattern)
        
        mock_brain.remember.assert_called_once()
        call_args = mock_brain.remember.call_args
        assert "Use caching" in call_args.kwargs["text"]

    def test_record_with_usage_tracking(self, mock_brain):
        memory = CreativePatternMemory(mock_brain)
        pattern = CreativePattern(
            id="pattern-002",
            content="Test",
            domain="test",
            technique="RPE",
            usage_count=5,
        )
        
        memory.record(pattern, track_usage=True)
        
        assert pattern.usage_count == 6
        assert pattern.last_used is not None

    def test_recall_patterns(self, mock_brain):
        mock_brain.recall_with_budget.return_value = [
            {
                "text": "Pattern content",
                "metadata": {
                    "type": "creative_pattern",
                    "pattern_id": "p001",
                    "domain": "api_design",
                    "technique": "SCAMPER",
                    "success_rate": 0.9,
                    "usage_count": 5,
                    "related_patterns": "[]",
                }
            }
        ]
        
        memory = CreativePatternMemory(mock_brain)
        patterns = memory.recall("api_design", "caching", top_k=5)
        
        assert len(patterns) == 1
        assert patterns[0].id == "p001"
        assert patterns[0].success_rate == 0.9

    def test_recall_filters_by_success_rate(self, mock_brain):
        mock_brain.recall_with_budget.return_value = [
            {
                "text": "Low success",
                "metadata": {
                    "type": "creative_pattern",
                    "pattern_id": "low",
                    "domain": "test",
                    "technique": "RPE",
                    "success_rate": 0.3,
                    "usage_count": 1,
                    "related_patterns": "[]",
                }
            },
            {
                "text": "High success",
                "metadata": {
                    "type": "creative_pattern",
                    "pattern_id": "high",
                    "domain": "test",
                    "technique": "RPE",
                    "success_rate": 0.9,
                    "usage_count": 10,
                    "related_patterns": "[]",
                }
            }
        ]
        
        memory = CreativePatternMemory(mock_brain)
        patterns = memory.recall("test", "query", min_success_rate=0.5)
        
        assert len(patterns) == 1
        assert patterns[0].id == "high"

    def test_cross_pollinate(self, mock_brain):
        mock_brain.recall_with_budget.return_value = [
            {
                "text": "Source pattern",
                "metadata": {
                    "type": "creative_pattern",
                    "pattern_id": "src",
                    "domain": "web",
                    "technique": "SCAMPER",
                    "success_rate": 0.8,
                    "usage_count": 5,
                    "related_patterns": "[]",
                }
            }
        ]
        
        memory = CreativePatternMemory(mock_brain)
        analogies = memory.cross_pollinate("web", "mobile", top_k=3)
        
        assert len(analogies) > 0
        assert analogies[0]["source_domain"] == "web"
        assert analogies[0]["target_domain"] == "mobile"

    def test_update_success_rate(self, mock_brain):
        mock_brain.get = Mock(return_value={
            "text": "Pattern",
            "metadata": {
                "type": "creative_pattern",
                "pattern_id": "test",
                "domain": "test",
                "technique": "RPE",
                "success_rate": 0.5,
                "usage_count": 10,
                "related_patterns": "[]",
            }
        })
        
        memory = CreativePatternMemory(mock_brain)
        # Pre-populate cache
        memory._cache["test"] = CreativePattern(
            id="test",
            content="Pattern",
            domain="test",
            technique="RPE",
            success_rate=0.5,
            usage_count=10,
        )
        
        memory.update_success_rate("test", success=True)
        
        # Success rate should increase (EMA formula)
        assert memory._cache["test"].success_rate > 0.5

    def test_get_domain_stats(self, mock_brain):
        mock_brain.recall_with_budget.return_value = [
            {
                "text": "Pattern 1",
                "metadata": {
                    "type": "creative_pattern",
                    "pattern_id": "p1",
                    "domain": "api",
                    "technique": "RPE",
                    "success_rate": 0.8,
                    "usage_count": 5,
                    "related_patterns": "[]",
                }
            },
            {
                "text": "Pattern 2",
                "metadata": {
                    "type": "creative_pattern",
                    "pattern_id": "p2",
                    "domain": "api",
                    "technique": "SCAMPER",
                    "success_rate": 0.9,
                    "usage_count": 10,
                    "related_patterns": "[]",
                }
            }
        ]
        
        memory = CreativePatternMemory(mock_brain)
        stats = memory.get_domain_stats("api")
        
        assert stats["pattern_count"] == 2
        assert abs(stats["avg_success_rate"] - 0.85) < 0.001
        assert stats["total_usage"] == 15

    def test_storage_error_handling(self, mock_brain):
        mock_brain.remember = Mock(side_effect=Exception("DB Error"))
        memory = CreativePatternMemory(mock_brain)
        
        with pytest.raises(StorageError):
            memory.record(CreativePattern(
                id="fail",
                content="Test",
                domain="test",
                technique="RPE",
            ))

    def test_recall_error_handling(self, mock_brain):
        mock_brain.recall_with_budget = Mock(side_effect=Exception("Recall Error"))
        memory = CreativePatternMemory(mock_brain)
        
        with pytest.raises(RecallError):
            memory.recall("domain", "query")
