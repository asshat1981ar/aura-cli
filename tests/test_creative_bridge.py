"""Tests for creative_bridge module."""
import pytest
from unittest.mock import Mock, AsyncMock
from core.creative_bridge import (
    CreativeIdea,
    ImplementationResult,
    CreativeImplementationBridge,
    CreativeBridgeError,
)


class TestCreativeIdea:
    """Test CreativeIdea dataclass."""

    def test_to_goal_formatting(self):
        idea = CreativeIdea(
            content="Add caching",
            requirements=["Use Redis", "TTL 5min"],
            validation_criteria=["Tests pass"],
            technique="SCAMPER",
            confidence=0.8,
        )
        goal = idea.to_goal()
        assert "Add caching" in goal
        assert "Use Redis" in goal
        assert "SCAMPER" in goal

    def test_to_goal_defaults(self):
        idea = CreativeIdea(content="Simple idea")
        goal = idea.to_goal()
        assert "Simple idea" in goal
        assert "unknown" in goal


class TestCreativeImplementationBridge:
    """Test CreativeImplementationBridge."""

    @pytest.fixture
    def mock_orchestrator(self):
        orch = Mock()
        orch.run_loop = AsyncMock(return_value={
            "success": True,
            "files_changed": ["file.py"],
            "tests_passed": True,
            "cycles": 3,
        })
        return orch

    @pytest.mark.asyncio
    async def test_implement_success(self, mock_orchestrator):
        bridge = CreativeImplementationBridge(mock_orchestrator)
        idea = CreativeIdea(
            content="Add feature",
            requirements=["Req 1"],
            technique="RPE",
        )
        
        result = await bridge.implement(idea)
        
        assert result.success is True
        assert result.files_changed == ["file.py"]
        assert result.tests_passed is True
        assert result.cycles_used == 3
        assert result.idea == idea

    @pytest.mark.asyncio
    async def test_implement_orchestrator_failure(self, mock_orchestrator):
        mock_orchestrator.run_loop = AsyncMock(side_effect=Exception("Failed"))
        bridge = CreativeImplementationBridge(mock_orchestrator)
        
        with pytest.raises(CreativeBridgeError):
            await bridge.implement(CreativeIdea(content="Test"))

    def test_get_success_rate_empty(self, mock_orchestrator):
        bridge = CreativeImplementationBridge(mock_orchestrator)
        assert bridge.get_success_rate() == 0.0

    @pytest.mark.asyncio
    async def test_get_success_rate_calculated(self, mock_orchestrator):
        bridge = CreativeImplementationBridge(mock_orchestrator)
        
        # Add successful implementation to history
        bridge._implementation_history.append(
            ImplementationResult(
                idea=CreativeIdea(content="Test", technique="RPE"),
                success=True,
            )
        )
        bridge._implementation_history.append(
            ImplementationResult(
                idea=CreativeIdea(content="Test2", technique="RPE"),
                success=False,
            )
        )
        
        assert bridge.get_success_rate() == 0.5
        assert bridge.get_success_rate(technique="RPE") == 0.5

    @pytest.mark.asyncio
    async def test_implement_batch(self, mock_orchestrator):
        bridge = CreativeImplementationBridge(mock_orchestrator, max_cycles=3)
        ideas = [
            CreativeIdea(content=f"Idea {i}", technique="SCAMPER")
            for i in range(3)
        ]
        
        results = await bridge.implement_batch(ideas, max_parallel=2)
        
        assert len(results) == 3
        assert all(r.success for r in results)


class TestImplementationResult:
    """Test ImplementationResult dataclass."""

    def test_to_dict(self):
        idea = CreativeIdea(content="Test", technique="RPE")
        result = ImplementationResult(
            idea=idea,
            success=True,
            files_changed=["a.py", "b.py"],
            cycles_used=5,
        )
        
        d = result.to_dict()
        assert d["success"] is True
        assert d["cycles_used"] == 5
        assert d["idea"]["technique"] == "RPE"
