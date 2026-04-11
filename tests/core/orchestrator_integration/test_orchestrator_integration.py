# tests/core/orchestrator_integration/test_orchestrator_integration.py
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
import os
import sys

# Mock dependencies
mock_langgraph = MagicMock()
mock_langchain = MagicMock()
mock_redis = MagicMock()
sys.modules["langgraph"] = mock_langgraph
sys.modules["langgraph.graph"] = mock_langgraph.graph
sys.modules["langgraph.checkpoint"] = mock_langgraph.checkpoint
sys.modules["langgraph.checkpoint.base"] = mock_langgraph.checkpoint.base
sys.modules["langgraph.checkpoint.redis"] = mock_langgraph.checkpoint.redis
sys.modules["langchain_core"] = mock_langchain
sys.modules["langchain_core.messages"] = mock_langchain.messages
sys.modules["redis"] = mock_redis
sys.modules["huggingface_hub"] = MagicMock()
sys.modules["tokenizers"] = MagicMock()
sys.modules["onnxruntime"] = MagicMock()

from core.orchestrator_integration import OrchestratorFactory, ContractGovernedReactOrchestrator, HumanReviewRequired

class TestOrchestratorIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_legacy = MagicMock()
        self.mock_legacy.run_loop = MagicMock(return_value={"run_id": "legacy-999"})
        self.factory = OrchestratorFactory()

    def test_factory_returns_react_when_enabled(self):
        with patch.dict("os.environ", {"REACT_GRAPH_ENABLED": "true"}):
            with patch("core.orchestrator_integration.orchestrator_factory.config.get", return_value=True):
                orch = self.factory.create_orchestrator(self.mock_legacy)
                self.assertIsInstance(orch, ContractGovernedReactOrchestrator)

    def test_factory_returns_legacy_when_disabled(self):
        with patch.dict("os.environ", {"REACT_GRAPH_ENABLED": "false"}):
            with patch("core.orchestrator_integration.orchestrator_factory.config.get", return_value=False):
                orch = self.factory.create_orchestrator(self.mock_legacy)
                self.assertIs(self.mock_legacy, orch)

    @patch("core.orchestrator_integration.contract_governed_react_orchestrator.ReActGraphEngine")
    def test_contract_governed_happy_path(self, MockGraphEngine):
        with patch.dict("os.environ", {"REACT_GRAPH_ENABLED": "true"}):
            orch = ContractGovernedReactOrchestrator(self.mock_legacy)
            orch.redis_engine.setup = AsyncMock()
            orch.redis_engine.run_with_semantic_injection = AsyncMock(return_value={"thread_id": "r-1", "drift_score": 0.1})
            
            mock_run_result = MagicMock()
            mock_run_result.checkpoint = {"thread_id": "cp-1"}
            mock_run_result.memory = {"key": "value"}
            
            mock_engine_instance = MockGraphEngine.return_value
            mock_engine_instance.run = AsyncMock(return_value=mock_run_result)
            
            orch.redis_engine.advanced.human_in_loop_pause = AsyncMock(return_value=True)
            
            result = asyncio.run(orch.run_pipeline("Implement docstrings", {"l2_semantic": [0.1]*128}))
            self.assertEqual(result["mode"], "react")
            self.assertIn("drift_score", result)

    @patch("core.orchestrator_integration.contract_governed_react_orchestrator.ReActGraphEngine")
    def test_drift_triggers_human_review(self, MockGraphEngine):
        with patch.dict("os.environ", {"REACT_GRAPH_ENABLED": "true"}):
            orch = ContractGovernedReactOrchestrator(self.mock_legacy)
            MockGraphEngine.return_value.run = AsyncMock()
            orch.redis_engine.run_with_semantic_injection = AsyncMock(return_value={"thread_id": "drift-1", "drift_score": 0.45})
            orch.redis_engine.advanced.human_in_loop_pause = AsyncMock(return_value=False)
            
            with self.assertRaises(HumanReviewRequired):
                asyncio.run(orch.run_pipeline("drifted goal"))

    def test_input_validation_contract(self):
        orch = ContractGovernedReactOrchestrator(self.mock_legacy)
        with self.assertRaises(ValueError):
            asyncio.run(orch.run_pipeline(123))

    @patch("core.orchestrator_integration.contract_governed_react_orchestrator.ReActGraphEngine")
    def test_structured_payload_contract(self, MockGraphEngine):
        with patch.dict("os.environ", {"REACT_GRAPH_ENABLED": "true"}):
            orch = ContractGovernedReactOrchestrator(self.mock_legacy)
            orch.redis_engine.setup = AsyncMock()
            orch.redis_engine.run_with_semantic_injection = AsyncMock(return_value={"thread_id": "r-1", "drift_score": 0.0})
            
            mock_run_result = MagicMock()
            mock_run_result.checkpoint = {"thread_id": "cp-1"}
            mock_run_result.memory = {"snapshot": "data"}
            MockGraphEngine.return_value.run = AsyncMock(return_value=mock_run_result)
            
            result = asyncio.run(orch.run_pipeline("goal"))
            self.assertIsInstance(result, dict)
            self.assertIn("status", result)
            self.assertIn("mode", result)

    def test_concurrency_idempotent_setup(self):
        orch = ContractGovernedReactOrchestrator(self.mock_legacy)
        orch.redis_engine.setup = AsyncMock()
        # Mock setup to prevent actual instantiation
        orch._setup = AsyncMock()
        async def concurrent():
            tasks = [orch.run_pipeline(f"goal-{i}") for i in range(10)]
            # We need to mock the engines since run_pipeline uses them
            orch.redis_engine.run_with_semantic_injection = AsyncMock(return_value={"thread_id": "t", "drift_score": 0})
            orch.graph_engine = MagicMock()
            orch.graph_engine.run = AsyncMock(return_value=MagicMock(checkpoint={}, memory={}))
            return await asyncio.gather(*tasks)
        asyncio.run(concurrent())

    def test_redis_tls_path(self):
        with patch.dict("os.environ", {"REDIS_CHECKPOINT_URL": "rediss://user:pass@redis:6379/0"}):
            orch = ContractGovernedReactOrchestrator(self.mock_legacy)
            orch.redis_engine.setup = AsyncMock()
            asyncio.run(orch._setup())

    def test_full_legacy_fallback_reason(self):
        with patch.dict("os.environ", {"REACT_GRAPH_ENABLED": "false"}):
            with patch("core.orchestrator_integration.orchestrator_factory.config.get", return_value=False):
                orch = ContractGovernedReactOrchestrator(None)
                result = asyncio.run(orch.run_pipeline("goal"))
                self.assertEqual(result["fallback_reason"], "no_legacy_orchestrator")

if __name__ == "__main__":
    unittest.main(verbosity=2)
