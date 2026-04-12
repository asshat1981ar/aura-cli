"""
TDD-generated Orchestrator Factory – Contract-Governed Integration Point
Zero breaking changes to aura-cli/core/orchestrator.py
Supports REACT_GRAPH_ENABLED, legacy fallback, Redis semantic checkpointing.
"""

import os
import logging
from typing import Optional
from core.orchestrator import LoopOrchestrator
from core.config_manager import config
from .contract_governed_react_orchestrator import ContractGovernedReactOrchestrator

logger = logging.getLogger(__name__)

class OrchestratorFactory:
    """Production factory – selects ReAct or legacy path at runtime."""

    @staticmethod
    def create_orchestrator(legacy_orchestrator: Optional[LoopOrchestrator] = None) -> LoopOrchestrator:
        # Check config manager first, fallback to env var
        react_enabled = config.get("REACT_GRAPH_ENABLED", False)
        if not react_enabled:
            react_enabled = os.getenv("REACT_GRAPH_ENABLED", "false").lower() == "true"
            
        if react_enabled:
            logger.info("OrchestratorFactory → ContractGovernedReactOrchestrator (LangGraph + Redis)")
            return ContractGovernedReactOrchestrator(legacy_orchestrator=legacy_orchestrator)
        logger.info("OrchestratorFactory → legacy AURA orchestrator (fallback)")
        return legacy_orchestrator or LoopOrchestrator()  # graceful no-op fallback
