"""Enhanced orchestrator integrating Simulation, Knowledge, Voting, and Adversarial features."""


__all__ = [
    "EnhancedOrchestrator",
    "enhance_orchestrator",
    "attach_enhanced_features_to_orchestrator",
]
from typing import Any, Dict, Optional

from core.logging_utils import log_json
from core.orchestrator import LoopOrchestrator


class EnhancedOrchestrator:
    """
    Enhanced orchestrator that integrates the new features:
    - Simulation Engine for "what-if" testing
    - Knowledge Base for insight sharing
    - LLM Voting for consensus decisions
    - Adversarial Agent for red-team critique
    """
    
    def __init__(
        self,
        base_orchestrator: Optional[LoopOrchestrator] = None,
        enable_simulation: bool = True,
        enable_knowledge: bool = True,
        enable_voting: bool = True,
        enable_adversarial: bool = True
    ):
        """
        Initialize the enhanced orchestrator.
        
        Args:
            base_orchestrator: Base orchestrator to wrap
            enable_simulation: Enable simulation features
            enable_knowledge: Enable knowledge base features
            enable_voting: Enable voting features
            enable_adversarial: Enable adversarial critique features
        """
        self.base = base_orchestrator
        self.enable_simulation = enable_simulation
        self.enable_knowledge = enable_knowledge
        self.enable_voting = enable_voting
        self.enable_adversarial = enable_adversarial
        
        # Initialize components
        self.simulation_engine = None
        self.knowledge_base = None
        self.voting_engine = None
        self.adversarial_agent = None
        
        self._init_components()
    
    def _init_components(self):
        """Initialize feature components."""
        if self.enable_simulation:
            try:
                from core.simulation import SimulationEngine
                self.simulation_engine = SimulationEngine()
                log_json("INFO", "simulation_engine_initialized")
            except Exception as e:
                log_json("WARN", "simulation_engine_init_failed", {"error": str(e)})
        
        if self.enable_knowledge:
            try:
                from core.knowledge import KnowledgeBase
                self.knowledge_base = KnowledgeBase()
                log_json("INFO", "knowledge_base_initialized")
            except Exception as e:
                log_json("WARN", "knowledge_base_init_failed", {"error": str(e)})
        
        if self.enable_voting:
            try:
                from core.voting import VotingEngine
                self.voting_engine = VotingEngine()
                log_json("INFO", "voting_engine_initialized")
            except Exception as e:
                log_json("WARN", "voting_engine_init_failed", {"error": str(e)})
        
        if self.enable_adversarial:
            try:
                from agents.adversarial import AdversarialAgent
                self.adversarial_agent = AdversarialAgent()
                log_json("INFO", "adversarial_agent_initialized")
            except Exception as e:
                log_json("WARN", "adversarial_agent_init_failed", {"error": str(e)})
    
    async def process_with_enhancements(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        use_simulation: bool = True,
        use_knowledge: bool = True,
        use_voting: bool = True,
        use_adversarial: bool = True
    ) -> Dict[str, Any]:
        """
        Process a goal with enhanced features.
        
        Args:
            goal: The goal to process
            context: Additional context
            use_simulation: Whether to use simulation
            use_knowledge: Whether to use knowledge base
            use_voting: Whether to use voting
            use_adversarial: Whether to use adversarial critique
            
        Returns:
            Processing result with enhancements
        """
        context = context or {}
        enhancements = {}
        
        # 1. Query knowledge base for relevant insights
        if use_knowledge and self.knowledge_base:
            relevant_knowledge = await self.knowledge_base.query(
                query_text=goal,
                max_results=5
            )
            if relevant_knowledge:
                context["relevant_knowledge"] = [
                    {
                        "content": r.entry.content,
                        "confidence": r.composite_score
                    }
                    for r in relevant_knowledge
                ]
                enhancements["knowledge_retrieved"] = len(relevant_knowledge)
        
        # 2. Run simulation if appropriate
        if use_simulation and self.simulation_engine:
            # Determine if this goal benefits from simulation
            if any(kw in goal.lower() for kw in ["test", "compare", "optimize", "config"]):
                from core.simulation import SimulationConfig
                
                sim_config = SimulationConfig(
                    name=f"sim_{goal[:30]}",
                    base_scenario="agent_configuration",
                    variables={"approach": ["conservative", "aggressive"]},
                    max_parallel=2
                )
                
                try:
                    sim_result = await self.simulation_engine.run_simulation(sim_config)
                    enhancements["simulation"] = {
                        "winner": sim_result.winner.scenario_id if sim_result.winner else None,
                        "insights_count": len(sim_result.insights)
                    }
                    
                    # Use simulation winner to inform context
                    if sim_result.winner:
                        context["recommended_approach"] = sim_result.winner.scenario_name
                        
                except Exception as e:
                    log_json("WARN", "simulation_failed", {"error": str(e)})
        
        # 3. Get adversarial critique if appropriate
        if use_adversarial and self.adversarial_agent:
            if len(goal) > 50:  # Only critique substantial goals
                try:
                    critique = await self.adversarial_agent.critique(
                        target=goal,
                        target_type="plan",  # Treat goal as plan
                        context=context,
                        intensity=0.7
                    )
                    
                    enhancements["adversarial_critique"] = {
                        "critique_id": critique.critique_id,
                        "risk_score": critique.risk_score,
                        "findings_count": len(critique.findings),
                        "assessment": critique.overall_assessment
                    }
                    
                    # Add critique findings to context
                    if critique.findings:
                        context["identified_risks"] = [
                            {
                                "severity": f.severity,
                                "description": f.description
                            }
                            for f in critique.findings[:3]
                        ]
                        
                except Exception as e:
                    log_json("WARN", "adversarial_critique_failed", {"error": str(e)})
        
        # 4. Vote on approach if multiple options
        if use_voting and self.voting_engine and "approaches" in context:
            try:
                from core.voting import VoteConfig
                
                vote_config = VoteConfig(
                    models=["model_a", "model_b", "model_c"],  # Configurable
                    strategy="weighted",
                    min_consensus=0.6
                )
                
                vote_result = await self.voting_engine.vote(
                    prompt=f"Select best approach for: {goal}",
                    options=context["approaches"],
                    config=vote_config
                )
                
                enhancements["voting"] = {
                    "winner": vote_result.winner,
                    "consensus": vote_result.consensus_level,
                    "confidence": vote_result.winner_confidence
                }
                
            except Exception as e:
                log_json("WARN", "voting_failed", {"error": str(e)})
        
        # 5. Execute with base orchestrator
        if self.base:
            result = self.base.run_cycle(goal, context)
        else:
            result = {"goal": goal, "context": context}
        
        result["enhancements"] = enhancements
        
        # 6. Store learnings in knowledge base
        if use_knowledge and self.knowledge_base:
            from core.knowledge import KnowledgeEntry, KnowledgeCategory
            
            lesson = KnowledgeEntry(
                content=f"Processed goal: {goal}. Result: {result.get('status', 'unknown')}",
                source="enhanced_orchestrator",
                category=KnowledgeCategory.LESSON_LEARNED,
                tags=["orchestration", "enhanced"],
                confidence=0.7
            )
            
            try:
                await self.knowledge_base.add(lesson)
            except Exception as e:
                log_json("WARN", "knowledge_store_failed", {"error": str(e)})
        
        return result
    
    def get_feature_status(self) -> Dict[str, Any]:
        """Get status of all features."""
        return {
            "simulation": {
                "enabled": self.enable_simulation,
                "initialized": self.simulation_engine is not None
            },
            "knowledge": {
                "enabled": self.enable_knowledge,
                "initialized": self.knowledge_base is not None
            },
            "voting": {
                "enabled": self.enable_voting,
                "initialized": self.voting_engine is not None
            },
            "adversarial": {
                "enabled": self.enable_adversarial,
                "initialized": self.adversarial_agent is not None
            }
        }


# Convenience function for quick enhancement
def enhance_orchestrator(
    orchestrator: Optional[LoopOrchestrator] = None,
    **kwargs
) -> EnhancedOrchestrator:
    """
    Create an enhanced orchestrator.
    
    Args:
        orchestrator: Base orchestrator to wrap
        **kwargs: Additional configuration options
        
    Returns:
        EnhancedOrchestrator instance
    """
    return EnhancedOrchestrator(
        base_orchestrator=orchestrator,
        enable_simulation=kwargs.get('simulation', True),
        enable_knowledge=kwargs.get('knowledge', True),
        enable_voting=kwargs.get('voting', True),
        enable_adversarial=kwargs.get('adversarial', True)
    )


def attach_enhanced_features_to_orchestrator(
    orchestrator: LoopOrchestrator,
    enable_simulation: bool = True,
    enable_knowledge: bool = True,
    enable_voting: bool = True,
    enable_adversarial: bool = True
) -> LoopOrchestrator:
    """
    Attach enhanced features to an existing LoopOrchestrator instance.
    
    This is the recommended way to enable the new features in existing code.
    
    Args:
        orchestrator: The LoopOrchestrator to enhance
        enable_simulation: Enable Simulation Engine
        enable_knowledge: Enable Knowledge Base
        enable_voting: Enable Voting Engine
        enable_adversarial: Enable Adversarial Agent
        
    Returns:
        The same orchestrator instance with enhanced features attached
        
    Example:
        >>> from core.orchestrator import LoopOrchestrator
        >>> from core.enhanced_orchestrator import attach_enhanced_features_to_orchestrator
        >>> 
        >>> orch = LoopOrchestrator(agents=default_agents())
        >>> orch = attach_enhanced_features_to_orchestrator(orch)
        >>> # Now orch has simulation, knowledge, voting, and adversarial capabilities
    """
    simulation_engine = None
    knowledge_base = None
    voting_engine = None
    adversarial_agent = None
    
    # Initialize Simulation Engine
    if enable_simulation:
        try:
            from core.simulation import SimulationEngine
            simulation_engine = SimulationEngine()
            log_json("INFO", "simulation_engine_attached")
        except Exception as e:
            log_json("WARN", "simulation_engine_attach_failed", {"error": str(e)})
    
    # Initialize Knowledge Base
    if enable_knowledge:
        try:
            from core.knowledge import KnowledgeBase
            knowledge_base = KnowledgeBase()
            log_json("INFO", "knowledge_base_attached")
        except Exception as e:
            log_json("WARN", "knowledge_base_attach_failed", {"error": str(e)})
    
    # Initialize Voting Engine
    if enable_voting:
        try:
            from core.voting import VotingEngine
            voting_engine = VotingEngine()
            log_json("INFO", "voting_engine_attached")
        except Exception as e:
            log_json("WARN", "voting_engine_attach_failed", {"error": str(e)})
    
    # Initialize Adversarial Agent
    if enable_adversarial:
        try:
            from agents.adversarial import AdversarialAgent
            adversarial_agent = AdversarialAgent(
                brain=getattr(orchestrator, 'brain', None),
                model=getattr(orchestrator, 'model', None)
            )
            log_json("INFO", "adversarial_agent_attached")
        except Exception as e:
            log_json("WARN", "adversarial_agent_attach_failed", {"error": str(e)})
    
    # Attach all features to the orchestrator
    orchestrator.attach_enhanced_features(
        simulation_engine=simulation_engine,
        knowledge_base=knowledge_base,
        voting_engine=voting_engine,
        adversarial_agent=adversarial_agent
    )
    
    return orchestrator
