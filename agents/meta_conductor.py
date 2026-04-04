"""
MetaConductor Agent - Orchestrates the complete Innovation Catalyst workflow.

This agent manages the 5-phase innovation process:
1. IMMERSION - Deep understanding of the problem
2. DIVERGENCE - Generate many ideas via InnovationSwarm
3. CONVERGENCE - Evaluate and select best ideas
4. INCUBATION - Let ideas develop (simulated pause)
5. TRANSFORMATION - Prepare solutions for implementation
"""

import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from agents.schemas import (
    InnovationPhase, InnovationSessionState
)
from agents.innovation_swarm import InnovationSwarm
from core.logging_utils import log_json


class MetaConductor:
    """
    High-level orchestrator for Innovation Catalyst sessions.
    
    Manages the complete 5-phase innovation workflow and coordinates
    between the InnovationSwarm and other AURA components.
    """
    
    capabilities = [
        "innovation",
        "orchestration", 
        "facilitation",
        "design_thinking",
        "session_management"
    ]
    
    description = "Orchestrates complete 5-phase innovation sessions following the Innovation Catalyst methodology"
    
    # Phase transition order
    PHASE_ORDER = [
        InnovationPhase.IMMERSION,
        InnovationPhase.DIVERGENCE,
        InnovationPhase.CONVERGENCE,
        InnovationPhase.INCUBATION,
        InnovationPhase.TRANSFORMATION
    ]
    
    def __init__(self, brain=None, model=None, use_llm: bool = True):
        """
        Initialize the MetaConductor.
        
        Args:
            brain: Optional memory/brain instance for context
            model: Optional model adapter for LLM interactions (legacy)
            use_llm: Whether to use LLM for idea generation
        """
        self.brain = brain
        self.model = model
        self.use_llm = use_llm
        self.innovation_swarm = InnovationSwarm(brain=brain, model=model, use_llm=use_llm)
        self.active_sessions: Dict[str, InnovationSessionState] = {}
    
    def run(self, input_data: dict) -> dict:
        """
        Standard agent interface for orchestrator integration.
        
        Args:
            input_data: Dict with keys:
                - "problem" or "goal": The problem statement
                - "phases": List of phases to run (optional, defaults to all)
                - "techniques": List of brainstorming techniques (optional)
                - "constraints": Dict with criteria (optional)
                - "session_id": Existing session ID to resume (optional)
                
        Returns:
            Dict with session state and results
        """
        problem = input_data.get("problem") or input_data.get("goal", "")
        phases = input_data.get("phases", [p.value for p in self.PHASE_ORDER])
        techniques = input_data.get("techniques")
        constraints = input_data.get("constraints", {})
        session_id = input_data.get("session_id")
        
        # Resume existing session or start new
        if session_id and session_id in self.active_sessions:
            result = self.resume_session(session_id, phases)
        else:
            result = self.start_session(problem, phases, techniques, constraints)
        
        return result.dict() if hasattr(result, 'dict') else result
    
    def start_session(
        self,
        problem_statement: str,
        phases: Optional[List[str]] = None,
        techniques: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> InnovationSessionState:
        """
        Start a new innovation session.
        
        Args:
            problem_statement: The challenge to solve
            phases: Which phases to run (defaults to all 5)
            techniques: Brainstorming techniques to use
            constraints: Session constraints and criteria
            
        Returns:
            InnovationSessionState tracking the session
        """
        session_id = str(uuid.uuid4())[:8]
        
        # Parse phases
        if phases:
            phase_list = [InnovationPhase(p) for p in phases if p in [ip.value for ip in self.PHASE_ORDER]]
        else:
            phase_list = list(self.PHASE_ORDER)
        
        # Create session state
        session = InnovationSessionState(
            session_id=session_id,
            problem_statement=problem_statement,
            current_phase=phase_list[0] if phase_list else InnovationPhase.IMMERSION,
            phases_completed=[],
            techniques=[t if isinstance(t, str) else t.value for t in (techniques or [])],
            constraints=constraints or {
                "selection_ratio": 0.2,
                "min_novelty": 0.5,
                "min_feasibility": 0.4,
                "max_ideas": 20,
                "diversity_threshold": 0.7
            },
            status="active"
        )
        
        self.active_sessions[session_id] = session
        
        log_json("INFO", "meta_conductor_session_started", details={
            "session_id": session_id,
            "problem": problem_statement[:50],
            "phases": [p.value for p in phase_list],
            "techniques": techniques
        })
        
        # Store in brain if available
        if self.brain:
            try:
                self.brain.save_innovation_session(session)
            except Exception as e:
                log_json("WARN", "brain_save_session_failed", details={"error": str(e)})
        
        return session
    
    def resume_session(
        self,
        session_id: str,
        phases: Optional[List[str]] = None
    ) -> InnovationSessionState:
        """
        Resume an existing innovation session.
        
        Args:
            session_id: The session to resume
            phases: Additional phases to run (optional)
            
        Returns:
            Updated InnovationSessionState
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        session.status = "active"
        session.updated_at = datetime.utcnow()
        
        log_json("INFO", "meta_conductor_session_resumed", details={
            "session_id": session_id,
            "current_phase": session.current_phase.value,
            "phases_completed": [p.value for p in session.phases_completed]
        })
        
        return session
    
    def execute_phase(
        self,
        session_id: str,
        phase: Optional[InnovationPhase] = None
    ) -> Dict[str, Any]:
        """
        Execute a single innovation phase.
        
        Args:
            session_id: The session ID
            phase: Phase to execute (defaults to current phase)
            
        Returns:
            Phase execution results
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        phase = phase or session.current_phase
        
        log_json("INFO", "meta_conductor_executing_phase", details={
            "session_id": session_id,
            "phase": phase.value
        })
        
        if phase == InnovationPhase.IMMERSION:
            result = self._execute_immersion(session)
        elif phase == InnovationPhase.DIVERGENCE:
            result = self._execute_divergence(session)
        elif phase == InnovationPhase.CONVERGENCE:
            result = self._execute_convergence(session)
        elif phase == InnovationPhase.INCUBATION:
            result = self._execute_incubation(session)
        elif phase == InnovationPhase.TRANSFORMATION:
            result = self._execute_transformation(session)
        else:
            raise ValueError(f"Unknown phase: {phase}")
        
        # Update session state
        session.phases_completed.append(phase)
        session.updated_at = datetime.utcnow()
        
        # Advance to next phase
        current_index = self.PHASE_ORDER.index(phase)
        if current_index < len(self.PHASE_ORDER) - 1:
            session.current_phase = self.PHASE_ORDER[current_index + 1]
        else:
            session.status = "completed"
        
        # Save to brain if available
        if self.brain:
            try:
                self.brain.save_innovation_session(session)
            except Exception as e:
                log_json("WARN", "brain_save_session_failed", details={"error": str(e)})
        
        return result
    
    def _execute_immersion(self, session: InnovationSessionState) -> Dict[str, Any]:
        """
        IMMERSION phase: Deep understanding of the problem.
        
        Analyzes the problem statement and gathers context.
        """
        problem = session.problem_statement
        
        # In a full implementation, this would use LLM to:
        # - Analyze problem components
        # - Identify stakeholders
        # - Define success criteria
        # - Gather context from memory
        
        result = {
            "phase": InnovationPhase.IMMERSION.value,
            "problem_analysis": f"Analyzed: {problem[:50]}...",
            "key_components": ["problem_core", "constraints", "opportunities"],
            "stakeholders": ["users", "developers", "business"],
            "success_criteria": ["feasibility", "impact", "novelty"],
            "ready_for_divergence": True
        }
        
        log_json("INFO", "immersion_complete", details={
            "session_id": session.session_id,
            "components_identified": len(result["key_components"])
        })
        
        return result
    
    def _execute_divergence(self, session: InnovationSessionState) -> Dict[str, Any]:
        """
        DIVERGENCE phase: Generate many ideas using InnovationSwarm.
        """
        output = self.innovation_swarm.brainstorm(
            problem_statement=session.problem_statement,
            techniques=session.techniques or None,
            constraints=session.constraints
        )
        
        session.output = output
        session.ideas_generated = output.total_ideas_generated
        
        result = {
            "phase": InnovationPhase.DIVERGENCE.value,
            "ideas_generated": output.total_ideas_generated,
            "techniques_used": output.techniques_used,
            "diversity_score": output.diversity_score,
            "ready_for_convergence": True
        }
        
        log_json("INFO", "divergence_complete", details={
            "session_id": session.session_id,
            "ideas": output.total_ideas_generated,
            "diversity": output.diversity_score
        })
        
        return result
    
    def _execute_convergence(self, session: InnovationSessionState) -> Dict[str, Any]:
        """
        CONVERGENCE phase: Select best ideas (already done in InnovationSwarm).
        """
        if not session.output:
            raise ValueError("No output from divergence phase")
        
        session.ideas_selected = session.output.total_ideas_selected
        
        result = {
            "phase": InnovationPhase.CONVERGENCE.value,
            "ideas_selected": session.output.total_ideas_selected,
            "novelty_score": session.output.novelty_score,
            "feasibility_score": session.output.feasibility_score,
            "selection_criteria": ["novelty", "feasibility", "impact"],
            "ready_for_incubation": True
        }
        
        log_json("INFO", "convergence_complete", details={
            "session_id": session.session_id,
            "selected": session.output.total_ideas_selected,
            "novelty": session.output.novelty_score
        })
        
        return result
    
    def _execute_incubation(self, session: InnovationSessionState) -> Dict[str, Any]:
        """
        INCUBATION phase: Let ideas develop (simulated).
        
        In a real implementation, this might:
        - Pause for reflection
        - Gather additional feedback
        - Allow subconscious processing
        """
        # Simulate incubation period
        selected_ideas = session.output.selected_ideas if session.output else []
        
        result = {
            "phase": InnovationPhase.INCUBATION.value,
            "ideas_refined": len(selected_ideas),
            "reflection_period": "simulated",
            "insights_emerged": [
                "Connections between ideas identified",
                "Potential implementation paths clarified"
            ],
            "ready_for_transformation": True
        }
        
        log_json("INFO", "incubation_complete", details={
            "session_id": session.session_id,
            "ideas": len(selected_ideas)
        })
        
        return result
    
    def _execute_transformation(self, session: InnovationSessionState) -> Dict[str, Any]:
        """
        TRANSFORMATION phase: Prepare solutions for implementation.
        
        Converts selected ideas into actionable tasks.
        """
        selected_ideas = session.output.selected_ideas if session.output else []
        
        # Create actionable tasks from ideas
        tasks = []
        for i, idea in enumerate(selected_ideas[:5], 1):  # Top 5 ideas
            tasks.append({
                "task_id": f"{session.session_id}_task_{i}",
                "description": idea.description[:100],
                "technique": idea.technique,
                "priority": "high" if idea.novelty > 0.8 else "medium",
                "estimated_effort": "medium",
                "next_steps": ["validate", "prototype", "implement"]
            })
        
        result = {
            "phase": InnovationPhase.TRANSFORMATION.value,
            "actionable_tasks": tasks,
            "task_count": len(tasks),
            "implementation_ready": True,
            "recommendations": [
                "Start with highest feasibility ideas",
                "Build prototypes for validation",
                "Iterate based on feedback"
            ]
        }
        
        session.status = "completed"
        
        log_json("INFO", "transformation_complete", details={
            "session_id": session.session_id,
            "tasks": len(tasks)
        })
        
        if self.brain:
            self.brain.remember(
                f"Innovation session {session.session_id} completed: "
                f"{len(tasks)} actionable tasks from {session.ideas_selected} ideas"
            )
        
        return result
    
    def get_session(self, session_id: str) -> Optional[InnovationSessionState]:
        """Get a session by ID."""
        # Check in-memory first
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Try to load from brain
        if self.brain:
            try:
                data = self.brain.get_innovation_session(session_id)
                if data:
                    # Convert back to InnovationSessionState
                    from agents.schemas import InnovationPhase, InnovationOutput
                    
                    session = InnovationSessionState(
                        session_id=data['session_id'],
                        problem_statement=data['problem_statement'],
                        status=data['status'],
                        current_phase=InnovationPhase(data['current_phase']),
                        phases_completed=[InnovationPhase(p) for p in data['phases_completed']],
                        techniques=data['techniques'],
                        constraints=data['constraints'],
                        ideas_generated=data['ideas_generated'],
                        ideas_selected=data['ideas_selected'],
                        output=InnovationOutput(**data['output_data']) if data.get('output_data') else None,
                    )
                    # Cache in memory
                    self.active_sessions[session_id] = session
                    return session
            except Exception as e:
                log_json("WARN", "brain_load_session_failed", details={"error": str(e)})
        
        return None
    
    def list_sessions(self, status: Optional[str] = None) -> List[InnovationSessionState]:
        """List all sessions, optionally filtered by status."""
        sessions = list(self.active_sessions.values())
        
        # Also load from brain to get persisted sessions
        if self.brain:
            try:
                from agents.schemas import InnovationOutput, InnovationPhase
                brain_sessions = self.brain.list_innovation_sessions(status=status)
                for data in brain_sessions:
                    if data['session_id'] not in self.active_sessions:
                        session = InnovationSessionState(
                            session_id=data['session_id'],
                            problem_statement=data['problem_statement'],
                            status=data['status'],
                            current_phase=InnovationPhase(data['current_phase']),
                            phases_completed=[InnovationPhase(p) for p in data['phases_completed']],
                            techniques=data['techniques'],
                            constraints=data.get('constraints', {}),
                            ideas_generated=data['ideas_generated'],
                            ideas_selected=data['ideas_selected'],
                            output=InnovationOutput(**data['output_data']) if data.get('output_data') else None,
                        )
                        sessions.append(session)
                        self.active_sessions[data['session_id']] = session
            except Exception as e:
                log_json("WARN", "brain_list_sessions_failed", details={"error": str(e)})
        
        if status:
            sessions = [s for s in sessions if s.status == status]
        return sessions
