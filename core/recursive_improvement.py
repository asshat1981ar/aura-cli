import logging
import json
import time
from typing import List, Dict, Optional
from core.fitness import FitnessFunction

logger = logging.getLogger(__name__)

class RecursiveImprovementService:
    """
    Runtime service that evaluates improvement opportunities from cycle history.
    Emits structured proposals for operator review.
    """
    def __init__(self, fitness_weights: Optional[Dict[str, float]] = None):
        self.fitness = FitnessFunction(weights=fitness_weights)
        self._recent_cycles: List[Dict] = []

    def normalize_cycle_entry(self, cycle_entry: Dict) -> Dict:
        """Normalize a cycle entry into the fields used for proposal scoring."""
        phase_outputs = cycle_entry.get("phase_outputs", {}) if isinstance(cycle_entry, dict) else {}
        verification = phase_outputs.get("verification", {}) if isinstance(phase_outputs, dict) else {}
        verification_status = cycle_entry.get("verification_status")
        if verification_status is None and isinstance(verification, dict):
            verification_status = verification.get("status")

        retry_count = cycle_entry.get("retries")
        if retry_count is None and isinstance(phase_outputs, dict):
            retry_count = phase_outputs.get("retry_count", 0)

        return {
            "cycle_id": cycle_entry.get("cycle_id", "unknown"),
            "goal": cycle_entry.get("goal", ""),
            "verification_status": verification_status or "unknown",
            "retries": int(retry_count or 0),
        }

    def observe_cycle(self, cycle_entry: Dict) -> List[Dict]:
        """Append a normalized cycle entry and return recent history."""
        normalized = self.normalize_cycle_entry(cycle_entry)
        self._recent_cycles.append(normalized)
        if len(self._recent_cycles) > 25:
            self._recent_cycles = self._recent_cycles[-25:]
        return list(self._recent_cycles)

    def evaluate_candidates(self, cycle_history: List[Dict]) -> List[Dict]:
        """
        Analyze recent cycle outcomes to identify improvement opportunities.
        """
        proposals = []
        if not cycle_history:
            return proposals

        # Simple analysis: look for high retry rates or frequent failures
        failed_cycles = [c for c in cycle_history if c.get("verification_status") == "fail"]
        
        if len(failed_cycles) > 2:
            proposal = self.create_proposal(
                proposal_id=f"ri_{int(time.time())}_001",
                summary="High failure rate detected in recent cycles.",
                source_cycles=[c.get("cycle_id", "unknown") for c in failed_cycles],
                metrics={
                    "success_rate": 1.0 - (len(failed_cycles) / len(cycle_history)),
                    "retry_rate": sum(c.get("retries", 0) for c in cycle_history) / len(cycle_history),
                    "complexity_delta": 0.05
                },
                hypotheses=["System instability or capability gap in specific goals"],
                actions=["Review failure logs", "Decompose complex goals", "Add targeted tests"],
                risk="low"
            )
            proposals.append(proposal)

        return proposals

    def create_proposal(self, proposal_id: str, summary: str, source_cycles: List[str], 
                        metrics: Dict, hypotheses: List[str], 
                        actions: List[str], risk: str = "medium") -> Dict:
        """
        Creates a structured proposal following the canonical data contract.
        """
        fitness_score = self.fitness.calculate(metrics)
        
        proposal = {
            "proposal_id": proposal_id,
            "source_cycles": source_cycles,
            "summary": summary,
            "fitness_snapshot": {
                "score": fitness_score,
                "success_rate": metrics.get("success_rate", 0.0),
                "retry_rate": metrics.get("retry_rate", 0.0),
                "complexity_delta": metrics.get("complexity_delta", 0.0),
            },
            "hypotheses": hypotheses,
            "recommended_actions": actions,
            "risk_level": risk,
            "requires_operator_review": True,
        }
        return proposal

    def log_proposal(self, proposal: Dict):
        logger.info(f"Generated Recursive Improvement Proposal: {proposal['proposal_id']}")
        logger.info(json.dumps(proposal, indent=2))
