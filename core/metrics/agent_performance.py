"""Agent performance tracking and scoring system.

Tracks individual agent effectiveness, task completion rates,
and provides performance-based routing recommendations.
"""

from __future__ import annotations

import time
import json
import statistics
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

from core.logging_utils import log_json


@dataclass
class AgentTaskRecord:
    """Record of an agent task execution."""
    timestamp: float
    task_type: str
    success: bool
    duration_seconds: float
    tokens_used: int
    error_message: Optional[str] = None


@dataclass
class AgentPerformance:
    """Performance metrics for an agent."""
    agent_id: str
    agent_name: str
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    total_duration: float = 0.0
    total_tokens: int = 0
    task_history: List[AgentTaskRecord] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks
    
    @property
    def avg_duration(self) -> float:
        """Calculate average task duration."""
        if self.total_tasks == 0:
            return 0.0
        return self.total_duration / self.total_tasks
    
    @property
    def avg_tokens(self) -> float:
        """Calculate average token usage."""
        if self.total_tasks == 0:
            return 0.0
        return self.total_tokens / self.total_tasks
    
    @property
    def performance_score(self) -> float:
        """Calculate overall performance score (0-100)."""
        if self.total_tasks < 5:
            return 50.0  # Neutral score for new agents
        
        # Weighted scoring
        success_weight = 0.5
        speed_weight = 0.3
        efficiency_weight = 0.2
        
        # Success score (0-100)
        success_score = self.success_rate * 100
        
        # Speed score (faster is better, but with diminishing returns)
        # Optimal is around 30 seconds
        speed_score = max(0, 100 - (self.avg_duration / 3))
        
        # Efficiency score (lower token usage is better)
        # Optimal is around 1000 tokens
        efficiency_score = max(0, 100 - (self.avg_tokens / 50))
        
        return (
            success_score * success_weight +
            speed_score * speed_weight +
            efficiency_score * efficiency_weight
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": round(self.success_rate, 4),
            "avg_duration_seconds": round(self.avg_duration, 2),
            "avg_tokens": round(self.avg_tokens, 2),
            "performance_score": round(self.performance_score, 2),
            "recent_tasks": [
                {
                    "timestamp": t.timestamp,
                    "task_type": t.task_type,
                    "success": t.success,
                    "duration": round(t.duration_seconds, 2),
                }
                for t in self.task_history[-10:]  # Last 10 tasks
            ],
        }


class AgentPerformanceTracker:
    """Tracks and analyzes agent performance.
    
    Maintains performance statistics for all agents and provides
    insights for agent selection and routing.
    """
    
    DATA_FILE = Path("memory/agent_performance.json")
    
    def __init__(self):
        self._agents: Dict[str, AgentPerformance] = {}
        self._load_data()
    
    def _load_data(self) -> None:
        """Load persisted performance data."""
        if self.DATA_FILE.exists():
            try:
                with open(self.DATA_FILE, "r") as f:
                    data = json.load(f)
                
                for agent_id, agent_data in data.get("agents", {}).items():
                    perf = AgentPerformance(
                        agent_id=agent_data["agent_id"],
                        agent_name=agent_data["agent_name"],
                        total_tasks=agent_data.get("total_tasks", 0),
                        successful_tasks=agent_data.get("successful_tasks", 0),
                        failed_tasks=agent_data.get("failed_tasks", 0),
                        total_duration=agent_data.get("total_duration", 0.0),
                        total_tokens=agent_data.get("total_tokens", 0),
                    )
                    self._agents[agent_id] = perf
                
                log_json("INFO", "agent_performance_loaded", {
                    "agent_count": len(self._agents),
                })
            except Exception as e:
                log_json("ERROR", "agent_performance_load_failed", {"error": str(e)})
    
    def _save_data(self) -> None:
        """Persist performance data."""
        try:
            self.DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "agents": {
                    agent_id: {
                        "agent_id": perf.agent_id,
                        "agent_name": perf.agent_name,
                        "total_tasks": perf.total_tasks,
                        "successful_tasks": perf.successful_tasks,
                        "failed_tasks": perf.failed_tasks,
                        "total_duration": perf.total_duration,
                        "total_tokens": perf.total_tokens,
                    }
                    for agent_id, perf in self._agents.items()
                },
                "saved_at": time.time(),
            }
            
            with open(self.DATA_FILE, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            log_json("ERROR", "agent_performance_save_failed", {"error": str(e)})
    
    def record_task(
        self,
        agent_id: str,
        agent_name: str,
        task_type: str,
        success: bool,
        duration_seconds: float,
        tokens_used: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        """Record a task execution for an agent.
        
        Args:
            agent_id: Unique agent identifier
            agent_name: Human-readable agent name
            task_type: Type of task performed
            success: Whether task succeeded
            duration_seconds: Task execution time
            tokens_used: Token consumption
            error_message: Error details if failed
        """
        # Get or create agent performance record
        if agent_id not in self._agents:
            self._agents[agent_id] = AgentPerformance(
                agent_id=agent_id,
                agent_name=agent_name,
            )
        
        perf = self._agents[agent_id]
        
        # Update stats
        perf.total_tasks += 1
        if success:
            perf.successful_tasks += 1
        else:
            perf.failed_tasks += 1
        
        perf.total_duration += duration_seconds
        perf.total_tokens += tokens_used
        
        # Add to history
        record = AgentTaskRecord(
            timestamp=time.time(),
            task_type=task_type,
            success=success,
            duration_seconds=duration_seconds,
            tokens_used=tokens_used,
            error_message=error_message,
        )
        perf.task_history.append(record)
        
        # Keep history manageable
        if len(perf.task_history) > 100:
            perf.task_history = perf.task_history[-100:]
        
        # Save periodically
        if perf.total_tasks % 10 == 0:
            self._save_data()
        
        log_json("INFO", "agent_task_recorded", {
            "agent_id": agent_id,
            "task_type": task_type,
            "success": success,
            "duration": round(duration_seconds, 2),
        })
    
    def get_performance(self, agent_id: str) -> Optional[AgentPerformance]:
        """Get performance data for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            AgentPerformance or None
        """
        return self._agents.get(agent_id)
    
    def get_all_performances(self) -> List[AgentPerformance]:
        """Get performance data for all agents.
        
        Returns:
            List of AgentPerformance, sorted by score
        """
        return sorted(
            self._agents.values(),
            key=lambda x: x.performance_score,
            reverse=True,
        )
    
    def get_leaderboard(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top performing agents.
        
        Args:
            limit: Number of top agents to return
            
        Returns:
            List of agent performance dicts
        """
        performances = self.get_all_performances()
        return [p.to_dict() for p in performances[:limit]]
    
    def get_recommendations(self) -> List[Dict[str, Any]]:
        """Get agent usage recommendations.
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        performances = self.get_all_performances()
        
        if not performances:
            return recommendations
        
        # Find underperforming agents
        underperforming = [p for p in performances if p.performance_score < 50]
        if underperforming:
            recommendations.append({
                "type": "warning",
                "title": "Underperforming Agents",
                "message": f"{len(underperforming)} agents have performance scores below 50",
                "agents": [p.agent_name for p in underperforming],
                "action": "Review agent configurations or consider replacement",
            })
        
        # Find top performers
        top_performers = [p for p in performances if p.performance_score > 80]
        if top_performers:
            recommendations.append({
                "type": "success",
                "title": "Top Performing Agents",
                "message": f"{len(top_performers)} agents are performing excellently",
                "agents": [p.agent_name for p in top_performers],
                "action": "Consider assigning more tasks to these agents",
            })
        
        # Check for agents with low success rates
        low_success = [p for p in performances 
                      if p.total_tasks > 5 and p.success_rate < 0.7]
        if low_success:
            recommendations.append({
                "type": "warning",
                "title": "Low Success Rates",
                "message": f"{len(low_success)} agents have success rates below 70%",
                "agents": [p.agent_name for p in low_success],
                "action": "Review error patterns and improve error handling",
            })
        
        return recommendations
    
    def get_agent_comparison(self, agent_ids: List[str]) -> Dict[str, Any]:
        """Compare performance of specific agents.
        
        Args:
            agent_ids: List of agent IDs to compare
            
        Returns:
            Comparison data
        """
        agents = [self._agents.get(aid) for aid in agent_ids]
        agents = [a for a in agents if a is not None]
        
        if not agents:
            return {"error": "No valid agents found"}
        
        return {
            "agents": [a.to_dict() for a in agents],
            "comparison": {
                "best_success_rate": max(a.success_rate for a in agents),
                "fastest_avg_duration": min(a.avg_duration for a in agents),
                "highest_score": max(a.performance_score for a in agents),
            },
        }
    
    def reset_agent_stats(self, agent_id: str) -> bool:
        """Reset statistics for an agent.
        
        Args:
            agent_id: Agent to reset
            
        Returns:
            True if reset successful
        """
        if agent_id in self._agents:
            agent = self._agents[agent_id]
            agent.total_tasks = 0
            agent.successful_tasks = 0
            agent.failed_tasks = 0
            agent.total_duration = 0.0
            agent.total_tokens = 0
            agent.task_history.clear()
            self._save_data()
            return True
        return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get overall performance summary.
        
        Returns:
            Summary statistics
        """
        performances = self.get_all_performances()
        
        if not performances:
            return {
                "total_agents": 0,
                "total_tasks": 0,
                "overall_success_rate": 0.0,
            }
        
        total_tasks = sum(p.total_tasks for p in performances)
        total_successful = sum(p.successful_tasks for p in performances)
        
        return {
            "total_agents": len(performances),
            "total_tasks": total_tasks,
            "total_successful": total_successful,
            "total_failed": sum(p.failed_tasks for p in performances),
            "overall_success_rate": round(total_successful / total_tasks, 4) if total_tasks > 0 else 0,
            "avg_performance_score": round(statistics.mean(p.performance_score for p in performances), 2),
            "top_agent": performances[0].agent_name if performances else None,
        }


# Global tracker instance
tracker: Optional[AgentPerformanceTracker] = None


def get_agent_tracker() -> AgentPerformanceTracker:
    """Get global agent performance tracker."""
    global tracker
    if tracker is None:
        tracker = AgentPerformanceTracker()
    return tracker
