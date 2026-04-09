"""Metrics collection system for AURA.

Collects system resource usage, goal execution metrics, and application performance data.
"""

from __future__ import annotations

import os
import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from threading import Lock

from core.logging_utils import log_json


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_percent: float
    load_avg: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat(),
            "cpu_percent": round(self.cpu_percent, 2),
            "memory_percent": round(self.memory_percent, 2),
            "memory_mb": round(self.memory_mb, 2),
            "disk_percent": round(self.disk_percent, 2),
            "load_avg": [round(x, 2) for x in self.load_avg],
        }


@dataclass
class GoalMetrics:
    """Goal execution metrics."""
    timestamp: float
    goal_id: str
    description: str
    status: str  # completed, failed, cancelled
    duration_seconds: float
    cycles_used: int
    agent_count: int
    tokens_used: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat(),
            "goal_id": self.goal_id,
            "description": self.description[:100] + "..." if len(self.description) > 100 else self.description,
            "status": self.status,
            "duration_seconds": round(self.duration_seconds, 2),
            "cycles_used": self.cycles_used,
            "agent_count": self.agent_count,
            "tokens_used": self.tokens_used,
        }


class MetricsCollector:
    """Collects and stores metrics for AURA.
    
    Maintains a rotating buffer of recent metrics and persists
    aggregated data to disk.
    """
    
    DEFAULT_BUFFER_SIZE = 1000
    METRICS_FILE = Path("memory/metrics_history.json")
    
    def __init__(self, buffer_size: int = DEFAULT_BUFFER_SIZE):
        self.buffer_size = buffer_size
        self._system_metrics: List[SystemMetrics] = []
        self._goal_metrics: List[GoalMetrics] = []
        self._lock = Lock()
        self._last_collection = 0
        
        # Load persisted metrics
        self._load_metrics()
    
    def _load_metrics(self) -> None:
        """Load persisted metrics from disk."""
        if self.METRICS_FILE.exists():
            try:
                with open(self.METRICS_FILE, "r") as f:
                    data = json.load(f)
                
                # Restore system metrics (last 100)
                for m in data.get("system_metrics", [])[-100:]:
                    self._system_metrics.append(SystemMetrics(**m))
                
                # Restore goal metrics (last 500)
                for m in data.get("goal_metrics", [])[-500:]:
                    self._goal_metrics.append(GoalMetrics(**m))
                
                log_json("INFO", "metrics_loaded", {
                    "system_count": len(self._system_metrics),
                    "goal_count": len(self._goal_metrics),
                })
            except Exception as e:
                log_json("ERROR", "metrics_load_failed", {"error": str(e)})
    
    def _save_metrics(self) -> None:
        """Persist metrics to disk."""
        try:
            self.METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "system_metrics": [asdict(m) for m in self._system_metrics[-100:]],
                "goal_metrics": [asdict(m) for m in self._goal_metrics[-500:]],
                "saved_at": datetime.now().isoformat(),
            }
            
            with open(self.METRICS_FILE, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            log_json("ERROR", "metrics_save_failed", {"error": str(e)})
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics.
        
        Returns:
            SystemMetrics with current values
        """
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_mb = memory.used / (1024 * 1024)
            
            # Disk usage
            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100
            
            # Load average
            load_avg = list(os.getloadavg()) if hasattr(os, "getloadavg") else [0.0, 0.0, 0.0]
            
            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_mb=memory_mb,
                disk_percent=disk_percent,
                load_avg=load_avg,
            )
            
            with self._lock:
                self._system_metrics.append(metrics)
                # Keep only recent metrics
                if len(self._system_metrics) > self.buffer_size:
                    self._system_metrics = self._system_metrics[-self.buffer_size:]
                
                # Save periodically (every 10 collections)
                if len(self._system_metrics) % 10 == 0:
                    self._save_metrics()
            
            return metrics
            
        except ImportError:
            # Fallback if psutil not available
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_mb=0.0,
                disk_percent=0.0,
                load_avg=[0.0, 0.0, 0.0],
            )
    
    def record_goal_completion(
        self,
        goal_id: str,
        description: str,
        status: str,
        duration_seconds: float,
        cycles_used: int,
        agent_count: int = 0,
        tokens_used: int = 0,
    ) -> None:
        """Record goal completion metrics.
        
        Args:
            goal_id: Unique goal identifier
            description: Goal description
            status: Final status (completed, failed, cancelled)
            duration_seconds: Total execution time
            cycles_used: Number of cycles consumed
            agent_count: Number of agents involved
            tokens_used: Token usage estimate
        """
        metrics = GoalMetrics(
            timestamp=time.time(),
            goal_id=goal_id,
            description=description,
            status=status,
            duration_seconds=duration_seconds,
            cycles_used=cycles_used,
            agent_count=agent_count,
            tokens_used=tokens_used,
        )
        
        with self._lock:
            self._goal_metrics.append(metrics)
            
            # Keep only recent metrics
            if len(self._goal_metrics) > self.buffer_size:
                self._goal_metrics = self._goal_metrics[-self.buffer_size:]
            
            # Save periodically
            if len(self._goal_metrics) % 10 == 0:
                self._save_metrics()
        
        log_json("INFO", "goal_metrics_recorded", metrics.to_dict())
    
    def get_system_metrics(self, limit: int = 100) -> List[SystemMetrics]:
        """Get recent system metrics.
        
        Args:
            limit: Maximum number of records
            
        Returns:
            List of SystemMetrics
        """
        with self._lock:
            return self._system_metrics[-limit:]
    
    def get_goal_metrics(self, limit: int = 100) -> List[GoalMetrics]:
        """Get recent goal metrics.
        
        Args:
            limit: Maximum number of records
            
        Returns:
            List of GoalMetrics
        """
        with self._lock:
            return self._goal_metrics[-limit:]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary.
        
        Returns:
            Summary statistics
        """
        with self._lock:
            # System metrics summary
            if self._system_metrics:
                recent = self._system_metrics[-10:]  # Last 10 readings
                avg_cpu = sum(m.cpu_percent for m in recent) / len(recent)
                avg_memory = sum(m.memory_percent for m in recent) / len(recent)
            else:
                avg_cpu = 0.0
                avg_memory = 0.0
            
            # Goal metrics summary
            total_goals = len(self._goal_metrics)
            completed = sum(1 for m in self._goal_metrics if m.status == "completed")
            failed = sum(1 for m in self._goal_metrics if m.status == "failed")
            
            if total_goals > 0:
                success_rate = completed / total_goals
                avg_duration = sum(m.duration_seconds for m in self._goal_metrics) / total_goals
            else:
                success_rate = 0.0
                avg_duration = 0.0
            
            return {
                "system": {
                    "avg_cpu_percent": round(avg_cpu, 2),
                    "avg_memory_percent": round(avg_memory, 2),
                    "collection_count": len(self._system_metrics),
                },
                "goals": {
                    "total": total_goals,
                    "completed": completed,
                    "failed": failed,
                    "success_rate": round(success_rate, 4),
                    "avg_duration_seconds": round(avg_duration, 2),
                },
                "last_updated": datetime.now().isoformat(),
            }
    
    def get_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get trends over time.
        
        Args:
            hours: Time window in hours
            
        Returns:
            Trend data
        """
        cutoff = time.time() - (hours * 3600)
        
        with self._lock:
            # Filter recent goals
            recent_goals = [m for m in self._goal_metrics if m.timestamp > cutoff]
            
            # Group by hour
            hourly: Dict[str, Dict[str, Any]] = {}
            for goal in recent_goals:
                hour_key = datetime.fromtimestamp(goal.timestamp).strftime("%Y-%m-%d %H:00")
                
                if hour_key not in hourly:
                    hourly[hour_key] = {
                        "total": 0,
                        "completed": 0,
                        "failed": 0,
                    }
                
                hourly[hour_key]["total"] += 1
                if goal.status == "completed":
                    hourly[hour_key]["completed"] += 1
                elif goal.status == "failed":
                    hourly[hour_key]["failed"] += 1
            
            return {
                "period_hours": hours,
                "total_goals": len(recent_goals),
                "hourly_breakdown": hourly,
                "trend_direction": "increasing" if len(recent_goals) > 10 else "stable",
            }


# Global collector instance
_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector."""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector
