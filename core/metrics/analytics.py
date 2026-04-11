"""Analytics engine for AURA metrics.

Provides trend analysis, predictions, and insights from collected metrics.
"""

from __future__ import annotations

import time
import statistics
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

from .collector import MetricsCollector


@dataclass
class TrendAnalysis:
    """Result of trend analysis."""

    metric_name: str
    direction: str  # increasing, decreasing, stable
    change_percent: float
    confidence: float  # 0.0 to 1.0
    forecast: List[float]
    recommendation: str


class AnalyticsEngine:
    """Analyzes metrics and generates insights.

    Provides trend analysis, anomaly detection, and recommendations
    based on collected metrics data.
    """

    def __init__(self, collector: Optional[MetricsCollector] = None):
        self.collector = collector or MetricsCollector()

    def analyze_goal_trends(self, days: int = 7) -> Dict[str, Any]:
        """Analyze goal completion trends.

        Args:
            days: Number of days to analyze

        Returns:
            Trend analysis results
        """
        cutoff = time.time() - (days * 86400)
        goals = [g for g in self.collector.get_goal_metrics(1000) if g.timestamp > cutoff]

        if not goals:
            return {
                "period_days": days,
                "total_goals": 0,
                "trend": "no_data",
                "message": "Not enough data for trend analysis",
            }

        # Group by day
        daily_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"total": 0, "completed": 0, "failed": 0, "duration_sum": 0})

        for goal in goals:
            day = datetime.fromtimestamp(goal.timestamp).strftime("%Y-%m-%d")
            daily_stats[day]["total"] += 1
            daily_stats[day]["completed"] += 1 if goal.status == "completed" else 0
            daily_stats[day]["failed"] += 1 if goal.status == "failed" else 0
            daily_stats[day]["duration_sum"] += goal.duration_seconds

        # Calculate success rates by day
        dates = sorted(daily_stats.keys())
        success_rates = []

        for date in dates:
            stats = daily_stats[date]
            rate = stats["completed"] / stats["total"] if stats["total"] > 0 else 0
            success_rates.append(rate)

        # Determine trend
        if len(success_rates) >= 3:
            recent_avg = statistics.mean(success_rates[-3:])
            older_avg = statistics.mean(success_rates[:3]) if len(success_rates) >= 6 else recent_avg

            change = recent_avg - older_avg

            if change > 0.1:
                trend = "improving"
                message = "Success rate is trending upward"
            elif change < -0.1:
                trend = "declining"
                message = "Success rate is trending downward"
            else:
                trend = "stable"
                message = "Success rate is stable"
        else:
            trend = "insufficient_data"
            message = "Need more data for trend analysis"

        # Calculate overall stats
        total = len(goals)
        completed = sum(1 for g in goals if g.status == "completed")
        failed = sum(1 for g in goals if g.status == "failed")

        avg_duration = statistics.mean([g.duration_seconds for g in goals]) if goals else 0

        return {
            "period_days": days,
            "total_goals": total,
            "completed": completed,
            "failed": failed,
            "success_rate": round(completed / total, 4) if total > 0 else 0,
            "avg_duration_seconds": round(avg_duration, 2),
            "trend": trend,
            "message": message,
            "daily_breakdown": [
                {
                    "date": date,
                    "total": daily_stats[date]["total"],
                    "completed": daily_stats[date]["completed"],
                    "success_rate": round(daily_stats[date]["completed"] / daily_stats[date]["total"], 4) if daily_stats[date]["total"] > 0 else 0,
                }
                for date in dates[-7:]  # Last 7 days
            ],
        }

    def analyze_system_performance(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze system performance trends.

        Args:
            hours: Number of hours to analyze

        Returns:
            Performance analysis
        """
        cutoff = time.time() - (hours * 3600)
        metrics = [m for m in self.collector.get_system_metrics(1000) if m.timestamp > cutoff]

        if not metrics:
            return {
                "period_hours": hours,
                "status": "no_data",
                "message": "No system metrics available",
            }

        # Calculate statistics
        cpu_values = [m.cpu_percent for m in metrics]
        memory_values = [m.memory_percent for m in metrics]

        cpu_stats = {
            "avg": round(statistics.mean(cpu_values), 2),
            "max": round(max(cpu_values), 2),
            "min": round(min(cpu_values), 2),
        }

        memory_stats = {
            "avg": round(statistics.mean(memory_values), 2),
            "max": round(max(memory_values), 2),
            "min": round(min(memory_values), 2),
        }

        # Detect anomalies (values > 2 std dev from mean)
        if len(cpu_values) > 10:
            cpu_std = statistics.stdev(cpu_values)
            cpu_mean = statistics.mean(cpu_values)
            anomalies = [v for v in cpu_values if abs(v - cpu_mean) > 2 * cpu_std]
            anomaly_count = len(anomalies)
        else:
            anomaly_count = 0

        # Health assessment
        health_score = 100
        if cpu_stats["avg"] > 80:
            health_score -= 20
        if memory_stats["avg"] > 80:
            health_score -= 20
        if anomaly_count > 5:
            health_score -= 10

        return {
            "period_hours": hours,
            "status": "healthy" if health_score >= 80 else "warning" if health_score >= 60 else "critical",
            "health_score": health_score,
            "cpu": cpu_stats,
            "memory": memory_stats,
            "anomaly_count": anomaly_count,
            "recommendations": self._generate_recommendations(cpu_stats, memory_stats),
        }

    def _generate_recommendations(
        self,
        cpu_stats: Dict[str, float],
        memory_stats: Dict[str, float],
    ) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []

        if cpu_stats["avg"] > 70:
            recommendations.append("Consider reducing concurrent goal execution to lower CPU usage")

        if memory_stats["avg"] > 75:
            recommendations.append("Memory usage is high - review agent memory consumption")

        if cpu_stats["max"] > 90:
            recommendations.append("CPU spikes detected - check for resource-intensive operations")

        if not recommendations:
            recommendations.append("System performance is optimal")

        return recommendations

    def predict_goal_completion(
        self,
        goal_complexity: str = "medium",
    ) -> Dict[str, Any]:
        """Predict goal completion time based on historical data.

        Args:
            goal_complexity: low, medium, high

        Returns:
            Prediction results
        """
        # Get historical data
        goals = self.collector.get_goal_metrics(100)
        completed = [g for g in goals if g.status == "completed"]

        if not completed:
            return {
                "predicted_duration_seconds": 300,  # Default 5 minutes
                "confidence": 0.0,
                "message": "No historical data for prediction",
            }

        durations = [g.duration_seconds for g in completed]

        # Complexity multipliers
        multipliers = {
            "low": 0.7,
            "medium": 1.0,
            "high": 1.5,
        }
        multiplier = multipliers.get(goal_complexity, 1.0)

        # Calculate prediction
        median_duration = statistics.median(durations)
        predicted = median_duration * multiplier

        # Confidence based on data quantity
        confidence = min(len(completed) / 20, 1.0)  # Max confidence at 20 samples

        return {
            "predicted_duration_seconds": round(predicted, 2),
            "predicted_duration_formatted": self._format_duration(predicted),
            "confidence": round(confidence, 2),
            "based_on_samples": len(completed),
            "message": f"Based on {len(completed)} completed goals",
        }

    def generate_daily_report(self) -> Dict[str, Any]:
        """Generate a daily analytics report.

        Returns:
            Complete daily report
        """
        now = datetime.now()
        yesterday = now - timedelta(days=1)

        # Get yesterday's goals
        cutoff = yesterday.timestamp()
        recent_goals = [g for g in self.collector.get_goal_metrics(1000) if g.timestamp > cutoff]

        report = {
            "date": yesterday.strftime("%Y-%m-%d"),
            "generated_at": now.isoformat(),
            "summary": {
                "total_goals": len(recent_goals),
                "completed": sum(1 for g in recent_goals if g.status == "completed"),
                "failed": sum(1 for g in recent_goals if g.status == "failed"),
            },
            "goal_trends": self.analyze_goal_trends(days=7),
            "system_performance": self.analyze_system_performance(hours=24),
            "predictions": {
                "next_goal": self.predict_goal_completion("medium"),
            },
        }

        # Add success rate
        total = report["summary"]["total_goals"]
        completed = report["summary"]["completed"]
        report["summary"]["success_rate"] = round(completed / total, 4) if total > 0 else 0

        return report

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds / 60)}m {int(seconds % 60)}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"

    def get_insights(self) -> List[Dict[str, Any]]:
        """Generate actionable insights.

        Returns:
            List of insights
        """
        insights = []

        # Goal insights
        goal_trends = self.analyze_goal_trends(days=7)
        if goal_trends["trend"] == "declining":
            insights.append(
                {
                    "type": "warning",
                    "category": "goals",
                    "title": "Success Rate Declining",
                    "message": goal_trends["message"],
                    "recommendation": "Review recent failed goals for common patterns",
                }
            )
        elif goal_trends["trend"] == "improving":
            insights.append(
                {
                    "type": "success",
                    "category": "goals",
                    "title": "Success Rate Improving",
                    "message": goal_trends["message"],
                    "recommendation": "Continue current practices",
                }
            )

        # System insights
        perf = self.analyze_system_performance(hours=24)
        if perf["status"] == "critical":
            insights.append(
                {
                    "type": "critical",
                    "category": "system",
                    "title": "System Performance Critical",
                    "message": f"Health score: {perf['health_score']}/100",
                    "recommendation": "Immediate attention required - review resource usage",
                }
            )
        elif perf["status"] == "warning":
            insights.append(
                {
                    "type": "warning",
                    "category": "system",
                    "title": "System Performance Warning",
                    "message": f"Health score: {perf['health_score']}/100",
                    "recommendation": perf["recommendations"][0] if perf["recommendations"] else "Monitor system resources",
                }
            )

        # Efficiency insights
        goals = self.collector.get_goal_metrics(100)
        if goals:
            avg_duration = statistics.mean([g.duration_seconds for g in goals])
            if avg_duration > 600:  # More than 10 minutes
                insights.append(
                    {
                        "type": "info",
                        "category": "efficiency",
                        "title": "Long Average Goal Duration",
                        "message": f"Average goal takes {self._format_duration(avg_duration)}",
                        "recommendation": "Consider breaking complex goals into smaller tasks",
                    }
                )

        return insights


# Global engine instance
_engine: Optional[AnalyticsEngine] = None


def get_analytics_engine() -> AnalyticsEngine:
    """Get global analytics engine."""
    global _engine
    if _engine is None:
        _engine = AnalyticsEngine()
    return _engine
