"""Outcome analyzer for extracting insights from simulation results."""

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.logging_utils import log_json
from core.simulation.scenario import ScenarioOutcome


@dataclass
class Insight:
    """An actionable insight extracted from simulation outcomes."""
    insight_type: str  # "statistical", "sensitivity", "outlier", "trend"
    title: str
    description: str
    confidence: float  # 0-1
    related_metrics: List[str] = field(default_factory=list)
    recommended_action: Optional[str] = None
    severity: str = "info"  # "info", "warning", "critical"


@dataclass
class ParameterSensitivity:
    """Sensitivity of outcomes to a specific parameter."""
    parameter_name: str
    correlation: float  # -1 to 1
    impact_score: float  # 0-1, how much this parameter affects outcomes
    optimal_range: Optional[Tuple[float, float]] = None
    recommendation: Optional[str] = None


class OutcomeAnalyzer:
    """Analyzes simulation outcomes to extract insights."""
    
    def __init__(self):
        self.min_samples_for_stats = 3
        self.outlier_threshold = 2.0  # Standard deviations
        
    def analyze(
        self, 
        outcomes: List[ScenarioOutcome], 
        baseline: Optional[Dict[str, float]] = None
    ) -> List[Insight]:
        """
        Extract insights from simulation results.
        
        Args:
            outcomes: List of scenario outcomes to analyze
            baseline: Optional baseline metrics for comparison
            
        Returns:
            List of actionable insights
        """
        insights = []
        
        if not outcomes:
            return [Insight(
                insight_type="statistical",
                title="No Outcomes",
                description="No simulation outcomes to analyze",
                confidence=1.0,
                severity="warning"
            )]
        
        # Statistical comparison
        insights.extend(self._statistical_comparison(outcomes, baseline))
        
        # Parameter sensitivity analysis
        insights.extend(self._sensitivity_analysis(outcomes))
        
        # Outlier detection
        insights.extend(self._detect_outliers(outcomes))
        
        # Trend identification
        insights.extend(self._identify_trends(outcomes))
        
        # Winner analysis
        winner_insight = self._analyze_winner(outcomes)
        if winner_insight:
            insights.append(winner_insight)
        
        log_json("INFO", "outcome_analysis_complete", {
            "outcomes_analyzed": len(outcomes),
            "insights_generated": len(insights),
            "insights_by_type": self._count_by_type(insights)
        })
        
        return insights
    
    def _statistical_comparison(
        self, 
        outcomes: List[ScenarioOutcome],
        baseline: Optional[Dict[str, float]]
    ) -> List[Insight]:
        """Generate statistical comparison insights."""
        insights = []
        
        if len(outcomes) < self.min_samples_for_stats:
            return insights
        
        # Collect all metrics
        all_metrics = set()
        for outcome in outcomes:
            all_metrics.update(outcome.metrics.keys())
        
        for metric in all_metrics:
            values = [o.metrics.get(metric, 0.0) for o in outcomes if metric in o.metrics]
            
            if len(values) < self.min_samples_for_stats:
                continue
            
            # Calculate statistics
            mean_val = statistics.mean(values)
            try:
                stdev_val = statistics.stdev(values) if len(values) > 1 else 0.0
            except statistics.StatisticsError:
                stdev_val = 0.0
            
            min_val = min(values)
            max_val = max(values)
            range_val = max_val - min_val
            
            # Compare with baseline if available
            if baseline and metric in baseline:
                baseline_val = baseline[metric]
                change = mean_val - baseline_val
                change_pct = (change / baseline_val * 100) if baseline_val != 0 else 0
                
                if abs(change_pct) > 10:  # Significant change threshold
                    direction = "improved" if change > 0 else "degraded"
                    insights.append(Insight(
                        insight_type="statistical",
                        title=f"{metric.replace('_', ' ').title()} {direction.title()}",
                        description=(
                            f"{metric} {direction} by {abs(change):.3f} "
                            f"({abs(change_pct):.1f}%) compared to baseline. "
                            f"Current mean: {mean_val:.3f}, Baseline: {baseline_val:.3f}"
                        ),
                        confidence=min(0.9, 0.5 + abs(change_pct) / 100),
                        related_metrics=[metric],
                        severity="info" if change > 0 else "warning"
                    ))
            
            # Report high variability
            if stdev_val > mean_val * 0.2 and mean_val > 0:  # CV > 20%
                insights.append(Insight(
                    insight_type="statistical",
                    title=f"High Variability in {metric.replace('_', ' ').title()}",
                    description=(
                        f"{metric} shows high variability (CV: {stdev_val/mean_val:.1%}). "
                        f"Range: {min_val:.3f} - {max_val:.3f}, "
                        f"Mean: {mean_val:.3f} ± {stdev_val:.3f}"
                    ),
                    confidence=0.8,
                    related_metrics=[metric],
                    recommended_action="Consider controlling for other variables or investigating causes of variability",
                    severity="warning"
                ))
        
        # Overall success rate
        success_rate = sum(1 for o in outcomes if o.success) / len(outcomes)
        if success_rate < 0.5:
            insights.append(Insight(
                insight_type="statistical",
                title="Low Success Rate",
                description=f"Only {success_rate:.1%} of scenarios succeeded. Review configuration or approach.",
                confidence=0.9,
                severity="critical",
                recommended_action="Investigate common failure patterns and adjust strategy"
            ))
        
        return insights
    
    def _sensitivity_analysis(self, outcomes: List[ScenarioOutcome]) -> List[Insight]:
        """Identify which parameters most affect outcomes."""
        insights = []
        
        if len(outcomes) < 4:
            return insights
        
        # Group outcomes by parameter values and analyze impact
        # This is a simplified version - in production, would use proper sensitivity analysis
        
        score_variance = statistics.variance([o.score for o in outcomes]) if len(outcomes) > 1 else 0
        
        if score_variance > 0.05:  # Significant variance in scores
            best = max(outcomes, key=lambda o: o.score)
            worst = min(outcomes, key=lambda o: o.score)
            
            # Identify differing parameters
            diff_params = []
            for key in best.parameter_overrides:
                if key in worst.parameter_overrides:
                    if best.parameter_overrides[key] != worst.parameter_overrides[key]:
                        diff_params.append(key)
            
            if diff_params:
                insights.append(Insight(
                    insight_type="sensitivity",
                    title="Key Performance Drivers Identified",
                    description=(
                        f"Parameters {diff_params} significantly affect outcomes. "
                        f"Score range: {worst.score:.3f} - {best.score:.3f} "
                        f"({(best.score - worst.score) / worst.score * 100:.1f}% variance)"
                    ),
                    confidence=0.75,
                    related_metrics=diff_params,
                    recommended_action=f"Focus optimization on: {', '.join(diff_params[:3])}",
                    severity="info"
                ))
        
        return insights
    
    def _detect_outliers(self, outcomes: List[ScenarioOutcome]) -> List[Insight]:
        """Detect outlier outcomes."""
        insights = []
        
        if len(outcomes) < 5:
            return insights
        
        scores = [o.score for o in outcomes]
        mean_score = statistics.mean(scores)
        try:
            stdev_score = statistics.stdev(scores)
        except statistics.StatisticsError:
            stdev_score = 0.0
        
        if stdev_score == 0:
            return insights
        
        outliers_high = []
        outliers_low = []
        
        for outcome in outcomes:
            z_score = (outcome.score - mean_score) / stdev_score
            
            if z_score > self.outlier_threshold:
                outliers_high.append((outcome, z_score))
            elif z_score < -self.outlier_threshold:
                outliers_low.append((outcome, z_score))
        
        if outliers_high:
            insights.append(Insight(
                insight_type="outlier",
                title="Exceptional Performance Detected",
                description=(
                    f"{len(outliers_high)} scenario(s) performed significantly better than average. "
                    f"Best outlier: {outliers_high[0][0].scenario_name} "
                    f"(z-score: {outliers_high[0][1]:.2f})"
                ),
                confidence=0.85,
                recommended_action="Analyze configuration of high-performing scenarios for replication",
                severity="info"
            ))
        
        if outliers_low:
            insights.append(Insight(
                insight_type="outlier",
                title="Underperforming Scenarios Detected",
                description=(
                    f"{len(outliers_low)} scenario(s) performed significantly worse than average. "
                    f"Worst outlier: {outliers_low[0][0].scenario_name} "
                    f"(z-score: {outliers_low[0][1]:.2f})"
                ),
                confidence=0.85,
                recommended_action="Investigate causes of poor performance and consider exclusion",
                severity="warning"
            ))
        
        return insights
    
    def _identify_trends(self, outcomes: List[ScenarioOutcome]) -> List[Insight]:
        """Identify trends in outcomes."""
        insights = []
        
        # Check for monotonic trends in parameters
        # This would be expanded with more sophisticated trend detection
        
        return insights
    
    def _analyze_winner(self, outcomes: List[ScenarioOutcome]) -> Optional[Insight]:
        """Analyze the winning scenario."""
        if not outcomes:
            return None
        
        winner = max(outcomes, key=lambda o: o.score)
        
        if winner.score < 0.5:
            return Insight(
                insight_type="trend",
                title="No Clear Winner",
                description=(
                    f"Best performing scenario '{winner.scenario_name}' "
                    f"only achieved score of {winner.score:.3f}. "
                    f"Consider expanding search space or adjusting approach."
                ),
                confidence=0.7,
                recommended_action="Expand parameter ranges or try different strategies",
                severity="warning"
            )
        
        return Insight(
            insight_type="trend",
            title=f"Winner: {winner.scenario_name}",
            description=(
                f"Best configuration achieved score {winner.score:.3f}. "
                f"Key parameters: {winner.parameter_overrides}"
            ),
            confidence=winner.score,
            related_metrics=list(winner.metrics.keys()),
            recommended_action="Use winning configuration as baseline for further optimization",
            severity="info"
        )
    
    def _count_by_type(self, insights: List[Insight]) -> Dict[str, int]:
        """Count insights by type."""
        counts = {}
        for insight in insights:
            counts[insight.insight_type] = counts.get(insight.insight_type, 0) + 1
        return counts
    
    def calculate_sensitivity(
        self, 
        outcomes: List[ScenarioOutcome], 
        parameter: str
    ) -> Optional[ParameterSensitivity]:
        """
        Calculate sensitivity of outcomes to a specific parameter.
        
        Args:
            outcomes: Simulation outcomes
            parameter: Parameter name to analyze
            
        Returns:
            ParameterSensitivity or None if insufficient data
        """
        # Extract parameter values and scores
        data_points = []
        for outcome in outcomes:
            if parameter in outcome.parameter_overrides:
                try:
                    value = float(outcome.parameter_overrides[parameter])
                    data_points.append((value, outcome.score))
                except (ValueError, TypeError):
                    continue
        
        if len(data_points) < 3:
            return None
        
        # Calculate correlation
        values = [p[0] for p in data_points]
        scores = [p[1] for p in data_points]
        
        try:
            correlation = statistics.correlation(values, scores)
        except (statistics.StatisticsError, AttributeError):
            # Fallback manual calculation
            n = len(values)
            mean_v = sum(values) / n
            mean_s = sum(scores) / n
            
            numerator = sum((v - mean_v) * (s - mean_s) for v, s in data_points)
            denom_v = math.sqrt(sum((v - mean_v) ** 2 for v in values))
            denom_s = math.sqrt(sum((s - mean_s) ** 2 for s in scores))
            
            correlation = numerator / (denom_v * denom_s) if denom_v * denom_s > 0 else 0
        
        # Calculate impact score (R² approximation)
        impact = abs(correlation)
        
        # Find optimal range (simplified)
        sorted_points = sorted(data_points, key=lambda x: x[1], reverse=True)
        top_performers = sorted_points[:max(1, len(sorted_points) // 3)]
        
        if top_performers:
            optimal_min = min(p[0] for p in top_performers)
            optimal_max = max(p[0] for p in top_performers)
            optimal_range = (optimal_min, optimal_max)
        else:
            optimal_range = None
        
        return ParameterSensitivity(
            parameter_name=parameter,
            correlation=correlation,
            impact_score=impact,
            optimal_range=optimal_range,
            recommendation=(
                f"{'Increase' if correlation > 0 else 'Decrease'} {parameter} "
                f"for better outcomes (correlation: {correlation:.2f})"
            ) if abs(correlation) > 0.3 else f"{parameter} has weak impact on outcomes"
        )
