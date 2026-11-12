"""Pattern recognition for successful executions."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from .feedback import ExecutionOutcome, ExecutionStatus


@dataclass
class SuccessPattern:
    """A recognized success pattern."""
    pattern_id: str
    description: str
    keywords: List[str]
    agent_name: str
    success_count: int
    total_count: int
    avg_quality: float
    examples: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.success_count / self.total_count
    
    def to_dict(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "keywords": self.keywords,
            "agent_name": self.agent_name,
            "success_count": self.success_count,
            "total_count": self.total_count,
            "success_rate": self.success_rate,
            "avg_quality": self.avg_quality,
            "examples": self.examples[:5],  # Limit examples
        }


class PatternRecognizer:
    """Recognize patterns in successful executions."""
    
    def __init__(self):
        self.patterns: Dict[str, SuccessPattern] = {}
        self._keyword_index: Dict[str, List[str]] = defaultdict(list)
    
    def analyze_outcomes(self, outcomes: List[ExecutionOutcome]) -> List[SuccessPattern]:
        """Analyze outcomes to identify success patterns."""
        # Group by agent
        by_agent: Dict[str, List[ExecutionOutcome]] = defaultdict(list)
        for outcome in outcomes:
            by_agent[outcome.agent_name].append(outcome)
        
        patterns = []
        for agent_name, agent_outcomes in by_agent.items():
            agent_patterns = self._analyze_agent_outcomes(agent_name, agent_outcomes)
            patterns.extend(agent_patterns)
        
        return patterns
    
    def _analyze_agent_outcomes(
        self,
        agent_name: str,
        outcomes: List[ExecutionOutcome],
    ) -> List[SuccessPattern]:
        """Analyze outcomes for a specific agent."""
        patterns = []
        
        # Separate successes and failures
        successes = [o for o in outcomes if o.status == ExecutionStatus.SUCCESS]
        failures = [o for o in outcomes if o.status == ExecutionStatus.FAILURE]
        
        if len(successes) < 3:
            return patterns  # Not enough data
        
        # Extract common keywords from successful goals
        success_keywords = self._extract_common_keywords(
            [o.goal for o in successes]
        )
        
        # Extract common keywords from failed goals
        failure_keywords = self._extract_common_keywords(
            [o.goal for o in failures]
        ) if failures else set()
        
        # Find distinguishing keywords (common in success, rare in failure)
        distinguishing = success_keywords - failure_keywords
        
        if distinguishing:
            pattern = SuccessPattern(
                pattern_id=f"{agent_name}_success",
                description=f"Successful {agent_name} executions",
                keywords=list(distinguishing)[:10],
                agent_name=agent_name,
                success_count=len(successes),
                total_count=len(outcomes),
                avg_quality=sum(o.output_quality for o in outcomes) / len(outcomes),
                examples=[o.goal for o in successes[:5]],
            )
            patterns.append(pattern)
            self._index_pattern(pattern)
        
        # Look for specific patterns by goal type
        goal_patterns = self._find_goal_type_patterns(agent_name, successes, failures)
        patterns.extend(goal_patterns)
        
        return patterns
    
    def _extract_common_keywords(self, texts: List[str], min_freq: float = 0.3) -> Set[str]:
        """Extract keywords that appear frequently in texts."""
        keyword_counts: Dict[str, int] = defaultdict(int)
        
        for text in texts:
            keywords = self._tokenize(text)
            for kw in keywords:
                keyword_counts[kw] += 1
        
        # Keep keywords that appear in at least min_freq of texts
        threshold = len(texts) * min_freq
        return {kw for kw, count in keyword_counts.items() if count >= threshold}
    
    def _find_goal_type_patterns(
        self,
        agent_name: str,
        successes: List[ExecutionOutcome],
        failures: List[ExecutionOutcome],
    ) -> List[SuccessPattern]:
        """Find patterns for specific goal types."""
        patterns = []
        
        # Group by goal type (refactor, test, add, fix, etc.)
        goal_types = self._categorize_goals([o.goal for o in successes])
        
        for goal_type, goals in goal_types.items():
            if len(goals) < 3:
                continue
            
            type_successes = [o for o in successes if goal_type in o.goal.lower()]
            type_failures = [o for o in failures if goal_type in o.goal.lower()]
            
            if not type_successes:
                continue
            
            success_rate = len(type_successes) / (len(type_successes) + len(type_failures) + 0.001)
            
            if success_rate > 0.7:  # High success rate for this type
                pattern = SuccessPattern(
                    pattern_id=f"{agent_name}_{goal_type}",
                    description=f"{agent_name} excels at {goal_type} tasks",
                    keywords=[goal_type],
                    agent_name=agent_name,
                    success_count=len(type_successes),
                    total_count=len(type_successes) + len(type_failures),
                    avg_quality=sum(o.output_quality for o in type_successes) / len(type_successes),
                    examples=[o.goal for o in type_successes[:5]],
                )
                patterns.append(pattern)
                self._index_pattern(pattern)
        
        return patterns
    
    def _categorize_goals(self, goals: List[str]) -> Dict[str, List[str]]:
        """Categorize goals by type."""
        categories: Dict[str, List[str]] = defaultdict(list)
        
        type_keywords = {
            "refactor": ["refactor", "restructure", "cleanup", "clean up"],
            "test": ["test", "testing", "spec", "assert"],
            "add": ["add", "create", "implement", "new"],
            "fix": ["fix", "bug", "error", "repair"],
            "optimize": ["optimize", "performance", "speed", "fast"],
            "document": ["doc", "comment", "readme", "documentation"],
        }
        
        for goal in goals:
            goal_lower = goal.lower()
            matched = False
            
            for category, keywords in type_keywords.items():
                if any(kw in goal_lower for kw in keywords):
                    categories[category].append(goal)
                    matched = True
                    break
            
            if not matched:
                categories["other"].append(goal)
        
        return dict(categories)
    
    def _index_pattern(self, pattern: SuccessPattern):
        """Add pattern to keyword index."""
        self.patterns[pattern.pattern_id] = pattern
        
        for keyword in pattern.keywords:
            if pattern.pattern_id not in self._keyword_index[keyword]:
                self._keyword_index[keyword].append(pattern.pattern_id)
    
    def find_matching_patterns(self, goal: str) -> List[SuccessPattern]:
        """Find patterns that match a given goal."""
        keywords = self._tokenize(goal)
        matching_ids: Set[str] = set()
        
        for keyword in keywords:
            matching_ids.update(self._keyword_index.get(keyword, []))
        
        patterns = [self.patterns[pid] for pid in matching_ids if pid in self.patterns]
        
        # Sort by success rate
        patterns.sort(key=lambda p: p.success_rate, reverse=True)
        
        return patterns
    
    def get_recommendations(self, goal: str) -> List[str]:
        """Get recommendations for a goal based on patterns."""
        patterns = self.find_matching_patterns(goal)
        
        recommendations = []
        for pattern in patterns[:3]:  # Top 3 patterns
            if pattern.success_rate > 0.8:
                recommendations.append(
                    f"This looks like a {pattern.description} "
                    f"(success rate: {pattern.success_rate:.0%})"
                )
        
        return recommendations
    
    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text into keywords."""
        text = text.lower()
        
        # Replace separators with spaces
        for sep in ['_', '-', '.', '(', ')', ',', ':']:
            text = text.replace(sep, ' ')
        
        # Extract words
        words = set()
        for word in text.split():
            word = word.strip()
            if len(word) >= 3:  # Filter short words
                words.add(word)
        
        return words
    
    def export_patterns(self) -> Dict[str, Any]:
        """Export all patterns as dictionary."""
        return {
            "patterns": [p.to_dict() for p in self.patterns.values()],
            "total_patterns": len(self.patterns),
        }
