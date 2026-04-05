"""Adversarial strategies for red-team critique."""

import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from agents.adversarial.agent import Finding, StrategyResult, TargetType
from core.logging_utils import log_json


class AdversarialStrategyBase(ABC):
    """Base class for adversarial strategies."""
    
    def __init__(self, model=None):
        """
        Initialize the strategy.
        
        Args:
            model: Optional model adapter for generating content
        """
        self.model = model
    
    @abstractmethod
    async def execute(
        self,
        target: str,
        target_type: TargetType,
        context: Dict[str, Any],
        intensity: float
    ) -> StrategyResult:
        """Execute the adversarial strategy."""
        pass
    
    def _generate_with_model(self, prompt: str) -> str:
        """Generate content using the model or fallback."""
        if self.model:
            try:
                if hasattr(self.model, 'generate'):
                    return self.model.generate(prompt)
                elif hasattr(self.model, 'respond'):
                    return self.model.respond(prompt)
            except Exception as e:
                log_json("WARN", "model_generation_failed", {"error": str(e)})
        
        # Fallback: return empty or use heuristic generation
        return ""
    
    def _estimate_confidence(self, findings: List[Finding], intensity: float) -> float:
        """Estimate confidence based on findings and intensity."""
        if not findings:
            return 0.3 * intensity
        
        base_confidence = min(0.9, 0.5 + len(findings) * 0.1)
        return min(1.0, base_confidence * intensity)


class DevilsAdvocateStrategy(AdversarialStrategyBase):
    """Argue against the proposed solution regardless of its merits."""
    
    async def execute(
        self,
        target: str,
        target_type: TargetType,
        context: Dict[str, Any],
        intensity: float
    ) -> StrategyResult:
        """Execute devil's advocate strategy."""
        start_time = time.time()
        
        # Generate critique via model or heuristic
        if self.model:
            prompt = self._build_prompt(target, target_type, context, intensity)
            response = self._generate_with_model(prompt)
            findings = self._parse_findings(response)
        else:
            # Heuristic findings
            findings = self._heuristic_findings(target, target_type, intensity)
        
        execution_time = time.time() - start_time
        
        return StrategyResult(
            strategy="devils_advocate",
            findings=findings,
            confidence=self._estimate_confidence(findings, intensity),
            execution_time=execution_time
        )
    
    def _build_prompt(
        self,
        target: str,
        target_type: TargetType,
        context: Dict[str, Any],
        intensity: float
    ) -> str:
        """Build the devil's advocate prompt."""
        return f"""
You are a skeptical principal engineer who must argue AGAINST this {target_type.value}.
Your job is to find the strongest arguments against it, even if it's actually good.
Be constructively critical and thorough.

Target ({target_type.value}):
{target}

Context: {context}

Argue against this solution. Find:
1. Hidden costs and trade-offs being ignored
2. Better alternatives that were dismissed too quickly
3. Premature optimization or over-engineering risks
4. Maintenance burden and technical debt concerns
5. Compatibility and integration risks
6. Scalability limitations
7. Edge cases not considered

For each issue, provide:
- Category (tradeoff|risk|alternative|limitation)
- Severity (critical|high|medium|low)
- Description of the issue
- Evidence from the target
- Recommendation for mitigation

Intensity level: {intensity*100:.0f}%
Respond with a structured list of findings.
"""
    
    def _parse_findings(self, response: str) -> List[Finding]:
        """Parse findings from model response."""
        findings = []
        
        # Simple parsing - look for numbered or bulleted items
        lines = response.strip().split('\n')
        current_finding = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_finding:
                    findings.append(self._build_finding(current_finding))
                    current_finding = {}
                continue
            
            # Look for category markers
            if line.lower().startswith('category:'):
                current_finding['category'] = line.split(':', 1)[1].strip()
            elif line.lower().startswith('severity:'):
                current_finding['severity'] = line.split(':', 1)[1].strip().lower()
            elif line.lower().startswith('description:'):
                current_finding['description'] = line.split(':', 1)[1].strip()
            elif line.lower().startswith('evidence:'):
                current_finding['evidence'] = line.split(':', 1)[1].strip()
            elif line.lower().startswith('recommendation:'):
                current_finding['recommendation'] = line.split(':', 1)[1].strip()
            elif line and not current_finding.get('description'):
                # First non-metadata line might be description
                current_finding['description'] = line
        
        if current_finding:
            findings.append(self._build_finding(current_finding))
        
        return findings
    
    def _build_finding(self, data: Dict[str, str]) -> Finding:
        """Build a Finding from parsed data."""
        return Finding(
            category=data.get('category', 'general'),
            severity=data.get('severity', 'medium'),
            description=data.get('description', 'No description provided'),
            evidence=data.get('evidence', 'No evidence provided'),
            recommendation=data.get('recommendation', 'No recommendation provided'),
            confidence=0.7
        )
    
    def _heuristic_findings(
        self,
        target: str,
        target_type: TargetType,
        intensity: float
    ) -> List[Finding]:
        """Generate heuristic findings when no model available."""
        findings = []
        
        # Common concerns for all targets
        findings.append(Finding(
            category="tradeoff",
            severity="medium",
            description="Solution may introduce unnecessary complexity",
            evidence="Complexity often increases with any solution",
            recommendation="Evaluate if a simpler approach could achieve the same goal",
            confidence=0.6 * intensity
        ))
        
        if target_type == TargetType.CODE:
            # Code-specific concerns
            findings.append(Finding(
                category="maintenance",
                severity="medium",
                description="Code may be difficult to maintain long-term",
                evidence="New code patterns may not be familiar to all team members",
                recommendation="Add comprehensive documentation and comments",
                confidence=0.5 * intensity
            ))
        
        elif target_type == TargetType.PLAN:
            # Plan-specific concerns
            findings.append(Finding(
                category="risk",
                severity="high",
                description="Plan may underestimate implementation challenges",
                evidence="Plans often overlook integration complexities",
                recommendation="Add buffer time and identify potential blockers early",
                confidence=0.7 * intensity
            ))
        
        return findings


class EdgeCaseHunterStrategy(AdversarialStrategyBase):
    """Find boundary conditions and edge cases that break the solution."""
    
    async def execute(
        self,
        target: str,
        target_type: TargetType,
        context: Dict[str, Any],
        intensity: float
    ) -> StrategyResult:
        """Hunt for edge cases."""
        start_time = time.time()
        
        if self.model:
            prompt = self._build_prompt(target, target_type, context, intensity)
            response = self._generate_with_model(prompt)
            findings = self._parse_edge_cases(response)
        else:
            findings = self._heuristic_edge_cases(target, target_type, intensity)
        
        execution_time = time.time() - start_time
        
        return StrategyResult(
            strategy="edge_case_hunter",
            findings=findings,
            confidence=min(1.0, len(findings) / 5) * intensity,
            execution_time=execution_time
        )
    
    def _build_prompt(
        self,
        target: str,
        target_type: TargetType,
        context: Dict[str, Any],
        intensity: float
    ) -> str:
        """Build the edge case hunter prompt."""
        return f"""
You are an expert at finding edge cases that break software and systems.
Analyze this {target_type.value} and find ALL possible edge cases, boundary conditions,
and exceptional scenarios that could cause failures.

Target:
{target}

Consider these categories of edge cases:
1. Empty/null/None inputs
2. Maximum size limits (files, arrays, strings, memory)
3. Unicode, special characters, encoding issues
4. Race conditions and concurrency scenarios
5. Resource exhaustion (memory, disk, CPU)
6. Network failures, timeouts, partial failures
7. Clock skew and timing edge cases
8. Numeric overflow/underflow, precision loss
9. Permission and authorization edge cases
10. State machine edge cases and invalid transitions

For each edge case found:
- Description of the input/condition
- Why it breaks the solution
- Suggested test case to verify
- Severity (critical|high|medium|low)

Intensity level: {intensity*100:.0f}%
Respond with a structured list of edge cases.
"""
    
    def _parse_edge_cases(self, response: str) -> List[Finding]:
        """Parse edge cases from response."""
        findings = []
        
        # Split by double newline or numbered items
        sections = re.split(r'\n\n+|\d+\.', response)
        
        for section in sections:
            section = section.strip()
            if not section or len(section) < 20:
                continue
            
            # Extract severity if mentioned
            severity = "medium"
            if "critical" in section.lower():
                severity = "critical"
            elif "high" in section.lower():
                severity = "high"
            elif "low" in section.lower():
                severity = "low"
            
            findings.append(Finding(
                category="edge_case",
                severity=severity,
                description=section[:200],
                evidence="Edge case analysis",
                recommendation=f"Add test case for: {section[:100]}...",
                confidence=0.75
            ))
        
        return findings
    
    def _heuristic_edge_cases(
        self,
        target: str,
        target_type: TargetType,
        intensity: float
    ) -> List[Finding]:
        """Generate common edge cases heuristically."""
        findings = []
        
        if target_type == TargetType.CODE:
            findings.extend([
                Finding(
                    category="edge_case",
                    severity="high",
                    description="Null/None input handling",
                    evidence="Functions may not validate inputs",
                    recommendation="Add null checks and input validation",
                    confidence=0.8 * intensity
                ),
                Finding(
                    category="edge_case",
                    severity="medium",
                    description="Empty collections or strings",
                    evidence="Edge case often overlooked",
                    recommendation="Test with empty inputs",
                    confidence=0.7 * intensity
                ),
                Finding(
                    category="edge_case",
                    severity="high",
                    description="Very large input sizes",
                    evidence="Memory and performance limits may be exceeded",
                    recommendation="Add size limits and pagination",
                    confidence=0.75 * intensity
                ),
            ])
        
        elif target_type == TargetType.API:
            findings.extend([
                Finding(
                    category="edge_case",
                    severity="critical",
                    description="Malformed or unexpected request payloads",
                    evidence="APIs may not validate all input formats",
                    recommendation="Add comprehensive request validation",
                    confidence=0.85 * intensity
                ),
                Finding(
                    category="edge_case",
                    severity="high",
                    description="Rate limiting and throttling scenarios",
                    evidence="Load handling may not be tested",
                    recommendation="Implement and test rate limiting",
                    confidence=0.8 * intensity
                ),
            ])
        
        return findings


class AssumptionChallengeStrategy(AdversarialStrategyBase):
    """Challenge underlying assumptions in the solution."""
    
    async def execute(
        self,
        target: str,
        target_type: TargetType,
        context: Dict[str, Any],
        intensity: float
    ) -> StrategyResult:
        """Challenge assumptions."""
        start_time = time.time()
        
        if self.model:
            prompt = self._build_prompt(target, target_type, context, intensity)
            response = self._generate_with_model(prompt)
            findings = self._parse_assumptions(response)
        else:
            findings = self._heuristic_assumptions(target, target_type, intensity)
        
        execution_time = time.time() - start_time
        
        return StrategyResult(
            strategy="assumption_challenge",
            findings=findings,
            confidence=self._estimate_confidence(findings, intensity),
            execution_time=execution_time
        )
    
    def _build_prompt(
        self,
        target: str,
        target_type: TargetType,
        context: Dict[str, Any],
        intensity: float
    ) -> str:
        """Build the assumption challenge prompt."""
        return f"""
Analyze this {target_type.value} and identify ALL assumptions being made.
Then challenge each assumption - what if it's wrong? What are the consequences?

Target:
{target}

Common assumptions to challenge:
- "The input will always be valid/well-formed"
- "This code won't be used concurrently"
- "The network is reliable and available"
- "Users will only use this as intended"
- "Data will always fit in memory"
- "This function is fast enough for all use cases"
- "Dependencies won't change or break"
- "The system has sufficient resources"
- "External services will respond quickly"
- "Security context is properly established"

For each assumption:
1. State the assumption clearly
2. Explain scenarios where it might be false
3. Describe the consequences if it fails
4. Suggest defensive measures or alternatives
5. Rate severity (critical|high|medium|low)

Intensity level: {intensity*100:.0f}%
Respond with a structured list of challenged assumptions.
"""
    
    def _parse_assumptions(self, response: str) -> List[Finding]:
        """Parse assumptions from response."""
        findings = []
        
        sections = re.split(r'\n\n+|\d+\.', response)
        
        for section in sections:
            section = section.strip()
            if not section or len(section) < 30:
                continue
            
            severity = "medium"
            if "critical" in section.lower():
                severity = "critical"
            elif "high" in section.lower():
                severity = "high"
            
            findings.append(Finding(
                category="assumption",
                severity=severity,
                description=section[:250],
                evidence="Assumption analysis",
                recommendation="Validate assumption or add defensive code",
                confidence=0.7
            ))
        
        return findings
    
    def _heuristic_assumptions(
        self,
        target: str,
        target_type: TargetType,
        intensity: float
    ) -> List[Finding]:
        """Generate common assumption challenges."""
        assumptions = [
            Finding(
                category="assumption",
                severity="high",
                description="Assumption: Input data is always valid",
                evidence="No validation logic may be present",
                recommendation="Add comprehensive input validation",
                confidence=0.8 * intensity
            ),
            Finding(
                category="assumption",
                severity="medium",
                description="Assumption: Sufficient system resources available",
                evidence="Resource constraints may not be considered",
                recommendation="Add resource checks and graceful degradation",
                confidence=0.6 * intensity
            ),
            Finding(
                category="assumption",
                severity="medium",
                description="Assumption: Single-threaded execution",
                evidence="Concurrency issues may not be addressed",
                recommendation="Review for thread safety",
                confidence=0.65 * intensity
            ),
        ]
        
        return assumptions


class WorstCaseStrategy(AdversarialStrategyBase):
    """Identify catastrophic failure modes."""
    
    async def execute(
        self,
        target: str,
        target_type: TargetType,
        context: Dict[str, Any],
        intensity: float
    ) -> StrategyResult:
        """Execute worst-case analysis."""
        start_time = time.time()
        
        findings = [
            Finding(
                category="failure_mode",
                severity="critical",
                description="Complete system failure under load",
                evidence="No circuit breaker or fallback mechanisms visible",
                recommendation="Implement circuit breakers and graceful degradation",
                confidence=0.7 * intensity
            ),
            Finding(
                category="failure_mode",
                severity="high",
                description="Data corruption or loss scenario",
                evidence="Transaction safety may not be guaranteed",
                recommendation="Add transactions and backup mechanisms",
                confidence=0.6 * intensity
            ),
        ]
        
        execution_time = time.time() - start_time
        
        return StrategyResult(
            strategy="worst_case",
            findings=findings,
            confidence=0.65 * intensity,
            execution_time=execution_time
        )


class SecurityMindsetStrategy(AdversarialStrategyBase):
    """Identify security vulnerabilities."""
    
    async def execute(
        self,
        target: str,
        target_type: TargetType,
        context: Dict[str, Any],
        intensity: float
    ) -> StrategyResult:
        """Execute security analysis."""
        start_time = time.time()
        
        findings = []
        
        # Common security concerns
        if target_type in [TargetType.CODE, TargetType.API]:
            findings.extend([
                Finding(
                    category="security",
                    severity="critical",
                    description="Input validation bypass possibility",
                    evidence="User input may reach sensitive operations",
                    recommendation="Implement strict input validation and sanitization",
                    confidence=0.75 * intensity
                ),
                Finding(
                    category="security",
                    severity="high",
                    description="Potential injection vulnerability",
                    evidence="String concatenation or formatting with user input",
                    recommendation="Use parameterized queries and safe APIs",
                    confidence=0.7 * intensity
                ),
                Finding(
                    category="security",
                    severity="medium",
                    description="Information disclosure through error messages",
                    evidence="Detailed error messages may leak implementation details",
                    recommendation="Sanitize error messages for production",
                    confidence=0.6 * intensity
                ),
            ])
        
        execution_time = time.time() - start_time
        
        return StrategyResult(
            strategy="security_mindset",
            findings=findings,
            confidence=0.7 * intensity,
            execution_time=execution_time
        )


class ScalabilityFocusStrategy(AdversarialStrategyBase):
    """Focus on scalability and performance limitations."""
    
    async def execute(
        self,
        target: str,
        target_type: TargetType,
        context: Dict[str, Any],
        intensity: float
    ) -> StrategyResult:
        """Execute scalability analysis."""
        start_time = time.time()
        
        findings = [
            Finding(
                category="scalability",
                severity="high",
                description="Algorithm may not scale with data size",
                evidence="Complexity may be O(n²) or worse",
                recommendation="Analyze time/space complexity and optimize",
                confidence=0.65 * intensity
            ),
            Finding(
                category="scalability",
                severity="medium",
                description="Memory usage may grow unbounded",
                evidence="No limits on caching or data retention",
                recommendation="Add memory limits and LRU caching",
                confidence=0.6 * intensity
            ),
            Finding(
                category="scalability",
                severity="medium",
                description="No horizontal scaling considerations",
                evidence="Design may assume single instance",
                recommendation="Design for statelessness and horizontal scaling",
                confidence=0.55 * intensity
            ),
        ]
        
        execution_time = time.time() - start_time
        
        return StrategyResult(
            strategy="scalability_focus",
            findings=findings,
            confidence=0.6 * intensity,
            execution_time=execution_time
        )
