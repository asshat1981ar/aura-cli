"""Multi-agent workflow orchestrator for architecture analysis and improvement.

This module coordinates multiple specialized agents to analyze project architecture,
predict failure modes, suggest improvements, and generate comprehensive summaries.
"""

from typing import Any, Dict, List
from datetime import datetime

from core.logging_utils import log_json


class _AgentStub:
    """Placeholder for unimplemented external agent."""

    @classmethod
    def analyze_architecture(cls, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {}

    @classmethod
    def assess_capabilities(cls, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {}

    @classmethod
    def create_execution_plan(cls, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {}


PythonAgent = _AgentStub
TypeScriptAgent = _AgentStub
CodeSearchAgent = _AgentStub


def create_multi_agent_workflow():
    """Execute the complete multi-agent architecture analysis workflow.

    This function orchestrates the following steps:
    1. Analyze Python project architecture using PythonAgent
    2. Assess TypeScript capabilities using TypeScriptAgent
    3. Create execution plan using CodeSearchAgent
    4. Predict and classify potential failure modes
    5. Suggest architectural improvements
    6. Compile findings into markdown summary
    7. Validate findings with MCP server
    8. Engage stakeholders with validation results

    Returns:
        str: Completion status message
    """
    # Step 1: Analyze Current Project Architecture
    python_analysis_tools = ["pylint", "mypy", "bandit"]
    python_results = PythonAgent.analyze_architecture(python_analysis_tools)

    # Step 2: Assess Typescript Capabilities
    typescript_analysis_metrics = ["dependency metrics", "code complexity"]
    typescript_results = TypeScriptAgent.assess_capabilities(typescript_analysis_metrics)

    # Step 3: Create Execution Plan with CodeSearchAgent
    execution_plan = CodeSearchAgent.create_execution_plan(python_results, typescript_results)

    # Step 4: Predict Potential Failure Modes
    failure_modes_analysis = predict_failure_modes(python_results)
    classified_failure_modes = classify_failure_modes(failure_modes_analysis)

    # Step 5: Suggest Architectural Improvements
    architectural_suggestions = suggest_architectural_improvements(python_results)

    # Step 6: Compile Findings into 'architecture_summary.md'
    summary_content = compile_summary(python_results, typescript_results, execution_plan, classified_failure_modes, architectural_suggestions)
    with open("architecture_summary.md", "w") as summary_file:
        summary_file.write(summary_content)

    log_json("INFO", "multi_agent_workflow_summary_generated", {"file": "architecture_summary.md"})

    # Step 7: Validate Findings with MCP Server
    validation_results = validate_with_mcp_server(summary_content)

    # Step 8: Engage Stakeholders
    engage_stakeholders(validation_results)

    return "Multi-agent workflow completed and architecture_summary.md generated."


def predict_failure_modes(architecture_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Analyze architecture results to predict potential failure modes.

    Examines Python analysis results including lint errors, type check issues,
    complexity metrics, and test coverage to identify potential failure points
    in the codebase.

    Args:
        architecture_results: Dictionary containing analysis results from
            PythonAgent including lint_results, type_check_results,
            complexity, and coverage data.

    Returns:
        List of dictionaries containing predicted failure modes with
        component, failure_type, description, and triggering_conditions.
    """
    failure_modes = []

    # Extract relevant data from architecture results
    lint_results = architecture_results.get("lint_results", {})
    type_check_results = architecture_results.get("type_check_results", {})
    complexity = architecture_results.get("complexity", {})
    coverage = architecture_results.get("coverage", {})

    # Analyze lint results for potential runtime failures
    if lint_results.get("status") != "skill_not_available":
        errors = lint_results.get("errors", [])
        if errors:
            failure_modes.append(
                {
                    "component": "Code Quality",
                    "failure_type": "Runtime Error",
                    "description": f"Linting detected {len(errors)} issues that may cause runtime failures",
                    "triggering_conditions": ["High error density in modified files", "Missing exception handling"],
                    "severity": "high" if len(errors) > 10 else "medium",
                }
            )

    # Analyze type check results for type-related failures
    if type_check_results.get("status") != "skill_not_available":
        type_errors = type_check_results.get("errors", "")
        if type_errors:
            failure_modes.append({"component": "Type System", "failure_type": "Type Mismatch", "description": "Type checking inconsistencies detected that may lead to AttributeError or TypeError", "triggering_conditions": ["Dynamic type usage", "Missing type annotations"], "severity": "high"})

    # Analyze complexity for maintainability failures
    complexity_score = complexity.get("score", 0)
    if complexity_score and complexity_score > 7:
        failure_modes.append(
            {
                "component": "Maintainability",
                "failure_type": "Technical Debt Accumulation",
                "description": f"High cyclomatic complexity ({complexity_score}) indicates difficult-to-maintain code",
                "triggering_conditions": ["Frequent modifications to complex functions", "Lack of unit tests"],
                "severity": "medium",
            }
        )

    # Analyze test coverage for quality failures
    coverage_pct = coverage.get("percentage", 100)
    if coverage_pct < 70:
        failure_modes.append(
            {
                "component": "Test Coverage",
                "failure_type": "Regression",
                "description": f"Low test coverage ({coverage_pct}%) increases risk of undetected regressions",
                "triggering_conditions": ["Code changes in uncovered regions", "Refactoring without tests"],
                "severity": "high" if coverage_pct < 50 else "medium",
            }
        )

    # Check for missing skills which indicate operational gaps
    missing_skills = [result.get("skill") for result in [lint_results, type_check_results, complexity, coverage] if result.get("status") == "skill_not_available"]
    if missing_skills:
        failure_modes.append({"component": "Operational", "failure_type": "Capability Gap", "description": f"Missing analysis skills: {', '.join(set(missing_skills))}", "triggering_conditions": ["Incomplete toolchain setup", "Missing dependencies"], "severity": "low"})

    log_json("INFO", "failure_modes_predicted", {"count": len(failure_modes)})
    return failure_modes


def classify_failure_modes(failure_modes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Classify failure modes by likelihood and severity.

    Applies a risk matrix to categorize each failure mode as Critical,
    High, Medium, or Low risk based on its severity and estimated
    likelihood of occurrence.

    Args:
        failure_modes: List of failure mode dictionaries from
            predict_failure_modes().

    Returns:
        List of classified failure modes with added 'risk_level',
        'likelihood', and 'priority' fields, sorted by priority
        (highest first).
    """
    classified = []

    for mode in failure_modes:
        severity = mode.get("severity", "medium")
        component = mode.get("component", "")

        # Estimate likelihood based on component type and severity
        likelihood_map = {"Code Quality": "high", "Type System": "medium", "Maintainability": "medium", "Test Coverage": "high", "Operational": "low"}
        likelihood = likelihood_map.get(component, "medium")

        # Calculate risk level using risk matrix
        risk_matrix = {("high", "high"): "Critical", ("high", "medium"): "High", ("high", "low"): "Medium", ("medium", "high"): "High", ("medium", "medium"): "Medium", ("medium", "low"): "Low", ("low", "high"): "Medium", ("low", "medium"): "Low", ("low", "low"): "Low"}
        risk_level = risk_matrix.get((severity, likelihood), "Medium")

        # Assign priority score (higher = more urgent)
        priority_map = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
        priority = priority_map.get(risk_level, 2)

        classified_mode = {**mode, "risk_level": risk_level, "likelihood": likelihood, "priority": priority}
        classified.append(classified_mode)

    # Sort by priority (descending)
    classified.sort(key=lambda x: x["priority"], reverse=True)

    log_json("INFO", "failure_modes_classified", {"count": len(classified)})
    return classified


def suggest_architectural_improvements(architecture_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Suggest specific architectural improvements based on analysis results.

    Analyzes the architecture results and generates targeted recommendations
    for improving code quality, test coverage, type safety, and overall
    maintainability.

    Args:
        architecture_results: Dictionary containing analysis results from
            PythonAgent including lint_results, type_check_results,
            complexity, and coverage data.

    Returns:
        List of improvement suggestions with target_component, improvement_type,
        description, expected_benefit, and implementation_effort.
    """
    suggestions = []

    # Extract relevant data
    lint_results = architecture_results.get("lint_results", {})
    type_check_results = architecture_results.get("type_check_results", {})
    complexity = architecture_results.get("complexity", {})
    coverage = architecture_results.get("coverage", {})

    # Suggest linting improvements
    if lint_results.get("status") != "skill_not_available":
        errors = lint_results.get("errors", [])
        if len(errors) > 5:
            suggestions.append(
                {"target_component": "Code Quality", "improvement_type": "Linting Automation", "description": "Integrate automated linting into pre-commit hooks to catch issues early", "expected_benefit": "Reduced runtime errors and consistent code style", "implementation_effort": "low"}
            )

    # Suggest type checking improvements
    if type_check_results.get("status") != "skill_not_available":
        type_errors = type_check_results.get("errors", "")
        if type_errors:
            suggestions.append(
                {
                    "target_component": "Type System",
                    "improvement_type": "Type Annotation Coverage",
                    "description": "Add comprehensive type annotations to public APIs and critical functions",
                    "expected_benefit": "Early detection of type errors and better IDE support",
                    "implementation_effort": "medium",
                }
            )

    # Suggest complexity reduction
    complexity_score = complexity.get("score", 0)
    if complexity_score and complexity_score > 7:
        suggestions.append({"target_component": "Maintainability", "improvement_type": "Refactoring Complex Functions", "description": "Refactor high-complexity functions into smaller, testable units", "expected_benefit": "Improved maintainability and testability", "implementation_effort": "high"})

    # Suggest test coverage improvements
    coverage_pct = coverage.get("percentage", 100)
    if coverage_pct < 80:
        suggestions.append({"target_component": "Test Coverage", "improvement_type": "Test Suite Expansion", "description": f"Increase test coverage from {coverage_pct}% to at least 80%", "expected_benefit": "Reduced regression risk and safer refactoring", "implementation_effort": "medium"})

    # Suggest CI/CD improvements if skills are missing
    missing_skills = [result.get("skill") for result in [lint_results, type_check_results, complexity, coverage] if result.get("status") == "skill_not_available"]
    if missing_skills:
        suggestions.append({"target_component": "Toolchain", "improvement_type": "Skill Dependencies", "description": f"Install missing analysis tools: {', '.join(set(missing_skills))}", "expected_benefit": "Complete analysis coverage and better insights", "implementation_effort": "low"})

    # Always suggest documentation improvement
    suggestions.append(
        {"target_component": "Documentation", "improvement_type": "Architecture Documentation", "description": "Create or update architecture decision records (ADRs) for key design choices", "expected_benefit": "Better onboarding and architectural alignment", "implementation_effort": "medium"}
    )

    log_json("INFO", "architectural_improvements_suggested", {"count": len(suggestions)})
    return suggestions


def _build_header_section(timestamp: str) -> List[str]:
    """Build the header and table of contents for the architecture summary."""
    return [
        "# Architecture Analysis Summary",
        "",
        f"**Generated:** {timestamp}",
        "**Project:** AURA CLI",
        "",
        "---",
        "",
        "## Table of Contents",
        "",
        "1. [Python Analysis Results](#python-analysis-results)",
        "2. [TypeScript Analysis Results](#typescript-analysis-results)",
        "3. [Execution Plan](#execution-plan)",
        "4. [Predicted Failure Modes](#predicted-failure-modes)",
        "5. [Architectural Improvement Suggestions](#architectural-improvement-suggestions)",
        "",
        "---",
        "",
    ]


def _build_python_analysis_section(python_results: Dict[str, Any]) -> List[str]:
    """Build the Python analysis results section."""
    lines: List[str] = ["## Python Analysis Results", "", "### Linting"]

    lint_results = python_results.get("lint_results", {})
    if lint_results.get("status") == "skill_not_available":
        lines.append(f"- Status: {lint_results.get('skill', 'Linter')} not available")
    else:
        errors = lint_results.get("errors", [])
        lines.append(f"- Issues found: {len(errors)}")
        if errors:
            lines.append("- Top issues:")
            for error in errors[:5]:
                lines.append(f"  - {error}")

    lines.extend(["", "### Type Checking"])
    type_results = python_results.get("type_check_results", {})
    if type_results.get("status") == "skill_not_available":
        lines.append(f"- Status: {type_results.get('skill', 'Type checker')} not available")
    else:
        type_errors = type_results.get("errors", "")
        lines.append(f"- Errors: {type_errors if type_errors else 'None'}")

    lines.extend(["", "### Complexity"])
    complexity = python_results.get("complexity", {})
    if complexity.get("status") == "skill_not_available":
        lines.append(f"- Status: {complexity.get('skill', 'Complexity scorer')} not available")
    else:
        score = complexity.get("score", "N/A")
        lines.append(f"- Complexity score: {score}")

    lines.extend(["", "### Test Coverage"])
    coverage = python_results.get("coverage", {})
    if coverage.get("status") == "skill_not_available":
        lines.append(f"- Status: {coverage.get('skill', 'Coverage analyzer')} not available")
    else:
        pct = coverage.get("percentage", "N/A")
        lines.append(f"- Coverage: {pct}%")

    return lines


def _build_typescript_section(typescript_results: Dict[str, Any]) -> List[str]:
    """Build the TypeScript analysis results section."""
    lines: List[str] = ["", "---", "", "## TypeScript Analysis Results", ""]

    action = typescript_results.get("action", "unknown")
    task = typescript_results.get("task", "N/A")
    lines.extend([f"- Action: {action}", f"- Task: {task}", ""])

    ts_lint = typescript_results.get("lint_results", {})
    if ts_lint:
        lines.append(f"- ESLint exit code: {ts_lint.get('exit_code', 0)}")

    ts_type = typescript_results.get("type_check_results", {})
    if ts_type:
        lines.append(f"- TypeScript compiler exit code: {ts_type.get('exit_code', 0)}")

    return lines


def _build_execution_plan_section(execution_plan: Dict[str, Any]) -> List[str]:
    """Build the execution plan section."""
    lines: List[str] = ["", "---", "", "## Execution Plan", ""]

    plan_status = execution_plan.get("status", "unknown")
    query = execution_plan.get("query", "N/A")
    results = execution_plan.get("results", [])

    lines.extend(
        [
            f"- Status: {plan_status}",
            f"- Query: {query}",
            f"- Results found: {len(results)}",
            "",
        ]
    )

    if results:
        lines.append("### Key Findings")
        for result in results[:3]:
            file_path = result.get("file", "unknown")
            score = result.get("score", 0)
            lines.append(f"- {file_path} (relevance: {score:.2f})")

    return lines


def _build_failure_modes_section(classified_failure_modes: List[Dict[str, Any]]) -> List[str]:
    """Build the predicted failure modes section."""
    lines: List[str] = ["", "---", "", "## Predicted Failure Modes", ""]

    if classified_failure_modes:
        lines.append("| Component | Failure Type | Risk Level | Likelihood | Description |")
        lines.append("|-----------|--------------|------------|------------|-------------|")
        for mode in classified_failure_modes:
            component = mode.get("component", "")
            failure_type = mode.get("failure_type", "")
            risk_level = mode.get("risk_level", "")
            likelihood = mode.get("likelihood", "")
            description = mode.get("description", "")[:50] + "..."
            lines.append(f"| {component} | {failure_type} | {risk_level} | {likelihood} | {description} |")
    else:
        lines.append("No failure modes predicted.")

    return lines


def _build_suggestions_section(architectural_suggestions: List[Dict[str, Any]]) -> List[str]:
    """Build the architectural improvement suggestions section."""
    lines: List[str] = ["", "---", "", "## Architectural Improvement Suggestions", ""]

    if architectural_suggestions:
        for i, suggestion in enumerate(architectural_suggestions, 1):
            target = suggestion.get("target_component", "")
            imp_type = suggestion.get("improvement_type", "")
            description = suggestion.get("description", "")
            benefit = suggestion.get("expected_benefit", "")
            effort = suggestion.get("implementation_effort", "")

            lines.extend(
                [
                    f"### {i}. {imp_type}",
                    "",
                    f"**Target:** {target}",
                    f"**Effort:** {effort}",
                    "",
                    f"{description}",
                    "",
                    f"**Expected Benefit:** {benefit}",
                    "",
                ]
            )
    else:
        lines.append("No improvement suggestions generated.")

    return lines


def _build_recommendations_section(
    classified_failure_modes: List[Dict[str, Any]],
    architectural_suggestions: List[Dict[str, Any]],
) -> List[str]:
    """Build the recommendations section based on analysis findings."""
    lines: List[str] = [
        "",
        "---",
        "",
        "## Recommendations",
        "",
        "Based on the analysis above, the following actions are recommended:",
        "",
    ]

    recommendations: List[str] = []
    if any(m.get("severity") == "high" for m in classified_failure_modes):
        recommendations.append("1. **Address high-severity failure modes** before adding new features")
    if len(architectural_suggestions) > 3:
        recommendations.append("2. **Prioritize low-effort improvements** for quick wins")
    if not recommendations:
        recommendations.append("1. **Maintain current practices** - no critical issues identified")

    lines.extend(recommendations)
    lines.extend(["", "---", "", "*Generated by AURA Multi-Agent Workflow*"])

    return lines


def compile_summary(python_results: Dict[str, Any], typescript_results: Dict[str, Any], execution_plan: Dict[str, Any], classified_failure_modes: List[Dict[str, Any]], architectural_suggestions: List[Dict[str, Any]]) -> str:
    """Compile and format the summary content into markdown.

    Creates a comprehensive architecture summary document combining results
    from all analysis phases into a well-structured markdown report.

    Args:
        python_results: Results from PythonAgent.analyze_architecture()
        typescript_results: Results from TypeScriptAgent.assess_capabilities()
        execution_plan: Execution plan from CodeSearchAgent
        classified_failure_modes: Classified failure modes from classify_failure_modes()
        architectural_suggestions: Improvement suggestions from
            suggest_architectural_improvements()

    Returns:
        str: Formatted markdown content for the architecture summary
    """
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: List[str] = []
    lines.extend(_build_header_section(timestamp))
    lines.extend(_build_python_analysis_section(python_results))
    lines.extend(_build_typescript_section(typescript_results))
    lines.extend(_build_execution_plan_section(execution_plan))
    lines.extend(_build_failure_modes_section(classified_failure_modes))
    lines.extend(_build_suggestions_section(architectural_suggestions))
    lines.extend(_build_recommendations_section(classified_failure_modes, architectural_suggestions))

    return "\n".join(lines)


def validate_with_mcp_server(summary_content: str) -> Dict[str, Any]:
    """Validate the summary content with an MCP server.

    Attempts to connect to a validation MCP server and submit the summary
    for external validation. Falls back to local validation if MCP server
    is unavailable.

    Args:
        summary_content: The markdown summary content to validate.

    Returns:
        Dictionary containing validation status, score, feedback, and
        any errors encountered during validation.
    """
    validation_result = {"status": "pending", "score": 0.0, "feedback": [], "errors": [], "timestamp": datetime.utcnow().isoformat()}

    try:
        # Attempt to use MCP client for validation
        from core.mcp_client import MCPAsyncClient

        # Try to connect to a validation MCP server (default port 8008)
        client = MCPAsyncClient(base_url="http://localhost:8008", timeout=10)

        # Check server health
        import asyncio

        health = asyncio.run(client.get_health())

        if health.get("status") == "healthy":
            # Submit content for validation
            response = asyncio.run(client.call_tool("validate_document", {"content": summary_content, "type": "architecture_summary"}))

            validation_result["status"] = "validated"
            validation_result["score"] = response.get("score", 0.0)
            validation_result["feedback"] = response.get("feedback", [])
            log_json("INFO", "mcp_validation_success", {"score": validation_result["score"]})
        else:
            validation_result["status"] = "fallback"
            validation_result["errors"].append("MCP server not healthy, using fallback validation")

    except Exception as e:
        validation_result["status"] = "fallback"
        validation_result["errors"].append(f"MCP validation failed: {str(e)}")
        log_json("WARN", "mcp_validation_failed", {"error": str(e)})

    # Fallback: Perform basic local validation
    if validation_result["status"] == "fallback":
        validation_result = _perform_fallback_validation(summary_content, validation_result)

    return validation_result


def _perform_fallback_validation(summary_content: str, partial_result: Dict[str, Any]) -> Dict[str, Any]:
    """Perform basic local validation of summary content.

    Args:
        summary_content: The markdown content to validate.
        partial_result: Partial validation result to extend.

    Returns:
        Updated validation result with local validation scores.
    """
    content_length = len(summary_content)
    has_headers = summary_content.count("##") > 0
    has_tables = "|" in summary_content
    has_recommendations = "Recommendations" in summary_content

    # Calculate basic quality score
    score = 0.5  # Base score
    if content_length > 1000:
        score += 0.2
    if has_headers:
        score += 0.1
    if has_tables:
        score += 0.1
    if has_recommendations:
        score += 0.1

    partial_result["score"] = round(min(score, 1.0), 2)
    partial_result["feedback"] = [f"Content length: {content_length} characters", f"Has proper headers: {has_headers}", f"Contains tables: {has_tables}", f"Includes recommendations: {has_recommendations}", "Note: This is fallback validation (MCP server unavailable)"]

    log_json("INFO", "fallback_validation_completed", {"score": partial_result["score"]})
    return partial_result


def engage_stakeholders(validation_results: Dict[str, Any]) -> None:
    """Process for engaging stakeholders with validation results.

    Logs stakeholder engagement activities and generates notifications
    based on the validation results. In a production environment, this
    might send emails, create tickets, or post to collaboration tools.

    Args:
        validation_results: Dictionary containing validation status,
            score, and feedback from validate_with_mcp_server().
    """
    status = validation_results.get("status", "unknown")
    score = validation_results.get("score", 0.0)
    feedback = validation_results.get("feedback", [])

    log_json("INFO", "stakeholder_engagement_started", {"validation_status": status, "score": score})

    # Determine engagement level based on validation score
    if score >= 0.8:
        engagement_level = "inform"
        message = "Architecture summary validated successfully with high confidence."
    elif score >= 0.5:
        engagement_level = "review"
        message = "Architecture summary requires stakeholder review before proceeding."
    else:
        engagement_level = "block"
        message = "Architecture summary validation failed - requires immediate attention."

    # Log engagement action
    log_json("INFO", "stakeholder_engagement_action", {"level": engagement_level, "message": message, "feedback_count": len(feedback)})

    # Simulate notification (in production, this would send actual notifications)
    notification = {"timestamp": datetime.utcnow().isoformat(), "level": engagement_level, "subject": f"AURA Architecture Analysis - {engagement_level.upper()}", "message": message, "validation_score": score, "action_required": engagement_level in ("review", "block")}

    # Log the notification for audit trail
    log_json("INFO", "stakeholder_notification_prepared", notification)
