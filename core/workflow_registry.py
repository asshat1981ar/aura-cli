from typing import Dict, Optional

from core.workflow_models import WorkflowDefinition, WorkflowStep


class WorkflowRegistry:
    """Central registry for workflow definitions."""
    _workflows: Dict[str, WorkflowDefinition] = {}

    @classmethod
    def register(cls, workflow: WorkflowDefinition) -> None:
        """Register a workflow definition."""
        cls._workflows[workflow.name] = workflow

    @classmethod
    def get(cls, name: str) -> Optional[WorkflowDefinition]:
        """Retrieve a workflow by name."""
        return cls._workflows.get(name)

    @classmethod
    def list_all(cls) -> Dict[str, WorkflowDefinition]:
        """Return all registered workflows."""
        return cls._workflows


def register_builtin_workflows() -> None:
    """Register default workflows (e.g., 'code_review')."""
    # 1. Code Review Workflow
    #    Steps:
    #      a. git_diff -> list changed files
    #      b. filter_files -> keep only src/test files
    #      c. lint_check -> run lint on those files
    #      d. summarize -> use LLM to summarize changes
    cr_steps = [
        WorkflowStep(
            name="get_diff",
            skill_name="git_diff",  # pseudo-skill; assumes "git_diff" is registered
            static_inputs={"format": "stat"},
        ),
        WorkflowStep(
            name="lint_check",
            skill_name="linter",
            inputs_from={"files": "get_diff.files"},
        ),
        WorkflowStep(
            name="summarize",
            skill_name="llm_summarizer",
            inputs_from={"diff": "get_diff.diff_content", "lint_errors": "lint_check.errors"},
        ),
    ]
    WorkflowRegistry.register(
        WorkflowDefinition(
            name="code_review",
            steps=cr_steps,
            description="Analyze diffs and run linters on changed files.",
        )
    )

    # 2. Security Scan
    #    Steps:
    #      a. scan_dependencies -> safety check
    #      b. static_analysis -> bandit/semgrep
    #      c. report -> aggregate results
    sec_steps = [
        WorkflowStep(name="dep_scan", skill_name="dependency_scanner"),
        WorkflowStep(name="code_scan", skill_name="static_analyzer"),
        WorkflowStep(
            name="report",
            skill_name="reporter",
            inputs_from={
                "dep_issues": "dep_scan.issues",
                "code_issues": "code_scan.issues",
            },
        ),
    ]
    WorkflowRegistry.register(
        WorkflowDefinition(
            name="security_scan",
            steps=sec_steps,
            description="Check dependencies and code for vulnerabilities.",
        )
    )

    # 3. Security audit (lightweight alias for legacy tests)
    audit_steps = [
        WorkflowStep(name="collect_findings", skill_name="security_collector"),
        WorkflowStep(name="summarize_findings", skill_name="security_reporter",
                     inputs_from={"findings": "collect_findings.*"}),
    ]
    WorkflowRegistry.register(
        WorkflowDefinition(
            name="security_audit",
            steps=audit_steps,
            description="Aggregate security signals and summarize findings.",
        )
    )
