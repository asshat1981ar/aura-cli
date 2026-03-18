"""Helpers for GitHub issue and pull-request automation."""

from core.github_automation.agent_dispatch import AgentDispatchPlan, AgentDispatchPlanner, slugify
from core.github_automation.agent_profiles import AgentProfile, default_agent_profiles
from core.github_automation.command_router import CommandRequest, CommandResult, CommandRouter, parse_slash_command
from core.github_automation.issue_planner import IssuePlan, IssuePlanner
from core.github_automation.issue_triage import IssueContext, IssueTriageResult, parse_issue_sections, triage_issue
from core.github_automation.merge_advisor import (
    CheckRunStatus,
    CodeownersRule,
    MergeAdvisor,
    MergeReadinessDecision,
    ReviewGateStatus,
    ReviewSynthesisSnapshot,
    codeowners_required,
    load_codeowners,
    parse_codeowners,
)
from core.github_automation.models import (
    PolicyDecision,
    ProviderReview,
    ReviewFinding,
    SynthesisResult,
)
from core.github_automation.policy import ReviewPolicy, evaluate_policy
from core.github_automation.pr_context import PRContext
from core.github_automation.provider_router import ProviderRouter
from core.github_automation.repo_health import (
    FailureCluster,
    FlakyWorkflow,
    FollowUpIssue,
    PathHotspot,
    PullRequestSnapshot,
    RepoHealthAnalyzer,
    RepoHealthReport,
    StalePullRequest,
    WorkflowRunSnapshot,
)
from core.github_automation.review_synthesizer import ReviewSynthesizer

__all__ = [
    "AgentDispatchPlan",
    "AgentDispatchPlanner",
    "AgentProfile",
    "CommandRequest",
    "CommandResult",
    "CommandRouter",
    "CodeownersRule",
    "CheckRunStatus",
    "IssueContext",
    "IssuePlan",
    "IssuePlanner",
    "IssueTriageResult",
    "MergeAdvisor",
    "MergeReadinessDecision",
    "PRContext",
    "PolicyDecision",
    "ProviderReview",
    "ProviderRouter",
    "ReviewGateStatus",
    "ReviewFinding",
    "ReviewPolicy",
    "ReviewSynthesisSnapshot",
    "ReviewSynthesizer",
    "SynthesisResult",
    "codeowners_required",
    "default_agent_profiles",
    "evaluate_policy",
    "FailureCluster",
    "FlakyWorkflow",
    "FollowUpIssue",
    "load_codeowners",
    "PathHotspot",
    "parse_issue_sections",
    "parse_codeowners",
    "parse_slash_command",
    "PullRequestSnapshot",
    "RepoHealthAnalyzer",
    "RepoHealthReport",
    "slugify",
    "StalePullRequest",
    "triage_issue",
    "WorkflowRunSnapshot",
]
