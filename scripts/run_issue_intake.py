"""Run deterministic issue intake planning for GitHub issue workflows."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.github_automation.issue_planner import IssuePlanner
from core.github_automation.issue_triage import IssueContext, triage_issue
from core.workflow_engine import WorkflowEngine

COMMENT_MARKER = "<!-- aura-issue-plan -->"


def _load_context() -> IssueContext:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_path and Path(event_path).exists():
        payload = json.loads(Path(event_path).read_text(encoding="utf-8"))
        if "issue" in payload:
            return IssueContext.from_github_event(payload)
    labels = [label for label in os.environ.get("ISSUE_LABELS", "").split(",") if label]
    return IssueContext(
        number=int(os.environ["ISSUE_NUMBER"]) if os.environ.get("ISSUE_NUMBER") else None,
        title=os.environ.get("ISSUE_TITLE", ""),
        body=os.environ.get("ISSUE_BODY", ""),
        labels=labels,
        author=os.environ.get("ISSUE_AUTHOR", ""),
        is_pull_request=False,
    )


def _write_outputs(*, labels: list[str], recommended_provider: str, queue_goal: str, should_queue: bool) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        return
    with open(github_output, "a", encoding="utf-8") as handle:
        handle.write(f"labels_json={json.dumps(labels)}\n")
        handle.write(f"recommended_provider={recommended_provider}\n")
        handle.write(f"queue_goal={queue_goal}\n")
        handle.write(f"should_queue={'true' if should_queue else 'false'}\n")


def main() -> None:
    context = _load_context()
    engine = WorkflowEngine()
    exec_id = engine.run_workflow("issue_intake_planning", context.to_workflow_inputs())
    triage_output = engine.get_step_output(exec_id, "triage_issue")
    plan_output = engine.get_step_output(exec_id, "plan_issue")

    Path("issue-plan-summary.json").write_text(
        json.dumps({"triage": triage_output, "plan": plan_output}, indent=2) + "\n",
        encoding="utf-8",
    )

    plan_markdown = plan_output.get("comment_markdown", "")
    if COMMENT_MARKER not in plan_markdown:
        plan_markdown = IssuePlanner(project_root=ROOT).plan(context, triage_issue(context)).render_markdown(marker=COMMENT_MARKER)
    Path("issue-plan-comment.md").write_text(plan_markdown + "\n", encoding="utf-8")

    labels = plan_output.get("labels_to_apply", triage_output.get("labels_to_apply", []))
    recommended_provider = plan_output.get("recommended_provider", triage_output.get("preferred_provider", "no preference"))
    queue_goal = plan_output.get("queue_goal", triage_output.get("queue_goal", context.title))
    should_queue = bool(triage_output.get("queue_candidate")) and recommended_provider in {"aura", "no preference"}
    _write_outputs(
        labels=list(labels),
        recommended_provider=str(recommended_provider),
        queue_goal=str(queue_goal),
        should_queue=should_queue,
    )


if __name__ == "__main__":
    main()
