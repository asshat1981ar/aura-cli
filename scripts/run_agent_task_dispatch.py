"""Build a deterministic coding-agent dispatch plan for an issue or manual request."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.github_automation.agent_dispatch import AgentDispatchPlanner
from core.github_automation.issue_triage import IssueContext


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


def _write_outputs(dispatch: dict[str, object]) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        return
    with open(github_output, "a", encoding="utf-8") as handle:
        handle.write(f"provider={dispatch['provider']}\n")
        handle.write(f"profile_name={dispatch['profile_name']}\n")
        handle.write(f"branch_name={dispatch['branch_name']}\n")
        handle.write(f"queue_goal={dispatch['queue_goal']}\n")
        handle.write(f"labels_json={json.dumps(dispatch['labels_to_apply'])}\n")
        handle.write(f"human_review_required={'true' if dispatch['human_review_required'] else 'false'}\n")


def main() -> None:
    context = _load_context()
    preferred_provider = os.environ.get("REQUESTED_PROVIDER") or None
    requested_profile = os.environ.get("REQUESTED_PROFILE") or None
    base_branch = os.environ.get("BASE_BRANCH", "main")

    planner = AgentDispatchPlanner(project_root=str(ROOT))
    dispatch = planner.dispatch(
        context,
        preferred_provider=preferred_provider,
        requested_profile=requested_profile,
        base_branch=base_branch,
    )
    Path("agent-dispatch-summary.json").write_text(
        json.dumps(dispatch.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    Path("agent-dispatch-comment.md").write_text(
        dispatch.render_markdown() + "\n",
        encoding="utf-8",
    )
    Path("patch-metadata.json").write_text(
        json.dumps(
            {
                "provider": dispatch.provider,
                "profile_name": dispatch.profile_name,
                "branch_name": dispatch.branch_name,
                "base_branch": dispatch.base_branch,
                "requires_pr": dispatch.requires_pr,
                "direct_main_allowed": dispatch.direct_main_allowed,
                "human_review_required": dispatch.human_review_required,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    _write_outputs(dispatch.to_dict())


if __name__ == "__main__":
    main()
