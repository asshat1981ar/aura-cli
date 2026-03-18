"""Handle slash commands from GitHub issue_comment events."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.github_automation.command_router import CommandRouter
from core.github_automation.issue_triage import IssueContext


def _load_payload() -> tuple[IssueContext, dict]:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path or not Path(event_path).exists():
        raise RuntimeError("GITHUB_EVENT_PATH is required for issue comment command handling.")
    payload = json.loads(Path(event_path).read_text(encoding="utf-8"))
    return IssueContext.from_github_event(payload), payload


def _write_outputs(result: dict[str, object]) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        return
    with open(github_output, "a", encoding="utf-8") as handle:
        handle.write(f"recognized={'true' if result['recognized'] else 'false'}\n")
        handle.write(f"action={result['action']}\n")
        handle.write(f"labels_json={json.dumps(result['labels_to_add'])}\n")
        handle.write(f"should_queue={'true' if result['should_queue'] else 'false'}\n")
        handle.write(f"queue_goal={result['queue_goal']}\n")
        handle.write(f"requested_provider={result['requested_provider']}\n")


def main() -> None:
    context, payload = _load_payload()
    comment = payload.get("comment", {})
    comment_id = comment.get("id", "unknown")
    marker = f"<!-- aura-command-response:{comment_id} -->"

    result = CommandRouter(project_root=ROOT).handle(context, comment.get("body", ""), response_marker=marker)
    Path("issue-command-summary.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    if result.comment_markdown:
        Path("issue-command-comment.md").write_text(result.comment_markdown + "\n", encoding="utf-8")
    _write_outputs(result.to_dict())


if __name__ == "__main__":
    main()
