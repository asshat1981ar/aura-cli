"""Build nightly repo-health artifacts from serialized GitHub data."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.github_automation.repo_health import (  # noqa: E402
    PullRequestSnapshot,
    RepoHealthAnalyzer,
    WorkflowRunSnapshot,
)


def _load_payload() -> dict[str, object]:
    input_path = os.environ.get("REPO_HEALTH_INPUT_PATH", "").strip()
    if input_path:
        return json.loads(Path(input_path).read_text(encoding="utf-8"))
    raw = os.environ.get("REPO_HEALTH_INPUT_JSON", "").strip()
    if raw:
        return json.loads(raw)
    return {"pull_requests": [], "workflow_runs": []}


def _load_now() -> datetime:
    value = os.environ.get("REPO_HEALTH_NOW", "").strip()
    if not value:
        return datetime.now(timezone.utc)
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _write_outputs(report) -> None:
    Path("repo-health-summary.json").write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    Path("repo-health-comment.md").write_text(report.render_markdown() + "\n", encoding="utf-8")
    Path("repo-health-followups.json").write_text(
        json.dumps([issue.to_dict() for issue in report.follow_up_issues], indent=2) + "\n",
        encoding="utf-8",
    )

    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        return

    with open(github_output, "a", encoding="utf-8") as handle:
        handle.write(f"stale_pull_requests={len(report.stale_pull_requests)}\n")
        handle.write(f"flaky_workflows={len(report.flaky_workflows)}\n")
        handle.write(f"recurring_failures={len(report.recurring_failures)}\n")
        handle.write(f"follow_up_issues={len(report.follow_up_issues)}\n")


def main() -> None:
    payload = _load_payload()
    analyzer = RepoHealthAnalyzer(now=_load_now())
    pull_requests = [PullRequestSnapshot.from_dict(item) for item in payload.get("pull_requests", [])]
    workflow_runs = [WorkflowRunSnapshot.from_dict(item) for item in payload.get("workflow_runs", [])]
    report = analyzer.build_report(pull_requests, workflow_runs)
    _write_outputs(report)


if __name__ == "__main__":
    main()
