from __future__ import annotations

import importlib.util
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module(relative_path: str, module_name: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_issue_intake_outputs_use_multiline_syntax_for_queue_goals(tmp_path, monkeypatch) -> None:
    module = _load_module("scripts/run_issue_intake.py", "run_issue_intake_test")
    output_path = tmp_path / "github-output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    module._write_outputs(
        labels=["bug"],
        recommended_provider="aura",
        queue_goal="Line one\nLine two",
        should_queue=True,
    )

    contents = output_path.read_text(encoding="utf-8")
    assert "labels_json=[\"bug\"]\n" in contents
    assert "recommended_provider=aura\n" in contents
    assert "queue_goal<<__AURA_EOF__\nLine one\nLine two\n__AURA_EOF__\n" in contents
    assert "should_queue=true\n" in contents


def test_issue_intake_workflow_queues_goal_via_environment_variable() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "issue-intake.yml").read_text(encoding="utf-8")

    assert "ISSUE_QUEUE_GOAL: ${{ steps.intake.outputs.queue_goal }}" in workflow
    assert 'python3 main.py --add-goal "$ISSUE_QUEUE_GOAL"' in workflow


def test_merge_readiness_keeps_codeowners_pending_without_matching_approval() -> None:
    module = _load_module("scripts/run_merge_readiness.py", "run_merge_readiness_test_pending")
    payload = {
        "reviews": {
            "approvals": 1,
            "changes_requested": 0,
            "review_decision": "APPROVED",
            "approved_reviewers": ["someone-else"],
            "requested_reviewers": [],
            "requested_teams": [],
        }
    }

    reviews = module._build_reviews(payload, ["core/workflow_engine.py"])

    assert reviews.codeowners_required is True
    assert reviews.codeowners_pending is True


def test_merge_readiness_clears_codeowners_pending_after_codeowner_approval() -> None:
    module = _load_module("scripts/run_merge_readiness.py", "run_merge_readiness_test_approved")
    payload = {
        "reviews": {
            "approvals": 1,
            "changes_requested": 0,
            "review_decision": "APPROVED",
            "approved_reviewers": ["asshat1981ar"],
            "requested_reviewers": [],
            "requested_teams": [],
        }
    }

    reviews = module._build_reviews(payload, ["core/workflow_engine.py"])

    assert reviews.codeowners_required is True
    assert reviews.codeowners_pending is False
