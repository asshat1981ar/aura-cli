"""Build a synthesized PR review artifact for GitHub Actions."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.github_automation import (
    PRContext,
    ProviderReview,
    ProviderRouter,
    ReviewFinding,
    ReviewSynthesizer,
    evaluate_policy,
)


def _changed_files(base_ref: str) -> list[str]:
    try:
        if base_ref:
            subprocess.run(["git", "fetch", "origin", base_ref, "--depth=1"], check=False)
            result = subprocess.run(
                ["git", "diff", "--name-only", f"origin/{base_ref}...HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD~1"],
                check=True,
                capture_output=True,
                text=True,
            )
    except subprocess.CalledProcessError:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _load_context_from_input_path(path: str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    pull_request = payload.get("pull_request", {})
    return {
        "number": pull_request.get("number"),
        "title": pull_request.get("title", ""),
        "base_ref": pull_request.get("base_ref", "main"),
        "head_ref": pull_request.get("head_ref", ""),
        "draft": bool(pull_request.get("draft")),
        "labels": [str(label) for label in pull_request.get("labels", []) if label],
        "changed_files": [str(path) for path in pull_request.get("changed_files", []) if path],
    }


def _build_findings(context: PRContext) -> list[ReviewFinding]:
    findings: list[ReviewFinding] = []
    if context.touches_workflows:
        findings.append(
            ReviewFinding(
                severity="medium",
                path=".github/workflows/",
                line=None,
                title="Workflow changes require manual review",
                detail="GitHub workflow changes can alter CI, permissions, or automation behavior.",
                confidence=0.96,
                category="maintainability",
            )
        )
    if context.touches_dependencies:
        dependency_path = next(
            (
                path
                for path in context.changed_files
                if Path(path).name in {"requirements.txt", "pyproject.toml", "package.json", "package-lock.json", "poetry.lock"}
            ),
            "requirements.txt",
        )
        findings.append(
            ReviewFinding(
                severity="medium",
                path=dependency_path,
                line=None,
                title="Dependency manifest changed",
                detail="Dependency updates should confirm lockfile, install, and CI behavior.",
                confidence=0.90,
                category="maintainability",
            )
        )
    if context.touches_core and not context.touches_tests:
        findings.append(
            ReviewFinding(
                severity="low",
                path="tests/",
                line=None,
                title="Core code changed without touching tests",
                detail="Consider adding or updating tests for changed core paths.",
                confidence=0.72,
                category="test",
            )
        )
    if context.touched_file_count >= 12:
        findings.append(
            ReviewFinding(
                severity="low",
                path=".",
                line=None,
                title="Large change set",
                detail="Large PRs are harder to review and may benefit from splitting or extra reviewer attention.",
                confidence=0.68,
                category="maintainability",
            )
        )
    return findings


def _write_outputs(synthesis_json: Path, comment_md: Path, synthesis) -> None:
    synthesis_json.write_text(json.dumps(synthesis.to_dict(), indent=2) + "\n", encoding="utf-8")
    comment_md.write_text(synthesis.comment_markdown + "\n", encoding="utf-8")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        return

    with open(github_output, "a", encoding="utf-8") as handle:
        handle.write(f"recommended_action={synthesis.recommended_action}\n")
        handle.write(f"human_review_required={'true' if synthesis.human_review_required else 'false'}\n")
        handle.write(f"providers={','.join(synthesis.providers_consulted)}\n")


def main() -> None:
    input_path = os.environ.get("PR_CONTEXT_INPUT_PATH", "").strip()
    if input_path:
        payload = _load_context_from_input_path(input_path)
        changed_files = payload["changed_files"]
        base_ref = str(payload["base_ref"] or "main")
        context = PRContext.from_changed_files(
            changed_files,
            number=int(payload["number"]) if payload.get("number") is not None else None,
            title=str(payload["title"] or ""),
            base_ref=base_ref,
            head_ref=str(payload["head_ref"] or ""),
            draft=bool(payload["draft"]),
            labels=list(payload["labels"]),
        )
    else:
        base_ref = os.environ.get("BASE_REF", "main")
        changed_files_json = os.environ.get("PR_CHANGED_FILES_JSON", "").strip()
        changed_files = json.loads(changed_files_json) if changed_files_json else _changed_files(base_ref)
        context = PRContext.from_changed_files(
            changed_files,
            number=int(os.environ["PR_NUMBER"]) if os.environ.get("PR_NUMBER") else None,
            title=os.environ.get("PR_TITLE", ""),
            base_ref=base_ref,
            head_ref=os.environ.get("HEAD_REF", ""),
            draft=os.environ.get("PR_DRAFT", "false").lower() == "true",
            labels=[label for label in os.environ.get("PR_LABELS", "").split(",") if label],
        )

    router = ProviderRouter()
    providers = router.select_providers(context)
    route_summary = router.route_summary(context)
    findings = _build_findings(context)

    aura_review = ProviderReview(
        provider="aura",
        summary=f"Synthesized deterministic review. {route_summary}",
        findings=findings,
        recommended_action="comment" if findings else "approve",
        artifacts={
            "changed_files": context.changed_files,
            "route_summary": route_summary,
        },
    )
    decision = evaluate_policy(context, [aura_review])
    synthesis = ReviewSynthesizer().synthesize(context, [aura_review], decision, planned_providers=providers)
    _write_outputs(Path("pr-review-summary.json"), Path("pr-review-comment.md"), synthesis)


if __name__ == "__main__":
    main()
