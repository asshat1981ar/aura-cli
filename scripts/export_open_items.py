#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = ROOT / "tmp"
PRS_PATH = TMP_DIR / "open_prs.json"
ISSUES_PATH = TMP_DIR / "open_issues.json"


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def parse_remote_repo() -> tuple[str | None, str | None, str | None]:
    """Return owner, repo, and source note."""
    try:
        remotes = run(["git", "remote", "-v"]).splitlines()
    except subprocess.CalledProcessError:
        remotes = []

    pattern = re.compile(r"github\.com[:/]([^/]+)/([^/.\s]+?)(?:\.git)?\s")
    for line in remotes:
        m = pattern.search(line + " ")
        if m:
            return m.group(1), m.group(2), "git remote -v"

    # Fallback: project docs reference canonical repo.
    prd = ROOT / "docs" / "PRD.md"
    if prd.exists():
        text = prd.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"github\.com/([^/]+)/([^)/\s]+)", text)
        if m:
            return m.group(1), m.group(2), "docs/PRD.md fallback"

    return None, None, "unresolved"


def gh_get(url: str) -> list | dict:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "aura-cli-triage-script",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read()
    return json.loads(body.decode("utf-8"))


def iso_to_dt(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def age_days(updated_at: str | None) -> int:
    return (datetime.now(timezone.utc) - iso_to_dt(updated_at)).days


def infer_pr_triage(pr: dict, check_rollup: dict) -> str:
    days = age_days(pr.get("updated_at"))
    merge_state = pr.get("mergeable_state")
    draft = bool(pr.get("draft"))

    if days >= 120:
        return "close-candidate"
    if merge_state in {"behind", "dirty"}:
        return "needs-rebase"
    if draft or merge_state in {"blocked", "unknown"}:
        return "blocked"

    if check_rollup.get("failed", 0) > 0:
        return "blocked"
    if days >= 30:
        return "stale"
    return "ready"


def infer_issue_triage(issue: dict) -> str:
    days = age_days(issue.get("updated_at"))
    labels = [lbl.get("name", "").lower() for lbl in issue.get("labels", []) if isinstance(lbl, dict)]
    assignee = issue.get("assignee", {}) or {}
    has_assignee = bool(assignee.get("login"))

    if any(k in labels for k in ["blocked", "on hold", "needs-info", "wip"]):
        return "blocked"
    if days >= 180 and not has_assignee:
        return "close-candidate"
    if days >= 45:
        return "stale"
    return "ready"


def collect_checks(owner: str, repo: str, head_sha: str) -> dict:
    status_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{head_sha}/status"
    checks_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{head_sha}/check-runs?per_page=100"

    combined = {"state": "unknown", "total_count": 0}
    rollup = {"total": 0, "passed": 0, "failed": 0, "pending": 0}

    try:
        status = gh_get(status_url)
        combined = {
            "state": status.get("state", "unknown"),
            "total_count": status.get("total_count", 0),
        }
    except Exception:
        pass

    try:
        check_runs = gh_get(checks_url)
        runs = check_runs.get("check_runs", [])
        for run in runs:
            rollup["total"] += 1
            conclusion = run.get("conclusion")
            status = run.get("status")
            if conclusion == "success":
                rollup["passed"] += 1
            elif conclusion in {"failure", "timed_out", "cancelled", "action_required", "startup_failure"}:
                rollup["failed"] += 1
            elif status != "completed" or conclusion is None:
                rollup["pending"] += 1
    except Exception:
        pass

    return {"combined_status": combined, "check_runs": rollup}


def main() -> int:
    owner, repo, repo_source = parse_remote_repo()
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    if not owner or not repo:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "repository": {"owner": owner, "name": repo, "source": repo_source},
            "error": "Unable to resolve GitHub owner/repo from git remotes or fallback metadata.",
            "items": [],
        }
        PRS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        ISSUES_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("Repository unresolved; wrote empty outputs.")
        return 1

    pulls_url = f"https://api.github.com/repos/{owner}/{repo}/pulls?state=open&per_page=100"
    issues_url = f"https://api.github.com/repos/{owner}/{repo}/issues?state=open&per_page=100"

    pulls = gh_get(pulls_url)
    issues = gh_get(issues_url)

    pr_items = []
    for pr in pulls:
        number = pr.get("number")
        detail_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}"
        try:
            detail = gh_get(detail_url)
        except Exception:
            detail = pr

        head_sha = (detail.get("head") or {}).get("sha")
        checks = collect_checks(owner, repo, head_sha) if head_sha else {"combined_status": {}, "check_runs": {}}
        labels = [lbl.get("name") for lbl in detail.get("labels", []) if isinstance(lbl, dict)]

        item = {
            "number": number,
            "title": detail.get("title"),
            "author": (detail.get("user") or {}).get("login"),
            "base": (detail.get("base") or {}).get("ref"),
            "head": (detail.get("head") or {}).get("ref"),
            "draft": bool(detail.get("draft")),
            "mergeable_state": detail.get("mergeable_state"),
            "updated_at": detail.get("updated_at"),
            "labels": labels,
            "checks": checks,
        }
        item["triage_status"] = infer_pr_triage(item, checks.get("check_runs", {}))
        pr_items.append(item)

    issue_items = []
    for issue in issues:
        if issue.get("pull_request"):
            continue
        labels = [lbl.get("name") for lbl in issue.get("labels", []) if isinstance(lbl, dict)]
        assignee = issue.get("assignee") or {}
        item = {
            "number": issue.get("number"),
            "title": issue.get("title"),
            "labels": labels,
            "assignee": assignee.get("login"),
            "updated_at": issue.get("updated_at"),
            "linked_pr": (issue.get("pull_request") or {}).get("html_url"),
        }
        item["triage_status"] = infer_issue_triage(issue)
        issue_items.append(item)

    pr_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository": {"owner": owner, "name": repo, "source": repo_source},
        "source": {"pulls": pulls_url},
        "items": pr_items,
    }
    issue_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository": {"owner": owner, "name": repo, "source": repo_source},
        "source": {"issues": issues_url},
        "items": issue_items,
    }

    PRS_PATH.write_text(json.dumps(pr_payload, indent=2) + "\n", encoding="utf-8")
    ISSUES_PATH.write_text(json.dumps(issue_payload, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(pr_items)} PRs to {PRS_PATH}")
    print(f"Wrote {len(issue_items)} issues to {ISSUES_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
