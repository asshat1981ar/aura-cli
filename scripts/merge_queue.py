#!/usr/bin/env python3
"""Build and execute a dependency-aware GitHub merge queue.

Workflow implemented:
1. Build queue ordered by smallest diff first, then oldest updated PR within risk band,
   while honoring explicit dependencies.
2. Classify mergeability (`clean`, `dirty`, `blocked`, `unstable`) from API data.
3. For `needs-rebase`/`dirty` PRs, attempt a local rebase onto latest base branch.
4. Merge only PRs that are clean, checks-passed, and approvals-satisfied.
5. Recompute queue and mergeability after each merge.

This script uses GitHub REST API and local git CLI. It requires:
- GITHUB_PAT (or GITHUB_TOKEN) environment variable
- A clean local checkout of the target repo
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests

API_BASE = "https://api.github.com"
DEPENDENCY_PATTERNS = [
    re.compile(r"depends[-_ ]on\s*:\s*#(\d+)", re.IGNORECASE),
    re.compile(r"requires\s*#(\d+)", re.IGNORECASE),
    re.compile(r"blocked\s+by\s*#(\d+)", re.IGNORECASE),
]


@dataclass
class PullRequestState:
    number: int
    title: str
    head_ref: str
    base_ref: str
    updated_at: str
    additions: int
    deletions: int
    changed_files: int
    draft: bool
    mergeable: Optional[bool]
    mergeable_state: str
    requested_reviewers: int
    dependency_prs: Set[int] = field(default_factory=set)
    approvals: int = 0
    required_checks: List[str] = field(default_factory=list)
    checks_passed: bool = False
    pending_checks: List[str] = field(default_factory=list)
    failed_checks: List[str] = field(default_factory=list)
    mergeability: str = "blocked"
    base_ahead_by: int = 0
    head_ahead_by: int = 0

    @property
    def diff_size(self) -> int:
        return self.additions + self.deletions

    @property
    def risk_band(self) -> int:
        files = self.changed_files
        if files <= 5:
            return 0
        if files <= 20:
            return 1
        return 2


class MergeQueueManager:
    def __init__(self, repo: str, token: str, execute: bool = False, require_approvals: int = 1):
        self.repo = repo
        self.execute = execute
        self.require_approvals = require_approvals
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self.session.request(method, f"{API_BASE}{path}", timeout=45, **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(f"GitHub API {method} {path} failed ({response.status_code}): {response.text[:300]}")
        if response.status_code == 204:
            return None
        return response.json()

    def fetch_open_prs(self) -> List[Dict[str, Any]]:
        prs: List[Dict[str, Any]] = []
        page = 1
        while True:
            batch = self._request(
                "GET",
                f"/repos/{self.repo}/pulls",
                params={"state": "open", "per_page": 100, "page": page, "sort": "updated", "direction": "asc"},
            )
            if not batch:
                break
            prs.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return prs

    def _parse_dependencies(self, body: str) -> Set[int]:
        deps: Set[int] = set()
        for pattern in DEPENDENCY_PATTERNS:
            deps.update(int(match) for match in pattern.findall(body or ""))
        return deps

    def _required_contexts(self, base_branch: str) -> List[str]:
        try:
            data = self._request("GET", f"/repos/{self.repo}/branches/{base_branch}/protection")
        except RuntimeError:
            return []
        contexts = data.get("required_status_checks", {}).get("contexts", []) or []
        return list(contexts)

    def _approval_count(self, pr_number: int) -> int:
        reviews = self._request("GET", f"/repos/{self.repo}/pulls/{pr_number}/reviews")
        by_user: Dict[str, str] = {}
        for review in reviews:
            user = (review.get("user") or {}).get("login")
            if user:
                by_user[user] = review.get("state", "")
        return sum(1 for state in by_user.values() if state == "APPROVED")

    def _check_rollup(self, head_sha: str, required_contexts: Iterable[str]) -> Tuple[bool, List[str], List[str]]:
        status = self._request("GET", f"/repos/{self.repo}/commits/{head_sha}/status")
        statuses = status.get("statuses", []) or []
        latest_by_context: Dict[str, str] = {}
        for item in statuses:
            ctx = item.get("context")
            if ctx and ctx not in latest_by_context:
                latest_by_context[ctx] = item.get("state", "pending")

        check_runs = self._request("GET", f"/repos/{self.repo}/commits/{head_sha}/check-runs")
        for run in check_runs.get("check_runs", []) or []:
            name = run.get("name")
            if not name:
                continue
            conclusion = run.get("conclusion")
            state = "pending" if conclusion is None else ("success" if conclusion == "success" else "failure")
            latest_by_context.setdefault(name, state)

        required = list(required_contexts)
        pending = [ctx for ctx in required if latest_by_context.get(ctx, "pending") == "pending"]
        failed = [ctx for ctx in required if latest_by_context.get(ctx) in {"failure", "error"}]
        passed = not pending and not failed and bool(required)
        if not required:
            passed = True
        return passed, pending, failed

    def _mergeability_label(self, mergeable: Optional[bool], mergeable_state: str, draft: bool) -> str:
        state = (mergeable_state or "").lower()
        if draft:
            return "blocked"
        if mergeable is False or state in {"dirty", "behind"}:
            return "dirty"
        if state in {"blocked"}:
            return "blocked"
        if state in {"unstable", "unknown"}:
            return "unstable"
        if mergeable is True and state in {"clean", "has_hooks", "unstable"}:
            return "clean" if state != "unstable" else "unstable"
        return "blocked"

    def _drift(self, base_ref: str, head_ref: str) -> Tuple[int, int]:
        compare = self._request("GET", f"/repos/{self.repo}/compare/{base_ref}...{head_ref}")
        return int(compare.get("behind_by", 0)), int(compare.get("ahead_by", 0))

    def collect_state(self) -> Dict[int, PullRequestState]:
        pr_states: Dict[int, PullRequestState] = {}
        for pr in self.fetch_open_prs():
            pr_number = pr["number"]
            detail = self._request("GET", f"/repos/{self.repo}/pulls/{pr_number}")
            base_ref = detail["base"]["ref"]
            required = self._required_contexts(base_ref)
            head_sha = detail["head"]["sha"]
            checks_passed, pending, failed = self._check_rollup(head_sha, required)
            approvals = self._approval_count(pr_number)
            base_ahead, head_ahead = self._drift(base_ref, detail["head"]["ref"])

            state = PullRequestState(
                number=pr_number,
                title=detail["title"],
                head_ref=detail["head"]["ref"],
                base_ref=base_ref,
                updated_at=detail["updated_at"],
                additions=detail.get("additions", 0),
                deletions=detail.get("deletions", 0),
                changed_files=detail.get("changed_files", 0),
                draft=detail.get("draft", False),
                mergeable=detail.get("mergeable"),
                mergeable_state=detail.get("mergeable_state", "unknown"),
                requested_reviewers=len(detail.get("requested_reviewers") or []),
                dependency_prs=self._parse_dependencies(detail.get("body") or ""),
                approvals=approvals,
                required_checks=required,
                checks_passed=checks_passed,
                pending_checks=pending,
                failed_checks=failed,
                base_ahead_by=base_ahead,
                head_ahead_by=head_ahead,
            )
            state.mergeability = self._mergeability_label(state.mergeable, state.mergeable_state, state.draft)
            if state.base_ahead_by > 0 and state.mergeability == "clean":
                state.mergeability = "dirty"
            pr_states[pr_number] = state
        return pr_states

    def _topo_with_priority(self, states: Dict[int, PullRequestState]) -> List[PullRequestState]:
        dependents: Dict[int, Set[int]] = defaultdict(set)
        indegree: Dict[int, int] = {n: 0 for n in states}
        for pr in states.values():
            deps = {d for d in pr.dependency_prs if d in states}
            indegree[pr.number] = len(deps)
            for dep in deps:
                dependents[dep].add(pr.number)

        queue = [pr.number for pr in states.values() if indegree[pr.number] == 0]

        def key(pr_num: int) -> Tuple[int, int, datetime, int]:
            pr = states[pr_num]
            updated = datetime.fromisoformat(pr.updated_at.replace("Z", "+00:00"))
            return (pr.risk_band, pr.diff_size, updated, pr.number)

        ordered: List[int] = []
        while queue:
            queue.sort(key=key)
            current = queue.pop(0)
            ordered.append(current)
            for dep in dependents[current]:
                indegree[dep] -= 1
                if indegree[dep] == 0:
                    queue.append(dep)

        # Cycle fallback: append remaining by priority
        remaining = [n for n in states if n not in set(ordered)]
        ordered.extend(sorted(remaining, key=key))
        return [states[n] for n in ordered]

    def _git(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], check=False, text=True, capture_output=True)

    def _attempt_rebase(self, state: PullRequestState) -> bool:
        print(f"[rebase] PR #{state.number} ({state.head_ref} -> {state.base_ref})")
        steps = [
            ("fetch", ["fetch", "origin", state.base_ref, state.head_ref]),
            ("checkout", ["checkout", state.head_ref]),
            ("rebase", ["rebase", f"origin/{state.base_ref}"]),
        ]
        for label, cmd in steps:
            result = self._git(*cmd)
            if result.returncode != 0:
                if label == "rebase":
                    conflicted = self._git("diff", "--name-only", "--diff-filter=U")
                    conflict_files = [line for line in conflicted.stdout.splitlines() if line.strip()]
                    note = "; ".join(conflict_files) if conflict_files else "unknown files"
                    print(f"  rebase conflict in: {note}")
                    self._git("rebase", "--abort")
                else:
                    print(f"  failed {label}: {result.stderr.strip()}")
                return False
        push = self._git("push", "--force-with-lease", "origin", state.head_ref)
        if push.returncode != 0:
            print(f"  push failed: {push.stderr.strip()}")
            return False
        return True

    def _eligible(self, state: PullRequestState) -> bool:
        approvals_ok = state.approvals >= self.require_approvals
        return state.mergeability == "clean" and state.checks_passed and approvals_ok

    def _merge(self, state: PullRequestState) -> bool:
        payload = {
            "merge_method": "squash",
            "commit_title": f"{state.title} (#{state.number})",
        }
        try:
            self._request("PUT", f"/repos/{self.repo}/pulls/{state.number}/merge", json=payload)
            return True
        except RuntimeError as exc:
            print(f"[merge] PR #{state.number} failed: {exc}")
            return False

    def process(self) -> int:
        merged_count = 0
        iteration = 1
        while True:
            states = self.collect_state()
            if not states:
                print("No open PRs.")
                break

            queue = self._topo_with_priority(states)
            print(f"\n=== Merge queue iteration {iteration} ===")
            for idx, pr in enumerate(queue, start=1):
                print(
                    f"{idx:>2}. #{pr.number} [{pr.mergeability}] risk={pr.risk_band} diff={pr.diff_size} "
                    f"updated={pr.updated_at} checks={'pass' if pr.checks_passed else 'wait'} "
                    f"approvals={pr.approvals} drift(main_ahead={pr.base_ahead_by}, head_ahead={pr.head_ahead_by}) deps={sorted(pr.dependency_prs)}"
                )

            progress = False
            for pr in queue:
                if pr.mergeability == "dirty":
                    if self.execute:
                        rebased = self._attempt_rebase(pr)
                        print(f"[rebase] PR #{pr.number}: {'ok' if rebased else 'failed'}")
                    else:
                        print(f"[plan] PR #{pr.number} needs rebase onto {pr.base_ref}")
                    continue

                if self._eligible(pr):
                    if self.execute:
                        if self._merge(pr):
                            merged_count += 1
                            progress = True
                            print(f"[merge] PR #{pr.number} merged")
                            break  # Recompute queue after each merge.
                    else:
                        print(f"[plan] PR #{pr.number} eligible to merge")
                else:
                    print(
                        f"[skip] PR #{pr.number}: mergeability={pr.mergeability}, "
                        f"checks_passed={pr.checks_passed}, approvals={pr.approvals}/{self.require_approvals}"
                    )

            if not self.execute:
                break
            if not progress:
                print("No further merge progress possible in current state.")
                break
            iteration += 1
        return merged_count


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dependency-aware GitHub merge queue")
    parser.add_argument("--repo", required=True, help="Repository in owner/name form")
    parser.add_argument("--execute", action="store_true", help="Actually perform rebases and merges")
    parser.add_argument("--require-approvals", type=int, default=1, help="Minimum approvals required")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    token = os.getenv("GITHUB_PAT") or os.getenv("GITHUB_TOKEN")
    if not token:
        print("GITHUB_PAT or GITHUB_TOKEN must be set", file=sys.stderr)
        return 2
    manager = MergeQueueManager(args.repo, token, execute=args.execute, require_approvals=args.require_approvals)
    merged = manager.process()
    print(f"Done. merged_count={merged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
