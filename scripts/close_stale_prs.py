#!/usr/bin/env python3
"""Close stale/duplicate open PRs that are superseded by existing fixes.

Usage:
    GITHUB_PAT=<token> python3 scripts/close_stale_prs.py [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import time
import sys

import requests

REPO = "asshat1981ar/aura-cli"
API_BASE = "https://api.github.com"

CLOSE_COMMENT = (
    "Closing as superseded: the dotenv/numpy/optional-import CI issue has been "
    "fixed comprehensively in `main` (via the `_MissingPackage` sentinel approach) "
    "and in PR #169 (`claude/fix-build-errors-ePrZJ`). "
    "Thank you for the contribution!"
)

# PRs to close: duplicate import-guard fixes, stale WIP PRs, and PRs with wrong base branches.
# These have all been superseded by main's existing fixes or PR #169.
CLOSE_PR_NUMBERS = [
    # Empty / zero-file WIP PRs
    199, 201, 200, 197, 183, 182, 181, 184, 185, 187, 179, 178, 170,
    # Wrong-base PRs (targeting stale feature branches, not main)
    196, 195, 159, 154, 153,
    # Duplicate dotenv/numpy/import-guard CI fixes (superseded by main + PR #169)
    192, 191, 190, 189, 188, 166, 165, 163, 162, 161, 160, 158, 157, 156, 152,
    144, 143, 142, 139, 138, 137, 136, 135, 129, 128, 127, 125, 124,
    123, 122, 121, 120, 117, 116, 115, 114, 113, 112, 110, 109, 108, 107,
    106, 105, 103, 102, 100, 99, 98, 97, 96, 95, 94, 92, 91, 90, 88, 87,
    86, 84, 81, 80, 78, 75, 74, 73, 72, 68, 67, 66, 64, 63, 61, 59, 58,
    57, 56, 55, 54, 52, 51, 50, 49, 48, 47, 46, 43, 42, 41, 39, 38, 36,
    35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 24, 23, 22, 21, 20, 18, 17,
    16, 15, 14, 11, 10, 9, 8, 7, 6, 3, 2,
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Close stale PRs")
    parser.add_argument("--dry-run", action="store_true", help="List PRs without closing")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_PAT") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: set GITHUB_PAT or GITHUB_TOKEN env var", file=sys.stderr)
        sys.exit(1)

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    closed = 0
    skipped = 0
    for pr_num in CLOSE_PR_NUMBERS:
        r = requests.get(f"{API_BASE}/repos/{REPO}/pulls/{pr_num}", headers=headers)
        if r.status_code != 200:
            print(f"  PR #{pr_num}: fetch failed ({r.status_code})")
            skipped += 1
            continue

        pr = r.json()
        if pr.get("state") != "open":
            skipped += 1
            continue

        title = pr["title"][:60]
        if args.dry_run:
            print(f"  [DRY-RUN] Would close PR #{pr_num}: {title}")
            continue

        # Post explanation comment
        requests.post(
            f"{API_BASE}/repos/{REPO}/issues/{pr_num}/comments",
            json={"body": CLOSE_COMMENT},
            headers=headers,
        )
        # Close the PR
        patch = requests.patch(
            f"{API_BASE}/repos/{REPO}/pulls/{pr_num}",
            json={"state": "closed"},
            headers=headers,
        )
        if patch.status_code == 200:
            print(f"  Closed PR #{pr_num}: {title}")
            closed += 1
        else:
            print(f"  FAILED to close PR #{pr_num}: {patch.status_code} {patch.text[:100]}")
            skipped += 1

        time.sleep(0.4)  # stay well within GitHub rate limits (5000 req/hr)

    print(f"\nDone: {closed} closed, {skipped} skipped/already closed.")


if __name__ == "__main__":
    main()
