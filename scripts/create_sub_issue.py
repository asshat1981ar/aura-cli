#!/usr/bin/env python3
"""
Script to create a sub-issue for a parent issue in GitHub.

This script creates sub-issues for asshat1981ar/project-chimera issue #22
(Automated Analysis & Insight Generation).

Usage:
    # Using GitHub PAT (Personal Access Token)
    GITHUB_PAT=<token> python3 scripts/create_sub_issue.py

    # Or using gh CLI (recommended)
    gh auth login
    python3 scripts/create_sub_issue.py --use-gh-cli

    # Create with custom content
    python3 scripts/create_sub_issue.py --title "Custom title" --body-file body.md

Requirements:
    - For API method: GITHUB_PAT or GITHUB_TOKEN with 'repo' scope
    - For gh CLI method: gh CLI installed and authenticated
    - The token must have write access to asshat1981ar/project-chimera
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_with_gh_cli(repo: str, title: str, body: str) -> int:
    """Create issue using gh CLI."""
    try:
        # Check if gh is available
        result = subprocess.run(
            ["gh", "--version"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode != 0:
            print("Error: gh CLI not found. Install from https://cli.github.com/")
            return 1

        # Create the issue
        cmd = [
            "gh", "issue", "create",
            "--repo", repo,
            "--title", title,
            "--body", body
        ]

        print(f"Creating issue in {repo} using gh CLI...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode == 0:
            print(f"\n✓ Sub-issue created successfully!")
            print(result.stdout)
            return 0
        else:
            print(f"\n✗ Failed to create sub-issue:")
            print(result.stderr)
            return 1

    except FileNotFoundError:
        print("Error: gh CLI not found")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


def create_with_api(repo: str, title: str, body: str) -> int:
    """Create issue using GitHub API via GitHubTools."""
    try:
        from tools.github_tools import GitHubTools
    except ImportError as e:
        print(f"Error importing GitHubTools: {e}")
        return 1

    # Try to get GitHub token from various sources
    github_token = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PAT")

    if not github_token:
        print("Error: No GitHub token found")
        print("Please set GITHUB_TOKEN or GITHUB_PAT environment variable")
        print("Or use --use-gh-cli flag to use gh CLI instead")
        return 1

    # Temporarily set GITHUB_PAT for GitHubTools
    os.environ["GITHUB_PAT"] = github_token

    # Initialize GitHub tools
    try:
        gh = GitHubTools()
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    print(f"Creating sub-issue in {repo}...")
    print(f"Title: {title}")
    print(f"Body preview: {body[:200]}...")

    try:
        result = gh.create_issue(repo, title, body)
        issue_url = result.get("html_url", "")
        issue_number = result.get("number", "")

        print(f"\n✓ Sub-issue created successfully!")
        print(f"  Issue #{issue_number}")
        print(f"  URL: {issue_url}")

        return 0
    except Exception as e:
        print(f"\n✗ Failed to create sub-issue: {e}")
        print("\nNote: If you get a 403 error, ensure your token has 'repo' scope")
        print("      and write access to the target repository.")
        print("\nAlternatively, try using gh CLI:")
        print(f"  python3 {sys.argv[0]} --use-gh-cli")
        return 1


def get_default_sub_issue_content() -> tuple[str, str]:
    """Get the default sub-issue title and body for issue #22."""
    parent_issue_number = 22

    title = "Implement metrics query interface for self_opt_metrics"
    body = f"""**Parent Issue:** #{parent_issue_number}

## Description
Implement the data query layer to retrieve recent self_opt_metrics for analysis.

This is a sub-task of the larger "Automated Analysis & Insight Generation" feature,
focusing specifically on the data retrieval component.

## Tasks
- [ ] Design data schema for self_opt_metrics storage
- [ ] Implement query interface to fetch last N runs
- [ ] Add filtering by date range
- [ ] Add support for specific metric selection
- [ ] Include tests for query functionality
- [ ] Document the query API

## Acceptance Criteria
- Can query self_opt_metrics for configurable number of recent runs
- Query performance is acceptable (< 100ms for typical queries)
- Query interface is well-documented
- Unit tests achieve >80% coverage

## Context
This is the first step in implementing automated analysis from issue #{parent_issue_number}.
The query layer will be used by subsequent tasks:
- Trend computation (future sub-issue)
- Anomaly detection (future sub-issue)
- Summary generation (future sub-issue)

## Technical Notes
The implementation should:
- Support both synchronous and asynchronous query patterns
- Handle pagination for large result sets
- Include proper error handling and logging
- Be compatible with the existing metrics storage system

/cc @Copilot
"""
    return title, body


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create a sub-issue for project-chimera issue #22",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using GitHub PAT
  GITHUB_PAT=ghp_xxx python3 scripts/create_sub_issue.py

  # Using gh CLI (recommended for local use)
  python3 scripts/create_sub_issue.py --use-gh-cli

  # Custom content
  python3 scripts/create_sub_issue.py --title "Custom" --body "Content"

  # Dry run (preview without creating)
  python3 scripts/create_sub_issue.py --dry-run
        """
    )
    parser.add_argument(
        "--use-gh-cli",
        action="store_true",
        help="Use gh CLI instead of GitHub API"
    )
    parser.add_argument(
        "--repo",
        default="asshat1981ar/project-chimera",
        help="Target repository (default: asshat1981ar/project-chimera)"
    )
    parser.add_argument(
        "--title",
        help="Custom issue title (default: from template)"
    )
    parser.add_argument(
        "--body",
        help="Custom issue body"
    )
    parser.add_argument(
        "--body-file",
        type=Path,
        help="Read issue body from file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the issue without creating it"
    )

    args = parser.parse_args()

    # Get title and body
    title, body = get_default_sub_issue_content()

    if args.title:
        title = args.title

    if args.body:
        body = args.body
    elif args.body_file:
        if not args.body_file.exists():
            print(f"Error: Body file not found: {args.body_file}")
            return 1
        body = args.body_file.read_text()

    # Dry run: just print and exit
    if args.dry_run:
        print("=" * 70)
        print("DRY RUN - Issue would be created in:", args.repo)
        print("=" * 70)
        print(f"\nTitle:\n{title}\n")
        print("=" * 70)
        print(f"\nBody:\n{body}\n")
        print("=" * 70)
        return 0

    # Create the issue
    if args.use_gh_cli:
        return create_with_gh_cli(args.repo, title, body)
    else:
        return create_with_api(args.repo, title, body)


if __name__ == "__main__":
    sys.exit(main())
