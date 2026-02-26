"""Skill: changelog generator — produce a structured changelog from git commit history."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.skills.base import SkillBase

# Conventional Commits type mapping
_TYPE_MAP = {
    "feat":     ("Features",      "minor"),
    "fix":      ("Bug Fixes",     "patch"),
    "perf":     ("Performance",   "patch"),
    "refactor": ("Refactoring",   "patch"),
    "docs":     ("Documentation", "patch"),
    "test":     ("Tests",         "patch"),
    "chore":    ("Chores",        "patch"),
    "ci":       ("CI/CD",         "patch"),
    "build":    ("Build",         "patch"),
    "revert":   ("Reverts",       "patch"),
    "style":    ("Style",         "patch"),
}

_CONVENTIONAL_RE = re.compile(
    r"^(?P<type>[a-z]+)(\((?P<scope>[^)]+)\))?(?P<breaking>!)?\s*:\s*(?P<subject>.+)$"
)

# Bump priority
_BUMP_PRIORITY = {"major": 3, "minor": 2, "patch": 1, "none": 0}


def _run_git(args: List[str], cwd: str) -> str:
    result = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _parse_commits(log_output: str) -> List[Dict]:
    commits = []
    for line in log_output.splitlines():
        if not line.strip():
            continue
        # format: <hash> <subject>
        parts = line.split(" ", 1)
        if len(parts) < 2:
            continue
        sha, subject = parts[0], parts[1].strip()
        m = _CONVENTIONAL_RE.match(subject)
        if m:
            commit_type = m.group("type")
            scope = m.group("scope") or ""
            breaking = bool(m.group("breaking"))
            clean_subject = m.group("subject")
            category, bump = _TYPE_MAP.get(commit_type, ("Other", "patch"))
            if breaking:
                bump = "major"
        else:
            commit_type = "other"
            scope = ""
            breaking = False
            clean_subject = subject
            category = "Other"
            bump = "patch"

        commits.append({
            "sha": sha[:8],
            "type": commit_type,
            "scope": scope,
            "breaking": breaking,
            "subject": clean_subject,
            "category": category,
            "bump": bump,
        })
    return commits


def _group_by_category(commits: List[Dict]) -> Dict[str, List[Dict]]:
    groups: Dict[str, List] = {}
    for c in commits:
        groups.setdefault(c["category"], []).append(c)
    return groups


def _determine_version_bump(commits: List[Dict]) -> str:
    best = "none"
    for c in commits:
        bump = c["bump"]
        if _BUMP_PRIORITY.get(bump, 0) > _BUMP_PRIORITY.get(best, 0):
            best = bump
    return best


def _format_markdown(groups: Dict[str, List[Dict]], version_bump: str, from_ref: str, to_ref: str) -> str:
    lines = [f"## Changelog ({from_ref}..{to_ref})", f"**Suggested version bump:** `{version_bump}`", ""]
    for category in sorted(groups.keys()):
        lines.append(f"### {category}")
        for c in groups[category]:
            scope_str = f"**{c['scope']}**: " if c["scope"] else ""
            breaking_str = " ⚠️ **BREAKING CHANGE**" if c["breaking"] else ""
            lines.append(f"- {scope_str}{c['subject']} (`{c['sha']}`){breaking_str}")
        lines.append("")
    return "\n".join(lines)


class ChangelogGeneratorSkill(SkillBase):
    """Generate a structured changelog from git commit history using Conventional Commits."""

    name = "changelog_generator"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        project_root: str = input_data.get("project_root", ".")
        from_ref: Optional[str] = input_data.get("from_ref")      # e.g. "v1.0.0" or a SHA
        to_ref: str = input_data.get("to_ref", "HEAD")
        limit: int = int(input_data.get("limit", 100))
        include_markdown: bool = input_data.get("include_markdown", True)

        root = Path(project_root).resolve()
        if not (root / ".git").exists():
            return {"error": f"No git repository found at '{project_root}'."}

        try:
            # Build git log range
            if from_ref:
                log_range = f"{from_ref}..{to_ref}"
            else:
                # Use all commits up to limit if no from_ref
                log_range = to_ref

            git_args = [
                "log",
                "--pretty=format:%h %s",
                f"-{limit}",
                log_range,
            ]
            log_output = _run_git(git_args, str(root))
        except RuntimeError as exc:
            return {"error": str(exc)}

        if not log_output:
            return {"error": "No commits found in the specified range.", "commits": [], "version_bump": "none"}

        commits = _parse_commits(log_output)
        groups = _group_by_category(commits)
        version_bump = _determine_version_bump(commits)

        breaking_commits = [c for c in commits if c["breaking"]]

        result: Dict[str, Any] = {
            "from_ref": from_ref or "(beginning)",
            "to_ref": to_ref,
            "commit_count": len(commits),
            "version_bump": version_bump,
            "breaking_changes": len(breaking_commits),
            "categories": {cat: len(items) for cat, items in groups.items()},
            "commits": commits,
        }

        if include_markdown:
            from_label = from_ref or "beginning"
            result["markdown"] = _format_markdown(groups, version_bump, from_label, to_ref)

        return result
