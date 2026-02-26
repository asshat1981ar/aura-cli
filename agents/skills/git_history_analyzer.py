"""Skill: analyse git commit history for hotspots and change patterns."""
from __future__ import annotations
import subprocess
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.skills.base import SkillBase
from core.logging_utils import log_json


def _run_git(args: List[str], cwd: Path) -> Optional[str]:
    try:
        result = subprocess.run(["git"] + args, cwd=str(cwd), capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return result.stdout
        return None
    except Exception:
        return None


class GitHistoryAnalyzerSkill(SkillBase):
    name = "git_history_analyzer"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        project_root = Path(input_data.get("project_root", "."))
        lookback_days = int(input_data.get("lookback_days", 30))
        since = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        # Total commits
        total_out = _run_git(["rev-list", "--count", "HEAD"], project_root)
        total_commits = int(total_out.strip()) if total_out and total_out.strip().isdigit() else 0

        # Recent commits
        recent_out = _run_git(["rev-list", "--count", f"--after={since}", "HEAD"], project_root)
        recent_commits = int(recent_out.strip()) if recent_out and recent_out.strip().isdigit() else 0

        # File change frequency
        log_out = _run_git(["log", "--name-only", "--format=", f"--after={since}"], project_root)
        file_counts: Counter = Counter()
        if log_out:
            for line in log_out.splitlines():
                line = line.strip()
                if line and not line.startswith("commit") and "." in line:
                    file_counts[line] += 1

        hotspot_files = [{"file": f, "change_count": c} for f, c in file_counts.most_common(20)]
        risky_areas = [h["file"] for h in hotspot_files if h["change_count"] >= 5]

        # Patterns
        patterns: List[Dict] = []
        if recent_commits > 50:
            patterns.append({"type": "high_velocity", "description": f"{recent_commits} commits in last {lookback_days} days – active development"})
        if risky_areas:
            patterns.append({"type": "hotspot_concentration", "description": f"{len(risky_areas)} files changed 5+ times – consider splitting"})

        # Authors
        authors_out = _run_git(["log", "--format=%ae", f"--after={since}"], project_root)
        authors: List[str] = []
        if authors_out:
            authors = list({a.strip() for a in authors_out.splitlines() if a.strip()})

        log_json("INFO", "git_history_analyzer_complete", details={"total_commits": total_commits, "hotspots": len(hotspot_files)})
        return {"hotspot_files": hotspot_files, "recent_commits": recent_commits, "total_commits": total_commits, "patterns": patterns, "risky_areas": risky_areas[:10], "unique_authors": authors[:20], "lookback_days": lookback_days}
