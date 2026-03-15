"""
Security Hardener Skill — Scans for and remediates hardcoded secrets.
"""
from __future__ import annotations

import re
import os
from pathlib import Path
from typing import Any, Dict, List

from agents.skills.base import SkillBase

class SecurityHardenerSkill(SkillBase):
    """
    Scans Python source files for hardcoded secrets and proposes os.environ.get replacements.
    """
    name = "security_hardener"

    # Regex that captures assignments where the variable name contains a secret-related keyword.
    _SECRET_ASSIGN_RE = re.compile(
        r"^\s*(?P<var>\w*?(?:secret|key|token|password|pwd)\w*)\s*=\s*(?P<quote>['\"])(?P<value>.*?)(?P=quote)\s*(?:#.*)?$",
        re.IGNORECASE,
    )

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        content = input_data.get("content")
        file_path = input_data.get("file_path", "unknown.py")
        project_root = input_data.get("project_root")

        if not content and project_root:
            # If no content but root provided, scan the whole project
            return self._scan_project(Path(project_root))
        
        if not content:
            return {"error": "Provide 'content' or 'project_root'"}

        findings = self._scan_content(content, file_path)
        return {
            "file_path": file_path,
            "findings_count": len(findings),
            "findings": findings
        }

    def _scan_content(self, content: str, file_path: str) -> List[Dict[str, str]]:
        findings = []
        for line in content.splitlines():
            match = self._SECRET_ASSIGN_RE.match(line)
            if match:
                var_name = match.group("var")
                # Extract indent manually
                indent = line[:line.find(var_name)]
                env_var = re.sub(r"\W+", "_", var_name).upper()
                new_line = f"{indent}{var_name} = os.environ.get('{env_var}')"
                
                findings.append({
                    "old_line": line,
                    "new_line": new_line,
                    "var_name": var_name,
                    "env_var": env_var,
                    "implement_block": (
                        f"# IMPLEMENT START ({file_path})\n"
                        f"# Replaced hardcoded secret with env var lookup\n"
                        f"{new_line}\n"
                        f"# IMPLEMENT END"
                    )
                })
        return findings

    def _scan_project(self, root: Path) -> Dict[str, Any]:
        all_findings = {}
        total_count = 0
        for py_file in root.rglob("*.py"):
            if any(skip in py_file.parts for skip in [".git", "__pycache__", "node_modules"]):
                continue
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                findings = self._scan_content(content, str(py_file.relative_to(root)))
                if findings:
                    all_findings[str(py_file)] = findings
                    total_count += len(findings)
            except Exception:
                continue
        return {
            "project_root": str(root),
            "total_findings": total_count,
            "file_findings": all_findings
        }
