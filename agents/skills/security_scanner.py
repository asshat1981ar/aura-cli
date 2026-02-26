"""Skill: static security analysis – secrets, injection patterns, unsafe calls."""
from __future__ import annotations
import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.skills.base import SkillBase
from core.logging_utils import log_json

_SECRET_PATTERNS = [
    (re.compile(r'(?i)(api_key|apikey|secret_key|access_token|auth_token)\s*=\s*["\'][^"\']{8,}["\']'), "hardcoded_secret", "critical"),
    (re.compile(r'(?i)(password|passwd|pwd)\s*=\s*["\'][^"\']{4,}["\']'), "hardcoded_password", "critical"),
    (re.compile(r'Bearer\s+[A-Za-z0-9\-_\.]{20,}'), "bearer_token_in_code", "high"),
    (re.compile(r'(?i)private[_\s]?key\s*=\s*["\'][^"\']{10,}'), "hardcoded_private_key", "critical"),
    (re.compile(r'(?:AKIA|ASIA)[A-Z0-9]{16}'), "aws_access_key", "critical"),
]

_SQL_PATTERNS = [
    (re.compile(r'(?i)(SELECT|INSERT|UPDATE|DELETE|DROP)\b.*?(%s|%d|\{[^}]+\}|"\s*\+|\'s*\+)', re.DOTALL), "sql_injection_risk", "high"),
]

_UNSAFE_CALLS = {
    "eval": ("unsafe_eval", "high"),
    "exec": ("unsafe_exec", "medium"),
    "pickle.loads": ("unsafe_pickle_loads", "high"),
    "os.system": ("os_system_call", "medium"),
    "subprocess_shell_true": ("subprocess_shell_true", "medium"),
}


def _scan_text(text: str, file_path: str) -> List[Dict]:
    findings = []
    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        for pattern, issue_type, severity in _SECRET_PATTERNS:
            if pattern.search(line):
                findings.append({"severity": severity, "issue": issue_type, "file": file_path, "line": i, "snippet": line.strip()[:120]})
        for pattern, issue_type, severity in _SQL_PATTERNS:
            if pattern.search(line):
                findings.append({"severity": severity, "issue": issue_type, "file": file_path, "line": i, "snippet": line.strip()[:120]})
    return findings


def _scan_ast(source: str, file_path: str) -> List[Dict]:
    findings = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return findings
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = ""
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = f"{getattr(func.value, 'id', '')}.{func.attr}"
            if name in ("eval", "exec"):
                issue, severity = _UNSAFE_CALLS[name]
                findings.append({"severity": severity, "issue": issue, "file": file_path, "line": node.lineno, "snippet": f"{name}() call"})
            elif "pickle" in name and "loads" in name:
                findings.append({"severity": "high", "issue": "unsafe_pickle_loads", "file": file_path, "line": node.lineno, "snippet": f"{name}() call"})
            elif name == "os.system":
                findings.append({"severity": "medium", "issue": "os_system_call", "file": file_path, "line": node.lineno, "snippet": "os.system() call"})
            # subprocess with shell=True
            for kw in node.keywords:
                if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    findings.append({"severity": "medium", "issue": "subprocess_shell_true", "file": file_path, "line": node.lineno, "snippet": "subprocess call with shell=True"})
    return findings


class SecurityScannerSkill(SkillBase):
    name = "security_scanner"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        code: Optional[str] = input_data.get("code")
        file_path: str = input_data.get("file_path", "<string>")
        project_root_str: Optional[str] = input_data.get("project_root")
        all_findings: List[Dict] = []

        if code:
            all_findings.extend(_scan_text(code, file_path))
            all_findings.extend(_scan_ast(code, file_path))
        elif project_root_str:
            root = Path(project_root_str)
            for f in root.rglob("*.py"):
                if ".git" in f.parts or "node_modules" in f.parts or "__pycache__" in f.parts:
                    continue
                try:
                    src = f.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                rel = str(f.relative_to(root))
                all_findings.extend(_scan_text(src, rel))
                all_findings.extend(_scan_ast(src, rel))
        else:
            return {"error": "Provide 'code' or 'project_root'"}

        critical = sum(1 for f in all_findings if f["severity"] == "critical")
        high = sum(1 for f in all_findings if f["severity"] == "high")
        summary = f"{len(all_findings)} finding(s): {critical} critical, {high} high."
        log_json("INFO", "security_scanner_complete", details={"findings": len(all_findings), "critical": critical})
        return {"findings": all_findings, "critical_count": critical, "high_count": high, "scan_summary": summary}
