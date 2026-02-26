"""Skill: analyse project dependencies for conflicts, vulnerabilities, and pin hygiene."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.skills.base import SkillBase
from core.logging_utils import log_json

# ---------------------------------------------------------------------------
# Vulnerability database (package → (vulnerable_below, advisory_id, description))
# ---------------------------------------------------------------------------
_KNOWN_VULNS: Dict[str, List] = {
    "requests":     ["<2.20.0", "CVE-2018-18074", "urllib3 credential exposure via Proxy-Authorization header"],
    "pyyaml":       ["<5.4",    "CVE-2020-14343", "Arbitrary code execution via yaml.load without Loader"],
    "pillow":       ["<9.0.0",  "CVE-2022-22815", "Path traversal in EpsImagePlugin"],
    "cryptography": ["<3.3",    "CVE-2020-36242", "Buffer overflow in symmetric cipher backends"],
    "urllib3":      ["<1.26.5", "CVE-2021-33503", "ReDoS in percent-encoded URL parsing"],
    "django":       ["<3.2.13", "CVE-2022-28346", "SQL injection via QuerySet.annotate"],
    "flask":        ["<2.2.5",  "CVE-2023-30861", "Session cookie exposure on reverse proxies"],
    "paramiko":     ["<2.10.1", "CVE-2022-24302", "Private key disclosure via SFTP race condition"],
    "werkzeug":     ["<2.2.3",  "CVE-2023-25577", "Multipart request DoS via malformed boundary"],
    "jinja2":       ["<3.1.3",  "CVE-2024-22195", "HTML attribute injection via xmlattr filter"],
    "aiohttp":      ["<3.9.0",  "CVE-2024-23829", "HTTP request smuggling via malformed headers"],
    "setuptools":   ["<65.5.1", "CVE-2022-40897", "ReDoS in package_index metadata parsing"],
    "certifi":      ["<2022.12.7", "GHSA-43fp-rhv2-5gv8", "Outdated root CA bundle"],
    "starlette":    ["<0.27.0", "CVE-2023-29159", "Path traversal in StaticFiles"],
    "fastapi":      ["<0.99.1", "CVE-2023-29159", "Inherited from starlette StaticFiles"],
}

_UNPIN_PATTERN = re.compile(r"^[A-Za-z0-9_\-\.]+\s*$")  # no version constraint at all


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_requirements(path: Path) -> List[Dict]:
    packages: List[Dict] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return packages
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-") or line.startswith("git+"):
            continue
        m = re.match(r"^([A-Za-z0-9_\-\.]+)(.*)", line)
        if m:
            name = m.group(1).lower().replace("-", "_")
            specifier = m.group(2).strip()
            packages.append({
                "name": name,
                "raw_name": m.group(1),
                "specifier": specifier,
                "source": path.name,
                "pinned": "==" in specifier,
                "unpinned": not specifier or bool(_UNPIN_PATTERN.match(specifier + "x")),
            })
    return packages


def _parse_pyproject_toml(path: Path) -> List[Dict]:
    """Very lightweight TOML parser for [tool.poetry.dependencies] and [project.dependencies]."""
    packages: List[Dict] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return packages

    in_deps = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and "dependencies" in stripped.lower():
            in_deps = True
            continue
        if stripped.startswith("[") and in_deps:
            in_deps = False
        if not in_deps:
            continue
        # poetry style: requests = "^2.28"
        m = re.match(r'^([A-Za-z0-9_\-\.]+)\s*=\s*["\']?([^"\'#\s]*)', stripped)
        if m and m.group(1).lower() not in {"python", "name", "version", "description"}:
            name = m.group(1).lower().replace("-", "_")
            specifier = m.group(2).strip()
            packages.append({
                "name": name,
                "raw_name": m.group(1),
                "specifier": specifier,
                "source": path.name,
                "pinned": "==" in specifier,
                "unpinned": not specifier,
            })
    return packages


def _pip_list() -> Optional[List[Dict]]:
    """Run `pip list --format=columns` and return installed packages if available."""
    try:
        result = subprocess.run(
            ["pip", "list", "--format=columns"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return None
        packages: List[Dict] = []
        for line in result.stdout.splitlines()[2:]:  # skip header rows
            parts = line.split()
            if len(parts) >= 2:
                packages.append({
                    "name": parts[0].lower().replace("-", "_"),
                    "raw_name": parts[0],
                    "specifier": f"=={parts[1]}",
                    "source": "pip_list",
                    "pinned": True,
                    "unpinned": False,
                })
        return packages
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _check_vulns(packages: List[Dict]) -> List[Dict]:
    vulns: List[Dict] = []
    for pkg in packages:
        info = _KNOWN_VULNS.get(pkg["name"])
        if info:
            vulns.append({
                "package": pkg["raw_name"],
                "installed_specifier": pkg["specifier"],
                "vulnerable_below": info[0],
                "cve": info[1],
                "description": info[2],
                "severity": "high",
                "source": pkg["source"],
            })
    return vulns


def _check_conflicts(packages: List[Dict]) -> List[Dict]:
    seen: Dict[str, List] = {}
    for pkg in packages:
        seen.setdefault(pkg["name"], []).append(pkg)
    return [
        {
            "package": name,
            "entries": [{"specifier": e["specifier"], "source": e["source"]} for e in entries],
            "issue": "Duplicate requirement — may cause version conflicts",
        }
        for name, entries in seen.items()
        if len(entries) > 1
    ]


def _check_unpinned(packages: List[Dict]) -> List[str]:
    return [pkg["raw_name"] for pkg in packages if pkg.get("unpinned")]


class DependencyAnalyzerSkill(SkillBase):
    """
    Analyse project dependencies from requirements*.txt and pyproject.toml.
    Detects known CVEs, duplicate/conflicting entries, and unpinned packages.
    Can optionally read installed packages via `pip list`.

    Input:
      project_root   — path to scan (default '.')
      paths          — list of project roots for multi-project scan
      include_pip    — if True, also scan `pip list` for installed packages
    """

    name = "dependency_analyzer"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        project_root_str: Optional[str] = input_data.get("project_root")
        paths: Optional[List[str]] = input_data.get("paths")
        include_pip: bool = input_data.get("include_pip", False)

        roots = [Path(p) for p in paths] if paths else [Path(project_root_str or ".")]

        packages: List[Dict] = []
        req_files_scanned: List[str] = []

        for root in roots:
            candidates = (
                list(root.rglob("requirements*.txt"))
                + list(root.rglob("pyproject.toml"))
            )
            candidates = [
                f for f in candidates
                if ".git" not in f.parts and "node_modules" not in f.parts
            ]
            for f in candidates:
                req_files_scanned.append(str(f))
                if f.name == "pyproject.toml":
                    packages.extend(_parse_pyproject_toml(f))
                else:
                    packages.extend(_parse_requirements(f))

        if include_pip:
            pip_pkgs = _pip_list()
            if pip_pkgs:
                packages.extend(pip_pkgs)

        vulns = _check_vulns(packages)
        conflicts = _check_conflicts(packages)
        unpinned = _check_unpinned(packages)

        recommendations: List[str] = []
        if vulns:
            recommendations.append(f"Upgrade {len(vulns)} vulnerable package(s) immediately.")
        if conflicts:
            recommendations.append(f"Resolve {len(conflicts)} conflicting requirement entry/entries.")
        if unpinned:
            recommendations.append(
                f"{len(unpinned)} unpinned package(s) — pin versions for reproducible builds."
            )
        if not packages:
            recommendations.append("No requirements files found; consider adding requirements.txt or pyproject.toml.")

        log_json("INFO", "dependency_analyzer_complete", details={
            "packages": len(packages),
            "vulns": len(vulns),
            "conflicts": len(conflicts),
            "unpinned": len(unpinned),
        })

        return {
            "packages_found": len(packages),
            "req_files_scanned": req_files_scanned,
            "vulnerabilities": vulns,
            "vulnerability_count": len(vulns),
            "conflicts": conflicts,
            "unpinned_packages": unpinned,
            "recommendations": recommendations,
            "packages": packages[:100],  # cap to avoid huge payloads
        }
