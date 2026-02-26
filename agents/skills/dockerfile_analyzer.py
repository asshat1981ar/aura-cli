"""Skill: Dockerfile static analysis — security, best-practices, and image hygiene."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.skills.base import SkillBase
from core.logging_utils import log_json

# Known slim / distroless base images (lower attack surface)
_SLIM_BASES = {
    "alpine", "scratch", "distroless", "slim", "debian:bookworm-slim",
    "debian:bullseye-slim", "ubuntu:22.04", "ubuntu:20.04",
}

# Known heavy / discouraged base images
_HEAVY_BASES = {"ubuntu:latest", "debian:latest", "centos", "fedora"}

# Sensitive env/arg patterns that should not be hardcoded
_SECRET_ENV_RE = re.compile(
    r"(?i)(password|passwd|secret|api_key|token|private_key|access_key)\s*=\s*\S+",
)


def _parse_dockerfile(content: str) -> List[Dict]:
    """Return a list of parsed Dockerfile instructions as {instruction, args, line_no} dicts."""
    instructions = []
    for line_no, raw in enumerate(content.splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Handle line continuations
        parts = stripped.split(None, 1)
        if parts:
            instructions.append({
                "instruction": parts[0].upper(),
                "args": parts[1] if len(parts) > 1 else "",
                "line_no": line_no,
            })
    return instructions


def _analyse(content: str, file_path: str) -> Dict[str, Any]:
    instructions = _parse_dockerfile(content)
    findings: List[Dict] = []
    metadata: Dict[str, Any] = {
        "file": file_path,
        "base_images": [],
        "exposed_ports": [],
        "has_healthcheck": False,
        "is_multistage": False,
        "user_set": False,
        "stages": 0,
    }

    from_count = 0

    for instr in instructions:
        cmd = instr["instruction"]
        args = instr["args"]
        ln = instr["line_no"]

        # ---- FROM ----
        if cmd == "FROM":
            from_count += 1
            # strip alias (AS name)
            base = args.split()[0].lower() if args.split() else ""
            metadata["base_images"].append(base)

            if base == "latest" or base.endswith(":latest"):
                findings.append({
                    "severity": "high",
                    "issue": "pinned_tag_missing",
                    "detail": f"Base image '{base}' uses :latest — pin to a specific version for reproducible builds.",
                    "line": ln,
                })
            if any(heavy in base for heavy in _HEAVY_BASES):
                findings.append({
                    "severity": "medium",
                    "issue": "heavy_base_image",
                    "detail": f"'{base}' is a large base image. Consider a slim or distroless variant.",
                    "line": ln,
                })

        # ---- RUN ----
        elif cmd == "RUN":
            if re.search(r"\bsudo\b", args):
                findings.append({
                    "severity": "medium",
                    "issue": "sudo_in_run",
                    "detail": "Avoid 'sudo' in RUN; use USER or run as root explicitly.",
                    "line": ln,
                })
            if re.search(r"apt-get install(?!.*-y)", args):
                findings.append({
                    "severity": "low",
                    "issue": "apt_get_missing_y_flag",
                    "detail": "Use 'apt-get install -y' to prevent interactive prompts.",
                    "line": ln,
                })
            if re.search(r"&&\s*rm\s+-rf\s+/var/lib/apt", args) is None and "apt-get install" in args:
                findings.append({
                    "severity": "low",
                    "issue": "apt_cache_not_cleaned",
                    "detail": "Clean apt cache in the same RUN layer: '&& rm -rf /var/lib/apt/lists/*'",
                    "line": ln,
                })
            if re.search(r"\bcurl\b.*\|\s*(sh|bash)", args):
                findings.append({
                    "severity": "high",
                    "issue": "curl_pipe_to_shell",
                    "detail": "Piping curl output directly to a shell is a supply-chain risk.",
                    "line": ln,
                })

        # ---- ENV / ARG — secrets ----
        elif cmd in ("ENV", "ARG"):
            if _SECRET_ENV_RE.search(args):
                findings.append({
                    "severity": "critical",
                    "issue": "secret_in_dockerfile",
                    "detail": f"{cmd} instruction appears to contain a secret: '{args[:80]}'.",
                    "line": ln,
                })

        # ---- EXPOSE ----
        elif cmd == "EXPOSE":
            ports = args.split()
            metadata["exposed_ports"].extend(ports)
            for p in ports:
                port_num = int(p.split("/")[0]) if p.split("/")[0].isdigit() else None
                if port_num and port_num < 1024:
                    findings.append({
                        "severity": "medium",
                        "issue": "privileged_port_exposed",
                        "detail": f"Port {port_num} is a privileged port (<1024); containers typically run as non-root.",
                        "line": ln,
                    })

        # ---- USER ----
        elif cmd == "USER":
            metadata["user_set"] = True
            if args.strip() in ("0", "root"):
                findings.append({
                    "severity": "high",
                    "issue": "running_as_root",
                    "detail": "Container is explicitly set to run as root. Use a non-root user.",
                    "line": ln,
                })

        # ---- HEALTHCHECK ----
        elif cmd == "HEALTHCHECK":
            if args.strip().upper() != "NONE":
                metadata["has_healthcheck"] = True

        # ---- ADD (prefer COPY) ----
        elif cmd == "ADD":
            if not (re.search(r"https?://", args) or args.endswith(".tar.gz") or args.endswith(".tgz")):
                findings.append({
                    "severity": "low",
                    "issue": "add_instead_of_copy",
                    "detail": "Prefer COPY over ADD unless you need URL fetching or tar auto-extraction.",
                    "line": ln,
                })

    metadata["stages"] = from_count
    metadata["is_multistage"] = from_count > 1

    # Post-scan checks
    if not metadata["has_healthcheck"]:
        findings.append({
            "severity": "low",
            "issue": "no_healthcheck",
            "detail": "No HEALTHCHECK instruction. Consider adding one for orchestration health monitoring.",
            "line": None,
        })
    if not metadata["user_set"]:
        findings.append({
            "severity": "medium",
            "issue": "no_user_instruction",
            "detail": "No USER instruction found. Container will run as root by default.",
            "line": None,
        })

    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    findings.sort(key=lambda f: severity_order.get(f["severity"], 9))

    critical_count = sum(1 for f in findings if f["severity"] == "critical")
    high_count = sum(1 for f in findings if f["severity"] == "high")

    return {
        **metadata,
        "findings": findings,
        "finding_count": len(findings),
        "critical_count": critical_count,
        "high_count": high_count,
        "score": max(0, 100 - critical_count * 30 - high_count * 15 - len(findings) * 3),
    }


class DockerfileAnalyzerSkill(SkillBase):
    """Analyse a Dockerfile (or all Dockerfiles in a project) for security and best-practice issues."""

    name = "dockerfile_analyzer"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        content: Optional[str] = input_data.get("content")
        file_path: str = input_data.get("file_path", "Dockerfile")
        project_root: Optional[str] = input_data.get("project_root")

        if content:
            return _analyse(content, file_path)

        if project_root:
            root = Path(project_root)
            dockerfiles = list(root.rglob("Dockerfile")) + list(root.rglob("Dockerfile.*"))
            if not dockerfiles:
                return {"error": f"No Dockerfiles found under '{project_root}'.", "files_scanned": 0}

            all_results = []
            for df in dockerfiles:
                try:
                    text = df.read_text(encoding="utf-8", errors="replace")
                    result = _analyse(text, str(df.relative_to(root)))
                    all_results.append(result)
                except Exception as exc:
                    log_json("WARN", "dockerfile_analyzer_read_error", details={"file": str(df), "error": str(exc)})

            total_findings = sum(r["finding_count"] for r in all_results)
            total_critical = sum(r["critical_count"] for r in all_results)
            total_high = sum(r["high_count"] for r in all_results)
            return {
                "files_scanned": len(all_results),
                "results": all_results,
                "total_findings": total_findings,
                "total_critical": total_critical,
                "total_high": total_high,
            }

        return {"error": "Provide 'content' (Dockerfile text) or 'project_root' (directory path)."}
