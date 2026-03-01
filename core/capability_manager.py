from __future__ import annotations

import json
import os
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from core.config_manager import ConfigManager, DEFAULT_CONFIG
from core.runtime_paths import resolve_project_path


@dataclass(frozen=True)
class CapabilityRule:
    capability_id: str
    keywords: tuple[str, ...]
    recommended_skills: tuple[str, ...] = ()
    mcp_tools: tuple[str, ...] = ()
    provisioning_action: str | None = None
    reason: str = ""


_CAPABILITY_RULES: tuple[CapabilityRule, ...] = (
    CapabilityRule(
        capability_id="github_mcp",
        keywords=("github", "pull request", "pull-request", "repository", "repo", "issue", "issues"),
        recommended_skills=("git_history_analyzer", "changelog_generator"),
        mcp_tools=("gh", "git", "fs"),
        provisioning_action="ensure_mcp_servers",
        reason="Goal references GitHub/repository workflows that benefit from MCP-backed repo tooling.",
    ),
    CapabilityRule(
        capability_id="docker_analysis",
        keywords=("docker", "dockerfile", "container", "image"),
        recommended_skills=("dockerfile_analyzer",),
        reason="Goal references container build/runtime concerns.",
    ),
    CapabilityRule(
        capability_id="observability_analysis",
        keywords=("observability", "logging", "metrics", "tracing", "telemetry"),
        recommended_skills=("observability_checker",),
        reason="Goal references logging, metrics, or telemetry quality.",
    ),
    CapabilityRule(
        capability_id="database_analysis",
        keywords=("database", "sql", "query", "postgres", "mysql", "sqlite"),
        recommended_skills=("database_query_analyzer",),
        reason="Goal references database/query behavior.",
    ),
    CapabilityRule(
        capability_id="structural_analysis",
        keywords=("architecture", "structural", "dependency graph", "module graph", "codebase shape"),
        recommended_skills=("structural_analyzer", "architecture_validator"),
        reason="Goal references architecture or structural analysis.",
    ),
    CapabilityRule(
        capability_id="release_notes",
        keywords=("release", "changelog", "version bump", "release prep"),
        recommended_skills=("changelog_generator",),
        reason="Goal references release management or changelog generation.",
    ),
    CapabilityRule(
        capability_id="web_research",
        keywords=("url", "website", "web", "fetch", "http", "https", "research"),
        recommended_skills=("web_fetcher",),
        reason="Goal references web content or external research.",
    ),
    CapabilityRule(
        capability_id="skills_mcp_server",
        keywords=("skills mcp", "skills server", "mcp skills", "expose skills", "skill server"),
        recommended_skills=("skill_composer",),
        mcp_tools=("aura_skills_server",),
        provisioning_action="start_skills_mcp_server",
        reason="Goal references exposing or consuming the dedicated AURA skills MCP server.",
    ),
    CapabilityRule(
        capability_id="github_mcp_bridge",
        keywords=("github bridge", "github mcp bridge", "copilot mcp", "server-github", "github tool bridge"),
        recommended_skills=("git_history_analyzer",),
        mcp_tools=("github_bridge",),
        provisioning_action="start_github_mcp_bridge",
        reason="Goal references the dedicated GitHub MCP bridge flow rather than the general MCP server setup.",
    ),
)

CAPABILITY_GOAL_PREFIX = "Add AURA skill '"
CAPABILITY_STATUS_PATH = "memory/capability_status.json"


def _goal_lower(goal: str) -> str:
    return (goal or "").strip().lower()


def _config_value(config, key: str, default):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return default


def analyze_capability_needs(
    goal: str,
    *,
    available_skills: Iterable[str],
    active_skills: Iterable[str] = (),
) -> dict:
    """Infer extra skills and MCP setup actions from a goal string."""
    goal_lower = _goal_lower(goal)
    available = set(available_skills)
    active = set(active_skills)
    matched_rules: list[CapabilityRule] = []

    for rule in _CAPABILITY_RULES:
        if any(keyword in goal_lower for keyword in rule.keywords):
            matched_rules.append(rule)

    recommended_skills: list[str] = []
    missing_skills: list[str] = []
    mcp_tools: list[str] = []
    provisioning_actions: list[dict] = []

    for rule in matched_rules:
        for skill_name in rule.recommended_skills:
            if skill_name in available:
                if skill_name not in active and skill_name not in recommended_skills:
                    recommended_skills.append(skill_name)
            elif skill_name not in missing_skills:
                missing_skills.append(skill_name)

        for tool_name in rule.mcp_tools:
            if tool_name not in mcp_tools:
                mcp_tools.append(tool_name)

        if rule.provisioning_action and not any(
            action["action"] == rule.provisioning_action for action in provisioning_actions
        ):
            provisioning_actions.append(
                {
                    "action": rule.provisioning_action,
                    "capability_id": rule.capability_id,
                    "reason": rule.reason,
                    "mcp_tools": list(rule.mcp_tools),
                }
            )

    return {
        "matched_capabilities": [
            {
                "capability_id": rule.capability_id,
                "reason": rule.reason,
                "recommended_skills": list(rule.recommended_skills),
                "mcp_tools": list(rule.mcp_tools),
                "provisioning_action": rule.provisioning_action,
            }
            for rule in matched_rules
        ],
        "recommended_skills": recommended_skills,
        "missing_skills": missing_skills,
        "mcp_tools": mcp_tools,
        "provisioning_actions": provisioning_actions,
    }


def _tail(text: str, limit: int = 400) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _project_config(project_root: Path) -> ConfigManager:
    return ConfigManager(config_file=Path(project_root) / "aura.config.json")


def _capability_status_path(project_root: Path) -> Path:
    return resolve_project_path(Path(project_root), CAPABILITY_STATUS_PATH, CAPABILITY_STATUS_PATH)


def _goal_queue_items(*, project_root: Path, goal_queue=None, config=None) -> list[str]:
    if goal_queue is not None and getattr(goal_queue, "queue", None) is not None:
        return list(goal_queue.queue)

    resolved_config = config or _project_config(project_root)
    queue_path = resolve_project_path(
        Path(project_root),
        _config_value(resolved_config, "goal_queue_path", DEFAULT_CONFIG["goal_queue_path"]),
        DEFAULT_CONFIG["goal_queue_path"],
    )
    if not queue_path.exists():
        return []

    try:
        payload = json.loads(queue_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    return payload if isinstance(payload, list) else []


def is_capability_goal(goal: str) -> bool:
    return isinstance(goal, str) and goal.startswith(CAPABILITY_GOAL_PREFIX)


def load_capability_status(project_root: Path) -> dict:
    path = _capability_status_path(project_root)
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    return payload if isinstance(payload, dict) else {}


def save_capability_status(project_root: Path, report: dict) -> dict:
    path = _capability_status_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _listening(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


def _start_background_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    health_host: str,
    health_port: int,
    action: str,
) -> dict:
    if _listening(health_host, health_port):
        return {
            "action": action,
            "status": "already_running",
            "host": health_host,
            "port": health_port,
        }

    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return {
        "action": action,
        "status": "started",
        "pid": proc.pid,
        "host": health_host,
        "port": health_port,
    }


def _run_mcp_setup(project_root: Path, *, start_servers: bool) -> dict:
    script_path = project_root / "scripts" / "mcp_server_setup.sh"
    if not script_path.is_file():
        return {
            "action": "ensure_mcp_servers",
            "status": "failed",
            "error": f"missing setup script: {script_path}",
        }

    env = os.environ.copy()
    if not start_servers:
        env["MCP_SETUP_NO_START"] = "1"

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return {
        "action": "ensure_mcp_servers",
        "status": "applied" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "started_servers": start_servers,
        "stdout_tail": _tail(result.stdout.strip()),
        "stderr_tail": _tail(result.stderr.strip()),
    }


def _start_skills_mcp_server(project_root: Path) -> dict:
    config = _project_config(project_root)
    port = config.get_mcp_server_port("skills")
    env = os.environ.copy()
    env.update(
        {
            "MCP_PORT": str(port),
            "MCP_HOST": "127.0.0.1",
            "MCP_APP_MOD": "tools.aura_mcp_skills_server:app",
        }
    )
    return _start_background_command(
        ["bash", str(project_root / "scripts" / "run_mcp_server.sh")],
        cwd=project_root,
        env=env,
        health_host="127.0.0.1",
        health_port=port,
        action="start_skills_mcp_server",
    )


def _start_github_mcp_bridge(project_root: Path) -> dict:
    port = int(os.environ.get("MCP_SERVER_PORT", DEFAULT_CONFIG["mcp_servers"]["dev_tools"]))
    env = os.environ.copy()
    return _start_background_command(
        ["bash", str(project_root / "scripts" / "start_mcp_github.sh")],
        cwd=project_root,
        env=env,
        health_host="127.0.0.1",
        health_port=port,
        action="start_github_mcp_bridge",
    )


def build_missing_skill_goals(missing_skills: Iterable[str], goal: str) -> list[str]:
    goal_text = (goal or "").strip()
    return [
        f"{CAPABILITY_GOAL_PREFIX}{skill_name}' so AURA can better handle goal: {goal_text}"
        for skill_name in missing_skills
    ]


def queue_missing_capability_goals(
    *,
    goal_queue,
    missing_skills: Iterable[str],
    goal: str,
    enabled: bool,
    dry_run: bool,
) -> dict:
    missing = list(dict.fromkeys(missing_skills))
    if not missing:
        return {"attempted": False, "queued": [], "skipped": []}
    if dry_run:
        return {
            "attempted": False,
            "queued": [],
            "skipped": [{"goal": item, "reason": "dry_run"} for item in build_missing_skill_goals(missing, goal)],
        }
    if not enabled:
        return {
            "attempted": False,
            "queued": [],
            "skipped": [{"goal": item, "reason": "auto_queue_disabled"} for item in build_missing_skill_goals(missing, goal)],
        }
    if goal_queue is None:
        return {
            "attempted": False,
            "queued": [],
            "skipped": [{"goal": item, "reason": "goal_queue_unavailable"} for item in build_missing_skill_goals(missing, goal)],
        }

    existing = set(getattr(goal_queue, "queue", []) or [])
    candidate_goals = build_missing_skill_goals(missing, goal)
    new_goals = [item for item in candidate_goals if item not in existing]
    skipped = [
        {"goal": item, "reason": "already_queued"}
        for item in candidate_goals
        if item in existing
    ]
    if not new_goals:
        return {"attempted": True, "queued": [], "skipped": skipped, "queue_strategy": None}

    queue_strategy = "prepend"
    if hasattr(goal_queue, "prepend_batch"):
        goal_queue.prepend_batch(new_goals)
    elif hasattr(goal_queue, "batch_add"):
        queue_strategy = "append"
        goal_queue.batch_add(new_goals)
    else:
        queue_strategy = "append"
        for item in new_goals:
            goal_queue.add(item)

    return {
        "attempted": True,
        "queued": new_goals,
        "skipped": skipped,
        "queue_strategy": queue_strategy,
    }


def record_capability_status(
    *,
    project_root: Path,
    goal: str,
    capability_plan: dict,
    capability_goal_queue: dict | None = None,
    capability_provisioning: dict | None = None,
    goal_queue=None,
) -> dict:
    report = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "last_goal": goal,
        "matched_capabilities": list(capability_plan.get("matched_capabilities", [])),
        "recommended_skills": list(capability_plan.get("recommended_skills", [])),
        "missing_skills": list(capability_plan.get("missing_skills", [])),
        "provisioning_actions": list(capability_plan.get("provisioning_actions", [])),
        "queued_goals": list((capability_goal_queue or {}).get("queued", [])),
        "skipped_goals": list((capability_goal_queue or {}).get("skipped", [])),
        "queue_strategy": (capability_goal_queue or {}).get("queue_strategy"),
        "provisioning_results": list((capability_provisioning or {}).get("results", [])),
        "pending_self_development_goals": [
            item
            for item in _goal_queue_items(project_root=project_root, goal_queue=goal_queue)
            if is_capability_goal(item)
        ],
    }
    return save_capability_status(project_root, report)


def build_capability_status_report(
    project_root: Path,
    *,
    goal_queue=None,
    config=None,
    last_status: dict | None = None,
) -> dict:
    project_root = Path(project_root)
    resolved_config = config or _project_config(project_root)
    stored = dict(last_status or load_capability_status(project_root) or {})
    queue_items = _goal_queue_items(project_root=project_root, goal_queue=goal_queue, config=resolved_config)
    pending_goals = [item for item in queue_items if is_capability_goal(item)]

    matched_capabilities = list(stored.get("matched_capabilities", []))
    provisioning_results = list(stored.get("provisioning_results", []))
    pending_bootstrap_actions: list[str] = []
    running_bootstrap_actions: list[str] = []
    applied_bootstrap_actions: list[str] = []
    failed_bootstrap_actions: list[str] = []

    for item in provisioning_results:
        action_name = item.get("action")
        status = item.get("status")
        if not action_name:
            continue
        if status == "planned":
            pending_bootstrap_actions.append(action_name)
        elif status in {"started", "already_running"}:
            running_bootstrap_actions.append(action_name)
        elif status == "applied":
            applied_bootstrap_actions.append(action_name)
        elif status == "failed":
            failed_bootstrap_actions.append(action_name)

    if not provisioning_results:
        pending_bootstrap_actions.extend(
            item.get("action")
            for item in stored.get("provisioning_actions", [])
            if item.get("action")
        )

    dedupe = lambda values: list(dict.fromkeys(item for item in values if item))

    return {
        "configured": {
            "auto_add_capabilities": bool(_config_value(resolved_config, "auto_add_capabilities", True)),
            "auto_queue_missing_capabilities": bool(
                _config_value(resolved_config, "auto_queue_missing_capabilities", True)
            ),
            "auto_provision_mcp": bool(_config_value(resolved_config, "auto_provision_mcp", False)),
            "auto_start_mcp_servers": bool(_config_value(resolved_config, "auto_start_mcp_servers", False)),
        },
        "last_updated": stored.get("updated_at"),
        "last_goal": stored.get("last_goal"),
        "matched_capabilities": matched_capabilities,
        "matched_capability_ids": [
            item.get("capability_id")
            for item in matched_capabilities
            if item.get("capability_id")
        ],
        "recommended_skills": list(stored.get("recommended_skills", [])),
        "missing_skills": list(stored.get("missing_skills", [])),
        "queued_goals": list(stored.get("queued_goals", [])),
        "skipped_goals": list(stored.get("skipped_goals", [])),
        "queue_strategy": stored.get("queue_strategy"),
        "pending_self_development_goals": pending_goals,
        "pending_bootstrap_actions": dedupe(pending_bootstrap_actions),
        "running_bootstrap_actions": dedupe(running_bootstrap_actions),
        "applied_bootstrap_actions": dedupe(applied_bootstrap_actions),
        "failed_bootstrap_actions": dedupe(failed_bootstrap_actions),
    }


def capability_doctor_check(
    project_root: Path,
    *,
    goal_queue=None,
    config=None,
    last_status: dict | None = None,
) -> tuple[str, str]:
    report = build_capability_status_report(
        Path(project_root),
        goal_queue=goal_queue,
        config=config,
        last_status=last_status,
    )
    configured = report["configured"]
    status = "WARN" if report["failed_bootstrap_actions"] else "PASS"
    matched = ", ".join(report["matched_capability_ids"]) or "none recorded"
    pending = ", ".join(report["pending_bootstrap_actions"]) or "none"
    running = ", ".join(report["running_bootstrap_actions"]) or "none"
    last_goal = report["last_goal"] or "none recorded"
    detail = (
        f"last goal: {last_goal}; "
        f"auto_add={'on' if configured['auto_add_capabilities'] else 'off'}; "
        f"auto_queue={'on' if configured['auto_queue_missing_capabilities'] else 'off'}; "
        f"auto_provision={'on' if configured['auto_provision_mcp'] else 'off'}; "
        f"auto_start={'on' if configured['auto_start_mcp_servers'] else 'off'}; "
        f"matched: {matched}; "
        f"pending skill goals: {len(report['pending_self_development_goals'])}; "
        f"bootstrap pending: {pending}; "
        f"bootstrap running: {running}"
    )
    return status, detail


def provision_capability_actions(
    *,
    project_root: Path,
    provisioning_actions: Iterable[dict],
    auto_provision: bool,
    start_servers: bool,
    dry_run: bool,
) -> dict:
    """Execute bounded local capability provisioning actions."""
    actions = list(provisioning_actions)
    if not actions:
        return {"attempted": False, "results": []}

    if dry_run:
        return {
            "attempted": False,
            "results": [
                {
                    **action,
                    "status": "planned",
                    "skipped_reason": "dry_run",
                }
                for action in actions
            ],
        }

    if not auto_provision:
        return {
            "attempted": False,
            "results": [
                {
                    **action,
                    "status": "planned",
                    "skipped_reason": "auto_provision_disabled",
                }
                for action in actions
            ],
        }

    results: list[dict] = []
    for action in actions:
        action_name = action.get("action")
        if action_name == "ensure_mcp_servers":
            result = _run_mcp_setup(project_root, start_servers=start_servers)
            results.append({**action, **result})
            continue
        if action_name in {"start_skills_mcp_server", "start_github_mcp_bridge"} and not start_servers:
            results.append({**action, "status": "planned", "skipped_reason": "auto_start_disabled"})
            continue
        if action_name == "start_skills_mcp_server":
            results.append({**action, **_start_skills_mcp_server(project_root)})
            continue
        if action_name == "start_github_mcp_bridge":
            results.append({**action, **_start_github_mcp_bridge(project_root)})
            continue
        results.append({**action, "status": "failed", "error": f"unknown action: {action_name}"})

    return {"attempted": True, "results": results}
