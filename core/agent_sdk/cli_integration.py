# core/agent_sdk/cli_integration.py
"""Wire the Agent SDK meta-controller into AURA CLI commands."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.agent_sdk.config import AgentSDKConfig

logger = logging.getLogger(__name__)


def build_controller_from_args(args: Any) -> Any:
    """Build an AuraController from parsed CLI args."""
    from core.agent_sdk.controller import AuraController

    # Load base config from aura.config.json if available
    config = AgentSDKConfig()
    try:
        config_path = Path("aura.config.json")
        if config_path.exists():
            with open(config_path) as f:
                aura_config = json.load(f)
            config = AgentSDKConfig.from_aura_config(aura_config)
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        pass  # Fall back to defaults

    # Apply CLI overrides
    if getattr(args, "model", None):
        config.model = args.model
    if getattr(args, "max_turns", None):
        config.max_turns = args.max_turns
    if getattr(args, "max_budget", None):
        config.max_budget_usd = args.max_budget
    if getattr(args, "permission_mode", None):
        config.permission_mode = args.permission_mode

    # Apply env overrides last
    config.apply_env_overrides()

    project_root = Path(getattr(args, "project_root", ".")).resolve()

    # Try to load brain and model adapter from existing AURA infra
    brain = None
    model_adapter = None
    try:
        from memory.brain import Brain
        brain = Brain()
    except (ImportError, OSError, RuntimeError):
        pass
    try:
        from core.model_adapter import ModelAdapter
        model_adapter = ModelAdapter()
    except (ImportError, OSError, RuntimeError):
        pass

    # Initialize production subsystems
    from core.agent_sdk.model_router import AdaptiveModelRouter
    from core.agent_sdk.session_persistence import SessionStore
    from core.agent_sdk.feedback import SkillWeightUpdater, FeedbackCollector

    model_router = AdaptiveModelRouter(
        stats_path=config.model_stats_path,
        ema_alpha=config.ema_alpha,
        min_success_rate=config.min_success_rate,
        escalation_threshold=config.escalation_threshold,
        de_escalation_threshold=config.de_escalation_threshold,
    )
    session_store = SessionStore(db_path=config.session_db_path)
    skill_updater = SkillWeightUpdater(
        weights_path=config.skill_weights_path,
        success_delta=config.skill_weight_success_delta,
        failure_delta=config.skill_weight_failure_delta,
        cap=config.skill_weight_cap,
        floor=config.skill_weight_floor,
    )
    feedback = FeedbackCollector(
        model_router=model_router,
        skill_updater=skill_updater,
        brain=brain,
        session_store=session_store,
    )

    return AuraController(
        config=config,
        project_root=project_root,
        brain=brain,
        model_adapter=model_adapter,
        model_router=model_router,
        session_store=session_store,
        feedback=feedback,
    )


def format_result(result: Dict[str, Any]) -> str:
    """Format controller result for CLI output."""
    parts = []

    if result.get("result"):
        parts.append(result["result"])

    if result.get("session_id"):
        parts.append(f"\n--- Session: {result['session_id']} ---")

    if result.get("total_cost_usd"):
        parts.append(f"Cost: ${result['total_cost_usd']:.2f}")

    if result.get("metrics"):
        m = result["metrics"]
        parts.append(
            f"Metrics: {m.get('total_calls', 0)} tool calls, "
            f"{m.get('success_rate', 0):.0%} success rate"
        )

    return "\n".join(parts)


def handle_agent_status(session_store: Any, limit: int = 20) -> list:
    """Return a list of recent sessions from the session store."""
    return session_store.list_sessions(limit=limit)


def handle_agent_cost(session_store: Any, days: int = 7) -> Dict[str, Any]:
    """Return a cost summary dict from the session store."""
    return session_store.get_cost_summary(days=days)


async def handle_agent_run(args: Any) -> int:
    """CLI handler for 'agent-run' command."""
    goal = getattr(args, "goal", None)
    if not goal:
        print("Error: --goal is required")
        return 1

    controller = build_controller_from_args(args)

    try:
        result = await controller.run(goal)
        print(format_result(result))
        return 0
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1
    except Exception as exc:  # Catch-all: CLI top-level handler must not crash
        logger.exception("Agent run failed")
        print(f"Error: {exc}")
        return 1


def handle_agent_scan(
    project_root: Path,
    db_path: Path,
    exclude_patterns: Optional[List[str]] = None,
    model_adapter: Any = None,
    no_llm: bool = False,
) -> Dict[str, Any]:
    """Run a semantic scan of the codebase."""
    from core.agent_sdk.semantic_scanner import SemanticScanner
    scanner = SemanticScanner(
        project_root=project_root,
        db_path=db_path,
        exclude_patterns=exclude_patterns or [".git", "__pycache__", "node_modules"],
        model_adapter=None if no_llm else model_adapter,
    )
    return scanner.scan_full()


def format_scan_stats(db_path: Path) -> str:
    """Format scan statistics for CLI output."""
    from core.agent_sdk.semantic_schema import SemanticDB
    db = SemanticDB(db_path)
    meta = db.get_last_scan()
    if not meta:
        return "No scan data available. Run 'agent scan' first."
    files = db.get_all_files()
    return (
        f"Last scan: {meta['scan_time']}\n"
        f"Type: {meta['scan_type']}, SHA: {meta['scan_sha'][:8]}\n"
        f"Files: {len(files)}, Symbols: {meta['symbols_found']}\n"
        f"LLM calls: {meta['llm_calls_made']}, Cost: ${meta['llm_cost_usd']:.3f}"
    )
