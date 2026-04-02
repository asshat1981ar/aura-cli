# core/agent_sdk/cli_integration.py
"""Wire the Agent SDK meta-controller into AURA CLI commands."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def build_controller_from_args(args: Any) -> Any:
    """Build an AuraController from parsed CLI args."""
    from core.agent_sdk.config import AgentSDKConfig
    from core.agent_sdk.controller import AuraController

    # Load base config from aura.config.json if available
    config = AgentSDKConfig()
    try:
        config_path = Path("aura.config.json")
        if config_path.exists():
            with open(config_path) as f:
                aura_config = json.load(f)
            config = AgentSDKConfig.from_aura_config(aura_config)
    except Exception:
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
    except Exception:
        pass
    try:
        from core.model_adapter import ModelAdapter
        model_adapter = ModelAdapter()
    except Exception:
        pass

    return AuraController(
        config=config,
        project_root=project_root,
        brain=brain,
        model_adapter=model_adapter,
    )


def format_result(result: Dict[str, Any]) -> str:
    """Format controller result for CLI output."""
    parts = []

    if result.get("result"):
        parts.append(result["result"])

    if result.get("session_id"):
        parts.append(f"\n--- Session: {result['session_id']} ---")

    if result.get("metrics"):
        m = result["metrics"]
        parts.append(
            f"Metrics: {m.get('total_calls', 0)} tool calls, "
            f"{m.get('success_rate', 0):.0%} success rate"
        )

    return "\n".join(parts)


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
    except Exception as exc:
        logger.exception("Agent run failed")
        print(f"Error: {exc}")
        return 1
