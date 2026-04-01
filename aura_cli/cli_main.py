import io
import json
import os
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import readline
except ImportError:
    readline = None

# Re-exported for backward compatibility — tests and external code access these
# as attributes of this module. Do not remove these imports.
from memory.brain import Brain
from memory.store import MemoryStore
from agents.debugger import DebuggerAgent
from agents.planner import PlannerAgent
from agents.registry import default_agents
from agents.router import RouterAgent
from core.beads_bridge import BeadsBridge
from core.config_manager import ConfigManager, DEFAULT_CONFIG, config
from core.git_tools import GitTools
from core.goal_archive import GoalArchive
from core.goal_queue import GoalQueue
from core.logging_utils import log_json
from core.model_adapter import ModelAdapter
from core.orchestrator import LoopOrchestrator
from core.policy import Policy
from core.runtime_auth import resolve_config_api_key, runtime_provider_status, runtime_provider_summary
from core.runtime_paths import resolve_project_path
from core.task_handler import _check_project_writability, run_goals_loop
from core.vector_store import VectorStore
from agents.scaffolder import ScaffolderAgent

from aura_cli.cli_options import (
    CLIParseError,
    attach_cli_warnings,
    cli_parse_error_payload,
    parse_cli_args,
    render_help,
    unknown_command_help_topic_payload,
)
from aura_cli.options import action_runtime_required
from aura_cli.dispatch import (
    COMMAND_DISPATCH_REGISTRY,
    DispatchContext,
    DispatchRule,
    _dispatch_rule,
    _handle_bootstrap_dispatch,
    _handle_contract_report_dispatch,
    _handle_diag_dispatch,
    _handle_doctor_dispatch,
    _handle_evolve_dispatch,
    _handle_goal_add_dispatch,
    _handle_goal_add_run_dispatch,
    _handle_goal_once_dispatch,
    _handle_goal_run_dispatch,
    _handle_goal_status_dispatch,
    _handle_help_dispatch,
    _handle_json_help_dispatch,
    _handle_logs_dispatch,
    _handle_memory_reindex_dispatch,
    _handle_memory_search_dispatch,
    _handle_metrics_show_dispatch,
    _handle_mcp_call_dispatch,
    _handle_mcp_tools_dispatch,
    _handle_queue_clear_dispatch,
    _handle_queue_list_dispatch,
    _handle_readiness_dispatch,
    _handle_sadd_resume_dispatch,
    _handle_sadd_run_dispatch,
    _handle_sadd_status_dispatch,
    _handle_scaffold_dispatch,
    _handle_show_config_dispatch,
    _handle_watch_dispatch,
    _handle_workflow_run_dispatch,
    _maybe_add_goal,
    _prepare_runtime_context,
    _print_json_payload,
    _resolve_beads_runtime_override,
    _resolve_dispatch_action,
    _resolve_runtime_mode,
    _run_json_printing_callable_with_warnings,
    dispatch_command,
)
from aura_cli.mcp_client import cmd_diag, cmd_mcp_call, cmd_mcp_tools
from aura_cli.commands import (
    _handle_add, _handle_run, _handle_status, _handle_exit, _handle_help,
    _handle_doctor, _handle_clear, _handle_readiness,
    _handle_innovate_start, _handle_innovate_list, _handle_innovate_show,
    _handle_innovate_resume, _handle_innovate_export,
)

from aura_cli.interactive_shell import cli_interaction_loop as _cli_interaction_loop
import aura_cli.entrypoint as _entrypoint_mod
import aura_cli.runtime_factory as _runtime_factory_mod
from aura_cli.runtime_factory import (
    _attach_advanced_loops as _attach_advanced_loops_impl,
    _build_runtime_config as _build_runtime_config_impl,
    _init_memory_and_brain as _init_memory_and_brain_impl,
    _resolve_runtime_paths as _resolve_runtime_paths_impl,
    _start_background_sync as _start_background_sync_impl,
    create_runtime as _create_runtime_impl,
)
from aura_cli.entrypoint import main as _entrypoint_main_impl


def _sync_runtime_factory_compat() -> None:
    for name in (
        "Brain",
        "MemoryStore",
        "DebuggerAgent",
        "PlannerAgent",
        "default_agents",
        "RouterAgent",
        "BeadsBridge",
        "ConfigManager",
        "DEFAULT_CONFIG",
        "config",
        "GitTools",
        "GoalArchive",
        "GoalQueue",
        "log_json",
        "ModelAdapter",
        "LoopOrchestrator",
        "Policy",
        "resolve_config_api_key",
        "runtime_provider_status",
        "runtime_provider_summary",
        "resolve_project_path",
        "VectorStore",
        "_build_runtime_config",
        "_resolve_runtime_paths",
        "_init_memory_and_brain",
        "_start_background_sync",
        "_attach_advanced_loops",
    ):
        setattr(_runtime_factory_mod, name, globals()[name])


def _build_runtime_config(overrides: dict | None = None) -> dict:
    _sync_runtime_factory_compat()
    return _build_runtime_config_impl(overrides)


def _resolve_runtime_paths(project_root: Path, runtime_config: dict | None = None) -> dict:
    _sync_runtime_factory_compat()
    return _resolve_runtime_paths_impl(project_root, runtime_config=runtime_config)


def _init_memory_and_brain(brain_db_path: Path):
    _sync_runtime_factory_compat()
    return _init_memory_and_brain_impl(brain_db_path)


def _start_background_sync(project_root: Path, vector_store, context_graph):
    _sync_runtime_factory_compat()
    return _start_background_sync_impl(project_root, vector_store, context_graph)


def _attach_advanced_loops(orchestrator, runtime_mode, brain, memory_store, goal_queue, momento, project_root):
    _sync_runtime_factory_compat()
    return _attach_advanced_loops_impl(orchestrator, runtime_mode, brain, memory_store, goal_queue, momento, project_root)


def create_runtime(project_root: Path, overrides: dict | None = None):
    _sync_runtime_factory_compat()
    runtime = _create_runtime_impl(project_root, overrides=overrides)
    if os.environ.get("AURA_ENABLE_SWARM", "0") == "1":
        try:
            from core.swarm_supervisor import install_swarm_runtime
            orchestrator = runtime.get("orchestrator")
            registry = runtime.get("agents", {})
            if orchestrator is not None:
                install_swarm_runtime(orchestrator=orchestrator, registry=registry)
                log_json("INFO", "swarm_runtime_activated", details={"project_root": str(project_root)})
        except (ImportError, OSError, RuntimeError) as _swarm_err:
            log_json("WARN", "swarm_runtime_activation_failed", details={"error": str(_swarm_err)})
    return runtime


def cli_interaction_loop(args, runtime):
    return _cli_interaction_loop(
        args,
        runtime,
        create_runtime,
        input_func=input,
        handle_add=_handle_add,
        handle_run=_handle_run,
        handle_status=_handle_status,
        handle_doctor=_handle_doctor,
        handle_clear=_handle_clear,
        handle_help=_handle_help,
        handle_exit=_handle_exit,
        log_json_func=log_json,
    )


def _sync_entrypoint_compat() -> None:
    for name in (
        "CLIParseError",
        "attach_cli_warnings",
        "cli_parse_error_payload",
        "dispatch_command",
        "json",
        "os",
        "parse_cli_args",
        "Path",
        "readline",
        "sys",
    ):
        setattr(_entrypoint_mod, name, globals()[name])


def main(project_root_override=None, argv=None):
    _sync_entrypoint_compat()
    return _entrypoint_main_impl(project_root_override=project_root_override, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
