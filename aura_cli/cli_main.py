import os
from pathlib import Path

try:
    import readline
except ImportError:
    readline = None

from core.logging_utils import log_json

from aura_cli.commands import _handle_add, _handle_run, _handle_status, _handle_exit, _handle_help, _handle_doctor, _handle_clear

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
        except Exception as _swarm_err:
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
