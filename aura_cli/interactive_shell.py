from __future__ import annotations

from builtins import input as builtin_input
from types import SimpleNamespace

from aura_cli.commands import (
    _handle_add,
    _handle_clear,
    _handle_doctor,
    _handle_exit,
    _handle_help,
    _handle_run,
    _handle_status,
)
from core.logging_utils import log_json


def cli_interaction_loop(
    args,
    runtime,
    create_runtime,
    *,
    input_func=builtin_input,
    handle_add=_handle_add,
    handle_run=_handle_run,
    handle_status=_handle_status,
    handle_doctor=_handle_doctor,
    handle_clear=_handle_clear,
    handle_help=_handle_help,
    handle_exit=_handle_exit,
    log_json_func=log_json,
):
    project_root = runtime["project_root"]

    while True:
        try:
            command_line = input_func("\nCommand (add/run/exit/status/help) > ").strip()
            command_parts = command_line.split(maxsplit=1)
            if not command_parts:
                continue
            cmd_name = command_parts[0]

            if cmd_name == "add":
                handle_add(runtime["goal_queue"], command_line)
            elif cmd_name == "run":
                orchestrator = runtime.get("orchestrator")
                if isinstance(orchestrator, SimpleNamespace):
                    print("Initializing full AURA runtime for execution...")
                    full_runtime = create_runtime(project_root, overrides={"runtime_mode": "full"})
                    runtime.update(full_runtime)

                orchestrator = runtime.get("orchestrator")
                handle_run(args, runtime["goal_queue"], runtime["goal_archive"], orchestrator, runtime["debugger"], runtime["planner"], project_root)
            elif cmd_name == "status":
                handle_status(
                    runtime["goal_queue"],
                    runtime["goal_archive"],
                    runtime.get("orchestrator"),
                    project_root=project_root,
                    memory_persistence_path=runtime.get("memory_persistence_path"),
                    memory_store=runtime.get("memory_store"),
                )
            elif cmd_name == "doctor":
                handle_doctor(project_root)
            elif cmd_name == "clear":
                handle_clear()
            elif cmd_name == "help":
                handle_help()
            elif cmd_name == "exit":
                handle_exit()
                break
            else:
                log_json_func("WARN", "invalid_cli_command", details={"command": cmd_name})
        except EOFError:
            log_json_func("INFO", "aura_cli_eof_received", details={"message": "End of input received, exiting."})
            break
