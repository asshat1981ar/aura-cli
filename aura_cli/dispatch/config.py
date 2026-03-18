"""Dispatch handlers for config/help/doctor commands (B4)."""
from __future__ import annotations

import json
import sys

from aura_cli.dispatch._helpers import _print_json_payload

from core.config_manager import config
from aura_cli.commands import _handle_doctor
from aura_cli.cli_options import attach_cli_warnings, render_help, unknown_command_help_topic_payload


def handle_help(ctx) -> int:
    try:
        print(render_help(getattr(ctx.args, "help_topics", None)))
    except ValueError as exc:
        if getattr(ctx.args, "json", False):
            print(json.dumps(attach_cli_warnings(unknown_command_help_topic_payload(str(exc)), ctx.parsed)))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


def handle_json_help(_ctx) -> int:
    print(render_help(format="json"))
    return 0


def handle_doctor(ctx) -> int:
    _handle_doctor(ctx.project_root)
    return 0


def handle_bootstrap(_ctx) -> int:
    config.interactive_bootstrap()
    return 0


def handle_show_config(_ctx) -> int:
    """Print the resolved effective configuration as JSON."""
    print(json.dumps(config.show_config(), indent=2, default=str))
    return 0


def handle_contract_report(ctx) -> int:
    from aura_cli.contract_report import (
        build_cli_contract_report,
        cli_contract_report_exit_code,
        cli_contract_report_failure_message,
        render_cli_contract_report,
    )

    # Import registry lazily to avoid circular import
    from aura_cli.cli_main import COMMAND_DISPATCH_REGISTRY

    report = build_cli_contract_report(
        include_dispatch=not getattr(ctx.args, "no_dispatch", False),
        dispatch_registry=COMMAND_DISPATCH_REGISTRY,
    )
    print(render_cli_contract_report(report, compact=getattr(ctx.args, "compact", False)), end="")

    exit_code = cli_contract_report_exit_code(report, check=getattr(ctx.args, "check", False))
    if exit_code:
        print(cli_contract_report_failure_message(report), file=sys.stderr)
    return exit_code
