from __future__ import annotations

import importlib
import io
import json
from contextlib import redirect_stdout
from typing import Any, Mapping

from aura_cli.cli_options import cli_contract_report


def build_cli_contract_report(
    *,
    include_dispatch: bool = True,
    dispatch_registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not include_dispatch:
        return cli_contract_report()
    if dispatch_registry is not None:
        return cli_contract_report(dispatch_registry=dispatch_registry)

    with redirect_stdout(io.StringIO()):
        cli_main = importlib.import_module("aura_cli.cli_main")
    return cli_contract_report(dispatch_registry=cli_main.COMMAND_DISPATCH_REGISTRY)


def render_cli_contract_report(report: Mapping[str, Any], *, compact: bool = False) -> str:
    if compact:
        return json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    return json.dumps(report, sort_keys=True, indent=2) + "\n"


def cli_contract_report_exit_code(report: Mapping[str, Any], *, check: bool) -> int:
    if check and not report.get("ok", False):
        return 1
    return 0


def cli_contract_report_failure_message(report: Mapping[str, Any]) -> str:
    failures = ", ".join(report.get("failure_keys", [])) or "unknown"
    return f"CLI contract failures: {failures}"


__all__ = [
    "build_cli_contract_report",
    "cli_contract_report_exit_code",
    "cli_contract_report_failure_message",
    "render_cli_contract_report",
]
