"""Deprecated compatibility shim for legacy ``cli.commands`` imports."""

from aura_cli.commands import (
    _handle_add,
    _handle_clear,
    _handle_doctor,
    _handle_exit,
    _handle_help,
    _handle_run,
    _handle_status,
)

__all__ = [
    "_handle_add",
    "_handle_clear",
    "_handle_doctor",
    "_handle_exit",
    "_handle_help",
    "_handle_run",
    "_handle_status",
]
