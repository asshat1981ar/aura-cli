"""Shared helpers for dispatch handler modules (B4)."""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

from aura_cli.cli_options import attach_cli_warnings


def _print_json_payload(payload: dict, *, parsed=None, **json_kwargs) -> None:
    print(json.dumps(attach_cli_warnings(payload, parsed), **json_kwargs))


def _run_json_printing_callable_with_warnings(ctx, func, *args, **kwargs) -> None:
    warning_records = getattr(ctx.parsed, "warning_records", None) or []
    if not warning_records:
        func(*args, **kwargs)
        return

    buf = io.StringIO()
    with redirect_stdout(buf):
        func(*args, **kwargs)
    raw = buf.getvalue()
    if raw == "":
        return

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        print(raw, end="")
        return

    _print_json_payload(payload, parsed=ctx.parsed, indent=2)
