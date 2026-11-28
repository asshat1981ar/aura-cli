"""Typer front end for the AURA CLI.

The argparse CLI in :mod:`aura_cli.cli_options` stays the source of truth:
its parser, help text, snapshots and contract report are unchanged. This
module adds a Typer application with one command per top-level command in
:data:`aura_cli.options.COMMAND_SPECS`. Each command forwards its arguments to
the legacy entry point, so both front ends dispatch exactly the same way.

:func:`build_main` returns a Typer app that is also callable the legacy way,
``main(project_root_override=..., argv=...)``, so ``aura_cli.cli_main.main``
works for existing callers, for the ``aura`` console script and for
``typer.testing.CliRunner``. Set ``AURA_USE_TYPER=1`` to run the Typer front
end when ``main()`` is called with no arguments (the console script); by
default that still runs the legacy argparse path.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import typer

from aura_cli.options import COMMAND_SPECS

LegacyMain = Callable[..., Optional[int]]

_PASSTHROUGH_SETTINGS = {"allow_extra_args": True, "ignore_unknown_options": True}


def _top_level_commands() -> dict[str, str]:
    """Map each top-level command name to its summary, in spec order."""
    commands: dict[str, str] = {}
    for spec in COMMAND_SPECS:
        name = spec.path[0]
        if len(spec.path) == 1:
            commands[name] = spec.summary
        else:
            commands.setdefault(name, "")
    return commands


def _version_text() -> str:
    try:
        from aura_cli import __version__
    except ImportError:  # pragma: no cover - package metadata missing
        __version__ = "unknown"
    return f"aura {__version__}"


class CompatTyper(typer.Typer):
    """A Typer app that can still be called like the legacy ``main``."""

    def __init__(self, legacy_main: LegacyMain, **kwargs) -> None:
        super().__init__(**kwargs)
        self._legacy_main = legacy_main

    def __call__(self, project_root_override=None, argv=None, *args, **kwargs):
        if (
            project_root_override is None
            and argv is None
            and not args
            and not kwargs
            and os.getenv("AURA_USE_TYPER") == "1"
        ):
            return super().__call__()
        return self._legacy_main(project_root_override=project_root_override, argv=argv)


def build_main(legacy_main: LegacyMain) -> CompatTyper:
    """Build the Typer app, forwarding every command to ``legacy_main``."""

    app = CompatTyper(
        legacy_main,
        name="aura",
        help="AURA autonomous development CLI.",
        add_completion=False,
        no_args_is_help=False,
        context_settings={"help_option_names": ["-h", "--help"]},
    )

    @app.callback(invoke_without_command=True)
    def _root(
        ctx: typer.Context,
        version: bool = typer.Option(False, "--version", is_eager=True, help="Show the version and exit."),
        dry_run: bool = typer.Option(False, "--dry-run", help="Do not apply filesystem changes."),
        json_output: bool = typer.Option(False, "--json", help="Emit JSON output where supported."),
        verbose: bool = typer.Option(False, "-v", "--verbose", help="Show more detail on errors."),
        quiet: bool = typer.Option(False, "-q", "--quiet", help="Show less output."),
    ) -> None:
        if version:
            typer.echo(_version_text())
            raise typer.Exit(0)
        ctx.obj = {"dry_run": dry_run, "json": json_output, "verbose": verbose, "quiet": quiet}
        if ctx.invoked_subcommand is None:
            typer.echo(ctx.get_help())
            raise typer.Exit(0)

    def _make_command(name: str) -> Callable[[typer.Context], None]:
        def _command(ctx: typer.Context) -> None:
            flags = (ctx.obj or {}) if isinstance(ctx.obj, dict) else {}
            argv = [name, *ctx.args]
            if flags.get("dry_run"):
                argv.append("--dry-run")
            if flags.get("json"):
                argv.append("--json")
            code = legacy_main(argv=argv)
            raise typer.Exit(code or 0)

        return _command

    for name, summary in _top_level_commands().items():
        app.command(name=name, help=summary or None, context_settings=_PASSTHROUGH_SETTINGS)(_make_command(name))

    return app
