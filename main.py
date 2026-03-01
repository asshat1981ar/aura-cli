# R1: Shim — canonical entry point lives in aura_cli/cli_main.py.
# Keep this wrapper lazy so lightweight/help paths do not trigger full runtime imports.
import json
import sys

from aura_cli.cli_options import CLIParseError, attach_cli_warnings, parse_cli_args, render_help


def main(*args, **kwargs):  # noqa: D401
    """Proxy to the canonical CLI entry point."""
    from aura_cli.cli_main import main as _main
    return _main(*args, **kwargs)

if __name__ == "__main__":
    raw_argv = sys.argv[1:]

    if "--json-help" in raw_argv:
        print(render_help(format="json"))
        raise SystemExit(0)

    try:
        parsed = parse_cli_args(raw_argv)
    except CLIParseError as exc:
        if "--json" in raw_argv:
            print(json.dumps(attach_cli_warnings({
                "status": "error",
                "code": "cli_parse_error",
                "message": str(exc),
                "usage": exc.usage,
            })))
        else:
            print(f"Error: {exc}", file=sys.stderr)
            if exc.usage:
                print(exc.usage, file=sys.stderr)
        raise SystemExit(exc.code)

    if parsed.command == "help":
        try:
            print(render_help(getattr(parsed.namespace, "help_topics", None)))
            raise SystemExit(0)
        except ValueError as exc:
            if getattr(parsed.namespace, "json", False):
                print(json.dumps(attach_cli_warnings({
                    "status": "error",
                    "code": "unknown_command_help_topic",
                    "message": str(exc),
                }, parsed)))
            else:
                print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(2)

    raise SystemExit(main(argv=raw_argv))
