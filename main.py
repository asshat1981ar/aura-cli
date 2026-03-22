# Shim — canonical entry point lives in aura_cli/cli_main.py.
# aura_cli.cli_main is imported lazily (line 48) to prevent module-level
# ConfigManager instantiation from emitting logs on lightweight paths.
import json
import sys
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from aura_cli.cli_options import (
    CLIParseError,
    attach_cli_warnings,
    cli_parse_error_payload,
    parse_cli_args,
    render_help,
    unknown_command_help_topic_payload,
)

if __name__ == "__main__":
    raw_argv = sys.argv[1:]

    if "--json-help" in raw_argv:
        print(render_help(format="json"))
        sys.exit(0)

    try:
        parsed = parse_cli_args(raw_argv)
    except CLIParseError as exc:
        if "--json" in raw_argv:
            print(json.dumps(attach_cli_warnings(cli_parse_error_payload(exc))))
        else:
            print(f"Error: {exc}", file=sys.stderr)
            if exc.usage:
                print(exc.usage, file=sys.stderr)
        sys.exit(exc.code)

    if parsed.command == "help":
        try:
            print(render_help(getattr(parsed.namespace, "help_topics", None)))
            sys.exit(0)
        except ValueError as exc:
            if getattr(parsed.namespace, "json", False):
                print(json.dumps(attach_cli_warnings(unknown_command_help_topic_payload(str(exc)), parsed)))
            else:
                print(f"Error: {exc}", file=sys.stderr)
            sys.exit(2)

    from aura_cli.cli_main import main
    try:
        sys.exit(main(argv=raw_argv))
    except Exception as exc:
        # Ensure JSON callers receive a structured error payload instead of a traceback.
        if getattr(parsed.namespace, "json", False) or "--json" in raw_argv:
            payload = {
                "error": "unexpected_runtime_error",
                "message": str(exc),
            }
            print(json.dumps(attach_cli_warnings(payload)))
            sys.exit(1)
        # For non-JSON paths, re-raise so the default traceback behavior is preserved.
        raise
