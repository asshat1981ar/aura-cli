# R1: Shim — canonical entry point lives in aura_cli/cli_main.py.
# Keep this wrapper lazy so lightweight/help paths do not trigger full runtime imports.
import json
import sys

# Load environment variables before any other imports (dotenv is optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from aura_cli.cli_options import CLIParseError, attach_cli_warnings, parse_cli_args, render_help
from aura_cli.cli_options import cli_parse_error_payload, unknown_command_help_topic_payload


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

    from aura_cli.cli_main import main as _main
    try:
        sys.exit(_main(argv=raw_argv))
    except Exception as exc:
        if "--json" in raw_argv:
            print(json.dumps(attach_cli_warnings({
                "status": "error",
                "code": "unexpected_runtime_error",
                "message": str(exc)
            }, parsed)))
        else:
            raise
        sys.exit(1)
