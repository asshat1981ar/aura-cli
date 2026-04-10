from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:
    import readline
except ImportError:  # pragma: no cover - platform-specific
    readline = None

from aura_cli.cli_options import CLIParseError, attach_cli_warnings, cli_parse_error_payload, parse_cli_args
from aura_cli.dispatch import dispatch_command

# Import new components from sub-agent deliverables
from core.container import Container
from aura_cli.error_presenter import present_error


def _initialize_container() -> None:
    """Initialize the dependency injection container with core services."""
    # Register configuration manager
    from core.config_manager import ConfigManager
    if "config_manager" not in Container._singletons:
        Container.register_singleton(ConfigManager, ConfigManager())
    
    # Register model adapter
    from core.model_adapter import ModelAdapter
    if "model_adapter" not in Container._singletons:
        Container.register_singleton(ModelAdapter, ModelAdapter())


def _resolve_project_root(project_root_override=None) -> Path:
    if project_root_override:
        return Path(project_root_override)

    project_root = Path(__file__).resolve().parent.parent
    if os.getenv("AURA_SKIP_CHDIR") == "1":
        return Path.cwd()

    os.chdir(project_root)
    return project_root


def _ensure_project_on_path(project_root: Path) -> None:
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _load_history(project_root: Path) -> None:
    if not readline:
        return

    history_file = project_root / "memory" / ".aura_history"
    try:
        if history_file.exists():
            readline.read_history_file(str(history_file))
        readline.set_history_length(1000)
    except Exception:
        pass


def _save_history(project_root: Path) -> None:
    if not readline:
        return

    try:
        readline.write_history_file(str(project_root / "memory" / ".aura_history"))
    except Exception:
        pass


def main(project_root_override=None, argv=None):
    # Initialize dependency injection container
    _initialize_container()
    
    project_root = _resolve_project_root(project_root_override)
    _ensure_project_on_path(project_root)
    _load_history(project_root)

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    
    # Check for global flags
    verbose = "--verbose" in raw_argv or "-v" in raw_argv
    json_output = "--json" in raw_argv
    
    try:
        parsed = parse_cli_args(raw_argv)
    except CLIParseError as exc:
        if json_output:
            print(json.dumps(attach_cli_warnings(cli_parse_error_payload(exc))))
        else:
            # Use enhanced error presenter for CLI parse errors
            present_error(
                exc,
                verbose=verbose,
                json_output=json_output,
                command="aura"
            )
        return exc.code

    try:
        return dispatch_command(parsed, project_root=project_root)
    except Exception as exc:
        # Use enhanced error presenter for runtime errors
        if json_output:
            import json
            print(json.dumps({"error": str(exc), "type": type(exc).__name__}))
        else:
            present_error(
                exc,
                verbose=verbose,
                json_output=json_output,
                command=getattr(parsed, 'command', 'aura')
            )
        return 1
    finally:
        _save_history(project_root)
