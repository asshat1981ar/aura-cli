import sys
from pathlib import Path
from core.logging_utils import log_json
from cli.cli_main import main as cli_main_main

def main(project_root_override=None):
    """Delegates to the CLI main entry point."""
    cli_main_main(project_root_override=project_root_override)

if __name__ == "__main__":
    main()
