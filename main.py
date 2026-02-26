# R1: Shim — canonical entry point lives in aura_cli/cli_main.py
from aura_cli.cli_main import main  # noqa: F401

if __name__ == "__main__":
    main()
