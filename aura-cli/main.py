# Refactored for clarity and performance


def setup():
    """Entry point setup — delegates to the main aura_cli package."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from aura_cli.cli_main import main
    main()


if __name__ == "__main__":
    setup()