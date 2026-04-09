from pathlib import Path
import os

from core.exceptions import ConfigurationError  # noqa: F401


def validate_api_key(api_key: str) -> None:
    if not api_key:
        raise ConfigurationError('API key is not set.')


def validate_cli_path(cli_path: str) -> None:
    path = Path(cli_path)
    if not path.is_file():
        raise ConfigurationError(f'CLI path {cli_path} is not configured or not executable.')
    if not os.access(cli_path, os.X_OK):
        raise ConfigurationError(f'CLI path {cli_path} is not executable.')
