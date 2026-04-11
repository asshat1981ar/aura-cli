import json
import logging
from pathlib import Path
from typing import Dict, Any

_logger = logging.getLogger(__name__)


class FileUtils:
    @staticmethod
    def save_json(data: Dict[str, Any], output_path: str, indent: int = 2) -> None:
        """Save dictionary as JSON to specified path"""
        with open(output_path, "w") as f:
            json.dump(data, f, indent=indent)

    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        """Load JSON file into dictionary"""
        with open(file_path, "r") as f:
            return json.load(f)

    @staticmethod
    def ensure_path_exists(path: str) -> None:
        """Ensure parent directories exist for a file path"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)


class ConfigLoader:
    def __init__(self, config_path: str = "config/settings.json"):
        self.config_path = config_path
        self.settings = self._load_settings()

    def _load_settings(self) -> Dict[str, Any]:
        """Load configuration settings from file"""
        try:
            return FileUtils.load_json(self.config_path)
        except FileNotFoundError:
            return {}

    def get(self, key: str, default=None):
        """Get a configuration value by key"""
        return self.settings.get(key, default)


class ErrorHandler:
    @staticmethod
    def handle_file_error(operation: str, file_path: str, error: Exception) -> None:
        """Standardized error handling for file operations"""
        _logger.error("Error during %s on %s: %s", operation, file_path, str(error))
        raise


class Logger:
    @staticmethod
    def log_info(message: str) -> None:
        """Log informational message"""
        _logger.info("%s", message)

    @staticmethod
    def log_error(message: str) -> None:
        """Log error message"""
        _logger.error("%s", message)
