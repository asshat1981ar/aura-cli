import os
import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    pass

@dataclass
class ServiceConfig:
    openai_api_key: str
    openrouter_api_key: str
    anthropic_api_key: str
    codex_cli_path: Path
    copilot_cli_path: Path
    gemini_cli_path: Path

    @staticmethod
    def load_from_env() -> 'ServiceConfig':
        openai_key = os.getenv('OPENAI_API_KEY')
        openrouter_key = os.getenv('OPENROUTER_API_KEY')
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        codex_path = Path(os.getenv('CODEX_CLI_PATH', ''))
        copilot_path = Path(os.getenv('COPILOT_CLI_PATH', ''))
        gemini_path = Path(os.getenv('GEMINI_CLI_PATH', ''))

        if not all([openai_key, openrouter_key, anthropic_key]):
            logger.error('Missing API keys.')
            raise ConfigurationError('One or more API keys are not set.')

        for path in [codex_path, copilot_path, gemini_path]:
            if path and not path.is_file():
                logger.error(f'Path to CLI not executable: {path}')
                raise ConfigurationError(f'Path to CLI not executable: {path}')

        return ServiceConfig(openai_api_key=openai_key, openrouter_api_key=openrouter_key,
                              anthropic_api_key=anthropic_key, codex_cli_path=codex_path,
                              copilot_cli_path=copilot_path, gemini_cli_path=gemini_path)
