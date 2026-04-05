from subprocess import run, CalledProcessError
from typing import Optional


def execute_api_command(api_key: Optional[str], command: str) -> str:
    """Executes a command using a given API key. Raises an exception if key is missing or command fails."""
    if not api_key:
        raise ValueError('API key must be provided.')
    try:
        result = run(command, shell=True, check=True, capture_output=True, env={'API_KEY': api_key})
        return result.stdout.decode('utf-8')
    except CalledProcessError as e:
        raise RuntimeError(f'Command failed with exit code {e.returncode}: {e.stderr.decode()}')

# Example usage in an agent class
class OpenAI:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def call(self, prompt: str) -> str:
        command = f"openai api complete --prompt '{prompt}'"
        return execute_api_command(self.api_key, command)
