"""E2E test infrastructure for AURA CLI.

Uses subprocess to invoke the AURA CLI with a temp directory as working dir.
Provides deterministic mock responses via AURA_DRY_RUN=1 env flag.
"""
import subprocess
import tempfile
import shutil
import os
import json
import pytest


@pytest.fixture
def temp_project(tmp_path):
    """Copy sample_project to a temp dir for isolated E2E testing."""
    src = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'sample_project')
    dst = tmp_path / 'project'
    shutil.copytree(src, dst)
    return dst


@pytest.fixture
def run_aura(temp_project):
    """Run aura CLI command in temp_project dir, return (returncode, stdout, stderr)."""
    def _run(*args, env_extra=None, timeout=60):
        env = {**os.environ, 'AURA_SKIP_CHDIR': '1', 'AURA_DRY_RUN': '1'}
        if env_extra:
            env.update(env_extra)
        result = subprocess.run(
            ['python3', os.path.join(os.path.dirname(__file__), '..', '..', 'main.py')] + list(args),
            cwd=str(temp_project),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return result.returncode, result.stdout, result.stderr
    return _run


# ---------------------------------------------------------------------------
# Mock LLM response for OpenRouter API (pytest-httpserver or env-var injection)
# ---------------------------------------------------------------------------

MOCK_LLM_RESPONSE = {
    "id": "mock-response",
    "choices": [{
        "message": {
            "content": '# AURA_TARGET: utils.py\ndef add(a, b):\n    """Add two numbers."""\n    return a + b\n',
            "role": "assistant"
        },
        "finish_reason": "stop"
    }],
    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
}


@pytest.fixture
def mock_llm_env():
    """Return env vars that stub out real LLM calls with a mock response.

    Injects MOCK_LLM_RESPONSE as AURA_MOCK_LLM_RESPONSE so the CLI can
    short-circuit real API calls when AURA_DRY_RUN=1 is set.
    """
    return {
        'AURA_DRY_RUN': '1',
        'AURA_MOCK_LLM_RESPONSE': json.dumps(MOCK_LLM_RESPONSE),
        'OPENROUTER_API_KEY': 'mock-key-not-used',
    }
