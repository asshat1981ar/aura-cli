"""
Golden file tests for CLI output of security-related components.
Verifies that help text, class names, and module repr don't drift.

Update snapshots with: pytest --snapshot-update
"""

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = str(Path(__file__).parent.parent.parent)


class TestCLIOutputGolden:

    def test_main_help_includes_expected_sections(self, snapshot):
        """Top-level help output structure must match snapshot."""
        result = subprocess.run(
            [sys.executable, "main.py", "help"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        # Extract only the first line (title) for deterministic snapshot
        first_line = result.stdout.strip().split("\n")[0]
        assert first_line == snapshot

    def test_dpop_session_class_name(self, snapshot):
        """DPoPSession type name must match snapshot for documentation purposes."""
        from core.security.http_client import DPoPSession

        session = DPoPSession()
        assert type(session).__name__ == snapshot

    def test_dpop_generator_class_name(self, snapshot):
        """DPoPProofGenerator type name must match snapshot."""
        from core.security.dpop import DPoPProofGenerator

        gen = DPoPProofGenerator()
        assert type(gen).__name__ == snapshot

    def test_credential_store_subclasses(self, snapshot):
        """Registered CredentialStore subclass names must match snapshot."""
        from core.security.credential_store import CredentialStore

        subclass_names = sorted(cls.__name__ for cls in CredentialStore.__subclasses__())
        assert subclass_names == snapshot

    def test_file_store_default_filename(self, snapshot):
        """FileStore credential filename must match snapshot."""
        from core.security.file_store import FileStore

        assert FileStore._FILENAME == snapshot

    def test_store_factory_fallback_warning_text(self, snapshot):
        """The fallback warning constant must match snapshot."""
        from core.security.store_factory import _FALLBACK_WARNING

        assert _FALLBACK_WARNING == snapshot
