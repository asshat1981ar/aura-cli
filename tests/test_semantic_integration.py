# tests/test_semantic_integration.py
"""Integration test: full scan → query → context injection."""

import tempfile
import unittest
from pathlib import Path


class TestSemanticEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.project = Path(self.tmpdir) / "project"
        self.project.mkdir()
        self.db_path = Path(self.tmpdir) / "index.db"

        # Create a small test project
        (self.project / "auth.py").write_text('def authenticate(user, pwd):\n    """Validate credentials."""\n    return True\n\ndef create_token(user):\n    """Create JWT token."""\n    return "token"\n')
        (self.project / "server.py").write_text("from auth import authenticate\n\ndef handle(req):\n    auth = authenticate(req.user, req.pwd)\n    return auth\n")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_scan_then_query(self):
        from core.agent_sdk.semantic_scanner import SemanticScanner
        from core.agent_sdk.semantic_querier import SemanticQuerier

        scanner = SemanticScanner(
            project_root=self.project,
            db_path=self.db_path,
            exclude_patterns=["__pycache__"],
        )
        result = scanner.scan_full()
        self.assertGreater(result["files_scanned"], 0)

        querier = SemanticQuerier(self.db_path)

        # Test all query types
        callers = querier.what_calls("authenticate")
        self.assertGreater(len(callers), 0)

        deps = querier.what_depends_on("auth.py")
        self.assertGreater(len(deps), 0)

        overview = querier.architecture_overview()
        self.assertGreater(overview["total_files"], 0)

        summary = querier.summarize("authenticate")
        self.assertIsInstance(summary, str)

    def test_context_builder_with_semantic_index(self):
        from core.agent_sdk.semantic_scanner import SemanticScanner
        from core.agent_sdk.semantic_querier import SemanticQuerier
        from core.agent_sdk.context_builder import ContextBuilder

        scanner = SemanticScanner(
            project_root=self.project,
            db_path=self.db_path,
            exclude_patterns=["__pycache__"],
        )
        scanner.scan_full()
        querier = SemanticQuerier(self.db_path)

        builder = ContextBuilder(
            project_root=self.project,
            semantic_querier=querier,
        )
        ctx = builder.build(goal="Fix authentication bug")
        self.assertIn("codebase_overview", ctx)
        self.assertIn("relevant_symbols", ctx)

        prompt = builder.build_system_prompt(
            goal="Fix auth bug",
            goal_type="bug_fix",
            context=ctx,
        )
        self.assertIn("Codebase Understanding", prompt)


if __name__ == "__main__":
    unittest.main()
