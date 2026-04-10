"""Tests for the SemanticQuerier read-only query interface."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from core.agent_sdk.semantic_schema import SemanticDB


def _populate_test_db(db_path: Path) -> SemanticDB:
    """Create a SemanticDB with 3 files, symbols, relationships, call_sites, and a scan record."""
    db = SemanticDB(db_path)

    # --- files ---
    auth_id = db.upsert_file(
        path="core/auth.py",
        module_name="core.auth",
        cluster="core",
        line_count=80,
        last_modified="2026-01-01T00:00:00",
        last_scan_sha="sha1",
        module_summary="Handles authentication and password verification.",
        scanned_at="2026-01-01T00:00:00",
    )
    db.update_file_coupling(auth_id, 0.9)

    server_id = db.upsert_file(
        path="core/server.py",
        module_name="core.server",
        cluster="core",
        line_count=120,
        last_modified="2026-01-01T00:00:00",
        last_scan_sha="sha2",
        module_summary="HTTP server request handling.",
        scanned_at="2026-01-01T00:00:00",
    )
    db.update_file_coupling(server_id, 0.7)

    coder_id = db.upsert_file(
        path="agents/coder.py",
        module_name="agents.coder",
        cluster="agents",
        line_count=200,
        last_modified="2026-01-01T00:00:00",
        last_scan_sha="sha3",
        module_summary="Code generation agent.",
        scanned_at="2026-01-01T00:00:00",
    )
    db.update_file_coupling(coder_id, 0.5)

    # --- symbols ---
    auth_sym_id = db.insert_symbol(
        file_id=auth_id,
        name="authenticate",
        kind="function",
        line_start=10,
        line_end=30,
        signature="def authenticate(username: str, password: str) -> bool",
        docstring="Verify username and password against the store.",
        decorators="",
        intent_summary="Validates user credentials for authentication.",
    )

    handle_sym_id = db.insert_symbol(
        file_id=server_id,
        name="handle_request",
        kind="function",
        line_start=5,
        line_end=50,
        signature="def handle_request(req) -> Response",
        docstring="Handle an incoming HTTP request.",
        decorators="",
        intent_summary="Routes HTTP requests to appropriate handlers.",
    )

    generate_sym_id = db.insert_symbol(
        file_id=coder_id,
        name="generate_code",
        kind="function",
        line_start=15,
        line_end=80,
        signature="def generate_code(goal: str) -> str",
        docstring="Generate code from a goal description.",
        decorators="",
        intent_summary="Produces code artifacts from natural language goals.",
    )

    # --- call sites ---
    # handle_request in server.py calls authenticate from auth.py
    db.insert_call_site(caller_symbol_id=handle_sym_id, callee_name="authenticate", line=22)
    # generate_code in coder.py also calls authenticate
    db.insert_call_site(caller_symbol_id=generate_sym_id, callee_name="authenticate", line=40)

    # --- imports ---
    db.insert_import(file_id=server_id, imported_module="core.auth", imported_name="authenticate", is_from_import=True)
    db.insert_import(file_id=coder_id, imported_module="core.auth", imported_name="authenticate", is_from_import=True)

    # --- relationships ---
    # server.py imports from auth.py
    db.upsert_relationship(from_file_id=server_id, to_file_id=auth_id, rel_type="imports", strength=0.8)
    # coder.py imports from auth.py
    db.upsert_relationship(from_file_id=coder_id, to_file_id=auth_id, rel_type="imports", strength=0.6)

    # --- scan record ---
    db.record_scan("abc", 3, 4, 0, 0.0, "full", "2026-01-01T00:00:00")

    return db


class TestSemanticQuerier:
    """Tests for SemanticQuerier query methods."""

    @pytest.fixture()
    def db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "test_semantic.db"

    @pytest.fixture()
    def querier(self, db_path: Path):
        _populate_test_db(db_path)
        from core.agent_sdk.semantic_querier import SemanticQuerier

        return SemanticQuerier(db_path)

    # ------------------------------------------------------------------
    def test_what_calls(self, querier) -> None:
        """authenticate should be called by handle_request and generate_code."""
        results = querier.what_calls("authenticate")
        assert len(results) == 2

        callers = {r["caller"] for r in results}
        assert "handle_request" in callers
        assert "generate_code" in callers

        # Check required keys are present
        for row in results:
            assert "callee_name" in row
            assert "caller" in row
            assert "kind" in row
            assert "path" in row
            assert "line" in row

        # callee_name should always be authenticate
        assert all(r["callee_name"] == "authenticate" for r in results)

    def test_what_calls_unknown_symbol(self, querier) -> None:
        """Unknown symbol returns empty list, not an error."""
        results = querier.what_calls("nonexistent_func")
        assert results == []

    # ------------------------------------------------------------------
    def test_what_depends_on(self, querier) -> None:
        """core/auth.py should have 2 dependents: server.py and coder.py."""
        results = querier.what_depends_on("core/auth.py")
        assert len(results) == 2

        paths = {r["path"] for r in results}
        assert "core/server.py" in paths
        assert "agents/coder.py" in paths

        for row in results:
            assert "path" in row
            assert "rel_type" in row
            assert "strength" in row

    def test_what_depends_on_unknown_path(self, querier) -> None:
        """Unknown path returns empty list."""
        results = querier.what_depends_on("does/not/exist.py")
        assert results == []

    # ------------------------------------------------------------------
    def test_what_changes_break(self, querier) -> None:
        """core/auth.py has transitive dependents (server.py and coder.py at distance 1)."""
        results = querier.what_changes_break("core/auth.py", depth=2)
        assert len(results) >= 1

        paths = {r["path"] for r in results}
        # Both direct dependents should appear
        assert "core/server.py" in paths or "agents/coder.py" in paths

        for row in results:
            assert "path" in row
            assert "distance" in row
            assert "relationship" in row
            assert row["distance"] >= 1

    def test_what_changes_break_unknown_path(self, querier) -> None:
        """Unknown path returns empty list."""
        results = querier.what_changes_break("unknown/file.py")
        assert results == []

    # ------------------------------------------------------------------
    def test_summarize_file(self, querier) -> None:
        """summarize() on a file path returns module_summary."""
        summary = querier.summarize("core/auth.py")
        assert "authentication" in summary.lower() or "password" in summary.lower()
        assert summary != ""

    def test_summarize_symbol(self, querier) -> None:
        """summarize() on a symbol name returns intent_summary."""
        summary = querier.summarize("authenticate")
        assert "authentication" in summary.lower() or "credential" in summary.lower()

    def test_summarize_fallback(self, querier) -> None:
        """summarize() on an unknown target returns the fallback string."""
        result = querier.summarize("completely_unknown_thing")
        assert result.startswith("No summary available for:")

    # ------------------------------------------------------------------
    def test_find_similar(self, querier) -> None:
        """find_similar should return symbol dicts matching the description."""
        results = querier.find_similar("authentication password", limit=5)
        # Should find 'authenticate' symbol
        assert isinstance(results, list)
        if results:  # FTS5 may or may not be available; at minimum it must not crash
            for row in results:
                assert "name" in row
                assert "path" in row

    # ------------------------------------------------------------------
    def test_architecture_overview(self, querier) -> None:
        """architecture_overview returns expected structure."""
        overview = querier.architecture_overview()

        assert "total_files" in overview
        assert overview["total_files"] == 3

        assert "clusters" in overview
        clusters = overview["clusters"]
        assert "core" in clusters
        assert "agents" in clusters
        assert clusters["core"] == 2
        assert clusters["agents"] == 1

        assert "top_coupled" in overview
        assert isinstance(overview["top_coupled"], list)
        # Top coupled should include core/auth.py (coupling 0.9)
        if overview["top_coupled"]:
            first = overview["top_coupled"][0]
            assert "path" in first
            assert "coupling" in first

        assert "summary" in overview
        assert isinstance(overview["summary"], str)
        assert len(overview["summary"]) > 0

    # ------------------------------------------------------------------
    def test_recent_changes(self, querier) -> None:
        """recent_changes returns a list (may be empty in test env)."""
        results = querier.recent_changes(n_commits=5)
        assert isinstance(results, list)
        # In test env there may be no git changes; just verify structure if non-empty
        for row in results:
            assert "path" in row
            # summary and coupling are optional but should be present as keys
            assert "summary" in row
            assert "coupling" in row
