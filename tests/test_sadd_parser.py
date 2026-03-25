"""Unit tests for the SADD design spec parser."""

import unittest
from pathlib import Path

from core.sadd.design_spec_parser import DesignSpecParser
from core.sadd.types import DesignSpec, WorkstreamSpec

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestDesignSpecParser(unittest.TestCase):
    """Tests for :class:`DesignSpecParser`."""

    def setUp(self) -> None:
        self.parser = DesignSpecParser()

    # ---- 1. Sample spec parsing ----

    def test_parse_sample_spec(self) -> None:
        """Parse sadd_sample_spec.md: correct title, 3 workstreams, high confidence."""
        spec = self.parser.parse_file(FIXTURES_DIR / "sadd_sample_spec.md")

        self.assertEqual(spec.title, "AURA Server Test Suite")
        self.assertEqual(len(spec.workstreams), 3)
        self.assertGreaterEqual(spec.parse_confidence, 0.8)

    # ---- 2. Dependency resolution ----

    def test_parse_dependencies_resolved(self) -> None:
        """Dependencies are correctly wired between the three workstreams."""
        spec = self.parser.parse_file(FIXTURES_DIR / "sadd_sample_spec.md")

        ws_by_title = {ws.title: ws for ws in spec.workstreams}

        server_ws = ws_by_title["Workstream: Server Unit Tests"]
        a2a_ws = ws_by_title["Workstream: A2A Framework Integration"]
        integration_ws = ws_by_title["Workstream: Integration Test Suite"]

        self.assertEqual(server_ws.depends_on, [])
        self.assertIn(server_ws.id, a2a_ws.depends_on)
        self.assertIn(server_ws.id, integration_ws.depends_on)
        self.assertIn(a2a_ws.id, integration_ws.depends_on)

    # ---- 3. Minimal spec ----

    def test_parse_minimal_spec(self) -> None:
        """Parse sadd_minimal_spec.md: correct title, 1 workstream, high confidence."""
        spec = self.parser.parse_file(FIXTURES_DIR / "sadd_minimal_spec.md")

        self.assertEqual(spec.title, "Fix Authentication Bug")
        self.assertEqual(len(spec.workstreams), 1)
        self.assertGreaterEqual(spec.parse_confidence, 0.8)

    # ---- 4. Empty document ----

    def test_parse_empty_document(self) -> None:
        """Empty string produces a fallback workstream with low confidence."""
        spec = self.parser.parse("")

        self.assertEqual(len(spec.workstreams), 1)
        self.assertEqual(spec.workstreams[0].id, "ws_full_document")
        # Confidence should be recalculated after fallback but still low
        # because there are no real heading sections.
        self.assertLessEqual(spec.parse_confidence, 0.5)

    # ---- 5. No headings ----

    def test_parse_no_headings(self) -> None:
        """Plain text with no markdown headings falls back to a single workstream."""
        text = "This is some plain text describing work.\nIt has no headings at all.\nJust a blob of prose."
        spec = self.parser.parse(text)

        self.assertEqual(len(spec.workstreams), 1)
        self.assertIn("plain text", spec.workstreams[0].goal_text)

    # ---- 6. Acceptance criteria ----

    def test_parse_acceptance_criteria(self) -> None:
        """Acceptance: lines are extracted into acceptance_criteria."""
        # The parser's _ACCEPTANCE_LABEL_RE matches "Acceptance:" at the start
        # of a line (not nested inside a bullet), so use a standalone block.
        md = "# My Spec\n\n## Workstream: Auth\n\n- Implement login flow\n\nAcceptance: all auth tests pass\n"
        spec = self.parser.parse(md)

        self.assertTrue(len(spec.workstreams) >= 1)
        auth_ws = spec.workstreams[0]
        self.assertTrue(len(auth_ws.acceptance_criteria) >= 1)
        criteria_text = " ".join(auth_ws.acceptance_criteria).lower()
        self.assertIn("auth tests pass", criteria_text)

    # ---- 7. parse_file with tmp_path ----

    def test_parse_file(self, tmp_path: Path = None) -> None:
        """parse_file reads a temporary file and returns a valid DesignSpec."""
        # Use a manual temp dir when run via unittest (tmp_path is a pytest fixture).
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "spec.md"
            p.write_text(
                "# Temp Spec\n\n## Task: Do Something\n\n- Add a feature\n",
                encoding="utf-8",
            )
            spec = self.parser.parse_file(p)

        self.assertIsInstance(spec, DesignSpec)
        self.assertEqual(spec.title, "Temp Spec")
        self.assertTrue(len(spec.workstreams) >= 1)

    # ---- 8. Unique workstream IDs ----

    def test_workstream_ids_are_unique(self) -> None:
        """Similar headings get deduplicated IDs with _2, _3 suffixes.

        Note: _extract_sections uses a dict keyed by heading text, so
        truly identical headings collapse into one section.  We use
        *slightly different* headings that produce the same slug to
        exercise the deduplication logic in _identify_workstreams.
        """
        md = "# Plan\n\n## Workstream: Setup\n\n- Create scaffolding\n\n## Workstream: Setup Phase-2\n\n- Create more scaffolding\n\n## Workstream: Setup Phase-3\n\n- Create even more scaffolding\n"
        spec = self.parser.parse(md)

        ids = [ws.id for ws in spec.workstreams]
        # All IDs must be unique.
        self.assertEqual(len(ids), len(set(ids)))
        # Should have 3 distinct workstreams.
        self.assertEqual(len(ids), 3)

    # ---- 9. Tags from headings ----

    def test_tags_extracted_from_headings(self) -> None:
        """Headings with keywords like 'test', 'api' get tagged."""
        md = "# Project\n\n## Workstream: API Integration Tests\n\n- Test the API endpoints\n"
        spec = self.parser.parse(md)

        self.assertTrue(len(spec.workstreams) >= 1)
        tags = spec.workstreams[0].tags
        self.assertIn("api", tags)
        self.assertIn("testing", tags)

    # ---- 10. Confidence scoring ----

    def test_confidence_scoring(self) -> None:
        """Well-structured spec gets high confidence; poorly structured gets low."""
        well_structured = "# Good Spec\n\n## Workstream: Alpha\n\n- Implement feature alpha\n- Depends on: Beta\n\n## Workstream: Beta\n\n- Build foundation\n"
        poorly_structured = "just some random text without any structure"

        good_spec = self.parser.parse(well_structured)
        bad_spec = self.parser.parse(poorly_structured)

        self.assertGreater(good_spec.parse_confidence, bad_spec.parse_confidence)
        self.assertGreaterEqual(good_spec.parse_confidence, 0.6)
        # Poorly structured text gets a fallback workstream, which brings
        # confidence to 0.5; verify it stays at or below that threshold.
        self.assertLessEqual(bad_spec.parse_confidence, 0.5)


if __name__ == "__main__":
    unittest.main()
