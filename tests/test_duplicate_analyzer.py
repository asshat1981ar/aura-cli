"""Unit tests for core/duplicate_analyzer.py — DuplicateCodeAnalyzer."""

import json
import os
import tempfile

from core.duplicate_analyzer import DuplicateCodeAnalyzer


class TestDuplicateCodeAnalyzerInit:
    def test_default_base_path(self):
        analyzer = DuplicateCodeAnalyzer()
        assert analyzer.base_path == "."

    def test_custom_base_path(self):
        analyzer = DuplicateCodeAnalyzer("/tmp")
        assert analyzer.base_path == "/tmp"

    def test_duplicate_patterns_starts_empty(self):
        analyzer = DuplicateCodeAnalyzer()
        assert len(analyzer.duplicate_patterns) == 0


class TestDuplicateCodeAnalyzerScan:
    def test_scan_for_duplicates_returns_none(self):
        analyzer = DuplicateCodeAnalyzer()
        result = analyzer.scan_for_duplicates()
        assert result is None

    def test_prioritize_segments_returns_none(self):
        analyzer = DuplicateCodeAnalyzer()
        result = analyzer.prioritize_segments()
        assert result is None


class TestDuplicateCodeAnalyzerExport:
    def test_export_findings_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "report.json")
            analyzer = DuplicateCodeAnalyzer()
            analyzer.export_findings(output_path)
            assert os.path.exists(output_path)

    def test_export_findings_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "report.json")
            analyzer = DuplicateCodeAnalyzer()
            analyzer.export_findings(output_path)
            with open(output_path) as f:
                data = json.load(f)
            assert isinstance(data, dict)

    def test_export_findings_reflects_patterns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "report.json")
            analyzer = DuplicateCodeAnalyzer()
            analyzer.duplicate_patterns["func_hash_123"] = ["file_a.py:foo", "file_b.py:foo"]
            analyzer.export_findings(output_path)
            with open(output_path) as f:
                data = json.load(f)
            assert "func_hash_123" in data
            assert data["func_hash_123"] == ["file_a.py:foo", "file_b.py:foo"]
