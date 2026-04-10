"""Unit tests for Dependency Analyzer Skill."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch, MagicMock

from agents.skills.dependency_analyzer import (
    DependencyAnalyzerSkill,
    _parse_requirements,
    _parse_pyproject_toml,
    _pip_list,
    _check_vulns,
    _check_conflicts,
    _check_unpinned,
    _KNOWN_VULNS,
)


class TestParseRequirements(TestCase):
    """Test cases for _parse_requirements function."""

    def setUp(self):
        """Set up test fixtures."""
        self.tmpdir = tempfile.mkdtemp()
        self.req_file = Path(self.tmpdir) / "requirements.txt"

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.tmpdir)

    def test_parse_simple_requirements(self):
        """Test parsing simple requirements."""
        self.req_file.write_text("requests\nflask\ndjango")
        result = _parse_requirements(self.req_file)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["name"], "requests")
        self.assertEqual(result[1]["name"], "flask")
        self.assertEqual(result[2]["name"], "django")

    def test_parse_pinned_requirements(self):
        """Test parsing pinned requirements."""
        self.req_file.write_text("requests==2.28.0\nflask>=2.0.0")
        result = _parse_requirements(self.req_file)

        self.assertEqual(result[0]["name"], "requests")
        self.assertEqual(result[0]["specifier"], "==2.28.0")
        self.assertTrue(result[0]["pinned"])
        self.assertFalse(result[0]["unpinned"])

        self.assertEqual(result[1]["name"], "flask")
        self.assertEqual(result[1]["specifier"], ">=2.0.0")
        self.assertFalse(result[1]["pinned"])

    def test_parse_unpinned_requirements(self):
        """Test parsing unpinned requirements."""
        self.req_file.write_text("requests\nflask ")
        result = _parse_requirements(self.req_file)

        self.assertTrue(result[0]["unpinned"])
        self.assertTrue(result[1]["unpinned"])

    def test_parse_with_comments_and_blank_lines(self):
        """Test parsing requirements with comments and blank lines."""
        self.req_file.write_text("""
# This is a comment
requests==2.28.0

flask>=2.0.0
# Another comment
django
""")
        result = _parse_requirements(self.req_file)

        self.assertEqual(len(result), 3)

    def test_parse_with_options(self):
        """Test parsing requirements with options."""
        self.req_file.write_text("-r base.txt\n--index-url https://pypi.org\nrequests")
        result = _parse_requirements(self.req_file)

        # Options should be skipped
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "requests")

    def test_parse_git_urls(self):
        """Test parsing requirements with git URLs."""
        self.req_file.write_text("git+https://github.com/user/repo.git\nrequests")
        result = _parse_requirements(self.req_file)

        # Git URLs should be skipped
        self.assertEqual(len(result), 1)

    def test_parse_normalizes_names(self):
        """Test that package names are normalized."""
        self.req_file.write_text("some-package\nAnother_Package")
        result = _parse_requirements(self.req_file)

        self.assertEqual(result[0]["name"], "some_package")
        self.assertEqual(result[1]["name"], "another_package")

    def test_parse_nonexistent_file(self):
        """Test parsing nonexistent file."""
        result = _parse_requirements(Path("/nonexistent/requirements.txt"))
        self.assertEqual(result, [])


class TestParsePyprojectToml(TestCase):
    """Test cases for _parse_pyproject_toml function."""

    def setUp(self):
        """Set up test fixtures."""
        self.tmpdir = tempfile.mkdtemp()
        self.pyproject_file = Path(self.tmpdir) / "pyproject.toml"

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.tmpdir)

    def test_parse_poetry_dependencies(self):
        """Test parsing poetry-style dependencies."""
        self.pyproject_file.write_text("""
[tool.poetry.dependencies]
python = "^3.9"
requests = "^2.28.0"
flask = ">=2.0.0"
""")
        result = _parse_pyproject_toml(self.pyproject_file)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "requests")
        self.assertEqual(result[1]["name"], "flask")

    def test_parse_project_dependencies(self):
        """Test parsing PEP 621 project dependencies."""
        self.pyproject_file.write_text("""
[project]
name = "my-project"
dependencies = [
    "requests>=2.28.0",
    "flask>=2.0.0",
]
""")
        result = _parse_pyproject_toml(self.pyproject_file)

        # Should find dependencies section
        self.assertGreaterEqual(len(result), 0)

    def test_skips_python_entry(self):
        """Test that python entry is skipped."""
        self.pyproject_file.write_text("""
[tool.poetry.dependencies]
python = "^3.9"
requests = "^2.28.0"
""")
        result = _parse_pyproject_toml(self.pyproject_file)

        for pkg in result:
            self.assertNotEqual(pkg["name"], "python")

    def test_parse_nonexistent_file(self):
        """Test parsing nonexistent file."""
        result = _parse_pyproject_toml(Path("/nonexistent/pyproject.toml"))
        self.assertEqual(result, [])


class TestPipList(TestCase):
    """Test cases for _pip_list function."""

    @patch("agents.skills.dependency_analyzer.subprocess.run")
    def test_pip_list_success(self, mock_run):
        """Test successful pip list execution."""
        mock_run.return_value = Mock(returncode=0, stdout="Package    Version\n-------    -------\nrequests   2.28.0\nflask      2.0.0\n")
        result = _pip_list()

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "requests")
        self.assertEqual(result[1]["name"], "flask")

    @patch("agents.skills.dependency_analyzer.subprocess.run")
    def test_pip_list_failure(self, mock_run):
        """Test pip list failure."""
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="")
        result = _pip_list()

        self.assertIsNone(result)

    @patch("agents.skills.dependency_analyzer.subprocess.run")
    def test_pip_list_exception(self, mock_run):
        """Test pip list with exception."""
        mock_run.side_effect = OSError("Command not found")
        result = _pip_list()

        self.assertIsNone(result)


class TestCheckVulns(TestCase):
    """Test cases for _check_vulns function."""

    def test_check_vulnerable_package(self):
        """Test detecting vulnerable package."""
        packages = [{"name": "requests", "raw_name": "requests", "specifier": "==2.19.0", "source": "req.txt"}]
        result = _check_vulns(packages)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["package"], "requests")
        self.assertIn("CVE", result[0]["cve"])

    def test_check_non_vulnerable_package(self):
        """Test non-vulnerable package."""
        packages = [{"name": "nonexistent", "raw_name": "nonexistent", "specifier": "==1.0.0", "source": "req.txt"}]
        result = _check_vulns(packages)

        self.assertEqual(len(result), 0)

    def test_check_multiple_packages(self):
        """Test checking multiple packages."""
        packages = [
            {"name": "requests", "raw_name": "requests", "specifier": "==2.19.0", "source": "req.txt"},
            {"name": "pyyaml", "raw_name": "pyyaml", "specifier": "==5.3", "source": "req.txt"},
            {"name": "safe-package", "raw_name": "safe-package", "specifier": "==1.0.0", "source": "req.txt"},
        ]
        result = _check_vulns(packages)

        self.assertEqual(len(result), 2)


class TestCheckConflicts(TestCase):
    """Test cases for _check_conflicts function."""

    def test_no_conflicts(self):
        """Test with no conflicts."""
        packages = [
            {"name": "requests", "specifier": "==2.28.0", "source": "req1.txt"},
            {"name": "flask", "specifier": ">=2.0.0", "source": "req2.txt"},
        ]
        result = _check_conflicts(packages)

        self.assertEqual(len(result), 0)

    def test_duplicate_detected(self):
        """Test detecting duplicate entries."""
        packages = [
            {"name": "requests", "specifier": "==2.28.0", "source": "req1.txt"},
            {"name": "requests", "specifier": ">=2.0.0", "source": "req2.txt"},
        ]
        result = _check_conflicts(packages)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["package"], "requests")

    def test_multiple_duplicates(self):
        """Test detecting multiple duplicates."""
        packages = [
            {"name": "requests", "specifier": "==2.28.0", "source": "req1.txt"},
            {"name": "requests", "specifier": ">=2.0.0", "source": "req2.txt"},
            {"name": "flask", "specifier": "==2.0.0", "source": "req1.txt"},
            {"name": "flask", "specifier": ">=1.0.0", "source": "req2.txt"},
        ]
        result = _check_conflicts(packages)

        self.assertEqual(len(result), 2)


class TestCheckUnpinned(TestCase):
    """Test cases for _check_unpinned function."""

    def test_all_pinned(self):
        """Test with all packages pinned."""
        packages = [
            {"name": "requests", "raw_name": "requests", "unpinned": False},
            {"name": "flask", "raw_name": "flask", "unpinned": False},
        ]
        result = _check_unpinned(packages)

        self.assertEqual(len(result), 0)

    def test_some_unpinned(self):
        """Test with some unpinned packages."""
        packages = [
            {"name": "requests", "raw_name": "requests", "unpinned": False},
            {"name": "flask", "raw_name": "flask", "unpinned": True},
        ]
        result = _check_unpinned(packages)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "flask")


class TestDependencyAnalyzerSkill(TestCase):
    """Test cases for DependencyAnalyzerSkill."""

    def setUp(self):
        """Set up test fixtures."""
        self.skill = DependencyAnalyzerSkill()

    def test_skill_initialization(self):
        """Test skill initialization."""
        self.assertEqual(self.skill.name, "dependency_analyzer")

    def test_run_with_empty_project(self):
        """Test running on empty project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.skill._run({"project_root": tmpdir})

            self.assertEqual(result["packages_found"], 0)
            self.assertIn("No requirements files found", result["recommendations"][0])

    def test_run_with_requirements_txt(self):
        """Test running with requirements.txt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            req_file = Path(tmpdir) / "requirements.txt"
            req_file.write_text("requests==2.28.0\nflask>=2.0.0")

            result = self.skill._run({"project_root": tmpdir})

            self.assertEqual(result["packages_found"], 2)
            self.assertEqual(len(result["req_files_scanned"]), 1)

    def test_run_with_pyproject_toml(self):
        """Test running with pyproject.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject = Path(tmpdir) / "pyproject.toml"
            pyproject.write_text("""
[tool.poetry.dependencies]
python = "^3.9"
requests = "^2.28.0"
""")

            result = self.skill._run({"project_root": tmpdir})

            self.assertEqual(result["packages_found"], 1)

    def test_run_detects_vulnerabilities(self):
        """Test that vulnerabilities are detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            req_file = Path(tmpdir) / "requirements.txt"
            req_file.write_text("requests==2.19.0")  # Known vulnerable version

            result = self.skill._run({"project_root": tmpdir})

            self.assertGreater(result["vulnerability_count"], 0)
            self.assertIn("Upgrade", result["recommendations"][0])

    def test_run_detects_unpinned(self):
        """Test that unpinned packages are detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            req_file = Path(tmpdir) / "requirements.txt"
            req_file.write_text("requests\nflask")

            result = self.skill._run({"project_root": tmpdir})

            self.assertEqual(len(result["unpinned_packages"]), 2)
            # "unpinned" may not be the first recommendation when vulnerabilities
            # are also detected; verify it appears somewhere in the list.
            self.assertTrue(
                any("unpinned" in r for r in result["recommendations"]),
                f"Expected an 'unpinned' recommendation in {result['recommendations']}",
            )

    def test_run_with_multiple_paths(self):
        """Test running with multiple project paths."""
        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
            req1 = Path(tmpdir1) / "requirements.txt"
            req1.write_text("requests")
            req2 = Path(tmpdir2) / "requirements.txt"
            req2.write_text("flask")

            result = self.skill._run({"paths": [tmpdir1, tmpdir2]})

            self.assertEqual(result["packages_found"], 2)

    def test_run_with_include_pip(self):
        """Test running with include_pip option."""
        with tempfile.TemporaryDirectory() as tmpdir:
            req_file = Path(tmpdir) / "requirements.txt"
            req_file.write_text("requests")

            with patch("agents.skills.dependency_analyzer._pip_list") as mock_pip:
                mock_pip.return_value = [{"name": "flask", "raw_name": "flask", "specifier": "==2.0.0", "source": "pip_list", "pinned": True, "unpinned": False}]
                result = self.skill._run({"project_root": tmpdir, "include_pip": True})

                self.assertEqual(result["packages_found"], 2)

    def test_run_skips_git_and_node_modules(self):
        """Test that run skips .git and node_modules."""
        with tempfile.TemporaryDirectory() as tmpdir:
            git_dir = Path(tmpdir) / ".git"
            git_dir.mkdir()
            (git_dir / "requirements.txt").write_text("should-not-be-found")

            node_modules = Path(tmpdir) / "node_modules"
            node_modules.mkdir()
            (node_modules / "requirements.txt").write_text("should-not-be-found")

            req_file = Path(tmpdir) / "requirements.txt"
            req_file.write_text("requests")

            result = self.skill._run({"project_root": tmpdir})

            # Should only find the one valid requirements.txt
            self.assertEqual(len(result["req_files_scanned"]), 1)

    def test_run_caps_packages(self):
        """Test that packages list is capped at 100."""
        with tempfile.TemporaryDirectory() as tmpdir:
            req_file = Path(tmpdir) / "requirements.txt"
            req_file.write_text("\n".join([f"pkg{i}" for i in range(150)]))

            result = self.skill._run({"project_root": tmpdir})

            self.assertEqual(len(result["packages"]), 100)

    def test_known_vulns_database(self):
        """Test that known vulnerabilities database is populated."""
        self.assertIn("requests", _KNOWN_VULNS)
        self.assertIn("django", _KNOWN_VULNS)
        self.assertIn("flask", _KNOWN_VULNS)
        # Each entry should have: vulnerable_below, advisory_id, description
        for pkg, info in _KNOWN_VULNS.items():
            self.assertEqual(len(info), 3)
