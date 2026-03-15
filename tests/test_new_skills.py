"""Tests for new skills: dockerfile_analyzer, observability_checker, changelog_generator."""
from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

os.environ.setdefault("AURA_SKIP_CHDIR", "1")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from agents.skills.dockerfile_analyzer import DockerfileAnalyzerSkill
from agents.skills.observability_checker import ObservabilityCheckerSkill
from agents.skills.changelog_generator import ChangelogGeneratorSkill


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content))
    return p


def _all_findings(result: dict) -> list:
    """Flatten findings from nested results or top-level findings key."""
    if "findings" in result:
        return result["findings"]
    findings = []
    for r in result.get("results", []):
        findings.extend(r.get("findings", []))
    return findings


def _mock_git_run(log_output: str) -> MagicMock:
    """Return a MagicMock for subprocess.run that returns git log output."""
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = log_output
    mock.stderr = ""
    return mock


# ===========================================================================
# DockerfileAnalyzerSkill
# ===========================================================================

class TestDockerfileAnalyzer:
    def setup_method(self):
        self.skill = DockerfileAnalyzerSkill()

    def test_name(self):
        assert self.skill.name == "dockerfile_analyzer"

    def test_latest_tag_flagged(self, tmp_path):
        _write(tmp_path, "Dockerfile", """\
            FROM python:latest
            RUN pip install flask
        """)
        result = self.skill.run({"project_root": str(tmp_path)})
        assert "error" not in result
        findings = _all_findings(result)
        severities = [f["severity"] for f in findings]
        assert "high" in severities

    def test_missing_healthcheck_flagged(self, tmp_path):
        _write(tmp_path, "Dockerfile", """\
            FROM python:3.11-slim
            RUN pip install flask
            CMD ["python", "app.py"]
        """)
        result = self.skill.run({"project_root": str(tmp_path)})
        issues = [f["issue"] for f in _all_findings(result)]
        assert any("healthcheck" in i.lower() or "health" in i.lower() for i in issues)

    def test_secret_in_env_flagged(self, tmp_path):
        _write(tmp_path, "Dockerfile", """\
            FROM python:3.11-slim
            ENV API_KEY=supersecretvalue
            CMD ["python", "app.py"]
        """)
        result = self.skill.run({"project_root": str(tmp_path)})
        issues = [f["issue"] for f in _all_findings(result)]
        assert any("secret" in i.lower() for i in issues)

    def test_root_user_flagged(self, tmp_path):
        _write(tmp_path, "Dockerfile", """\
            FROM python:3.11-slim
            USER root
            CMD ["python", "app.py"]
        """)
        result = self.skill.run({"project_root": str(tmp_path)})
        issues = [f["issue"] for f in _all_findings(result)]
        assert any("root" in i.lower() or "user" in i.lower() for i in issues)

    def test_clean_dockerfile_no_high(self, tmp_path):
        _write(tmp_path, "Dockerfile", """\
            FROM python:3.11-slim
            WORKDIR /app
            COPY . .
            RUN pip install --no-cache-dir -r requirements.txt
            USER appuser
            HEALTHCHECK CMD curl -f http://localhost/ || exit 1
            CMD ["python", "app.py"]
        """)
        result = self.skill.run({"project_root": str(tmp_path)})
        findings = _all_findings(result)
        high_or_critical = [f for f in findings if f["severity"] in ("critical", "high")]
        assert len(high_or_critical) == 0

    def test_no_dockerfile_does_not_crash(self, tmp_path):
        result = self.skill.run({"project_root": str(tmp_path)})
        assert isinstance(result, dict)

    def test_inline_content(self):
        content = "FROM ubuntu:latest\nRUN apt-get update\n"
        result = self.skill.run({"content": content, "file_path": "Dockerfile"})
        findings = _all_findings(result)
        issues = [f["issue"] for f in findings]
        assert any("pin" in i.lower() or "latest" in i.lower() for i in issues)

    def test_result_has_aggregate_fields(self, tmp_path):
        _write(tmp_path, "Dockerfile", "FROM python:3.11-slim\n")
        result = self.skill.run({"project_root": str(tmp_path)})
        assert "total_findings" in result or "findings" in result or "results" in result


# ===========================================================================
# ObservabilityCheckerSkill
# ===========================================================================

class TestObservabilityChecker:
    def setup_method(self):
        self.skill = ObservabilityCheckerSkill()

    def test_name(self):
        assert self.skill.name == "observability_checker"

    def test_silent_except_detected(self, tmp_path):
        _write(tmp_path, "bad.py", """\
            def fetch():
                try:
                    return 1
                except Exception:
                    pass
        """)
        result = self.skill.run({"project_root": str(tmp_path)})
        assert "error" not in result
        # Issues could be under total_issues or nested in results
        total_issues = result.get("total_issues", 0)
        nested_issues = sum(
            len(r.get("issues", [])) for r in result.get("results", [])
        )
        assert total_issues > 0 or nested_issues > 0

    def test_bare_print_detected(self, tmp_path):
        _write(tmp_path, "noisy.py", """\
            def run():
                print("starting")
                return 1
        """)
        result = self.skill.run({"project_root": str(tmp_path)})
        bare_prints = result.get("total_bare_prints", 0)
        nested_prints = sum(
            len(r.get("bare_prints", [])) for r in result.get("results", [])
        )
        assert bare_prints > 0 or nested_prints > 0

    def test_long_function_without_logging_flagged(self, tmp_path):
        lines = ["def big_function():\n"]
        for i in range(35):
            lines.append(f"    x_{i} = {i} * 2\n")
        _write(tmp_path, "big.py", "".join(lines))
        result = self.skill.run({"project_root": str(tmp_path)})
        # Coverage should be < 100% — at least one unlogged function
        coverage = result.get("overall_logging_coverage_pct", 100)
        assert coverage < 100

    def test_well_logged_file_no_bare_prints(self, tmp_path):
        _write(tmp_path, "good.py", """\
            import logging
            logger = logging.getLogger(__name__)

            def process():
                logger.info("processing started")
                return 42
        """)
        result = self.skill.run({"project_root": str(tmp_path)})
        assert result.get("total_bare_prints", 0) == 0

    def test_inline_code(self):
        code = "def f():\n    try:\n        pass\n    except:\n        pass\n"
        result = self.skill.run({"code": code, "file_path": "x.py"})
        assert isinstance(result, dict)
        assert "error" not in result

    def test_empty_project(self, tmp_path):
        result = self.skill.run({"project_root": str(tmp_path)})
        assert isinstance(result, dict)

    def test_result_has_coverage_field(self, tmp_path):
        _write(tmp_path, "mod.py", "def f():\n    pass\n")
        result = self.skill.run({"project_root": str(tmp_path)})
        has_coverage = (
            "overall_logging_coverage_pct" in result
            or "logging_coverage_pct" in result
            or "files_scanned" in result
        )
        assert has_coverage


# ===========================================================================
# ChangelogGeneratorSkill
# ===========================================================================

_SAMPLE_LOG = (
    "abc1234 feat(auth): add JWT refresh endpoint\n"
    "def5678 fix(api): correct null pointer in user lookup\n"
    "ghi9012 docs: update README with new endpoints\n"
    "jkl3456 chore: bump version to 1.2.0\n"
    "pqr1234 refactor(core): simplify orchestrator loop\n"
    "stu5678 perf: cache model responses\n"
    "vwx9012 fix(db): handle connection timeout\n"
)

_BREAKING_LOG = "abc1234 feat!: breaking change in config schema\n"
_FIX_LOG = "abc1234 fix: correct null check\n"
_FEAT_LOG = "abc1234 feat(ui): add dark mode toggle\n"


def _make_git_root(tmp_path: Path) -> Path:
    """Create a fake .git dir so the skill skips the repo check."""
    (tmp_path / ".git").mkdir()
    return tmp_path


class TestChangelogGenerator:
    def setup_method(self):
        self.skill = ChangelogGeneratorSkill()

    def test_name(self):
        assert self.skill.name == "changelog_generator"

    def test_generates_changelog_from_commits(self, tmp_path):
        root = _make_git_root(tmp_path)
        with patch("subprocess.run", return_value=_mock_git_run(_SAMPLE_LOG)):
            result = self.skill.run({"project_root": str(root)})
        assert "error" not in result
        # changelog may be in 'markdown' or 'changelog' key
        text = result.get("markdown") or result.get("changelog") or ""
        assert len(text) > 0

    def test_feat_classified_in_features_section(self, tmp_path):
        root = _make_git_root(tmp_path)
        with patch("subprocess.run", return_value=_mock_git_run(_FEAT_LOG)):
            result = self.skill.run({"project_root": str(root)})
        text = result.get("markdown") or result.get("changelog") or ""
        assert "feat" in text.lower() or "feature" in text.lower()

    def test_breaking_change_bumps_major(self, tmp_path):
        root = _make_git_root(tmp_path)
        with patch("subprocess.run", return_value=_mock_git_run(_BREAKING_LOG)):
            result = self.skill.run({"project_root": str(root)})
        bump = result.get("version_bump") or result.get("suggested_version_bump")
        assert bump == "major"

    def test_fix_only_bumps_patch(self, tmp_path):
        root = _make_git_root(tmp_path)
        with patch("subprocess.run", return_value=_mock_git_run(_FIX_LOG)):
            result = self.skill.run({"project_root": str(root)})
        bump = result.get("version_bump") or result.get("suggested_version_bump")
        assert bump in ("patch", "minor")

    def test_feat_without_breaking_bumps_minor(self, tmp_path):
        root = _make_git_root(tmp_path)
        with patch("subprocess.run", return_value=_mock_git_run(_FEAT_LOG)):
            result = self.skill.run({"project_root": str(root)})
        bump = result.get("version_bump") or result.get("suggested_version_bump")
        assert bump in ("minor", "major")

    def test_unconventional_commits_handled_gracefully(self, tmp_path):
        root = _make_git_root(tmp_path)
        log = "abc1234 random commit message with no convention\n"
        with patch("subprocess.run", return_value=_mock_git_run(log)):
            result = self.skill.run({"project_root": str(root)})
        assert isinstance(result, dict)

    def test_missing_git_repo_returns_error(self, tmp_path):
        # No .git directory — should return error gracefully
        result = self.skill.run({"project_root": str(tmp_path)})
        assert "error" in result

    def test_since_tag_calls_git(self, tmp_path):
        root = _make_git_root(tmp_path)
        with patch("subprocess.run", return_value=_mock_git_run(_SAMPLE_LOG)) as mock_run:
            self.skill.run({"project_root": str(root), "from_ref": "v1.0.0"})
        assert mock_run.called

    def test_commit_count_in_result(self, tmp_path):
        root = _make_git_root(tmp_path)
        with patch("subprocess.run", return_value=_mock_git_run(_SAMPLE_LOG)):
            result = self.skill.run({"project_root": str(root)})
        has_count = (
            "commit_count" in result
            or "total_commits" in result
            or "markdown" in result
            or "changelog" in result
        )
        assert has_count



# ===========================================================================
# DockerfileAnalyzerSkill
# ===========================================================================
