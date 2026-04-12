"""Tests for security auditor."""

import pytest
from pathlib import Path

from aura.security.auditor import ContinuousAuditor, SecurityAuditor
from aura.security.models import AuditConfig, FindingCategory, SecurityFinding, Severity


class TestSecurityAuditor:
    @pytest.fixture
    def auditor(self):
        return SecurityAuditor()
    
    def test_audit_single_file(self, tmp_path, auditor):
        test_file = tmp_path / "test.py"
        test_file.write_text('password = "secret123"')
        
        report = auditor.audit(test_file)
        
        assert report.total_count >= 1
        assert report.scanned_files == 1
        assert any(f.category == FindingCategory.SECRETS for f in report.findings)
    
    def test_audit_directory(self, tmp_path, auditor):
        # Create test files
        (tmp_path / "file1.py").write_text('api_key = "secret1"')
        (tmp_path / "file2.py").write_text('password = "secret2"')
        
        report = auditor.audit(tmp_path)
        
        assert report.total_count >= 1
        assert report.scanned_files == 2
    
    def test_audit_file_method(self, tmp_path, auditor):
        test_file = tmp_path / "config.py"
        test_file.write_text('token = "abc123"')
        
        report = auditor.audit_file(test_file)
        
        assert report.scanned_files == 1
    
    def test_audit_directory_method(self, tmp_path, auditor):
        (tmp_path / "test.py").write_text('key = "value"')
        
        report = auditor.audit_directory(tmp_path)
        
        assert report.scanned_files >= 1
    
    def test_audit_with_config(self, tmp_path):
        config = AuditConfig(severity_threshold=Severity.CRITICAL)
        auditor = SecurityAuditor(config)
        
        test_file = tmp_path / "test.py"
        test_file.write_text('api_key = "secret"')  # HIGH severity
        
        report = auditor.audit(test_file)
        
        # Should not report HIGH severity findings when threshold is CRITICAL
        assert report.total_count == 0
    
    def test_audit_empty_directory(self, tmp_path, auditor):
        report = auditor.audit(tmp_path)
        
        assert report.total_count == 0
        assert report.scanned_files == 0


class TestContinuousAuditor:
    @pytest.fixture
    def paths(self, tmp_path):
        # Create test files
        (tmp_path / "file1.py").write_text('password = "pass1"')
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file2.py").write_text('api_key = "key2"')
        return [tmp_path]
    
    @pytest.fixture
    def continuous_auditor(self, paths):
        return ContinuousAuditor(paths=paths, interval=60)
    
    def test_run_once(self, paths, continuous_auditor):
        reports = continuous_auditor.run_once()
        
        assert len(reports) == 1
        assert reports[0].total_count >= 1
    
    def test_get_summary(self, paths, continuous_auditor):
        continuous_auditor.run_once()
        
        summary = continuous_auditor.get_summary()
        
        assert summary["audits_run"] == 1
        assert summary["total_findings"] >= 1
        assert summary["critical"] >= 0
    
    def test_has_critical_issues(self, paths, continuous_auditor):
        continuous_auditor.run_once()
        
        assert continuous_auditor.has_critical_issues() is True
    
    def test_no_critical_issues(self, tmp_path):
        # Create file with only LOW severity issue
        (tmp_path / "clean.py").write_text('# No secrets here')
        
        auditor = ContinuousAuditor(paths=[tmp_path])
        auditor.run_once()
        
        assert auditor.has_critical_issues() is False
    
    def test_multiple_paths(self, tmp_path):
        path1 = tmp_path / "dir1"
        path1.mkdir()
        (path1 / "file.py").write_text('key = "val1"')
        
        path2 = tmp_path / "dir2"
        path2.mkdir()
        (path2 / "file.py").write_text('pwd = "val2"')
        
        auditor = ContinuousAuditor(paths=[path1, path2])
        reports = auditor.run_once()
        
        assert len(reports) == 2
        assert auditor.get_summary()["audits_run"] == 2
