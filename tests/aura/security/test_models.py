"""Tests for security models."""

import pytest
from datetime import datetime

from aura.security.models import (
    AuditConfig,
    AuditReport,
    FindingCategory,
    SecurityFinding,
    Severity,
)


class TestSeverity:
    def test_severity_values(self):
        assert Severity.CRITICAL.value == "critical"
        assert Severity.HIGH.value == "high"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.LOW.value == "low"
        assert Severity.INFO.value == "info"


class TestFindingCategory:
    def test_category_values(self):
        assert FindingCategory.SECRETS.value == "secrets"
        assert FindingCategory.PERMISSIONS.value == "permissions"
        assert FindingCategory.DEPENDENCIES.value == "dependencies"
        assert FindingCategory.CONFIGURATION.value == "configuration"
        assert FindingCategory.CODE.value == "code"
        assert FindingCategory.NETWORK.value == "network"


class TestSecurityFinding:
    def test_basic_finding(self):
        finding = SecurityFinding(
            category=FindingCategory.SECRETS,
            severity=Severity.HIGH,
            title="API Key exposed",
            description="Found in file.py line 10",
        )
        
        assert finding.category == FindingCategory.SECRETS
        assert finding.severity == Severity.HIGH
        assert finding.title == "API Key exposed"
        assert finding.file_path is None
        assert finding.line_number is None
    
    def test_finding_with_location(self):
        finding = SecurityFinding(
            category=FindingCategory.CODE,
            severity=Severity.MEDIUM,
            title="Issue",
            description="Details",
            file_path="/path/to/file.py",
            line_number=42,
            remediation="Fix it",
            references=["https://example.com"],
        )
        
        assert finding.file_path == "/path/to/file.py"
        assert finding.line_number == 42
        assert finding.remediation == "Fix it"
        assert finding.references == ["https://example.com"]
    
    def test_to_dict(self):
        finding = SecurityFinding(
            category=FindingCategory.SECRETS,
            severity=Severity.CRITICAL,
            title="Password",
            description="Hardcoded password",
        )
        
        data = finding.to_dict()
        
        assert data["category"] == "secrets"
        assert data["severity"] == "critical"
        assert data["title"] == "Password"
        assert "timestamp" in data


class TestAuditConfig:
    def test_default_config(self):
        config = AuditConfig()
        
        assert config.scan_secrets is True
        assert config.scan_permissions is True
        assert config.scan_dependencies is True
        assert config.scan_configuration is True
        assert config.severity_threshold == Severity.LOW
        assert config.exclude_patterns == []
    
    def test_custom_config(self):
        config = AuditConfig(
            scan_secrets=False,
            severity_threshold=Severity.HIGH,
            exclude_patterns=["node_modules", ".git"],
        )
        
        assert config.scan_secrets is False
        assert config.severity_threshold == Severity.HIGH
        assert "node_modules" in config.exclude_patterns
    
    def test_should_scan(self):
        config = AuditConfig(severity_threshold=Severity.MEDIUM)
        
        assert config.should_scan(Severity.CRITICAL) is True
        assert config.should_scan(Severity.HIGH) is True
        assert config.should_scan(Severity.MEDIUM) is True
        assert config.should_scan(Severity.LOW) is False
        assert config.should_scan(Severity.INFO) is False


class TestAuditReport:
    @pytest.fixture
    def sample_findings(self):
        return [
            SecurityFinding(FindingCategory.SECRETS, Severity.CRITICAL, "C1", "Desc"),
            SecurityFinding(FindingCategory.SECRETS, Severity.HIGH, "H1", "Desc"),
            SecurityFinding(FindingCategory.CODE, Severity.HIGH, "H2", "Desc"),
            SecurityFinding(FindingCategory.PERMISSIONS, Severity.MEDIUM, "M1", "Desc"),
            SecurityFinding(FindingCategory.CONFIGURATION, Severity.LOW, "L1", "Desc"),
        ]
    
    def test_report_creation(self, sample_findings):
        report = AuditReport(
            findings=sample_findings,
            scan_duration_ms=100.0,
            scanned_files=10,
        )
        
        assert report.total_count == 5
        assert report.critical_count == 1
        assert report.high_count == 2
        assert report.medium_count == 1
        assert report.low_count == 1
        assert report.scanned_files == 10
    
    def test_has_critical(self, sample_findings):
        report = AuditReport(findings=sample_findings, scan_duration_ms=1.0)
        assert report.has_critical is True
    
    def test_no_critical(self):
        findings = [SecurityFinding(FindingCategory.CODE, Severity.LOW, "L1", "Desc")]
        report = AuditReport(findings=findings, scan_duration_ms=1.0)
        assert report.has_critical is False
    
    def test_has_issues(self):
        report = AuditReport(findings=[], scan_duration_ms=1.0)
        assert report.has_issues is False
        
        report = AuditReport(findings=[SecurityFinding(FindingCategory.CODE, Severity.INFO, "I1", "Desc")], scan_duration_ms=1.0)
        assert report.has_issues is True
    
    def test_get_by_severity(self, sample_findings):
        report = AuditReport(findings=sample_findings, scan_duration_ms=1.0)
        
        high_findings = report.get_by_severity(Severity.HIGH)
        assert len(high_findings) == 2
    
    def test_get_by_category(self, sample_findings):
        report = AuditReport(findings=sample_findings, scan_duration_ms=1.0)
        
        secret_findings = report.get_by_category(FindingCategory.SECRETS)
        assert len(secret_findings) == 2
    
    def test_to_dict(self, sample_findings):
        report = AuditReport(findings=sample_findings, scan_duration_ms=50.0, scanned_files=5)
        
        data = report.to_dict()
        
        assert data["scan_duration_ms"] == 50.0
        assert data["scanned_files"] == 5
        assert data["summary"]["total"] == 5
        assert data["summary"]["critical"] == 1
        assert len(data["findings"]) == 5
