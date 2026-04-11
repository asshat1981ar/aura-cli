"""Data models for security audit."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class Severity(Enum):
    """Severity levels for security findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FindingCategory(Enum):
    """Category of security finding."""

    SECRETS = "secrets"
    PERMISSIONS = "permissions"
    DEPENDENCIES = "dependencies"
    CONFIGURATION = "configuration"
    CODE = "code"
    NETWORK = "network"


@dataclass
class SecurityFinding:
    """A single security finding."""

    category: FindingCategory
    severity: Severity
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    remediation: Optional[str] = None
    references: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "remediation": self.remediation,
            "references": self.references,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AuditReport:
    """Complete security audit report."""

    findings: List[SecurityFinding]
    scan_duration_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    scanned_files: int = 0

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.HIGH)

    @property
    def medium_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.MEDIUM)

    @property
    def low_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.LOW)

    @property
    def total_count(self) -> int:
        return len(self.findings)

    @property
    def has_critical(self) -> bool:
        return self.critical_count > 0

    @property
    def has_issues(self) -> bool:
        return self.total_count > 0

    def get_by_severity(self, severity: Severity) -> List[SecurityFinding]:
        """Get findings by severity."""
        return [f for f in self.findings if f.severity == severity]

    def get_by_category(self, category: FindingCategory) -> List[SecurityFinding]:
        """Get findings by category."""
        return [f for f in self.findings if f.category == category]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "scan_duration_ms": self.scan_duration_ms,
            "scanned_files": self.scanned_files,
            "summary": {
                "total": self.total_count,
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": self.medium_count,
                "low": self.low_count,
            },
            "findings": [f.to_dict() for f in self.findings],
        }


@dataclass
class AuditConfig:
    """Configuration for security audit."""

    scan_secrets: bool = True
    scan_permissions: bool = True
    scan_dependencies: bool = True
    scan_configuration: bool = True
    exclude_patterns: List[str] = field(default_factory=list)
    severity_threshold: Severity = Severity.LOW

    def should_scan(self, severity: Severity) -> bool:
        """Check if a finding severity should be reported."""
        severity_order = {
            Severity.CRITICAL: 4,
            Severity.HIGH: 3,
            Severity.MEDIUM: 2,
            Severity.LOW: 1,
            Severity.INFO: 0,
        }
        return severity_order[severity] >= severity_order[self.severity_threshold]
