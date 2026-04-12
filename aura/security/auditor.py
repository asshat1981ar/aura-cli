"""Security auditor that orchestrates all scanners."""

import time
from pathlib import Path
from typing import List, Optional

from .models import AuditConfig, AuditReport, FindingCategory, SecurityFinding, Severity
from .scanners import (
    ConfigurationScanner,
    DependencyScanner,
    PermissionScanner,
    SecretScanner,
)


class SecurityAuditor:
    """Main security auditor that runs all scanners."""
    
    def __init__(self, config: Optional[AuditConfig] = None):
        self.config = config or AuditConfig()
        self._scanners = [
            SecretScanner(self.config),
            PermissionScanner(self.config),
            DependencyScanner(self.config),
            ConfigurationScanner(self.config),
        ]
    
    def audit(self, path: Path) -> AuditReport:
        """Run security audit on a path."""
        start = time.time()
        findings: List[SecurityFinding] = []
        scanned_files = 0
        
        # Count files to scan
        if path.is_file():
            scanned_files = 1
        elif path.is_dir():
            scanned_files = sum(1 for _ in path.rglob('*') if _.is_file())
        
        # Run all scanners
        for scanner in self._scanners:
            try:
                for finding in scanner.scan(path):
                    findings.append(finding)
            except Exception as e:
                # Log scanner error but continue
                findings.append(SecurityFinding(
                    category=FindingCategory.CODE,
                    severity=Severity.INFO,
                    title=f"Scanner error: {type(scanner).__name__}",
                    description=str(e),
                ))
        
        duration_ms = (time.time() - start) * 1000
        
        return AuditReport(
            findings=findings,
            scan_duration_ms=duration_ms,
            scanned_files=scanned_files,
        )
    
    def audit_file(self, file_path: Path) -> AuditReport:
        """Audit a single file."""
        return self.audit(file_path)
    
    def audit_directory(self, directory: Path) -> AuditReport:
        """Audit a directory."""
        return self.audit(directory)


class ContinuousAuditor:
    """Continuous security monitoring."""
    
    def __init__(self, paths: List[Path], interval: int = 3600):
        self.paths = paths
        self.interval = interval
        self.auditor = SecurityAuditor()
        self._reports: List[AuditReport] = []
    
    def run_once(self) -> List[AuditReport]:
        """Run audit on all paths once."""
        reports = []
        for path in self.paths:
            report = self.auditor.audit(path)
            reports.append(report)
        self._reports.extend(reports)
        return reports
    
    def get_summary(self) -> dict:
        """Get summary of all audits."""
        if not self._reports:
            return {"total_findings": 0, "audits_run": 0}
        
        total_findings = sum(r.total_count for r in self._reports)
        critical = sum(r.critical_count for r in self._reports)
        high = sum(r.high_count for r in self._reports)
        
        return {
            "total_findings": total_findings,
            "audits_run": len(self._reports),
            "critical": critical,
            "high": high,
            "medium": sum(r.medium_count for r in self._reports),
            "low": sum(r.low_count for r in self._reports),
        }
    
    def has_critical_issues(self) -> bool:
        """Check if any critical issues were found."""
        return any(r.has_critical for r in self._reports)
