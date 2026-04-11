"""Security scanners for various categories."""

import json
import os
import re
from pathlib import Path
from typing import Iterator, Optional

from .models import AuditConfig, FindingCategory, SecurityFinding, Severity


class SecretScanner:
    """Scan for hardcoded secrets in files."""
    
    PATTERNS = [
        # API Keys
        (r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\']?([a-zA-Z0-9_\-]{16,})["\']?', 
         "Potential API key exposed", Severity.HIGH),
        # Passwords
        (r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']?([^"\'\s]{4,})["\']?',
         "Potential password in code", Severity.CRITICAL),
        # Private keys
        (r'-----BEGIN (RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----',
         "Private key found in code", Severity.CRITICAL),
        # AWS Access Key
        (r'AKIA[0-9A-Z]{16}',
         "AWS Access Key ID found", Severity.CRITICAL),
        # Generic tokens
        (r'(?i)(token|secret)\s*[=:]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?',
         "Potential token exposed", Severity.HIGH),
        # URLs with credentials
        (r'https?://[^:]+:[^@]+@[a-zA-Z0-9.-]+',
         "URL with embedded credentials", Severity.CRITICAL),
    ]
    
    EXCLUDED_EXTENSIONS = {'.pyc', '.pyo', '.so', '.dll', '.exe', '.jpg', '.png', '.gif', '.pdf'}
    
    def __init__(self, config: Optional[AuditConfig] = None):
        self.config = config or AuditConfig()
        # Compile regex patterns once for performance
        self._compiled_patterns = [
            (re.compile(pattern), description, severity)
            for pattern, description, severity in self.PATTERNS
        ]
    
    def scan(self, path: Path) -> Iterator[SecurityFinding]:
        """Scan a file or directory for secrets."""
        if path.is_file():
            yield from self._scan_file(path)
        elif path.is_dir():
            yield from self._scan_directory(path)
    
    def _scan_directory(self, directory: Path) -> Iterator[SecurityFinding]:
        """Scan all files in a directory."""
        for root, _, files in os.walk(directory):
            # Skip excluded directories
            if any(pattern in root for pattern in self.config.exclude_patterns):
                continue
            
            for filename in files:
                file_path = Path(root) / filename
                if file_path.suffix.lower() not in self.EXCLUDED_EXTENSIONS:
                    yield from self._scan_file(file_path)
    
    def _scan_file(self, file_path: Path) -> Iterator[SecurityFinding]:
        """Scan a single file."""
        # Skip excluded extensions
        if file_path.suffix.lower() in self.EXCLUDED_EXTENSIONS:
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception:
            return
        
        # Use pre-compiled patterns for better performance
        for compiled_pattern, description, severity in self._compiled_patterns:
            if not self.config.should_scan(severity):
                continue
            
            for match in compiled_pattern.finditer(content):
                # Find line number efficiently
                line_num = content[:match.start()].count('\n') + 1
                line_content = lines[line_num - 1] if line_num <= len(lines) else ""
                
                yield SecurityFinding(
                    category=FindingCategory.SECRETS,
                    severity=severity,
                    title=description,
                    description=f"Found in: {line_content.strip()[:100]}",
                    file_path=str(file_path),
                    line_number=line_num,
                    remediation="Remove hardcoded secrets and use environment variables or secure vaults",
                    references=["https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html"],
                )


class PermissionScanner:
    """Scan for permission issues."""
    
    DANGEROUS_PERMISSIONS = {
        0o777: "World-writable file/directory",
        0o666: "World-readable/writable file",
    }
    
    def __init__(self, config: Optional[AuditConfig] = None):
        self.config = config or AuditConfig()
    
    def scan(self, path: Path) -> Iterator[SecurityFinding]:
        """Scan for permission issues."""
        if not self.config.scan_permissions:
            return
        
        if path.is_file():
            yield from self._check_permissions(path)
        elif path.is_dir():
            for root, _, files in os.walk(path):
                for filename in files:
                    file_path = Path(root) / filename
                    yield from self._check_permissions(file_path)
    
    def _check_permissions(self, file_path: Path) -> Iterator[SecurityFinding]:
        """Check file permissions."""
        try:
            stat = file_path.stat()
            mode = stat.st_mode & 0o777
            
            if mode in self.DANGEROUS_PERMISSIONS:
                severity = Severity.HIGH if mode == 0o777 else Severity.MEDIUM
                
                if self.config.should_scan(severity):
                    yield SecurityFinding(
                        category=FindingCategory.PERMISSIONS,
                        severity=severity,
                        title=self.DANGEROUS_PERMISSIONS[mode],
                        description=f"File {file_path} has permissions {oct(mode)}",
                        file_path=str(file_path),
                        remediation=f"Change permissions with: chmod 644 {file_path}",
                    )
        except Exception:
            pass


class DependencyScanner:
    """Scan for vulnerable dependencies."""
    
    def __init__(self, config: Optional[AuditConfig] = None):
        self.config = config or AuditConfig()
    
    def scan(self, path: Path) -> Iterator[SecurityFinding]:
        """Scan dependency files."""
        if not self.config.scan_dependencies:
            return
        
        requirements_file = path / "requirements.txt"
        if requirements_file.exists():
            yield from self._scan_requirements(requirements_file)
        
        package_json = path / "package.json"
        if package_json.exists():
            yield from self._scan_package_json(package_json)
    
    def _scan_requirements(self, file_path: Path) -> Iterator[SecurityFinding]:
        """Scan Python requirements."""
        try:
            content = file_path.read_text()
            
            # Check for unpinned dependencies (security risk)
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and '==' not in line and '>=' not in line:
                    if self.config.should_scan(Severity.MEDIUM):
                        yield SecurityFinding(
                            category=FindingCategory.DEPENDENCIES,
                            severity=Severity.MEDIUM,
                            title="Unpinned dependency",
                            description=f"Dependency '{line}' is not pinned to a specific version",
                            file_path=str(file_path),
                            remediation="Pin dependencies with 'package==version' format",
                            references=["https://pip.pypa.io/en/stable/cli/pip_install/#requirement-specifiers"],
                        )
        except Exception:
            pass
    
    def _scan_package_json(self, file_path: Path) -> Iterator[SecurityFinding]:
        """Scan Node.js package.json."""
        try:
            content = json.loads(file_path.read_text())
            
            # Check for devDependencies that might have vulnerabilities
            deps = content.get('dependencies', {})
            dev_deps = content.get('devDependencies', {})
            
            # Look for known vulnerable packages (simplified check)
            vulnerable_packages = {'lodash': '<4.17.21', 'jquery': '<3.5.0'}
            
            for pkg, version in {**deps, **dev_deps}.items():
                if pkg in vulnerable_packages:
                    if self.config.should_scan(Severity.HIGH):
                        yield SecurityFinding(
                            category=FindingCategory.DEPENDENCIES,
                            severity=Severity.HIGH,
                            title=f"Potentially vulnerable package: {pkg}",
                            description=f"Package {pkg}@{version} may have known vulnerabilities",
                            file_path=str(file_path),
                            remediation=f"Update {pkg} to version {vulnerable_packages[pkg]}",
                            references=["https://www.npmjs.com/advisories"],
                        )
        except Exception:
            pass


class ConfigurationScanner:
    """Scan for security configuration issues."""
    
    def __init__(self, config: Optional[AuditConfig] = None):
        self.config = config or AuditConfig()
    
    def scan(self, path: Path) -> Iterator[SecurityFinding]:
        """Scan configuration files."""
        if not self.config.scan_configuration:
            return
        
        # Scan .env files
        env_file = path / ".env"
        if env_file.exists():
            yield from self._scan_env_file(env_file)
        
        # Scan config files
        config_files = list(path.glob("*.config.*")) + list(path.glob("config.*"))
        for config_file in config_files:
            yield from self._scan_config_file(config_file)
    
    def _scan_env_file(self, file_path: Path) -> Iterator[SecurityFinding]:
        """Scan .env file for issues."""
        try:
            content = file_path.read_text()
            
            # Check if DEBUG is enabled
            if re.search(r'^DEBUG\s*=\s*(true|1|yes)', content, re.IGNORECASE | re.MULTILINE):
                if self.config.should_scan(Severity.MEDIUM):
                    yield SecurityFinding(
                        category=FindingCategory.CONFIGURATION,
                        severity=Severity.MEDIUM,
                        title="DEBUG mode enabled",
                        description="DEBUG mode is enabled in .env file",
                        file_path=str(file_path),
                        remediation="Set DEBUG=false in production",
                    )
        except Exception:
            pass
    
    def _scan_config_file(self, file_path: Path) -> Iterator[SecurityFinding]:
        """Scan generic config files."""
        try:
            content = file_path.read_text()
            
            # Check for hardcoded credentials
            if 'password' in content.lower() or 'secret' in content.lower():
                # This is a simplified check - would need more sophisticated parsing
                pass
        except Exception:
            pass
