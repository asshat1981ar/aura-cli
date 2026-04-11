"""Tests for security scanners."""

import os
import pytest
from pathlib import Path

from aura.security.models import AuditConfig, FindingCategory, Severity
from aura.security.scanners import (
    ConfigurationScanner,
    DependencyScanner,
    PermissionScanner,
    SecretScanner,
)


class TestSecretScanner:
    @pytest.fixture
    def scanner(self):
        return SecretScanner()
    
    def test_scan_file_with_api_key(self, tmp_path, scanner):
        test_file = tmp_path / "test.py"
        test_file.write_text('api_key = "sk_test_1234567890abcdef"')
        
        findings = list(scanner.scan(test_file))
        
        assert len(findings) >= 1
        assert any(f.category == FindingCategory.SECRETS for f in findings)
    
    def test_scan_file_with_password(self, tmp_path, scanner):
        test_file = tmp_path / "config.py"
        test_file.write_text('password = "supersecret123"')
        
        findings = list(scanner.scan(test_file))
        
        # Should detect password
        password_findings = [f for f in findings if "password" in f.title.lower()]
        assert len(password_findings) >= 1
        assert any(f.severity == Severity.CRITICAL for f in password_findings)
    
    def test_scan_file_with_private_key(self, tmp_path, scanner):
        test_file = tmp_path / "key.pem"
        test_file.write_text('-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...')
        
        findings = list(scanner.scan(test_file))
        
        key_findings = [f for f in findings if "private key" in f.title.lower()]
        assert len(key_findings) >= 1
    
    def test_scan_file_with_aws_key(self, tmp_path, scanner):
        test_file = tmp_path / "aws.py"
        test_file.write_text('aws_access_key = "AKIAIOSFODNN7EXAMPLE"')
        
        findings = list(scanner.scan(test_file))
        
        aws_findings = [f for f in findings if "AWS" in f.title]
        assert len(aws_findings) >= 1
    
    def test_scan_clean_file(self, tmp_path, scanner):
        test_file = tmp_path / "clean.py"
        test_file.write_text('def hello():\n    print("Hello World")')
        
        findings = list(scanner.scan(test_file))
        
        assert len(findings) == 0
    
    def test_excluded_extensions_not_scanned(self, tmp_path, scanner):
        test_file = tmp_path / "image.png"
        test_file.write_bytes(b'fake_png_data_with_password = "secret"')
        
        findings = list(scanner.scan(test_file))
        
        assert len(findings) == 0
    
    def test_scan_directory(self, tmp_path, scanner):
        # Create multiple files
        (tmp_path / "file1.py").write_text('api_key = "secret1"')
        (tmp_path / "file2.py").write_text('password = "secret2"')
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file3.py").write_text('token = "secret3"')
        
        findings = list(scanner.scan(tmp_path))
        
        # Should find at least one finding per file
        assert len(findings) >= 1


class TestPermissionScanner:
    @pytest.fixture
    def scanner(self):
        return PermissionScanner()
    
    def test_scan_world_writable_file(self, tmp_path, scanner):
        test_file = tmp_path / "writable.txt"
        test_file.write_text("content")
        os.chmod(test_file, 0o777)
        
        findings = list(scanner.scan(test_file))
        
        assert len(findings) == 1
        assert findings[0].severity == Severity.HIGH
        assert "World-writable" in findings[0].title
    
    def test_scan_normal_permissions(self, tmp_path, scanner):
        test_file = tmp_path / "normal.txt"
        test_file.write_text("content")
        os.chmod(test_file, 0o644)
        
        findings = list(scanner.scan(test_file))
        
        assert len(findings) == 0


class TestDependencyScanner:
    @pytest.fixture
    def scanner(self):
        return DependencyScanner()
    
    def test_scan_requirements_with_unpinned_deps(self, tmp_path, scanner):
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("requests\nflask>=1.0\ndjango==3.0")
        
        findings = list(scanner.scan(tmp_path))
        
        unpinned = [f for f in findings if "Unpinned" in f.title]
        assert len(unpinned) == 1  # Only 'requests' is unpinned
        assert unpinned[0].category == FindingCategory.DEPENDENCIES
    
    def test_scan_requirements_all_pinned(self, tmp_path, scanner):
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("requests==2.28.0\nflask==2.0.0")
        
        findings = list(scanner.scan(tmp_path))
        
        unpinned = [f for f in findings if "Unpinned" in f.title]
        assert len(unpinned) == 0


class TestConfigurationScanner:
    @pytest.fixture
    def scanner(self):
        return ConfigurationScanner()
    
    def test_scan_env_with_debug_enabled(self, tmp_path, scanner):
        env_file = tmp_path / ".env"
        env_file.write_text("DEBUG=true\nAPI_KEY=test")
        
        findings = list(scanner.scan(tmp_path))
        
        debug_findings = [f for f in findings if "DEBUG" in f.title]
        assert len(debug_findings) == 1
        assert debug_findings[0].severity == Severity.MEDIUM
