# Security Policy

## Reporting a Vulnerability

We take the security of AURA CLI seriously. If you discover a security vulnerability, please follow these steps:

### 🚨 DO NOT open a public GitHub issue for security vulnerabilities

Instead, please report security issues through one of these channels:

1. **Preferred:** Use [GitHub Security Advisories](https://github.com/asshat1981ar/aura-cli/security/advisories/new) to privately report the vulnerability
2. **Alternative:** Email the maintainers (check repository for current contact info)

### What to Include

Please include as much of the following information as possible:

- **Type of vulnerability**: e.g., RCE, XSS, authentication bypass, etc.
- **Affected component**: File, module, or feature affected
- **Steps to reproduce**: Detailed steps to reproduce the vulnerability
- **Impact**: What an attacker could do by exploiting this
- **Suggested fix**: If you have ideas for how to fix it (optional)
- **Any supporting materials**: PoC code, screenshots, logs (sanitize sensitive data!)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt within 48 hours
- **Updates**: We will provide updates on our investigation every 3-5 business days
- **Timeline**: We aim to release a fix within 30 days for critical vulnerabilities
- **Credit**: We will credit you in the security advisory (unless you prefer to remain anonymous)

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

We provide security updates for the latest stable release and the main development branch.

## Security Considerations for Users

### API Keys and Secrets

**Never commit API keys or tokens to version control!**

- Use environment variables (`AURA_API_KEY`, `AGENT_API_TOKEN`, etc.)
- Use `.env` files (already in `.gitignore`)
- Store keys in `aura.config.json` (already in `.gitignore`)
- Use secret management tools (e.g., 1Password, AWS Secrets Manager) in production

### Authentication

- **Always set `AGENT_API_TOKEN`** when exposing the server API to a network
- Use strong, random tokens (e.g., `openssl rand -hex 32`)
- Rotate tokens periodically
- Don't share tokens via insecure channels

### Hook Commands

- **Review hook commands carefully** - they execute with shell privileges
- Prefer specific commands over shell scripts where possible
- Avoid user-supplied input in hook commands
- Use absolute paths for executables
- Set appropriate timeouts

Example of safer hook configuration:

```json
{
  "hooks": {
    "pre_apply": {
      "command": "/usr/local/bin/my-validator --strict",
      "timeout_seconds": 30,
      "on_failure": "block"
    }
  }
}
```

### A2A Protocol

- **Disable A2A in production** unless specifically needed (`"a2a": {"enabled": false}`)
- If enabled, ensure it's behind authentication
- Only expose to trusted networks
- Use firewall rules to restrict access

### Memory and Logs

- AURA logs automatically mask secrets, but verify before sharing logs
- Memory database files may contain sensitive context - don't share publicly
- Use `--dry-run` when testing with sensitive code

### Sandboxing

- AURA sandboxes generated code execution, but it's not a security boundary
- Don't run AURA on untrusted goals in production environments
- Review generated code before applying in critical systems

### Dependencies

- Keep dependencies updated: `pip install --upgrade -r requirements.txt`
- Monitor security advisories (Dependabot will alert if enabled)
- Review dependency changes in PR diffs

## Known Security Features

### Implemented Protections

- ✅ Secret masking in logs
- ✅ Bearer token authentication for API
- ✅ Command sanitization for shell execution
- ✅ Timeout enforcement for hook execution
- ✅ Sandbox for code execution (non-security boundary)
- ✅ Optional dependency guards prevent import errors

### Planned Enhancements

- 🔄 Rate limiting for API endpoints
- 🔄 Audit logging for sensitive operations
- 🔄 Input validation hardening
- 🔄 RBAC for multi-user scenarios

## Security Best Practices

### Development

1. **Never hardcode secrets** in source code
2. **Validate all inputs** before processing
3. **Use parameterized queries** not string concatenation
4. **Catch specific exceptions** not broad `except Exception`
5. **Log security events** (authentication failures, access denials)
6. **Use type hints** to catch type-related bugs early
7. **Write tests** for security-critical paths

### Deployment

1. **Use environment variables** for configuration
2. **Enable authentication** with strong tokens
3. **Run with minimal privileges** (don't run as root)
4. **Use HTTPS** for API access
5. **Restrict network access** with firewalls
6. **Monitor logs** for suspicious activity
7. **Keep system updated** with security patches

### Code Review

When reviewing PRs, check for:

- [ ] No hardcoded secrets or API keys
- [ ] Authentication required for sensitive endpoints
- [ ] Input validation on all external inputs
- [ ] Proper error handling (no information leakage)
- [ ] Safe command execution (no shell injection)
- [ ] Secure defaults (opt-in for dangerous features)
- [ ] Tests for security-critical logic

## Security Updates

We announce security updates through:

- GitHub Security Advisories
- Release notes
- GitHub Discussions (for major vulnerabilities)

Subscribe to repository notifications to stay informed.

## Related Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md) - Contributing guidelines
- [README.md](README.md) - Project overview and setup
- [CLAUDE.md](CLAUDE.md) - Architecture and conventions

## Questions?

For general security questions (not vulnerabilities):
- Open a discussion in GitHub Discussions
- Ask in existing issues

For vulnerability reports: Use the process described at the top of this document.

---

Last updated: 2026-03-21
