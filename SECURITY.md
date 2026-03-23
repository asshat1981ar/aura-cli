# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |

## Reporting a Vulnerability

We take the security of aura-cli seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

1. **Do NOT open a public issue** for security vulnerabilities.
2. Instead, please use one of the following methods:
   - **GitHub Private Vulnerability Reporting:** Use the "Report a vulnerability" button on the [Security tab](https://github.com/asshat1981ar/aura-cli/security/advisories/new).
   - **Email:** Send details to the repository owner via their GitHub profile.

### What to Include

- A description of the vulnerability
- Steps to reproduce the issue
- The potential impact
- Any suggested fixes (if you have them)

### What to Expect

- **Acknowledgment:** We will acknowledge receipt of your report within 48 hours.
- **Assessment:** We will investigate and assess the vulnerability within 7 days.
- **Resolution:** We aim to release a fix within 30 days for confirmed vulnerabilities.
- **Credit:** We will credit reporters in the release notes (unless you prefer to remain anonymous).

### Scope

The following are in scope for security reports:

- Authentication and authorization flaws
- Injection vulnerabilities (SQL, command, etc.)
- Secrets or credentials exposed in code or logs
- Dependency vulnerabilities with a known exploit path
- Configuration issues that could lead to data exposure

### Out of Scope

- Issues in third-party dependencies without a demonstrated exploit path in aura-cli
- Social engineering attacks
- Denial of service attacks that require excessive resources

## Security Best Practices for Contributors

- Never commit secrets, API keys, or credentials to the repository
- Use environment variables for sensitive configuration
- Keep dependencies updated and review security advisories
- Follow the principle of least privilege in code and configuration
- Run `pre-commit` hooks which include security checks before committing

## Dependencies

We monitor our dependencies for known vulnerabilities. If you notice a vulnerable dependency, please open an issue or report it through the security process above.
