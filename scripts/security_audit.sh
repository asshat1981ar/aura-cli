#!/usr/bin/env bash
# Security audit script for AURA CLI
# Run: ./scripts/security_audit.sh
set -euo pipefail

echo "=== AURA CLI Security Audit ==="
echo ""

# 1. Bandit static analysis
echo "--- Bandit: Python security linting ---"
if command -v bandit &>/dev/null; then
    bandit -c pyproject.toml -r aura_cli/ core/ agents/ memory/ tools/ -f screen || true
else
    echo "SKIP: bandit not installed (pip install bandit[toml])"
fi
echo ""

# 2. Dependency audit
echo "--- pip-audit: Dependency vulnerability scan ---"
if command -v pip-audit &>/dev/null; then
    pip-audit --strict --desc on || true
else
    echo "SKIP: pip-audit not installed (pip install pip-audit)"
fi
echo ""

# 3. Secret detection
echo "--- detect-secrets: Scanning for leaked secrets ---"
if command -v detect-secrets &>/dev/null; then
    detect-secrets scan --exclude-files '(node_modules|\.git|\.secrets\.baseline|package-lock\.json)' \
        --exclude-lines '(YOUR_API_KEY_HERE|YOUR_OPENROUTER_API_KEY|placeholder)' || true
else
    echo "SKIP: detect-secrets not installed (pip install detect-secrets)"
fi
echo ""

# 4. Check for common dangerous patterns
echo "--- Pattern check: Common security anti-patterns ---"
ISSUES=0

# Check for eval/exec usage outside tests
if grep -rn 'eval(' --include='*.py' aura_cli/ core/ agents/ memory/ tools/ 2>/dev/null | grep -v '__pycache__' | grep -v 'test_'; then
    echo "WARNING: eval() usage found in production code"
    ISSUES=$((ISSUES + 1))
fi

# Check for hardcoded secrets patterns
if grep -rn 'password\s*=\s*["\x27][^"\x27]\+["\x27]' --include='*.py' aura_cli/ core/ agents/ memory/ 2>/dev/null | grep -v '__pycache__' | grep -v 'test_' | grep -v 'placeholder' | grep -v 'YOUR_'; then
    echo "WARNING: Possible hardcoded password found"
    ISSUES=$((ISSUES + 1))
fi

if [ "$ISSUES" -eq 0 ]; then
    echo "No common anti-patterns detected."
fi

echo ""
echo "=== Audit complete ==="
