# Sandbox Security Audit Report

**Version:** 1.0.0  
**Date:** 2026-04-09  
**Auditor:** AURA Security Agent  
**Scope:** `agents/sandbox.py` — Isolated code execution for pre-apply validation

---

## Executive Summary

The AURA Sandbox Agent provides defense-in-depth isolation for executing LLM-generated code before it reaches the project filesystem. This audit reviews the security controls implemented and identifies residual risks.

**Overall Security Posture:** GOOD  
**Risk Level:** MEDIUM (acceptable for intended use case)

---

## Security Controls Implemented

### 1. Process Isolation ✅

| Control | Implementation | Status |
|---------|---------------|--------|
| Subprocess isolation | Code runs in separate Python subprocess | ✅ Implemented |
| Temp directory isolation | Each execution uses `tempfile.TemporaryDirectory()` | ✅ Implemented |
| Automatic cleanup | Temp directories deleted after execution | ✅ Implemented |
| Working directory restriction | `cwd` locked to temp directory | ✅ Implemented |

### 2. Resource Limits ✅

| Control | Implementation | Status |
|---------|---------------|--------|
| CPU time limit | 30 seconds (RLIMIT_CPU) | ✅ Implemented |
| Memory limit | 512 MiB virtual address space (RLIMIT_AS) | ✅ Implemented |
| Wall-clock timeout | Configurable (default 30s) | ✅ Implemented |
| Output size limits | Clamped to 64KB-256KB | ✅ Implemented |

*Note: Resource limits use Unix `resource` module; silently skipped on Windows.*

### 3. Network Restrictions ✅

| Control | Implementation | Status |
|---------|---------------|--------|
| Outbound blocking | HTTP(S)_PROXY routed to 127.0.0.1:1 | ✅ Implemented |
| SSL bypass prevention | REQUESTS_CA_BUNDLE cleared | ✅ Implemented |
| Environment isolation | Clean env with only allowlisted vars | ✅ Implemented |

### 4. Filesystem Restrictions ✅

| Control | Implementation | Status |
|---------|---------------|--------|
| User site-packages disabled | PYTHONNOUSERSITE=1 | ✅ Implemented |
| Bytecode disabled | PYTHONDONTWRITEBYTECODE=1 | ✅ Implemented |
| Python path restricted | sys.path filtered in wrapper | ✅ Implemented |
| Open() wrapper | Runtime check for path allowlist | ✅ Implemented |
| Temp directory enforcement | Execution only in temp dirs | ✅ Implemented |

### 5. Command Security ✅

| Control | Implementation | Status |
|---------|---------------|--------|
| Command sanitization | `core.sanitizer.sanitize_command()` | ✅ Implemented |
| Denylist patterns | `rm -rf /`, `mkfs`, `reboot`, etc. | ✅ Implemented |
| SecurityError handling | Catches and logs violations | ✅ Implemented |

### 6. Monitoring & Auditing ✅

| Control | Implementation | Status |
|---------|---------------|--------|
| Execution logging | All runs logged via `log_json()` | ✅ Implemented |
| Violation detection | Regex patterns for common attacks | ✅ Implemented |
| Telemetry integration | Results persisted to brain memory | ✅ Implemented |
| Prometheus metrics | `aura_sandbox_violations_total` counter | ✅ Implemented |

---

## Residual Risks

### HIGH Severity

*None identified.*

### MEDIUM Severity

1. **Windows Compatibility Gap**
   - Resource limits (CPU/memory) not enforced on Windows
   - **Mitigation:** Documented; timeout still enforced via `communicate(timeout=)`
   - **Recommendation:** Consider Windows Job Objects for parity

2. **Path Traversal in Wrapped Code**
   - The `open()` wrapper provides defense-in-depth but relies on string prefix matching
   - **Mitigation:** Multiple layers (temp dir + wrapper + cwd restriction)
   - **Recommendation:** Consider seccomp-bpf or Landlock on Linux

### LOW Severity

1. **Temporary Directory Exposure**
   - Temp directories use standard system paths (predictable prefix)
   - **Mitigation:** Random suffix, immediate cleanup
   - **Recommendation:** Consider private temp directory (e.g., `dir_fd` on Linux)

2. **Pytest Plugin Risks**
   - `PYTEST_DISABLE_PLUGIN_AUTOLOAD` reduces but doesn't eliminate plugin risks
   - **Mitigation:** Tests run in isolated temp directory
   - **Recommendation:** Consider `conftest.py` scanning

---

## Recommendations

### Short-term (v1.0.x)

1. **Add unit tests for filesystem restrictions** — Verify wrapper prevents writes outside temp dir
2. **Add integration test for network blocking** — Confirm proxy routing works
3. **Document Windows limitations** — Clear guidance for Windows deployments

### Medium-term (v1.1.x)

1. **Evaluate seccomp-bpf** — Linux-specific syscall filtering
2. **Evaluate Landlock LSM** — Native Linux path restriction
3. **Windows Job Objects** — Resource limits parity

### Long-term (v2.x)

1. **Container sandbox option** — Run sandbox in minimal container
2. **Wasm sandbox** — Compile Python to Wasm for stronger isolation
3. **MicroVM sandbox** — Firecracker/gVisor for cloud deployments

---

## Compliance Mapping

| Framework | Requirement | Status |
|-----------|-------------|--------|
| OWASP ASVS L1 | V5.2 Input Validation | ✅ Met |
| OWASP ASVS L1 | V12.3 File Execution | ✅ Met |
| NIST 800-53 | SC-7 Boundary Protection | ⚠️ Partial (network) |
| NIST 800-53 | SC-39 Process Isolation | ✅ Met |

---

## Sign-off

**Security Assessment:** APPROVED for production use with documented limitations.

**Conditions:**
1. Do not enable `AGENT_API_ENABLE_RUN=1` on untrusted networks
2. Monitor `aura_sandbox_violations_total` metric
3. Review sandbox logs regularly for attack patterns

---

## Appendix: Violation Patterns

The following patterns are monitored in sandbox output:

```python
_VIOLATION_PATTERNS = [
    (re.compile(r"\bPermissionError\b"), "restricted_path_access"),
    (re.compile(r"\b(?:ModuleNotFoundError|ImportError)\b", re.IGNORECASE), "blocked_import"),
    (re.compile(r"\bSecurityError\b|\bAccess denied\b"), "blocked_command"),
    (re.compile(r"\bBlockingIOError\b|Operation not permitted", re.IGNORECASE), "blocked_syscall"),
]
```

Each match triggers a `sandbox_violation_attempt` log entry with:
- `violation_type`: Pattern category
- `attempted_value`: Matching text
- `goal`: Contextual goal (when available)
- `agent`: Always "sandbox"
