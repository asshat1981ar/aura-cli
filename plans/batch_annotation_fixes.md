# SADD Spec: Batch Annotation Fixes

**Summary:** Fix all noqa and type_ignore annotations in a single batch operation.

**Estimated Time:** 10 minutes (vs 15-20 minutes sequential)

---

## Workstream: Fix All Annotations

Fix all noqa and type:ignore annotations across all modules.

**Noqa Files:**
- agents/autogen_agent.py line 22
- core/sadd/session_coordinator.py lines 200, 207
- aura_cli/dispatch.py lines 847, 895
- memory/brain.py lines 18, 31
- tools/github_copilot_mcp.py line 729
- tools/mcp_server.py line 279

**Type:Ignore Files:**
- core/sadd/session_store.py line 179
- core/sadd/workstream_graph.py line 91
- aura_cli/dispatch.py line 842
- tools/mcp_server.py lines 478, 482

**Acceptance:**
- [ ] All noqa annotations resolved or properly justified
- [ ] All type:ignore annotations resolved with proper types
- [ ] Code passes flake8 without warnings
- [ ] Code passes mypy without errors
- [ ] No functional changes to code behavior

---

## Workstream: Validate Linters

Run linters to verify all fixes.

**Acceptance:**
- [ ] flake8 passes with 0 warnings
- [ ] mypy passes with 0 errors
- [ ] Overall code quality score improved

Depends on: Fix All Annotations
