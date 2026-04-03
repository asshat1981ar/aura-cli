# SADD Spec: Fleet Prompt Dispatcher Implementation

**Summary:** Build a fully autonomous GitHub Issue → code → PR → merge pipeline using n8n workflows, AURA CLI APIs, and GitHub Actions. The dispatcher classifies issues via LLM, runs AURA Skills analysis, generates code changes via AURA Loop, quality-gates output, pushes PRs, and auto-merges on CI green.

Reference spec: `docs/superpowers/specs/2026-03-29-fleet-prompt-dispatcher-design.md`
Reference plan: `docs/superpowers/specs/2026-03-29-fleet-prompt-dispatcher-plan.md`

---

## Workstream: GitHub Labels and Fleet Trigger Workflow

Create the GitHub labels for the fleet dispatcher and the GitHub Actions trigger workflow.

**Acceptance:**
- [ ] `fleet-trigger.yml` exists at `.github/workflows/fleet-trigger.yml`
- [ ] Workflow triggers on `issues: [labeled]` with `fleet:trigger` label
- [ ] Workflow POSTs to `${{ secrets.N8N_FLEET_WEBHOOK_URL }}` with the full event payload
- [ ] Workflow uses pinned `actions/checkout` SHA (not a tag)
- [ ] `docs/fleet-labels-setup.sh` script created to run `gh label create` for all 4 fleet labels

---

## Workstream: WF-0 Master Dispatcher n8n Workflow

Create the n8n Master Dispatcher workflow as a JSON export file that can be imported into n8n.

**Depends on:** GitHub Labels and Fleet Trigger Workflow

**Acceptance:**
- [ ] File `n8n-workflows/WF-0-master-dispatcher.json` created
- [ ] Workflow contains: Webhook node (path `/fleet`), Filter node (skip done/blocked), Set node (extract issue fields), LLM Classifier node (OpenAI/generic HTTP), Switch node (routes to 5 sub-flows), Label node (apply fleet:in-progress)
- [ ] LLM classifier prompt template matches spec exactly (classifies to bug/feature/refactor/security/docs)
- [ ] Switch has fallback to `feature` for unknown types
- [ ] Workflow is valid n8n JSON (has `nodes`, `connections`, `settings` keys)

---

## Workstream: WF-1 through WF-5 Handler Workflows

Create the 5 type-specific handler workflows as n8n JSON export files.

**Depends on:** WF-0 Master Dispatcher n8n Workflow

**Acceptance:**
- [ ] Files `n8n-workflows/WF-{1,2,3,4,5}-*.json` created (one per handler)
- [ ] WF-1 (bug): calls `complexity_scorer` and `error_pattern_matcher` with correct args
- [ ] WF-2 (feature): calls `architecture_validator` and `dependency_analyzer`
- [ ] WF-3 (refactor): calls `code_clone_detector`, `refactoring_advisor`, `tech_debt_quantifier`
- [ ] WF-4 (security): calls `security_scanner` and `type_checker`
- [ ] WF-5 (docs): calls `doc_generator` and `linter_enforcer`
- [ ] Each workflow: parallel HTTP nodes for skills fan-out, Merge node, Set node builds analyst_context, ExecuteWorkflow node calls WF-6
- [ ] Skills called via `POST {{ $env.AURA_SKILLS_URL }}/call` with Bearer `{{ $env.MCP_API_TOKEN }}`
- [ ] Each workflow is valid n8n JSON

---

## Workstream: WF-6 Code Gen and PR Push Workflow

Create the core code generation and PR push workflow as a n8n JSON export file. This is the most complex workflow — it calls AURA Loop, polls status, commits changes, quality-gates, and creates/merges PRs.

**Depends on:** WF-1 through WF-5 Handler Workflows

**Acceptance:**
- [ ] File `n8n-workflows/WF-6-code-gen-pr-push.json` created
- [ ] Node 1: GitHub API branch create (`fleet/{type}/{issue}` from main HEAD SHA)
- [ ] Node 2: Render goal prompt from analyst_context template
- [ ] Node 3: POST to `{{ $env.AURA_API_URL }}/webhook/goal` with goal, priority:5, metadata
- [ ] Node 4: Poll `GET /webhook/status/{goal_id}` at 30s intervals; handles done/failed/queued/running
- [ ] Node 5: POST to `/execute` to git checkout branch + git add -A + git commit + git push using `{{ $env.AURA_REPO_PATH }}`
- [ ] Node 6: Extract `applied_files[0]` from `result.history[-1].cycle_summary.applied_files`; read file via `/execute`
- [ ] Node 7: Quality gate via `generation_quality_checker`; threshold 70; up to 2 retries (re-dispatches to Node 3)
- [ ] Node 8: GitHub API create PR with `fleet(#{issue}): {title}` naming and issue link
- [ ] Node 9: Poll PR check-runs; wait for all green; up to 3 CI retries
- [ ] Node 10: GitHub API merge PR (squash)
- [ ] Node 11: Apply `fleet:done` label; close issue
- [ ] Node 12: Escalation node — apply `fleet:blocked`, post structured blocker comment, remove `fleet:in-progress`
- [ ] Workflow is valid n8n JSON

---

## Workstream: Integration Test Scripts

Create shell/Python test scripts that validate each layer of the fleet dispatcher independently, plus a README for the n8n-workflows directory.

**Depends on:** WF-6 Code Gen and PR Push Workflow

**Acceptance:**
- [ ] `n8n-workflows/README.md` created documenting all 7 workflows, import instructions, required credentials, and env vars
- [ ] `scripts/test_fleet_api.sh` created — tests AURA API endpoints (`/webhook/goal`, `/webhook/status`, `/execute`) and Skills API; prints PASS/FAIL per endpoint
- [ ] `scripts/test_fleet_quality_gate.sh` created — invokes `generation_quality_checker` with a sample payload; validates score is numeric 0-100
- [ ] Scripts are executable (`chmod +x`)
