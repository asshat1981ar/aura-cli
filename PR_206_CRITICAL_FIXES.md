# Critical Fixes Required for PR #206

This document catalogs all critical security, correctness, and functionality issues found in PR #206 that must be addressed before merge.

## 🚨 P1 - Critical Security Issues

### 1. A2A Routes Bypass Authentication
**File:** `aura_cli/server.py:76`
**Issue:** A2A routes (`/a2a/tasks`) are registered without `Depends(require_auth)`, allowing unauthenticated remote goal execution even when `AGENT_API_TOKEN` is set.

**Fix:**
```python
# Option 1: Add auth dependency to FastAPI app globally
app = FastAPI(
    title="AURA Agent API",
    version="0.2.0",
    dependencies=[Depends(require_auth)]  # Protect all endpoints
)

# Option 2: Gate A2A behind config toggle and add auth
if os.getenv("AURA_A2A_ENABLED") == "1":
    # ... A2A setup code ...
    # When registering routes, ensure they require auth
```

### 2. Event Streaming Endpoints Lack Authentication
**File:** `aura_cli/server.py:113+`
**Issue:** `/events/stream`, `/events/publish`, and `/events/history` endpoints do not enforce authentication. `/events/publish` allows any client to inject events.

**Fix:**
```python
@app.get("/events/stream")
async def event_stream(auth=Depends(require_auth)):
    # ... existing code ...

@app.post("/events/publish")
async def event_publish(event_data: dict, auth=Depends(require_auth)):
    # ... existing code ...
```

### 3. Shell Command Injection Risk
**File:** `core/hooks.py:163`
**Issue:** Using `subprocess.run(..., shell=True)` with hook commands can introduce command injection vulnerabilities.

**Fix:**
```python
# Use shlex.split to properly parse commands without shell=True
import shlex

# Replace:
# subprocess.run(hook.command, shell=True, ...)

# With:
try:
    cmd_parts = shlex.split(hook.command)
    result = subprocess.run(
        cmd_parts,  # No shell=True
        capture_output=True,
        text=True,
        timeout=hook.timeout_seconds,
        env=merged_env,
    )
except ValueError as e:
    log_json("ERROR", "hook_command_parse_failed",
             details={"command": hook.command, "error": str(e)})
    return HookResult.BLOCK
```

## 🔴 P1 - Critical Functionality Bugs

### 4. Experiment Tracker Git Revert Fails
**File:** `core/evolution_loop.py:287`
**Issue:** Calls `self.git.run(...)` but `GitTools` doesn't have a `run` method. Regressions are never reverted.

**Fix:**
```python
# In discard method, replace:
# self.git.run(["git", "revert", "--no-commit", experiment.baseline_commit])

# With GitTools API:
try:
    # GitTools uses subprocess directly, not a run() method
    import subprocess
    result = subprocess.run(
        ["git", "revert", "--no-commit", experiment.baseline_commit],
        cwd=self.project_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log_json("ERROR", "revert_failed",
                 details={"stderr": result.stderr})
except Exception as exc:
    log_json("ERROR", "experiment_discard_revert_failed",
             details={"error": str(exc)})
```

### 5. NBest Sandbox Adapter Signature Mismatch
**File:** `core/nbest.py:91` and `core/orchestrator.py:636`
**Issue:** `NBestEngine.sandbox_all()` calls `sandbox_agent.run(changes, context)` but `SandboxAdapter.run()` expects `input_data: dict`.

**Fix in `core/nbest.py`:**
```python
def sandbox_all(self, sandbox_agent, candidates: list[CodeCandidate],
                context: dict | None = None) -> list[CodeCandidate]:
    """Run all candidates through sandbox, mark pass/fail."""
    for candidate in candidates:
        if not candidate.changes:
            continue
        try:
            # Adapt to SandboxAdapter's expected input shape
            input_data = {
                "act": {"changes": candidate.changes},
                "dry_run": context.get("dry_run", True) if context else True,
            }
            if hasattr(context, "get") and "project_root" in context:
                input_data["project_root"] = context["project_root"]

            run_fn = getattr(sandbox_agent, "run", None)
            if callable(run_fn):
                result = run_fn(input_data)
            else:
                result = {"success": True, "output": "sandbox_skipped"}
            candidate.sandbox_passed = result.get("success", False)
            candidate.sandbox_output = str(result.get("output", ""))[:2000]
        except Exception as exc:
            log_json("WARN", "sandbox_candidate_failed",
                     details={"variant": candidate.variant_id, "error": str(exc)})
            candidate.sandbox_output = str(exc)
    return candidates
```

**Fix in `core/orchestrator.py`:**
```python
# Around line 636, wrap sandbox_agent for NBest compatibility
def _sandbox_wrapper(changes, context):
    """Adapt NBest signature to SandboxAdapter input shape."""
    input_data = {
        "act": {"changes": changes},
        "dry_run": context.get("dry_run", False) if context else False,
    }
    return sandbox_agent.run(input_data)

candidates = engine.sandbox_all(_sandbox_wrapper, candidates, context)
```

### 6. Missing `.secrets.baseline` File
**File:** `.pre-commit-config.yaml:36`
**Issue:** detect-secrets hook configured with `--baseline .secrets.baseline` but file doesn't exist. Pre-commit will fail for all contributors.

**Fix:**
```bash
# Generate the baseline file:
detect-secrets scan > .secrets.baseline

# Or remove the --baseline argument:
```
```yaml
- repo: https://github.com/Yelp/detect-secrets
  rev: v1.5.0
  hooks:
    - id: detect-secrets
      # Remove: args: ['--baseline', '.secrets.baseline']
      exclude: '(package-lock\.json)$'
```

## 🟡 P2 - Important Correctness Issues

### 7. A2A Port Mismatch
**File:** `aura_cli/server.py:63`
**Issue:** `AgentCard.default()` uses port 8000 but server's `__main__` uses 8080 by default.

**Fix:**
```python
# Use consistent default
a2a_server = A2AServer(AgentCard.default(port=int(os.getenv("PORT", "8080"))))
```

### 8. NBest Temperature Not Applied
**File:** `core/nbest.py:68`
**Issue:** Temperature is set on candidates but never passed to model calls.

**Fix:**
```python
# In generate_candidates method:
try:
    respond_fn = getattr(model, "respond_for_role", None)
    if callable(respond_fn):
        # Try to pass temperature if supported
        try:
            response = respond_fn("code_generation", variant_prompt, temperature=temp)
        except TypeError:
            response = respond_fn("code_generation", variant_prompt)
    else:
        try:
            response = model.respond(variant_prompt, temperature=temp)
        except TypeError:
            response = model.respond(variant_prompt)
    # ... rest of code
```

### 9. Memory Consolidation Never Runs Every 10 Cycles
**File:** `core/orchestrator.py:1126`
**Issue:** Uses random UUID hex suffix instead of actual cycle counter. Also doesn't check config toggle or persist results.

**Fix:**
```python
# Add instance variable to track cycles
def __init__(self, ...):
    # ... existing init ...
    self._cycle_count = 0

# In run_cycle:
def run_cycle(self, goal: str, dry_run: bool = False) -> dict:
    self._cycle_count += 1
    # ... existing code ...

    # At the end, before return:
    config = runtime.get("config", {})
    consolidation_enabled = config.get("memory_consolidation", {}).get("enabled", True)

    if consolidation_enabled and self._cycle_count % 10 == 0:
        try:
            from memory.consolidation import MemoryConsolidator
            consolidator = MemoryConsolidator(self.brain, threshold=0.7)
            result = consolidator.consolidate()

            # Actually persist the consolidated memories back
            if result.memories_after < result.memories_before:
                log_json("INFO", "memory_consolidation_complete",
                         details={"before": result.memories_before,
                                  "after": result.memories_after,
                                  "compression": f"{result.compression_ratio:.1%}"})
        except Exception as exc:
            log_json("WARN", "memory_consolidation_error",
                     details={"error": str(exc)})
```

### 10. Experiment Tracker Uses Wrong Path
**File:** `core/evolution_loop.py:52`
**Issue:** Uses `Path(__file__).parent.parent / "memory"` which may be read-only or wrong location.

**Fix:**
```python
def __init__(self, orchestrator, brain, config: dict | None = None, project_root: Path | None = None):
    self.orchestrator = orchestrator
    self.brain = brain
    self.config = config or {}
    self.project_root = project_root or Path.cwd()

    # Use configured memory directory or project_root/memory
    memory_dir = self.config.get("memory_dir")
    if memory_dir:
        self.experiments_file = Path(memory_dir) / "experiments.json"
    else:
        self.experiments_file = self.project_root / "memory" / "experiments.json"

    self.experiments_file.parent.mkdir(parents=True, exist_ok=True)
    # ... rest of init
```

### 11. NBest Critic Scoring Has No Fallback
**File:** `core/nbest.py:125`
**Issue:** If critic response doesn't parse, `max(..., key=total_score)` will pick arbitrary candidate.

**Fix:**
```python
def critic_tournament(self, model, candidates: list[CodeCandidate],
                      goal: str) -> CodeCandidate:
    scoreable = [c for c in candidates if c.changes]
    if not scoreable:
        raise ValueError("No candidates produced valid changes")
    if len(scoreable) == 1:
        scoreable[0].total_score = 1.0
        return scoreable[0]

    comparison_prompt = self._build_comparison_prompt(scoreable, goal)
    scores_parsed = False
    try:
        respond_fn = getattr(model, "respond_for_role", None)
        if callable(respond_fn):
            scoring_response = respond_fn("critique", comparison_prompt)
        else:
            scoring_response = model.respond(comparison_prompt)
        scores_parsed = self._parse_scores(scoring_response, scoreable)
    except Exception:
        pass

    # Apply fallback scoring if parsing failed
    if not scores_parsed:
        log_json("WARN", "critic_scoring_failed_using_fallback")
        for c in scoreable:
            c.total_score = (1.0 if c.sandbox_passed else 0.0) + (1.0 - c.temperature)

    winner = max(scoreable, key=lambda c: c.total_score)
    # ... rest of method
```

Then update `_parse_scores` to return bool:
```python
def _parse_scores(self, response: str, candidates: list[CodeCandidate]) -> bool:
    """Parse JSON scoring response, return True if any scores were parsed."""
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if not json_match:
        return False

    try:
        data = json.loads(json_match.group())
        scores_dict = data.get("scores", {})
        if not scores_dict:
            return False

        any_scored = False
        for candidate in candidates:
            vid_str = str(candidate.variant_id)
            if vid_str in scores_dict:
                candidate.scores = scores_dict[vid_str]
                candidate.total_score = sum(candidate.scores.values()) / len(SCORING_AXES)
                any_scored = True

        return any_scored
    except (json.JSONDecodeError, KeyError, ValueError):
        return False
```

### 12. EventBus Callback Registry Memory Leak
**File:** `core/mcp_events.py:195`
**Issue:** `register_callback()` subscribes without unsubscribe mechanism. Memory leak over time.

**Fix:**
```python
class CallbackRegistry:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._active_callbacks: dict[str, callable] = {}

    def register_callback(self, correlation_id: str, callback: callable, timeout_sec: float = 30.0):
        """Register a callback that auto-unsubscribes after firing or timeout."""
        def wrapper(event: MCPEvent):
            if event.correlation_id == correlation_id:
                # Remove callback after firing
                if correlation_id in self._active_callbacks:
                    self.event_bus.unsubscribe(wrapper)
                    del self._active_callbacks[correlation_id]
                callback(event)

        self._active_callbacks[correlation_id] = wrapper
        self.event_bus.subscribe(wrapper)

        # Schedule cleanup after timeout
        import threading
        def cleanup():
            if correlation_id in self._active_callbacks:
                self.event_bus.unsubscribe(self._active_callbacks[correlation_id])
                del self._active_callbacks[correlation_id]

        timer = threading.Timer(timeout_sec, cleanup)
        timer.daemon = True
        timer.start()
```

### 13. A2A Client Blocks Event Loop
**File:** `core/a2a/client.py:52`
**Issue:** `async` functions use blocking `urllib.request.urlopen()`.

**Fix:**
```python
# Add httpx as dependency or use asyncio.to_thread
import asyncio
from urllib.request import Request, urlopen

async def discover(self, peer_url: str) -> AgentCard | None:
    """Discover peer agent capabilities via /.well-known/agent.json."""
    well_known = f"{peer_url.rstrip('/')}/.well-known/agent.json"
    try:
        # Run blocking IO in thread pool
        def _fetch():
            with urlopen(well_known, timeout=5) as resp:
                return json.loads(resp.read())

        data = await asyncio.to_thread(_fetch)
        return AgentCard(**data)
    except Exception as exc:
        log_json("WARN", "a2a_discovery_failed",
                 details={"peer": peer_url, "error": str(exc)})
        return None

async def delegate(self, peer_url: str, task: Task) -> dict | None:
    """Delegate a task to a peer agent."""
    task_url = f"{peer_url.rstrip('/')}/a2a/tasks"
    payload = json.dumps(task.to_dict()).encode("utf-8")

    try:
        def _post():
            req = Request(
                task_url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())

        return await asyncio.to_thread(_post)
    except Exception as exc:
        log_json("WARN", "a2a_delegation_failed",
                 details={"peer": peer_url, "error": str(exc)})
        return None
```

### 14. Similarity Threshold Not Used
**File:** `memory/consolidation.py:85`
**Issue:** `similarity_threshold` parameter accepted but never used in merging logic.

**Fix:**
Either implement it or remove it:
```python
# Option 1: Remove if not implementing
def __init__(self, brain, threshold: float = 0.9):
    # Remove: similarity_threshold parameter

# Option 2: Implement similarity-based merging
def _merge_duplicates(self, memories: list[dict]) -> list[dict]:
    """Merge semantically similar memories using threshold."""
    # Use embeddings or text similarity to merge similar memories
    # This would require implementing actual similarity comparison
```

## 🔵 P3 - Code Quality Improvements

### 15. Docker Healthcheck Needs Timeout
**File:** `Dockerfile:44`
**Fix:**
```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; import socket; socket.setdefaulttimeout(2); urllib.request.urlopen('http://localhost:8000/health')" || exit 1
```

### 16. Narrow Exception Handling
**Files:** Multiple (`core/mcp_events.py:113`, `core/experiment_tracker.py:69,84`, etc.)
**Issue:** Broad `except Exception` blocks hide unexpected errors.

**Fix:** Catch specific exceptions:
```python
# Instead of:
except Exception as exc:
    ...

# Use:
except (json.JSONDecodeError, KeyError, OSError) as exc:
    ...
# Let other exceptions propagate
```

### 17. AURA_CONTEXT Truncation
**File:** `core/hooks.py:150`
**Issue:** Context truncated to 10000 chars may lose critical information.

**Fix:**
```python
# Option 1: Increase limit
aura_context = json.dumps(context or {}, default=str)[:50000]

# Option 2: Write to temp file
import tempfile
context_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
json.dump(context or {}, context_file, default=str)
context_file.close()
merged_env["AURA_CONTEXT_FILE"] = context_file.name
# Remember to clean up file after subprocess completes
```

### 18. NBest JSON Parsing Too Permissive
**File:** `core/nbest.py:157`
**Issue:** `re.search(r'{.*}', response, re.DOTALL)` matches unintended structures.

**Fix:**
```python
def _parse_scores(self, response: str, candidates: list[CodeCandidate]) -> bool:
    """Parse JSON scoring response from critic."""
    # Try to find JSON block with scores key
    json_match = re.search(
        r'\{\s*"scores"\s*:\s*\{[^}]*\}[^}]*\}',
        response,
        re.DOTALL
    )
    if not json_match:
        # Fallback to broader match
        json_match = re.search(r'\{.*"scores".*\}', response, re.DOTALL)

    if not json_match:
        return False
    # ... rest of parsing
```

## 📋 Integration Test Requirements (Issue #207)

The following integration tests are needed but blocked by the above bugs:

1. ✅ Can add once fixes 5 & 8 are applied: N-best with critic tournament
2. ✅ Can add once fix 3 is applied: Hook blocking behavior
3. ⚠️ Hook modification test - needs documentation on expected JSON format
4. ⚠️ Confidence routing test - needs phase_result.py review
5. ⚠️ Confidence escalation test - needs phase_result.py review
6. ❌ Blocked by fix 4: Experiment tracker keep/discard
7. ✅ Can add once fix 7 is applied: A2A discovery
8. ✅ Can add once fix 1 is applied: A2A task delegation
9. ✅ Can add once fix 2 is applied: SSE event streaming
10. ❌ Blocked by fix 9: Memory consolidation

## 🎯 Recommended Merge Order

1. **Must fix before merge:** Issues 1, 2, 3, 4, 5, 6
2. **Should fix before merge:** Issues 7, 8, 9, 10, 11, 12, 13
3. **Can fix post-merge:** Issues 14, 15, 16, 17, 18

## 📚 Additional Suggestions for Improvement

### Issue #198 - Merge Queue Bug
Review the status check logic in the merge queue script for the high-severity bug mentioned in the code review.

### Issue #210 - Skill Correlation Learning
This is a good enhancement but should be a separate PR after #206 is stable.

### Issue #209 - JSON-RPC/stdio Transport
Strategic feature but defer until after v0.1.0 release to maintain focus.

### General Code Quality
- Add type hints to all new modules
- Ensure all new code has docstrings
- Add logging to all error paths
- Consider adding retry logic for transient failures
