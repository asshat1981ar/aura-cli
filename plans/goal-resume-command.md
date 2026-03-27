# Goal Resume Command — Implementation Plan

**Story:** [AF-STORY-0011](.aura_forge/backlog/ready/AF-STORY-0011.yaml)  
**Sprint:** S001_queue_ready_intelligence  
**Status:** Ready for implementation

---

## Overview

`goal resume` recovers an AURA goal that was interrupted mid-execution — for example, by a `SIGKILL`,
OOM kill, or machine power loss — and re-queues it at the front of the goal queue so it can be
retried on the next run.

### Why this matters

`GoalQueue.next()` in `core/goal_queue.py` pops the goal with `popleft()` and persists the queue
to disk before execution begins. From that moment until `goal_archive.record()` is called at the
end of `run_goals_loop()`, the goal exists only in a local Python variable. Any process kill in that
window discards the goal silently — no archive entry, no log event, no way to recover it.

The SADD subsystem already has resume via SQLite checkpoints (`core/sadd/session_store.py`), but
regular goals have no equivalent. This feature closes that gap with a lightweight JSON file and a
new `goal resume` subcommand.

---

## Architecture

### InFlightTracker

`core/in_flight_tracker.py` is a thin wrapper around `memory/in_flight_goal.json`. Its only
responsibility is to record what goal is currently executing, so that a later `goal resume` invocation
can find it.

**In-flight record shape:**

```json
{
  "goal": "Fix the auth bug",
  "started_at": "2026-03-26T18:00:00Z",
  "cycle_limit": 5,
  "phase": "plan"
}
```

| Field | Type | Purpose |
|-------|------|---------|
| `goal` | string | The goal title exactly as dequeued |
| `started_at` | ISO-8601 string | When execution began — shown to the operator in `goal resume` |
| `cycle_limit` | int | The cycle limit at the time of execution, preserved for context |
| `phase` | string | The last known phase (currently always "ingest" at write time; reserved for future phase-aware resume) |

**Write strategy:** The file is written atomically using a `.tmp` suffix and `os.replace()` to prevent
partial-write corruption in the event of a kill during the write itself.

**Clear strategy:** `clear()` uses `Path.unlink(missing_ok=True)` so it is safe to call in a `finally`
block even when the file was never created (e.g., dry-run mode).

### Data flow

```
goal_queue.next()
    │
    ▼
tracker.write(goal, cycle_limit)          ← IN-FLIGHT FILE CREATED
    │
    ▼
try:
    orchestrator.run_loop(goal, ...)       ← EXECUTION (crash window)
    goal_archive.record(goal, score)
finally:
    tracker.clear()                        ← IN-FLIGHT FILE DELETED
```

### Recovery flow

```
$ python3 main.py goal resume

  Reads memory/in_flight_goal.json
      │
      ├── File absent → print "No interrupted goal found." → exit 0
      │
      └── File present → print goal title + started_at
                       → tracker.clear()
                       → goal_queue.prepend_batch([goal])
                       → print "Goal re-queued at front of queue."
                       → if --run: invoke goal run loop
                       → exit 0
```

---

## Files to Create

### `core/in_flight_tracker.py`

```python
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


_DEFAULT_PATH = Path("memory") / "in_flight_goal.json"


class InFlightTracker:
    """Tracks the goal currently executing to enable crash recovery."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _DEFAULT_PATH

    def write(self, goal: str, cycle_limit: int, phase: str = "ingest") -> None:
        """Record goal as in-flight. Atomic write to prevent partial-file corruption."""
        data = {
            "goal": goal,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "cycle_limit": cycle_limit,
            "phase": phase,
        }
        tmp = self._path.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, self._path)

    def read(self) -> dict | None:
        """Return in-flight record, or None if no interrupted goal exists."""
        if not self._path.exists():
            return None
        try:
            return json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def clear(self) -> None:
        """Remove in-flight record. Safe to call even if file does not exist."""
        self._path.unlink(missing_ok=True)

    def exists(self) -> bool:
        """Return True if an in-flight record is present."""
        return self._path.exists()
```

### `tests/test_in_flight_tracker.py`

Unit tests covering all four public methods plus edge cases (absent file, corrupted JSON, atomic write).
Tests use a `tmp_path` fixture (pytest) so they never touch `memory/in_flight_goal.json`.

---

## Files to Modify

### 1. `core/task_handler.py`

**Where:** In `run_goals_loop()`, immediately after `goal = goal_queue.next()` and before
`orchestrator.run_loop()`.

**Change:**

```python
from core.in_flight_tracker import InFlightTracker

tracker = InFlightTracker()

# ... existing while-loop ...
goal = goal_queue.next()
if goal is None:
    break

tracker.write(goal, cycle_limit)
try:
    result = orchestrator.run_loop(goal, max_cycles=cycle_limit, dry_run=dry_run)
    # ... existing result handling ...
    goal_archive.record(goal, final_score)
finally:
    tracker.clear()
```

**Key invariant:** `tracker.clear()` is in a `finally` block, so it runs on PASS, MAX_CYCLES,
ERROR, and even unhandled exceptions. Only a hard process kill (SIGKILL, power loss) bypasses it.

### 2. `aura_cli/options.py`

**Where:** After the `("goal", "once")` `CommandSpec` and before the `("workflow",)` spec.

**Add CommandSpec:**

```python
CommandSpec(
    path=("goal", "resume"),
    summary="Resume an interrupted goal",
    description=(
        "Re-queue a goal that was interrupted mid-execution due to a crash or process kill. "
        "Reads memory/in_flight_goal.json written by the goal run loop. "
        "Use --run to immediately execute the re-queued goal."
    ),
    examples=(
        "python3 main.py goal resume",
        "python3 main.py goal resume --run",
    ),
),
```

**Add CLIActionSpec** (after the `goal_run` entry):

```python
CLIActionSpec("goal_resume", True, ("goal", "resume")),
```

`requires_runtime=True` because `--run` delegates to the goal run loop which needs runtime objects.

### 3. `aura_cli/dispatch.py`

**Add handler** (near `_handle_sadd_resume_dispatch`, after the other goal handlers):

```python
def _handle_goal_resume_dispatch(ctx: DispatchContext) -> int:
    from core.in_flight_tracker import InFlightTracker

    tracker = InFlightTracker()
    entry = tracker.read()

    if not entry:
        print("No interrupted goal found.", file=sys.stderr)
        return 0

    goal = entry.get("goal", "<unknown>")
    started_at = entry.get("started_at", "unknown")
    cycle_limit = entry.get("cycle_limit", "unknown")

    print(f"Interrupted goal found: {goal!r}")
    print(f"  Started:     {started_at}")
    print(f"  Cycle limit: {cycle_limit}")
    print(f"  Last phase:  {entry.get('phase', 'unknown')}")

    # Clear before re-queuing to prevent double-entry on concurrent resume calls.
    tracker.clear()
    ctx.runtime["goal_queue"].prepend_batch([goal])
    print("Goal re-queued at front of queue.")

    if getattr(ctx.args, "run", False):
        return _handle_goal_run_dispatch(ctx)

    return 0
```

**Add to `COMMAND_DISPATCH_REGISTRY`** (after `goal_run` entry):

```python
"goal_resume": _dispatch_rule("goal_resume", _handle_goal_resume_dispatch),
```

### 4. `aura_cli/cli_main.py`

If `dispatch.py` handlers are imported explicitly in `cli_main.py`, add:

```python
from aura_cli.dispatch import _handle_goal_resume_dispatch  # noqa: F401
```

Verify by checking the existing import block — if `_handle_sadd_resume_dispatch` is imported
there, follow the same pattern.

### 5. `run_aura.sh`

**In the help/usage block**, add `resume` alongside the other aliases:

```bash
  resume             Forward to `python3 main.py goal resume`
```

**In the case block**, add after the `status)` case:

```bash
  resume)
    shift
    exec_main goal resume "$@"
    ;;
```

### 6. `tests/snapshots/` — snapshot updates

After implementing changes, run:

```bash
python3 -m pytest tests/test_cli_help_snapshots.py tests/test_cli_error_snapshots.py -v
```

If snapshots are stale (expected after adding a new subcommand), update them:

```bash
python3 update_snapshots.py
```

or follow the update workflow documented in `tests/snapshots/README.md` if present.

The following snapshot files are likely affected:
- `tests/snapshots/goal_help.json` — now includes `goal resume`
- `tests/snapshots/cli_action_specs.json` — now includes `goal_resume`
- `tests/snapshots/command_specs.json` — now includes `("goal", "resume")`

### 7. `docs/CLI_REFERENCE.md`

Regenerate after all code changes:

```bash
python3 scripts/generate_cli_reference.py
```

Verify it is current:

```bash
python3 scripts/generate_cli_reference.py --check
```

---

## Implementation Steps (ordered)

1. **Create `core/in_flight_tracker.py`** — implement `InFlightTracker` with `write`, `read`, `clear`, `exists`.

2. **Write `tests/test_in_flight_tracker.py`** — cover:
   - `write()` creates file with expected keys
   - `read()` returns `None` when file absent
   - `read()` returns dict when file present
   - `clear()` removes file; subsequent `read()` returns `None`
   - `exists()` returns `True`/`False` correctly
   - `clear()` is safe to call when file does not exist (no exception)
   - `read()` returns `None` if file contains invalid JSON (corruption tolerance)
   - Atomic write: `.tmp` file replaced, not left behind

3. **Integrate tracker into `core/task_handler.py`** — wrap dequeue and archive with `tracker.write()` / `tracker.clear()` in `try/finally`.

4. **Add `CommandSpec` and `CLIActionSpec` to `aura_cli/options.py`** — follow the exact formatting of adjacent goal subcommand specs.

5. **Implement `_handle_goal_resume_dispatch` in `aura_cli/dispatch.py`** — follow `_handle_sadd_resume_dispatch` pattern; add entry to `COMMAND_DISPATCH_REGISTRY`.

6. **Update imports in `aura_cli/cli_main.py`** — add `_handle_goal_resume_dispatch` if the file imports dispatch handlers explicitly.

7. **Add `resume` alias to `run_aura.sh`** — in both the usage block and the case statement.

8. **Run targeted tests:**
   ```bash
   python3 -m pytest tests/test_in_flight_tracker.py -v
   ```

9. **Run snapshot tests and update if needed:**
   ```bash
   python3 -m pytest tests/test_cli_help_snapshots.py tests/test_cli_error_snapshots.py -v
   python3 update_snapshots.py   # if snapshots are stale
   ```

10. **Regenerate CLI reference:**
    ```bash
    python3 scripts/generate_cli_reference.py
    python3 scripts/generate_cli_reference.py --check
    ```

11. **Run full test suite:**
    ```bash
    python3 -m pytest
    ```

---

## Dispatch Handler — Full Pseudocode

```python
def _handle_goal_resume_dispatch(ctx: DispatchContext) -> int:
    """Re-queue an interrupted goal from memory/in_flight_goal.json."""
    from core.in_flight_tracker import InFlightTracker

    tracker = InFlightTracker()
    entry = tracker.read()

    if not entry:
        # No in-flight file — nothing to resume. Exit cleanly.
        print("No interrupted goal found.", file=sys.stderr)
        return 0

    goal = entry.get("goal", "<unknown>")
    started_at = entry.get("started_at", "unknown")
    cycle_limit = entry.get("cycle_limit", "unknown")
    phase = entry.get("phase", "unknown")

    print(f"Interrupted goal found: {goal!r}")
    print(f"  Started:     {started_at}")
    print(f"  Cycle limit: {cycle_limit}")
    print(f"  Last phase:  {phase}")

    # Clear first to prevent double-entry if resume is called concurrently.
    tracker.clear()
    ctx.runtime["goal_queue"].prepend_batch([goal])
    print("Goal re-queued at front of queue.")

    if getattr(ctx.args, "run", False):
        # Delegate to the standard goal run loop.
        return _handle_goal_run_dispatch(ctx)

    return 0
```

---

## Forge Story Link

This plan describes **how** to implement AF-STORY-0011. The story file describes **why** and **what**:

- **Why:** [AF-STORY-0011 — inbox](.aura_forge/backlog/inbox/AF-STORY-0011.yaml)  
- **What + design passes:** [AF-STORY-0011 — ready](.aura_forge/backlog/ready/AF-STORY-0011.yaml)  
- **How:** this file (`plans/goal-resume-command.md`)

The separation keeps the Forge backlog focused on shaping decisions and design rationale, while the
plans file focuses on implementation detail, pseudocode, and ordered steps.

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| In-flight file left on disk after hard crash (power loss) | Medium — requires SIGKILL or power loss | Explicit `goal resume` check; file is never auto-consumed |
| Race condition: two AURA instances write the same in-flight file simultaneously | Low — single-user CLI | Document as known limitation; out of scope for v1 |
| `goal resume` run when goal already in queue → double entry | Low — requires unusual manual state | `tracker.clear()` called before `prepend_batch()`, preventing a second clear; queue dedup is a v2 concern |
| Stale in-flight file if AURA is upgraded between crash and resume | Low | Use `.get()` for all field access; unknown fields are ignored gracefully |
| `--run` flag not wired in argparse → `getattr(ctx.args, "run", False)` silently returns False | Medium if forgotten | Verify argparse `--run` flag exists for `goal resume` subparser during implementation step 5 |

---

## Acceptance Verification

After implementation, verify each acceptance criterion:

```bash
# AC1 + AC2: in-flight file created and cleared
python3 main.py goal add "Test interrupted goal"
# Kill during execution with SIGKILL
ls memory/in_flight_goal.json          # Should exist
python3 main.py goal run               # Run to completion normally
ls memory/in_flight_goal.json 2>&1     # Should be absent

# AC3: resume shows goal and re-queues
# (with in-flight file present from a crash)
python3 main.py goal resume

# AC4: no-op when no in-flight file
rm -f memory/in_flight_goal.json
python3 main.py goal resume            # Should print "No interrupted goal found." and exit 0

# AC5: resume --run re-queues and runs
python3 main.py goal resume --run

# AC6: CLI docs and snapshots
python3 scripts/generate_cli_reference.py --check
python3 -m pytest tests/test_cli_help_snapshots.py -v
```
