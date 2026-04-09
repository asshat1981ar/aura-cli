# Goal Resume Feature

The Goal Resume feature enables recovery of goals that were interrupted mid-execution due to crashes, power loss, or SIGKILL.

## Overview

When AURA starts executing a goal, it writes an "in-flight" record to disk. If the process terminates unexpectedly before the goal completes, this record persists. The `goal resume` command allows you to recover and re-queue these interrupted goals.

## How It Works

1. **Tracking**: When a goal is dequeued for execution, an in-flight record is written to `memory/in_flight_goal.json`
2. **Cleanup**: When the goal completes (success or failure), the record is cleared
3. **Recovery**: If a crash occurs, the record survives and can be used to resume

## CLI Usage

### Check for Interrupted Goals

```bash
python3 main.py goal resume
```

Output if goal exists:
```
Found interrupted goal: "Implement user authentication"
  Started:    2024-01-15T10:30:00+00:00
  Last phase: ingest
  Cycle limit: 10
Re-queued at front of queue.
Run 'goal run' (or --run) to execute.
```

Output if no goal:
```
No interrupted goal found. Nothing to resume.
```

### Resume and Run Immediately

```bash
python3 main.py goal resume --run
```

This re-queues the goal and immediately starts execution.

### With Goal Run

```bash
# Check and resume before running new goals
python3 main.py goal run --resume
```

## Web UI

The Dashboard displays an alert when an interrupted goal is detected:

![Goal Resume Card](docs/assets/goal-resume-card.png)

Features:
- Visual indicator of interrupted goal
- Elapsed time since interruption
- One-click re-queue
- Resume & Run button

## API Endpoints

### Get In-Flight Goal

```http
GET /api/goals/in-flight
```

Response:
```json
{
  "exists": true,
  "summary": {
    "goal": "Implement user authentication",
    "started_at": "2024-01-15T10:30:00+00:00",
    "started_at_formatted": "2024-01-15 10:30:00 UTC",
    "elapsed_seconds": 3600,
    "elapsed_formatted": "1h 0m",
    "cycle_limit": 10
  }
}
```

### Resume Goal

```http
POST /api/goals/resume
Content-Type: application/json

{
  "run": false
}
```

Response:
```json
{
  "status": "resumed",
  "message": "Goal re-queued at front: Implement user authentication...",
  "goal": "Implement user authentication"
}
```

## Safety Features

### Atomic Writes

The in-flight record is written atomically (to a temp file, then renamed) to prevent corruption during crashes.

### Double-Execution Prevention

Before re-queuing, the tracker clears the in-flight record. This ensures the goal enters the queue exactly once, even if `resume` is called multiple times.

### Process Isolation

The tracker records the process ID (PID) that started the goal. This helps detect if the goal was started by a different process.

## Technical Details

### File Location

```
memory/in_flight_goal.json
```

### Record Format

```json
{
  "goal": "Implement user authentication",
  "started_at": "2024-01-15T10:30:00+00:00",
  "cycle_limit": 10,
  "phase": "ingest"
}
```

### Implementation

The feature is implemented in:
- `core/in_flight_tracker.py` - Core tracking logic
- `core/task_handler.py` - Integration with goal execution
- `aura_cli/dispatch.py` - CLI command handling
- `aura_cli/api_server.py` - API endpoints
- `web-ui/src/components/GoalResumeCard.tsx` - Web UI

## Limitations

1. **Hard Crashes**: If the system loses power during the write operation, the file may be corrupted (handled gracefully by ignoring invalid JSON)

2. **No Progress Checkpointing**: The feature tracks which goal was running but not how far it progressed. The goal restarts from the beginning.

3. **Single Goal**: Only one goal can be in-flight at a time per AURA instance.

4. **No Multi-Instance Coordination**: If multiple AURA processes run simultaneously, they don't coordinate on the in-flight file.

## Testing

Run the in-flight tracker tests:

```bash
python3 -m pytest tests/test_in_flight_tracker.py -v
```

Test scenarios covered:
- Writing and reading records
- Atomic write operations
- Clearing records
- Handling corrupted files
- Edge cases (empty goals, missing files)

## Troubleshooting

### Goal Not Resuming

Check if the file exists:
```bash
cat memory/in_flight_goal.json
```

### Stale Record

If a record is left behind from a previous run, manually clear it:
```bash
rm memory/in_flight_goal.json
```

### Permission Errors

Ensure AURA has write access to the `memory/` directory:
```bash
ls -la memory/
```

## Future Enhancements

- Phase-aware resume (continue from last completed phase)
- Multiple in-flight goals support
- Distributed coordination for multi-instance setups
- Automatic resume on startup option
