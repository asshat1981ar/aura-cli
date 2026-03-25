# Real Run Tool Execution Design

## Summary

Replace the fake SSE-based `run` tool in `aura_cli/server.py` with real subprocess execution while keeping the current `POST /execute` contract for `tool_name="run"`.

The initial implementation will support arbitrary shell commands with hard safety limits:

- explicit feature flag gate via `AGENT_API_ENABLE_RUN=1`
- bounded wall-clock timeout
- bounded streamed output size
- fixed working directory
- scrubbed subprocess environment
- deterministic SSE event contract for success, failure, timeout, and truncation

## Goals

- Make the `run` tool actually execute commands instead of returning simulated output.
- Preserve the current shape of the API so existing clients and tests continue to work.
- Bound execution tightly enough that the endpoint is useful without becoming an unrestricted remote shell.
- Keep the implementation small and isolated inside the server module.

## Non-Goals

- No arbitrary client-controlled working directory in the first version.
- No interactive stdin support.
- No PTY emulation.
- No shell allowlist in the first version.
- No redesign of the `ExecuteRequest` model in this pass.

## Interface

The existing request shape remains:

- `tool_name="run"`
- `args[0]` is a shell command string

The response remains an SSE stream with `text/event-stream`.

Events:

- `start`
- `stdout`
- `stderr`
- `exit`
- `error`
- `truncated`

## Execution Model

Implementation will use `asyncio.create_subprocess_shell(...)` because the current API already takes a single command string and the goal is to preserve compatibility.

The subprocess will run:

- from the repository root
- with a reduced environment
- with stdout/stderr read asynchronously and streamed as SSE events

## Safety Limits

Default limits for the first version:

- timeout: 15 seconds
- combined output cap: 64 KiB
- per-chunk emission trimmed to avoid giant SSE frames

Environment policy:

- start from an empty environment or a small safe baseline
- preserve only minimal runtime variables such as `PATH`, `HOME`, `LANG`, `TERM`
- do not forward obvious secret-bearing variables by default

Working directory policy:

- always use repo root / current project root

Termination policy:

- on timeout, terminate then kill if needed
- on output-cap breach, terminate and emit a `truncated` event

## Error Handling

Validation errors:

- if `AGENT_API_ENABLE_RUN != "1"`, return `403`
- if command string is missing or empty, return `400`

Runtime errors:

- process spawn failures emit `error`
- timeout emits `error` with timeout metadata followed by final exit semantics
- output truncation emits `truncated`

## Testing

Update server tests to cover:

- real successful command execution
- stderr streaming
- disabled-run rejection
- missing-args rejection
- timeout behavior
- output truncation behavior

Use cheap deterministic commands such as:

- `python3 -c "print('ok')"`
- `python3 -c "import sys; sys.stderr.write('err\\n')"`
- `python3 -c "import time; time.sleep(...)"` for timeout coverage

## Implementation Notes

- keep `_execute_run(...)` as the public handler
- add a small helper for subprocess streaming and limit accounting
- prefer explicit constants for timeout and output caps
- keep later extension paths open for structured exec mode or configurable limits

## Follow-Up Options

Possible next steps after this pass:

- optional structured `exec` mode using argument arrays instead of shell strings
- per-request timeout/output overrides under strict ceilings
- allowlist/denylist command policy
- audit logging for run-tool invocations
