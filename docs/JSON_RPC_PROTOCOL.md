# AURA JSON-RPC 2.0 Protocol

This document describes the JSON-RPC 2.0 over stdio transport for AURA CLI.
Third-party integrations (e.g. VS Code extensions, language server clients)
can launch AURA as a subprocess and communicate with it over stdin/stdout
using this protocol.

---

## Starting the Transport

```bash
python main.py --stdio-rpc
# or, with a custom project root:
python main.py --stdio-rpc --project-root /path/to/project
```

The process reads newline-delimited JSON-RPC 2.0 request objects from **stdin**
and writes newline-delimited JSON-RPC 2.0 response objects to **stdout**.

> **Note:** Each request and each response is a single line (no pretty printing).
> Lines may be up to any length; use a line-buffered reader.

---

## Protocol Specification

AURA implements a subset of [JSON-RPC 2.0](https://www.jsonrpc.org/specification).

### Request format

```json
{
  "jsonrpc": "2.0",
  "method": "goal.add",
  "params": {
    "goal": "Add unit tests for the file_tools module"
  },
  "id": 1
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `jsonrpc` | `"2.0"` | ✅ | Must be the string `"2.0"` |
| `method` | string | ✅ | JSON-RPC method name (see [Methods](#methods)) |
| `params` | object | ❌ | Method parameters (see per-method docs below) |
| `id` | string \| number \| null | ❌ | Request ID echoed in the response |

### Response format (success)

```json
{
  "jsonrpc": "2.0",
  "result": {
    "exit_code": 0,
    "output": { "status": "ok", "added": "Add unit tests for the file_tools module" }
  },
  "id": 1
}
```

### Response format (error)

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32601,
    "message": "Method not found: \"goal.unknown\""
  },
  "id": 1
}
```

### Standard error codes

| Code | Name | Meaning |
|------|------|---------|
| `-32700` | Parse error | The input could not be parsed as JSON |
| `-32600` | Invalid request | The JSON object is not a valid request |
| `-32601` | Method not found | The method name is not registered |
| `-32602` | Invalid params | Invalid or missing method parameters |
| `-32603` | Internal error | An internal server error occurred |

---

## Methods

All `result` objects contain at minimum:

```json
{
  "exit_code": 0,
  "output": <any>
}
```

`output` is the JSON payload returned by the underlying CLI command.
`stderr` is included only when there is stderr output.

### `goal.add`

Add a goal to the AURA goal queue.

**Params:**

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `goal` | string | ✅ | Goal description |

**Example:**

```json
{"jsonrpc":"2.0","method":"goal.add","params":{"goal":"Refactor core/model_adapter.py"},"id":1}
```

---

### `goal.run`

Run the AURA goal loop (processes queued goals).

**Params:**

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `max_cycles` | integer | ❌ | Maximum number of loop cycles |
| `dry_run` | boolean | ❌ | Perform a dry run (no file writes) |

**Example:**

```json
{"jsonrpc":"2.0","method":"goal.run","params":{"max_cycles":3},"id":2}
```

---

### `goal.once`

Run AURA for a single goal without queueing.

**Params:**

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `goal` | string | ✅ | Goal description |

**Example:**

```json
{"jsonrpc":"2.0","method":"goal.once","params":{"goal":"Write tests for transport.py"},"id":3}
```

---

### `goal.status`

Get the current goal queue status.

**Params:** none

**Example:**

```json
{"jsonrpc":"2.0","method":"goal.status","params":{},"id":4}
```

---

### `queue.list`

List all goals currently in the queue.

**Params:** none

---

### `queue.clear`

Clear the goal queue.

**Params:** none

---

### `doctor`

Run the AURA system diagnostics doctor.

**Params:** none

---

### `bootstrap`

Bootstrap AURA configuration.

**Params:** none

---

### `config.show`

Show the current AURA runtime configuration.

**Params:** none

---

### `diag`

Show diagnostics information.

**Params:** none

---

### `logs`

Show recent AURA logs.

**Params:** none

---

### `memory.search`

Search semantic memory.

**Params:**

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `query` | string | ✅ | Search query |

---

### `memory.reindex`

Trigger a memory re-index.

**Params:** none

---

### `metrics`

Show runtime metrics.

**Params:** none

---

### `workflow.run`

Run an AURA workflow.

**Params:**

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `goal` | string | ✅ | Workflow goal description |

---

### `scaffold`

Scaffold a new project or component.

**Params:** none

---

### `evolve`

Run the AURA self-improvement (RSI) evolution loop.

**Params:** none

---

### `mcp.tools`

List available MCP tools.

**Params:** none

---

### `mcp.call`

Call an MCP tool.

**Params:**

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `name` | string | ✅ | Tool name |
| `input` | object | ❌ | Tool input payload |

---

### `help`

Show CLI help text.

**Params:** none

---

## Integration Example (Python)

```python
import subprocess
import json

proc = subprocess.Popen(
    ["python", "main.py", "--stdio-rpc"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
    bufsize=1,  # line-buffered
)

def rpc(method, params=None, req_id=1):
    request = {"jsonrpc": "2.0", "method": method, "id": req_id}
    if params:
        request["params"] = params
    proc.stdin.write(json.dumps(request) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    return json.loads(line)

# Add a goal
result = rpc("goal.add", {"goal": "Write tests for transport module"})
print(result)

# Check status
status = rpc("goal.status", req_id=2)
print(status)

proc.stdin.close()
proc.wait()
```

---

## Integration Example (Node.js)

```javascript
const { spawn } = require("child_process");
const readline = require("readline");

const proc = spawn("python", ["main.py", "--stdio-rpc"]);
const rl = readline.createInterface({ input: proc.stdout });

const pending = new Map();
let nextId = 1;

rl.on("line", (line) => {
  const response = JSON.parse(line);
  const resolve = pending.get(response.id);
  if (resolve) {
    resolve(response);
    pending.delete(response.id);
  }
});

function rpc(method, params = {}) {
  return new Promise((resolve) => {
    const id = nextId++;
    pending.set(id, resolve);
    const request = JSON.stringify({ jsonrpc: "2.0", method, params, id }) + "\n";
    proc.stdin.write(request);
  });
}

// Usage
(async () => {
  const result = await rpc("goal.add", { goal: "Add CI integration" });
  console.log(result);
  proc.stdin.end();
})();
```

---

## Limitations

- **Batch requests** (arrays) are not supported; send one request per line.
- Long-running operations (e.g. `goal.run`) block until completion; consider
  a timeout on the client side.
- The `interactive` action is not exposed via the transport (it requires a TTY).
