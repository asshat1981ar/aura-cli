# AURA CLI VS Code Extension

VS Code extension for the AURA autonomous development loop.

## Transport

The extension communicates with the AURA HTTP API server using a JSON-RPC
transport layer over HTTP. Start the server with:

```bash
uvicorn aura_cli.server:app --port 8001
```

## Commands

- **AURA: Run Goal** (`aura.runGoal`) — Submit a goal to the AURA loop
- **AURA: Show Status** (`aura.status`) — Show current goal queue status
