# n8n Integration

`n8n` is an optional automation bridge for AURA. It is not part of the core CLI runtime and AURA must remain fully usable without it.

## Local setup

1. Copy the example env file:
   - `cp .env.n8n.example .env.n8n`
2. Start the local stack:
   - `docker compose -f docker-compose.n8n.yml --env-file .env.n8n up -d`
3. Open `http://localhost:5678`.

## Intended use

- outbound AURA events for SADD/session automation
- approval and notification workflows
- external service choreography around the CLI runtime

## AURA event contract

The thin adapter in `core/integrations/n8n.py` emits JSON shaped like:

```json
{
  "event_type": "sadd.session.started",
  "source": "aura-cli",
  "payload": {
    "session_id": "abc123",
    "title": "Refactor runtime factory"
  }
}
```

Recommended initial event types:

- `sadd.session.started`
- `sadd.plan.generated`
- `sadd.workstream.ready`
- `sadd.workstream.completed`
- `sadd.session.failed`
- `aura.goal.completed`

## Workflow recommendation

Start with a single inbound webhook workflow that:

1. receives AURA events
2. branches on `event_type`
3. posts notifications or approval requests
4. optionally calls back into AURA or a related automation service

Keep `n8n` optional and sidecar-only. It should not replace AURA’s orchestrator.
