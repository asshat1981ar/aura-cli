"""A2A client for discovering and delegating tasks to peer agents.

Enables AURA to discover other A2A-compatible agents via their Agent Cards
and delegate tasks to them.
"""
import json
import time
import urllib.request
from dataclasses import dataclass

from core.a2a.agent_card import AgentCard
from core.logging_utils import log_json


@dataclass
class PeerAgent:
    """A discovered peer agent."""
    card: AgentCard
    url: str = ""
    last_seen: float = 0.0
    healthy: bool = True


class A2AClient:
    """Client for discovering peer agents and delegating tasks."""

    def __init__(self):
        self.peers: dict[str, PeerAgent] = {}

    async def discover(self, url: str) -> AgentCard | None:
        """Discover a peer agent by fetching its Agent Card."""
        try:
            agent_json_url = f"{url.rstrip('/')}/.well-known/agent.json"
            req = urllib.request.Request(
                agent_json_url,
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:  # nosec B310 — URL from trusted peer config
                data = json.loads(resp.read())
                card = AgentCard.from_dict(data)
                self.peers[url] = PeerAgent(
                    card=card, url=url, last_seen=time.time(),
                )
                log_json("INFO", "a2a_peer_discovered",
                         details={"name": card.name, "url": url,
                                  "capabilities": [c.name for c in card.capabilities]})
                return card
        except Exception as exc:
            log_json("WARN", "a2a_discovery_failed",
                     details={"url": url, "error": str(exc)})
            return None

    async def delegate(self, peer_url: str, capability: str,
                       message: str,
                       metadata: dict | None = None) -> dict | None:
        """Delegate a task to a peer agent."""
        try:
            task_url = f"{peer_url.rstrip('/')}/a2a/tasks"
            payload = json.dumps({
                "capability": capability,
                "message": message,
                "metadata": metadata or {},
            }).encode()
            req = urllib.request.Request(
                task_url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:  # nosec B310
                result = json.loads(resp.read())
                log_json("INFO", "a2a_task_delegated",
                         details={"peer": peer_url, "capability": capability,
                                  "task_id": result.get("id")})
                return result
        except Exception as exc:
            log_json("WARN", "a2a_delegation_failed",
                     details={"peer": peer_url, "error": str(exc)})
            return None

    def find_capable_peer(self, capability: str) -> PeerAgent | None:
        """Find a peer that supports a given capability."""
        for peer in self.peers.values():
            if not peer.healthy:
                continue
            for cap in peer.card.capabilities:
                if cap.name == capability:
                    return peer
        return None

    def list_peers(self) -> list[dict]:
        """List all known peers and their capabilities."""
        return [
            {
                "name": p.card.name,
                "url": p.url,
                "capabilities": [c.name for c in p.card.capabilities],
                "healthy": p.healthy,
                "last_seen": p.last_seen,
            }
            for p in self.peers.values()
        ]
