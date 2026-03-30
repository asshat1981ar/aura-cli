"""Server lifecycle management — start, stop, health check, auto-restart.

Extends the _start_background_command pattern from core/capability_manager.py.
"""
from __future__ import annotations

import os
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ServerState:
    """Runtime state for a managed MCP server."""

    name: str
    port: int
    command: List[str]
    health_path: str = "/health"
    host: str = "127.0.0.1"
    pid: Optional[int] = None
    status: str = "stopped"  # stopped, starting, running, unhealthy, failed
    start_time: float = 0.0
    last_health_check: float = 0.0
    consecutive_failures: int = 0
    max_retries: int = 3
    env_overrides: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict:
        return {
            "name": self.name,
            "port": self.port,
            "host": self.host,
            "pid": self.pid,
            "status": self.status,
            "start_time": self.start_time,
            "last_health_check": self.last_health_check,
            "consecutive_failures": self.consecutive_failures,
        }


class ServerLifecycle:
    """Manages lifecycle of MCP servers with health checking and auto-restart."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self._servers: Dict[str, ServerState] = {}

    def register(
        self,
        name: str,
        port: int,
        command: List[str],
        *,
        health_path: str = "/health",
        host: str = "127.0.0.1",
        max_retries: int = 3,
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> ServerState:
        """Register a server for lifecycle management."""
        state = ServerState(
            name=name,
            port=port,
            command=command,
            health_path=health_path,
            host=host,
            max_retries=max_retries,
            env_overrides=env_overrides or {},
        )
        # Check if already running
        if _listening(host, port):
            state.status = "running"
        self._servers[name] = state
        return state

    def start(self, name: str) -> dict:
        """Start a registered server.

        Returns:
            Status dict with action result.
        """
        state = self._servers.get(name)
        if state is None:
            return {"action": "start", "name": name, "status": "not_registered"}

        if _listening(state.host, state.port):
            state.status = "running"
            return {
                "action": "start",
                "name": name,
                "status": "already_running",
                "port": state.port,
            }

        env = os.environ.copy()
        env.update(state.env_overrides)

        try:
            proc = subprocess.Popen(
                state.command,
                cwd=str(self.project_root),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            state.pid = proc.pid
            state.status = "starting"
            state.start_time = time.time()
            state.consecutive_failures = 0

            return {
                "action": "start",
                "name": name,
                "status": "started",
                "pid": proc.pid,
                "port": state.port,
            }
        except (OSError, subprocess.SubprocessError) as exc:
            state.status = "failed"
            return {
                "action": "start",
                "name": name,
                "status": "failed",
                "error": str(exc),
            }

    def stop(self, name: str) -> dict:
        """Stop a registered server by sending SIGTERM."""
        state = self._servers.get(name)
        if state is None:
            return {"action": "stop", "name": name, "status": "not_registered"}

        if state.pid is None:
            state.status = "stopped"
            return {"action": "stop", "name": name, "status": "no_pid"}

        try:
            os.kill(state.pid, 15)  # SIGTERM
            state.status = "stopped"
            state.pid = None
            return {"action": "stop", "name": name, "status": "stopped"}
        except ProcessLookupError:
            state.status = "stopped"
            state.pid = None
            return {"action": "stop", "name": name, "status": "already_stopped"}
        except OSError as exc:
            return {"action": "stop", "name": name, "status": "error", "error": str(exc)}

    def health_check(self, name: str) -> dict:
        """Check health of a single server.

        Returns:
            Dict with health status.
        """
        state = self._servers.get(name)
        if state is None:
            return {"name": name, "status": "not_registered"}

        is_up = _listening(state.host, state.port)
        state.last_health_check = time.time()

        if is_up:
            state.status = "running"
            state.consecutive_failures = 0
        else:
            state.consecutive_failures += 1
            state.status = "unhealthy"

        return {
            "name": name,
            "status": state.status,
            "port": state.port,
            "consecutive_failures": state.consecutive_failures,
        }

    def health_check_all(self) -> Dict[str, dict]:
        """Check health of all registered servers."""
        return {name: self.health_check(name) for name in self._servers}

    def restart_unhealthy(self, max_retries: int = 3) -> List[dict]:
        """Attempt to restart any unhealthy servers within their retry limit.

        Returns:
            List of restart attempt results.
        """
        results = []
        for name, state in self._servers.items():
            if state.status == "unhealthy" and state.consecutive_failures <= max_retries:
                self.stop(name)
                result = self.start(name)
                result["action"] = "restart"
                results.append(result)
            elif state.status == "unhealthy" and state.consecutive_failures > max_retries:
                state.status = "failed"
                results.append({
                    "action": "restart",
                    "name": name,
                    "status": "max_retries_exceeded",
                    "consecutive_failures": state.consecutive_failures,
                })
        return results

    def list_servers(self) -> List[dict]:
        """List all managed servers with their current state."""
        return [state.as_dict() for state in self._servers.values()]

    def get_server(self, name: str) -> Optional[ServerState]:
        """Get server state by name."""
        return self._servers.get(name)


def _listening(host: str, port: int) -> bool:
    """Check if a TCP port is listening."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.3)
            return sock.connect_ex((host, port)) == 0
    except OSError:
        return False
