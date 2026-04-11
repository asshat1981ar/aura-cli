"""Sub-Agent Integration Layer — Wires Phase 2 sub-agents into the orchestrator.

This module provides the bridge between standalone Phase 2 sub-agents
(IOTA, KAPPA, NU, PI, RHO, SIGMA, TAU) and the main LoopOrchestrator.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Protocol

from core.logging_utils import log_json

# Import Phase 2 sub-agents
try:
    from aura.error_resolution import ErrorResolutionEngine

    _IOTA_AVAILABLE = True
except ImportError:
    _IOTA_AVAILABLE = False

try:
    from aura.recording import WorkflowRecorder, WorkflowReplayer  # noqa: F401

    _KAPPA_AVAILABLE = True
except ImportError:
    _KAPPA_AVAILABLE = False

try:
    from aura.offline import OfflineExecutor, ConnectivityMonitor  # noqa: F401

    _NU_AVAILABLE = True
except ImportError:
    _NU_AVAILABLE = False

try:
    from aura.encryption import EncryptedConfigManager  # noqa: F401

    _PI_AVAILABLE = True
except ImportError:
    _PI_AVAILABLE = False

try:
    from aura.health import HealthMonitor, SystemHealthChecker  # noqa: F401

    _RHO_AVAILABLE = True
except ImportError:
    _RHO_AVAILABLE = False

try:
    from aura.security import SecurityAuditor, SecretScanner  # noqa: F401

    _SIGMA_AVAILABLE = True
except ImportError:
    _SIGMA_AVAILABLE = False

try:
    from aura.scheduler import TaskScheduler, CronEngine  # noqa: F401

    _TAU_AVAILABLE = True
except ImportError:
    _TAU_AVAILABLE = False


class SubAgent(Protocol):
    """Protocol for sub-agent integration."""

    name: str

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]: ...


class SubAgentRegistry:
    """Registry for Phase 2 sub-agents.

    Provides centralized access to all sub-agents with lazy initialization
    and health checking.

    Example:
        >>> registry = SubAgentRegistry()
        >>> result = registry.iota.resolve_error(error_message="...", context={})
        >>> workflow = registry.kappa.record_workflow("deploy", steps)
    """

    def __init__(self):
        self._agents: Dict[str, Any] = {}
        self._availability: Dict[str, bool] = {
            "iota": _IOTA_AVAILABLE,
            "kappa": _KAPPA_AVAILABLE,
            "nu": _NU_AVAILABLE,
            "pi": _PI_AVAILABLE,
            "rho": _RHO_AVAILABLE,
            "sigma": _SIGMA_AVAILABLE,
            "tau": _TAU_AVAILABLE,
        }

    # ------------------------------------------------------------------
    # IOTA — AI Error Resolution
    # ------------------------------------------------------------------

    @property
    def iota(self) -> Optional[Any]:
        """IOTA: AI-powered error resolution engine."""
        if not _IOTA_AVAILABLE:
            return None
        if "iota" not in self._agents:
            self._agents["iota"] = ErrorResolutionEngine()
        return self._agents["iota"]

    def resolve_error(
        self,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        """Use IOTA to resolve an error.

        Args:
            error_message: The error to resolve
            context: Additional context (stack trace, logs, etc.)
            timeout_seconds: Maximum time to wait for resolution

        Returns:
            Resolution result with suggested fix
        """
        if not self.iota:
            return {"resolved": False, "reason": "IOTA not available"}

        start = time.time()
        try:
            result = self.iota.resolve(
                error=error_message,
                context=context or {},
                timeout=timeout_seconds,
            )
            log_json(
                "INFO",
                "iota_resolution_complete",
                details={
                    "duration_ms": (time.time() - start) * 1000,
                    "resolved": result.get("resolved", False),
                },
            )
            return result
        except Exception as e:
            log_json("ERROR", "iota_resolution_failed", details={"error": str(e)})
            return {"resolved": False, "reason": str(e)}

    # ------------------------------------------------------------------
    # KAPPA — Recording & Replay
    # ------------------------------------------------------------------

    @property
    def kappa(self) -> Optional[Any]:
        """KAPPA: Workflow recording and replay system."""
        if not _KAPPA_AVAILABLE:
            return None
        if "kappa" not in self._agents:
            from aura.recording import WorkflowRecorder

            self._agents["kappa"] = WorkflowRecorder()
        return self._agents["kappa"]

    def record_workflow(
        self,
        workflow_id: str,
        steps: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Record a workflow for later replay.

        Args:
            workflow_id: Unique identifier for the workflow
            steps: List of workflow steps

        Returns:
            Recording result
        """
        if not self.kappa:
            return {"recorded": False, "reason": "KAPPA not available"}

        try:
            recording = self.kappa.record(workflow_id=workflow_id, steps=steps)
            log_json("INFO", "kappa_workflow_recorded", details={"workflow_id": workflow_id})
            return {"recorded": True, "recording": recording}
        except Exception as e:
            log_json("ERROR", "kappa_recording_failed", details={"error": str(e)})
            return {"recorded": False, "reason": str(e)}

    def replay_workflow(
        self,
        workflow_id: str,
        variables: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Replay a recorded workflow.

        Args:
            workflow_id: Workflow to replay
            variables: Variables to substitute during replay

        Returns:
            Replay result
        """
        if not _KAPPA_AVAILABLE:
            return {"replayed": False, "reason": "KAPPA not available"}

        try:
            from aura.recording import WorkflowReplayer

            replayer = WorkflowReplayer()
            result = replayer.replay(workflow_id=workflow_id, variables=variables or {})
            log_json(
                "INFO",
                "kappa_workflow_replayed",
                details={
                    "workflow_id": workflow_id,
                    "success": result.get("success", False),
                },
            )
            return result
        except Exception as e:
            log_json("ERROR", "kappa_replay_failed", details={"error": str(e)})
            return {"replayed": False, "reason": str(e)}

    # ------------------------------------------------------------------
    # NU — Offline Mode
    # ------------------------------------------------------------------

    @property
    def nu(self) -> Optional[Any]:
        """NU: Offline mode and connectivity management."""
        if not _NU_AVAILABLE:
            return None
        if "nu" not in self._agents:
            from aura.offline import ConnectivityMonitor

            self._agents["nu"] = ConnectivityMonitor()
        return self._agents["nu"]

    def check_connectivity(self) -> Dict[str, Any]:
        """Check system connectivity status.

        Returns:
            Connectivity status with recommendations
        """
        if not self.nu:
            return {"online": True, "mode": "unknown", "reason": "NU not available"}

        try:
            status = self.nu.check_connectivity()
            return status
        except Exception as e:
            log_json("ERROR", "nu_connectivity_check_failed", details={"error": str(e)})
            return {"online": True, "mode": "error", "reason": str(e)}

    def execute_offline_aware(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute an operation with offline awareness.

        Args:
            operation: Function to execute
            *args, **kwargs: Arguments for the operation

        Returns:
            Operation result, or queued for later if offline
        """
        if not _NU_AVAILABLE:
            # No NU available, just execute directly
            return operation(*args, **kwargs)

        try:
            from aura.offline import OfflineExecutor

            executor = OfflineExecutor()
            return executor.execute(operation, *args, **kwargs)
        except Exception as e:
            log_json("ERROR", "nu_offline_execution_failed", details={"error": str(e)})
            raise

    # ------------------------------------------------------------------
    # PI — Config Encryption
    # ------------------------------------------------------------------

    @property
    def pi(self) -> Optional[Any]:
        """PI: Secure configuration encryption."""
        if not _PI_AVAILABLE:
            return None
        if "pi" not in self._agents:
            from aura.encryption import EncryptedConfigManager

            self._agents["pi"] = EncryptedConfigManager()
        return self._agents["pi"]

    def load_encrypted_config(
        self,
        config_path: str,
        key_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load an encrypted configuration file.

        Args:
            config_path: Path to encrypted config
            key_id: Optional key identifier

        Returns:
            Decrypted configuration
        """
        if not self.pi:
            # Fall back to regular config loading
            import json

            with open(config_path) as f:
                return json.load(f)

        try:
            config = self.pi.load(config_path, key_id=key_id)
            log_json("INFO", "pi_config_loaded", details={"path": config_path})
            return config
        except Exception as e:
            log_json("ERROR", "pi_config_load_failed", details={"error": str(e)})
            raise

    def save_encrypted_config(
        self,
        config_path: str,
        config: Dict[str, Any],
        key_id: Optional[str] = None,
    ) -> bool:
        """Save configuration with encryption.

        Args:
            config_path: Path to save config
            config: Configuration to encrypt and save
            key_id: Optional key identifier

        Returns:
            True if saved successfully
        """
        if not self.pi:
            import json

            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            return True

        try:
            self.pi.save(config_path, config, key_id=key_id)
            log_json("INFO", "pi_config_saved", details={"path": config_path})
            return True
        except Exception as e:
            log_json("ERROR", "pi_config_save_failed", details={"error": str(e)})
            return False

    # ------------------------------------------------------------------
    # RHO — Health Monitoring
    # ------------------------------------------------------------------

    @property
    def rho(self) -> Optional[Any]:
        """RHO: System health monitoring."""
        if not _RHO_AVAILABLE:
            return None
        if "rho" not in self._agents:
            from aura.health import HealthMonitor

            self._agents["rho"] = HealthMonitor()
        return self._agents["rho"]

    def run_health_checks(self) -> Dict[str, Any]:
        """Run comprehensive health checks.

        Returns:
            Health check results by component
        """
        if not self.rho:
            return {"healthy": True, "checks": {}, "reason": "RHO not available"}

        try:
            results = self.rho.check_all()
            log_json(
                "INFO",
                "rho_health_checks_complete",
                details={
                    "healthy": results.get("healthy", False),
                    "check_count": len(results.get("checks", {})),
                },
            )
            return results
        except Exception as e:
            log_json("ERROR", "rho_health_checks_failed", details={"error": str(e)})
            return {"healthy": False, "error": str(e)}

    def check_component_health(self, component: str) -> Dict[str, Any]:
        """Check health of a specific component.

        Args:
            component: Component name to check

        Returns:
            Component health status
        """
        if not self.rho:
            return {"healthy": True, "component": component, "reason": "RHO not available"}

        try:
            return self.rho.check_component(component)
        except Exception as e:
            return {"healthy": False, "component": component, "error": str(e)}

    # ------------------------------------------------------------------
    # SIGMA — Security Gate
    # ------------------------------------------------------------------

    @property
    def sigma(self) -> Optional[Any]:
        """SIGMA: Security auditing and secret detection."""
        if not _SIGMA_AVAILABLE:
            return None
        if "sigma" not in self._agents:
            from aura.security import SecurityAuditor

            self._agents["sigma"] = SecurityAuditor()
        return self._agents["sigma"]

    def scan_for_secrets(self, content: str) -> List[Dict[str, Any]]:
        """Scan content for secrets.

        Args:
            content: Text content to scan

        Returns:
            List of detected secrets
        """
        if not self.sigma:
            return []

        try:
            from aura.security import SecretScanner

            scanner = SecretScanner()
            findings = scanner.scan(content)
            log_json(
                "INFO",
                "sigma_secret_scan_complete",
                details={
                    "findings_count": len(findings),
                },
            )
            return findings
        except Exception as e:
            log_json("ERROR", "sigma_scan_failed", details={"error": str(e)})
            return []

    def security_gate_check(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run security gate checks on proposed changes.

        Args:
            changes: List of file changes to validate

        Returns:
            Gate check result with allow/block decision
        """
        if not self.sigma:
            return {"allowed": True, "reason": "SIGMA not available"}

        try:
            result = self.sigma.validate_changes(changes)
            log_json(
                "INFO",
                "sigma_security_gate_complete",
                details={
                    "allowed": result.get("allowed", False),
                    "violations": len(result.get("violations", [])),
                },
            )
            return result
        except Exception as e:
            log_json("ERROR", "sigma_gate_check_failed", details={"error": str(e)})
            return {"allowed": True, "reason": f"Check failed: {e}"}

    # ------------------------------------------------------------------
    # TAU — Background Tasks
    # ------------------------------------------------------------------

    @property
    def tau(self) -> Optional[Any]:
        """TAU: Background task scheduling."""
        if not _TAU_AVAILABLE:
            return None
        if "tau" not in self._agents:
            from aura.scheduler import TaskScheduler

            self._agents["tau"] = TaskScheduler()
        return self._agents["tau"]

    def schedule_task(
        self,
        task_id: str,
        task: Callable,
        schedule: str,  # cron expression or "now"
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        """Schedule a background task.

        Args:
            task_id: Unique task identifier
            task: Function to execute
            schedule: When to run (cron expression or "now")
            *args, **kwargs: Arguments for the task

        Returns:
            Scheduling result
        """
        if not self.tau:
            # No scheduler available, execute immediately
            if schedule == "now":
                try:
                    result = task(*args, **kwargs)
                    return {"scheduled": False, "executed": True, "result": result}
                except Exception as e:
                    return {"scheduled": False, "executed": False, "error": str(e)}
            return {"scheduled": False, "reason": "TAU not available"}

        try:
            job = self.tau.schedule(task_id, task, schedule, *args, **kwargs)
            log_json(
                "INFO",
                "tau_task_scheduled",
                details={
                    "task_id": task_id,
                    "schedule": schedule,
                },
            )
            return {"scheduled": True, "job": job}
        except Exception as e:
            log_json("ERROR", "tau_schedule_failed", details={"error": str(e)})
            return {"scheduled": False, "reason": str(e)}

    def get_scheduled_tasks(self) -> List[Dict[str, Any]]:
        """Get list of scheduled tasks.

        Returns:
            List of scheduled task info
        """
        if not self.tau:
            return []

        try:
            return self.tau.list_tasks()
        except Exception as e:
            log_json("ERROR", "tau_list_tasks_failed", details={"error": str(e)})
            return []

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all sub-agents.

        Returns:
            Status report for each sub-agent
        """
        return {
            name: {
                "available": available,
                "initialized": name in self._agents,
            }
            for name, available in self._availability.items()
        }

    def is_agent_available(self, agent_name: str) -> bool:
        """Check if a specific agent is available.

        Args:
            agent_name: Name of the agent (iota, kappa, etc.)

        Returns:
            True if agent is available
        """
        return self._availability.get(agent_name.lower(), False)


# Global registry instance
_global_registry: Optional[SubAgentRegistry] = None


def get_subagent_registry() -> SubAgentRegistry:
    """Get the global sub-agent registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = SubAgentRegistry()
    return _global_registry
