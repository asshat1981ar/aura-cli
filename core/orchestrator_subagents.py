"""Sub-Agent Integration Mixin for LoopOrchestrator.

Wires Phase 2 sub-agents into the orchestrator pipeline at appropriate phases:
- IOTA: Error handling phase (auto-resolve errors)
- KAPPA: Workflow recording/replay
- NU: Pre-flight connectivity checks
- PI: Config loading with encryption
- RHO: Pre-flight health validation
- SIGMA: Security gate on changes
- TAU: Background task scheduling
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from core.logging_utils import log_json
from core.subagent_integration import get_subagent_registry


class SubAgentMixin:
    """Mixin class adding sub-agent integration to LoopOrchestrator."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._subagents = get_subagent_registry()
        self._subagent_metrics = {
            "iota_invocations": 0,
            "kappa_recordings": 0,
            "kappa_replays": 0,
            "nu_offline_switches": 0,
            "rho_health_checks": 0,
            "sigma_gate_blocks": 0,
            "tau_tasks_scheduled": 0,
        }
    
    # ------------------------------------------------------------------
    # IOTA Integration — Error Resolution
    # ------------------------------------------------------------------
    
    def _attempt_error_resolution(
        self,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Attempt to resolve an error using IOTA.
        
        Called from error handling phase when a recoverable error occurs.
        
        Args:
            error_message: The error to resolve
            context: Additional context (stack trace, phase info, etc.)
        
        Returns:
            Resolution result if successful, None otherwise
        """
        if not self._subagents.is_agent_available("iota"):
            return None
        
        log_json("INFO", "iota_error_resolution_start", details={
            "error_preview": error_message[:100],
        })
        
        result = self._subagents.resolve_error(
            error_message=error_message,
            context=context,
            timeout_seconds=30.0,
        )
        
        if result.get("resolved"):
            self._subagent_metrics["iota_invocations"] += 1
            log_json("INFO", "iota_resolution_success", details={
                "fix_type": result.get("fix_type", "unknown"),
            })
            return result
        
        log_json("INFO", "iota_resolution_failed", details={
            "reason": result.get("reason", "unknown"),
        })
        return None
    
    def _should_retry_with_iota(self, phase_result: Dict[str, Any]) -> bool:
        """Determine if IOTA should attempt to resolve a phase failure.
        
        Args:
            phase_result: The failed phase result
        
        Returns:
            True if IOTA should attempt resolution
        """
        if not self._subagents.is_agent_available("iota"):
            return False
        
        # Only attempt on specific error types
        error_type = phase_result.get("error_type", "")
        recoverable_types = [
            "syntax_error",
            "import_error",
            "attribute_error",
            "type_error",
            "test_failure",
        ]
        
        return error_type in recoverable_types
    
    # ------------------------------------------------------------------
    # KAPPA Integration — Workflow Recording
    # ------------------------------------------------------------------
    
    def _record_workflow_if_enabled(
        self,
        workflow_id: str,
        steps: List[Dict[str, Any]],
    ) -> None:
        """Record a workflow if KAPPA is available.
        
        Args:
            workflow_id: Unique workflow identifier
            steps: Workflow steps to record
        """
        if not self._subagents.is_agent_available("kappa"):
            return
        
        result = self._subagents.record_workflow(workflow_id, steps)
        if result.get("recorded"):
            self._subagent_metrics["kappa_recordings"] += 1
    
    def _try_replay_workflow(
        self,
        workflow_id: str,
        variables: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Attempt to replay a recorded workflow.
        
        Args:
            workflow_id: Workflow to replay
            variables: Variables for substitution
        
        Returns:
            Replay result if successful, None otherwise
        """
        if not self._subagents.is_agent_available("kappa"):
            return None
        
        result = self._subagents.replay_workflow(workflow_id, variables)
        if result.get("replayed"):
            self._subagent_metrics["kappa_replays"] += 1
        return result
    
    # ------------------------------------------------------------------
    # NU Integration — Offline Mode
    # ------------------------------------------------------------------
    
    def _check_connectivity_before_cycle(self) -> Dict[str, Any]:
        """Check connectivity before starting an orchestrator cycle.
        
        Returns:
            Connectivity status with recommended mode
        """
        if not self._subagents.is_agent_available("nu"):
            return {"online": True, "mode": "normal"}
        
        status = self._subagents.check_connectivity()
        
        if not status.get("online", True):
            self._subagent_metrics["nu_offline_switches"] += 1
            log_json("WARN", "nu_offline_mode_activated", details={
                "reason": status.get("reason", "connectivity_lost"),
            })
        
        return status
    
    def _execute_offline_aware(
        self,
        operation: callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute an operation with offline awareness.
        
        Args:
            operation: Function to execute
            *args, **kwargs: Arguments for operation
        
        Returns:
            Operation result
        """
        return self._subagents.execute_offline_aware(operation, *args, **kwargs)
    
    # ------------------------------------------------------------------
    # PI Integration — Encrypted Config
    # ------------------------------------------------------------------
    
    def _load_secure_config(
        self,
        config_path: str,
        key_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load configuration with encryption support.
        
        Args:
            config_path: Path to config file
            key_id: Optional key identifier
        
        Returns:
            Decrypted configuration dict
        """
        return self._subagents.load_encrypted_config(config_path, key_id)
    
    def _save_secure_config(
        self,
        config_path: str,
        config: Dict[str, Any],
        key_id: Optional[str] = None,
    ) -> bool:
        """Save configuration with encryption.
        
        Args:
            config_path: Path to save config
            config: Configuration dict
            key_id: Optional key identifier
        
        Returns:
            True if saved successfully
        """
        return self._subagents.save_encrypted_config(config_path, config, key_id)
    
    # ------------------------------------------------------------------
    # RHO Integration — Health Checks
    # ------------------------------------------------------------------
    
    def _run_preflight_health_checks(self) -> Dict[str, Any]:
        """Run health checks before starting orchestration.
        
        Returns:
            Health check results
        """
        if not self._subagents.is_agent_available("rho"):
            return {"healthy": True, "checks": {}}
        
        results = self._subagents.run_health_checks()
        self._subagent_metrics["rho_health_checks"] += 1
        
        if not results.get("healthy"):
            log_json("WARN", "rho_health_checks_failed", details={
                "failed_checks": [
                    name for name, check in results.get("checks", {}).items()
                    if not check.get("healthy", True)
                ],
            })
        
        return results
    
    def _is_system_healthy_for_phase(self, phase: str) -> bool:
        """Check if system is healthy enough for a specific phase.
        
        Args:
            phase: Pipeline phase to check
        
        Returns:
            True if system can proceed
        """
        if not self._subagents.is_agent_available("rho"):
            return True
        
        # Map phases to required components
        phase_requirements = {
            "plan": ["memory", "llm"],
            "act": ["filesystem", "git"],
            "sandbox": ["subprocess", "filesystem"],
            "verify": ["test_runner", "git"],
        }
        
        requirements = phase_requirements.get(phase, [])
        for component in requirements:
            check = self._subagents.check_component_health(component)
            if not check.get("healthy", True):
                return False
        
        return True
    
    # ------------------------------------------------------------------
    # SIGMA Integration — Security Gate
    # ------------------------------------------------------------------
    
    def _run_security_gate(
        self,
        changes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run security gate on proposed changes.
        
        Args:
            changes: List of file changes to validate
        
        Returns:
            Gate result with allow/block decision
        """
        if not self._subagents.is_agent_available("sigma"):
            return {"allowed": True, "reason": "SIGMA not available"}
        
        result = self._subagents.security_gate_check(changes)
        
        if not result.get("allowed", True):
            self._subagent_metrics["sigma_gate_blocks"] += 1
            log_json("WARN", "sigma_security_gate_blocked", details={
                "violations": result.get("violations", []),
            })
        
        return result
    
    def _scan_content_for_secrets(self, content: str) -> List[Dict[str, Any]]:
        """Scan content for secrets before commit.
        
        Args:
            content: Text content to scan
        
        Returns:
            List of detected secrets
        """
        return self._subagents.scan_for_secrets(content)
    
    # ------------------------------------------------------------------
    # TAU Integration — Background Tasks
    # ------------------------------------------------------------------
    
    def _schedule_background_task(
        self,
        task_id: str,
        task: callable,
        schedule: str,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """Schedule a background task.
        
        Args:
            task_id: Unique task identifier
            task: Function to execute
            schedule: Cron expression or "now"
            *args, **kwargs: Arguments for task
        
        Returns:
            Scheduling result
        """
        result = self._subagents.schedule_task(task_id, task, schedule, *args, **kwargs)
        
        if result.get("scheduled"):
            self._subagent_metrics["tau_tasks_scheduled"] += 1
        
        return result
    
    def _schedule_maintenance_tasks(self) -> None:
        """Schedule routine maintenance tasks.
        
        Called at orchestrator startup.
        """
        if not self._subagents.is_agent_available("tau"):
            return
        
        # Schedule memory cleanup
        self._schedule_background_task(
            "memory_cleanup",
            self._cleanup_old_memories,
            "0 2 * * *",  # Daily at 2 AM
        )
        
        # Schedule health check
        self._schedule_background_task(
            "health_check",
            self._run_preflight_health_checks,
            "*/15 * * * *",  # Every 15 minutes
        )
    
    def _cleanup_old_memories(self) -> None:
        """Cleanup old memories (called by scheduled task)."""
        try:
            if hasattr(self, 'memory_store') and self.memory_store:
                # Remove memories older than 30 days
                self.memory_store.clear_old(days=30)
                log_json("INFO", "memory_cleanup_completed")
        except Exception as e:
            log_json("ERROR", "memory_cleanup_failed", details={"error": str(e)})
    
    # ------------------------------------------------------------------
    # Metrics & Status
    # ------------------------------------------------------------------
    
    def get_subagent_metrics(self) -> Dict[str, Any]:
        """Get sub-agent invocation metrics.
        
        Returns:
            Metrics dict for all sub-agents
        """
        return {
            "metrics": self._subagent_metrics.copy(),
            "availability": self._subagents.get_agent_status(),
        }
    
    def get_subagent_status(self) -> Dict[str, Any]:
        """Get detailed sub-agent status.
        
        Returns:
            Status report for all sub-agents
        """
        return self._subagents.get_agent_status()
