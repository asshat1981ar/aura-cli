"""Capability management, pipeline configuration, and BEADS gate methods."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from core.logging_utils import log_json
from core.beads_bridge import build_beads_runtime_input


class CapabilitiesMixin:
    """Mixin providing capability management, pipeline config, and BEADS methods."""

    def _get_beads_skill(self):
        """Return the BEADS skill only when runtime BEADS integration is enabled."""
        if not self.beads_enabled:
            return None
        return self.skills.get("beads_skill")

    def _configure_pipeline(self, goal: str, goal_type: str, phase_outputs: Dict) -> Any:
        """Section 0: ADAPTIVE PIPELINE CONFIG."""
        if self.adaptive_pipeline:
            pipeline_cfg = self.adaptive_pipeline.configure(
                goal,
                goal_type,
                consecutive_fails=self._consecutive_fails,
                past_failures=list(phase_outputs.get("_failure_context", {}).get("failures", [])),
            )
        else:
            from core.adaptive_pipeline import AdaptivePipeline

            pipeline_cfg = AdaptivePipeline()._default_config(goal_type)

        phase_outputs["pipeline_config"] = {
            "intensity": pipeline_cfg.intensity,
            "phases": pipeline_cfg.phases,
            "skills": pipeline_cfg.skill_set,
        }
        self._notify_ui("on_pipeline_configured", phase_outputs["pipeline_config"])
        return pipeline_cfg

    def _handle_capabilities(self, goal: str, pipeline_cfg: Any, phase_outputs: Dict, dry_run: bool):
        """Section 0.1: CAPABILITY MANAGEMENT."""
        import sys
        _orch = sys.modules['core.orchestrator']

        capability_plan = {"matched_capabilities": [], "recommended_skills": [], "missing_skills": [], "mcp_tools": [], "provisioning_actions": []}
        capability_goal_queue = {"attempted": False, "queued": [], "skipped": [], "queue_strategy": None}
        capability_provisioning = {"attempted": False, "results": []}

        if self.auto_add_capabilities:
            capability_plan = _orch.analyze_capability_needs(goal, available_skills=self.skills.keys(), active_skills=pipeline_cfg.skill_set)
            if capability_plan["recommended_skills"]:
                pipeline_cfg.skill_set = list(dict.fromkeys(list(pipeline_cfg.skill_set) + list(capability_plan["recommended_skills"])))
                phase_outputs["pipeline_config"]["skills"] = pipeline_cfg.skill_set
            phase_outputs["capability_plan"] = capability_plan
            capability_goal_queue = _orch.queue_missing_capability_goals(goal_queue=self.goal_queue, missing_skills=capability_plan["missing_skills"], goal=goal, enabled=self.auto_queue_missing_capabilities, dry_run=dry_run)
            if capability_goal_queue["queued"] or capability_goal_queue["skipped"]:
                phase_outputs["capability_goal_queue"] = capability_goal_queue
            if capability_plan["provisioning_actions"]:
                capability_provisioning = _orch.provision_capability_actions(project_root=self.project_root, provisioning_actions=capability_plan["provisioning_actions"], auto_provision=self.auto_provision_mcp, start_servers=self.auto_start_mcp_servers, dry_run=dry_run)
                phase_outputs["capability_provisioning"] = capability_provisioning

        self.last_capability_plan = capability_plan
        self.last_capability_goal_queue = capability_goal_queue
        self.last_capability_provisioning = capability_provisioning
        self.last_capability_status = _orch.record_capability_status(project_root=self.project_root, goal=goal, capability_plan=capability_plan, capability_goal_queue=capability_goal_queue, capability_provisioning=capability_provisioning, goal_queue=self.goal_queue)

    def _beads_gate_applies(self) -> bool:
        if not self.beads_enabled or self.beads_bridge is None:
            return False
        if self.beads_scope == "all_runtime":
            return True
        return self.beads_scope == "goal_run"

    def _run_beads_gate(
        self,
        goal: str,
        goal_type: str,
        context: Dict,
        skill_context: Dict,
    ) -> Dict[str, Any]:
        active_context = {
            "context": context,
            "skill_context": skill_context,
            "capability_plan": self.last_capability_plan,
            "capability_goal_queue": self.last_capability_goal_queue,
            "capability_provisioning": self.last_capability_provisioning,
        }
        payload = build_beads_runtime_input(
            goal=goal,
            goal_type=goal_type,
            project_root=self.project_root,
            runtime_mode=self.runtime_mode,
            goal_queue=self.goal_queue,
            goal_archive=self.goal_archive,
            active_goal=goal,
            active_context=active_context,
        )
        log_json("INFO", "beads_gate_start", details={"goal": goal, "goal_type": goal_type, "scope": self.beads_scope})
        result = self.beads_bridge.run(payload)
        decision = result.get("decision") or {}
        beads_state = {
            "ok": bool(result.get("ok")),
            "status": decision.get("status") if decision else ("error" if not result.get("ok") else None),
            "decision_id": decision.get("decision_id"),
            "summary": decision.get("summary"),
            "required_constraints": list(decision.get("required_constraints", [])) if isinstance(decision, dict) else [],
            "required_skills": list(decision.get("required_skills", [])) if isinstance(decision, dict) else [],
            "required_tests": list(decision.get("required_tests", [])) if isinstance(decision, dict) else [],
            "follow_up_goals": list(decision.get("follow_up_goals", [])) if isinstance(decision, dict) else [],
            "stop_reason": decision.get("stop_reason") if isinstance(decision, dict) else None,
            "error": result.get("error"),
            "stderr": result.get("stderr"),
            "duration_ms": result.get("duration_ms", 0),
        }
        log_json(
            "INFO" if beads_state["ok"] else "WARN",
            "beads_gate_complete",
            details={
                "goal": goal,
                "ok": beads_state["ok"],
                "status": beads_state["status"],
                "decision_id": beads_state["decision_id"],
                "error": beads_state["error"],
            },
        )
        return beads_state

    def _build_early_stop_entry(
        self,
        *,
        cycle_id: str,
        goal: str,
        goal_type: str,
        phase_outputs: Dict,
        started_at: float,
        stop_reason: str,
        beads: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        entry = {
            "cycle_id": cycle_id,
            "goal": goal,
            "goal_type": goal_type,
            "phase_outputs": phase_outputs,
            "stop_reason": stop_reason,
            "started_at": started_at,
            "completed_at": time.time(),
        }
        if beads is not None:
            entry["beads"] = beads
        self._refresh_cycle_summary(entry)
        self.current_goal = None
        self.active_cycle_summary = None
        return entry

    def _parse_bead_id(self, goal: str) -> Optional[str]:
        """Extract bead ID from goal string if present (format: 'bead:ID: Title')."""
        if goal.startswith("bead:"):
            parts = goal.split(":", 2)
            if len(parts) >= 2:
                return parts[1]
        return None

    def _claim_bead(self, bead_id: str):
        """Mark a bead as in_progress using BeadsSkill."""
        beads_skill = self._get_beads_skill()
        if beads_skill is not None:
            log_json("INFO", "orchestrator_claiming_bead", details={"bead_id": bead_id})
            beads_skill.run({"cmd": "update", "id": bead_id, "args": ["--status", "in_progress"]})

    def _close_bead(self, bead_id: str, reason: str):
        """Close a bead using BeadsSkill."""
        beads_skill = self._get_beads_skill()
        if beads_skill is not None:
            log_json("INFO", "orchestrator_closing_bead", details={"bead_id": bead_id})
            beads_skill.run({"cmd": "close", "id": bead_id, "args": ["--reason", reason]})
