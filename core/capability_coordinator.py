"""CapabilityCoordinator — extracted from LoopOrchestrator (B3).

Manages automatic capability analysis, skill recommendation, goal queuing
for missing capabilities, and MCP provisioning.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from core.capability_manager import (
    analyze_capability_needs,
    provision_capability_actions,
    queue_missing_capability_goals,
    record_capability_status,
)


class CapabilityCoordinator:
    """Coordinates capability analysis, skill augmentation, and MCP provisioning."""

    def __init__(
        self,
        *,
        auto_add: bool = True,
        auto_queue_missing: bool = True,
        auto_provision_mcp: bool = False,
        auto_start_mcp_servers: bool = False,
        project_root: Path | None = None,
    ) -> None:
        self.auto_add = auto_add
        self.auto_queue_missing = auto_queue_missing
        self.auto_provision_mcp = auto_provision_mcp
        self.auto_start_mcp_servers = auto_start_mcp_servers
        self.project_root = project_root or Path(".")

        # Last-run snapshots (inspectable by UI / telemetry)
        self.last_plan: dict = {}
        self.last_goal_queue: dict = {}
        self.last_provisioning: dict = {}
        self.last_status: dict = {}

    def handle(
        self,
        goal: str,
        pipeline_cfg: Any,
        phase_outputs: Dict,
        *,
        skills: Dict[str, Any],
        goal_queue: Any | None = None,
        dry_run: bool = False,
    ) -> None:
        """Run capability analysis and update pipeline_cfg / phase_outputs in place.

        This is the main entry point called once per cycle, replacing the old
        ``LoopOrchestrator._handle_capabilities`` method.
        """
        capability_plan: dict = {
            "matched_capabilities": [],
            "recommended_skills": [],
            "missing_skills": [],
            "mcp_tools": [],
            "provisioning_actions": [],
        }
        capability_goal_queue: dict = {
            "attempted": False,
            "queued": [],
            "skipped": [],
            "queue_strategy": None,
        }
        capability_provisioning: dict = {"attempted": False, "results": []}

        if self.auto_add:
            capability_plan = analyze_capability_needs(
                goal,
                available_skills=skills.keys(),
                active_skills=pipeline_cfg.skill_set,
            )
            if capability_plan["recommended_skills"]:
                pipeline_cfg.skill_set = list(
                    dict.fromkeys(
                        list(pipeline_cfg.skill_set)
                        + list(capability_plan["recommended_skills"])
                    )
                )
                phase_outputs["pipeline_config"]["skills"] = pipeline_cfg.skill_set

            phase_outputs["capability_plan"] = capability_plan

            capability_goal_queue = queue_missing_capability_goals(
                goal_queue=goal_queue,
                missing_skills=capability_plan["missing_skills"],
                goal=goal,
                enabled=self.auto_queue_missing,
                dry_run=dry_run,
            )
            if capability_goal_queue["queued"] or capability_goal_queue["skipped"]:
                phase_outputs["capability_goal_queue"] = capability_goal_queue

            if capability_plan["provisioning_actions"]:
                capability_provisioning = provision_capability_actions(
                    project_root=self.project_root,
                    provisioning_actions=capability_plan["provisioning_actions"],
                    auto_provision=self.auto_provision_mcp,
                    start_servers=self.auto_start_mcp_servers,
                    dry_run=dry_run,
                )
                phase_outputs["capability_provisioning"] = capability_provisioning

        self.last_plan = capability_plan
        self.last_goal_queue = capability_goal_queue
        self.last_provisioning = capability_provisioning
        self.last_status = record_capability_status(
            project_root=self.project_root,
            goal=goal,
            capability_plan=capability_plan,
            capability_goal_queue=capability_goal_queue,
            capability_provisioning=capability_provisioning,
            goal_queue=goal_queue,
        )
