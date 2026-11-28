"""Reflection, learning, and feedback methods for the orchestrator."""

from __future__ import annotations

import dataclasses
import json
import time
from typing import Dict

from core.logging_utils import log_json
from core.cycle_outcome import CycleOutcome
from core.operator_runtime import build_cycle_summary
from memory.controller import MemoryTier


class LearnMixin:
    """Mixin providing reflection, outcome recording, and feedback methods."""

    def _run_reflection_phase(self, verification: Dict, skill_context: Dict, goal_type: str, cycle_id: str, phase_outputs: Dict) -> Dict:
        """Section 7: REFLECT."""
        import sys

        _orch = sys.modules["core.orchestrator"]

        self._notify_ui("on_phase_start", "reflect")
        t0 = time.time()
        reflection = self._run_phase(
            "reflect",
            {
                "verification": verification,
                "skill_context": skill_context,
                "goal_type": goal_type,
                "pipeline_run_id": getattr(self, "_cycle_context", {}).get("pipeline_run_id"),
            },
        )
        self._notify_ui("on_phase_complete", "reflect", (time.time() - t0) * 1000)
        errors = _orch.validate_phase_output("reflection", reflection)
        if errors:
            log_json("ERROR", "phase_schema_invalid", details={"phase": "reflection", "errors": errors})
        phase_outputs["reflection"] = reflection
        if reflection.get("summary"):
            self.memory_controller.store(MemoryTier.SESSION, reflection["summary"], metadata={"cycle_id": cycle_id, "type": "reflection"})
        return reflection

    def _refresh_cycle_summary(self, entry: Dict, *, state: str = "complete", current_phase: str | None = None) -> Dict:
        """Rebuild and cache the canonical operator-facing cycle summary."""
        summary = build_cycle_summary(entry, state=state, current_phase=current_phase)
        entry["cycle_summary"] = summary
        if state == "running":
            self.active_cycle_summary = summary
        else:
            self.last_cycle_summary = summary
        return summary

    def _build_cycle_outcome(self, goal: str, goal_type: str, started_at: float, phase_outputs: Dict, passed: bool) -> tuple:
        """Create CycleOutcome and build the cycle entry dict.

        Returns (entry, outcome, changed_files, quality) tuple.
        """
        from core.quality_snapshot import run_quality_snapshot

        # Phase 8: measure()
        self._notify_ui("on_phase_start", "measure")
        t0_measure = time.time()
        changed_files = phase_outputs.get("apply_result", {}).get("applied", [])
        quality = run_quality_snapshot(self.project_root, changed_files=changed_files)
        phase_outputs["quality"] = quality
        self._notify_ui("on_phase_complete", "measure", (time.time() - t0_measure) * 1000)

        outcome = CycleOutcome(
            goal=goal,
            goal_type=goal_type,
            started_at=started_at,
            phases_completed=list(phase_outputs.keys()),
            changes_applied=len(changed_files),
            tests_after=quality.get("test_count", 0),
            strategy_used=phase_outputs.get("pipeline_config", {}).get("intensity", "normal"),
        )

        if self.adaptive_pipeline:
            try:
                self.adaptive_pipeline.record_outcome(goal_type, outcome.strategy_used, passed)
            except Exception as exc:
                log_json("WARN", "adaptive_pipeline_outcome_record_failed", details={"error": str(exc)})

        completed_at = time.time()
        outcome.mark_complete(success=passed)
        entry = {
            "cycle_id": None,  # filled by caller
            "goal": goal,
            "goal_type": goal_type,
            "phase_outputs": phase_outputs,
            "dry_run": bool(phase_outputs.get("dry_run")),
            "beads": phase_outputs.get("beads_gate"),
            "stop_reason": None,
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_s": outcome.duration_s(),
            "outcome": dataclasses.asdict(outcome),
        }
        return entry, outcome, changed_files, quality

    def _persist_cycle_entry(self, entry: Dict, goal: str, cycle_id: str, outcome: "CycleOutcome", passed: bool) -> None:
        """Phase 9: learn -- persist cycle entry to memory stores and brain."""
        self._notify_ui("on_phase_start", "learn")
        t0_learn = time.time()
        summary = self._refresh_cycle_summary(entry)
        self._notify_ui("on_cycle_complete", summary)
        self.current_goal = None
        self.active_cycle_summary = None

        if self.memory_controller.persistent_store:
            self.memory_controller.persistent_store.append_log(entry)
            self.memory_controller.store(
                MemoryTier.PROJECT,
                json.dumps(summary),
                metadata={"type": "cycle_summary", "goal": goal, "cycle_id": cycle_id},
            )

        if self.brain:
            try:
                self.brain.set(f"outcome:{cycle_id}", outcome.to_json())
                self.brain.remember(f"Cycle completed: {goal} -> {'SUCCESS' if passed else 'FAILED'}")
            except Exception as exc:
                log_json("WARN", "brain_outcome_storage_failed", details={"error": str(exc)})

        self._notify_ui("on_phase_complete", "learn", (time.time() - t0_learn) * 1000)

    def _run_post_cycle_hooks(self, entry: Dict, goal: str, cycle_id: str, phase_outputs: Dict, passed: bool) -> None:
        """Run n8n feedback, context graph, improvement loops, discovery, evolution, propagation, and memory consolidation."""
        # P4 Feedback Loop
        log_json("INFO", "n8n_feedback_attempting", details={"cycle_id": cycle_id, "passed": passed})
        try:
            self._notify_n8n_feedback(goal, cycle_id, passed, phase_outputs)
        except Exception as exc:
            log_json("WARN", "n8n_feedback_notify_failed", details={"error": str(exc)})

        if self.context_graph is not None:
            try:
                self.context_graph.update_from_cycle(entry)
            except Exception as exc:
                log_json("WARN", "context_graph_update_failed", details={"error": str(exc)})

        for loop in self._improvement_loops:
            try:
                loop.on_cycle_complete(entry)
            except Exception as exc:
                log_json("WARN", "improvement_loop_error", details={"loop": type(loop).__name__, "error": str(exc)})

        # Phase 10: discover() — drain pending learning backlog into the goal queue
        self._notify_ui("on_phase_start", "discover")
        t0_disc = time.time()
        if getattr(self, "learning_coordinator", None) is not None and self.goal_queue:
            try:
                backlog = self.learning_coordinator.generate_backlog(limit=3)
                for g in backlog:
                    self.goal_queue.add(g)
                    log_json("INFO", "discover_backlog_goal_enqueued", details={"goal": g[:100]})
            except Exception as exc:
                log_json("WARN", "discover_backlog_error", details={"error": str(exc)})
        self._notify_ui("on_phase_complete", "discover", (time.time() - t0_disc) * 1000)

        # Phase 11: evolve()
        self._notify_ui("on_phase_start", "evolve")
        self._notify_ui("on_phase_complete", "evolve", 0)

        if self.propagation_engine is not None:
            try:
                self.propagation_engine.on_cycle_complete(entry)
            except Exception as exc:
                log_json("WARN", "propagation_engine_error", details={"error": str(exc)})

        # Memory Consolidation (periodic, every 10 cycles)
        cycle_num = int(cycle_id.split("_")[-1], 16) if "_" in cycle_id else 0
        if self.brain and cycle_num % 10 == 0:
            try:
                from memory.consolidation import MemoryConsolidator, MemoryEntry

                consolidator = MemoryConsolidator()
                raw_memories = self.brain.recall_with_budget(max_tokens=50000)
                entries = [MemoryEntry(id=str(i), content=m, memory_type="decision") for i, m in enumerate(raw_memories)]
                if len(entries) > 50:
                    retained, result = consolidator.consolidate(entries)
                    log_json(
                        "INFO",
                        "memory_consolidation_complete",
                        details={
                            "before": result.memories_before,
                            "after": result.memories_after,
                            "compression": f"{result.compression_ratio:.1%}",
                        },
                    )
            except Exception as exc:
                log_json("WARN", "memory_consolidation_error", details={"error": str(exc)})

    def _record_cycle_outcome(self, cycle_id: str, goal: str, goal_type: str, phase_outputs: Dict, started_at: float):
        """Final persistence and loop notification."""
        verify_status = phase_outputs.get("verification", {}).get("status", "skip")
        passed = verify_status in ("pass", "skip")
        if passed:
            self._consecutive_fails = 0
        elif verify_status == "fail":
            self._consecutive_fails += 1

        # Build outcome and entry
        entry, outcome, changed_files, quality = self._build_cycle_outcome(
            goal,
            goal_type,
            started_at,
            phase_outputs,
            passed,
        )
        entry["cycle_id"] = cycle_id

        # Quality Trend Analysis
        alerts: list = []
        try:
            alerts = self.quality_trends.record_from_cycle(
                {
                    "cycle_id": cycle_id,
                    "goal": goal,
                    "completed_at": time.time(),
                    "duration_s": time.time() - started_at,
                    "phase_outputs": phase_outputs,
                }
            )
            if alerts and self.goal_queue:
                for goal_text in self.quality_trends.get_remediation_goals():
                    self.goal_queue.add(goal_text)
                    log_json("INFO", "quality_remediation_goal_enqueued", details={"goal": goal_text[:100]})
        except Exception as exc:
            log_json("WARN", "quality_trend_record_failed", details={"error": str(exc)})

        # Learning Coordinator (PRD-003): convert signals to artifacts + immediate goals
        if getattr(self, "learning_coordinator", None) is not None:
            try:
                reflection = phase_outputs.get("reflection", {})
                lc_goals = self.learning_coordinator.on_cycle_complete(entry, reflection, alerts or [])
                if self.goal_queue:
                    for g in lc_goals:
                        self.goal_queue.add(g)
                        log_json("INFO", "learning_artifact_goal_enqueued", details={"goal": g[:100]})
            except Exception as exc:
                log_json("WARN", "learning_coordinator_cycle_failed", details={"error": str(exc)})

        # Persist and run post-cycle hooks
        self._persist_cycle_entry(entry, goal, cycle_id, outcome, passed)
        self._run_post_cycle_hooks(entry, goal, cycle_id, phase_outputs, passed)

        return entry

    def _notify_n8n_feedback(self, goal: str, cycle_id: str, passed: bool, phase_outputs: Dict) -> None:
        """POST cycle reflection + skill summary to n8n P4 Feedback Loop webhook.

        Fires non-blocking after the reflect phase completes.  Gated by
        ``n8n_connector.enabled`` in config.  Failures are swallowed and
        logged — never allowed to interrupt cycle completion.
        """
        try:
            config = self._load_config_file()
            n8n_cfg = config.get("n8n_connector", {})
            if not n8n_cfg.get("enabled", False):
                return

            webhook_url = n8n_cfg.get("feedback_loop_webhook", "")
            if not webhook_url:
                return

            reflection: Dict = phase_outputs.get("reflection", {})
            payload = {
                "pipeline_run_id": getattr(self, "_cycle_context", {}).get("pipeline_run_id", cycle_id),
                "cycle_id": cycle_id,
                "goal": goal,
                "passed": passed,
                "verification_status": phase_outputs.get("verification", {}).get("status", "skip"),
                "learnings": reflection.get("learnings", []),
                "summary": reflection.get("summary", ""),
                "skill_summary": reflection.get("skill_summary", {}),
                "act_confidence": phase_outputs.get("act_confidence"),
                "plan_confidence": phase_outputs.get("plan_confidence"),
                "retry_count": phase_outputs.get("retry_count", 0),
                "quality": phase_outputs.get("quality", {}),
            }

            import urllib.request

            def _post(url: str, body: bytes) -> int:
                req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    return resp.status

            data = json.dumps(payload).encode()
            status = _post(webhook_url, data)
            log_json(
                "INFO",
                "n8n_feedback_sent",
                details={
                    "cycle_id": cycle_id,
                    "passed": passed,
                    "learnings": len(payload["learnings"]),
                    "status_code": status,
                },
            )

            # Fan-out to P5 Observability Collector (best-effort)
            obs_url = n8n_cfg.get("observability_webhook", "")
            if obs_url:
                try:
                    _post(obs_url, data)
                except Exception as obs_exc:
                    log_json("WARN", "n8n_observability_send_failed", details={"error": str(obs_exc)})

        except Exception as exc:
            log_json("WARN", "n8n_feedback_notify_failed", details={"error": str(exc)})

    def _enrich_act_context(self, task_bundle: Dict) -> Dict:
        """Inject quality-gate critique from n8n P3 Pipeline Coordinator into task_bundle.

        P3 pre-fetches Dev Suite review results and injects them via
        WebhookGoalRequest.metadata["quality_gate_critique"] into the cycle
        context before the act phase.  This method reads from that context
        field and prepends the critique to the task bundle so the CoderAgent
        sees it as part of the prompt — no blocking HTTP call.

        If quality_gate is not enabled or no critique is available, returns
        task_bundle unchanged.
        """
        try:
            config = self._load_config_file()
            n8n_cfg = config.get("n8n_connector", {})
            if not n8n_cfg.get("quality_gate_enabled", False):
                return task_bundle

            critique = getattr(self, "_cycle_context", {}).get("quality_gate_critique") or (task_bundle.get("_cycle_context") or {}).get("quality_gate_critique") or task_bundle.get("quality_gate_critique")
            if not critique:
                return task_bundle

            task_bundle = dict(task_bundle) if isinstance(task_bundle, dict) else {}
            existing = task_bundle.get("critique", "") or ""
            task_bundle["critique"] = f"[Dev Suite Quality Gate Review]\n{critique}\n\n{existing}".strip()
            log_json(
                "INFO",
                "aura_act_context_enriched",
                details={
                    "critique_len": len(critique),
                    "quality_gate_enabled": True,
                },
            )
        except Exception as exc:
            log_json("WARN", "aura_act_context_enrich_failed", details={"error": str(exc)})
        return task_bundle
