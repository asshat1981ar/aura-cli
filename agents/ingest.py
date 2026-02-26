import os
from pathlib import Path
from typing import Dict, List

from agents.base import Agent
from core.logging_utils import log_json
from memory.controller import MemoryTier
from core.context_manager import ContextManager


class IngestAgent(Agent):
    name = "ingest"

    def __init__(self, brain, context_manager: ContextManager = None):
        self.brain = brain
        self.cm = context_manager

    def _snapshot(self, project_root: Path) -> List[str]:
        files = []
        try:
            for root, dirs, filenames in os.walk(project_root):
                dirs[:] = [d for d in dirs if d not in [".git", "__pycache__", ".pytest_cache", ".ralph-state", ".gemini", "memory", "node_modules"]]
                for fname in filenames:
                    rel_path = os.path.relpath(os.path.join(root, fname), project_root)
                    files.append(rel_path)
        except Exception as exc:
            log_json("WARN", "snapshot_file_listing_failed", details={"error": str(exc)})
        return sorted(files)

    def run(self, input_data: Dict) -> Dict:
        goal = input_data.get("goal", "")
        project_root = Path(input_data.get("project_root", "."))
        memory_entries = self.brain.recall_all(MemoryTier.SESSION)
        
        # Use ContextManager if available, else fallback to basic logic
        if self.cm:
            file_list = self._snapshot(project_root)
            bundle = self.cm.get_context_bundle(
                goal=goal,
                goal_type=input_data.get("goal_type", "default"),
                recent_memory=[str(m) for m in memory_entries],
                file_list=file_list
            )
            context = {
                "goal": goal,
                "bundle": bundle,
                "prompt_segment": self.cm.format_as_prompt(bundle),
                "memory_summary": "\n".join(str(m) for m in bundle["memory"]),
                "snapshot": "\n".join(bundle["files"]),
                "hints_summary": "\n".join(bundle["related_insights"]),
                "constraints": input_data.get("constraints", {}),
            }
        else:
            # Fallback to old basic logic
            memory_summary = "\n".join(str(m) for m in memory_entries[-10:])
            hints = input_data.get("hints") or []
            hints_summary = []
            for h in hints:
                if not isinstance(h, dict):
                    continue
                cid = h.get("cycle_id") or h.get("id")
                status = h.get("verification_status") or h.get("status")
                stop = h.get("stop_reason")
                failure = ""
                failures = h.get("failures") or h.get("phase_outputs", {}).get("verification", {}).get("failures")
                if isinstance(failures, list) and failures:
                    failure = f" first_failure={failures[0]}"
                desc = h.get("summary") or h.get("notes") or ""
                hints_summary.append(f"- cycle={cid or 'n/a'} status={status or 'n/a'} stop={stop or 'n/a'}{failure} {desc}".strip())
            hints_text = "\n".join(hints_summary)
            snapshot = "\n".join(self._snapshot(project_root)[:150])
            context = {
                "goal": goal,
                "snapshot": snapshot,
                "memory_summary": memory_summary,
                "constraints": input_data.get("constraints", {}),
                "hints": hints,
                "hints_summary": hints_text,
                "instructions": (
                    "HINTS BELOW: summarize recent cycles touching this goal. "
                    "Do not repeat failures; prefer paths that avoid listed stop reasons/failures."
                ),
            }
        return context
