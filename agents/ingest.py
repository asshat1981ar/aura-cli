import os
from pathlib import Path
from typing import Dict

from agents.base import Agent
from core.logging_utils import log_json
from memory.controller import MemoryTier


class IngestAgent(Agent):
    name = "ingest"

    def __init__(self, brain):
        self.brain = brain

    def _snapshot(self, project_root: Path) -> str:
        files = []
        try:
            for root, dirs, filenames in os.walk(project_root):
                dirs[:] = [d for d in dirs if d not in [".git", "__pycache__", ".pytest_cache", ".ralph-state", ".gemini", "memory", "node_modules"]]
                for fname in filenames:
                    rel_path = os.path.relpath(os.path.join(root, fname), project_root)
                    files.append(rel_path)
        except Exception as exc:
            log_json("WARN", "snapshot_file_listing_failed", details={"error": str(exc)})
        return "\n".join(sorted(files)[:150])

    def run(self, input_data: Dict) -> Dict:
        goal = input_data.get("goal", "")
        project_root = Path(input_data.get("project_root", "."))
        memory_entries = self.brain.recall_all(MemoryTier.SESSION)
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
        snapshot = self._snapshot(project_root)
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
