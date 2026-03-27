"""Persistent lesson storage for reflection-driven cycles."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List

from core.swarm_models import CycleLesson, CycleReport


class LessonStore:
    """Stores reusable cycle lessons as JSONL for lightweight retrieval."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.lessons_file = self.root_dir / "cycle_lessons.jsonl"

    async def record_cycle(self, report: CycleReport) -> None:
        """Persist lessons derived from the most recent cycle report."""
        lessons = report.lessons_injected[:]
        if report.debug_report:
            lessons.append(
                CycleLesson(
                    cycle_number=report.cycle_number,
                    lesson=f"Debugger recovery plan: {'; '.join(report.debug_report.recovery_plan)}",
                    source_task_id=report.debug_report.task_id,
                    confidence=0.9,
                )
            )

        if not lessons:
            return

        payload = "".join(json.dumps(item.model_dump()) + "\n" for item in lessons)
        await asyncio.to_thread(self._append_text, payload)

    async def injectable_lessons(self, limit: int = 5) -> List[CycleLesson]:
        """Return the most recent lessons for injection into the next cycle."""
        if not self.lessons_file.exists():
            return []

        raw = await asyncio.to_thread(self.lessons_file.read_text, "utf-8")
        lines = [line for line in raw.splitlines() if line.strip()]
        selected = lines[-limit:]
        return [CycleLesson.model_validate(json.loads(line)) for line in selected]

    def _append_text(self, payload: str) -> None:
        with self.lessons_file.open("a", encoding="utf-8") as handle:
            handle.write(payload)
