import json
import time
from dataclasses import dataclass, field
from typing import List, Optional, Any
from pathlib import Path
from core.logging_utils import log_json
from core.config_manager import config
from memory.controller import memory_controller, MemoryTier

@dataclass
class Task:
    id: str
    title: str
    status: str = "pending"
    description: str = ""
    subtasks: List['Task'] = field(default_factory=list)
    result: Optional[str] = None

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "description": self.description,
            "subtasks": [s.to_dict() for s in self.subtasks],
            "result": self.result
        }

    @classmethod
    def from_dict(cls, data):
        subtasks = [cls.from_dict(s) for s in data.get("subtasks", [])]
        return cls(
            id=data["id"],
            title=data["title"],
            status=data.get("status", "pending"),
            description=data.get("description", ""),
            subtasks=subtasks,
            result=data.get("result")
        )

    def display(self, indent: int = 0) -> str:
        status = self.status.replace("_", " ")
        lines = [f"{'  ' * indent}- [{status}] {self.title}"]
        for subtask in self.subtasks:
            lines.append(subtask.display(indent + 1))
        return "\n".join(lines)

class TaskManager:
    """
    Unified Control Plane: Manages task hierarchies via MemoryController.
    """
    def __init__(self, persistence_path=None):
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self.root_tasks: List[Task] = []
        self.load()

    def add_task(self, task: Task):
        self.root_tasks.append(task)
        memory_controller.store(MemoryTier.PROJECT, {"type": "task", "task_id": task.id, "title": task.title})
        self.save()

    def save(self):
        data = [t.to_dict() for t in self.root_tasks]
        if self.persistence_path is not None:
            try:
                self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
                self.persistence_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception as e:
                log_json("WARN", "task_hierarchy_file_save_failed", details={"error": str(e), "path": str(self.persistence_path)})
        memory_controller.store(MemoryTier.PROJECT, {"type": "task_hierarchy", "data": data}, metadata={"authority": "task_manager"})
        log_json("INFO", "task_hierarchy_persisted_to_controller", details={"root_count": len(self.root_tasks)})

    def load(self):
        if self.persistence_path is not None:
            if not self.persistence_path.exists():
                self.root_tasks = []
                log_json("INFO", "task_hierarchy_file_not_found", details={"path": str(self.persistence_path)})
                return
            try:
                data = json.loads(self.persistence_path.read_text(encoding="utf-8"))
                self.root_tasks = [Task.from_dict(t) for t in data]
                log_json("INFO", "task_hierarchy_loaded_from_file", details={"root_count": len(self.root_tasks), "path": str(self.persistence_path)})
            except Exception as e:
                self.root_tasks = []
                log_json("WARN", "task_hierarchy_file_load_failed", details={"error": str(e), "path": str(self.persistence_path)})
            return

        # Retrieve the hierarchy from the project memory tier
        records = memory_controller.retrieve(MemoryTier.PROJECT, limit=500)
        # Look for the most recent task_hierarchy record
        hierarchy_record = next((r for r in reversed(records) if isinstance(r, dict) and r.get("type") == "task_hierarchy"), None)
        
        if hierarchy_record:
            try:
                data = hierarchy_record.get("data", [])
                self.root_tasks = [Task.from_dict(t) for t in data]
                log_json("INFO", "task_hierarchy_loaded_from_controller", details={"root_count": len(self.root_tasks)})
            except Exception as e:
                log_json("WARN", "task_hierarchy_load_failed", details={"error": str(e)})
        else:
            log_json("INFO", "task_hierarchy_not_found_in_controller")

    def find_task(self, task_id: str, tasks: Optional[List[Task]] = None) -> Optional[Task]:
        if tasks is None:
            tasks = self.root_tasks
        for task in tasks:
            if task.id == task_id:
                return task
            found = self.find_task(task_id, task.subtasks)
            if found:
                return found
        return None

    def get_pending_tasks(self, tasks: Optional[List[Task]] = None) -> List[Task]:
        if tasks is None:
            tasks = self.root_tasks
        pending = []
        for task in tasks:
            if task.status == "pending":
                pending.append(task)
            pending.extend(self.get_pending_tasks(task.subtasks))
        return pending

    def decompose_goal(self, goal: str, planner_agent: Any,
                       memory_snapshot: str = "", similar_past_problems: str = "",
                       known_weaknesses: str = "") -> Task:
        log_json("INFO", "decomposing_goal", details={"goal": goal})
        root = Task(id=f"goal_{int(time.time())}", title=goal)
        try:
            steps = planner_agent.plan(goal, memory_snapshot, similar_past_problems, known_weaknesses)
            for i, step in enumerate(steps):
                if isinstance(step, str) and not step.startswith("ERROR:"):
                    subtask = Task(
                        id=f"{root.id}_step_{i}",
                        title=step,
                        description=step,
                    )
                    root.subtasks.append(subtask)
            log_json("INFO", "decompose_goal_complete", details={"goal": goal, "subtask_count": len(root.subtasks)})
        except Exception as e:
            log_json("ERROR", "decompose_goal_failed", details={"goal": goal, "error": str(e)})
        return root
