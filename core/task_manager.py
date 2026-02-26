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

class TaskManager:
    """
    Unified Control Plane: Manages task hierarchies via MemoryController.
    """
    def __init__(self, persistence_path=None):
        self.persistence_path = Path(persistence_path or config.get("memory_persistence_path", "memory/task_hierarchy_v2.json"))
        self.root_tasks: List[Task] = []
        self.load()

    def add_task(self, task: Task):
        self.root_tasks.append(task)
        memory_controller.store(MemoryTier.PROJECT, {"type": "task", "task_id": task.id, "title": task.title})
        self.save()

    def save(self):
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        data = [t.to_dict() for t in self.root_tasks]
        with open(self.persistence_path, 'w') as f:
            json.dump(data, f, indent=4)
        log_json("INFO", "task_hierarchy_saved", details={"root_count": len(self.root_tasks)})

    def load(self):
        if self.persistence_path.exists():
            try:
                with open(self.persistence_path, 'r') as f:
                    data = json.load(f)
                    self.root_tasks = [Task.from_dict(t) for t in data]
            except Exception as e:
                log_json("WARN", "task_hierarchy_load_failed", details={"error": str(e)})

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
