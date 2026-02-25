import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
from pathlib import Path
from core.logging_utils import log_json

@dataclass
class Task:
    id: str
    title: str
    status: str = "pending" # pending, in_progress, completed, failed
    description: str = ""
    subtasks: List['Task'] = field(default_factory=list)
    result: Optional[str] = None

    def to_dict(self):
        """Recursively convert Task to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "description": self.description,
            "subtasks": [st.to_dict() for st in self.subtasks],
            "result": self.result
        }

    @classmethod
    def from_dict(cls, data):
        """Recursively create Task from dictionary."""
        subtasks = [cls.from_dict(st) for st in data.get("subtasks", [])]
        return cls(
            id=data["id"],
            title=data["title"],
            status=data.get("status", "pending"),
            description=data.get("description", ""),
            subtasks=subtasks,
            result=data.get("result")
        )

    def display(self, indent=0):
        """Return a formatted string representation of the task hierarchy."""
        status_icon = "✅" if self.status == "completed" else "❌" if self.status == "failed" else "⏳" if self.status == "pending" else "🔄"
        lines = [f"{'  ' * indent}{status_icon} [{self.id}] {self.title}"]
        for st in self.subtasks:
            lines.append(st.display(indent + 1))
        return "\n".join(lines)

class TaskManager:
    """
    Manages a hierarchical graph of tasks and sub-tasks for AURA projects.
    """

    def __init__(self, persistence_path="memory/task_hierarchy.json"):
        self.persistence_path = Path(persistence_path)
        self.root_tasks: List[Task] = self._load()

    def add_task(self, task: Task):
        """Add a root task to the hierarchy."""
        self.root_tasks.append(task)
        self.save()
        log_json("INFO", "task_hierarchy_added", details={"task_id": task.id, "title": task.title})

    def decompose_goal(self, goal: str, planner):
        """Use a planner agent to decompose a goal into a hierarchy of tasks."""
        log_json("INFO", "decomposing_goal", details={"goal": goal})
        
        # Get context from brain for planning
        memory_snapshot = planner.brain.reflect()
        similar_past = "" # Placeholder
        known_weaknesses = "\n".join(planner.brain.recall_weaknesses())
        
        plan_steps = planner.plan(goal, memory_snapshot, similar_past, known_weaknesses)
        
        root_task = Task(id=f"goal_{int(time.time())}", title=goal)
        for i, step in enumerate(plan_steps):
            if step.startswith("ERROR"):
                continue
            subtask = Task(id=f"{root_task.id}_{i}", title=step)
            root_task.subtasks.append(subtask)
        
        self.add_task(root_task)
        return root_task

    def execute_next(self):
        """Retrieve the next pending task from the hierarchy."""
        pending = self.get_pending_tasks()
        if pending:
            task = pending[0]
            task.status = "in_progress"
            self.save()
            return task
        return None

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

    def save(self):
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        data = [task.to_dict() for task in self.root_tasks]
        with open(self.persistence_path, 'w') as f:
            json.dump(data, f, indent=4)
        log_json("INFO", "task_hierarchy_saved", details={"path": str(self.persistence_path), "root_count": len(self.root_tasks)})

    def _load(self) -> List[Task]:
        if not self.persistence_path.exists():
            return []
        try:
            with open(self.persistence_path, 'r') as f:
                data = json.load(f)
                return [Task.from_dict(d) for d in data]
        except (json.JSONDecodeError, KeyError) as e:
            log_json("WARN", "task_hierarchy_load_failed", details={"error": str(e)})
            return []
