import os
import json
from pathlib import Path
from core.task_manager import TaskManager, Task

def test_task_hierarchy_persistence(tmp_path):
    persistence_file = tmp_path / "test_hierarchy.json"
    manager = TaskManager(persistence_path=str(persistence_file))
    
    # Create a hierarchy
    child = Task(id="sub-1", title="Subtask 1", description="A subtask")
    parent = Task(id="parent-1", title="Parent Task", description="A parent task", subtasks=[child])
    
    manager.add_task(parent)
    
    # Reload from disk
    new_manager = TaskManager(persistence_path=str(persistence_file))
    found_parent = new_manager.find_task("parent-1")
    
    assert found_parent is not None
    assert found_parent.title == "Parent Task"
    assert len(found_parent.subtasks) == 1
    assert found_parent.subtasks[0].id == "sub-1"

def test_find_task(tmp_path):
    persistence_file = tmp_path / "test_find.json"
    manager = TaskManager(persistence_path=str(persistence_file))
    child = Task(id="sub-1", title="Subtask 1")
    parent = Task(id="parent-1", title="Parent Task", subtasks=[child])
    manager.add_task(parent)
    
    assert manager.find_task("sub-1") == child
    assert manager.find_task("non-existent") is None

def test_get_pending_tasks(tmp_path):
    persistence_file = tmp_path / "test_pending.json"
    manager = TaskManager(persistence_path=str(persistence_file))
    t1 = Task(id="t1", title="T1", status="completed")
    t2 = Task(id="t2", title="T2", status="pending")
    t3 = Task(id="t3", title="T3", status="pending")
    t2.subtasks.append(t3)
    
    manager.add_task(t1)
    manager.add_task(t2)
    
    pending = manager.get_pending_tasks()
    assert len(pending) == 2
    assert t2 in pending
    assert t3 in pending
    assert t1 not in pending
