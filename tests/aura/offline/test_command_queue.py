"""Tests for aura/offline/queue.py — CommandQueue."""

import pytest
from aura.offline.queue import CommandQueue
from aura.offline.models import CommandPriority, CommandStatus, QueuedCommand


@pytest.fixture
def queue(tmp_path):
    return CommandQueue(db_path=str(tmp_path / "test_queue.db"))


def _cmd(command="run_test", priority=CommandPriority.NORMAL) -> QueuedCommand:
    return QueuedCommand(command=command, priority=priority)


class TestCommandQueueInit:
    def test_creates_db_file(self, tmp_path):
        db = tmp_path / "q.db"
        CommandQueue(db_path=str(db))
        assert db.exists()

    def test_parent_dir_created(self, tmp_path):
        db = tmp_path / "nested" / "dir" / "q.db"
        CommandQueue(db_path=str(db))
        assert db.exists()


class TestCommandQueueAdd:
    async def test_add_returns_id(self, queue):
        cmd = _cmd()
        result = await queue.add(cmd)
        assert result == cmd.id

    async def test_add_increments_size(self, queue):
        await queue.add(_cmd())
        size = await queue.size()
        assert size == 1

    async def test_add_multiple(self, queue):
        await queue.add(_cmd("cmd_a"))
        await queue.add(_cmd("cmd_b"))
        assert await queue.size() == 2


class TestCommandQueueGetPending:
    async def test_get_pending_empty(self, queue):
        result = await queue.get_pending()
        assert result == []

    async def test_get_pending_returns_commands(self, queue):
        await queue.add(_cmd("do_work"))
        pending = await queue.get_pending()
        assert len(pending) == 1
        assert pending[0].command == "do_work"

    async def test_get_pending_respects_limit(self, queue):
        for i in range(5):
            await queue.add(_cmd(f"cmd_{i}"))
        pending = await queue.get_pending(limit=2)
        assert len(pending) == 2

    async def test_get_pending_ordered_by_priority(self, queue):
        await queue.add(_cmd("low", CommandPriority.LOW))
        await queue.add(_cmd("critical", CommandPriority.CRITICAL))
        await queue.add(_cmd("normal", CommandPriority.NORMAL))
        pending = await queue.get_pending()
        # CRITICAL(1) < NORMAL(3) < LOW(4)
        assert pending[0].command == "critical"

    async def test_non_pending_not_returned(self, queue):
        cmd = _cmd("done")
        await queue.add(cmd)
        await queue.update_status(cmd.id, CommandStatus.COMPLETED)
        pending = await queue.get_pending()
        assert len(pending) == 0


class TestCommandQueueUpdateStatus:
    async def test_update_to_processing(self, queue):
        cmd = _cmd()
        await queue.add(cmd)
        await queue.update_status(cmd.id, CommandStatus.PROCESSING)
        pending = await queue.get_pending()
        assert len(pending) == 0

    async def test_update_to_failed(self, queue):
        cmd = _cmd()
        await queue.add(cmd)
        await queue.update_status(cmd.id, CommandStatus.FAILED)
        pending = await queue.get_pending()
        assert len(pending) == 0


class TestCommandQueueRemove:
    async def test_remove_existing_returns_true(self, queue):
        cmd = _cmd()
        await queue.add(cmd)
        result = await queue.remove(cmd.id)
        assert result is True

    async def test_remove_nonexistent_returns_false(self, queue):
        result = await queue.remove("nonexistent_id")
        assert result is False

    async def test_remove_decrements_size(self, queue):
        cmd = _cmd()
        await queue.add(cmd)
        await queue.remove(cmd.id)
        assert await queue.size() == 0


class TestCommandQueueClear:
    async def test_clear_empties_queue(self, queue):
        for i in range(3):
            await queue.add(_cmd(f"cmd_{i}"))
        await queue.clear()
        assert await queue.size() == 0

    async def test_clear_empty_queue_no_crash(self, queue):
        await queue.clear()
        assert await queue.size() == 0
