"""Tests for aura/offline/executor.py — OfflineExecutor."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aura.offline.executor import OfflineExecutor
from aura.offline.models import CommandResult, CommandStatus, CommandPriority
from aura.offline.queue import CommandQueue


def _make_executor(tmp_path, online=True):
    queue = CommandQueue(db_path=str(tmp_path / "exec.db"))
    executor = OfflineExecutor(
        queue=queue,
        connectivity_check=lambda: online,
    )
    return executor, queue


# ---------------------------------------------------------------------------
# Init / register_handler
# ---------------------------------------------------------------------------

class TestOfflineExecutorInit:
    def test_default_is_online(self, tmp_path):
        queue = CommandQueue(db_path=str(tmp_path / "q.db"))
        executor = OfflineExecutor(queue=queue)
        assert executor._is_online() is True

    def test_register_handler_stored(self, tmp_path):
        executor, _ = _make_executor(tmp_path)
        handler = lambda: None
        executor.register_handler("cmd", handler)
        assert executor._handlers["cmd"] is handler

    def test_connectivity_check_respected(self, tmp_path):
        queue = CommandQueue(db_path=str(tmp_path / "q.db"))
        executor = OfflineExecutor(queue=queue, connectivity_check=lambda: False)
        assert executor._is_online() is False


# ---------------------------------------------------------------------------
# _do_execute
# ---------------------------------------------------------------------------

class TestDoExecute:
    async def test_no_handler_returns_failure(self, tmp_path):
        executor, _ = _make_executor(tmp_path)
        result = await executor._do_execute("unknown", (), {})
        assert result.success is False
        assert "No handler" in result.error

    async def test_sync_handler_called(self, tmp_path):
        executor, _ = _make_executor(tmp_path)
        executor.register_handler("greet", lambda name: f"hi {name}")
        result = await executor._do_execute("greet", ("alice",), {})
        assert result.success is True
        assert result.output == "hi alice"

    async def test_async_handler_called(self, tmp_path):
        executor, _ = _make_executor(tmp_path)

        async def async_greet(name):
            return f"async hi {name}"

        executor.register_handler("agreet", async_greet)
        result = await executor._do_execute("agreet", ("bob",), {})
        assert result.success is True
        assert result.output == "async hi bob"

    async def test_handler_exception_returns_failure(self, tmp_path):
        executor, _ = _make_executor(tmp_path)
        executor.register_handler("boom", lambda: (_ for _ in ()).throw(RuntimeError("kaboom")))
        result = await executor._do_execute("boom", (), {})
        assert result.success is False
        assert "kaboom" in result.error

    async def test_handler_kwargs_passed(self, tmp_path):
        executor, _ = _make_executor(tmp_path)
        received = {}

        def capture(**kwargs):
            received.update(kwargs)
            return "ok"

        executor.register_handler("kw_cmd", capture)
        result = await executor._do_execute("kw_cmd", (), {"x": 1, "y": 2})
        assert result.success is True
        assert received == {"x": 1, "y": 2}


# ---------------------------------------------------------------------------
# execute — online vs offline routing
# ---------------------------------------------------------------------------

class TestExecuteOnlineOffline:
    async def test_execute_online_runs_directly(self, tmp_path):
        executor, _ = _make_executor(tmp_path, online=True)
        executor.register_handler("ping", lambda: "pong")
        result = await executor.execute("ping")
        assert result.success is True
        assert result.output == "pong"
        assert result.queued_id is None

    async def test_execute_offline_queues_command(self, tmp_path):
        executor, queue = _make_executor(tmp_path, online=False)
        result = await executor.execute("deferred_cmd")
        assert result.success is True
        assert result.output["queued"] is True
        assert result.queued_id is not None
        assert await queue.size() == 1

    async def test_execute_offline_no_handler_still_queues(self, tmp_path):
        executor, queue = _make_executor(tmp_path, online=False)
        result = await executor.execute("any_cmd")
        assert result.success is True
        assert await queue.size() == 1


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------

class TestSync:
    async def test_sync_offline_returns_zero(self, tmp_path):
        executor, _ = _make_executor(tmp_path, online=False)
        count = await executor.sync()
        assert count == 0

    async def test_sync_processes_pending_commands(self, tmp_path):
        executor, queue = _make_executor(tmp_path, online=True)
        executor.register_handler("do_work", lambda: "done")
        # Queue a command while offline
        offline_executor = OfflineExecutor(
            queue=queue,
            connectivity_check=lambda: False,
        )
        offline_executor.register_handler("do_work", lambda: "done")
        await offline_executor.execute("do_work")
        assert await queue.size() == 1
        # Now sync online
        processed = await executor.sync()
        assert processed == 1
        assert await queue.size() == 0

    async def test_sync_failed_command_marked_failed(self, tmp_path):
        executor, queue = _make_executor(tmp_path, online=True)
        # Queue a command for which there's no handler
        offline_executor = OfflineExecutor(
            queue=queue,
            connectivity_check=lambda: False,
        )
        await offline_executor.execute("unknown_cmd")
        assert await queue.size() == 1
        processed = await executor.sync()
        assert processed == 0
        # Command should still be in queue but with FAILED status
        pending = await queue.get_pending()
        assert len(pending) == 0  # FAILED not returned by get_pending


# ---------------------------------------------------------------------------
# sync_loop lifecycle
# ---------------------------------------------------------------------------

class TestSyncLoop:
    async def test_start_and_stop_loop(self, tmp_path):
        executor, _ = _make_executor(tmp_path)
        await executor.start_sync_loop()
        assert executor._running is True
        assert executor._task is not None
        await executor.stop_sync_loop()
        assert executor._running is False
        assert executor._task is None

    async def test_stop_without_start_is_safe(self, tmp_path):
        executor, _ = _make_executor(tmp_path)
        await executor.stop_sync_loop()  # Should not raise
