"""Unit tests for core/checkpoint.py -- State Checkpointing System (Issue #434)."""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from core.checkpoint import (
    Checkpoint,
    CheckpointCorruptedError,
    CheckpointManager,
    CheckpointNotFoundError,
    CheckpointSummary,
    SQLiteCheckpointStorage,
)


class _TempStorageMixin:
    """Helper that creates a temporary SQLiteCheckpointStorage per test."""

    def _make_storage(self, tmp_dir: str) -> SQLiteCheckpointStorage:
        return SQLiteCheckpointStorage(
            db_path=str(Path(tmp_dir) / "test_checkpoints.db")
        )

    def _make_manager(
        self, tmp_dir: str, checkpoint_every: int = 1
    ) -> CheckpointManager:
        storage = self._make_storage(tmp_dir)
        return CheckpointManager(storage=storage, checkpoint_every=checkpoint_every)


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


class TestCheckpointDataclass(unittest.TestCase):
    def test_defaults(self):
        cp = Checkpoint(
            checkpoint_id="abc",
            workflow_name="wf",
            node_id="n1",
            state_data={"x": 1},
            completed_nodes=["n0"],
            pending_nodes=["n2"],
        )
        self.assertEqual(cp.memory_snapshot, {})
        self.assertEqual(cp.metadata, {})
        self.assertEqual(cp.checksum, "")
        self.assertEqual(cp.schema_version, 1)
        self.assertEqual(cp.created_at, 0.0)
        self.assertIsNone(cp.parent_id)

    def test_summary_fields(self):
        s = CheckpointSummary(
            checkpoint_id="id1",
            workflow_name="wf",
            node_id="n1",
            created_at=1000.0,
            state_size_bytes=42,
        )
        self.assertIsNone(s.parent_id)
        self.assertEqual(s.state_size_bytes, 42)


# ---------------------------------------------------------------------------
# SQLiteCheckpointStorage tests
# ---------------------------------------------------------------------------


class TestSQLiteCheckpointStorage(unittest.TestCase, _TempStorageMixin):
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = self._make_storage(tmp)
            cp = Checkpoint(
                checkpoint_id="cp1",
                workflow_name="test_wf",
                node_id="node_a",
                state_data={"key": "value", "num": 42},
                completed_nodes=["node_a"],
                pending_nodes=["node_b", "node_c"],
                memory_snapshot={"mem": "data"},
                metadata={"step_count": 5},
                checksum="abc123",
                schema_version=1,
                created_at=time.time(),
                parent_id=None,
            )
            returned_id = storage.save(cp)
            self.assertEqual(returned_id, "cp1")

            loaded = storage.load("cp1")
            self.assertEqual(loaded.checkpoint_id, "cp1")
            self.assertEqual(loaded.workflow_name, "test_wf")
            self.assertEqual(loaded.state_data, {"key": "value", "num": 42})
            self.assertEqual(loaded.completed_nodes, ["node_a"])
            self.assertEqual(loaded.pending_nodes, ["node_b", "node_c"])
            self.assertEqual(loaded.memory_snapshot, {"mem": "data"})
            self.assertEqual(loaded.metadata, {"step_count": 5})
            self.assertEqual(loaded.checksum, "abc123")
            storage.close()

    def test_load_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = self._make_storage(tmp)
            with self.assertRaises(CheckpointNotFoundError):
                storage.load("nonexistent")
            storage.close()

    def test_list_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = self._make_storage(tmp)
            for i in range(5):
                cp = Checkpoint(
                    checkpoint_id=f"cp{i}",
                    workflow_name="wf_a" if i < 3 else "wf_b",
                    node_id=f"n{i}",
                    state_data={"i": i},
                    completed_nodes=[],
                    pending_nodes=[],
                    checksum="x",
                    created_at=1000.0 + i,
                )
                storage.save(cp)

            # List all
            summaries = storage.list()
            self.assertEqual(len(summaries), 5)
            # Newest first
            self.assertEqual(summaries[0].checkpoint_id, "cp4")

            # Filter by workflow
            wf_a = storage.list(workflow_name="wf_a")
            self.assertEqual(len(wf_a), 3)

            # Limit
            limited = storage.list(limit=2)
            self.assertEqual(len(limited), 2)
            storage.close()

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = self._make_storage(tmp)
            cp = Checkpoint(
                checkpoint_id="del1",
                workflow_name="wf",
                node_id="n",
                state_data={},
                completed_nodes=[],
                pending_nodes=[],
                checksum="x",
                created_at=time.time(),
            )
            storage.save(cp)
            self.assertTrue(storage.delete("del1"))
            self.assertFalse(storage.delete("del1"))
            with self.assertRaises(CheckpointNotFoundError):
                storage.load("del1")
            storage.close()

    def test_gc(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = self._make_storage(tmp)
            # Create 15 checkpoints for workflow "wf_gc"
            for i in range(15):
                cp = Checkpoint(
                    checkpoint_id=f"gc{i}",
                    workflow_name="wf_gc",
                    node_id=f"n{i}",
                    state_data={"step": i},
                    completed_nodes=[],
                    pending_nodes=[],
                    checksum="x",
                    created_at=1000.0 + i,
                )
                storage.save(cp)

            deleted = storage.gc(keep_last=5)
            self.assertEqual(deleted, 10)
            remaining = storage.list(workflow_name="wf_gc")
            self.assertEqual(len(remaining), 5)
            # The 5 newest should remain
            ids = {s.checkpoint_id for s in remaining}
            for i in range(10, 15):
                self.assertIn(f"gc{i}", ids)
            storage.close()

    def test_gc_multiple_workflows(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = self._make_storage(tmp)
            # 8 checkpoints in wf_a, 4 in wf_b
            for i in range(8):
                storage.save(
                    Checkpoint(
                        checkpoint_id=f"a{i}",
                        workflow_name="wf_a",
                        node_id=f"n{i}",
                        state_data={},
                        completed_nodes=[],
                        pending_nodes=[],
                        checksum="x",
                        created_at=1000.0 + i,
                    )
                )
            for i in range(4):
                storage.save(
                    Checkpoint(
                        checkpoint_id=f"b{i}",
                        workflow_name="wf_b",
                        node_id=f"n{i}",
                        state_data={},
                        completed_nodes=[],
                        pending_nodes=[],
                        checksum="x",
                        created_at=2000.0 + i,
                    )
                )

            deleted = storage.gc(keep_last=3)
            # wf_a: 8-3=5 deleted, wf_b: 4-3=1 deleted
            self.assertEqual(deleted, 6)
            self.assertEqual(len(storage.list(workflow_name="wf_a")), 3)
            self.assertEqual(len(storage.list(workflow_name="wf_b")), 3)
            storage.close()

    def test_save_with_null_optional_fields(self):
        """memory_snapshot and metadata can be empty."""
        with tempfile.TemporaryDirectory() as tmp:
            storage = self._make_storage(tmp)
            cp = Checkpoint(
                checkpoint_id="null_fields",
                workflow_name="wf",
                node_id="n",
                state_data={"a": 1},
                completed_nodes=[],
                pending_nodes=[],
                checksum="x",
                created_at=time.time(),
            )
            storage.save(cp)
            loaded = storage.load("null_fields")
            self.assertEqual(loaded.memory_snapshot, {})
            self.assertEqual(loaded.metadata, {})
            storage.close()

    def test_state_size_bytes_in_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = self._make_storage(tmp)
            state = {"big_key": "a" * 1000}
            state_json = json.dumps(state, sort_keys=True)
            expected_size = len(state_json.encode("utf-8"))
            cp = Checkpoint(
                checkpoint_id="size1",
                workflow_name="wf",
                node_id="n",
                state_data=state,
                completed_nodes=[],
                pending_nodes=[],
                checksum="x",
                created_at=time.time(),
            )
            storage.save(cp)
            summaries = storage.list()
            self.assertEqual(summaries[0].state_size_bytes, expected_size)
            storage.close()


# ---------------------------------------------------------------------------
# CheckpointManager tests
# ---------------------------------------------------------------------------


class TestCheckpointManager(unittest.TestCase, _TempStorageMixin):
    def test_create_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._make_manager(tmp)
            cp_id = mgr.create_checkpoint(
                workflow_name="wf1",
                node_id="step_3",
                state={"progress": 0.5},
                completed_nodes=["step_1", "step_2", "step_3"],
                pending_nodes=["step_4"],
                metadata={"step_count": 3},
            )
            self.assertIsInstance(cp_id, str)
            self.assertEqual(len(cp_id), 32)  # UUID hex

            restored = mgr.resume(cp_id)
            self.assertEqual(restored.workflow_name, "wf1")
            self.assertEqual(restored.node_id, "step_3")
            self.assertEqual(restored.state_data, {"progress": 0.5})
            self.assertEqual(
                restored.completed_nodes, ["step_1", "step_2", "step_3"]
            )
            self.assertEqual(restored.pending_nodes, ["step_4"])

    def test_resume_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._make_manager(tmp)
            with self.assertRaises(CheckpointNotFoundError):
                mgr.resume("does_not_exist")

    def test_integrity_check_on_resume(self):
        """Tampering with state_data after save must cause CheckpointCorruptedError."""
        with tempfile.TemporaryDirectory() as tmp:
            storage = self._make_storage(tmp)
            mgr = CheckpointManager(storage=storage)
            cp_id = mgr.create_checkpoint(
                workflow_name="wf",
                node_id="n",
                state={"key": "original"},
                completed_nodes=[],
                pending_nodes=[],
            )
            # Tamper directly in the DB
            import sqlite3 as _sq

            conn = _sq.connect(str(Path(tmp) / "test_checkpoints.db"))
            conn.execute(
                "UPDATE checkpoints SET state_json = ? WHERE id = ?",
                (json.dumps({"key": "tampered"}), cp_id),
            )
            conn.commit()
            conn.close()

            with self.assertRaises(CheckpointCorruptedError):
                mgr.resume(cp_id)
            storage.close()

    def test_fork(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._make_manager(tmp)
            original_id = mgr.create_checkpoint(
                workflow_name="wf_fork",
                node_id="n5",
                state={"data": [1, 2, 3]},
                completed_nodes=["n1", "n2", "n5"],
                pending_nodes=["n6"],
            )
            forked_id = mgr.fork(original_id)
            self.assertNotEqual(forked_id, original_id)

            forked = mgr.resume(forked_id)
            self.assertEqual(forked.parent_id, original_id)
            self.assertEqual(forked.state_data, {"data": [1, 2, 3]})
            self.assertEqual(forked.workflow_name, "wf_fork")
            self.assertIn("forked_from", forked.metadata)

    def test_rollback_is_resume_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._make_manager(tmp)
            cp_id = mgr.create_checkpoint(
                workflow_name="wf",
                node_id="n2",
                state={"v": 99},
                completed_nodes=["n1", "n2"],
                pending_nodes=["n3"],
            )
            via_resume = mgr.resume(cp_id)
            via_rollback = mgr.rollback(cp_id)
            self.assertEqual(via_resume.state_data, via_rollback.state_data)
            self.assertEqual(via_resume.checkpoint_id, via_rollback.checkpoint_id)

    def test_list_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._make_manager(tmp)
            for i in range(3):
                mgr.create_checkpoint(
                    workflow_name="wf_list",
                    node_id=f"n{i}",
                    state={"i": i},
                    completed_nodes=[],
                    pending_nodes=[],
                )
            summaries = mgr.list_checkpoints(workflow_name="wf_list")
            self.assertEqual(len(summaries), 3)

    def test_delete_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._make_manager(tmp)
            cp_id = mgr.create_checkpoint(
                workflow_name="wf",
                node_id="n",
                state={},
                completed_nodes=[],
                pending_nodes=[],
            )
            self.assertTrue(mgr.delete_checkpoint(cp_id))
            self.assertFalse(mgr.delete_checkpoint(cp_id))

    def test_gc_via_manager(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._make_manager(tmp)
            for i in range(12):
                mgr.create_checkpoint(
                    workflow_name="wf_gc",
                    node_id=f"n{i}",
                    state={"i": i},
                    completed_nodes=[],
                    pending_nodes=[],
                )
            deleted = mgr.gc(keep_last=5)
            self.assertEqual(deleted, 7)
            remaining = mgr.list_checkpoints(workflow_name="wf_gc")
            self.assertEqual(len(remaining), 5)

    def test_checksum_deterministic(self):
        mgr = CheckpointManager.__new__(CheckpointManager)
        state = {"z": 1, "a": 2, "m": [3, 4]}
        c1 = mgr._compute_checksum(state)
        c2 = mgr._compute_checksum(state)
        self.assertEqual(c1, c2)
        # Different order same data
        state2 = {"a": 2, "z": 1, "m": [3, 4]}
        c3 = mgr._compute_checksum(state2)
        self.assertEqual(c1, c3)

    def test_checksum_differs_for_different_state(self):
        mgr = CheckpointManager.__new__(CheckpointManager)
        c1 = mgr._compute_checksum({"a": 1})
        c2 = mgr._compute_checksum({"a": 2})
        self.assertNotEqual(c1, c2)


# ---------------------------------------------------------------------------
# on_node_complete / checkpoint_every tests
# ---------------------------------------------------------------------------


class TestOnNodeComplete(unittest.TestCase, _TempStorageMixin):
    def test_checkpoint_every_1(self):
        """Every node completion triggers a checkpoint."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._make_manager(tmp, checkpoint_every=1)
            results = []
            for i in range(5):
                r = mgr.on_node_complete(
                    workflow_name="wf",
                    node_id=f"n{i}",
                    state={"step": i},
                    completed_nodes=[f"n{j}" for j in range(i + 1)],
                    pending_nodes=[],
                )
                results.append(r)
            self.assertTrue(all(r is not None for r in results))
            self.assertEqual(len(mgr.list_checkpoints(workflow_name="wf")), 5)

    def test_checkpoint_every_3(self):
        """Only every 3rd node triggers a checkpoint."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._make_manager(tmp, checkpoint_every=3)
            results = []
            for i in range(9):
                r = mgr.on_node_complete(
                    workflow_name="wf",
                    node_id=f"n{i}",
                    state={"step": i},
                    completed_nodes=[],
                    pending_nodes=[],
                )
                results.append(r)
            # Checkpoints at node_counter 3, 6, 9
            created = [r for r in results if r is not None]
            self.assertEqual(len(created), 3)
            skipped = [r for r in results if r is None]
            self.assertEqual(len(skipped), 6)

    def test_checkpoint_every_minimum_clamped(self):
        """checkpoint_every < 1 is clamped to 1."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._make_manager(tmp, checkpoint_every=0)
            r = mgr.on_node_complete(
                workflow_name="wf",
                node_id="n0",
                state={},
                completed_nodes=[],
                pending_nodes=[],
            )
            self.assertIsNotNone(r)


# ---------------------------------------------------------------------------
# Thread safety tests
# ---------------------------------------------------------------------------


class TestCheckpointThreadSafety(unittest.TestCase, _TempStorageMixin):
    def test_concurrent_saves(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = self._make_storage(tmp)
            errors: list = []
            count_per_thread = 20

            def writer(prefix: str):
                try:
                    for i in range(count_per_thread):
                        cp = Checkpoint(
                            checkpoint_id=f"{prefix}_{i}",
                            workflow_name="concurrent_wf",
                            node_id=f"n{i}",
                            state_data={"t": prefix, "i": i},
                            completed_nodes=[],
                            pending_nodes=[],
                            checksum="x",
                            created_at=time.time(),
                        )
                        storage.save(cp)
                except Exception as exc:
                    errors.append(exc)

            t1 = threading.Thread(target=writer, args=("A",))
            t2 = threading.Thread(target=writer, args=("B",))
            t1.start()
            t2.start()
            t1.join(timeout=15)
            t2.join(timeout=15)

            self.assertFalse(errors, f"Concurrent saves raised: {errors}")
            total = len(storage.list(limit=100))
            self.assertEqual(total, count_per_thread * 2)
            storage.close()

    def test_concurrent_reads_and_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = self._make_storage(tmp)
            # Seed data
            for i in range(10):
                storage.save(
                    Checkpoint(
                        checkpoint_id=f"seed_{i}",
                        workflow_name="rw_wf",
                        node_id=f"n{i}",
                        state_data={"i": i},
                        completed_nodes=[],
                        pending_nodes=[],
                        checksum="x",
                        created_at=1000.0 + i,
                    )
                )
            errors: list = []

            def writer():
                try:
                    for i in range(20):
                        storage.save(
                            Checkpoint(
                                checkpoint_id=f"rw_{i}",
                                workflow_name="rw_wf",
                                node_id=f"n{i}",
                                state_data={"i": i},
                                completed_nodes=[],
                                pending_nodes=[],
                                checksum="x",
                                created_at=2000.0 + i,
                            )
                        )
                except Exception as exc:
                    errors.append(exc)

            def reader():
                try:
                    for _ in range(20):
                        storage.list(workflow_name="rw_wf")
                except Exception as exc:
                    errors.append(exc)

            threads = [
                threading.Thread(target=writer),
                threading.Thread(target=reader),
                threading.Thread(target=reader),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=15)

            self.assertFalse(errors, f"Concurrent r/w raised: {errors}")
            storage.close()


# ---------------------------------------------------------------------------
# Memory snapshot and metadata tests
# ---------------------------------------------------------------------------


class TestMemorySnapshotAndMetadata(unittest.TestCase, _TempStorageMixin):
    def test_memory_snapshot_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._make_manager(tmp)
            snapshot = {
                "recent_memories": ["mem1", "mem2"],
                "weaknesses": ["w1"],
                "kv": {"key1": "val1"},
            }
            cp_id = mgr.create_checkpoint(
                workflow_name="wf",
                node_id="n",
                state={"s": 1},
                completed_nodes=[],
                pending_nodes=[],
                memory_snapshot=snapshot,
            )
            loaded = mgr.resume(cp_id)
            self.assertEqual(loaded.memory_snapshot, snapshot)

    def test_metadata_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._make_manager(tmp)
            meta = {"step_count": 42, "token_usage": 12345}
            cp_id = mgr.create_checkpoint(
                workflow_name="wf",
                node_id="n",
                state={},
                completed_nodes=[],
                pending_nodes=[],
                metadata=meta,
            )
            loaded = mgr.resume(cp_id)
            self.assertEqual(loaded.metadata["step_count"], 42)
            self.assertEqual(loaded.metadata["token_usage"], 12345)
            # created_ts_iso is added automatically
            self.assertIn("created_ts_iso", loaded.metadata)

    def test_parent_id_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._make_manager(tmp)
            parent = mgr.create_checkpoint(
                workflow_name="wf",
                node_id="n1",
                state={"v": 1},
                completed_nodes=["n1"],
                pending_nodes=[],
            )
            child = mgr.create_checkpoint(
                workflow_name="wf",
                node_id="n2",
                state={"v": 2},
                completed_nodes=["n1", "n2"],
                pending_nodes=[],
                parent_id=parent,
            )
            loaded = mgr.resume(child)
            self.assertEqual(loaded.parent_id, parent)


if __name__ == "__main__":
    unittest.main()
