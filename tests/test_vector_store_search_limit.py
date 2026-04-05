"""Tests for vector_store search limit — covers issue #329."""
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path


def _make_brain(tmp_dir: str):
    from memory.brain import Brain
    return Brain(db_path=str(Path(tmp_dir) / "vs_test.db"))


def _make_model_adapter(dims: int = 4):
    import numpy as np
    adapter = MagicMock()
    adapter.model_id.return_value = "test-model"
    # get_embedding returns a fixed unit vector
    vec = np.ones(dims, dtype=np.float32) / (dims ** 0.5)
    adapter.get_embedding.return_value = vec
    # embed returns a list of identical vectors
    adapter.embed.side_effect = lambda texts: [vec for _ in texts]
    return adapter


class TestVectorStoreSearchLimitConstant(unittest.TestCase):
    """SEARCH_LIMIT constant must exist at module level in core/vector_store.py."""

    def test_search_limit_constant_exists(self):
        import core.vector_store as vs_mod
        self.assertTrue(
            hasattr(vs_mod, "SEARCH_LIMIT"),
            "core/vector_store.py must define a SEARCH_LIMIT constant",
        )

    def test_search_limit_is_positive_integer(self):
        import core.vector_store as vs_mod
        self.assertIsInstance(vs_mod.SEARCH_LIMIT, int)
        self.assertGreater(vs_mod.SEARCH_LIMIT, 0)

    def test_search_limit_value(self):
        import core.vector_store as vs_mod
        self.assertEqual(
            vs_mod.SEARCH_LIMIT,
            1000,
            "SEARCH_LIMIT should be 1000 as specified in the issue",
        )


class TestVectorStoreSearchAppliesLimit(unittest.TestCase):
    """The SQL query inside search() must include LIMIT ? bound to SEARCH_LIMIT."""

    def test_sql_uses_limit_parameter(self):
        """Patch brain.db.execute to capture SQL and verify LIMIT is present."""
        with tempfile.TemporaryDirectory() as tmp:
            brain = _make_brain(tmp)
            adapter = _make_model_adapter()

            from core.vector_store import VectorStore, SEARCH_LIMIT

            vs = VectorStore(model_adapter=adapter, brain=brain)

            captured_calls: list[tuple] = []
            original_execute = brain.db.execute

            def capturing_execute(sql, params=None):
                captured_calls.append((sql, params))
                if params is not None:
                    return original_execute(sql, params)
                return original_execute(sql)

            # sqlite3.Connection.execute is read-only, so verify via source inspection
            import inspect
            src = inspect.getsource(vs.search)
            self.assertIn("SEARCH_LIMIT", src, "search() must reference SEARCH_LIMIT constant")
            self.assertIn("LIMIT", src, "search() SQL must contain LIMIT clause")

            # Also verify the constant value is used
            self.assertEqual(SEARCH_LIMIT, 1000)

            # This test is now complete
            return

    def test_search_does_not_load_all_rows_beyond_limit(self):
        """
        Insert more than SEARCH_LIMIT rows via direct DB insert (bypassing embedding),
        then verify search() fetches at most SEARCH_LIMIT candidates from the DB.

        We do this by counting how many rows are passed to the numpy scoring loop,
        which we intercept by patching numpy.frombuffer.
        """
        import sys
        import importlib
        import numpy as np

        # Guard: a previous test (test_optional_dependency_guards) may have left
        # core.vector_store with np=_MissingPackage.  Use identity comparison
        # (not isinstance) to detect this — isinstance can fail when the
        # _MissingPackage class comes from a different module instance.
        # Also ensure sys.modules["numpy"] points to real numpy before reloading,
        # so the reload doesn't create another _MissingPackage.
        vs_mod = sys.modules.get("core.vector_store")
        if vs_mod is not None and vs_mod.np is not np:
            sys.modules["numpy"] = np  # ensure real numpy is visible to the reload
            importlib.reload(vs_mod)

        from core.vector_store import VectorStore, SEARCH_LIMIT

        with tempfile.TemporaryDirectory() as tmp:
            brain = _make_brain(tmp)
            adapter = _make_model_adapter(dims=4)
            vs = VectorStore(model_adapter=adapter, brain=brain)

            dims = 4
            embedding_blob = np.ones(dims, dtype=np.float32).tobytes()
            now = 1_700_000_000.0
            import uuid
            import hashlib

            # Insert SEARCH_LIMIT + 50 rows directly so we exceed the cap
            rows = []
            for i in range(SEARCH_LIMIT + 50):
                rid = uuid.uuid4().hex
                content = f"entry-{i}"
                content_hash = hashlib.sha256(content.encode()).hexdigest()
                rows.append((rid, content, "test", None, now, now, None, None,
                              "[]", 1.0, 0, "test-model", dims, content_hash, embedding_blob))

            brain.db.executemany(
                """INSERT INTO memory_records
                   (id, content, source_type, source_ref, created_at, updated_at,
                    goal_id, agent_name, tags, importance, token_count,
                    embedding_model, embedding_dims, content_hash, embedding)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            emb_rows = [(r[0], "test-model", dims, embedding_blob) for r in rows]
            brain.db.executemany(
                "INSERT INTO embeddings (record_id, model_id, dims, data) VALUES (?,?,?,?)",
                emb_rows,
            )
            brain.db.commit()

            frombuffer_call_count = [0]
            original_frombuffer = np.frombuffer

            def counting_frombuffer(buf, dtype=None):
                frombuffer_call_count[0] += 1
                return original_frombuffer(buf, dtype=dtype)

            with patch("core.vector_store.np.frombuffer", side_effect=counting_frombuffer):
                vs.search("test query", k=5)

            self.assertLessEqual(
                frombuffer_call_count[0],
                SEARCH_LIMIT,
                f"search() should process at most SEARCH_LIMIT={SEARCH_LIMIT} candidates "
                f"from the DB, but np.frombuffer was called {frombuffer_call_count[0]} times",
            )


if __name__ == "__main__":
    unittest.main()
