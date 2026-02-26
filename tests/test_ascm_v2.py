"""Tests for ASCM v2: VectorStoreV2, EmbeddingProvider, ContextBudgetManager."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
import time
import uuid

import pytest

os.environ.setdefault("AURA_SKIP_CHDIR", "1")

from memory.vector_store_v2 import MemoryRecord, RetrievalQuery, SearchHit, VectorStoreV2
from memory.embedding_provider import (
    LocalEmbeddingProvider,
    OpenAIEmbeddingProvider,
    get_default_provider,
)
from core.context_budget import ContextBudgetManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tmp_store() -> VectorStoreV2:
    """Create a VectorStoreV2 backed by a temp file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = VectorStoreV2(db_path=path)
    return store


def _make_record(**kwargs) -> MemoryRecord:
    defaults = dict(
        content="Hello world test content",
        source_type="memory",
        source_ref="tests/test_ascm_v2.py",
    )
    defaults.update(kwargs)
    return MemoryRecord(**defaults)


# ===========================================================================
# TestMemoryRecord
# ===========================================================================

class TestMemoryRecord:
    def test_id_auto_generated(self):
        r = MemoryRecord(content="test")
        assert r.id and len(r.id) > 0

    def test_content_hash_auto_set(self):
        r = MemoryRecord(content="hello")
        expected = hashlib.sha256("hello".encode()).hexdigest()
        assert r.content_hash == expected

    def test_content_hash_not_overwritten_if_set(self):
        r = MemoryRecord(content="hello", content_hash="manual")
        assert r.content_hash == "manual"

    def test_token_count_auto_set(self):
        r = MemoryRecord(content="a" * 100)
        assert r.token_count == 25

    def test_token_count_min_one(self):
        r = MemoryRecord(content="x")
        assert r.token_count >= 1

    def test_default_importance(self):
        r = MemoryRecord(content="x")
        assert r.importance == 1.0

    def test_default_tags_empty_list(self):
        r = MemoryRecord(content="x")
        assert r.tags == []

    def test_optional_fields_none_by_default(self):
        r = MemoryRecord(content="x")
        assert r.goal_id is None
        assert r.agent_name is None

    def test_custom_importance(self):
        r = MemoryRecord(content="x", importance=0.5)
        assert r.importance == 0.5

    def test_created_at_reasonable(self):
        before = time.time()
        r = MemoryRecord(content="x")
        after = time.time()
        assert before <= r.created_at <= after


# ===========================================================================
# TestVectorStoreV2Schema
# ===========================================================================

class TestVectorStoreV2Schema:
    def test_tables_created(self):
        store = _tmp_store()
        conn = store._get_conn()
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "memory_records" in tables
        assert "embeddings" in tables
        store.close()

    def test_indexes_exist(self):
        store = _tmp_store()
        conn = store._get_conn()
        indexes = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()}
        assert any("goal_id" in i for i in indexes)
        assert any("source_type" in i for i in indexes)
        assert any("content_hash" in i for i in indexes)
        assert any("created_at" in i for i in indexes)
        store.close()

    def test_memory_records_columns(self):
        store = _tmp_store()
        conn = store._get_conn()
        cols = {r[1] for r in conn.execute("PRAGMA table_info(memory_records)").fetchall()}
        expected = {"id", "content", "source_type", "source_ref", "created_at", "updated_at",
                    "goal_id", "agent_name", "tags", "importance", "token_count",
                    "embedding_model", "content_hash"}
        assert expected <= cols
        store.close()

    def test_embeddings_columns(self):
        store = _tmp_store()
        conn = store._get_conn()
        cols = {r[1] for r in conn.execute("PRAGMA table_info(embeddings)").fetchall()}
        assert {"record_id", "model_id", "dims", "data"} <= cols
        store.close()

    def test_empty_count_on_new_store(self):
        store = _tmp_store()
        assert store.count() == 0
        store.close()


# ===========================================================================
# TestVectorStoreV2Upsert
# ===========================================================================

class TestVectorStoreV2Upsert:
    def test_upsert_returns_id(self):
        store = _tmp_store()
        r = _make_record()
        rid = store.upsert(r)
        assert rid and isinstance(rid, str)
        store.close()

    def test_count_increments(self):
        store = _tmp_store()
        store.upsert(_make_record(content="alpha"))
        store.upsert(_make_record(content="beta"))
        assert store.count() == 2
        store.close()

    def test_deduplication_by_content_hash(self):
        store = _tmp_store()
        r1 = _make_record(content="same content")
        r2 = _make_record(content="same content")
        store.upsert(r1)
        store.upsert(r2)
        assert store.count() == 1
        store.close()

    def test_update_on_same_hash(self):
        store = _tmp_store()
        r1 = _make_record(content="my content", importance=1.0)
        store.upsert(r1)
        r2 = _make_record(content="my content", importance=0.5)
        store.upsert(r2)
        assert store.count() == 1
        store.close()

    def test_different_content_creates_new_record(self):
        store = _tmp_store()
        store.upsert(_make_record(content="content A"))
        store.upsert(_make_record(content="content B"))
        assert store.count() == 2
        store.close()

    def test_upsert_with_embedding_stores_it(self):
        store = _tmp_store()
        r = _make_record(content="embed this", embedding_model="test-model")
        embedding = [0.1, 0.2, 0.3]
        rid = store.upsert(r, embedding=embedding)
        loaded = store._load_embedding(rid, "test-model")
        assert loaded is not None
        assert len(loaded) == 3
        store.close()

    def test_delete_works(self):
        store = _tmp_store()
        r = _make_record(content="delete me")
        rid = store.upsert(r)
        result = store.delete(rid)
        assert result is True
        assert store.count() == 0
        store.close()

    def test_delete_nonexistent_returns_false(self):
        store = _tmp_store()
        result = store.delete("nonexistent-id")
        assert result is False
        store.close()

    def test_tags_serialized_and_restored(self):
        store = _tmp_store()
        r = _make_record(content="tagged", tags=["a", "b", "c"])
        rid = store.upsert(r)
        records = store.filter_by()
        assert any(set(rec.tags) == {"a", "b", "c"} for rec in records)
        store.close()


# ===========================================================================
# TestVectorStoreV2Search
# ===========================================================================

class TestVectorStoreV2Search:
    def test_keyword_search_returns_results(self):
        store = _tmp_store()
        store.upsert(_make_record(content="python async coroutine event loop"))
        store.upsert(_make_record(content="database SQL query optimization"))
        q = RetrievalQuery(query_text="python coroutine", min_score=0.1)
        hits = store.search(q)
        assert len(hits) >= 1
        assert hits[0].record.content == "python async coroutine event loop"
        store.close()

    def test_search_returns_searchhit_objects(self):
        store = _tmp_store()
        store.upsert(_make_record(content="hello world"))
        q = RetrievalQuery(query_text="hello", min_score=0.1)
        hits = store.search(q)
        for hit in hits:
            assert isinstance(hit, SearchHit)
            assert isinstance(hit.record, MemoryRecord)
            assert isinstance(hit.score, float)
        store.close()

    def test_search_empty_store_returns_empty(self):
        store = _tmp_store()
        q = RetrievalQuery(query_text="anything")
        hits = store.search(q)
        assert hits == []
        store.close()

    def test_search_min_score_filters_low_matches(self):
        store = _tmp_store()
        store.upsert(_make_record(content="completely unrelated xyz"))
        q = RetrievalQuery(query_text="python code", min_score=0.9)
        hits = store.search(q)
        assert len(hits) == 0
        store.close()

    def test_search_sorted_by_score_desc(self):
        store = _tmp_store()
        store.upsert(_make_record(content="python python python"))
        store.upsert(_make_record(content="python code"))
        q = RetrievalQuery(query_text="python", min_score=0.1)
        hits = store.search(q)
        if len(hits) >= 2:
            assert hits[0].score >= hits[1].score
        store.close()

    def test_search_with_source_type_filter(self):
        store = _tmp_store()
        store.upsert(_make_record(content="file content", source_type="file"))
        store.upsert(_make_record(content="memory content", source_type="memory"))
        q = RetrievalQuery(
            query_text="content",
            min_score=0.1,
            filters={"source_type": "file"},
        )
        hits = store.search(q)
        assert all(h.record.source_type == "file" for h in hits)
        store.close()

    def test_search_dedupe_removes_duplicates(self):
        store = _tmp_store()
        store.upsert(_make_record(content="unique content here"))
        # The dedup by content_hash means only one record exists anyway
        q = RetrievalQuery(query_text="unique content", min_score=0.1, dedupe=True)
        hits = store.search(q)
        hashes = [h.record.content_hash for h in hits]
        assert len(hashes) == len(set(hashes))
        store.close()

    def test_filter_by_source_type(self):
        store = _tmp_store()
        store.upsert(_make_record(content="file A", source_type="file"))
        store.upsert(_make_record(content="mem A", source_type="memory"))
        records = store.filter_by(source_type="file")
        assert all(r.source_type == "file" for r in records)
        store.close()

    def test_filter_by_goal_id(self):
        store = _tmp_store()
        store.upsert(_make_record(content="goal1 content", goal_id="goal-1"))
        store.upsert(_make_record(content="goal2 content", goal_id="goal-2"))
        records = store.filter_by(goal_id="goal-1")
        assert all(r.goal_id == "goal-1" for r in records)
        store.close()

    def test_filter_by_agent_name(self):
        store = _tmp_store()
        store.upsert(_make_record(content="agent content", agent_name="planner"))
        store.upsert(_make_record(content="other content", agent_name="coder"))
        records = store.filter_by(agent_name="planner")
        assert all(r.agent_name == "planner" for r in records)
        store.close()

    def test_filter_by_tags(self):
        store = _tmp_store()
        store.upsert(_make_record(content="tagged A", tags=["important", "reviewed"]))
        store.upsert(_make_record(content="tagged B", tags=["draft"]))
        records = store.filter_by(tags=["important"])
        assert all("important" in r.tags for r in records)
        store.close()

    def test_filter_by_limit(self):
        store = _tmp_store()
        for i in range(10):
            store.upsert(_make_record(content=f"record number {i} content"))
        records = store.filter_by(limit=3)
        assert len(records) <= 3
        store.close()

    def test_search_with_embedding_cosine(self):
        store = _tmp_store()
        r = _make_record(content="vector search test", embedding_model="test-model")
        rid = store.upsert(r, embedding=[1.0, 0.0, 0.0])
        q = RetrievalQuery(query_text="vector search", min_score=0.0)
        hits = store.search(q, query_embedding=[0.9, 0.1, 0.0])
        assert any(h.record.id == rid for h in hits)
        store.close()


# ===========================================================================
# TestVectorStoreV2Migration
# ===========================================================================

class TestVectorStoreV2Migration:
    def test_migrate_from_v1_empty_store(self):
        store = _tmp_store()

        class EmptyV1:
            _cache_records = {}

        count = store.migrate_from_v1(EmptyV1())
        assert count == 0
        store.close()

    def test_migrate_from_v1_imports_records(self):
        store = _tmp_store()

        class MockV1Record:
            def __init__(self, content):
                self.content = content
                self.source_ref = "legacy/file.py"
                self.created_at = time.time()
                self.goal_id = None
                self.agent_name = None
                self.tags = []
                self.importance = 1.0

        class MockV1Store:
            _cache_records = {
                "r1": MockV1Record("legacy content one"),
                "r2": MockV1Record("legacy content two"),
            }

        count = store.migrate_from_v1(MockV1Store())
        assert count == 2
        assert store.count() == 2
        store.close()

    def test_migrate_marks_source_type_legacy(self):
        store = _tmp_store()

        class MockRec:
            content = "imported from v1"
            source_ref = "old/path.py"
            created_at = time.time()
            goal_id = None
            agent_name = None
            tags = []
            importance = 1.0

        class MockV1:
            _cache_records = {"r1": MockRec()}

        store.migrate_from_v1(MockV1())
        records = store.filter_by(source_type="legacy")
        assert len(records) == 1
        store.close()

    def test_migrate_deduplicates(self):
        store = _tmp_store()

        class MockRec:
            content = "same content"
            source_ref = ""
            created_at = time.time()
            goal_id = None
            agent_name = None
            tags = []
            importance = 1.0

        class MockV1:
            _cache_records = {"r1": MockRec(), "r2": MockRec()}

        store.migrate_from_v1(MockV1())
        # Deduplication: same content_hash → only 1 record
        assert store.count() == 1
        store.close()


# ===========================================================================
# TestLocalEmbeddingProvider
# ===========================================================================

class TestLocalEmbeddingProvider:
    def test_available_returns_true(self):
        p = LocalEmbeddingProvider()
        assert p.available() is True

    def test_embed_returns_list_of_lists(self):
        p = LocalEmbeddingProvider()
        result = p.embed(["hello world", "test text"])
        assert isinstance(result, list)
        assert len(result) == 2
        for vec in result:
            assert isinstance(vec, list)

    def test_embed_correct_dimensions(self):
        p = LocalEmbeddingProvider()
        result = p.embed(["sample text for dimension check"])
        assert len(result) == 1
        assert len(result[0]) == LocalEmbeddingProvider.DIMS

    def test_embed_returns_floats(self):
        p = LocalEmbeddingProvider()
        result = p.embed(["check types"])
        for val in result[0]:
            assert isinstance(val, float)

    def test_embed_unit_normalized(self):
        p = LocalEmbeddingProvider()
        result = p.embed(["normalize this vector"])
        vec = result[0]
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 0.01

    def test_embed_empty_list_returns_empty(self):
        p = LocalEmbeddingProvider()
        result = p.embed([])
        assert result == []

    def test_model_id_string(self):
        p = LocalEmbeddingProvider()
        assert isinstance(p.model_id(), str)
        assert len(p.model_id()) > 0

    def test_dimensions_returns_50(self):
        p = LocalEmbeddingProvider()
        assert p.dimensions() == 50

    def test_embed_single_text(self):
        p = LocalEmbeddingProvider()
        result = p.embed(["single"])
        assert len(result) == 1
        assert len(result[0]) == 50

    def test_embed_never_raises(self):
        p = LocalEmbeddingProvider()
        # Even with weird inputs, should not raise
        result = p.embed([""])
        assert isinstance(result, list)


# ===========================================================================
# TestOpenAIEmbeddingProvider
# ===========================================================================

class TestOpenAIEmbeddingProvider:
    def test_falls_back_to_local_without_adapter(self):
        p = OpenAIEmbeddingProvider(model_adapter=None)
        result = p.embed(["fallback test"])
        assert isinstance(result, list)
        assert len(result) == 1

    def test_available_returns_true(self):
        p = OpenAIEmbeddingProvider(model_adapter=None)
        assert p.available() is True

    def test_get_default_provider_no_adapter(self):
        p = get_default_provider()
        assert isinstance(p, LocalEmbeddingProvider)

    def test_get_default_provider_with_adapter(self):
        class FakeAdapter:
            pass
        p = get_default_provider(model_adapter=FakeAdapter())
        assert isinstance(p, OpenAIEmbeddingProvider)


# ===========================================================================
# TestContextBudgetManager
# ===========================================================================

class TestContextBudgetManager:
    def _make_hit(self, content: str, score: float = 0.8, source_ref: str = "src/file.py") -> SearchHit:
        rec = MemoryRecord(content=content, source_ref=source_ref, importance=1.0)
        return SearchHit(record=rec, score=score, explanation="test")

    def test_assemble_markdown_format(self):
        mgr = ContextBudgetManager()
        hits = [self._make_hit("First result", score=0.9, source_ref="a.py")]
        result = mgr.assemble(hits, budget_tokens=1000, format="markdown")
        assert "a.py" in result
        assert "score=" in result
        assert "First result" in result

    def test_assemble_plain_format(self):
        mgr = ContextBudgetManager()
        hits = [
            self._make_hit("Alpha content", score=0.9),
            self._make_hit("Beta content", score=0.7),
        ]
        result = mgr.assemble(hits, budget_tokens=1000, format="plain")
        assert "Alpha content" in result
        assert "Beta content" in result
        assert "score=" not in result

    def test_assemble_json_format(self):
        mgr = ContextBudgetManager()
        hits = [self._make_hit("JSON content", score=0.85, source_ref="test.py")]
        result = mgr.assemble(hits, budget_tokens=1000, format="json")
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["content"] == "JSON content"
        assert data[0]["source_ref"] == "test.py"
        assert abs(data[0]["score"] - 0.85) < 0.001

    def test_budget_respected(self):
        mgr = ContextBudgetManager()
        # 1 token budget ≈ 4 chars; budget 10 tokens ≈ 40 chars
        big_content = "x" * 400  # 100 tokens
        small_content = "y" * 20  # 5 tokens
        hits = [
            self._make_hit(big_content, score=0.9),
            self._make_hit(small_content, score=0.8),
        ]
        result = mgr.assemble(hits, budget_tokens=10, format="plain")
        # Should include small content or truncated big content, not full big content
        assert len(result) <= 400 + 50  # some slack for separator

    def test_empty_hits_returns_empty(self):
        mgr = ContextBudgetManager()
        result = mgr.assemble([], budget_tokens=1000, format="markdown")
        assert result == ""

    def test_empty_hits_json_returns_empty_list(self):
        mgr = ContextBudgetManager()
        result = mgr.assemble([], budget_tokens=1000, format="json")
        assert result == "[]"

    def test_sorted_by_score_times_importance(self):
        mgr = ContextBudgetManager()
        low_score_rec = MemoryRecord(content="LOW", importance=1.0)
        high_score_rec = MemoryRecord(content="HIGH", importance=1.0)
        hits = [
            SearchHit(record=low_score_rec, score=0.3),
            SearchHit(record=high_score_rec, score=0.9),
        ]
        result = mgr.assemble(hits, budget_tokens=1000, format="plain")
        assert result.index("HIGH") < result.index("LOW")

    def test_markdown_attribution_format(self):
        mgr = ContextBudgetManager()
        hits = [self._make_hit("content here", score=0.75, source_ref="core/module.py")]
        result = mgr.assemble(hits, budget_tokens=1000, format="markdown")
        assert "[core/module.py]" in result
        assert "score=0.75" in result

    def test_never_raises_on_bad_input(self):
        mgr = ContextBudgetManager()
        # Should not raise even with weird hits
        result = mgr.assemble(None, budget_tokens=100, format="markdown")
        assert result == ""

    def test_json_never_raises(self):
        mgr = ContextBudgetManager()
        result = mgr.assemble(None, budget_tokens=100, format="json")
        assert result == "[]"


# ===========================================================================
# TestASCMV2Integration
# ===========================================================================

class TestASCMV2Integration:
    def test_upsert_search_budget_end_to_end(self):
        store = _tmp_store()
        provider = LocalEmbeddingProvider()
        mgr = ContextBudgetManager()

        texts = [
            "Python async programming with asyncio and event loops",
            "Machine learning model training with PyTorch",
            "Database optimization using indexes and query planning",
        ]
        for i, text in enumerate(texts):
            rec = MemoryRecord(
                content=text,
                source_type="file",
                source_ref=f"docs/topic{i}.md",
                importance=1.0,
                embedding_model="local-tfidf-svd-50d",
            )
            embeddings = provider.embed([text])
            store.upsert(rec, embedding=embeddings[0] if embeddings else None)

        assert store.count() == 3

        q = RetrievalQuery(query_text="asyncio event loop", min_score=0.0)
        hits = store.search(q)
        assert len(hits) >= 1

        result = mgr.assemble(hits, budget_tokens=500, format="markdown")
        assert isinstance(result, str)
        assert len(result) > 0

        store.close()

    def test_filter_and_assemble_json(self):
        store = _tmp_store()
        mgr = ContextBudgetManager()

        store.upsert(_make_record(content="plan step one", source_type="goal", goal_id="g1"))
        store.upsert(_make_record(content="plan step two", source_type="goal", goal_id="g1"))
        store.upsert(_make_record(content="other content", source_type="memory", goal_id="g2"))

        records = store.filter_by(goal_id="g1")
        assert len(records) == 2

        hits = [SearchHit(record=r, score=0.8) for r in records]
        result = mgr.assemble(hits, budget_tokens=200, format="json")
        data = json.loads(result)
        assert len(data) >= 1

        store.close()

    def test_migrate_and_search(self):
        store = _tmp_store()

        class MockRec:
            def __init__(self, content):
                self.content = content
                self.source_ref = "legacy/file.py"
                self.created_at = time.time()
                self.goal_id = None
                self.agent_name = None
                self.tags = []
                self.importance = 1.0

        class MockV1:
            _cache_records = {
                "r1": MockRec("legacy python asyncio code"),
                "r2": MockRec("legacy SQL database query"),
            }

        count = store.migrate_from_v1(MockV1())
        assert count == 2

        q = RetrievalQuery(query_text="python asyncio", min_score=0.1)
        hits = store.search(q)
        assert len(hits) >= 1
        assert hits[0].record.source_type == "legacy"

        store.close()

    def test_full_pipeline_with_embeddings(self):
        store = _tmp_store()
        provider = LocalEmbeddingProvider()
        mgr = ContextBudgetManager()

        content = "semantic search with vector embeddings"
        rec = MemoryRecord(
            content=content,
            source_ref="memory/vector_store_v2.py",
            embedding_model=provider.model_id(),
        )
        emb = provider.embed([content])
        store.upsert(rec, embedding=emb[0])

        query_emb = provider.embed(["vector search embeddings"])
        # min_score=-1.0 to allow any cosine similarity (random unit vectors may be negative)
        q = RetrievalQuery(query_text="vector search embeddings", min_score=-1.0)
        hits = store.search(q, query_embedding=query_emb[0])

        assert len(hits) >= 1
        result = mgr.assemble(hits, budget_tokens=100, format="plain")
        assert isinstance(result, str)

        store.close()
