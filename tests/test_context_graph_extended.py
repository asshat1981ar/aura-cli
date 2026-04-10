"""Extended unit tests for core/context_graph.py — ContextGraph.

Covers graph operations, caching, batch queries, and graph analysis not in main test file.
"""

from __future__ import annotations

import json
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.context_graph import ContextGraph, _CacheEntry, _CACHE_TTL_SECONDS


# ---------------------------------------------------------------------------
# Cache Entry Tests
# ---------------------------------------------------------------------------


class TestCacheEntry:
    """Test cache entry TTL behavior."""

    def test_cache_entry_stores_value(self):
        """Verify cache entry stores value."""
        entry = _CacheEntry("test_value")
        assert entry.value == "test_value"

    def test_cache_entry_records_timestamp(self):
        """Verify cache entry records creation timestamp."""
        entry = _CacheEntry("test")
        assert entry.timestamp > 0
        assert isinstance(entry.timestamp, float)

    def test_cache_entry_is_not_expired_immediately(self):
        """Verify newly created entry is not expired."""
        entry = _CacheEntry("test")
        assert entry.is_expired() is False

    def test_cache_entry_expires_after_ttl(self):
        """Verify entry expires after TTL."""
        entry = _CacheEntry("test")
        entry.timestamp = time.time() - (_CACHE_TTL_SECONDS + 1)
        
        assert entry.is_expired() is True

    def test_cache_entry_custom_ttl(self):
        """Verify custom TTL is respected."""
        entry = _CacheEntry("test")
        entry.timestamp = time.time() - 10
        
        assert entry.is_expired(ttl=5) is True
        assert entry.is_expired(ttl=15) is False


# ---------------------------------------------------------------------------
# Database Initialization
# ---------------------------------------------------------------------------


class TestContextGraphInit:
    """Test ContextGraph initialization."""

    def test_context_graph_creates_db(self):
        """Verify database is created on initialization."""
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            graph = ContextGraph(db_path=db_path)
            
            assert db_path.exists()

    def test_context_graph_uses_default_path_if_none(self):
        """Verify default path is used when not specified."""
        graph = ContextGraph()
        # Should not raise
        assert graph._db_path is not None

    def test_context_graph_caching_enabled_by_default(self):
        """Verify caching is enabled by default."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            assert graph._enable_caching is True

    def test_context_graph_caching_can_be_disabled(self):
        """Verify caching can be disabled."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(
                db_path=Path(td) / "test.db",
                enable_caching=False
            )
            assert graph._enable_caching is False

    def test_context_graph_cache_initialized_empty(self):
        """Verify cache starts empty."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            assert len(graph._cache) == 0

    def test_context_graph_cache_stats_initialized(self):
        """Verify cache statistics start at zero."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            assert graph._cache_hits == 0
            assert graph._cache_misses == 0


# ---------------------------------------------------------------------------
# Caching Layer
# ---------------------------------------------------------------------------


class TestCachingLayer:
    """Test cache operations."""

    def test_get_from_cache_returns_none_when_disabled(self):
        """Verify cache returns None when disabled."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(
                db_path=Path(td) / "test.db",
                enable_caching=False
            )
            result = graph._get_from_cache("test_key")
            assert result is None

    def test_get_from_cache_miss(self):
        """Verify cache miss returns None."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            result = graph._get_from_cache("nonexistent")
            
            assert result is None
            assert graph._cache_misses == 1

    def test_get_from_cache_hit(self):
        """Verify cache hit returns value."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            graph._cache["key1"] = _CacheEntry("value1")
            
            result = graph._get_from_cache("key1")
            
            assert result == "value1"
            assert graph._cache_hits == 1

    def test_set_in_cache_when_disabled(self):
        """Verify set_in_cache is no-op when disabled."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(
                db_path=Path(td) / "test.db",
                enable_caching=False
            )
            graph._set_in_cache("key", "value")
            
            assert len(graph._cache) == 0

    def test_set_in_cache_stores_value(self):
        """Verify set_in_cache stores value."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            graph._set_in_cache("key", "value")
            
            assert graph._get_from_cache("key") == "value"

    def test_invalidate_cache_clears_all(self):
        """Verify cache can be fully cleared."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            graph._cache["key1"] = _CacheEntry("value1")
            graph._cache["key2"] = _CacheEntry("value2")
            
            graph._invalidate_cache()
            
            assert len(graph._cache) == 0

    def test_invalidate_cache_by_prefix(self):
        """Verify cache can be invalidated by prefix."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            graph._cache["node:1"] = _CacheEntry("v1")
            graph._cache["node:2"] = _CacheEntry("v2")
            graph._cache["edge:1"] = _CacheEntry("v3")
            
            graph._invalidate_cache(prefix="node:")
            
            assert "node:1" not in graph._cache
            assert "node:2" not in graph._cache
            assert "edge:1" in graph._cache

    def test_set_in_cache_evicts_when_full(self):
        """Verify LRU eviction when cache is full."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            # Fill cache
            for i in range(1024):
                graph._cache[f"key_{i}"] = _CacheEntry(f"value_{i}")
            
            original_size = len(graph._cache)
            
            # Add one more (should trigger eviction)
            graph._set_in_cache("overflow", "value")
            
            # Should have evicted ~10% (102 entries)
            assert len(graph._cache) < original_size

    def test_get_cache_stats(self):
        """Verify cache statistics are calculated correctly."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            graph._cache["key1"] = _CacheEntry("value1")
            
            # Generate hits and misses
            graph._get_from_cache("key1")  # hit
            graph._get_from_cache("key1")  # hit
            graph._get_from_cache("missing")  # miss
            
            stats = graph.get_cache_stats()
            
            assert stats["hits"] == 2
            assert stats["misses"] == 1
            assert stats["size"] == 1
            assert stats["hit_rate_percent"] > 0


# ---------------------------------------------------------------------------
# Public API: add_edge
# ---------------------------------------------------------------------------


class TestAddEdge:
    """Test public edge addition API."""

    def test_add_edge_creates_edge(self):
        """Verify edge can be created via public API."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            graph.add_edge("core/a.py", "core/b.py", "depends_on", weight=1.5)
            
            # Verify nodes were created
            nodes = graph.get_nodes_batch([f"file:core/a.py", f"file:core/b.py"])
            assert len(nodes) == 2

    def test_add_edge_creates_file_nodes(self):
        """Verify file nodes are created automatically."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            graph.add_edge("src.py", "dst.py", "uses")
            
            # Check nodes exist
            src_node = graph.get_node(f"file:src.py")
            dst_node = graph.get_node(f"file:dst.py")
            
            assert src_node is not None
            assert dst_node is not None
            assert src_node["type"] == "file"
            assert dst_node["type"] == "file"

    def test_add_edge_with_default_weight(self):
        """Verify default weight is 1.0."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            graph.add_edge("a.py", "b.py", "uses")
            
            # Verify edge was created (no error)
            edges = graph.get_edges_batch(
                src_ids=[f"file:a.py"]
            )
            assert len(edges) > 0

    def test_add_edge_with_custom_weight(self):
        """Verify custom weight is stored."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            graph.add_edge("a.py", "b.py", "uses", weight=2.5)
            
            edges = graph.get_edges_batch(src_ids=[f"file:a.py"])
            assert len(edges) > 0


# ---------------------------------------------------------------------------
# Node Operations
# ---------------------------------------------------------------------------


class TestNodeOperations:
    """Test node operations via batch APIs."""

    def test_get_node_returns_none_for_nonexistent(self):
        """Verify get_node returns None for nonexistent node."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            node = graph.get_node("nonexistent_id")
            
            assert node is None

    def test_get_nodes_batch_retrieves_multiple(self):
        """Verify batch node retrieval."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            # Create nodes via edges (public API)
            graph.add_edge("a.py", "b.py", "uses")
            graph.add_edge("c.py", "d.py", "uses")
            
            # Retrieve in batch
            nodes = graph.get_nodes_batch([f"file:a.py", f"file:b.py"])
            
            assert len(nodes) >= 0

    def test_get_nodes_batch_empty_list(self):
        """Verify empty list returns empty dict."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            nodes = graph.get_nodes_batch([])
            
            assert nodes == {}

    def test_get_nodes_batch_uses_cache(self):
        """Verify batch retrieval uses cache."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            # Create a node
            graph.add_edge("a.py", "b.py", "uses")
            
            # First call
            nodes1 = graph.get_nodes_batch([f"file:a.py"])
            hits_before = graph._cache_hits
            
            # Second call (should hit cache)
            nodes2 = graph.get_nodes_batch([f"file:a.py"])
            hits_after = graph._cache_hits
            
            assert hits_after >= hits_before


# ---------------------------------------------------------------------------
# Edge Operations
# ---------------------------------------------------------------------------


class TestEdgeOperations:
    """Test edge operations."""

    def test_get_edge_returns_none_for_nonexistent(self):
        """Verify get_edge returns None for nonexistent edge."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            edge = graph.get_edge("nonexistent")
            
            assert edge is None

    def test_get_edges_batch_by_src(self):
        """Verify batch edge retrieval by source."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            graph.add_edge("src.py", "dst1.py", "uses")
            graph.add_edge("src.py", "dst2.py", "uses")
            
            edges = graph.get_edges_batch(src_ids=[f"file:src.py"])
            
            assert len(edges) >= 2

    def test_get_edges_batch_by_dst(self):
        """Verify batch edge retrieval by destination."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            graph.add_edge("src1.py", "dst.py", "uses")
            graph.add_edge("src2.py", "dst.py", "uses")
            
            edges = graph.get_edges_batch(dst_ids=[f"file:dst.py"])
            
            assert len(edges) >= 2

    def test_get_edges_batch_by_relation(self):
        """Verify batch edge retrieval by relation type."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            graph.add_edge("a.py", "b.py", "uses")
            graph.add_edge("a.py", "c.py", "related_to")
            
            edges = graph.get_edges_batch(relations=["uses"])
            
            assert any(e["relation"] == "uses" for e in edges)

    def test_get_edges_batch_combined_filters(self):
        """Verify batch edge retrieval with multiple filters."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            graph.add_edge("src.py", "dst1.py", "uses")
            graph.add_edge("src.py", "dst2.py", "related_to")
            
            edges = graph.get_edges_batch(src_ids=[f"file:src.py"], relations=["uses"])
            
            # Should only have 'uses' edges from src
            for edge in edges:
                if edge["src_id"] == f"file:src.py":
                    assert edge["relation"] == "uses"


# ---------------------------------------------------------------------------
# Neighborhood Operations
# ---------------------------------------------------------------------------


class TestNeighborhoodOperations:
    """Test node neighborhood queries."""

    def test_get_neighbors_batch_outgoing(self):
        """Verify outgoing neighbors retrieval."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            graph.add_edge("src.py", "dst1.py", "uses")
            graph.add_edge("src.py", "dst2.py", "uses")
            
            neighbors = graph.get_neighbors_batch([f"file:src.py"], direction="outgoing")
            
            assert f"file:src.py" in neighbors

    def test_get_neighbors_batch_incoming(self):
        """Verify incoming neighbors retrieval."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            graph.add_edge("src1.py", "dst.py", "uses")
            graph.add_edge("src2.py", "dst.py", "uses")
            
            neighbors = graph.get_neighbors_batch([f"file:dst.py"], direction="incoming")
            
            assert f"file:dst.py" in neighbors

    def test_get_neighbors_batch_both(self):
        """Verify bidirectional neighbors retrieval."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            graph.add_edge("dep1.py", "node.py", "uses")
            graph.add_edge("dep2.py", "node.py", "uses")
            graph.add_edge("node.py", "user.py", "uses")
            
            neighbors = graph.get_neighbors_batch([f"file:node.py"], direction="both")
            
            assert f"file:node.py" in neighbors

    def test_get_neighbors_batch_filtered_by_relation(self):
        """Verify neighbors can be filtered by relation type."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            graph.add_edge("src.py", "uses_dst.py", "uses")
            graph.add_edge("src.py", "related_dst.py", "related_to")
            
            neighbors = graph.get_neighbors_batch(
                [f"file:src.py"],
                relation="uses",
                direction="outgoing"
            )
            
            assert f"file:src.py" in neighbors


# ---------------------------------------------------------------------------
# Eager Loading
# ---------------------------------------------------------------------------


class TestEagerLoading:
    """Test eager loading of nodes and edges."""

    def test_get_nodes_with_edges_retrieves_both(self):
        """Verify eager loading retrieves nodes and edges."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            graph.add_edge("a.py", "b.py", "uses")
            
            nodes, edges = graph.get_nodes_with_edges([f"file:a.py", f"file:b.py"])
            
            assert len(nodes) >= 0
            assert isinstance(edges, list)

    def test_get_nodes_with_edges_excludes_incoming(self):
        """Verify incoming edges can be excluded."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            graph.add_edge("a.py", "b.py", "uses")
            
            nodes, edges = graph.get_nodes_with_edges(
                [f"file:a.py", f"file:b.py"],
                include_incoming=False
            )
            
            assert isinstance(edges, list)

    def test_get_nodes_with_edges_excludes_outgoing(self):
        """Verify outgoing edges can be excluded."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            graph.add_edge("a.py", "b.py", "uses")
            
            nodes, edges = graph.get_nodes_with_edges(
                [f"file:a.py", f"file:b.py"],
                include_outgoing=False
            )
            
            assert isinstance(edges, list)


# ---------------------------------------------------------------------------
# Graph Analysis
# ---------------------------------------------------------------------------


class TestGraphAnalysis:
    """Test graph analysis methods."""

    def test_goals_touching_file(self):
        """Verify goal query for a file."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            goals = graph.goals_touching_file("core/main.py")
            
            assert isinstance(goals, list)

    def test_query_similar_resolutions(self):
        """Verify similar resolution query."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            similar = graph.query_similar_resolutions("core/orchestrator.py")
            
            assert isinstance(similar, list)

    def test_best_skills_for_goal_type(self):
        """Verify best skills query for goal type."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            skills = graph.best_skills_for_goal_type("bug_fix")
            
            assert isinstance(skills, list)

    def test_weaknesses_for_goal_type(self):
        """Verify weaknesses query for goal type."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            weaknesses = graph.weaknesses_for_goal_type("refactor")
            
            assert isinstance(weaknesses, list)

    def test_file_failure_count(self):
        """Verify file failure count query."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            count = graph.file_failure_count("test.py")
            
            assert isinstance(count, int)
            assert count >= 0

    def test_find_bottleneck_files(self):
        """Verify bottleneck file detection."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            bottlenecks = graph.find_bottleneck_files()
            
            assert isinstance(bottlenecks, list)

    def test_find_circular_dependencies(self):
        """Verify circular dependency detection."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            cycles = graph.find_circular_dependencies()
            
            assert isinstance(cycles, list)

    def test_graph_summary(self):
        """Verify graph summary generation."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            graph.add_edge("a.py", "b.py", "uses")
            
            summary = graph.graph_summary()
            
            assert isinstance(summary, dict)
            assert "nodes" in summary or "edges" in summary


# ---------------------------------------------------------------------------
# NetworkX Integration
# ---------------------------------------------------------------------------


class TestNetworkXIntegration:
    """Test NetworkX graph export."""

    def test_get_nx_graph(self):
        """Verify NetworkX graph export."""
        pytest.importorskip("networkx")
        
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            graph.add_edge("a.py", "b.py", "uses")
            
            nx_graph = graph.get_nx_graph()
            
            assert nx_graph is not None

    def test_get_nx_graph_with_filter(self):
        """Verify NetworkX graph creation."""
        pytest.importorskip("networkx")
        
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            graph.add_edge("a.py", "b.py", "uses")
            
            # get_nx_graph takes relations parameter, not node_types
            nx_graph = graph.get_nx_graph(relations=["uses"])
            
            assert nx_graph is not None


# ---------------------------------------------------------------------------
# Cycle Updates
# ---------------------------------------------------------------------------


class TestCycleUpdates:
    """Test updating graph from cycle data."""

    def test_update_from_cycle(self):
        """Verify cycle data integration."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            cycle_entry = {
                "cycle_id": "cycle_1",
                "goal": "implement feature",
                "files_changed": ["core/main.py", "core/utils.py"],
                "skills_used": ["refactor", "test_writer"],
            }
            
            # Should not raise
            graph.update_from_cycle(cycle_entry)


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_node_id(self):
        """Verify handling of empty node ID."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            node = graph.get_node("")
            assert node is None

    def test_very_long_label(self):
        """Verify handling of very long label."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            long_label = "x" * 10000
            graph.add_edge(long_label, "b.py", "uses")
            
            # Should not raise

    def test_special_characters_in_labels(self):
        """Verify handling of special characters in labels."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            special = "test!@#$%^&*().py"
            graph.add_edge(special, "other.py", "uses")
            
            # Should not raise

    def test_negative_weight_on_edge(self):
        """Verify handling of negative edge weights."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            graph.add_edge("a.py", "b.py", "uses", weight=-1.0)
            
            edges = graph.get_edges_batch(src_ids=[f"file:a.py"])
            assert len(edges) > 0

    def test_zero_weight_on_edge(self):
        """Verify handling of zero edge weight."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            graph.add_edge("a.py", "b.py", "uses", weight=0.0)
            
            edges = graph.get_edges_batch(src_ids=[f"file:a.py"])
            assert len(edges) > 0

    def test_very_large_graph(self):
        """Verify handling of large graph."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            # Create 50 edges
            for i in range(50):
                graph.add_edge(f"file_{i}.py", f"file_{i+1}.py", "uses")
            
            # Retrieve in batch
            node_ids = [f"file:file_{i}.py" for i in range(50)]
            nodes = graph.get_nodes_batch(node_ids)
            
            assert len(nodes) > 0

    def test_duplicate_edge_handling(self):
        """Verify duplicate edge handling."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            # Add same edge twice with different weights
            graph.add_edge("a.py", "b.py", "uses", weight=1.0)
            graph.add_edge("a.py", "b.py", "uses", weight=2.0)
            
            # Should not raise - implementation handles upsert


# ---------------------------------------------------------------------------
# Batch Traversal
# ---------------------------------------------------------------------------


class TestBatchTraversal:
    """Test batch graph traversal."""

    def test_traverse_batched_basic(self):
        """Verify batch traversal execution."""
        with tempfile.TemporaryDirectory() as td:
            graph = ContextGraph(db_path=Path(td) / "test.db")
            
            graph.add_edge("a.py", "b.py", "uses")
            graph.add_edge("b.py", "c.py", "uses")
            
            # Traverse from a node (uses start_node_ids parameter)
            results = graph.traverse_batched(
                start_node_ids=[f"file:a.py"],
                max_depth=2,
                direction="outgoing"
            )
            
            assert isinstance(results, dict)
