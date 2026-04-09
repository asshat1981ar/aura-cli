"""
Benchmark for ContextGraph N+1 query optimizations.

This benchmark measures the performance improvement of batch operations
compared to individual queries.

Usage:
    python -m pytest tests/benchmark_context_graph.py -v
    python tests/benchmark_context_graph.py
"""
import tempfile
import time
import unittest
from pathlib import Path
from typing import List, Dict, Any

from core.context_graph import ContextGraph


class TestContextGraphBenchmark(unittest.TestCase):
    """Benchmark tests for ContextGraph performance."""

    def setUp(self):
        """Set up a temporary database for each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = self.temp_dir / "benchmark.db"
        self.graph = ContextGraph(db_path=self.db_path)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_graph(self, num_nodes: int, edges_per_node: int = 3) -> List[str]:
        """Create a test graph with specified number of nodes and edges."""
        node_ids = []
        
        # Create nodes in batch using internal method
        nodes_to_create = []
        for i in range(num_nodes):
            node_id = f"file:test_file_{i}.py"
            nodes_to_create.append((node_id, "file", f"test_file_{i}.py", {"index": i}))
            node_ids.append(node_id)
        
        self.graph._upsert_nodes_batch(nodes_to_create)
        
        # Create edges in batch
        edges_to_create = []
        for i in range(num_nodes):
            for j in range(1, min(edges_per_node + 1, num_nodes)):
                target_idx = (i + j) % num_nodes
                edges_to_create.append((
                    node_ids[i],
                    node_ids[target_idx],
                    "related_to",
                    1.0,
                    {}
                ))
        
        self.graph._upsert_edges_batch(edges_to_create)
        return node_ids

    def test_benchmark_batch_vs_individual_node_queries(self):
        """Compare batch node queries vs individual queries (N+1 pattern)."""
        num_nodes = 500
        node_ids = self._create_test_graph(num_nodes)
        
        # Clear cache to get accurate measurements
        self.graph._invalidate_cache()
        
        # Measure individual queries (N+1 pattern)
        start_time = time.time()
        individual_results = {}
        for node_id in node_ids:
            node = self.graph.get_node(node_id)
            if node:
                individual_results[node_id] = node
        individual_time = time.time() - start_time
        
        # Clear cache again
        self.graph._invalidate_cache()
        
        # Measure batch query
        start_time = time.time()
        batch_results = self.graph.get_nodes_batch(node_ids)
        batch_time = time.time() - start_time
        
        # Verify results are equivalent
        self.assertEqual(len(individual_results), len(batch_results))
        
        # Calculate improvement
        speedup = individual_time / batch_time if batch_time > 0 else float('inf')
        improvement_pct = ((individual_time - batch_time) / individual_time * 100) if individual_time > 0 else 0
        
        print(f"\n--- Batch vs Individual Node Queries ({num_nodes} nodes) ---")
        print(f"Individual queries (N+1): {individual_time:.4f}s")
        print(f"Batch query: {batch_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Improvement: {improvement_pct:.1f}%")
        
        # Batch should be significantly faster
        self.assertLess(batch_time, individual_time * 0.5, 
                       "Batch query should be at least 2x faster")

    def test_benchmark_batch_vs_individual_edge_queries(self):
        """Compare batch edge queries vs individual queries."""
        num_nodes = 500
        node_ids = self._create_test_graph(num_nodes, edges_per_node=5)
        
        # Clear cache
        self.graph._invalidate_cache()
        
        # Measure individual edge queries (N+1)
        start_time = time.time()
        individual_edges = []
        for node_id in node_ids[:100]:  # Test subset for speed
            edges = self.graph.get_edges_batch(src_ids=[node_id])
            individual_edges.extend(edges)
        individual_time = time.time() - start_time
        
        # Clear cache
        self.graph._invalidate_cache()
        
        # Measure batch edge query
        start_time = time.time()
        batch_edges = self.graph.get_edges_batch(src_ids=node_ids[:100])
        batch_time = time.time() - start_time
        
        print(f"\n--- Batch vs Individual Edge Queries ({len(node_ids[:100])} nodes) ---")
        print(f"Individual queries (N+1): {individual_time:.4f}s")
        print(f"Batch query: {batch_time:.4f}s")
        speedup = individual_time / batch_time if batch_time > 0 else float('inf')
        print(f"Speedup: {speedup:.2f}x")
        
        # Batch should be faster
        self.assertLess(batch_time, individual_time)

    def test_benchmark_eager_loading(self):
        """Test eager loading performance."""
        num_nodes = 300
        node_ids = self._create_test_graph(num_nodes, edges_per_node=3)
        sample_ids = node_ids[:50]
        
        # Clear cache
        self.graph._invalidate_cache()
        
        # Method 1: Naive N+1 loading
        start_time = time.time()
        naive_nodes = []
        naive_edges = []
        for node_id in sample_ids:
            node = self.graph.get_node(node_id)
            if node:
                naive_nodes.append(node)
                edges = self.graph.get_edges_batch(src_ids=[node_id])
                naive_edges.extend(edges)
        naive_time = time.time() - start_time
        
        # Clear cache
        self.graph._invalidate_cache()
        
        # Method 2: Eager loading (2 queries)
        start_time = time.time()
        eager_nodes, eager_edges = self.graph.get_nodes_with_edges(sample_ids)
        eager_time = time.time() - start_time
        
        print(f"\n--- Eager Loading ({len(sample_ids)} nodes) ---")
        print(f"Naive N+1 loading: {naive_time:.4f}s")
        print(f"Eager loading (2 queries): {eager_time:.4f}s")
        speedup = naive_time / eager_time if eager_time > 0 else float('inf')
        print(f"Speedup: {speedup:.2f}x")
        
        # Verify results
        self.assertEqual(len(naive_nodes), len(eager_nodes))
        self.assertLess(eager_time, naive_time)

    def test_benchmark_neighbor_batching(self):
        """Test batch neighbor fetching."""
        num_nodes = 300
        node_ids = self._create_test_graph(num_nodes, edges_per_node=5)
        sample_ids = node_ids[:30]
        
        # Clear cache
        self.graph._invalidate_cache()
        
        # Method 1: Individual neighbor queries
        start_time = time.time()
        naive_neighbors = {}
        for node_id in sample_ids:
            edges = self.graph.get_edges_batch(src_ids=[node_id])
            neighbors = []
            for edge in edges:
                neighbor = self.graph.get_node(edge["dst_id"])
                if neighbor:
                    neighbors.append(neighbor)
            naive_neighbors[node_id] = neighbors
        naive_time = time.time() - start_time
        
        # Clear cache
        self.graph._invalidate_cache()
        
        # Method 2: Batch neighbor fetching
        start_time = time.time()
        batch_neighbors = self.graph.get_neighbors_batch(sample_ids, direction="outgoing")
        batch_time = time.time() - start_time
        
        print(f"\n--- Neighbor Batching ({len(sample_ids)} nodes) ---")
        print(f"Individual queries: {naive_time:.4f}s")
        print(f"Batch neighbors: {batch_time:.4f}s")
        speedup = naive_time / batch_time if batch_time > 0 else float('inf')
        print(f"Speedup: {speedup:.2f}x")
        
        # Results should be equivalent
        self.assertEqual(set(naive_neighbors.keys()), set(batch_neighbors.keys()))
        self.assertLess(batch_time, naive_time)

    def test_benchmark_caching(self):
        """Test cache performance."""
        num_nodes = 200
        node_ids = self._create_test_graph(num_nodes)
        
        # First access (cache miss)
        start_time = time.time()
        for node_id in node_ids[:50]:
            self.graph.get_node(node_id)
        first_access_time = time.time() - start_time
        
        # Second access (cache hit)
        start_time = time.time()
        for node_id in node_ids[:50]:
            self.graph.get_node(node_id)
        second_access_time = time.time() - start_time
        
        stats = self.graph.get_cache_stats()
        
        print(f"\n--- Caching Performance ({len(node_ids[:50])} nodes) ---")
        print(f"First access (cache miss): {first_access_time:.4f}s")
        print(f"Second access (cache hit): {second_access_time:.4f}s")
        print(f"Cache stats: {stats}")
        speedup = first_access_time / second_access_time if second_access_time > 0 else float('inf')
        print(f"Speedup: {speedup:.2f}x")
        
        # Cache hits should be much faster
        self.assertLess(second_access_time, first_access_time * 0.5)
        self.assertGreater(stats["hits"], 0)

    def test_benchmark_graph_summary_caching(self):
        """Test graph summary caching."""
        self._create_test_graph(100)
        
        # First call
        start_time = time.time()
        summary1 = self.graph.graph_summary()
        first_time = time.time() - start_time
        
        # Second call (should be cached)
        start_time = time.time()
        summary2 = self.graph.graph_summary()
        second_time = time.time() - start_time
        
        self.assertEqual(summary1, summary2)
        
        print(f"\n--- Graph Summary Caching ---")
        print(f"First call: {first_time:.4f}s")
        print(f"Cached call: {second_time:.6f}s")
        
        # Cached call should be much faster
        self.assertLess(second_time, first_time * 0.5)

    def test_benchmark_batch_ingestion(self):
        """Test batch ingestion vs individual."""
        num_nodes = 500
        
        # Create test data
        nodes = [(f"file:test_{i}.py", "file", f"test_{i}.py", {}) 
                for i in range(num_nodes)]
        edges = [(f"file:test_{i}.py", f"file:test_{(i+1) % num_nodes}.py", 
                 "related_to", 1.0, {}) 
                for i in range(num_nodes)]
        
        # Measure individual ingestion
        graph1 = ContextGraph(db_path=self.temp_dir / "individual.db")
        start_time = time.time()
        for node_id, node_type, label, meta in nodes:
            graph1._upsert_node(node_id, node_type, label, meta)
        for src, dst, rel, weight, meta in edges:
            graph1._upsert_edge(src, dst, rel, weight, meta)
        individual_time = time.time() - start_time
        
        # Measure batch ingestion
        graph2 = ContextGraph(db_path=self.temp_dir / "batch.db")
        start_time = time.time()
        graph2._upsert_nodes_batch(nodes)
        graph2._upsert_edges_batch(edges)
        batch_time = time.time() - start_time
        
        print(f"\n--- Batch Ingestion ({num_nodes} nodes, {num_nodes} edges) ---")
        print(f"Individual ingestion: {individual_time:.4f}s")
        print(f"Batch ingestion: {batch_time:.4f}s")
        speedup = individual_time / batch_time if batch_time > 0 else float('inf')
        print(f"Speedup: {speedup:.2f}x")
        
        # Batch should be faster
        self.assertLess(batch_time, individual_time)

    def test_benchmark_scales(self):
        """Test performance at different scales."""
        scales = [100, 500, 1000]
        
        print("\n--- Scale Performance Test ---")
        print(f"{'Nodes':<10} {'Create Time':<15} {'Batch Query':<15} {'Cache Hit':<15}")
        print("-" * 60)
        
        for num_nodes in scales:
            # Create graph
            start_time = time.time()
            node_ids = self._create_test_graph(num_nodes, edges_per_node=3)
            create_time = time.time() - start_time
            
            # Batch query test
            self.graph._invalidate_cache()
            start_time = time.time()
            nodes = self.graph.get_nodes_batch(node_ids[:100])
            batch_time = time.time() - start_time
            
            # Cache test
            start_time = time.time()
            for node_id in node_ids[:100]:
                self.graph.get_node(node_id)
            cache_time = time.time() - start_time
            
            print(f"{num_nodes:<10} {create_time:<15.4f} {batch_time:<15.4f} {cache_time:<15.4f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
