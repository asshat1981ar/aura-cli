"""
Context Graph — Persistent knowledge graph for AURA.

A growing SQLite-backed property graph that tracks relationships between
files, goals, skills, weaknesses, learnings, and cycles.  Every cycle adds
nodes and edges.  The graph is queried by AdaptivePipeline to make planning
decisions that leverage everything the system has learned.

Node types:   file | goal | skill | weakness | learning | cycle
Edge relations: caused_by | fixed_by | uses | related_to | generated |
                mentions | depends_on | failed_on | succeeded_on

Usage::

    from core.context_graph import ContextGraph
    cg = ContextGraph()
    cg.update_from_cycle(cycle_entry)

    # What fixed files like this before?
    similar = cg.query_similar_resolutions("core/orchestrator.py")

    # What skills are most useful for bug_fix goals?
    skills = cg.best_skills_for_goal_type("bug_fix")

    # What goals relate to this file?
    goals = cg.goals_touching_file("core/file_tools.py")
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from functools import lru_cache

from core.logging_utils import log_json

_DB_PATH = Path(__file__).resolve().parent.parent / "memory" / "context_graph.db"

# Cache configuration
_CACHE_MAXSIZE = 1024
_CACHE_TTL_SECONDS = 300  # 5 minutes


class _CacheEntry:
    """Simple TTL cache entry."""
    __slots__ = ('value', 'timestamp')
    
    def __init__(self, value: Any):
        self.value = value
        self.timestamp = time.time()
    
    def is_expired(self, ttl: float = _CACHE_TTL_SECONDS) -> bool:
        return time.time() - self.timestamp > ttl


class ContextGraph:
    """Persistent property graph — grows richer with every cycle.

    Thread-safe via SQLite WAL mode + per-connection isolation.
    Optimized with batch queries and caching to eliminate N+1 query problems.
    """

    def __init__(self, db_path: Optional[Path] = None, enable_caching: bool = True):
        self._db_path = db_path or _DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._enable_caching = enable_caching
        self._cache: Dict[str, _CacheEntry] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._init_db()

    # ── Schema ───────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._conn() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS nodes (
                id        TEXT PRIMARY KEY,
                type      TEXT NOT NULL,
                label     TEXT NOT NULL,
                meta      TEXT DEFAULT '{}',
                created   REAL NOT NULL,
                updated   REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS edges (
                id        TEXT PRIMARY KEY,
                src_id    TEXT NOT NULL REFERENCES nodes(id),
                dst_id    TEXT NOT NULL REFERENCES nodes(id),
                relation  TEXT NOT NULL,
                weight    REAL DEFAULT 1.0,
                meta      TEXT DEFAULT '{}',
                created   REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_nodes_type  ON nodes(type);
            CREATE INDEX IF NOT EXISTS idx_nodes_label ON nodes(label);
            CREATE INDEX IF NOT EXISTS idx_edges_src   ON edges(src_id);
            CREATE INDEX IF NOT EXISTS idx_edges_dst   ON edges(dst_id);
            CREATE INDEX IF NOT EXISTS idx_edges_rel   ON edges(relation);
            """)

    # ── Caching Layer ────────────────────────────────────────────────────────

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache if exists and not expired."""
        if not self._enable_caching:
            return None
        
        entry = self._cache.get(key)
        if entry is None:
            self._cache_misses += 1
            return None
        
        if entry.is_expired():
            del self._cache[key]
            self._cache_misses += 1
            return None
        
        self._cache_hits += 1
        return entry.value

    def _set_in_cache(self, key: str, value: Any) -> None:
        """Set value in cache with TTL."""
        if not self._enable_caching:
            return
        
        # Simple LRU eviction if cache is full
        if len(self._cache) >= _CACHE_MAXSIZE:
            # Remove oldest 10% of entries
            sorted_keys = sorted(self._cache.keys(), 
                               key=lambda k: self._cache[k].timestamp)
            for key_to_remove in sorted_keys[:_CACHE_MAXSIZE // 10]:
                del self._cache[key_to_remove]
        
        self._cache[key] = _CacheEntry(value)

    def _invalidate_cache(self, prefix: str = "") -> None:
        """Invalidate cache entries matching prefix."""
        if prefix:
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(prefix)]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            self._cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "size": len(self._cache)
        }

    # ── Batch Query Methods ──────────────────────────────────────────────────

    def get_nodes_batch(self, node_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch multiple nodes in a single query (avoids N+1).
        
        Returns:
            Dict mapping node_id to node data.
        """
        if not node_ids:
            return {}
        
        # Check cache first
        cached_nodes = {}
        ids_to_fetch = []
        
        for nid in node_ids:
            cache_key = f"node:{nid}"
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                cached_nodes[nid] = cached
            else:
                ids_to_fetch.append(nid)
        
        if not ids_to_fetch:
            return cached_nodes
        
        # Batch query for remaining nodes
        placeholders = ",".join(["?"] * len(ids_to_fetch))
        query = f"SELECT id, type, label, meta, created, updated FROM nodes WHERE id IN ({placeholders})"
        
        with self._conn() as db:
            rows = db.execute(query, ids_to_fetch).fetchall()
        
        result = cached_nodes.copy()
        for row in rows:
            node_data = {
                "id": row[0],
                "type": row[1],
                "label": row[2],
                "meta": self._parse_meta(row[3]),
                "created": row[4],
                "updated": row[5]
            }
            result[row[0]] = node_data
            self._set_in_cache(f"node:{row[0]}", node_data)
        
        return result

    def get_edges_batch(
        self, 
        node_ids: Optional[List[str]] = None,
        src_ids: Optional[List[str]] = None,
        dst_ids: Optional[List[str]] = None,
        relations: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Fetch edges in batch with multiple filter options.
        
        Args:
            node_ids: Match edges where src_id OR dst_id in list
            src_ids: Match edges where src_id in list
            dst_ids: Match edges where dst_id in list
            relations: Filter by relation types
            
        Returns:
            List of edge dictionaries.
        """
        conditions = []
        params = []
        
        if node_ids:
            placeholders = ",".join(["?"] * len(node_ids))
            conditions.append(f"(src_id IN ({placeholders}) OR dst_id IN ({placeholders}))")
            params.extend(node_ids)
            params.extend(node_ids)
        
        if src_ids:
            placeholders = ",".join(["?"] * len(src_ids))
            conditions.append(f"src_id IN ({placeholders})")
            params.extend(src_ids)
        
        if dst_ids:
            placeholders = ",".join(["?"] * len(dst_ids))
            conditions.append(f"dst_id IN ({placeholders})")
            params.extend(dst_ids)
        
        if relations:
            placeholders = ",".join(["?"] * len(relations))
            conditions.append(f"relation IN ({placeholders})")
            params.extend(relations)
        
        query = "SELECT id, src_id, dst_id, relation, weight, meta, created FROM edges"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        with self._conn() as db:
            rows = db.execute(query, params).fetchall()
        
        return [
            {
                "id": row[0],
                "src_id": row[1],
                "dst_id": row[2],
                "relation": row[3],
                "weight": row[4],
                "meta": self._parse_meta(row[5]),
                "created": row[6]
            }
            for row in rows
        ]

    def get_nodes_with_edges(
        self, 
        node_ids: List[str],
        include_incoming: bool = True,
        include_outgoing: bool = True
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        """Eager loading: Fetch nodes and their edges in 2 queries.
        
        Returns:
            Tuple of (nodes_dict, edges_list)
        """
        # Batch fetch nodes
        nodes = self.get_nodes_batch(node_ids)
        
        # Batch fetch edges
        edges = []
        if include_outgoing:
            edges.extend(self.get_edges_batch(src_ids=node_ids))
        if include_incoming:
            edges.extend(self.get_edges_batch(dst_ids=node_ids))
        
        # Remove duplicates if both incoming and outgoing
        if include_incoming and include_outgoing:
            seen_ids = set()
            unique_edges = []
            for edge in edges:
                if edge["id"] not in seen_ids:
                    seen_ids.add(edge["id"])
                    unique_edges.append(edge)
            edges = unique_edges
        
        return nodes, edges

    def get_neighbors_batch(
        self, 
        node_ids: List[str],
        relation: Optional[str] = None,
        direction: str = "both"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get neighbors for multiple nodes in batch.
        
        Args:
            node_ids: List of node IDs
            relation: Optional relation filter
            direction: 'outgoing', 'incoming', or 'both'
            
        Returns:
            Dict mapping node_id to list of neighbor nodes.
        """
        relations = [relation] if relation else None
        
        # Get all edges in batch
        edges = []
        if direction in ("outgoing", "both"):
            edges.extend(self.get_edges_batch(src_ids=node_ids, relations=relations))
        if direction in ("incoming", "both"):
            edges.extend(self.get_edges_batch(dst_ids=node_ids, relations=relations))
        
        # Collect all neighbor node IDs
        neighbor_ids = set()
        edge_map: Dict[str, List[Tuple[str, str]]] = {}  # node_id -> [(neighbor_id, edge_id)]
        
        for edge in edges:
            if direction in ("outgoing", "both") and edge["src_id"] in node_ids:
                neighbor_ids.add(edge["dst_id"])
                edge_map.setdefault(edge["src_id"], []).append((edge["dst_id"], edge["id"]))
            if direction in ("incoming", "both") and edge["dst_id"] in node_ids:
                neighbor_ids.add(edge["src_id"])
                edge_map.setdefault(edge["dst_id"], []).append((edge["src_id"], edge["id"]))
        
        # Batch fetch all neighbor nodes
        neighbors = self.get_nodes_batch(list(neighbor_ids))
        
        # Build result mapping
        result: Dict[str, List[Dict[str, Any]]] = {nid: [] for nid in node_ids}
        for node_id, neighbor_edge_pairs in edge_map.items():
            for neighbor_id, edge_id in neighbor_edge_pairs:
                if neighbor_id in neighbors:
                    neighbor_data = neighbors[neighbor_id].copy()
                    neighbor_data["_edge_id"] = edge_id
                    result[node_id].append(neighbor_data)
        
        return result

    def traverse_batched(
        self,
        start_node_ids: List[str],
        max_depth: int = 3,
        relation: Optional[str] = None,
        direction: str = "outgoing"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Batched graph traversal avoiding N+1 queries.
        
        Args:
            start_node_ids: Starting node IDs
            max_depth: Maximum traversal depth
            relation: Optional relation filter
            direction: 'outgoing' or 'incoming'
            
        Returns:
            Dict mapping start_node_id to list of reachable nodes at each depth.
        """
        results: Dict[str, List[Dict[str, Any]]] = {nid: [] for nid in start_node_ids}
        current_level = set(start_node_ids)
        visited = set(start_node_ids)
        
        for depth in range(max_depth):
            if not current_level:
                break
            
            # Get neighbors for all current nodes in one batch
            neighbors = self.get_neighbors_batch(
                list(current_level), 
                relation=relation, 
                direction=direction
            )
            
            next_level = set()
            for node_id, neighbor_list in neighbors.items():
                for neighbor in neighbor_list:
                    neighbor_id = neighbor["id"]
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_level.add(neighbor_id)
                        # Find the original start node this belongs to
                        for start_id in start_node_ids:
                            if node_id == start_id or any(
                                r.get("id") == node_id for r in results[start_id]
                            ):
                                neighbor["_depth"] = depth + 1
                                results[start_id].append(neighbor)
            
            current_level = next_level
        
        return results

    # ── Public API ───────────────────────────────────────────────────────────

    def update_from_cycle(self, cycle_entry: Dict[str, Any]) -> None:
        """Ingest a completed cycle entry and add/update nodes + edges."""
        try:
            self._ingest(cycle_entry)
            # Invalidate cache after ingestion
            self._invalidate_cache()
        except Exception as exc:
            log_json("WARN", "context_graph_ingest_failed", details={"error": str(exc)})

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a single node by ID (with caching)."""
        cache_key = f"node:{node_id}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        
        nodes = self.get_nodes_batch([node_id])
        return nodes.get(node_id)

    def get_edge(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """Get a single edge by ID (with caching)."""
        cache_key = f"edge:{edge_id}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        
        with self._conn() as db:
            row = db.execute(
                "SELECT id, src_id, dst_id, relation, weight, meta, created FROM edges WHERE id=?",
                (edge_id,)
            ).fetchone()
        
        if row:
            edge_data = {
                "id": row[0],
                "src_id": row[1],
                "dst_id": row[2],
                "relation": row[3],
                "weight": row[4],
                "meta": self._parse_meta(row[5]),
                "created": row[6]
            }
            self._set_in_cache(cache_key, edge_data)
            return edge_data
        return None

    def best_skills_for_goal_type(self, goal_type: str, limit: int = 5) -> List[str]:
        """Return skill names most associated with successful cycles of this goal_type."""
        with self._conn() as db:
            rows = db.execute("""
            SELECT n_skill.label, SUM(e2.weight) AS score
            FROM nodes n_cycle
            JOIN edges e1 ON e1.src_id = n_cycle.id AND e1.relation = 'uses'
            JOIN nodes n_skill ON n_skill.id = e1.dst_id AND n_skill.type = 'skill'
            JOIN edges e2 ON e2.src_id = n_cycle.id AND e2.relation = 'succeeded_on'
            WHERE n_cycle.type = 'cycle'
              AND json_extract(n_cycle.meta, '$.goal_type') = ?
            GROUP BY n_skill.label
            ORDER BY score DESC
            LIMIT ?
            """, (goal_type, limit)).fetchall()
        return [r[0] for r in rows]

    def goals_touching_file(self, file_path: str, limit: int = 10) -> List[Dict]:
        """Return recent goals that modified or mentioned a file."""
        with self._conn() as db:
            rows = db.execute("""
            SELECT n_goal.label, n_goal.meta, e.relation
            FROM nodes n_file
            JOIN edges e ON (e.src_id = n_file.id OR e.dst_id = n_file.id)
            JOIN nodes n_goal ON
                (n_goal.id = e.dst_id OR n_goal.id = e.src_id)
                AND n_goal.type = 'goal'
            WHERE n_file.type = 'file'
              AND n_file.label = ?
            ORDER BY e.created DESC
            LIMIT ?
            """, (file_path, limit)).fetchall()
        return [{"goal": r[0], "meta": self._parse_meta(r[1]), "relation": r[2]}
                for r in rows]

    def query_similar_resolutions(self, file_path: str, limit: int = 5) -> List[Dict]:
        """Find past goals that fixed failures related to a file."""
        with self._conn() as db:
            rows = db.execute("""
            SELECT n_goal.label, n_goal.meta
            FROM nodes n_file
            JOIN edges e ON e.src_id = n_file.id AND e.relation = 'fixed_by'
            JOIN nodes n_goal ON n_goal.id = e.dst_id AND n_goal.type = 'goal'
            WHERE n_file.label = ?
            ORDER BY e.created DESC
            LIMIT ?
            """, (file_path, limit)).fetchall()
        return [{"goal": r[0], "meta": self._parse_meta(r[1])} for r in rows]

    def weaknesses_for_goal_type(self, goal_type: str, limit: int = 5) -> List[str]:
        """Return weakness labels associated with a goal type."""
        with self._conn() as db:
            rows = db.execute("""
            SELECT DISTINCT n_w.label
            FROM nodes n_cycle
            JOIN edges e ON e.src_id = n_cycle.id AND e.relation = 'generated'
            JOIN nodes n_w ON n_w.id = e.dst_id AND n_w.type = 'weakness'
            WHERE n_cycle.type = 'cycle'
              AND json_extract(n_cycle.meta, '$.goal_type') = ?
            ORDER BY e.created DESC
            LIMIT ?
            """, (goal_type, limit)).fetchall()
        return [r[0] for r in rows]

    def file_failure_count(self, file_path: str) -> int:
        """How many times has a file appeared in failed change sets?"""
        with self._conn() as db:
            row = db.execute("""
            SELECT COUNT(*) FROM nodes n
            JOIN edges e ON e.src_id = n.id AND e.relation = 'failed_on'
            WHERE n.type = 'file' AND n.label = ?
            """, (file_path,)).fetchone()
        return row[0] if row else 0

    def graph_summary(self) -> Dict[str, int]:
        """Return node/edge counts per type for observability."""
        cache_key = "graph_summary"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        
        with self._conn() as db:
            node_counts = dict(db.execute(
                "SELECT type, COUNT(*) FROM nodes GROUP BY type"
            ).fetchall())
            edge_counts = dict(db.execute(
                "SELECT relation, COUNT(*) FROM edges GROUP BY relation"
            ).fetchall())
        
        result = {"nodes": node_counts, "edges": edge_counts}
        self._set_in_cache(cache_key, result)
        return result

    def get_nx_graph(self, relations: Optional[List[str]] = None):
        """Export the graph to a networkx.DiGraph for complex analysis.
        
        Optimized to use batch queries instead of N+1 pattern.
        """
        import networkx as nx
        G = nx.DiGraph()
        
        with self._conn() as db:
            # Batch fetch all nodes
            nodes = db.execute("SELECT id, type, label, meta FROM nodes").fetchall()
            for nid, ntype, label, meta in nodes:
                G.add_node(nid, type=ntype, label=label, meta=self._parse_meta(meta))
            
            # Build query for edges
            query = "SELECT src_id, dst_id, relation, weight FROM edges"
            params = []
            if relations:
                query += " WHERE relation IN ({})".format(",".join(["?"] * len(relations)))
                params = relations
            
            # Batch fetch all edges
            edges = db.execute(query, params).fetchall()
            for src, dst, rel, weight in edges:
                G.add_edge(src, dst, relation=rel, weight=weight)
        
        return G

    def find_circular_dependencies(self) -> List[List[str]]:
        """Identify cycles in the 'imports' and 'calls' graph."""
        import networkx as nx
        G = self.get_nx_graph(relations=["imports", "imports_from", "calls"])
        cycles = list(nx.simple_cycles(G))
        # Map IDs back to labels
        labels = nx.get_node_attributes(G, 'label')
        return [[labels.get(node, node) for node in cycle] for cycle in cycles]

    def find_bottleneck_files(self, limit: int = 5) -> List[Dict]:
        """Find files with high centrality (highly depended on)."""
        import networkx as nx
        G = self.get_nx_graph(relations=["imports", "imports_from", "calls"])
        if not G.nodes:
            return []
        
        try:
            # PageRank often requires scipy for performance
            centrality = nx.pagerank(G, weight='weight')
        except (ImportError, OSError, IOError, ValueError):
            # Fallback to degree centrality (no extra dependencies)
            log_json("INFO", "pagerank_failed_fallback_to_degree")
            centrality = nx.degree_centrality(G)
            
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        
        labels = nx.get_node_attributes(G, 'label')
        results = []
        for nid, score in sorted_nodes:
            if len(results) >= limit:
                break
            # Ensure node still exists and is a file
            if nid in G.nodes and G.nodes[nid].get('type') == 'file':
                results.append({
                    "file": labels.get(nid, nid),
                    "centrality_score": round(score, 4)
                })
        return results

    # ── Ingest logic (optimized with batching) ────────────────────────────────

    def _ingest(self, entry: Dict) -> None:
        now = time.time()
        cycle_id = entry.get("cycle_id", str(uuid.uuid4()))
        goal_type = entry.get("goal_type", "unknown")
        po = entry.get("phase_outputs", {})

        # ── Cycle node ───────────────────────────────────────────────────────
        cycle_nid = self._upsert_node(
            cycle_id, "cycle", cycle_id,
            {"goal_type": goal_type, "ts": now},
        )

        # ── Verification outcome ─────────────────────────────────────────────
        verif = po.get("verification", {})
        passed = isinstance(verif, dict) and verif.get("status") in ("pass", "skip")

        # ── Applied files → nodes + edges ────────────────────────────────────
        apply_result = po.get("apply_result", {})
        
        # Batch upsert nodes for applied files
        applied_files = apply_result.get("applied", [])
        if applied_files:
            file_nodes = [(f"file:{fp}", "file", fp, {}) for fp in applied_files]
            self._upsert_nodes_batch(file_nodes)
            
            # Batch create edges
            edges = []
            for fp in applied_files:
                fnid = f"file:{fp}"
                edges.append((cycle_nid, fnid, "succeeded_on" if passed else "failed_on", 1.0, {}))
                
                if passed:
                    goal_text = po.get("context", {}).get("goal", "")
                    if goal_text:
                        gnid = f"goal:{goal_text[:60]}"
                        self._upsert_node(gnid, "goal", goal_text[:120], {})
                        edges.append((fnid, gnid, "fixed_by", 1.0, {}))
                        edges.append((cycle_nid, gnid, "generated", 1.0, {}))
            
            if edges:
                self._upsert_edges_batch(edges)

        # Failed files
        failed_items = apply_result.get("failed", [])
        if failed_items:
            failed_files = [item.get("file", "") for item in failed_items if item.get("file")]
            file_nodes = [(f"file:{fp}", "file", fp, {}) for fp in failed_files]
            self._upsert_nodes_batch(file_nodes)
            
            edges = [(cycle_nid, f"file:{fp}", "failed_on", 1.0, {}) for fp in failed_files]
            self._upsert_edges_batch(edges)

        # ── Skills used → nodes + edges ──────────────────────────────────────
        skill_context = po.get("skill_context", {})
        if skill_context:
            skill_nodes = [(f"skill:{sk_name}", "skill", sk_name, {}) 
                          for sk_name in skill_context.keys()]
            self._upsert_nodes_batch(skill_nodes)
            
            edges = []
            for sk_name, sk_result in skill_context.items():
                snid = f"skill:{sk_name}"
                edges.append((cycle_nid, snid, "uses", 1.0, {}))
                if passed:
                    edges.append((cycle_nid, snid, "succeeded_on", 1.0, {}))
            
            self._upsert_edges_batch(edges)

        # ── Learnings → nodes + edges ────────────────────────────────────────
        learnings = po.get("reflection", {}).get("learnings", [])
        if learnings:
            learning_nodes = [(f"learning:{item[:50]}", "learning", item[:200], {}) 
                             for item in learnings]
            self._upsert_nodes_batch(learning_nodes)
            
            edges = [(cycle_nid, f"learning:{item[:50]}", "generated", 1.0, {}) 
                    for item in learnings]
            self._upsert_edges_batch(edges)

    # ── Node / Edge helpers (with batching support) ───────────────────────────

    def _upsert_node(
        self, node_id: str, node_type: str, label: str, meta: Dict
    ) -> str:
        """Upsert a single node."""
        now = time.time()
        with self._conn() as db:
            existing = db.execute(
                "SELECT id FROM nodes WHERE id=?", (node_id,)
            ).fetchone()
            if existing:
                db.execute(
                    "UPDATE nodes SET updated=?, meta=? WHERE id=?",
                    (now, json.dumps(meta), node_id),
                )
            else:
                db.execute(
                    "INSERT INTO nodes VALUES (?,?,?,?,?,?)",
                    (node_id, node_type, label, json.dumps(meta), now, now),
                )
        
        # Invalidate cache for this node
        self._invalidate_cache(f"node:{node_id}")
        return node_id

    def _upsert_nodes_batch(self, nodes: List[Tuple[str, str, str, Dict]]) -> None:
        """Batch upsert multiple nodes (avoids N+1 inserts)."""
        if not nodes:
            return
        
        now = time.time()
        with self._conn() as db:
            # Use transaction for atomicity
            db.execute("BEGIN TRANSACTION")
            try:
                for node_id, node_type, label, meta in nodes:
                    existing = db.execute(
                        "SELECT id FROM nodes WHERE id=?", (node_id,)
                    ).fetchone()
                    if existing:
                        db.execute(
                            "UPDATE nodes SET updated=?, meta=? WHERE id=?",
                            (now, json.dumps(meta), node_id),
                        )
                    else:
                        db.execute(
                            "INSERT INTO nodes VALUES (?,?,?,?,?,?)",
                            (node_id, node_type, label, json.dumps(meta), now, now),
                        )
                db.execute("COMMIT")
            except Exception:
                db.execute("ROLLBACK")
                raise
        
        # Invalidate cache for all affected nodes
        for node_id, _, _, _ in nodes:
            self._invalidate_cache(f"node:{node_id}")

    def _upsert_edge(
        self,
        src: str, dst: str, relation: str,
        weight: float = 1.0,
        meta: Optional[Dict] = None,
    ) -> None:
        """Upsert a single edge."""
        edge_id = f"{src}:{relation}:{dst}"
        now = time.time()
        with self._conn() as db:
            existing = db.execute(
                "SELECT id, weight FROM edges WHERE id=?", (edge_id,)
            ).fetchone()
            if existing:
                # Reinforce existing edges
                db.execute(
                    "UPDATE edges SET weight=weight+?, created=? WHERE id=?",
                    (weight * 0.1, now, edge_id),
                )
            else:
                db.execute(
                    "INSERT INTO edges VALUES (?,?,?,?,?,?,?)",
                    (edge_id, src, dst, relation,
                     weight, json.dumps(meta or {}), now),
                )
        
        # Invalidate cache
        self._invalidate_cache(f"edge:{edge_id}")

    def _upsert_edges_batch(self, edges: List[Tuple[str, str, str, float, Dict]]) -> None:
        """Batch upsert multiple edges (avoids N+1 inserts)."""
        if not edges:
            return
        
        now = time.time()
        with self._conn() as db:
            db.execute("BEGIN TRANSACTION")
            try:
                for src, dst, relation, weight, meta in edges:
                    edge_id = f"{src}:{relation}:{dst}"
                    existing = db.execute(
                        "SELECT id, weight FROM edges WHERE id=?", (edge_id,)
                    ).fetchone()
                    if existing:
                        # Reinforce existing edges
                        db.execute(
                            "UPDATE edges SET weight=weight+?, created=? WHERE id=?",
                            (weight * 0.1, now, edge_id),
                        )
                    else:
                        db.execute(
                            "INSERT INTO edges VALUES (?,?,?,?,?,?,?)",
                            (edge_id, src, dst, relation,
                             weight, json.dumps(meta or {}), now),
                        )
                db.execute("COMMIT")
            except Exception:
                db.execute("ROLLBACK")
                raise
        
        # Invalidate cache for all affected edges
        for src, dst, relation, _, _ in edges:
            edge_id = f"{src}:{relation}:{dst}"
            self._invalidate_cache(f"edge:{edge_id}")

    def add_edge(self, src_label: str, dst_label: str, relation: str, weight: float = 1.0):
        """Public helper to add a relationship between two generic nodes (defaults to 'file' type)."""
        src_id = f"file:{src_label}"
        dst_id = f"file:{dst_label}"
        self._upsert_node(src_id, "file", src_label, {})
        self._upsert_node(dst_id, "file", dst_label, {})
        self._upsert_edge(src_id, dst_id, relation, weight)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _parse_meta(raw: str) -> Dict:
        try:
            return json.loads(raw)
        except (OSError, IOError, ValueError):
            return {}
