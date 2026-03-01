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
from typing import Any, Dict, List, Optional

from core.logging_utils import log_json

_DB_PATH = Path(__file__).resolve().parent.parent / "memory" / "context_graph.db"


class ContextGraph:
    """Persistent property graph — grows richer with every cycle.

    Thread-safe via SQLite WAL mode + per-connection isolation.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path or _DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
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

    # ── Public API ───────────────────────────────────────────────────────────

    def update_from_cycle(self, cycle_entry: Dict[str, Any]) -> None:
        """Ingest a completed cycle entry and add/update nodes + edges."""
        try:
            self._ingest(cycle_entry)
        except Exception as exc:
            log_json("WARN", "context_graph_ingest_failed", details={"error": str(exc)})

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
        with self._conn() as db:
            node_counts = dict(db.execute(
                "SELECT type, COUNT(*) FROM nodes GROUP BY type"
            ).fetchall())
            edge_counts = dict(db.execute(
                "SELECT relation, COUNT(*) FROM edges GROUP BY relation"
            ).fetchall())
        return {"nodes": node_counts, "edges": edge_counts}

    def get_nx_graph(self, relations: Optional[List[str]] = None):
        """Export the graph to a networkx.DiGraph for complex analysis."""
        import networkx as nx
        G = nx.DiGraph()
        
        with self._conn() as db:
            nodes = db.execute("SELECT id, type, label, meta FROM nodes").fetchall()
            for nid, ntype, label, meta in nodes:
                G.add_node(nid, type=ntype, label=label, meta=self._parse_meta(meta))
            
            query = "SELECT src_id, dst_id, relation, weight FROM edges"
            params = []
            if relations:
                query += " WHERE relation IN ({})".format(",".join(["?"] * len(relations)))
                params = relations
            
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
        except Exception:
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

    # ── Ingest logic ─────────────────────────────────────────────────────────

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
        for fp in apply_result.get("applied", []):
            fnid = self._upsert_node(f"file:{fp}", "file", fp, {})
            self._upsert_edge(cycle_nid, fnid, "succeeded_on" if passed else "failed_on")
            if passed:
                goal_text = po.get("context", {}).get("goal", "")
                if goal_text:
                    gnid = self._upsert_node(f"goal:{goal_text[:60]}", "goal", goal_text[:120], {})
                    self._upsert_edge(fnid, gnid, "fixed_by")
                    self._upsert_edge(cycle_nid, gnid, "generated")

        for item in apply_result.get("failed", []):
            fp = item.get("file", "")
            if fp:
                fnid = self._upsert_node(f"file:{fp}", "file", fp, {})
                self._upsert_edge(cycle_nid, fnid, "failed_on")

        # ── Skills used → nodes + edges ──────────────────────────────────────
        for sk_name, sk_result in po.get("skill_context", {}).items():
            snid = self._upsert_node(f"skill:{sk_name}", "skill", sk_name, {})
            self._upsert_edge(cycle_nid, snid, "uses")
            if passed:
                self._upsert_edge(cycle_nid, snid, "succeeded_on")

        # ── Weaknesses → nodes + edges ───────────────────────────────────────
        for item in po.get("reflection", {}).get("learnings", []):
            wnid = self._upsert_node(f"learning:{item[:50]}", "learning", item[:200], {})
            self._upsert_edge(cycle_nid, wnid, "generated")

    # ── Node / Edge helpers ───────────────────────────────────────────────────

    def _upsert_node(
        self, node_id: str, node_type: str, label: str, meta: Dict
    ) -> str:
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
        return node_id

    def _upsert_edge(
        self,
        src: str, dst: str, relation: str,
        weight: float = 1.0,
        meta: Optional[Dict] = None,
    ) -> None:
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
        except Exception:
            return {}
