"""Query interface for the semantic codebase index."""
from __future__ import annotations

import logging
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class SemanticQuerier:
    """Read-only query interface for the semantic index."""

    def __init__(self, db_path: Path) -> None:
        from core.agent_sdk.semantic_schema import SemanticDB

        self._db = SemanticDB(db_path)

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def what_calls(self, symbol_name: str) -> List[Dict[str, Any]]:
        """Find all callers of a symbol.

        Queries call_sites joined with symbols and files where callee_name matches.
        Returns [{callee_name, caller, kind, path, line}].
        """
        rows = self._db.conn.execute(
            """
            SELECT cs.callee_name, s.name AS caller, s.kind, f.path, cs.line
            FROM call_sites cs
            JOIN symbols s ON s.id = cs.caller_symbol_id
            JOIN files f ON f.id = s.file_id
            WHERE cs.callee_name = ?
            """,
            (symbol_name,),
        ).fetchall()
        return [dict(row) for row in rows]

    def what_depends_on(self, file_path: str) -> List[Dict[str, Any]]:
        """Find files that import from the given file.

        Looks up file by path, then queries relationships where to_file_id matches.
        Returns [{path, rel_type, strength}].
        """
        target = self._db.get_file_by_path(file_path)
        if target is None:
            return []

        rows = self._db.conn.execute(
            """
            SELECT f.path, r.rel_type, r.strength
            FROM relationships r
            JOIN files f ON f.id = r.from_file_id
            WHERE r.to_file_id = ?
            """,
            (target["id"],),
        ).fetchall()
        return [dict(row) for row in rows]

    def what_changes_break(self, file_path: str, depth: int = 2) -> List[Dict[str, Any]]:
        """Transitive dependents — ripple analysis.

        BFS walk from target file through reverse relationships up to depth.
        Returns [{path, distance, relationship}].
        """
        target = self._db.get_file_by_path(file_path)
        if target is None:
            return []

        results: List[Dict[str, Any]] = []
        visited: set[int] = set()

        def _walk(file_id: int, dist: int) -> None:
            if dist > depth or file_id in visited:
                return
            visited.add(file_id)
            rows = self._db.conn.execute(
                """
                SELECT f.path, r.rel_type, r.from_file_id
                FROM relationships r
                JOIN files f ON f.id = r.from_file_id
                WHERE r.to_file_id = ?
                """,
                (file_id,),
            ).fetchall()
            for row in rows:
                results.append(
                    {
                        "path": row["path"],
                        "distance": dist,
                        "relationship": row["rel_type"],
                    }
                )
                _walk(row["from_file_id"], dist + 1)

        _walk(target["id"], 1)
        return results

    def summarize(self, target: str) -> str:
        """Return summary for a file or symbol.

        Tries target as a file path first (returns module_summary).
        Then tries as symbol name (returns intent_summary or docstring).
        Fallback: 'No summary available for: {target}'.
        """
        # Try as file path
        file_row = self._db.get_file_by_path(target)
        if file_row is not None:
            summary = file_row.get("module_summary")
            if summary:
                return summary

        # Try as symbol name
        symbol_rows = self._db.get_symbol_by_name(target)
        if symbol_rows:
            sym = symbol_rows[0]
            intent = sym.get("intent_summary")
            if intent:
                return intent
            doc = sym.get("docstring")
            if doc:
                return doc

        return f"No summary available for: {target}"

    def find_similar(self, description: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find symbols matching a description via FTS5.

        Falls back to keyword_search if FTS5 is not available.
        Returns list of symbol dicts with path.
        """
        try:
            return self._db.fts_search(description, limit=limit)
        except RuntimeError:
            logger.debug("FTS5 unavailable, falling back to keyword_search")
            return self._db.keyword_search(description, limit=limit)

    def architecture_overview(self) -> Dict[str, Any]:
        """Compact architectural overview.

        Returns {total_files, clusters: {name: count},
                 top_coupled: [{path, coupling, summary}], summary: str}.
        Groups files by cluster field; sorts by coupling_score desc, takes top 5.
        """
        all_files = self._db.get_all_files()
        total_files = len(all_files)

        # Group by cluster
        clusters: Dict[str, int] = defaultdict(int)
        for f in all_files:
            cluster = f.get("cluster") or "unknown"
            clusters[cluster] += 1

        # Top 5 by coupling score
        sorted_files = sorted(all_files, key=lambda f: f.get("coupling_score") or 0.0, reverse=True)
        top_coupled = [
            {
                "path": f["path"],
                "coupling": f.get("coupling_score") or 0.0,
                "summary": f.get("module_summary") or "",
            }
            for f in sorted_files[:5]
        ]

        summary = self._build_overview_text(total_files, clusters, top_coupled)

        return {
            "total_files": total_files,
            "clusters": dict(clusters),
            "top_coupled": top_coupled,
            "summary": summary,
        }

    def recent_changes(self, n_commits: int = 5) -> List[Dict[str, Any]]:
        """Changed files with summaries since N commits ago.

        Uses git log --name-only to find changed .py files and looks them up in DB.
        Returns [{path, summary, coupling}].
        """
        try:
            result = subprocess.run(
                ["git", "log", f"-{n_commits}", "--name-only", "--pretty=format:"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                logger.warning("git_log_failed: %s", result.stderr.strip())
                return []
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            logger.warning("git_log_unavailable: %s", exc)
            return []

        # Collect unique .py paths preserving first-seen order
        seen: set[str] = set()
        changed_paths: list[str] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.endswith(".py") and line not in seen:
                seen.add(line)
                changed_paths.append(line)

        output: List[Dict[str, Any]] = []
        for path in changed_paths:
            file_row = self._db.get_file_by_path(path)
            if file_row is not None:
                output.append(
                    {
                        "path": path,
                        "summary": file_row.get("module_summary") or "",
                        "coupling": file_row.get("coupling_score") or 0.0,
                    }
                )

        return output

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_overview_text(
        self,
        total: int,
        clusters: Dict[str, int],
        top_coupled: List[Dict[str, Any]],
    ) -> str:
        """Produce a compact overview string."""
        n_clusters = len(clusters)
        cluster_parts = ", ".join(
            f"{name} ({count} {'file' if count == 1 else 'files'})"
            for name, count in sorted(clusters.items(), key=lambda kv: -kv[1])
        )
        hub_paths = ", ".join(item["path"] for item in top_coupled[:2]) if top_coupled else "none"
        return (
            f"{total} {'file' if total == 1 else 'files'} in {n_clusters} "
            f"{'cluster' if n_clusters == 1 else 'clusters'}: {cluster_parts}. "
            f"Key hubs: {hub_paths}."
        )
