"""
Project Knowledge Syncer — Semantic indexing and graph synchronization.

Performs recursive semantic chunking of the codebase using AST parsing,
hydrates the VectorStore with symbol-level MemoryRecords, and syncs
structural relationships (imports/calls) into the ContextGraph.
"""
from __future__ import annotations

import ast
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from core.logging_utils import log_json
from core.memory_types import MemoryRecord

# Directories to index
_SOURCE_DIRS = ["agents", "core", "aura_cli", "memory", "tools", "cli"]
_SKIP_PATTERNS = ["__pycache__", ".git", "node_modules", ".pytest_cache", ".ralph-state", ".gemini"]

class ProjectKnowledgeSyncer:
    """Recursively indexes the project for semantic and relational depth."""

    def __init__(
        self,
        vector_store,
        context_graph,
        project_root: str = "."
    ):
        self.vs = vector_store
        self.cg = context_graph
        self.root = Path(project_root)
        
    def sync_all(self) -> Dict[str, Any]:
        """Perform a full sync of the project knowledge. Never raises."""
        log_json("INFO", "project_sync_start", details={"root": str(self.root)})
        start_time = time.time()
        
        stats = {
            "files_processed": 0,
            "chunks_created": 0,
            "relationships_extracted": 0,
            "errors": 0
        }

        try:
            for src_dir in _SOURCE_DIRS:
                dir_path = self.root / src_dir
                if not dir_path.is_dir():
                    continue
                
                for py_file in dir_path.rglob("*.py"):
                    if self._should_skip(py_file):
                        continue
                    
                    try:
                        f_stats = self._sync_file(py_file)
                        stats["files_processed"] += 1
                        stats["chunks_created"] += f_stats["chunks"]
                        stats["relationships_extracted"] += f_stats["relations"]
                    except Exception as exc:
                        stats["errors"] += 1
                        log_json("WARN", "project_sync_file_failed", 
                                 details={"file": str(py_file), "error": str(exc)})

            log_json("INFO", "project_sync_complete", details={
                "duration_sec": round(time.time() - start_time, 2),
                **stats
            })
        except Exception as exc:
            log_json("ERROR", "project_sync_critical_failure", details={"error": str(exc)})
            
        return stats

    def _sync_file(self, path: Path) -> Dict[str, int]:
        """Index a single file: chunking + relationships."""
        rel_path = str(path.relative_to(self.root))
        content = path.read_text(encoding="utf-8", errors="replace")
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # 1. Check for incremental update skip
        # (For now, we re-index, but in v2.1 we check content_hash in DB)
        
        tree = ast.parse(content)
        chunks: List[MemoryRecord] = []
        relations_count = 0

        # 2. Extract Chunks (Symbols)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk_content = ast.get_source_segment(content, node)
                if not chunk_content:
                    continue
                
                symbol_name = node.name
                line_start = node.lineno
                
                rec = MemoryRecord(
                    id=f"{rel_path}:{symbol_name}",
                    content=chunk_content,
                    source_type="file",
                    source_ref=f"{rel_path}:{line_start}",
                    created_at=time.time(),
                    updated_at=time.time(),
                    content_hash=hashlib.sha256(chunk_content.encode()).hexdigest(),
                    tags=[type(node).__name__, symbol_name]
                )
                chunks.append(rec)

        # 3. Upsert to VectorStore
        if chunks:
            self.vs.upsert(chunks)

        # 4. Extract Relationships (Imports)
        if self.cg:
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        try:
                            self.cg.add_edge(rel_path, alias.name, "imports")
                            relations_count += 1
                        except Exception: continue
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        try:
                            self.cg.add_edge(rel_path, node.module, "imports_from")
                            relations_count += 1
                        except Exception: continue

        return {"chunks": len(chunks), "relations": relations_count}

    def _should_skip(self, path: Path) -> bool:
        parts = path.parts
        return any(skip in parts for skip in _SKIP_PATTERNS)
