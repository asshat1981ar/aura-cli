"""
Project Knowledge Syncer — Semantic indexing and graph synchronization.

Performs recursive semantic chunking of the codebase using AST parsing,
hydrates the VectorStore with symbol-level MemoryRecords, and syncs
structural relationships (imports/calls) into the ContextGraph.
"""
from __future__ import annotations

import ast
import json
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
        self._hash_map_path = self.root / "memory" / "project_sync_hashes.json"
        self._hashes: Dict[str, str] = self._load_hashes()
        
    def _load_hashes(self) -> Dict[str, str]:
        try:
            if self._hash_map_path.exists():
                return json.loads(self._hash_map_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def _save_hashes(self):
        try:
            self._hash_map_path.parent.mkdir(parents=True, exist_ok=True)
            self._hash_map_path.write_text(json.dumps(self._hashes, indent=2), encoding="utf-8")
        except Exception as e:
            log_json("WARN", "project_sync_save_hashes_failed", details={"error": str(e)})

    def sync_all(self) -> Dict[str, Any]:
        """Perform a full sync of the project knowledge. Never raises."""
        log_json("INFO", "project_sync_start", details={"root": str(self.root)})
        start_time = time.time()
        
        stats = {
            "files_processed": 0,
            "files_skipped": 0,
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
                        rel_path = str(py_file.relative_to(self.root))
                        content = py_file.read_text(encoding="utf-8", errors="replace")
                        current_hash = hashlib.sha256(content.encode()).hexdigest()
                        
                        if self._hashes.get(rel_path) == current_hash:
                            stats["files_skipped"] += 1
                            continue

                        f_stats = self._sync_file(py_file, content)
                        self._hashes[rel_path] = current_hash
                        
                        stats["files_processed"] += 1
                        stats["chunks_created"] += f_stats["chunks"]
                        stats["relationships_extracted"] += f_stats["relations"]
                    except Exception as exc:
                        stats["errors"] += 1
                        log_json("WARN", "project_sync_file_failed", 
                                 details={"file": str(py_file), "error": str(exc)})

            self._save_hashes()
            log_json("INFO", "project_sync_complete", details={
                "duration_sec": round(time.time() - start_time, 2),
                **stats
            })
        except Exception as exc:
            log_json("ERROR", "project_sync_critical_failure", details={"error": str(exc)})
            
        return stats

    def _sync_file(self, path: Path, content: str) -> Dict[str, int]:
        """Index a single file: chunking + relationships."""
        rel_path = str(path.relative_to(self.root))
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return {"chunks": 0, "relations": 0}

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

        # 4. Extract Relationships (Imports & Calls)
        if self.cg:
            imports_map = {} # alias -> module_name
            
            # First pass: gather imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        target = alias.name
                        name = alias.asname or alias.name
                        imports_map[name] = target
                        try:
                            self.cg.add_edge(rel_path, target, "imports")
                            relations_count += 1
                        except Exception: continue
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            target = f"{node.module}.{alias.name}"
                            name = alias.asname or alias.name
                            imports_map[name] = target
                            try:
                                self.cg.add_edge(rel_path, node.module, "imports_from")
                                relations_count += 1
                            except Exception: continue

            # Second pass: gather calls and link to imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    called_name = None
                    if isinstance(node.func, ast.Name):
                        called_name = node.func.id
                    elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                        # e.g. os.path.join -> module 'os.path' if 'os' imported, or alias 'os'
                        called_name = node.func.value.id
                    
                    if called_name and called_name in imports_map:
                        target_module = imports_map[called_name]
                        try:
                            # Edge: file -> module (relation="calls")
                            # We interpret this as "depends on functionality from"
                            self.cg.add_edge(rel_path, target_module, "calls", weight=0.5)
                            relations_count += 1
                        except Exception: continue

        return {"chunks": len(chunks), "relations": relations_count}

    def _should_skip(self, path: Path) -> bool:
        parts = path.parts
        return any(skip in parts for skip in _SKIP_PATTERNS)
