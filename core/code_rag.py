"""RAG-augmented code generation: retrieve past implementations before Act phase.

Before generating code, retrieves the top-K most similar past successful
implementations from the vector store and injects them as few-shot examples.
This reduces retries by grounding generation in proven patterns.
"""

import time
from dataclasses import dataclass, field
from core.logging_utils import log_json


@dataclass
class RAGContext:
    """Retrieved context for code generation."""

    examples: list[dict] = field(default_factory=list)  # past implementations
    patterns: list[str] = field(default_factory=list)  # code patterns
    anti_patterns: list[str] = field(default_factory=list)  # what to avoid
    retrieval_time_ms: float = 0.0
    total_tokens: int = 0


class CodeRAG:
    """Retrieval-Augmented Generation for code — grounds generation in past successes."""

    def __init__(self, vector_store=None, brain=None, max_examples: int = 3, max_tokens: int = 2000, min_similarity: float = 0.6):
        self.vector_store = vector_store
        self.brain = brain
        self.max_examples = max_examples
        self.max_tokens = max_tokens
        self.min_similarity = min_similarity

    def retrieve_context(self, goal: str, task_bundle: dict | None = None) -> RAGContext:
        """Retrieve relevant past implementations for a goal."""
        t0 = time.time()
        context = RAGContext()

        if not self.vector_store and not self.brain:
            return context

        # 1. Search for similar past goals/implementations
        examples = self._search_implementations(goal)
        context.examples = examples[: self.max_examples]

        # 2. Search for relevant code patterns
        if task_bundle:
            files = self._extract_target_files(task_bundle)
            patterns = self._search_patterns(goal, files)
            context.patterns = patterns

        # 3. Search for anti-patterns (past failures)
        anti_patterns = self._search_failures(goal)
        context.anti_patterns = anti_patterns

        context.retrieval_time_ms = (time.time() - t0) * 1000
        context.total_tokens = sum(len(str(e).split()) for e in context.examples)

        log_json(
            "INFO",
            "code_rag_retrieved",
            details={
                "examples": len(context.examples),
                "patterns": len(context.patterns),
                "anti_patterns": len(context.anti_patterns),
                "retrieval_ms": round(context.retrieval_time_ms, 1),
            },
        )
        return context

    def augment_prompt(self, base_prompt: str, rag_context: RAGContext) -> str:
        """Inject RAG context into the code generation prompt."""
        sections = []

        if rag_context.examples:
            sections.append("## Similar Past Implementations (proven to work)")
            for i, ex in enumerate(rag_context.examples[: self.max_examples]):
                content = ex.get("content", str(ex))[:500]
                source = ex.get("source", "memory")
                sections.append(f"### Example {i + 1} (from {source}):\n{content}\n")

        if rag_context.patterns:
            sections.append("## Relevant Patterns")
            for p in rag_context.patterns[:3]:
                sections.append(f"- {p[:200]}")

        if rag_context.anti_patterns:
            sections.append("## Known Pitfalls (avoid these)")
            for ap in rag_context.anti_patterns[:3]:
                sections.append(f"- {ap[:200]}")

        if not sections:
            return base_prompt

        rag_section = "\n".join(sections)
        return f"{base_prompt}\n\n{rag_section}"

    def store_successful_implementation(self, goal: str, changes: list[dict], goal_type: str = ""):
        """Store a successful implementation for future RAG retrieval."""
        if not self.vector_store:
            return

        try:
            from core.memory_types import MemoryRecord
            import uuid
            import hashlib

            for change in changes[:5]:  # Limit storage
                file_path = change.get("file_path", "unknown")
                new_code = change.get("new_code", "")
                if not new_code or len(new_code) < 10:
                    continue

                content = f"Goal: {goal}\nFile: {file_path}\nCode:\n{new_code[:1000]}"
                record = MemoryRecord(
                    id=uuid.uuid4().hex[:16],
                    content=content,
                    source_type="implementation",
                    source_ref=file_path,
                    created_at=time.time(),
                    updated_at=time.time(),
                    tags=["implementation", goal_type] if goal_type else ["implementation"],
                    importance=0.8,
                    content_hash=hashlib.sha256(content.encode()).hexdigest(),
                )
                self.vector_store.upsert([record])
        except Exception as exc:
            log_json("WARN", "code_rag_store_failed", details={"error": str(exc)})

    def _search_implementations(self, goal: str) -> list[dict]:
        """Search vector store for similar past implementations."""
        results = []
        try:
            if self.vector_store:
                from core.memory_types import RetrievalQuery

                query = RetrievalQuery(
                    query_text=goal,
                    k=self.max_examples * 2,
                    min_score=self.min_similarity,
                    filters={"source_type": "implementation"},
                    budget_tokens=self.max_tokens,
                )
                hits = self.vector_store.search(query)
                if isinstance(hits, list):
                    for hit in hits:
                        if isinstance(hit, str):
                            results.append({"content": hit, "source": "vector_store"})
                        elif hasattr(hit, "content"):
                            results.append({"content": hit.content, "source": hit.source_ref, "score": getattr(hit, "score", 0)})
        except Exception as exc:
            log_json("WARN", "code_rag_search_failed", details={"error": str(exc)})

        # Also check brain memory
        if self.brain and not results:
            try:
                memories = self.brain.recall_with_budget(max_tokens=self.max_tokens)
                for m in memories[: self.max_examples]:
                    if any(kw in m.lower() for kw in ["implement", "code", "function", "class"]):
                        results.append({"content": m, "source": "brain"})
            except (OSError, IOError, ValueError):
                pass

        return results

    def _search_patterns(self, goal: str, files: list[str]) -> list[str]:
        """Search for code patterns relevant to target files."""
        patterns = []
        if self.brain:
            try:
                for f in files[:3]:
                    memories = self.brain.recall_with_budget(max_tokens=500)
                    for m in memories:
                        if f in m or any(kw in m.lower() for kw in ["pattern", "convention", "style"]):
                            patterns.append(m[:200])
            except (OSError, IOError, ValueError):
                pass
        return patterns[:5]

    def _search_failures(self, goal: str) -> list[str]:
        """Search for past failures related to this goal."""
        failures = []
        try:
            from memory.consolidation import NegativeExampleStore
            from pathlib import Path

            store_path = Path(__file__).parent.parent / "memory" / "negative_examples.json"
            if store_path.exists():
                store = NegativeExampleStore(store_path)
                similar = store.find_similar_failures(goal, limit=3)
                for f in similar:
                    failures.append(f"Failed: {f['goal']} — Reason: {f['failure_reason']}")
        except (ImportError, OSError, IOError, ValueError):
            pass
        return failures

    def _extract_target_files(self, task_bundle: dict) -> list[str]:
        """Extract target file paths from task bundle."""
        files = []
        tasks = task_bundle.get("tasks", [])
        for task in tasks[:5]:
            if isinstance(task, dict):
                for f in task.get("files", []):
                    files.append(str(f))
                target = task.get("target_file", "")
                if target:
                    files.append(str(target))
        return list(set(files))
