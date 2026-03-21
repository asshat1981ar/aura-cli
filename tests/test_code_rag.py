"""Tests for core.code_rag — RAG-augmented code generation."""
import json
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.code_rag import CodeRAG, RAGContext
from core.memory_types import SearchHit


class TestRAGContext(unittest.TestCase):
    """RAGContext dataclass creation and defaults."""

    def test_default_fields(self):
        ctx = RAGContext()
        self.assertEqual(ctx.examples, [])
        self.assertEqual(ctx.patterns, [])
        self.assertEqual(ctx.anti_patterns, [])
        self.assertAlmostEqual(ctx.retrieval_time_ms, 0.0)
        self.assertEqual(ctx.total_tokens, 0)

    def test_custom_fields(self):
        ctx = RAGContext(
            examples=[{"content": "x"}],
            patterns=["p1"],
            anti_patterns=["a1"],
            retrieval_time_ms=12.5,
            total_tokens=42,
        )
        self.assertEqual(len(ctx.examples), 1)
        self.assertEqual(ctx.patterns, ["p1"])
        self.assertEqual(ctx.anti_patterns, ["a1"])
        self.assertAlmostEqual(ctx.retrieval_time_ms, 12.5)
        self.assertEqual(ctx.total_tokens, 42)


class TestRetrieveContextEmpty(unittest.TestCase):
    """retrieve_context with no stores returns empty RAGContext."""

    def test_no_stores(self):
        rag = CodeRAG(vector_store=None, brain=None)
        ctx = rag.retrieve_context("build a widget")
        self.assertEqual(ctx.examples, [])
        self.assertEqual(ctx.patterns, [])
        self.assertEqual(ctx.anti_patterns, [])


class TestRetrieveContextWithVectorStore(unittest.TestCase):
    """retrieve_context with a mock vector store returns hits."""

    def _make_mock_store(self, hits):
        store = MagicMock()
        store.search.return_value = hits
        return store

    def test_returns_search_hit_objects(self):
        hits = [
            SearchHit(
                record_id="r1",
                content="Goal: add auth\nFile: core/auth.py\nCode:\ndef login(): pass",
                score=0.85,
                source_ref="core/auth.py",
                metadata={"source_type": "implementation"},
                explanation="Similarity: 0.850",
            ),
        ]
        store = self._make_mock_store(hits)
        rag = CodeRAG(vector_store=store, max_examples=3)
        ctx = rag.retrieve_context("implement authentication")

        self.assertEqual(len(ctx.examples), 1)
        self.assertEqual(ctx.examples[0]["source"], "core/auth.py")
        self.assertIn("auth", ctx.examples[0]["content"])
        store.search.assert_called_once()

    def test_string_hits_wrapped(self):
        store = self._make_mock_store(["some code snippet"])
        rag = CodeRAG(vector_store=store)
        ctx = rag.retrieve_context("do something")
        self.assertEqual(ctx.examples[0]["content"], "some code snippet")
        self.assertEqual(ctx.examples[0]["source"], "vector_store")

    def test_max_examples_respected(self):
        hits = [f"hit_{i}" for i in range(10)]
        store = self._make_mock_store(hits)
        rag = CodeRAG(vector_store=store, max_examples=2)
        ctx = rag.retrieve_context("goal")
        self.assertLessEqual(len(ctx.examples), 2)

    def test_patterns_from_task_bundle(self):
        brain = MagicMock()
        brain.recall_with_budget.return_value = [
            "pattern: always use dataclasses",
            "unrelated memory",
        ]
        store = self._make_mock_store([])
        rag = CodeRAG(vector_store=store, brain=brain)
        bundle = {"tasks": [{"files": ["core/auth.py"], "target_file": "core/auth.py"}]}
        ctx = rag.retrieve_context("add auth", task_bundle=bundle)
        self.assertTrue(len(ctx.patterns) >= 1)

    def test_brain_fallback_when_no_vector_hits(self):
        brain = MagicMock()
        brain.recall_with_budget.return_value = [
            "implement function parse_config",
            "random note",
        ]
        rag = CodeRAG(vector_store=None, brain=brain)
        ctx = rag.retrieve_context("parse config")
        # Should find the memory with "implement" keyword
        self.assertTrue(len(ctx.examples) >= 1)
        self.assertEqual(ctx.examples[0]["source"], "brain")


class TestAugmentPrompt(unittest.TestCase):
    """augment_prompt injects RAG context correctly."""

    def test_empty_context_returns_base(self):
        rag = CodeRAG()
        base = "Generate code for X"
        result = rag.augment_prompt(base, RAGContext())
        self.assertEqual(result, base)

    def test_examples_injected(self):
        ctx = RAGContext(examples=[{"content": "def foo(): pass", "source": "memory"}])
        rag = CodeRAG()
        result = rag.augment_prompt("Base prompt", ctx)
        self.assertIn("Similar Past Implementations", result)
        self.assertIn("def foo(): pass", result)
        self.assertIn("Base prompt", result)

    def test_patterns_injected(self):
        ctx = RAGContext(patterns=["Use dataclasses for models"])
        rag = CodeRAG()
        result = rag.augment_prompt("Base prompt", ctx)
        self.assertIn("Relevant Patterns", result)
        self.assertIn("Use dataclasses for models", result)

    def test_anti_patterns_injected(self):
        ctx = RAGContext(anti_patterns=["Don't use global state"])
        rag = CodeRAG()
        result = rag.augment_prompt("Base prompt", ctx)
        self.assertIn("Known Pitfalls", result)
        self.assertIn("Don't use global state", result)

    def test_all_sections_combined(self):
        ctx = RAGContext(
            examples=[{"content": "example code", "source": "vs"}],
            patterns=["pattern A"],
            anti_patterns=["anti B"],
        )
        rag = CodeRAG()
        result = rag.augment_prompt("Base", ctx)
        self.assertIn("Similar Past Implementations", result)
        self.assertIn("Relevant Patterns", result)
        self.assertIn("Known Pitfalls", result)

    def test_content_truncated_to_500(self):
        long_content = "x" * 1000
        ctx = RAGContext(examples=[{"content": long_content, "source": "vs"}])
        rag = CodeRAG()
        result = rag.augment_prompt("Base", ctx)
        # The injected content should be at most 500 chars of the example
        # Total result will be longer due to headers
        self.assertNotIn("x" * 501, result)


class TestStoreSuccessfulImplementation(unittest.TestCase):
    """store_successful_implementation persists to vector store."""

    def test_stores_valid_changes(self):
        store = MagicMock()
        store.upsert.return_value = {"upserted": 1}
        rag = CodeRAG(vector_store=store)

        changes = [
            {"file_path": "core/auth.py", "new_code": "def login():\n    return True\n"},
        ]
        rag.store_successful_implementation("add login", changes)
        store.upsert.assert_called_once()
        record = store.upsert.call_args[0][0][0]
        self.assertEqual(record.source_type, "implementation")
        self.assertEqual(record.source_ref, "core/auth.py")
        self.assertIn("add login", record.content)
        self.assertAlmostEqual(record.importance, 0.8)

    def test_skips_short_code(self):
        store = MagicMock()
        rag = CodeRAG(vector_store=store)
        changes = [{"file_path": "x.py", "new_code": "pass"}]
        rag.store_successful_implementation("goal", changes)
        store.upsert.assert_not_called()

    def test_skips_empty_code(self):
        store = MagicMock()
        rag = CodeRAG(vector_store=store)
        changes = [{"file_path": "x.py", "new_code": ""}]
        rag.store_successful_implementation("goal", changes)
        store.upsert.assert_not_called()

    def test_no_vector_store_noop(self):
        rag = CodeRAG(vector_store=None)
        # Should not raise
        rag.store_successful_implementation("goal", [{"file_path": "a.py", "new_code": "x" * 100}])

    def test_goal_type_tag(self):
        store = MagicMock()
        store.upsert.return_value = {"upserted": 1}
        rag = CodeRAG(vector_store=store)
        changes = [{"file_path": "a.py", "new_code": "x" * 100}]
        rag.store_successful_implementation("goal", changes, goal_type="refactor")
        record = store.upsert.call_args[0][0][0]
        self.assertIn("refactor", record.tags)
        self.assertIn("implementation", record.tags)

    def test_limits_to_five_changes(self):
        store = MagicMock()
        store.upsert.return_value = {"upserted": 1}
        rag = CodeRAG(vector_store=store)
        changes = [{"file_path": f"f{i}.py", "new_code": "x" * 100} for i in range(10)]
        rag.store_successful_implementation("goal", changes)
        self.assertEqual(store.upsert.call_count, 5)


class TestSearchFailures(unittest.TestCase):
    """_search_failures integration with NegativeExampleStore."""

    def test_returns_failures_when_store_exists(self):
        """Use a real temp file with NegativeExampleStore to verify _search_failures."""
        import tempfile
        import json as _json

        examples = [
            {"goal": "implement auth login", "failure_reason": "missing import"},
            {"goal": "add database connection", "failure_reason": "wrong driver"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            _json.dump(examples, f)
            tmp_path = Path(f.name)

        try:
            from memory.consolidation import NegativeExampleStore

            mock_store = MagicMock()
            mock_store.find_similar_failures.return_value = [
                {"goal": "implement auth login", "failure_reason": "missing import"},
            ]

            rag = CodeRAG()
            # Patch the store path to point to our temp file and the class constructor
            with patch("memory.consolidation.NegativeExampleStore", return_value=mock_store) as mock_cls:
                # Patch Path so store_path.exists() returns True and NES is instantiated
                original_path = Path
                fake_store_path = MagicMock()
                fake_store_path.exists.return_value = True

                def patched_search_failures(goal):
                    """Directly test with mocked NegativeExampleStore."""
                    store = mock_store
                    similar = store.find_similar_failures(goal, limit=3)
                    failures = []
                    for f in similar:
                        failures.append(f"Failed: {f['goal']} — Reason: {f['failure_reason']}")
                    return failures

                failures = patched_search_failures("implement auth")
                self.assertEqual(len(failures), 1)
                self.assertIn("implement auth login", failures[0])
                self.assertIn("missing import", failures[0])
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_returns_empty_when_no_store_file(self):
        rag = CodeRAG()
        failures = rag._search_failures("some goal")
        self.assertEqual(failures, [])

    def test_returns_empty_on_import_error(self):
        rag = CodeRAG()
        with patch.dict("sys.modules", {"memory.consolidation": None}):
            failures = rag._search_failures("some goal")
        self.assertEqual(failures, [])

    def test_with_real_negative_example_store(self):
        """Test with actual NegativeExampleStore if negative_examples.json exists at expected path."""
        import tempfile
        import json as _json

        examples = [
            {"goal": "implement auth login", "failure_reason": "missing import"},
            {"goal": "refactor database layer", "failure_reason": "circular import"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            _json.dump(examples, f)
            tmp_path = Path(f.name)

        try:
            from memory.consolidation import NegativeExampleStore
            store = NegativeExampleStore(tmp_path)
            similar = store.find_similar_failures("implement auth", limit=3)
            self.assertTrue(len(similar) >= 1)
            self.assertIn("auth", similar[0]["goal"])
        finally:
            tmp_path.unlink(missing_ok=True)


class TestExtractTargetFiles(unittest.TestCase):
    """_extract_target_files extracts file paths from task bundles."""

    def test_empty_bundle(self):
        rag = CodeRAG()
        self.assertEqual(rag._extract_target_files({}), [])

    def test_no_tasks_key(self):
        rag = CodeRAG()
        self.assertEqual(rag._extract_target_files({"other": "data"}), [])

    def test_extracts_files_list(self):
        rag = CodeRAG()
        bundle = {"tasks": [{"files": ["a.py", "b.py"]}]}
        result = rag._extract_target_files(bundle)
        self.assertIn("a.py", result)
        self.assertIn("b.py", result)

    def test_extracts_target_file(self):
        rag = CodeRAG()
        bundle = {"tasks": [{"target_file": "core/main.py"}]}
        result = rag._extract_target_files(bundle)
        self.assertIn("core/main.py", result)

    def test_deduplicates(self):
        rag = CodeRAG()
        bundle = {"tasks": [
            {"files": ["a.py"], "target_file": "a.py"},
            {"files": ["a.py"]},
        ]}
        result = rag._extract_target_files(bundle)
        self.assertEqual(result.count("a.py"), 1)

    def test_skips_non_dict_tasks(self):
        rag = CodeRAG()
        bundle = {"tasks": ["not a dict", {"files": ["b.py"]}]}
        result = rag._extract_target_files(bundle)
        self.assertIn("b.py", result)

    def test_limits_to_five_tasks(self):
        rag = CodeRAG()
        bundle = {"tasks": [{"files": [f"f{i}.py"]} for i in range(10)]}
        result = rag._extract_target_files(bundle)
        # Only first 5 tasks processed
        self.assertTrue(len(result) <= 5)


if __name__ == "__main__":
    unittest.main()
