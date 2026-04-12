"""Tests for core.session_init — session initialization protocol."""

import unittest
from unittest.mock import MagicMock, patch

from core.session_init import SessionContext, SessionInitializer


class TestSessionContext(unittest.TestCase):
    """Tests for the SessionContext dataclass."""

    def test_to_prompt_context_empty(self):
        ctx = SessionContext()
        result = ctx.to_prompt_context()
        self.assertIn("No context available", result)

    def test_to_prompt_context_with_agent_config(self):
        ctx = SessionContext(agent_config={"model_name": "gpt-4", "dry_run": False})
        result = ctx.to_prompt_context()
        self.assertIn("## Agent Configuration", result)
        self.assertIn("model_name: gpt-4", result)
        self.assertIn("dry_run: False", result)

    def test_to_prompt_context_with_project_context(self):
        ctx = SessionContext(project_context={"has_git": True, "has_pyproject": True})
        result = ctx.to_prompt_context()
        self.assertIn("## Project Context", result)
        self.assertIn("has_git: True", result)

    def test_to_prompt_context_with_recent_memories(self):
        ctx = SessionContext(recent_memories=["memory_a", "memory_b"])
        result = ctx.to_prompt_context()
        self.assertIn("## Recent Memories", result)
        self.assertIn("- memory_a", result)
        self.assertIn("- memory_b", result)

    def test_to_prompt_context_with_long_term_memories(self):
        ctx = SessionContext(long_term_memories=["lt_1", "lt_2"])
        result = ctx.to_prompt_context()
        self.assertIn("## Long-Term Memories", result)
        self.assertIn("- lt_1", result)

    def test_to_prompt_context_all_sections(self):
        ctx = SessionContext(
            agent_config={"model_name": "test"},
            project_context={"has_git": True},
            recent_memories=["recent"],
            long_term_memories=["long_term"],
        )
        result = ctx.to_prompt_context()
        self.assertIn("# Session Context", result)
        self.assertIn("## Agent Configuration", result)
        self.assertIn("## Project Context", result)
        self.assertIn("## Recent Memories", result)
        self.assertIn("## Long-Term Memories", result)

    def test_timestamp_auto_set(self):
        ctx = SessionContext()
        self.assertGreater(ctx.timestamp, 0)


class TestSessionInitializerNoDeps(unittest.TestCase):
    """SessionInitializer with no optional dependencies."""

    def test_initialize_no_deps(self):
        init = SessionInitializer()
        ctx = init.initialize(task_hint="test")
        self.assertIsInstance(ctx, SessionContext)
        # Should still load agent config from global config
        self.assertIsInstance(ctx.agent_config, dict)
        self.assertIsInstance(ctx.project_context, dict)
        self.assertEqual(ctx.recent_memories, [])
        self.assertEqual(ctx.long_term_memories, [])


class TestSessionInitializerWithMemoryController(unittest.TestCase):
    """SessionInitializer with mock memory_controller."""

    def test_loads_session_memories(self):
        mc = MagicMock()
        mc.retrieve.return_value = ["session_entry_1", "session_entry_2"]

        init = SessionInitializer(memory_controller=mc)
        ctx = init.initialize()

        self.assertIn("session_entry_1", ctx.recent_memories)
        self.assertIn("session_entry_2", ctx.recent_memories)

    def test_loads_project_memories(self):
        mc = MagicMock()
        mc.retrieve.return_value = ["project_entry_1"]

        init = SessionInitializer(memory_controller=mc)
        ctx = init.initialize()

        self.assertIn("project_entry_1", ctx.long_term_memories)

    def test_memory_controller_error_handled(self):
        mc = MagicMock()
        mc.retrieve.side_effect = RuntimeError("boom")

        init = SessionInitializer(memory_controller=mc)
        ctx = init.initialize()
        # Should not raise, just return empty
        self.assertIsInstance(ctx, SessionContext)


class TestSessionInitializerWithMemoryStore(unittest.TestCase):
    """SessionInitializer with mock memory_store."""

    def test_loads_decision_log(self):
        ms = MagicMock()
        ms.query.return_value = [{"decision": "use_pytest"}]

        init = SessionInitializer(memory_store=ms)
        ctx = init.initialize()

        self.assertIn({"decision": "use_pytest"}, ctx.recent_memories)

    def test_memory_store_error_handled(self):
        ms = MagicMock()
        ms.query.side_effect = IOError("disk error")

        init = SessionInitializer(memory_store=ms)
        ctx = init.initialize()
        self.assertIsInstance(ctx, SessionContext)


class TestSessionInitializerWithMemoryManager(unittest.TestCase):
    """SessionInitializer with mock memory_manager."""

    def test_loads_semantic_memories(self):
        mm = MagicMock()
        mm.recall.return_value = ["semantic_1"]
        mm.get_procedures.return_value = ["proc_1"]

        init = SessionInitializer(memory_manager=mm)
        ctx = init.initialize()

        self.assertIn("semantic_1", ctx.long_term_memories)
        self.assertIn("proc_1", ctx.long_term_memories)

    def test_memory_manager_without_recall(self):
        mm = MagicMock(spec=[])  # No methods at all
        init = SessionInitializer(memory_manager=mm)
        ctx = init.initialize()
        self.assertIsInstance(ctx, SessionContext)

    def test_memory_manager_error_handled(self):
        mm = MagicMock()
        mm.recall.side_effect = Exception("model error")

        init = SessionInitializer(memory_manager=mm)
        ctx = init.initialize()
        self.assertIsInstance(ctx, SessionContext)


class TestSessionInitializerProjectRoot(unittest.TestCase):
    """Test project root detection."""

    def test_detect_project_root_from_cwd(self):
        # The test runs inside aura-cli which has .git and pyproject.toml
        init = SessionInitializer()
        ctx = init.initialize()
        # Should detect some project context
        self.assertIsInstance(ctx.project_context, dict)

    def test_detect_project_root_static(self):
        root = SessionInitializer._detect_project_root()
        # Running from inside aura-cli, should find a root
        # (or None if we are in a bare environment — both are valid)
        self.assertTrue(root is None or root.exists())


if __name__ == "__main__":
    unittest.main()
