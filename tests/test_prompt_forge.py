"""Tests for agents/prompt_forge.py — PromptForge semantic prompt generation."""

import unittest
from unittest.mock import MagicMock, patch

from agents.prompt_forge import (
    TEMPLATES,
    ProjectContext,
    PromptForgeAgent,
    SemanticInfo,
    assemble_prompt,
    extract_semantics,
    parse_file_tree,
    semantics_to_text,
)


class TestExtractSemantics(unittest.TestCase):
    """Tests for extract_semantics."""

    def test_empty_code_returns_none(self):
        self.assertIsNone(extract_semantics(""))
        self.assertIsNone(extract_semantics("   "))
        self.assertIsNone(extract_semantics(None))

    def test_detects_async(self):
        code = "async def fetch_data():\n    await get()\n"
        sem = extract_semantics(code)
        self.assertTrue(sem.has_async)
        self.assertEqual(sem.name, "fetch_data")

    def test_detects_recursion(self):
        code = "def factorial(n):\n    return n * factorial(n - 1)\n"
        sem = extract_semantics(code)
        self.assertTrue(sem.has_recursion)

    def test_no_recursion_for_unknown(self):
        code = "x = 1\ny = x + 1\n"
        sem = extract_semantics(code)
        self.assertFalse(sem.has_recursion)

    def test_detects_loops(self):
        code = "def process():\n    for item in items:\n        pass\n"
        sem = extract_semantics(code)
        self.assertTrue(sem.has_loop)

    def test_detects_while_loop(self):
        code = "def run():\n    while True:\n        break\n"
        sem = extract_semantics(code)
        self.assertTrue(sem.has_loop)

    def test_detects_error_handling(self):
        code = "def safe():\n    try:\n        pass\n    except Exception:\n        pass\n"
        sem = extract_semantics(code)
        self.assertTrue(sem.has_error_handling)

    def test_detects_network(self):
        code = "import requests\ndef call():\n    requests.get('url')\n"
        sem = extract_semantics(code)
        self.assertTrue(sem.has_network)

    def test_detects_state(self):
        code = "function App() {\n    const [count, setCount] = useState(0);\n}\n"
        sem = extract_semantics(code)
        self.assertTrue(sem.has_state)

    def test_extracts_imports(self):
        code = "import os\nfrom pathlib import Path\ndef f(): pass\n"
        sem = extract_semantics(code)
        self.assertIn("os", sem.imports)
        self.assertIn("pathlib", sem.imports)

    def test_extracts_exports(self):
        code = "def public_func():\n    pass\n\ndef _private():\n    pass\n"
        sem = extract_semantics(code)
        self.assertIn("public_func", sem.exports)
        self.assertNotIn("_private", sem.exports)

    def test_line_count(self):
        code = "def f():\n    pass\n    return\n"
        sem = extract_semantics(code)
        self.assertEqual(sem.lines, 3)

    def test_language_passthrough(self):
        sem = extract_semantics("x = 1", language="javascript")
        self.assertEqual(sem.language, "javascript")

    def test_class_detection(self):
        code = "class MyClass:\n    pass\n"
        sem = extract_semantics(code)
        self.assertEqual(sem.name, "MyClass")


class TestSemanticsToText(unittest.TestCase):
    def test_basic_output(self):
        sem = SemanticInfo(name="foo", lines=10, has_async=True, has_loop=True, has_error_handling=False)
        text = semantics_to_text(sem)
        self.assertIn("foo", text)
        self.assertIn("10 lines", text)
        self.assertIn("asynchronous", text)
        self.assertIn("iterative", text)
        self.assertIn("lacks error handling", text)

    def test_with_error_handling(self):
        sem = SemanticInfo(name="bar", lines=5, has_error_handling=True)
        text = semantics_to_text(sem)
        self.assertIn("includes error handling", text)

    def test_with_imports_and_exports(self):
        sem = SemanticInfo(name="baz", lines=3, imports=["os", "sys"], exports=["baz"])
        text = semantics_to_text(sem)
        self.assertIn("os", text)
        self.assertIn("Exports", text)


class TestParseFileTree(unittest.TestCase):
    """Tests for parse_file_tree."""

    def test_empty_tree(self):
        ctx = parse_file_tree("")
        self.assertEqual(ctx.languages, [])
        self.assertEqual(ctx.frameworks, [])

    def test_none_tree(self):
        ctx = parse_file_tree(None)
        self.assertEqual(ctx.languages, [])

    def test_detects_python(self):
        tree = "src/main.py\nsrc/utils.py\ntests/test_main.py\n"
        ctx = parse_file_tree(tree)
        self.assertIn("Python", ctx.languages)

    def test_detects_javascript(self):
        tree = "src/index.js\nlib/utils.js\n"
        ctx = parse_file_tree(tree)
        self.assertIn("JavaScript", ctx.languages)

    def test_detects_frameworks(self):
        tree = "package.json\nDockerfile\npytest.ini\nrequirements.txt\n"
        ctx = parse_file_tree(tree)
        self.assertIn("Node.js", ctx.frameworks)
        self.assertIn("Docker", ctx.frameworks)
        self.assertIn("pytest", ctx.frameworks)

    def test_limits_to_5_languages(self):
        tree = "a.py\nb.js\nc.ts\nd.rs\ne.go\nf.rb\ng.java\n"
        ctx = parse_file_tree(tree)
        self.assertLessEqual(len(ctx.languages), 5)

    def test_structure_preserved(self):
        tree = "src/main.py\n"
        ctx = parse_file_tree(tree)
        self.assertEqual(ctx.structure, "src/main.py")


class TestAssemblePrompt(unittest.TestCase):
    """Tests for assemble_prompt with all 6 templates."""

    def test_function_template(self):
        prompt = assemble_prompt("create a parser", template="function")
        self.assertIn("Implement a Function", prompt)
        self.assertIn("create a parser", prompt)

    def test_bugfix_template(self):
        prompt = assemble_prompt("fix null pointer", template="bugfix")
        self.assertIn("Fix a Bug", prompt)

    def test_refactor_template(self):
        prompt = assemble_prompt("simplify logic", template="refactor")
        self.assertIn("Refactor Code", prompt)

    def test_feature_template(self):
        prompt = assemble_prompt("add caching", template="feature")
        self.assertIn("New Feature", prompt)

    def test_architecture_template(self):
        prompt = assemble_prompt("design microservices", template="architecture")
        self.assertIn("Architectural Design", prompt)

    def test_test_template(self):
        prompt = assemble_prompt("test the parser", template="test")
        self.assertIn("Write Tests", prompt)

    def test_unknown_template_fallback(self):
        prompt = assemble_prompt("do something", template="weird")
        self.assertIn("Task", prompt)
        self.assertIn("do something", prompt)

    def test_chain_of_thought_enabled(self):
        prompt = assemble_prompt("task", chain_of_thought=True)
        self.assertIn("Chain-of-Thought", prompt)

    def test_chain_of_thought_disabled(self):
        prompt = assemble_prompt("task", chain_of_thought=False, iterative_refine=False)
        self.assertNotIn("Chain-of-Thought", prompt)

    def test_multi_candidate(self):
        prompt = assemble_prompt("task", multi_candidate=True)
        self.assertIn("Multi-Candidate", prompt)

    def test_iterative_refine(self):
        prompt = assemble_prompt("task", iterative_refine=True)
        self.assertIn("Iterative Refinement", prompt)

    def test_code_context_included(self):
        prompt = assemble_prompt("fix bug", code_context="def foo(): pass", template="bugfix")
        self.assertIn("def foo(): pass", prompt)

    def test_constraints_included(self):
        prompt = assemble_prompt("task", constraints="must be O(n)")
        self.assertIn("must be O(n)", prompt)

    def test_file_tree_context(self):
        prompt = assemble_prompt("task", file_tree="src/main.py\npackage.json\n")
        self.assertIn("Node.js", prompt)


class TestPromptForgeAgent(unittest.TestCase):
    """Tests for PromptForgeAgent.run."""

    def test_run_basic(self):
        agent = PromptForgeAgent(project_root="/nonexistent")
        result = agent.run({"task": "implement sorting", "template": "function"})
        self.assertEqual(result["status"], "success")
        self.assertIn("prompt", result)
        self.assertIn("token_estimate", result)
        self.assertIn("Implement a Function", result["prompt"])

    def test_run_with_code_context(self):
        agent = PromptForgeAgent(project_root="/nonexistent")
        result = agent.run(
            {
                "task": "fix the bug",
                "template": "bugfix",
                "code_context": "async def fetch():\n    await get()\n",
            }
        )
        self.assertIsNotNone(result["semantics"])
        self.assertTrue(result["semantics"]["has_async"])

    def test_run_no_code_context_semantics_none(self):
        agent = PromptForgeAgent(project_root="/nonexistent")
        result = agent.run({"task": "design system"})
        self.assertIsNone(result["semantics"])

    def test_run_auto_detects_file_tree(self):
        """When project_root exists, auto-scans files."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path := __import__("pathlib").Path)(tmpdir, "main.py").write_text("x=1")
            agent = PromptForgeAgent(project_root=tmpdir)
            result = agent.run({"task": "task"})
            self.assertEqual(result["status"], "success")

    def test_capabilities(self):
        self.assertIn("prompt_engineering", PromptForgeAgent.capabilities)

    def test_templates_set(self):
        self.assertEqual(TEMPLATES, {"function", "bugfix", "refactor", "feature", "architecture", "test"})


if __name__ == "__main__":
    unittest.main()
