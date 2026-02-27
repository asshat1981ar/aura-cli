import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR_PATH = REPO_ROOT / "scripts" / "generate_cli_reference.py"


def _load_generator_module():
    spec = importlib.util.spec_from_file_location("generate_cli_reference", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load generator module from {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCLIDocsGenerator(unittest.TestCase):
    def test_render_cli_reference_groups_by_top_level_with_toc(self):
        mod = _load_generator_module()
        text = mod.render_cli_reference()

        self.assertIn("## Contributor Notes\n", text)
        self.assertIn("python3 scripts/generate_cli_reference.py", text)
        self.assertIn("tests/test_cli_main_dispatch.py -k snapshot", text)
        self.assertIn("## JSON Output Contracts\n", text)
        self.assertIn("### `cli_warnings`\n", text)
        self.assertIn("Known record codes:\n", text)
        self.assertIn("## Table of Contents\n", text)
        self.assertIn("## `goal`\n", text)
        self.assertIn("## `mcp`\n", text)
        self.assertIn("## `studio`\n", text)
        self.assertIn("python3 main.py watch --autonomous", text)
        self.assertIn("python3 main.py studio --autonomous", text)
        self.assertIn("- [`goal`](#goal)\n", text)
        self.assertIn("- [`studio`](#studio)\n", text)
        self.assertIn("- [`aura goal add`](#aura-goal-add)\n", text)

        self.assertLess(text.index("## Contributor Notes"), text.index("## Table of Contents"))
        self.assertLess(text.index("## JSON Output Contracts"), text.index("## Table of Contents"))
        self.assertLess(text.index("## Table of Contents"), text.index("## `goal`"))
        self.assertLess(text.index("## `watch`"), text.index("## `studio`"))
        self.assertLess(text.index("## `goal`"), text.index("### `aura goal add`"))

    def test_render_cli_reference_is_deterministic(self):
        mod = _load_generator_module()
        first = mod.render_cli_reference()
        second = mod.render_cli_reference()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
