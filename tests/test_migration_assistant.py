import unittest


class TestMigrationPrompts(unittest.TestCase):
    def setUp(self):
        from aura_cli.migration_assistant import MigrationPrompts

        self.prompts = MigrationPrompts

    def test_has_seven_steps(self):
        all_p = self.prompts.get_all_migration_prompts()
        self.assertEqual(len(all_p), 7)

    def test_all_prompts_have_required_fields(self):
        for p in self.prompts.get_all_migration_prompts():
            self.assertIn("step", p)
            self.assertIn("title", p)
            self.assertIn("prompt", p)
            self.assertIsInstance(p["step"], int)
            self.assertGreater(len(p["prompt"]), 100)

    def test_all_prompts_expanded_and_complete(self):
        all_p = self.prompts.get_all_migration_prompts()
        for p in all_p:
            self.assertGreater(len(p["prompt"]), 800)
            self.assertIn("COMPLETE CODE TEMPLATE", p["prompt"])

    def test_no_secrets(self):
        all_text = "\n".join(p["prompt"] for p in self.prompts.get_all_migration_prompts())
        self.assertNotIn("sk-", all_text)
        self.assertNotIn("hardcoded", all_text.lower())

    def test_steps_are_sequential(self):
        all_p = self.prompts.get_all_migration_prompts()
        steps = [p["step"] for p in all_p]
        self.assertEqual(steps, list(range(1, 8)))
