"""
Unit tests for agents/brainstorming_bots.py

Covers:
- BaseBrainstormingBot abstract interface
- All 8 bot implementations (uniformly parameterized)
- Module-level registry (BRAINSTORMING_BOTS, get_bot, list_techniques)
- Idea quality checks
"""

import unittest
from typing import List

from agents.brainstorming_bots import (
    BaseBrainstormingBot,
    BIABot,
    BRAINSTORMING_BOTS,
    LotusBlossomBot,
    MindMappingBot,
    ReverseBrainstormingBot,
    SCAMPERBot,
    SixThinkingHatsBot,
    StarBrainstormingBot,
    WorstIdeaBot,
    get_bot,
    list_techniques,
)
from agents.schemas import Idea


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOTS_TO_TEST = [
    ("scamper", SCAMPERBot, "SCAMPER"),
    ("six_hats", SixThinkingHatsBot, "Six Thinking Hats"),
    ("mind_map", MindMappingBot, "Mind Mapping"),
    ("reverse", ReverseBrainstormingBot, "Reverse Brainstorming"),
    ("worst_idea", WorstIdeaBot, "Worst Idea"),
    ("lotus", LotusBlossomBot, "Lotus Blossom"),
    ("star", StarBrainstormingBot, "Star Brainstorming"),
    ("bia", BIABot, "Bisociative Association"),
]

SAMPLE_TASK = "Improve developer onboarding for a large open-source project"
SAMPLE_CONTEXT = "The project has 200+ contributors and a complex monorepo structure"


def _all_scores_valid(ideas: List[Idea]) -> bool:
    """Return True if every idea's novelty/feasibility/impact are in [0, 1]."""
    for idea in ideas:
        for score in (idea.novelty, idea.feasibility, idea.impact):
            if not (0.0 <= score <= 1.0):
                return False
    return True


# ---------------------------------------------------------------------------
# TestBaseBrainstormingBot
# ---------------------------------------------------------------------------


class TestBaseBrainstormingBot(unittest.TestCase):
    """Tests for the abstract base class."""

    def test_cannot_instantiate_abstract(self):
        """BaseBrainstormingBot cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseBrainstormingBot()  # type: ignore[abstract]

    def test_create_idea_helper_returns_idea_with_correct_technique(self):
        """A concrete subclass can call _create_idea and the technique field is set."""

        class _MinimalBot(BaseBrainstormingBot):
            @property
            def technique_name(self) -> str:
                return "TestTechnique"

            @property
            def technique_key(self) -> str:
                return "test_technique"

            def _generate_template(self, task: str, context: str = "") -> List[Idea]:
                return [self._create_idea("A test idea")]

            def generate(self, task: str, context: str = "") -> List[Idea]:
                return [self._create_idea("A test idea")]

        bot = _MinimalBot()
        ideas = bot.generate("some task")
        self.assertEqual(len(ideas), 1)
        idea = ideas[0]
        self.assertIsInstance(idea, Idea)
        self.assertEqual(idea.technique, "TestTechnique")
        self.assertEqual(idea.description, "A test idea")

    def test_create_idea_helper_default_scores(self):
        """_create_idea uses 0.5 defaults for all three scores."""

        class _MinimalBot(BaseBrainstormingBot):
            @property
            def technique_name(self) -> str:
                return "Defaults"

            @property
            def technique_key(self) -> str:
                return "defaults"

            def _generate_template(self, task: str, context: str = "") -> List[Idea]:
                return [self._create_idea("default scores")]

            def generate(self, task: str, context: str = "") -> List[Idea]:
                return [self._create_idea("default scores")]

        bot = _MinimalBot()
        idea = bot.generate("task")[0]
        self.assertEqual(idea.novelty, 0.5)
        self.assertEqual(idea.feasibility, 0.5)
        self.assertEqual(idea.impact, 0.5)

    def test_create_idea_helper_custom_scores(self):
        """_create_idea passes through explicit score values."""

        class _MinimalBot(BaseBrainstormingBot):
            @property
            def technique_name(self) -> str:
                return "Custom"

            @property
            def technique_key(self) -> str:
                return "custom"

            def _generate_template(self, task: str, context: str = "") -> List[Idea]:
                return [self._create_idea("custom", novelty=0.9, feasibility=0.3, impact=0.7)]

            def generate(self, task: str, context: str = "") -> List[Idea]:
                return [self._create_idea("custom", novelty=0.9, feasibility=0.3, impact=0.7)]

        bot = _MinimalBot()
        idea = bot.generate("task")[0]
        self.assertAlmostEqual(idea.novelty, 0.9)
        self.assertAlmostEqual(idea.feasibility, 0.3)
        self.assertAlmostEqual(idea.impact, 0.7)

    def test_create_idea_helper_metadata_defaults_to_empty_dict(self):
        """_create_idea sets metadata to {} when not provided."""

        class _MinimalBot(BaseBrainstormingBot):
            @property
            def technique_name(self) -> str:
                return "Meta"

            @property
            def technique_key(self) -> str:
                return "meta"

            def _generate_template(self, task: str, context: str = "") -> List[Idea]:
                return [self._create_idea("no meta")]

            def generate(self, task: str, context: str = "") -> List[Idea]:
                return [self._create_idea("no meta")]

        bot = _MinimalBot()
        idea = bot.generate("task")[0]
        self.assertEqual(idea.metadata, {})

    def test_capabilities_set_on_init(self):
        """Every concrete bot inherits the capabilities list."""
        bot = SCAMPERBot()
        self.assertIn("brainstorming", bot.capabilities)
        self.assertIn("idea_generation", bot.capabilities)
        self.assertIn("creativity", bot.capabilities)


# ---------------------------------------------------------------------------
# TestEachBot — parameterized via subTest
# ---------------------------------------------------------------------------


class TestEachBot(unittest.TestCase):
    """Uniform tests run against all 8 bots."""

    def test_returns_non_empty_list_of_ideas(self):
        for key, bot_cls, _ in BOTS_TO_TEST:
            with self.subTest(bot=key):
                bot = bot_cls()
                ideas = bot.generate(SAMPLE_TASK)
                self.assertIsInstance(ideas, list, msg=f"{key}: generate() must return a list")
                self.assertGreater(len(ideas), 0, msg=f"{key}: generate() must return at least one idea")

    def test_technique_name_matches_expected(self):
        for key, bot_cls, expected_name in BOTS_TO_TEST:
            with self.subTest(bot=key):
                bot = bot_cls()
                self.assertEqual(
                    bot.technique_name,
                    expected_name,
                    msg=f"{key}: unexpected technique_name",
                )

    def test_ideas_have_valid_scores(self):
        """All novelty/feasibility/impact values must be in [0, 1]."""
        for key, bot_cls, _ in BOTS_TO_TEST:
            with self.subTest(bot=key):
                bot = bot_cls()
                ideas = bot.generate(SAMPLE_TASK)
                self.assertTrue(
                    _all_scores_valid(ideas),
                    msg=f"{key}: one or more ideas have scores outside [0, 1]",
                )

    def test_ideas_have_technique_set_to_bot_technique_name(self):
        """Each idea's .technique must equal the bot's technique_name."""
        for key, bot_cls, _ in BOTS_TO_TEST:
            with self.subTest(bot=key):
                bot = bot_cls()
                ideas = bot.generate(SAMPLE_TASK)
                for idea in ideas:
                    self.assertEqual(
                        idea.technique,
                        bot.technique_name,
                        msg=f"{key}: idea.technique mismatch",
                    )

    def test_generate_with_context_still_works(self):
        """Passing an extra context string must not raise or return empty."""
        for key, bot_cls, _ in BOTS_TO_TEST:
            with self.subTest(bot=key):
                bot = bot_cls()
                ideas = bot.generate(SAMPLE_TASK, context=SAMPLE_CONTEXT)
                self.assertIsInstance(ideas, list)
                self.assertGreater(len(ideas), 0, msg=f"{key}: no ideas returned when context provided")

    def test_all_ideas_are_idea_instances(self):
        """generate() must return a list of Idea objects, not plain dicts."""
        for key, bot_cls, _ in BOTS_TO_TEST:
            with self.subTest(bot=key):
                bot = bot_cls()
                ideas = bot.generate(SAMPLE_TASK)
                for idea in ideas:
                    self.assertIsInstance(idea, Idea, msg=f"{key}: non-Idea object in results")

    def test_all_ideas_have_non_empty_description(self):
        """Every idea must have a non-empty description string."""
        for key, bot_cls, _ in BOTS_TO_TEST:
            with self.subTest(bot=key):
                bot = bot_cls()
                ideas = bot.generate(SAMPLE_TASK)
                for i, idea in enumerate(ideas):
                    self.assertIsInstance(idea.description, str, msg=f"{key}[{i}]: description is not str")
                    self.assertTrue(
                        len(idea.description.strip()) > 0,
                        msg=f"{key}[{i}]: description is empty",
                    )

    def test_generate_with_empty_string_task(self):
        """Bots should not raise when given an empty task string."""
        for key, bot_cls, _ in BOTS_TO_TEST:
            with self.subTest(bot=key):
                bot = bot_cls()
                try:
                    ideas = bot.generate("")
                    self.assertIsInstance(ideas, list)
                except Exception as exc:  # noqa: BLE001
                    self.fail(f"{key}: generate('') raised {type(exc).__name__}: {exc}")

    def test_generate_called_twice_returns_lists(self):
        """generate() is safe to call multiple times on the same bot instance."""
        for key, bot_cls, _ in BOTS_TO_TEST:
            with self.subTest(bot=key):
                bot = bot_cls()
                first = bot.generate(SAMPLE_TASK)
                second = bot.generate(SAMPLE_TASK)
                self.assertIsInstance(first, list)
                self.assertIsInstance(second, list)
                self.assertGreater(len(first), 0)
                self.assertGreater(len(second), 0)


# ---------------------------------------------------------------------------
# TestBotRegistry
# ---------------------------------------------------------------------------


class TestBotRegistry(unittest.TestCase):
    """Tests for BRAINSTORMING_BOTS dict, get_bot(), and list_techniques()."""

    def test_all_8_bots_registered(self):
        self.assertEqual(len(BRAINSTORMING_BOTS), 8)

    def test_registry_contains_expected_keys(self):
        expected_keys = {"scamper", "six_hats", "mind_map", "reverse", "worst_idea", "lotus", "star", "bia"}
        self.assertEqual(set(BRAINSTORMING_BOTS.keys()), expected_keys)

    def test_registry_values_are_classes_not_instances(self):
        for key, value in BRAINSTORMING_BOTS.items():
            with self.subTest(key=key):
                self.assertTrue(
                    isinstance(value, type),
                    msg=f"BRAINSTORMING_BOTS['{key}'] should be a class, not an instance",
                )

    def test_get_bot_valid_lowercase_key(self):
        bot = get_bot("scamper")
        self.assertIsInstance(bot, SCAMPERBot)

    def test_get_bot_returns_instantiated_bot(self):
        """get_bot() must return an instance, not a class."""
        for key, bot_cls, _ in BOTS_TO_TEST:
            with self.subTest(key=key):
                bot = get_bot(key)
                self.assertIsInstance(bot, bot_cls)

    def test_get_bot_case_insensitive(self):
        """get_bot() lowercases the key internally."""
        bot_upper = get_bot("SCAMPER")
        self.assertIsInstance(bot_upper, SCAMPERBot)

        bot_mixed = get_bot("Six_Hats")
        self.assertIsInstance(bot_mixed, SixThinkingHatsBot)

    def test_get_bot_unknown_raises_value_error(self):
        with self.assertRaises(ValueError):
            get_bot("nonexistent_technique")

    def test_get_bot_unknown_error_message_contains_technique_name(self):
        try:
            get_bot("bogus")
        except ValueError as exc:
            self.assertIn("bogus", str(exc))

    def test_list_techniques_returns_list(self):
        result = list_techniques()
        self.assertIsInstance(result, list)

    def test_list_techniques_returns_8_strings(self):
        result = list_techniques()
        self.assertEqual(len(result), 8)
        for item in result:
            self.assertIsInstance(item, str)

    def test_list_techniques_matches_registry_keys(self):
        self.assertEqual(set(list_techniques()), set(BRAINSTORMING_BOTS.keys()))


# ---------------------------------------------------------------------------
# TestIdeaQuality — structural minimums and content checks
# ---------------------------------------------------------------------------


class TestIdeaQuality(unittest.TestCase):
    """Tests that specific bots produce the expected structural minimum of ideas."""

    def test_scamper_produces_at_least_7_ideas(self):
        """SCAMPER has 7 operations; each generates ≥1 idea, so total >= 7."""
        bot = SCAMPERBot()
        ideas = bot.generate(SAMPLE_TASK)
        self.assertGreaterEqual(len(ideas), 7)

    def test_scamper_produces_at_least_14_ideas(self):
        """SCAMPER generates 2-3 ideas per operation (7 ops × 2 min = 14 min)."""
        bot = SCAMPERBot()
        ideas = bot.generate(SAMPLE_TASK)
        self.assertGreaterEqual(len(ideas), 14)

    def test_six_hats_produces_at_least_6_ideas(self):
        """Six Thinking Hats has 6 hat colors; each produces ≥1 idea."""
        bot = SixThinkingHatsBot()
        ideas = bot.generate(SAMPLE_TASK)
        self.assertGreaterEqual(len(ideas), 6)

    def test_six_hats_produces_at_least_12_ideas(self):
        """Six hats × 2 ideas minimum per hat = 12."""
        bot = SixThinkingHatsBot()
        ideas = bot.generate(SAMPLE_TASK)
        self.assertGreaterEqual(len(ideas), 12)

    def test_lotus_blossom_produces_center_plus_petals(self):
        """Lotus Blossom always has 1 center + 8 petals + sub-petals."""
        bot = LotusBlossomBot()
        ideas = bot.generate(SAMPLE_TASK)
        # 1 center + 8 petals + 8*2 sub-petals = 25 minimum
        self.assertGreaterEqual(len(ideas), 25)

    def test_star_brainstorming_produces_center_plus_rays(self):
        """Star has 1 center + 6 rays × 2 variations = 13 minimum."""
        bot = StarBrainstormingBot()
        ideas = bot.generate(SAMPLE_TASK)
        self.assertGreaterEqual(len(ideas), 13)

    def test_reverse_brainstorming_produces_at_least_4_ideas(self):
        """ReverseBrainstormingBot reverses 4 fixed anti-solutions."""
        bot = ReverseBrainstormingBot()
        ideas = bot.generate(SAMPLE_TASK)
        self.assertGreaterEqual(len(ideas), 4)

    def test_worst_idea_produces_at_least_4_ideas(self):
        """WorstIdeaBot iterates over the first 4 worst-idea templates."""
        bot = WorstIdeaBot()
        ideas = bot.generate(SAMPLE_TASK)
        self.assertGreaterEqual(len(ideas), 4)

    def test_bia_produces_exactly_5_ideas(self):
        """BIABot samples 5 domains, producing exactly 5 ideas."""
        bot = BIABot()
        ideas = bot.generate(SAMPLE_TASK)
        self.assertEqual(len(ideas), 5)

    def test_mind_mapping_produces_center_and_branches(self):
        """MindMappingBot produces 1 center + 8 main branches + sub-branches."""
        bot = MindMappingBot()
        ideas = bot.generate(SAMPLE_TASK)
        # 1 center + 8 branches + 8*(2 min sub-branches) = 25 minimum
        self.assertGreaterEqual(len(ideas), 25)

    def test_scamper_ideas_reference_task_in_description(self):
        """SCAMPER descriptions embed the original task string."""
        bot = SCAMPERBot()
        ideas = bot.generate(SAMPLE_TASK)
        for idea in ideas:
            self.assertIn(SAMPLE_TASK, idea.description, msg="SCAMPER idea description should include the task")

    def test_six_hats_ideas_contain_hat_color(self):
        """Six Thinking Hats descriptions contain the hat colour in brackets."""
        hat_colors = {"WHITE", "RED", "BLACK", "YELLOW", "GREEN", "BLUE"}
        bot = SixThinkingHatsBot()
        ideas = bot.generate(SAMPLE_TASK)
        found_colors = set()
        for idea in ideas:
            for color in hat_colors:
                if color in idea.description.upper():
                    found_colors.add(color)
        self.assertEqual(
            found_colors,
            hat_colors,
            msg=f"Expected all hat colors in descriptions; found: {found_colors}",
        )

    def test_scamper_metadata_contains_scamper_action(self):
        """SCAMPER ideas carry a 'scamper_action' key in metadata."""
        bot = SCAMPERBot()
        ideas = bot.generate(SAMPLE_TASK)
        for idea in ideas:
            self.assertIn("scamper_action", idea.metadata, msg="SCAMPER idea missing 'scamper_action' in metadata")

    def test_six_hats_metadata_contains_hat_color(self):
        """Six Thinking Hats ideas carry 'hat_color' in metadata."""
        bot = SixThinkingHatsBot()
        ideas = bot.generate(SAMPLE_TASK)
        for idea in ideas:
            self.assertIn("hat_color", idea.metadata, msg="SixThinkingHats idea missing 'hat_color' in metadata")

    def test_lotus_blossom_metadata_has_level_field(self):
        """Lotus Blossom ideas carry a 'level' key in metadata."""
        bot = LotusBlossomBot()
        ideas = bot.generate(SAMPLE_TASK)
        for idea in ideas:
            self.assertIn("level", idea.metadata, msg="LotusBlossomBot idea missing 'level' in metadata")

    def test_bia_metadata_contains_domain(self):
        """BIABot ideas carry a 'domain' key in metadata."""
        bot = BIABot()
        ideas = bot.generate(SAMPLE_TASK)
        for idea in ideas:
            self.assertIn("domain", idea.metadata, msg="BIABot idea missing 'domain' in metadata")
            self.assertIn(idea.metadata["domain"], BIABot.DOMAINS, msg=f"BIABot domain '{idea.metadata['domain']}' not in DOMAINS list")

    def test_all_ideas_have_descriptions(self):
        """All ideas from every bot have non-empty description strings."""
        for key, bot_cls, _ in BOTS_TO_TEST:
            with self.subTest(bot=key):
                bot = bot_cls()
                ideas = bot.generate(SAMPLE_TASK)
                for idea in ideas:
                    self.assertTrue(
                        len(idea.description.strip()) > 0,
                        msg=f"{key}: found idea with empty description",
                    )


if __name__ == "__main__":
    unittest.main()
