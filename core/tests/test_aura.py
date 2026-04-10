import unittest
from core.aura_manager import AuraManager


class TestAuraManager(unittest.TestCase):
    def setUp(self):
        self.aura_manager = AuraManager()

    def test_integration_functions(self):
        def dummy_function():
            return "Hello, World!"

        results = self.aura_manager.integrate_functions([dummy_function])
        self.assertEqual(results["dummy_function"], "Hello, World!")


if __name__ == "__main__":
    unittest.main()
