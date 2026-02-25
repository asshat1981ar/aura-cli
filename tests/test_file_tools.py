import unittest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch # Import patch

# Ensure the project root is on the path for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.file_tools import replace_code, OldCodeNotFoundError, FileToolsError

class TestFileTools(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_file_path = self.test_dir / "test_file.txt"
        self.initial_content = """Line 1
Line 2 - original
Line 3"""
        self.test_file_path.write_text(self.initial_content)

    def tearDown(self):
        # Clean up the temporary directory and its contents
        for item in self.test_dir.iterdir():
            item.unlink()
        self.test_dir.rmdir()

    def test_normal_replacement(self):
        old_code = "Line 2 - original"
        new_code = "Line 2 - replaced"
        replace_code(str(self.test_file_path), old_code, new_code)
        self.assertEqual(self.test_file_path.read_text(), """Line 1
Line 2 - replaced
Line 3""")

    def test_dry_run_replacement(self):
        old_code = "Line 2 - original"
        new_code = "Line 2 - replaced"
        # Capture stdout to check dry-run output
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output

        replace_code(str(self.test_file_path), old_code, new_code, dry_run=True)
        
        sys.stdout = sys.__stdout__ # Restore stdout
        output = captured_output.getvalue()

        self.assertIn("DRY RUN: Changes for", output)
        self.assertIn("--- OLD CODE ---\nLine 2 - original", output)
        self.assertIn("--- NEW CODE ---\nLine 2 - replaced", output)
        self.assertEqual(self.test_file_path.read_text(), self.initial_content) # File should not change

    def test_old_code_not_found_raises_exception(self):
        old_code = "Non-existent line"
        new_code = "Some new code"
        with self.assertRaisesRegex(OldCodeNotFoundError, f"'{old_code}' not found in '{self.test_file_path}'."):
            replace_code(str(self.test_file_path), old_code, new_code)
        self.assertEqual(self.test_file_path.read_text(), self.initial_content) # File should remain unchanged

    def test_overwrite_file_flag(self):
        new_content = "Completely new file content."
        replace_code(str(self.test_file_path), old_code="", new_code=new_content, overwrite_file=True)
        self.assertEqual(self.test_file_path.read_text(), new_content)

    def test_overwrite_file_flag_with_non_empty_old_code_raises_error(self):
        with self.assertRaisesRegex(ValueError, "When overwrite_file is True, old_code must be an empty string."):
            replace_code(str(self.test_file_path), old_code="something", new_code="new", overwrite_file=True)
        self.assertEqual(self.test_file_path.read_text(), self.initial_content) # File should remain unchanged

    def test_file_not_found_raises_exception(self):
        non_existent_file = self.test_dir / "non_existent.txt"
        with self.assertRaisesRegex(FileNotFoundError, f"File not found at '{non_existent_file}'"):
            replace_code(str(non_existent_file), "old", "new")

    def test_atomic_write_integrity_on_success(self):
        old_code = "Line 2 - original"
        new_code = "Line 2 - replaced atomically"
        original_inode = os.stat(self.test_file_path).st_ino
        
        replace_code(str(self.test_file_path), old_code, new_code)
        
        self.assertEqual(self.test_file_path.read_text(), """Line 1
Line 2 - replaced atomically
Line 3""")
        # Inode might change on os.replace, but the content should be correct.
        # This primarily tests successful atomic write, not specifically inode preservation.

    def test_atomic_write_failure_preserves_original(self):
        old_code = "Line 2 - original"
        new_code = "Line 2 - replaced, but failed atomically"
        
        # Ensure the file exists with initial content
        self.assertEqual(self.test_file_path.read_text(), self.initial_content)

        # Mock os.replace to raise an exception, simulating atomic write failure
        with patch('os.replace') as mock_os_replace:
            mock_os_replace.side_effect = OSError("Simulated atomic write failure")
            
            # Call replace_code; it should catch the OSError and re-raise as FileToolsError or similar
            with self.assertRaises(FileToolsError): # replace_code wraps exceptions in FileToolsError
                replace_code(str(self.test_file_path), old_code, new_code)
        
        # Assert that the original file content remains unchanged
        self.assertEqual(self.test_file_path.read_text(), self.initial_content)

        # Verify that the temporary file used for atomic write is cleaned up.
        # This is implicitly handled by tempfile.NamedTemporaryFile's context manager,
        # but we can't directly assert its absence here without exposing internal tempfile logic.
        # The key assertion is the original file's integrity.

if __name__ == '__main__':
    unittest.main()
