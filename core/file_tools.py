import os
import tempfile
from pathlib import Path
import re # Added for _aura_clean_json
import json # Added for _aura_safe_loads
from core.logging_utils import log_json

# Custom Exception for FileTools
class FileToolsError(Exception):
    """Base exception for FileTools operations."""
    pass

class OldCodeNotFoundError(FileToolsError):
    """Exception raised when old_code is not found in the file."""
    pass

def replace_code(file_path: str, old_code: str, new_code: str, dry_run: bool = False, overwrite_file: bool = False):
    """
    Replaces a block of code in a file or overwrites the entire file.
    If dry_run is True, it prints the changes without modifying the file.
    If overwrite_file is True and old_code is an empty string, the file is entirely overwritten.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found at '{file_path}'")

    try:
        current_content = path.read_text()
        new_content = ""

        if overwrite_file:
            if old_code != "":
                # If overwrite_file is True, old_code must be empty string to signify full overwrite
                raise ValueError("When overwrite_file is True, old_code must be an empty string.")
            new_content = new_code
        elif old_code == "":
            # If old_code is empty but overwrite_file is False, this is likely an error or no-op.
            # If overwrite_file is True, we proceed with overwriting the entire file,
            # even if it's not empty, as implied by an empty old_code.
            if not overwrite_file:
                # If old_code is empty and overwrite_file is False, do nothing.
                return
            # If overwrite_file is True, and old_code is empty, new_content is already new_code.
            # The next block will handle the atomic write.
        else: # Normal replacement logic
            if old_code not in current_content:
                raise OldCodeNotFoundError(f"'{old_code}' not found in '{file_path}'.")
            new_content = current_content.replace(old_code, new_code)

        if dry_run:
            print(f"--- DRY RUN: Changes for {file_path} ---")
            print("--- OLD CODE ---")
            print(old_code if not overwrite_file else "Entire file content")
            print("--- NEW CODE ---")
            print(new_code)
            print("--------------------------------------")
        else:
            # Atomic write: write to temp file, then rename/replace
            # Using tempfile.NamedTemporaryFile for safety and automatic cleanup (on close/delete)
            # Use os.replace for atomic replacement on POSIX systems.
            with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=path.parent, encoding='utf-8') as tmp_file:
                tmp_file.write(new_content)
            
            try:
                # Atomically replace the original file
                os.replace(tmp_file.name, path)
            except OSError as e:
                # If os.replace fails with "File exists" (Errno 17), try explicit removal then rename.
                if e.errno == 17: # errno 17 is EEXIST (File exists)
                    log_json("WARN", "replace_code_initial_replace_failed_errno_17", details={"file": str(path), "error": str(e)})
                    try:
                        log_json("INFO", "replace_code_attempting_remove", details={"file": str(path)})
                        os.remove(path) # Remove the old file
                        log_json("INFO", "replace_code_old_file_removed", details={"file": str(path)})
                        log_json("INFO", "replace_code_attempting_rename", details={"temp_file": tmp_file.name, "target_file": str(path)})
                        os.rename(tmp_file.name, path) # Then rename the new file
                        log_json("INFO", "replace_code_new_file_renamed", details={"file": str(path)})
                    except OSError as remove_rename_e: # Catch OSError specifically here for more details
                        log_json("ERROR", "replace_code_remove_rename_os_error", details={"file": str(path), "error": str(remove_rename_e), "errno": remove_rename_e.errno})
                        raise FileToolsError(f"Failed to replace file '{file_path}' during remove/rename fallback: {remove_rename_e}")
                    except Exception as remove_rename_e:
                        log_json("ERROR", "replace_code_remove_rename_unexpected_error", details={"file": str(path), "error": str(remove_rename_e)})
                        raise FileToolsError(f"Unexpected error during remove/rename fallback for '{file_path}': {remove_rename_e}")
                else:
                    log_json("ERROR", "replace_code_os_error", details={"file": str(path), "error": str(e), "errno": e.errno})
                    raise # Re-raise other OSErrors

            print(f"Successfully replaced code in '{file_path}'")

    except FileToolsError as e:
        print(f"Error in FileTools: {e}")
        raise
    except OSError as e: # Catch OSError specifically during os.replace
        print(f"File system error during atomic replacement: {e}")
        raise FileToolsError(f"File system error during atomic replacement: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during code replacement: {e}")
        raise

def _aura_clean_json(raw: str) -> str:
    """Strip markdown code blocks from LLM responses"""
    if not raw:
        return raw
    # Use the imported 're' module
    text = re.sub(r"^\s*```[a-zA-Z]*\s*\n?", "", raw.strip())
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()

def _aura_safe_loads(raw: str, ctx: str = "unknown"):
    """Parse JSON with markdown cleaning, with robustness for common encoding/character issues."""
    try:
        # Attempt to clean up potential encoding issues or hidden characters by re-encoding
        # and then decoding. This can sometimes fix subtle parsing problems.
        cleaned_for_encoding = raw.encode('utf-8', 'ignore').decode('utf-8')
        return json.loads(cleaned_for_encoding)
    except json.JSONDecodeError:
        # If direct load fails, try with markdown fence cleaning (if not already done implicitly)
        # Note: _aura_clean_json already handles markdown fences, so it might be redundant here if raw is already cleaned.
        # But this path is for when initial json.loads(cleaned_for_encoding) fails.
        cleaned = _aura_clean_json(cleaned_for_encoding)
        return json.loads(cleaned)

def _safe_apply_change(project_root: Path, file_path: str, old_code: str, new_code: str, overwrite_file: bool = False):
    """
    Safe wrapper around replace_code:
    - Creates file if missing
    - Overwrites empty files
    - Falls back to replace_code otherwise
    """
    path_obj = project_root / file_path
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    if not path_obj.exists():
        path_obj.write_text(new_code or "", encoding="utf-8")
        return

    current = path_obj.read_text(encoding="utf-8", errors="ignore")

    if not current.strip():
        path_obj.write_text(new_code or "", encoding="utf-8")
        return

    # If old_code is empty and the file is not empty, assume the intent is to overwrite the entire file.
    # This addresses the scenario where the model generates initial content for an existing file.
    if not old_code and current.strip():
        overwrite_file = True

    # If old_code is provided but not found, and new_code is present, assume full overwrite intention.
    if old_code and old_code not in current:
        if new_code:
            overwrite_file = True
            old_code = "" # Set old_code to empty to trigger full overwrite behavior in replace_code
        else:
            # If old_code not found and no new_code, this is an invalid replacement.
            raise OldCodeNotFoundError(f"'{old_code}' not found in '{path_obj}' and no new_code for overwrite.")

    replace_code(str(path_obj), old_code, new_code, overwrite_file=overwrite_file)
