"""Low-level filesystem utilities for AURA's code-change apply pipeline.

This module provides atomic file-write helpers and JSON parsing utilities
used by the orchestrator when applying LLM-generated change-sets to the
project tree.

Key functions:

* :func:`replace_code`       — targeted or full-file atomic replacement.
* :func:`_safe_apply_change` — resilient wrapper with auto-create and
  mismatch-overwrite fallback logic.
* :func:`_aura_clean_json`   — strips markdown fences from LLM responses.
* :func:`_aura_safe_loads`   — JSON parser with fence cleaning and UTF-8
  encoding fix-up.

Custom exceptions:

* :exc:`FileToolsError`      — base class for all module errors.
* :exc:`OldCodeNotFoundError`— raised when the target snippet is absent.
"""
import os
import tempfile
import difflib
import subprocess
from pathlib import Path
from typing import Optional
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

def find_historical_match(old_code: str, file_path: str, project_root: Path) -> Optional[str]:
    """Search the last 10 git commits for a version of *file_path* containing *old_code*.

    Uses :class:`difflib.SequenceMatcher` for fuzzy matching when an exact
    match is not found.

    Args:
        old_code: The code snippet we are trying to locate.
        file_path: Path to the file relative to *project_root*.
        project_root: Root of the git repository.

    Returns:
        The historical file content (as a string) from the commit where
        *old_code* appears (or has the highest similarity), provided the
        best similarity ratio exceeds 0.8.  Returns ``None`` otherwise.
    """
    try:
        log_result = subprocess.run(
            ["git", "log", "--oneline", "-10", "--", file_path],
            cwd=str(project_root), capture_output=True, text=True, timeout=15,
        )
        if log_result.returncode != 0 or not log_result.stdout.strip():
            return None

        commits = [line.split()[0] for line in log_result.stdout.strip().splitlines() if line.strip()]
        best_content: Optional[str] = None
        best_ratio = 0.0

        for sha in commits:
            show = subprocess.run(
                ["git", "show", f"{sha}:{file_path}"],
                cwd=str(project_root), capture_output=True, text=True, timeout=15,
            )
            if show.returncode != 0:
                continue
            hist = show.stdout
            if old_code in hist:
                return hist  # exact match – return immediately
            ratio = difflib.SequenceMatcher(None, old_code, hist).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_content = hist

        if best_ratio > 0.8:
            return best_content
        return None
    except Exception:
        return None


def recover_old_code_from_git(old_code: str, file_path: str, project_root: Path) -> Optional[str]:
    """Return the most recent historical file content that contains *old_code*.

    Thin wrapper around :func:`find_historical_match` provided as a
    callable hook for callers that only need a yes/no recovery path.
    """
    return find_historical_match(old_code, file_path, project_root)


def replace_code(file_path: str, old_code: str, new_code: str, dry_run: bool = False, overwrite_file: bool = False):
    """Replace a block of code inside a file, or overwrite the entire file.

    Performs an atomic write by staging the new content in a sibling temp file
    and then calling :func:`os.replace` to rename it over the target.  On
    POSIX systems ``os.replace`` is guaranteed to be atomic.

    Args:
        file_path: Path to the file to modify.  The file must exist.
        old_code: The exact substring to find and replace.  Pass an empty
            string (``""``) only when *overwrite_file* is ``True``.
        new_code: The replacement content.  When *overwrite_file* is ``True``
            this becomes the entire file content.
        dry_run: When ``True``, print a diff-style preview to stdout without
            modifying the file.  Defaults to ``False``.
        overwrite_file: When ``True`` the entire file is replaced with
            *new_code* regardless of the existing content.  *old_code* must
            be an empty string when this flag is set.  Defaults to ``False``.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        ValueError: If *overwrite_file* is ``True`` but *old_code* is not
            empty.
        OldCodeNotFoundError: If *old_code* is non-empty and cannot be found
            in the file content.
        FileToolsError: On any filesystem error during the atomic write.
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
    """Strip markdown code-fence wrappers from an LLM JSON response.

    LLMs frequently wrap JSON output in `` ```json … ``` `` or `` ``` … ``` ``
    fences.  This function removes those fences so that the result can be
    passed directly to :func:`json.loads`.

    Args:
        raw: The raw string returned by the LLM.  May or may not contain
            markdown fences.

    Returns:
        The cleaned string with leading/trailing fences and whitespace
        removed.  Returns *raw* unchanged when it is falsy.
    """
    if not raw:
        return raw
    # Use the imported 're' module
    text = re.sub(r"^\s*```[a-zA-Z]*\s*\n?", "", raw.strip())
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()

def _aura_safe_loads(raw: str, ctx: str = "unknown"):
    """Parse *raw* as JSON with resilient pre-processing for common LLM quirks.

    Applies two layers of defence before handing off to :func:`json.loads`:

    1. **UTF-8 round-trip** — re-encodes the string through ``utf-8`` with
       ``'ignore'`` to silently drop any hidden control characters or
       surrogate pairs that trip up the JSON parser.
    2. **Markdown fence stripping** — if the UTF-8-clean string still fails
       to parse, :func:`_aura_clean_json` is applied to remove code fences
       before a second parse attempt.

    Args:
        raw: Raw string from an LLM response that is expected to be JSON.
        ctx: Optional caller-context label for debugging.  Not used in the
            current implementation but kept for future logging.

    Returns:
        The parsed Python object (dict, list, etc.).

    Raises:
        json.JSONDecodeError: If the string cannot be parsed as valid JSON
            even after both cleaning passes.
    """
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
    """Resilient wrapper around :func:`replace_code` with auto-create and mismatch handling.

    Handles three edge cases that would otherwise cause :func:`replace_code`
    to fail or behave unexpectedly:

    *   **Missing file** — creates the file with *new_code* as content.
    *   **Empty file** — overwrites the empty file with *new_code* directly.
    *   **Old-code mismatch** — when *old_code* is provided but cannot be found
        in the existing file and *new_code* is non-empty, logs a ``WARN`` and
        falls back to a full-file overwrite rather than raising.

    Args:
        project_root: Absolute :class:`~pathlib.Path` of the project root.
            All relative *file_path* values are resolved against this.
        file_path: Relative (or absolute) path to the file to modify.
            Parent directories are created automatically when the file is new.
        old_code: The exact substring to replace.  Pass ``""`` to request a
            full-file overwrite (also set *overwrite_file* to ``True``).
        new_code: Replacement content.
        overwrite_file: When ``True``, force a full-file overwrite regardless
            of *old_code*.  Defaults to ``False``; may be promoted to ``True``
            internally on mismatch.

    Raises:
        OldCodeNotFoundError: When *old_code* is non-empty, is not found in
            the existing file, *and* *new_code* is also empty (no safe
            fallback exists).
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
    # Log a warning so the caller can audit unexpected overwrites.
    if old_code and old_code not in current:
        # Attempt to recover the correct base from git history before overwriting.
        hist = recover_old_code_from_git(old_code, file_path, project_root)
        if hist and old_code in hist:
            recovered = hist.replace(old_code, new_code)
            path_obj.write_text(recovered, encoding="utf-8")
            log_json("INFO", "file_tools_recovered_from_git", details={"file": str(path_obj)})
            return

        if new_code:
            log_json("WARN", "file_tools_old_code_mismatch_overwrite", details={
                "file": str(path_obj),
                "old_code_preview": old_code[:120],
            })
            overwrite_file = True
            old_code = ""  # trigger full-overwrite path in replace_code
        else:
            # No new_code and no match → truly invalid
            raise OldCodeNotFoundError(f"'{old_code}' not found in '{path_obj}' and no new_code for overwrite.")

    replace_code(str(path_obj), old_code, new_code, overwrite_file=overwrite_file)


class AtomicChangeSet:
    """Apply a list of file changes atomically with in-memory rollback.

    If any change fails, all previously modified files are restored to
    their pre-apply content.  Backups are kept in memory (no temp files).

    Args:
        changes: List of change dicts, each with keys
            ``file_path``, ``old_code``, ``new_code``, and
            optionally ``overwrite_file`` (default ``False``).
        project_root: Root directory used to resolve relative paths.
    """

    def __init__(self, changes: list, project_root: Path):
        self.changes = changes
        self.project_root = project_root

    def apply(self) -> list:
        """Apply all changes atomically.

        Returns:
            List of file paths that were successfully written.

        Raises:
            Exception: Re-raises the first failure after restoring all
                previously modified files to their original content.
        """
        backups: dict = {}  # file_path_str -> original content (or None if new)
        applied: list = []

        for change in self.changes:
            file_path = change["file_path"]
            old_code = change.get("old_code", "")
            new_code = change.get("new_code", "")
            overwrite_file = change.get("overwrite_file", False)

            path_obj = self.project_root / file_path
            key = str(path_obj)

            # Capture backup before first touch
            if key not in backups:
                backups[key] = path_obj.read_text(encoding="utf-8", errors="ignore") if path_obj.exists() else None

            try:
                _safe_apply_change(self.project_root, file_path, old_code, new_code, overwrite_file)
                applied.append(file_path)
            except Exception as exc:
                log_json("ERROR", "atomic_change_set_failure", details={
                    "file": file_path, "error": str(exc),
                })
                # Restore all modified files
                for restore_key, original in backups.items():
                    try:
                        restore_path = Path(restore_key)
                        if original is None:
                            if restore_path.exists():
                                restore_path.unlink()
                        else:
                            restore_path.write_text(original, encoding="utf-8")
                    except Exception as restore_exc:  # pragma: no cover
                        log_json("ERROR", "atomic_change_set_restore_failed", details={
                            "file": restore_key, "error": str(restore_exc),
                        })
                raise
        return applied


def apply_atomic(changes: list, project_root: Path) -> list:
    """Apply *changes* as an atomic transaction using :class:`AtomicChangeSet`.

    Args:
        changes: List of change dicts (``file_path``, ``old_code``,
            ``new_code``, ``overwrite_file``).
        project_root: Project root for resolving relative paths.

    Returns:
        List of file paths successfully applied.
    """
    return AtomicChangeSet(changes, project_root).apply()
