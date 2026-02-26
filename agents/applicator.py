"""LLM output parser and filesystem writer for AURA's apply phase.

:class:`ApplicatorAgent` is responsible for the final "last mile" of the
code-generation loop: extracting clean Python source from an LLM response,
resolving the destination file path, creating a timestamped backup of any
existing file, and atomically writing the new content to disk.

It is intentionally decoupled from the orchestrator — any component that
receives a raw LLM response string can use it directly.

Typical usage::

    applicator = ApplicatorAgent(brain, backup_dir=".aura/backups")

    # LLM output with embedded AURA_TARGET directive:
    result = applicator.apply(llm_output)

    # Explicit destination path:
    result = applicator.apply(llm_output, target_path="agents/new_agent.py")

    if not result.success:
        print(result.error)
    else:
        print(f"Wrote {len(result.code)} chars → {result.target_path}")
"""
import re
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ApplyResult:
    """Outcome of a single :meth:`ApplicatorAgent.apply` or :meth:`~ApplicatorAgent.rollback` call.

    Attributes:
        success: ``True`` when the file was written (or restored) successfully.
        target_path: Relative or absolute path of the file that was written.
            ``None`` when the path could not be determined.
        backup_path: Path to the timestamped backup created before overwriting
            the target.  ``None`` when no pre-existing file was found.
        code: The extracted Python source that was (or would be) written.
            ``None`` when extraction failed.
        error: Human-readable error description.  ``None`` on success.
        metadata: Arbitrary extra information populated on success, e.g.
            ``{"lines": 42, "timestamp": "2024-01-01T00:00:00+00:00"}``.
    """

    success: bool
    target_path: Optional[str]
    backup_path: Optional[str]
    code: Optional[str]
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __str__(self):
        if self.success:
            return (
                f"[ApplyResult OK] wrote {len(self.code or '')} chars "
                f"→ {self.target_path}  (backup: {self.backup_path})"
            )
        return f"[ApplyResult FAIL] {self.error}"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ApplicatorAgent:
    """Parses LLM-generated code and writes it to the real filesystem.

    The agent handles three concerns:

    1. **Extraction** — strips markdown fences (`` ```python … ``` ``) from
       the raw LLM response to obtain clean Python source.
    2. **Path resolution** — uses an explicit *target_path* argument or falls
       back to a ``# AURA_TARGET: <path>`` directive embedded in the code.
    3. **Safe write** — backs up any existing file before overwriting, and
       restores the backup on ``OSError``.

    All public methods return :class:`ApplyResult`; they do **not** raise.

    Class Attributes:
        DIRECTIVE_RE: Regex that matches ``# AURA_TARGET: path/to/file.py``
            anywhere in a line of code.
        CODE_BLOCK_RE: Regex that extracts the content of the first
            `` ```python … ``` `` fence in the LLM output.
    """

    DIRECTIVE_RE = re.compile(r"#\s*AURA_TARGET:\s*(.+)")
    CODE_BLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)

    def __init__(self, brain, backup_dir: str = ".aura/backups"):
        """Initialise the applicator with a brain and backup directory.

        Args:
            brain: Brain instance used for memory persistence via
                ``brain.remember()``.
            backup_dir: Directory path where timestamped file backups are
                stored before overwriting.  Created automatically if it does
                not exist.  Defaults to ``".aura/backups"``.
        """
        self.brain = brain
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(
        self,
        llm_output: str,
        target_path: Optional[str] = None,
        allow_overwrite: bool = True,
    ) -> ApplyResult:
        """Parse *llm_output*, resolve the target path, and write the code to disk.

        Steps performed in order:

        1. Extract the first `` ```python … ``` `` block from *llm_output*.
        2. Resolve the destination: use *target_path* if given, otherwise scan
           the code for a ``# AURA_TARGET: path`` directive.
        3. Bail out (with a failed :class:`ApplyResult`) if *allow_overwrite*
           is ``False`` and the target already exists.
        4. Create a timestamped backup of any existing target file.
        5. Write the extracted code atomically.  On ``OSError``, restore the
           backup and return a failed result.

        Args:
            llm_output: Raw text from the LLM, expected to contain a
                `` ```python … ``` `` code block.
            target_path: Explicit destination file path.  When provided, any
                ``# AURA_TARGET:`` directive inside the code is ignored.
            allow_overwrite: When ``False``, writing to an existing file is
                treated as an error.  Defaults to ``True``.

        Returns:
            :class:`ApplyResult` with ``success=True`` and populated
            ``target_path``, ``backup_path``, ``code``, and ``metadata``
            fields on success.  On failure, ``success=False`` and ``error``
            contains a human-readable description.

        Raises:
            Does not raise.  All errors are captured in the returned
            :class:`ApplyResult`.
        """
        code = self._extract_code(llm_output)
        if code is None:
            return ApplyResult(
                success=False,
                target_path=target_path,
                backup_path=None,
                code=None,
                error="No ```python``` code block found in LLM output.",
            )

        resolved_path = target_path or self._detect_target(code)
        if resolved_path is None:
            return ApplyResult(
                success=False,
                target_path=None,
                backup_path=None,
                code=code,
                error=(
                    "No target path provided and no # AURA_TARGET: directive "
                    "found in the code block."
                ),
            )

        path = Path(resolved_path)

        if path.exists() and not allow_overwrite:
            return ApplyResult(
                success=False,
                target_path=resolved_path,
                backup_path=None,
                code=code,
                error=f"File {resolved_path} exists and allow_overwrite=False.",
            )

        backup_path = self._backup(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(code, encoding="utf-8")
        except OSError as exc:
            # Restore backup on write failure
            if backup_path and Path(backup_path).exists():
                shutil.copy2(backup_path, path)
            return ApplyResult(
                success=False,
                target_path=resolved_path,
                backup_path=backup_path,
                code=code,
                error=f"Filesystem write failed: {exc}",
            )

        self.brain.remember(
            f"ApplicatorAgent wrote {len(code)} chars to {resolved_path} "
            f"(backup: {backup_path})"
        )

        return ApplyResult(
            success=True,
            target_path=resolved_path,
            backup_path=backup_path,
            code=code,
            metadata={
                "lines": code.count("\n") + 1,
                "timestamp": datetime.now(tz=__import__("datetime").timezone.utc).isoformat(),
            },
        )

    def rollback(self, apply_result: ApplyResult) -> bool:
        """Restore the file that was overwritten during a previous :meth:`apply` call.

        Copies the backup file back to the original target path.  The backup
        file is **not** deleted after restoration so that it remains available
        for audit purposes.

        Args:
            apply_result: The :class:`ApplyResult` returned by the previous
                :meth:`apply` call.  Both ``backup_path`` and ``target_path``
                must be set for a rollback to succeed.

        Returns:
            ``True`` when the backup was successfully restored.  ``False``
            when *apply_result* has no ``backup_path`` or the backup file no
            longer exists on disk.
        """
        if not apply_result.backup_path:
            return False
        backup = Path(apply_result.backup_path)
        target = Path(apply_result.target_path)
        if not backup.exists():
            return False
        shutil.copy2(backup, target)
        self.brain.remember(f"ApplicatorAgent rolled back {target} from {backup}")
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_code(self, text: str) -> Optional[str]:
        """Return the first python code block found in text, stripped."""
        match = self.CODE_BLOCK_RE.search(text)
        return match.group(1).strip() if match else None

    def _detect_target(self, code: str) -> Optional[str]:
        """Look for a # AURA_TARGET: path directive in the code."""
        for line in code.splitlines():
            match = self.DIRECTIVE_RE.search(line)
            if match:
                return match.group(1).strip()
        return None

    def _backup(self, path: Path) -> Optional[str]:
        """Copy existing file to backup_dir. Returns backup path string or None."""
        if not path.exists():
            return None
        timestamp = datetime.now(tz=__import__("datetime").timezone.utc).strftime("%Y%m%dT%H%M%S")
        safe_name = str(path).replace("/", "__")
        backup_path = self.backup_dir / f"{safe_name}.{timestamp}.bak"
        shutil.copy2(path, backup_path)
        return str(backup_path)