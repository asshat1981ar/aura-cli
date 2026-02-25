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
    """
    Parses LLM-generated code and writes it to the real filesystem.

    Usage
    -----
    applicator = ApplicatorAgent(brain, backup_dir=".aura/backups")

    # Direct path:
    result = applicator.apply(coder_output, target_path="agents/new_agent.py")

    # Auto-detect from directive comment inside the code block:
    # (CoderAgent should include: # AURA_TARGET: agents/new_agent.py)
    result = applicator.apply(coder_output)
    """

    DIRECTIVE_RE = re.compile(r"#\s*AURA_TARGET:\s*(.+)")
    CODE_BLOCK_RE = re.compile(r"```(?:python)?
(.*?)```", re.DOTALL)

    def __init__(self, brain, backup_dir: str = ".aura/backups"):
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
        """
        Primary entry point.  Parses llm_output, resolves the target path,
        backs up any existing file, then writes the new code.
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
                "lines": code.count("
") + 1,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    def rollback(self, apply_result: ApplyResult) -> bool:
        """
        Restore the file that was overwritten during a previous apply().
        Returns True on success.
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
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        safe_name = str(path).replace("/", "__")
        backup_path = self.backup_dir / f"{safe_name}.{timestamp}.bak"
        shutil.copy2(path, backup_path)
        return str(backup_path)