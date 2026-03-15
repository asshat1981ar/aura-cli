import warnings
from core.logging_utils import log_json
from core.file_tools import (
    OldCodeNotFoundError,
    FileToolsError,
    MismatchOverwriteBlockedError,
    apply_change_with_explicit_overwrite_policy,
)


class HybridClosedLoop:
    """
    DEPRECATED: Use core.orchestrator.LoopOrchestrator instead.
    This class exists only to support legacy imports during migration.
    """
    def __init__(self, model=None, brain=None, git=None, *args, **kwargs):
        warnings.warn(
            "HybridClosedLoop is deprecated. Use core.orchestrator.LoopOrchestrator instead.",
            DeprecationWarning,
            stacklevel=2
        )
        log_json("WARN", "deprecated_class_usage", details={"class": "HybridClosedLoop", "replacement": "LoopOrchestrator"})
        self.model = model
        self.brain = brain
        self.git = git

    def _log_and_diagnose_error(self, project_root, sanitized_file_path, old_code, new_code, overwrite_file, error_code, extra_details=None):
        extra_details = extra_details or {}
        try:
            payload = {
                "summary": "apply_error",
                "diagnosis": error_code,
                "fix_strategy": "retry_with_overwrite" if overwrite_file else "inspect_change",
                "severity": "HIGH",
                "file": sanitized_file_path,
                "details": extra_details,
            }
            if self.model and hasattr(self.model, "respond"):
                self.model.respond(payload)
        finally:
            return False

    def _apply_change_with_debug(
        self,
        project_root,
        sanitized_file_path,
        old_code,
        new_code,
        overwrite_file,
        current_goal,
        change_idx,
        result_json,
        change,
    ):
        try:
            apply_change_with_explicit_overwrite_policy(
                project_root,
                sanitized_file_path,
                old_code,
                new_code,
                overwrite_file=overwrite_file,
            )
            return True
        except MismatchOverwriteBlockedError:
            return self._log_and_diagnose_error(
                project_root,
                sanitized_file_path,
                old_code,
                new_code,
                overwrite_file,
                "old_code_mismatch_overwrite_blocked",
                extra_details={"policy": "explicit_overwrite_file_required"},
            )
        except OldCodeNotFoundError:
            return False
        except FileToolsError:
            return False

    def run(self, goal):
        raise NotImplementedError("HybridClosedLoop is deprecated and non-functional. Please use the new CLI entry point 'aura'.")
