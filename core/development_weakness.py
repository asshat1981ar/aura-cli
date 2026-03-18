"""Repo-grounded development weakness scans for BEADS and RSI work."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


_RSI_KEYWORDS = (
    "recursive self-improvement",
    "recursive improvement",
    "self-improvement",
    "evolution loop",
    "rsi",
)

_RSI_CANONICAL_RUNTIME_PATH = "core/recursive_improvement.py"
_RSI_CANONICAL_PATHS = (
    "core/recursive_improvement.py",
    "core/fitness.py",
    "core/evolution_loop.py",
    "core/rsi_integration_verification.py",
    "scripts/run_rsi_evolution.py",
    "conductor/tracks/recursive_self_improvement_20260301/plan.md",
)
_RSI_DEPRECATED_PATHS = (
    "core/evolution_plan.py",
    "scripts/autonomous_rsi_run.py",
)
_RSI_TRACK_PLAN_PATH = "conductor/tracks/recursive_self_improvement_20260301/plan.md"
_RSI_TRACK_VERIFICATION_PATH = (
    "conductor/tracks/recursive_self_improvement_20260301/verification_20260308.md"
)


def _existing_paths(project_root: Path, relative_paths: tuple[str, ...]) -> list[str]:
    existing: list[str] = []
    for relative_path in relative_paths:
        if (project_root / relative_path).exists():
            existing.append(relative_path)
    return existing


def _read_text(project_root: Path, relative_path: str) -> str:
    path = project_root / relative_path
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _mentions_needles(haystack: Any, needles: list[str]) -> bool:
    lowered = str(haystack or "").lower()
    return any(needle.lower() in lowered for needle in needles)


def _infer_rsi_validation_status(project_root: Path) -> tuple[str, list[str]]:
    plan_text = _read_text(project_root, _RSI_TRACK_PLAN_PATH).lower()
    verification_text = _read_text(project_root, _RSI_TRACK_VERIFICATION_PATH).lower()
    notes: list[str] = []

    if not plan_text and not verification_text:
        notes.append("RSI conductor evidence could not be loaded from the tracked rollout artifacts.")
        return "unknown", notes

    if "- [ ] final integration" in plan_text:
        notes.append("Final live RSI integration remains open in the conductor track plan.")
        return "pending_live_evolution_audit", notes

    if "remaining open step is a non-dry-run 50-cycle audit" in verification_text:
        notes.append("The tracked RSI verification still requires a non-dry-run 50-cycle audit.")
        return "pending_live_evolution_audit", notes

    notes.append("Tracked RSI rollout artifacts do not report any remaining open validation steps.")
    return "validated", notes


def identify_development_target(goal: str, active_context: Mapping[str, Any] | None = None) -> str:
    deprecated_labels = [Path(path).name for path in _RSI_DEPRECATED_PATHS]
    needles = list(_RSI_KEYWORDS) + list(_RSI_CANONICAL_PATHS) + deprecated_labels
    if _mentions_needles(goal, needles) or _mentions_needles(active_context, needles):
        return "recursive_self_improvement"
    return "general"


def build_recursive_self_improvement_inventory(project_root: Path) -> dict[str, Any]:
    project_root = Path(project_root)
    canonical_paths = _existing_paths(project_root, _RSI_CANONICAL_PATHS)
    deprecated_paths = _existing_paths(project_root, _RSI_DEPRECATED_PATHS)
    validation_status, validation_notes = _infer_rsi_validation_status(project_root)

    weaknesses: list[dict[str, Any]] = []
    if deprecated_paths:
        weaknesses.append(
            {
                "code": "prototype_overlap",
                "severity": "high",
                "summary": "Legacy RSI prototype entrypoints still overlap the canonical runtime path.",
                "paths": deprecated_paths,
            }
        )
    
    # Validation status only matters if we are actually doing RSI work
    # but we'll report it as a weakness regardless if incomplete.
    if validation_status != "validated":
        weaknesses.append(
            {
                "code": "live_validation_pending",
                "severity": "medium" if validation_status == "pending_live_evolution_audit" else "high",
                "summary": validation_notes[0] if validation_notes else "RSI validation status is incomplete.",
                "paths": [
                    relative_path
                    for relative_path in (_RSI_TRACK_PLAN_PATH, _RSI_TRACK_VERIFICATION_PATH)
                    if (project_root / relative_path).exists()
                ],
            }
        )

    canonical_runtime_path = (
        _RSI_CANONICAL_RUNTIME_PATH
        if (project_root / _RSI_CANONICAL_RUNTIME_PATH).exists()
        else None
    )

    return {
        "subsystem": "recursive_self_improvement",
        "canonical_runtime_path": canonical_runtime_path,
        "canonical_paths": canonical_paths,
        "deprecated_paths": deprecated_paths,
        "overlap_classification": "legacy_overlap_present" if deprecated_paths else "canonical_only",
        "validation_status": validation_status,
        "validation_notes": validation_notes,
        "track_plan_path": _RSI_TRACK_PLAN_PATH if (project_root / _RSI_TRACK_PLAN_PATH).exists() else None,
        "track_verification_path": (
            _RSI_TRACK_VERIFICATION_PATH
            if (project_root / _RSI_TRACK_VERIFICATION_PATH).exists()
            else None
        ),
        "weaknesses": weaknesses,
    }


def build_development_context(
    project_root: Path,
    *,
    goal: str,
    active_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    project_root = Path(project_root)
    inventory = build_recursive_self_improvement_inventory(project_root)
    target_subsystem = identify_development_target(goal, active_context)

    if target_subsystem == inventory["subsystem"]:
        # Only classify as overlapping if deprecated paths actually exist
        if inventory["deprecated_paths"]:
            overlap_classification = inventory["overlap_classification"]
        else:
            overlap_classification = "canonical_only"
        
        canonical_path = inventory["canonical_runtime_path"]
        validation_status = inventory["validation_status"]
    else:
        canonical_path = None
        overlap_classification = "not_targeted"
        validation_status = "not_applicable"

    return {
        "target_subsystem": target_subsystem,
        "canonical_path": canonical_path,
        "overlap_classification": overlap_classification,
        "validation_status": validation_status,
        "prototype_inventory": inventory,
        "weaknesses": list(inventory.get("weaknesses", [])),
    }
