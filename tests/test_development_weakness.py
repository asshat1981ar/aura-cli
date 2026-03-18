from __future__ import annotations

from pathlib import Path

from core.development_weakness import (
    build_development_context,
    build_recursive_self_improvement_inventory,
    identify_development_target,
)


def test_recursive_self_improvement_inventory_flags_overlap_and_pending_validation(tmp_path: Path):
    (tmp_path / "core").mkdir()
    (tmp_path / "scripts").mkdir()
    track_dir = tmp_path / "conductor" / "tracks" / "recursive_self_improvement_20260301"
    track_dir.mkdir(parents=True)

    (tmp_path / "core" / "recursive_improvement.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "core" / "fitness.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "core" / "evolution_loop.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "core" / "rsi_integration_verification.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "core" / "evolution_plan.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "scripts" / "run_rsi_evolution.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "scripts" / "autonomous_rsi_run.py").write_text("pass\n", encoding="utf-8")
    (track_dir / "plan.md").write_text("- [ ] Final Integration: run the live audit\n", encoding="utf-8")
    (track_dir / "verification_20260308.md").write_text(
        "The remaining open step is a non-dry-run 50-cycle audit.\n",
        encoding="utf-8",
    )

    inventory = build_recursive_self_improvement_inventory(tmp_path)

    assert inventory["canonical_runtime_path"] == "core/recursive_improvement.py"
    assert inventory["overlap_classification"] == "legacy_overlap_present"
    assert inventory["validation_status"] == "pending_live_evolution_audit"
    assert "core/evolution_plan.py" in inventory["deprecated_paths"]
    assert "scripts/autonomous_rsi_run.py" in inventory["deprecated_paths"]
    assert any(item["code"] == "prototype_overlap" for item in inventory["weaknesses"])


def test_build_development_context_targets_recursive_self_improvement_goals(tmp_path: Path):
    (tmp_path / "core").mkdir()
    (tmp_path / "scripts").mkdir()
    track_dir = tmp_path / "conductor" / "tracks" / "recursive_self_improvement_20260301"
    track_dir.mkdir(parents=True)

    (tmp_path / "core" / "recursive_improvement.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "core" / "evolution_plan.py").write_text("pass\n", encoding="utf-8")
    (track_dir / "plan.md").write_text("- [ ] Final Integration: run the live audit\n", encoding="utf-8")

    context = build_development_context(
        tmp_path,
        goal="Retire core/evolution_plan.py and move recursive self-improvement logic to the canonical path",
        active_context={"notes": ["recursive self-improvement"]},
    )

    assert context["target_subsystem"] == "recursive_self_improvement"
    assert context["canonical_path"] == "core/recursive_improvement.py"
    assert context["overlap_classification"] == "legacy_overlap_present"


def test_identify_development_target_ignores_unrelated_goals():
    assert identify_development_target("Refresh GitHub review automation") == "general"
