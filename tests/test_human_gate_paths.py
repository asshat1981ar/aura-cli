from __future__ import annotations

from core.human_gate import HumanGate


def test_dependency_manifests_are_protected_paths() -> None:
    assert HumanGate.is_protected_path("package.json") is True
    assert HumanGate.is_protected_path("package-lock.json") is True
    assert HumanGate.is_protected_path("poetry.lock") is True


def test_should_block_paths_reports_dependency_manifest_changes() -> None:
    blocked, reason = HumanGate().should_block_paths(["README.md", "package.json"])
    assert blocked is True
    assert "package.json" in reason
