from __future__ import annotations

from core.github_automation import PRContext, ProviderRouter


def test_docs_only_routes_to_aura_and_copilot() -> None:
    context = PRContext.from_changed_files(["README.md", "docs/CLI_REFERENCE.md"])
    providers = ProviderRouter().select_providers(context)
    assert providers == ["aura", "copilot"]


def test_python_core_change_adds_gemini_and_claude() -> None:
    context = PRContext.from_changed_files(
        ["core/workflow_engine.py", "core/human_gate.py", "requirements.txt"]
    )
    providers = ProviderRouter().select_providers(context)
    assert providers == ["aura", "copilot", "gemini", "claude"]


def test_large_core_change_adds_codex() -> None:
    context = PRContext.from_changed_files(
        [
            "core/workflow_engine.py",
            "core/human_gate.py",
            "aura_cli/server.py",
            "agents/registry.py",
        ]
    )
    providers = ProviderRouter().select_providers(context)
    assert providers == ["aura", "copilot", "gemini", "claude", "codex"]
