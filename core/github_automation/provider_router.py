"""Provider routing rules for PR automation."""
from __future__ import annotations

from core.github_automation.pr_context import PRContext


class ProviderRouter:
    """Select providers based on changed-file patterns."""

    def select_providers(self, context: PRContext) -> list[str]:
        providers = ["aura", "copilot"]

        if context.docs_only:
            return providers

        if context.touches_python:
            providers.append("gemini")

        if context.touches_workflows or context.touches_dependencies or context.touches_protected_paths:
            providers.append("claude")

        if context.touches_core and context.touched_file_count >= 4:
            providers.append("codex")

        deduped: list[str] = []
        for provider in providers:
            if provider not in deduped:
                deduped.append(provider)
        return deduped

    def route_summary(self, context: PRContext) -> str:
        providers = self.select_providers(context)
        reasons: list[str] = []
        if context.docs_only:
            reasons.append("docs-only change")
        if context.touches_python:
            reasons.append("python code touched")
        if context.touches_workflows:
            reasons.append("workflow changes detected")
        if context.touches_dependencies:
            reasons.append("dependency manifest changed")
        if context.touches_core and context.touched_file_count >= 4:
            reasons.append("broad core change set")
        if not reasons:
            reasons.append("default repo review lane")
        return f"Providers: {', '.join(providers)}. Reasons: {', '.join(reasons)}."
