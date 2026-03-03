from __future__ import annotations

import os
from pathlib import Path


from core.config_manager import config

_PLACEHOLDER_VALUES = {
    "",
    "YOUR_API_KEY_HERE",
    "YOUR_OPENROUTER_API_KEY",
    "YOUR_OPENROUTER_API_KEY_HERE",
    "placeholder",
}


def _clean_secret(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or cleaned in _PLACEHOLDER_VALUES:
        return None
    return cleaned


def resolve_config_api_key(config_api_key: object | None = None) -> str | None:
    if config_api_key is None:
        config_api_key = config.get("api_key", "")
    return _clean_secret(config_api_key)


def resolve_openrouter_api_key(
    config_api_key: object | None = None,
    *,
    cli_arg: object | None = None,
) -> str | None:
    for candidate in (
        cli_arg,
        config.get("api_key"),
        resolve_config_api_key(config_api_key),
    ):
        resolved = _clean_secret(candidate)
        if resolved:
            return resolved
    return None


def resolve_openai_api_key() -> str | None:
    return _clean_secret(config.get("openai_api_key"))


def resolve_local_model_command() -> str | None:
    return _clean_secret(config.get("local_model_command"))


def resolve_local_model_profiles() -> dict[str, dict]:
    raw = config.get("local_model_profiles", {})
    if not isinstance(raw, dict):
        return {}
    return {name: profile for name, profile in raw.items() if isinstance(name, str) and isinstance(profile, dict)}


def resolve_local_embedding_profile_name() -> str | None:
    profiles = resolve_local_model_profiles()
    semantic_memory = config.get("semantic_memory", {}) or {}
    embedding_model = semantic_memory.get("embedding_model")
    if isinstance(embedding_model, str) and embedding_model.startswith("local_profile:"):
        profile_name = embedding_model.split(":", 1)[1]
        if profile_name in profiles:
            return profile_name

    local_routing = config.get("local_model_routing", {}) or {}
    if isinstance(local_routing, dict):
        profile_name = local_routing.get("embedding")
        if isinstance(profile_name, str) and profile_name in profiles:
            return profile_name
    return None


def resolve_local_embedding_mode() -> str | None:
    semantic_memory = config.get("semantic_memory", {}) or {}
    embedding_model = semantic_memory.get("embedding_model")
    if not isinstance(embedding_model, str):
        return None
    if embedding_model in {"local-tfidf-svd-50d", "local/tfidf-svd-50d"}:
        return "local-tfidf-svd-50d"
    if embedding_model.startswith("local_profile:"):
        return embedding_model
    return None


def resolve_gemini_cli_path() -> str | None:
    raw_path = _clean_secret(config.get("gemini_cli_path"))
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_file() and os.access(path, os.X_OK):
        return str(path)
    return None


def runtime_provider_status(
    config_api_key: object | None = None,
    *,
    openrouter_api_key_arg: object | None = None,
) -> dict[str, bool]:
    openai_ready = bool(resolve_openai_api_key())
    openrouter_ready = bool(
        resolve_openrouter_api_key(config_api_key=config_api_key, cli_arg=openrouter_api_key_arg)
    )
    local_ready = bool(resolve_local_model_command() or resolve_local_model_profiles())
    local_embedding_ready = bool(resolve_local_embedding_profile_name() or resolve_local_embedding_mode() == "local-tfidf-svd-50d")
    gemini_ready = bool(resolve_gemini_cli_path())
    chat_ready = openai_ready or openrouter_ready or local_ready or gemini_ready
    return {
        "openai": openai_ready,
        "openrouter": openrouter_ready,
        "local_model": local_ready,
        "gemini_cli": gemini_ready,
        "chat_ready": chat_ready,
        "embedding_ready": openai_ready or local_embedding_ready,
    }


def runtime_provider_summary(status: dict[str, bool]) -> str:
    chat_sources: list[str] = []
    if status.get("openai"):
        chat_sources.append("OPENAI_API_KEY")
    if status.get("openrouter"):
        chat_sources.append("OPENROUTER_API_KEY/AURA_API_KEY/api_key")
    if status.get("gemini_cli"):
        chat_sources.append("GEMINI_CLI_PATH")
    if status.get("local_model"):
        chat_sources.append("AURA_LOCAL_MODEL_COMMAND/local_model_profiles")

    chat_detail = ", ".join(chat_sources) if chat_sources else "none"
    local_embedding_profile = resolve_local_embedding_profile_name()
    local_embedding_mode = resolve_local_embedding_mode()
    if status.get("openai"):
        embedding_detail = "OPENAI_API_KEY"
    elif local_embedding_profile:
        embedding_detail = f"local:{local_embedding_profile}"
    elif local_embedding_mode == "local-tfidf-svd-50d":
        embedding_detail = "local-tfidf-svd-50d"
    else:
        embedding_detail = "disabled"
    return f"chat providers: {chat_detail}; embeddings: {embedding_detail}"


__all__ = [
    "resolve_config_api_key",
    "resolve_gemini_cli_path",
    "resolve_local_model_command",
    "resolve_local_model_profiles",
    "resolve_local_embedding_profile_name",
    "resolve_local_embedding_mode",
    "resolve_openai_api_key",
    "resolve_openrouter_api_key",
    "runtime_provider_status",
    "runtime_provider_summary",
]
