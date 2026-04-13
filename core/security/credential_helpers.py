"""Helpers for credential resolution with environment variable overrides."""
from __future__ import annotations

import os
from typing import Optional

from core.security.credential_store import CredentialStore

# Map of credential names to environment variable overrides.
# If the env var is set, it takes priority over the credential store.
_ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "api_key": ("AURA_API_KEY", "OPENROUTER_API_KEY"),
    "openai_api_key": ("OPENAI_API_KEY",),
    "anthropic_api_key": ("ANTHROPIC_API_KEY",),
    "client_secret": ("AURA_CLIENT_SECRET",),
}

# Keys in the config that are secrets and should go to the credential store
SECRET_KEYS: frozenset[str] = frozenset({
    "api_key",
    "openai_api_key",
    "anthropic_api_key",
})


def get_credential(store: CredentialStore, name: str) -> Optional[str]:
    """Retrieve a credential, checking env vars first, then the store."""
    for env_var in _ENV_OVERRIDES.get(name, ()):
        value = os.environ.get(env_var)
        if value:
            return value
    return store.get(name)


def set_credential(store: CredentialStore, name: str, value: str) -> None:
    """Store a credential in the credential store."""
    store.set(name, value)


def is_secret_key(key: str) -> bool:
    """Return True if the config key holds a secret that should use the credential store."""
    return key in SECRET_KEYS
