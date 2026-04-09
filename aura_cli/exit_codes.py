"""Structured exit codes for AURA CLI.

These constants define the meaning of non-zero exit codes returned by the
AURA CLI so that shell scripts and CI pipelines can distinguish failure
categories without parsing stderr output.
"""

# 0 — Pipeline completed successfully (also used for read-only commands).
EXIT_SUCCESS = 0

# 1 — General failure: config error, unknown error, or any unclassified exception.
EXIT_FAILURE = 1

# 2 — Sandbox execution failed after all retries.
EXIT_SANDBOX_ERROR = 2

# 3 — Atomic apply / filesystem write error (e.g. OldCodeNotFoundError).
EXIT_APPLY_ERROR = 3

# 4 — Pipeline was cancelled by the user (KeyboardInterrupt / SIGINT).
EXIT_CANCELLED = 4

# 5 — LLM provider unavailable or rate-limited.
EXIT_LLM_ERROR = 5

__all__ = [
    "EXIT_SUCCESS",
    "EXIT_FAILURE",
    "EXIT_SANDBOX_ERROR",
    "EXIT_APPLY_ERROR",
    "EXIT_CANCELLED",
    "EXIT_LLM_ERROR",
]
