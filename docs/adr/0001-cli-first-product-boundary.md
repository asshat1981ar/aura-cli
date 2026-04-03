# ADR 0001: CLI-First Product Boundary

## Status

Accepted

## Decision

AURA ships as a CLI-first product. The installed `aura` console script is the canonical entrypoint.

The following are not treated as shipped product surfaces:

- `main.py` as a primary entrypoint
- `run_aura.sh` as a product contract
- `vscode-extension` as a functional shipped integration
- `orchestrator_hub` as a shipped runtime

Those surfaces may exist for development, experimentation, or future work, but they do not define release readiness.

## Consequences

- docs and help text must describe `aura` as canonical
- broken transport/editor claims must be removed or marked experimental
- unfinished subsystems should be moved under `experimental/`
- architecture cleanup should optimize for one CLI runtime path first
