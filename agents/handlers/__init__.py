"""agents.handlers — thin dispatch layer for AURA agent pipeline.

Two complementary interfaces are available:

**Low-level** — each ``<name>.py`` module exposes::

    handle(task: dict, context: dict) -> dict

**High-level** — each ``<name>_handler.py`` module exposes::

    run_<name>_phase(context: dict, **kwargs) -> dict

The high-level interface is preferred in ``aura_cli/dispatch.py`` because it
maps naturally to pipeline phase names and accepts keyword arguments directly
without callers having to build a ``task`` dict manually.

Available handlers
------------------
* :mod:`agents.handlers.planner`    / :mod:`agents.handlers.planner_handler`
* :mod:`agents.handlers.coder`      / :mod:`agents.handlers.coder_handler`
* :mod:`agents.handlers.critic`     / :mod:`agents.handlers.critic_handler`
* :mod:`agents.handlers.debugger`   / :mod:`agents.handlers.debugger_handler`
* :mod:`agents.handlers.reflector`  / :mod:`agents.handlers.reflector_handler`
* :mod:`agents.handlers.applicator` / :mod:`agents.handlers.applicator_handler`

Importing
---------
High-level phase functions (preferred in dispatch.py)::

    from agents.handlers import run_planner_phase, run_coder_phase

Low-level handle functions::

    from agents.handlers import planner, coder, critic

Dynamic dispatch via the ``HANDLER_MAP`` registry::

    from agents.handlers import HANDLER_MAP
    result = HANDLER_MAP["plan"](task, context)

Adding a new handler
--------------------
1. Create ``agents/handlers/<name>.py`` that exposes ``handle(task, context) -> dict``.
2. Create ``agents/handlers/<name>_handler.py`` that exposes
   ``run_<name>_phase(context, **kwargs) -> dict``.
3. Add entries to :data:`HANDLER_MAP` and :data:`PHASE_MAP` below.
4. Optionally add an entry in ``agents/registry.py`` for orchestrator wiring.
"""

from __future__ import annotations

from agents.handlers import applicator  # noqa: F401
from agents.handlers import coder  # noqa: F401
from agents.handlers import critic  # noqa: F401
from agents.handlers import debugger  # noqa: F401
from agents.handlers import planner  # noqa: F401
from agents.handlers import reflector  # noqa: F401

# High-level ``run_<name>_phase`` wrappers (preferred by dispatch.py)
from agents.handlers.planner_handler import run_planner_phase  # noqa: F401
from agents.handlers.coder_handler import run_coder_phase  # noqa: F401
from agents.handlers.critic_handler import run_critic_phase  # noqa: F401
from agents.handlers.debugger_handler import run_debugger_phase  # noqa: F401
from agents.handlers.reflector_handler import run_reflector_phase  # noqa: F401
from agents.handlers.applicator_handler import run_applicator_phase  # noqa: F401

# ---------------------------------------------------------------------------
# Dynamic dispatch registries
# ---------------------------------------------------------------------------

# Low-level: handle(task, context) -> dict
HANDLER_MAP: dict[str, object] = {
    "plan": planner.handle,
    "code": coder.handle,
    "critique": critic.handle,
    "debug": debugger.handle,
    "reflect": reflector.handle,
    "apply": applicator.handle,
    # Aliases
    "planner": planner.handle,
    "coder": coder.handle,
    "critic": critic.handle,
    "debugger": debugger.handle,
    "reflector": reflector.handle,
    "applicator": applicator.handle,
}

# High-level: run_<name>_phase(context, **kwargs) -> dict
PHASE_MAP: dict[str, object] = {
    "planner": run_planner_phase,
    "coder": run_coder_phase,
    "critic": run_critic_phase,
    "debugger": run_debugger_phase,
    "reflector": run_reflector_phase,
    "applicator": run_applicator_phase,
    # Canonical phase-name aliases
    "plan": run_planner_phase,
    "code": run_coder_phase,
    "critique": run_critic_phase,
    "debug": run_debugger_phase,
    "reflect": run_reflector_phase,
    "apply": run_applicator_phase,
}

__all__ = [
    # Low-level modules
    "planner",
    "coder",
    "critic",
    "debugger",
    "reflector",
    "applicator",
    # High-level phase functions
    "run_planner_phase",
    "run_coder_phase",
    "run_critic_phase",
    "run_debugger_phase",
    "run_reflector_phase",
    "run_applicator_phase",
    # Registries
    "HANDLER_MAP",
    "PHASE_MAP",
]
