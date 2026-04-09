"""agents.handlers — thin dispatch layer for AURA agent pipeline.

Each module in this package exposes a single public function::

    handle(task: dict, context: dict) -> dict

This standard interface decouples the orchestrator's routing logic from the
concrete agent implementations in ``agents/``.

Available handlers
------------------
* :mod:`agents.handlers.planner`    — wraps :class:`~agents.planner.PlannerAgent`
* :mod:`agents.handlers.coder`      — wraps :class:`~agents.coder.CoderAgent`
* :mod:`agents.handlers.critic`     — wraps :class:`~agents.critic.CriticAgent`
* :mod:`agents.handlers.debugger`   — wraps :class:`~agents.debugger.DebuggerAgent`
* :mod:`agents.handlers.reflector`  — wraps :class:`~agents.reflector.ReflectorAgent`
* :mod:`agents.handlers.applicator` — wraps :class:`~agents.applicator.ApplicatorAgent`

Importing
---------
Individual handlers can be imported directly::

    from agents.handlers import planner, coder, critic

or via the ``HANDLER_MAP`` registry for dynamic dispatch::

    from agents.handlers import HANDLER_MAP
    result = HANDLER_MAP["plan"](task, context)

Adding a new handler
--------------------
1. Create ``agents/handlers/<name>.py`` that exposes ``handle(task, context) -> dict``.
2. Add it to :data:`HANDLER_MAP` below.
3. Optionally add an entry in ``agents/registry.py`` for orchestrator wiring.
"""

from __future__ import annotations

from agents.handlers import applicator  # noqa: F401
from agents.handlers import coder       # noqa: F401
from agents.handlers import critic      # noqa: F401
from agents.handlers import debugger    # noqa: F401
from agents.handlers import planner     # noqa: F401
from agents.handlers import reflector   # noqa: F401

# ---------------------------------------------------------------------------
# Dynamic dispatch registry
# Maps logical phase/capability names → handler callables.
# ---------------------------------------------------------------------------

HANDLER_MAP: dict[str, object] = {
    "plan":      planner.handle,
    "code":      coder.handle,
    "critique":  critic.handle,
    "debug":     debugger.handle,
    "reflect":   reflector.handle,
    "apply":     applicator.handle,
    # Aliases
    "planner":   planner.handle,
    "coder":     coder.handle,
    "critic":    critic.handle,
    "debugger":  debugger.handle,
    "reflector": reflector.handle,
    "applicator": applicator.handle,
}

__all__ = [
    "planner",
    "coder",
    "critic",
    "debugger",
    "reflector",
    "applicator",
    "HANDLER_MAP",
]
