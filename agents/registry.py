"""Agent registry and pipeline-phase adapters for the AURA orchestration loop.

Each concrete agent class in this module wraps a specialised agent
(``PlannerAgent``, ``CriticAgent``, etc.) and exposes a uniform
``run(input_data: dict) -> dict`` interface so that
:class:`~core.orchestrator.LoopOrchestrator` can treat every pipeline phase
identically.

The :func:`default_agents` factory wires all adapters together and returns the
dict that :class:`~core.orchestrator.LoopOrchestrator` consumes as its
``agents`` parameter.

Adapters defined here:

* :class:`PlannerAdapter`  — wraps :class:`~agents.planner.PlannerAgent`
* :class:`CriticAdapter`   — wraps :class:`~agents.critic.CriticAgent`
* :class:`ActAdapter`      — wraps :class:`~agents.coder.CoderAgent`; also
  handles smart file-path selection for generated code.
* :class:`SandboxAdapter`  — wraps :class:`~agents.sandbox.SandboxAgent`
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.types import AgentSpec
    from agents.planner import PlannerAgent
    from agents.critic import CriticAgent
    from agents.coder import CoderAgent
    from agents.sandbox import SandboxAgent

# ---------------------------------------------------------------------------
# Capability declarations
# ---------------------------------------------------------------------------

# Fallback capability list for agents that do not declare a native `capabilities`
# class attribute. First entry is the PRIMARY capability used for sort tie-breaking.
# Agents with a native `capabilities` attribute take precedence over this dict.
FALLBACK_CAPABILITIES: dict[str, list[str]] = {
    "ingest": ["ingest", "context_gathering", "memory_hints"],
    "plan": ["planning", "decomposition", "design", "tree_of_thought", "strategy"],
    "critique": ["critique", "review", "adversarial", "quality_gate"],
    "synthesize": ["synthesis", "merge", "consolidation"],
    "act": ["code_generation", "coding", "implement", "refactor"],
    "sandbox": ["sandbox", "execution", "isolated_run"],
    "verify": ["testing", "verification", "lint", "quality", "test_runner"],
    "reflect": ["reflection", "quality_analysis", "skill_update", "learning"],
    "python_agent": ["python", "code_generation", "coding", "pep8"],
    "typescript_agent": ["typescript", "javascript", "code_generation", "coding", "npm"],
    "debugging": ["debugging", "root_cause", "failure_analysis", "fix"],
    "self_correction": ["self_correction", "error_recovery", "retry"],
    "monitoring": ["monitoring", "observability", "health_check", "mcp_health"],
    "notification": ["notification", "alerting", "slack", "discord"],
    "telemetry": ["telemetry", "metrics", "tracing"],
    "mcp_discovery": ["mcp_discovery", "tool_discovery", "capability_discovery"],
    "mcp_health": ["mcp_health", "monitoring", "mcp_monitoring"],
    "root_cause_analysis": ["root_cause", "rca", "failure_analysis", "debugging"],
    "code_search": ["code_search", "symbol_lookup", "rag", "retrieval"],
    "investigation": ["investigation", "research", "analysis"],
    "external_llm": ["routing", "proxy", "llm_proxy", "model_routing", "external_llm"],
    "documentation": ["doc_generation", "readme", "inline_docs", "commenting", "documentation"],
    "innovation_swarm": ["innovation", "brainstorming", "divergence", "convergence", "creativity", "ideation"],
    "meta_conductor": ["innovation", "orchestration", "facilitation", "design_thinking", "session_management"],
    "debugger": ["debugging", "error_analysis", "fix_strategy"],
    "tester": ["testing", "unit_tests", "evaluation"],
    "scaffolder": ["scaffolding", "project_creation", "bootstrap"],
    "code_refactor": ["refactoring", "duplicate_code_reduction", "DRY"],
    "technical_debt": ["tech_debt", "code_quality", "hotspot_analysis"],
    "prompt_forge": ["prompt_engineering", "prompt_assembly", "semantic_analysis"],
}


def _make_spec(name: str, agent: object) -> "AgentSpec":
    """Build an AgentSpec, prioritising the agent's native ``capabilities`` attribute
    over the :data:`FALLBACK_CAPABILITIES` dict, falling back to ``[name]`` if neither
    is available.  The first capability in the list is treated as the *primary*
    capability by :meth:`~core.mcp_agent_registry.TypedAgentRegistry.resolve_by_capability`.
    """
    from core.types import AgentSpec

    native = getattr(agent, "capabilities", None)
    capabilities: list[str] = list(native) if native else FALLBACK_CAPABILITIES.get(name, [name])
    return AgentSpec(
        name=name,
        description=getattr(agent, "description", f"Local {name} agent"),
        capabilities=capabilities,
        source="local",
    )


# ---------------------------------------------------------------------------
# Lazy import cache — populated on first access, keyed by short agent name.
# This avoids importing all agent modules at module-load time, keeping the
# startup cost proportional to what is actually used (closes #321).
# ---------------------------------------------------------------------------

_agent_cache: dict[str, object] = {}

# Maps short names to (module_path, class_name) for deferred imports.
_AGENT_MODULE_MAP: dict[str, tuple[str, str]] = {
    "ingest": ("agents.ingest", "IngestAgent"),
    "verifier": ("agents.verifier", "VerifierAgent"),
    "synthesizer": ("agents.synthesizer", "SynthesizerAgent"),
    "reflector": ("agents.reflector", "ReflectorAgent"),
    "planner": ("agents.planner", "PlannerAgent"),
    "critic": ("agents.critic", "CriticAgent"),
    "coder": ("agents.coder", "CoderAgent"),
    "sandbox": ("agents.sandbox", "SandboxAgent"),
    "telemetry": ("agents.telemetry_agent", "TelemetryAgent"),
    "self_correction": ("agents.self_correction_agent", "SelfCorrectionAgent"),
    "code_search": ("agents.code_search_agent", "CodeSearchAgent"),
    "investigation": ("agents.investigation_agent", "InvestigationAgent"),
    "root_cause_analysis": ("agents.root_cause_analysis", "RootCauseAnalysisAgent"),
    "mcp_discovery": ("agents.mcp_discovery_agent", "MCPDiscoveryAgent"),
    "mcp_health": ("agents.mcp_health_agent", "MCPHealthAgent"),
    "innovation_swarm": ("agents.innovation_swarm", "InnovationSwarm"),
    "meta_conductor": ("agents.meta_conductor", "MetaConductor"),
    "debugger": ("agents.debugger", "DebuggerAgent"),
    "tester": ("agents.tester", "TesterAgent"),
    "scaffolder": ("agents.scaffolder", "ScaffolderAgent"),
    "code_refactor": ("agents.code_refactor_agent", "DuplicateCodeReducer"),
    "technical_debt": ("agents.technical_debt_agent", "TechnicalDebtAgent"),
    "prompt_forge": ("agents.prompt_forge", "PromptForgeAgent"),
}


def _lazy_import(agent_name: str) -> object:
    """Return the agent *class* for *agent_name*, importing it only once.

    Results are stored in the module-level :data:`_agent_cache` dict so
    subsequent calls return the cached class without re-importing the module.

    Args:
        agent_name: Short agent key as defined in :data:`_AGENT_MODULE_MAP`.

    Returns:
        The agent class object, or ``None`` if the key is unknown.
    """
    if agent_name in _agent_cache:
        return _agent_cache[agent_name]

    if agent_name not in _AGENT_MODULE_MAP:
        return None

    module_path, class_name = _AGENT_MODULE_MAP[agent_name]
    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    _agent_cache[agent_name] = cls
    return cls


class PlannerAdapter:
    """Pipeline adapter that exposes :class:`~agents.planner.PlannerAgent` as a
    ``run()``-compatible phase handler (phase name: ``"plan"``).

    Translates the generic ``input_data`` dict consumed by the orchestrator
    into the positional arguments expected by
    :meth:`~agents.planner.PlannerAgent.plan`, then normalises the return
    value into the ``{"steps": [...], "risks": [...]}`` shape required by the
    plan-phase schema.
    """

    name = "plan"

    def __init__(self, agent: PlannerAgent):
        """Initialise the adapter with a configured planner agent.

        Args:
            agent: The :class:`~agents.planner.PlannerAgent` instance to wrap.
        """
        self.agent = agent

    def run(self, input_data):
        """Execute the planning phase.

        Args:
            input_data: Dict with the following keys:

                * ``"goal"``                 — the top-level coding task.
                * ``"memory_snapshot"``      — serialised recent memory context.
                * ``"similar_past_problems"``— summaries of analogous past goals.
                * ``"known_weaknesses"``     — optional agent-level weakness hints.

        Returns:
            Dict with keys:

            * ``"steps"`` (list) — ordered plan steps returned by the agent.
            * ``"risks"`` (list) — always ``[]``; reserved for future use.
        """
        goal = input_data.get("goal", "")
        memory_snapshot = input_data.get("memory_snapshot", "")
        similar = input_data.get("similar_past_problems", "")
        weaknesses = input_data.get("known_weaknesses", "")
        steps = self.agent.plan(goal, memory_snapshot, similar, weaknesses)
        return {"steps": steps, "risks": []}


class CriticAdapter:
    """Pipeline adapter for :class:`~agents.critic.CriticAgent` (phase name: ``"critique"``).

    Calls :meth:`~agents.critic.CriticAgent.critique_plan` and wraps the
    string result into the ``{"issues": [...], "fixes": [...]}`` schema shape
    expected by the orchestrator.
    """

    name = "critique"

    def __init__(self, agent: CriticAgent):
        """Initialise the adapter with a configured critic agent.

        Args:
            agent: The :class:`~agents.critic.CriticAgent` instance to wrap.
        """
        self.agent = agent

    def run(self, input_data):
        """Execute the critique phase.

        Args:
            input_data: Dict with the following keys:

                * ``"task"`` — the natural-language goal being planned.
                * ``"plan"`` — list of plan steps produced by :class:`PlannerAdapter`.

        Returns:
            Dict with keys:

            * ``"issues"`` (list[str]) — single-element list containing the
              critique text returned by the agent.
            * ``"fixes"``  (list)      — always ``[]``; reserved for future use.
        """
        task = input_data.get("task", "")
        plan = input_data.get("plan", [])
        critique = self.agent.critique_plan(task, plan)
        return {"issues": [critique], "fixes": []}


class ActAdapter:
    """Pipeline adapter for :class:`~agents.coder.CoderAgent` (phase name: ``"act"``).

    Responsibilities beyond simple delegation:

    * Parses the optional ``# AURA_TARGET: <path>`` directive that
      :class:`~agents.coder.CoderAgent` may embed at the top of generated code
      to declare its intended destination file.
    * Falls back to :meth:`_choose_file_path` — a heuristic that scores
      existing project files against the task keywords — when no directive is
      present.
    * Returns a change-set dict compatible with the orchestrator's apply phase.
    """

    name = "act"

    def __init__(self, agent: CoderAgent):
        """Initialise the adapter with a configured coder agent.

        Args:
            agent: The :class:`~agents.coder.CoderAgent` instance to wrap.
        """
        self.agent = agent

    def _keywords(self, *texts):
        """Extract a sorted list of unique lowercase tokens from *texts*.

        Splits on any sequence of non-alphanumeric/underscore characters and
        discards tokens shorter than 3 characters.

        Args:
            *texts: One or more strings to tokenise.  ``None`` values are
                treated as empty strings.

        Returns:
            Sorted list of unique lowercase keyword strings.
        """
        words = set()
        for text in texts:
            for token in re.split(r"[^a-zA-Z0-9_]+", text or ""):
                token = token.strip().lower()
                if len(token) >= 3:
                    words.add(token)
        return sorted(words)

    def _score_path(self, path: str, keywords) -> int:
        """Return the number of *keywords* that appear in *path* (case-insensitive).

        Args:
            path: Relative or absolute file-system path string to score.
            keywords: Iterable of lowercase keyword strings.

        Returns:
            Integer count of keywords matched in the lowercased *path*.
        """
        lower = path.lower()
        return sum(1 for k in keywords if k in lower)

    def _choose_generated_name(self, directory: Path, keywords) -> str:
        """Return a non-colliding ``aura_<keyword>.py`` filename in *directory*.

        Uses the first keyword as a short descriptive suffix.  Appends ``_1``,
        ``_2``, … if the base name already exists, up to 50 attempts, then
        falls back to ``<base>_new.py``.

        Args:
            directory: Directory :class:`~pathlib.Path` in which the file will
                be created.  Must exist before calling this method.
            keywords: Ordered iterable of keyword strings.  The first element
                is used as the filename suffix (truncated to 24 chars).

        Returns:
            Absolute string path for a new ``.py`` file that does not yet exist.
        """
        base = "aura_generated"
        if keywords:
            base = f"aura_{keywords[0][:24]}"
        candidate = directory / f"{base}.py"
        if not candidate.exists():
            return str(candidate)
        for i in range(1, 50):
            alt = directory / f"{base}_{i}.py"
            if not alt.exists():
                return str(alt)
        return str(directory / f"{base}_new.py")

    def _choose_file_path(self, task: str, task_bundle: dict, project_root: Path) -> str:
        """Heuristically select the best target file path for generated code.

        Selection strategy (in priority order):

        1. Use files explicitly listed in the task bundle's first task entry.
        2. Fall back to all ``.py`` files under ``core/``, ``agents/``, and
           ``memory/`` in *project_root*.
        3. If no scored candidate has a positive keyword match, generate a new
           file via :meth:`_choose_generated_name` inside ``core/``.

        Args:
            task: The natural-language goal string, used for keyword extraction.
            task_bundle: The synthesiser output dict.  Its ``tasks[0].files``
                list and ``tasks[0].intent`` string are inspected.
            project_root: Absolute :class:`~pathlib.Path` of the project root.

        Returns:
            A relative or absolute path string that the coder's output should
            be written to.
        """
        tasks = task_bundle.get("tasks", []) if isinstance(task_bundle, dict) else []
        intent = ""
        files = []
        if tasks:
            intent = tasks[0].get("intent", "")
            files = tasks[0].get("files", []) or []

        keywords = self._keywords(task, intent)
        candidates = []

        for entry in files:
            if not entry:
                continue
            entry_path = (project_root / entry).resolve()
            if entry.endswith("/") or entry_path.is_dir():
                for path in entry_path.rglob("*.py"):
                    if ".git" in path.parts or "__pycache__" in path.parts:
                        continue
                    candidates.append(str(path.relative_to(project_root)))
            else:
                candidates.append(entry)

        if not candidates:
            for base in ["core", "agents", "memory"]:
                entry_path = project_root / base
                if entry_path.is_dir():
                    for path in entry_path.rglob("*.py"):
                        if ".git" in path.parts or "__pycache__" in path.parts:
                            continue
                        candidates.append(str(path.relative_to(project_root)))

        if candidates:
            scored = sorted(
                ((self._score_path(p, keywords), p) for p in candidates),
                reverse=True,
            )
            best_score, best_path = scored[0]
            if best_score > 0:
                return best_path

        # Fallback to a generated file in core/
        core_dir = project_root / "core"
        core_dir.mkdir(parents=True, exist_ok=True)
        return self._choose_generated_name(core_dir, keywords)

    def run(self, input_data):
        """Execute the code-generation (act) phase.

        Calls :meth:`~agents.coder.CoderAgent.implement`, optionally strips the
        ``AURA_TARGET`` directive line from the generated code, and returns a
        change-set dict ready for the orchestrator's apply step.

        Args:
            input_data: Dict with the following keys:

                * ``"task"``         — the natural-language coding goal.
                * ``"task_bundle"``  — synthesiser output used for file-path
                  selection when no directive is present.
                * ``"project_root"`` — root path for file resolution (str).
                * ``"dry_run"``      — ignored here; passed through for context.
                * ``"fix_hints"``    — optional list of error messages from a
                  prior failed verification, injected into the task bundle.

        Returns:
            A change-set dict of the form::

                {
                    "changes": [{
                        "file_path":      str,   # destination path
                        "old_code":       "",    # always empty (full overwrite)
                        "new_code":       str,   # generated source code
                        "overwrite_file": True,
                    }]
                }
        """
        task = input_data.get("task", "")
        task_bundle = input_data.get("task_bundle", {}) or {}
        project_root = Path(input_data.get("project_root", Path.cwd()))
        code = self.agent.implement(task)
        file_path = ""
        new_code = code
        if code:
            lines = code.splitlines()
            if lines and lines[0].startswith(self.agent.AURA_TARGET_DIRECTIVE):
                file_path = lines[0].replace(self.agent.AURA_TARGET_DIRECTIVE, "").strip()
                new_code = "\n".join(lines[1:]).lstrip()
        if not file_path and isinstance(task_bundle, dict):
            file_path = self._choose_file_path(task, task_bundle, project_root)
        return {
            "changes": [
                {
                    "file_path": file_path,
                    "old_code": "",
                    "new_code": new_code,
                    "overwrite_file": True,
                }
            ]
        }


class SandboxAdapter:
    """Pipeline adapter for :class:`~agents.sandbox.SandboxAgent` (phase name: ``"sandbox"``).

    Extracts generated code snippets from the act-phase output and executes
    each in an isolated subprocess **before** the changes are written to disk.
    A single failing snippet causes the entire batch to be reported as failed,
    allowing the orchestrator to skip the apply step and retry code generation
    instead.

    Input keys consumed:

    * ``act``          — change-set dict from :class:`ActAdapter` (contains
      ``"changes"`` list).
    * ``dry_run``      — when ``True`` execution is skipped and ``"skip"`` is
      returned immediately.
    * ``project_root`` — str path (unused by this adapter; kept for interface
      consistency).

    Output keys:

    * ``status``  — ``"pass"`` | ``"fail"`` | ``"skip"``
    * ``passed``  — bool mirror of ``status == "pass"``
    * ``summary`` — human-readable one-liner from :meth:`~agents.sandbox.SandboxResult.summary`
    * ``details`` — dict with ``exit_code``, ``stdout``, ``stderr``,
      ``timed_out``, and ``snippet_count``.
    """

    name = "sandbox"

    def __init__(self, agent: SandboxAgent):
        """Initialise the adapter with a configured sandbox agent.

        Args:
            agent: The :class:`~agents.sandbox.SandboxAgent` instance to wrap.
        """
        self.agent = agent

    def run(self, input_data: dict) -> dict:
        """Execute all code snippets from the act phase in isolation.

        Args:
            input_data: Dict with keys ``"act"``, ``"dry_run"``, and optionally
                ``"project_root"``.

        Returns:
            A result dict with keys ``"status"``, ``"passed"``, ``"summary"``,
            and ``"details"``.  Returns ``status="skip"`` when *dry_run* is
            ``True``, when no snippets are present, or when all snippets are
            empty.  Returns ``status="fail"`` when any snippet exits non-zero.
        """
        if input_data.get("dry_run"):
            return {"status": "skip", "passed": True, "summary": "dry_run", "details": {}}

        act_output = input_data.get("act") or {}
        changes = act_output.get("changes") or []

        # Collect all new_code snippets to validate
        snippets = [c.get("new_code", "") for c in changes if c.get("new_code")]
        if not snippets:
            return {"status": "skip", "passed": True, "summary": "no_code_to_sandbox", "details": {}}

        results = []
        for snippet in snippets:
            if not snippet.strip():
                continue
            res = self.agent.run_code(snippet)
            results.append(res)

        if not results:
            return {"status": "skip", "passed": True, "summary": "empty_snippets", "details": {}}

        # Aggregate: all must pass
        all_pass = all(r.passed for r in results)
        first = results[0]
        return {
            "status": "pass" if all_pass else "fail",
            "passed": all_pass,
            "summary": first.summary(),
            "details": {
                "exit_code": first.exit_code,
                "stdout": first.stdout[:500],
                "stderr": first.stderr[:500],
                "timed_out": first.timed_out,
                "snippet_count": len(results),
            },
        }


def default_agents(brain, model, context_manager=None, skills=None, health_monitor=None, config=None):
    """Build and return the full agent dict used by :class:`~core.orchestrator.LoopOrchestrator`.

    Instantiates every pipeline phase adapter with the provided *brain* and
    *model*, wires them together, and returns a dict whose keys match the phase
    names the orchestrator expects.

    Args:
        brain: The LLM brain object (must expose a ``remember()`` method and
            be accepted by all agent constructors).  Typically a
            :class:`~core.brain.Brain` instance.
        model: Model identifier string passed to agents that call the LLM
            (e.g. ``"gpt-4o"``).
        context_manager: Optional ContextManager instance for semantic ingestion.
        skills: Optional dict of skill instances for specialized agents.
        health_monitor: Optional HealthMonitor instance for monitoring agent.

    Returns:
        Dict mapping phase-name strings to their adapter/agent instances:

        * ``"ingest"``        → :class:`~agents.ingest.IngestAgent`
        * ``"plan"``          → :class:`PlannerAdapter`
        * ``"critique"``      → :class:`CriticAdapter`
        * ``"synthesize"``    → :class:`~agents.synthesizer.SynthesizerAgent`
        * ``"act"``           → :class:`ActAdapter`
        * ``"sandbox"``       → :class:`SandboxAdapter`
        * ``"verify"``        → :class:`~agents.verifier.VerifierAgent`
        * ``"reflect"``       → :class:`~agents.reflector.ReflectorAgent`
        * ``"python_agent"``  → :class:`~agents.python_agent.PythonAgentAdapter`
        * ``"typescript_agent"`` → :class:`~agents.typescript_agent.TypeScriptAgentAdapter`
        * ``"external_llm"``  → :class:`~agents.external_llm_agent.ExternalLLMAgentAdapter`
        * ``"monitoring"``    → :class:`~agents.monitoring_agent.MonitoringAgentAdapter`
        * ``"notification"``  → :class:`~agents.notification_agent.NotificationAgentAdapter`

    Example::

        from core.brain import Brain
        agents = default_agents(Brain(api_key="..."), model="gpt-4o")
        orchestrator = LoopOrchestrator(agents=agents, memory_store=store)
    """
    from agents.python_agent import PythonAgentAdapter
    from agents.typescript_agent import TypeScriptAgentAdapter
    from agents.external_llm_agent import ExternalLLMAgentAdapter
    from agents.monitoring_agent import MonitoringAgentAdapter
    from agents.notification_agent import NotificationAgentAdapter

    # Use lazy imports for core agent classes (populated into _agent_cache)
    IngestAgent = _lazy_import("ingest")
    PlannerAgent = _lazy_import("planner")
    CriticAgent = _lazy_import("critic")
    CoderAgent = _lazy_import("coder")
    SandboxAgent = _lazy_import("sandbox")
    SynthesizerAgent = _lazy_import("synthesizer")
    VerifierAgent = _lazy_import("verifier")
    ReflectorAgent = _lazy_import("reflector")
    TelemetryAgent = _lazy_import("telemetry")
    SelfCorrectionAgent = _lazy_import("self_correction")
    CodeSearchAgent = _lazy_import("code_search")
    InvestigationAgent = _lazy_import("investigation")
    RootCauseAnalysisAgent = _lazy_import("root_cause_analysis")
    MCPDiscoveryAgent = _lazy_import("mcp_discovery")
    MCPHealthAgent = _lazy_import("mcp_health")
    InnovationSwarm = _lazy_import("innovation_swarm")
    MetaConductor = _lazy_import("meta_conductor")

    sandbox_agent = SandboxAgent(brain, timeout=30)
    planner = PlannerAdapter(PlannerAgent(brain, model))
    critic = CriticAdapter(CriticAgent(brain, model))
    act = ActAdapter(CoderAgent(brain, model))
    sandbox = SandboxAdapter(sandbox_agent)

    agent_dict = {
        # Core pipeline agents
        "ingest": IngestAgent(brain, context_manager=context_manager),
        "plan": planner,
        "critique": critic,
        "synthesize": SynthesizerAgent(),
        "act": act,
        "sandbox": sandbox,
        "verify": VerifierAgent(),
        "reflect": ReflectorAgent(),
        # Specialized agents
        "python_agent": PythonAgentAdapter(model_adapter=None, skills=skills or {}),
        "typescript_agent": TypeScriptAgentAdapter(model_adapter=None, skills=skills or {}),
        "external_llm": ExternalLLMAgentAdapter(model_adapter=None),
        "monitoring": MonitoringAgentAdapter(health_monitor=health_monitor),
        "notification": NotificationAgentAdapter(),
        "telemetry": TelemetryAgent(),
        "self_correction": SelfCorrectionAgent(brain=brain),
        "code_search": CodeSearchAgent(vector_store=getattr(brain, "vector_store", None) if brain else None),
        "investigation": InvestigationAgent(),
        "root_cause_analysis": RootCauseAnalysisAgent(),
        "mcp_discovery": MCPDiscoveryAgent(),
        "mcp_health": MCPHealthAgent(),
        "innovation_swarm": InnovationSwarm(brain=brain, model=model),
        "meta_conductor": MetaConductor(brain=brain, model=model),
    }

    # Specialized utility agents
    DebuggerAgent = _lazy_import("debugger")
    TesterAgent = _lazy_import("tester")
    ScaffolderAgent = _lazy_import("scaffolder")
    DuplicateCodeReducer = _lazy_import("code_refactor")
    TechnicalDebtAgent = _lazy_import("technical_debt")

    PromptForgeAgent = _lazy_import("prompt_forge")

    agent_dict.update(
        {
            "debugger": DebuggerAgent(brain=brain, model=model),
            "tester": TesterAgent(brain=brain, model=model, sandbox=sandbox_agent),
            "scaffolder": ScaffolderAgent(brain=brain, model=model),
            "code_refactor": DuplicateCodeReducer(base_path=str(getattr(context_manager, "project_root", "."))),
            "technical_debt": TechnicalDebtAgent(),
            "prompt_forge": PromptForgeAgent(project_root=str(getattr(context_manager, "project_root", "."))),
        }
    )

    from agents.autogen_agent import AutoGenGroupChatAgent

    agent_dict["autogen_group_chat"] = AutoGenGroupChatAgent(brain=brain, model=model, config=(config or {}).get("autogen", {}))

    # Register in typed registry using rich multi-capability specs
    from core.mcp_agent_registry import agent_registry

    for name, agent in agent_dict.items():
        agent_registry.register(_make_spec(name, agent), overwrite=True)

    return agent_dict
