from __future__ import annotations

import atexit
import copy
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

from agents.debugger import DebuggerAgent
from agents.planner import PlannerAgent
from agents.registry import default_agents
from agents.router import RouterAgent
from core.beads_bridge import BeadsBridge
from core.config_manager import ConfigManager, DEFAULT_CONFIG, config
from core.git_tools import GitTools
from core.goal_archive import GoalArchive
from core.goal_queue import GoalQueue
from core.logging_utils import log_json
from core.model_adapter import ModelAdapter
from core.orchestrator import LoopOrchestrator
from core.policy import Policy
from core.runtime_auth import resolve_config_api_key, runtime_provider_status, runtime_provider_summary
from core.runtime_paths import resolve_project_path
from memory.vector_store_v2 import VectorStoreV2 as VectorStore
from memory.brain import Brain
from memory.store import MemoryStore


class _WeaknessRemediatorLoop:
    """Runs WeaknessRemediator every N cycles via on_cycle_complete interface."""

    EVERY_N = 5

    def __init__(self, remediator, brain, goal_queue):
        self._r, self._brain, self._queue, self._n = remediator, brain, goal_queue, 0

    def on_cycle_complete(self, _entry):
        self._n += 1
        if self._n % self.EVERY_N == 0:
            self._r.run(self._brain, self._queue, limit=3)


class _ConvergenceEscapeLoop:
    """Checks convergence escape after every cycle via on_cycle_complete interface."""

    def __init__(self, escape_loop):
        self._escape = escape_loop

    def on_cycle_complete(self, entry):
        goal = str(entry.get("phase_outputs", {}).get("context", {}).get("goal", ""))
        if not goal:
            goal = str(entry.get("phase_outputs", {}).get("plan", {}).get("goal", ""))
        if goal:
            self._escape.check_and_escape(goal, entry)


def _build_runtime_config(overrides: dict | None = None) -> dict:
    """Build a runtime-local effective config without mutating global config state."""
    base_config = ConfigManager(config_file=config.config_file).show_config()
    merged = copy.deepcopy(base_config)
    runtime_overrides = dict(overrides or {})

    for nested_key in (
        "beads",
        "model_routing",
        "semantic_memory",
        "local_model_profiles",
        "local_model_routing",
    ):
        nested_value = runtime_overrides.pop(nested_key, None)
        if isinstance(nested_value, dict):
            merged.setdefault(nested_key, {})
            merged[nested_key].update(nested_value)

    merged.update(runtime_overrides)
    return merged


def _resolve_runtime_paths(project_root: Path, runtime_config: dict | None = None) -> dict:
    """Resolve all required storage paths against the project root."""
    runtime_config = runtime_config or config.show_config()
    paths = {}
    for key in ["goal_queue_path", "goal_archive_path", "brain_db_path", "memory_store_path", "memory_persistence_path"]:
        paths[key] = resolve_project_path(project_root, runtime_config.get(key, DEFAULT_CONFIG[key]), DEFAULT_CONFIG[key])
    return paths


def _init_memory_and_brain(brain_db_path: Path):
    """Initialize the brain instance and cache adapter."""
    try:
        from memory.cache_adapter_factory import create_cache_adapter
        from memory.momento_brain import MomentoBrain

        _momento = create_cache_adapter()
        brain_instance = MomentoBrain(_momento, db_path=str(brain_db_path))
        log_json("INFO", "runtime_cache_adapter_active", details={"adapter": type(_momento).__name__})
        return brain_instance, _momento
    except Exception as _exc:
        log_json("WARN", "cache_adapter_init_failed", details={"error": str(_exc)})
        return Brain(db_path=str(brain_db_path)), None


def _start_background_sync(project_root: Path, vector_store, context_graph):
    """Start the project knowledge syncer in a background thread."""
    from core.project_syncer import ProjectKnowledgeSyncer

    syncer = ProjectKnowledgeSyncer(vector_store=vector_store, context_graph=context_graph, project_root=project_root)
    _stop_event = threading.Event()

    def _bg_sync():
        try:
            syncer.sync_all()
        except Exception as exc:
            log_json("WARN", "background_sync_failed", details={"error": str(exc)})

    _sync_thread = threading.Thread(target=_bg_sync, daemon=True, name="aura-bg-sync")
    _sync_thread.start()

    def _on_exit():
        _stop_event.set()
        _sync_thread.join(timeout=5)

    atexit.register(_on_exit)
    log_json("INFO", "background_sync_started")


def _attach_advanced_loops(orchestrator, runtime_mode, brain, memory_store, goal_queue, momento, project_root):
    """Attach improvement loops and CASPA-W workflow to the orchestrator."""
    if runtime_mode in ("lean", "queue"):
        return

    try:
        from core.reflection_loop import DeepReflectionLoop
        from core.health_monitor import HealthMonitor
        from core.weakness_remediator import WeaknessRemediator
        from core.skill_weight_adapter import SkillWeightAdapter
        from core.convergence_escape import ConvergenceEscapeLoop
        from core.memory_compaction import MemoryCompactionLoop

        _reflection = DeepReflectionLoop(memory_store, brain)
        _health = HealthMonitor(orchestrator.skills, goal_queue, memory_store, project_root)
        _remediator = _WeaknessRemediatorLoop(WeaknessRemediator(), brain, goal_queue)
        _skill_adapt = SkillWeightAdapter(memory_root=str(memory_store.root.parent), momento=momento)
        _conv_escape = _ConvergenceEscapeLoop(ConvergenceEscapeLoop(memory_store, goal_queue))
        _compaction = MemoryCompactionLoop(memory_store)

        orchestrator.attach_improvement_loops(_reflection, _health, _remediator, _skill_adapt, _conv_escape, _compaction)

        # LearningCoordinator (PRD-003): ties all learning signals into artifacts + backlog
        try:
            from core.learning_coordinator import LearningCoordinator
            _learning = LearningCoordinator(memory_store)
            orchestrator.attach_learning_coordinator(_learning)
        except Exception as _exc:
            log_json("WARN", "learning_coordinator_setup_failed", details={"error": str(_exc)})

        if getattr(orchestrator, "beads_enabled", False) and "beads_skill" in orchestrator.skills:
            from core.orchestrator import BeadsSyncLoop

            _beads_sync = BeadsSyncLoop(orchestrator.skills["beads_skill"])
            orchestrator.attach_improvement_loops(_beads_sync)
    except Exception as _exc:
        log_json("WARN", "improvement_loops_setup_failed", details={"error": str(_exc)})

    try:
        from agents.mutator import MutatorAgent
        from core.adaptive_pipeline import AdaptivePipeline
        from core.autonomous_discovery import AutonomousDiscovery
        from core.context_graph import ContextGraph
        from core.evolution_loop import EvolutionLoop
        from core.propagation_engine import PropagationEngine
        from core.recursive_improvement import RecursiveImprovementService

        _context_graph = ContextGraph()
        _adaptive_pipeline = AdaptivePipeline(
            context_graph=_context_graph,
            skill_weight_adapter=_skill_adapt if "_skill_adapt" in locals() else None,
            memory_store=memory_store,
            brain=brain,
        )
        _propagation = PropagationEngine(goal_queue, _context_graph, memory_store)
        _discovery = AutonomousDiscovery(goal_queue, memory_store, project_root=str(project_root))

        _ri_service = RecursiveImprovementService()
        _coder_adapter = orchestrator.agents.get("act")
        _critic_adapter = orchestrator.agents.get("critique")
        _planner_adapter = orchestrator.agents.get("plan")
        _coder = getattr(_coder_adapter, "agent", _coder_adapter)
        _critic = getattr(_critic_adapter, "agent", _critic_adapter)
        _planner = getattr(_planner_adapter, "agent", _planner_adapter)
        _git = GitTools(repo_path=str(project_root))
        _mutator = MutatorAgent(project_root)
        _evo = EvolutionLoop(
            _planner,
            _coder,
            _critic,
            brain,
            getattr(brain, "vector_store", None),
            _git,
            _mutator,
            improvement_service=_ri_service,
            goal_queue=goal_queue,
            orchestrator=orchestrator,
            project_root=project_root,
            skills=orchestrator.skills,
            auto_execute_queued=False,
        )

        orchestrator.attach_caspa(
            adaptive_pipeline=_adaptive_pipeline,
            propagation_engine=_propagation,
            context_graph=_context_graph,
        )
        orchestrator.attach_improvement_loops(_discovery, _evo)
    except Exception as _exc:
        log_json("WARN", "caspa_setup_failed", details={"error": str(_exc)})


def _get_momento_store(root, momento):
    try:
        from memory.momento_memory_store import MomentoMemoryStore

        return MomentoMemoryStore(root, momento)
    except Exception as exc:
        log_json("WARN", "memory_store_init_failed", details={"error": str(exc)})
        return MemoryStore(root)


def create_runtime(project_root: Path, overrides: dict | None = None):
    """
    Library-friendly initializer that sets up shared AURA runtime objects
    without user input loops or forced chdir. Used by the HTTP server.
    """
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    runtime_overrides = dict(overrides or {})
    runtime_mode = runtime_overrides.pop("runtime_mode", "full")
    beads_cli_override = runtime_overrides.pop("beads_cli_override", None)
    runtime_config = _build_runtime_config(runtime_overrides)

    paths = _resolve_runtime_paths(project_root, runtime_config=runtime_config)
    goal_queue = GoalQueue(str(paths["goal_queue_path"]))
    goal_archive = GoalArchive(str(paths["goal_archive_path"]))

    config_api_key = resolve_config_api_key(runtime_config.get("api_key", ""))
    provider_status = runtime_provider_status(config_api_key=config_api_key)

    if not provider_status["chat_ready"]:
        log_json("WARN", "aura_api_key_missing", details={"providers": runtime_provider_summary(provider_status)})
        print("⚠️  AURA: No chat provider configured.", file=sys.stderr)

    if runtime_mode == "queue":
        return {
            "goal_queue": goal_queue,
            "goal_archive": goal_archive,
            "orchestrator": SimpleNamespace(),
            "beads_bridge": None,
            "debugger": None,
            "planner": None,
            "loop": None,
            "git_tools": None,
            "project_root": project_root,
            "model_adapter": None,
            "memory_store": None,
            "brain": None,
            "vector_store": None,
            "router": None,
            "config_api_key": config_api_key,
            "provider_status": provider_status,
            "memory_persistence_path": paths["memory_persistence_path"],
            "beads_cli_override": beads_cli_override,
        }

    brain_instance, _momento = _init_memory_and_brain(paths["brain_db_path"])
    model_adapter = ModelAdapter()
    model_adapter.enable_cache(brain_instance.db, ttl_seconds=3600, momento=_momento)

    vector_store = VectorStore(model_adapter, brain_instance)
    brain_instance.set_vector_store(vector_store)

    router = RouterAgent(brain_instance, model_adapter)
    model_adapter.set_router(router)

    debugger_instance = DebuggerAgent(brain_instance, model_adapter)
    planner_instance = PlannerAgent(brain_instance, model_adapter)

    try:
        from core.context_manager import ContextManager

        context_manager = ContextManager(vector_store=vector_store, project_root=project_root)
        if runtime_mode != "lean":
            _start_background_sync(project_root, vector_store, None)
    except Exception as _exc:
        log_json("WARN", "context_manager_init_failed", details={"error": str(_exc)})
        context_manager = None

    memory_store = _get_momento_store(paths["memory_store_path"], _momento) if _momento else MemoryStore(paths["memory_store_path"])
    from memory.controller import memory_controller

    memory_controller.set_store(memory_store)

    beads_config = runtime_config.get("beads", DEFAULT_CONFIG["beads"]) or {}
    bridge_command = beads_config.get("bridge_command")
    beads_bridge = BeadsBridge.from_defaults(
        project_root,
        command=bridge_command,
        timeout_seconds=float(beads_config.get("timeout_seconds", 20)),
        enabled=bool(beads_config.get("enabled", True)),
        required=bool(beads_config.get("required", True)),
        persist_artifacts=bool(beads_config.get("persist_artifacts", True)),
        scope=str(beads_config.get("scope", "goal_run")),
    )

    agents = default_agents(brain_instance, model_adapter, context_manager=context_manager)
    telemetry_agent = agents.get("telemetry")
    if telemetry_agent:
        model_adapter.set_telemetry_agent(telemetry_agent)

    orchestrator = LoopOrchestrator(
        agents=agents,
        memory_store=memory_store,
        policy=Policy.from_config(copy.deepcopy(runtime_config)),
        project_root=project_root,
        strict_schema=bool(runtime_config.get("strict_schema", False)),
        debugger=debugger_instance,
        goal_queue=goal_queue,
        goal_archive=goal_archive,
        brain=brain_instance,
        model=model_adapter,
        runtime_mode=runtime_mode,
        beads_bridge=beads_bridge,
        beads_enabled=bool(beads_config.get("enabled", True)),
        beads_required=bool(beads_config.get("required", True)),
        beads_scope=str(beads_config.get("scope", "goal_run")),
    )
    orchestrator.beads_runtime_override = beads_cli_override

    _attach_advanced_loops(orchestrator, runtime_mode, brain_instance, memory_store, goal_queue, _momento, project_root)

    return {
        "goal_queue": goal_queue,
        "goal_archive": goal_archive,
        "orchestrator": orchestrator,
        "beads_bridge": beads_bridge,
        "debugger": debugger_instance,
        "planner": planner_instance,
        "git_tools": GitTools(repo_path=str(project_root)),
        "project_root": project_root,
        "model_adapter": model_adapter,
        "memory_store": memory_store,
        "brain": brain_instance,
        "vector_store": vector_store,
        "router": router,
        "config_api_key": config_api_key,
        "provider_status": provider_status,
        "memory_persistence_path": paths["memory_persistence_path"],
        "beads_cli_override": beads_cli_override,
    }


class RuntimeFactory:
    """Thin facade for constructing AURA runtimes from external callers."""

    @staticmethod
    def create(project_root: Path, overrides: dict | None = None) -> dict:
        """Create and return a fully-configured AURA runtime dict."""
        return create_runtime(project_root, overrides=overrides)
