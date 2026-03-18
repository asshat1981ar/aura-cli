import os
import sys
import json
import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

try:
    import readline
except ImportError:
    readline = None

from core.config_manager import ConfigManager, DEFAULT_CONFIG, config
from core.goal_queue import GoalQueue
from core.goal_archive import GoalArchive
from memory.brain import Brain
from core.model_adapter import ModelAdapter
from core.orchestrator import LoopOrchestrator
from core.policy import Policy
from memory.store import MemoryStore
from agents.registry import default_agents
from core.git_tools import GitTools
from core.logging_utils import log_json
from agents.debugger import DebuggerAgent
from agents.planner import PlannerAgent
from agents.router import RouterAgent
from agents.scaffolder import ScaffolderAgent
from core.vector_store import VectorStore
from core.beads_bridge import BeadsBridge
from core.runtime_paths import resolve_project_path
from core.runtime_state import validate_brain_schema
from core.runtime_auth import (
    resolve_config_api_key,
    runtime_provider_status,
    runtime_provider_summary,
)

from core.task_handler import _check_project_writability, run_goals_loop
from aura_cli.commands import _handle_add, _handle_run, _handle_status, _handle_exit, _handle_help, _handle_doctor, _handle_clear
from aura_cli.cli_options import (
    CLIParseError,
    attach_cli_warnings,
    cli_parse_error_payload,
    parse_cli_args,
    render_help,
)
from aura_cli.options import action_runtime_required, action_default_canonical_path

# Dispatch handler modules (B4 extraction)
from aura_cli.dispatch._helpers import _print_json_payload, _run_json_printing_callable_with_warnings
from aura_cli.dispatch.goal import (
    handle_goal_once as _handle_goal_once_dispatch,
    handle_goal_run as _handle_goal_run_dispatch,
    handle_goal_add as _handle_goal_add_dispatch,
    handle_goal_add_run as _handle_goal_add_run_dispatch,
    handle_goal_status as _handle_goal_status_dispatch,
    handle_interactive as _handle_interactive_dispatch,
)
from aura_cli.dispatch.memory import (
    handle_memory_search as _handle_memory_search_dispatch,
    handle_memory_reindex as _handle_memory_reindex_dispatch,
)
from aura_cli.dispatch.mcp import (
    _mcp_headers, _mcp_base_url, _mcp_request,
    cmd_mcp_tools, cmd_mcp_call, cmd_diag, cmd_mcp_check, cmd_mcp_setup,
    handle_mcp_tools as _handle_mcp_tools_dispatch,
    handle_mcp_call as _handle_mcp_call_dispatch,
    handle_diag as _handle_diag_dispatch,
    handle_mcp_check as _handle_mcp_check_dispatch,
    handle_mcp_setup as _handle_mcp_setup_dispatch,
)
from aura_cli.dispatch.ops import (
    handle_queue_list as _handle_queue_list_dispatch,
    handle_queue_clear as _handle_queue_clear_dispatch,
    handle_skills_list as _handle_skills_list_dispatch,
    handle_metrics_show as _handle_metrics_show_dispatch,
    handle_workflow_run as _handle_workflow_run_dispatch,
    handle_scaffold as _handle_scaffold_dispatch,
    handle_evolve as _handle_evolve_dispatch,
    handle_logs as _handle_logs_dispatch,
    handle_watch as _handle_watch_dispatch,
)
from aura_cli.dispatch.config import (
    handle_help as _handle_help_dispatch,
    handle_json_help as _handle_json_help_dispatch,
    handle_doctor as _handle_doctor_dispatch,
    handle_bootstrap as _handle_bootstrap_dispatch,
    handle_show_config as _handle_show_config_dispatch,
    handle_contract_report as _handle_contract_report_dispatch,
)
from aura_cli.dispatch.chat import interactive_chat


def cli_interaction_loop(args, runtime):
    def _get_runtime_part(key):
        return runtime.get(key)

    project_root = runtime["project_root"]
    chat_history = []

    print("Type 'help' for commands, or just chat with AURA in natural language.")
    
    # --- Iteration 3: Proactive State Management (Greeting) ---
    pending_count = len(runtime["goal_queue"].queue)
    startup_prompt = f"System: AURA has just started. There are {pending_count} pending goals in the queue. Provide a very brief, friendly 1-2 sentence greeting to the user summarizing this and asking what they would like to do. Use the 'reply' action."
    interactive_chat(runtime, startup_prompt, chat_history)
    
    while True:
        try:
            command_line = input("\nAURA > ").strip()
            command_parts = command_line.split(maxsplit=1)
            if not command_parts:
                continue
            cmd_name = command_parts[0]
            
            if cmd_name == "add":
                _handle_add(runtime["goal_queue"], command_line)
            elif cmd_name == "run":
                # Upgrade runtime if it's currently a queue-only runtime
                orchestrator = runtime.get("orchestrator")
                if isinstance(orchestrator, SimpleNamespace):
                    print("Initializing full AURA runtime for execution...")
                    full_runtime = create_runtime(project_root, overrides={"runtime_mode": "full"})
                    runtime.update(full_runtime)
                
                orchestrator = runtime.get("orchestrator")
                _handle_run(
                    args, 
                    runtime["goal_queue"], 
                    runtime["goal_archive"], 
                    orchestrator, 
                    runtime["debugger"], 
                    runtime["planner"], 
                    project_root
                )
            elif cmd_name == "status":
                _handle_status(
                    runtime["goal_queue"], 
                    runtime["goal_archive"], 
                    runtime.get("orchestrator"), 
                    project_root=project_root, 
                    memory_persistence_path=runtime.get("memory_persistence_path")
                )
            elif cmd_name == "doctor":
                _handle_doctor(project_root)
            elif cmd_name == "clear":
                _handle_clear()
            elif cmd_name == "help":
                _handle_help()
            elif cmd_name == "exit":
                _handle_exit()
                break
            else:
                action = interactive_chat(runtime, command_line, chat_history)
                if action == "run_goals":
                    # Re-inject the "run" command logic
                    orchestrator = runtime.get("orchestrator")
                    if isinstance(orchestrator, SimpleNamespace):
                        print("Initializing full AURA runtime for execution...")
                        full_runtime = create_runtime(project_root, overrides={"runtime_mode": "full"})
                        runtime.update(full_runtime)
                    
                    orchestrator = runtime.get("orchestrator")
                    _handle_run(
                        args, 
                        runtime["goal_queue"], 
                        runtime["goal_archive"], 
                        orchestrator, 
                        runtime["debugger"], 
                        runtime["planner"], 
                        project_root
                    )

        except EOFError:
            log_json("INFO", "aura_cli_eof_received", details={"message": "End of input received, exiting."})
            break



# ── Thin adapter classes for improvement loops ────────────────────────────────

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
            # fallback: try to find goal string anywhere in phase_outputs
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
    for key in [
        "goal_queue_path", "goal_archive_path", "brain_db_path",
        "memory_store_path", "memory_persistence_path"
    ]:
        paths[key] = resolve_project_path(
            project_root,
            runtime_config.get(key, DEFAULT_CONFIG[key]),
            DEFAULT_CONFIG[key]
        )
    return paths


def _init_memory_and_brain(brain_db_path: Path):
    """Initialize the brain instance and cache adapter."""
    try:
        from memory.cache_adapter_factory import create_cache_adapter
        from memory.momento_brain import MomentoBrain
        _momento = create_cache_adapter()
        brain_instance = MomentoBrain(_momento, db_path=str(brain_db_path))
        log_json("INFO", "runtime_cache_adapter_active", details={"adapter": type(_momento).__name__})
        validation = validate_brain_schema(brain_db_path)
        log_json(
            "INFO" if validation["ok"] else "WARN",
            "brain_schema_validated",
            details=validation,
        )
        return brain_instance, _momento
    except Exception as _exc:
        log_json("WARN", "cache_adapter_init_failed", details={"error": str(_exc)})
        brain_instance = Brain(db_path=str(brain_db_path))
        validation = validate_brain_schema(brain_db_path)
        log_json(
            "INFO" if validation["ok"] else "WARN",
            "brain_schema_validated",
            details=validation,
        )
        return brain_instance, None


def _start_background_sync(project_root: Path, vector_store, context_graph):
    """Start the project knowledge syncer in a background thread."""
    from core.project_syncer import ProjectKnowledgeSyncer
    import threading
    import atexit

    syncer = ProjectKnowledgeSyncer(
        vector_store=vector_store,
        context_graph=context_graph,
        project_root=project_root
    )
    _stop_event = threading.Event()

    def _bg_sync():
        try:
            syncer.sync_all()
        except Exception as e:
            log_json("WARN", "background_sync_failed", details={"error": str(e)})

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

    # 1. Improvement Loops
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

        # 1.1 Beads Sync Loop
        if getattr(orchestrator, "beads_enabled", False) and "beads_skill" in orchestrator.skills:
            from core.orchestrator import BeadsSyncLoop
            _beads_sync = BeadsSyncLoop(orchestrator.skills["beads_skill"])
            orchestrator.attach_improvement_loops(_beads_sync)
    except Exception as _exc:
        log_json("WARN", "improvement_loops_setup_failed", details={"error": str(_exc)})

    # 2. CASPA-W
    try:
        from core.context_graph import ContextGraph
        from core.adaptive_pipeline import AdaptivePipeline
        from core.propagation_engine import PropagationEngine
        from core.autonomous_discovery import AutonomousDiscovery
        from core.evolution_loop import EvolutionLoop
        from agents.mutator import MutatorAgent

        _context_graph = ContextGraph()
        _adaptive_pipeline = AdaptivePipeline(
            context_graph=_context_graph,
            skill_weight_adapter=_skill_adapt if "_skill_adapt" in locals() else None,
            memory_store=memory_store,
            brain=brain,
        )
        _propagation = PropagationEngine(goal_queue, _context_graph, memory_store)
        _discovery = AutonomousDiscovery(goal_queue, memory_store, project_root=str(project_root))
        
        # Evolution Loop
        from core.recursive_improvement import RecursiveImprovementService
        _ri_service = RecursiveImprovementService()
        # EvolutionLoop expects the raw agents, not the orchestrator's adapters
        _coder_adapter = orchestrator.agents.get("act")
        _critic_adapter = orchestrator.agents.get("critique")
        _planner_adapter = orchestrator.agents.get("plan")
        _coder = getattr(_coder_adapter, "agent", _coder_adapter)
        _critic = getattr(_critic_adapter, "agent", _critic_adapter)
        _planner = getattr(_planner_adapter, "agent", _planner_adapter)
        _git = GitTools(repo_path=str(project_root))
        _mutator = MutatorAgent(project_root)
        _evo = EvolutionLoop(_planner, _coder, _critic, brain, getattr(brain, "vector_store", None), _git, _mutator, improvement_service=_ri_service)

        orchestrator.attach_caspa(
            adaptive_pipeline=_adaptive_pipeline,
            propagation_engine=_propagation,
            context_graph=_context_graph,
        )
        orchestrator.discovery_loop = _discovery
        orchestrator.evolution_loop = _evo
        # No longer attaching to improvement_loops as they are explicit phases in PRD-003
    except Exception as _exc:
        log_json("WARN", "caspa_setup_failed", details={"error": str(_exc)})


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

    # Context Manager
    try:
        from core.context_manager import ContextManager
        context_manager = ContextManager(vector_store=vector_store, project_root=project_root)
        if runtime_mode != "lean":
            _start_background_sync(project_root, vector_store, None)
    except Exception as _exc:
        log_json("WARN", "context_manager_init_failed", details={"error": str(_exc)})
        context_manager = None

    memory_store = (
        _get_momento_store(paths["memory_store_path"], _momento) 
        if _momento else MemoryStore(paths["memory_store_path"])
    )
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

    orchestrator = LoopOrchestrator(
        agents=default_agents(brain_instance, model_adapter, context_manager=context_manager),
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
    try:
        git_tools = GitTools(repo_path=str(project_root))
    except Exception as exc:
        log_json("WARN", "git_tools_runtime_unavailable", details={"error": str(exc), "project_root": str(project_root)})
        git_tools = None

    return {
        "goal_queue": goal_queue,
        "goal_archive": goal_archive,
        "orchestrator": orchestrator,
        "beads_bridge": beads_bridge,
        "debugger": debugger_instance,
        "planner": planner_instance,
        "git_tools": git_tools,
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


def _get_momento_store(root, momento):
    try:
        from memory.momento_memory_store import MomentoMemoryStore
        return MomentoMemoryStore(root, momento)
    except Exception as e:
        log_json("WARN", "memory_store_init_failed", details={"error": str(e)})
        return MemoryStore(root)


@dataclass
class DispatchContext:
    parsed: object
    project_root: Path
    runtime_factory: object
    args: object
    runtime: dict | None = None


@dataclass(frozen=True)
class DispatchRule:
    action: str
    requires_runtime: bool
    handler: object


def _resolve_dispatch_action(parsed) -> str:
    action = getattr(parsed, "action", None)
    if action:
        return action
    return "interactive"


def _prepare_runtime_context(ctx: DispatchContext) -> int | None:
    if ctx.runtime is not None:
        return None

    args = ctx.args
    overrides: dict[str, object] = {}
    if getattr(args, "dry_run", False):
        overrides["dry_run"] = True
    if getattr(args, "decompose", False):
        overrides["decompose"] = True
    if getattr(args, "model", None):
        overrides["model_name"] = args.model

    beads_config = dict(config.get("beads", DEFAULT_CONFIG["beads"]) or {})
    beads_override_requested = False
    beads_cli_override: dict[str, object] | None = None
    if getattr(args, "beads", False):
        beads_config["enabled"] = True
        beads_override_requested = True
    if getattr(args, "no_beads", False):
        beads_config["enabled"] = False
        beads_override_requested = True
    if getattr(args, "beads_required", False):
        beads_config["enabled"] = True
        beads_config["required"] = True
        beads_override_requested = True
    if getattr(args, "beads_optional", False):
        beads_config["enabled"] = True
        beads_config["required"] = False
        beads_override_requested = True
    if beads_override_requested:
        overrides["beads"] = beads_config
        beads_cli_override = {
            "source": "cli",
            "enabled": bool(beads_config.get("enabled", True)),
            "required": bool(beads_config.get("required", True)),
        }
        overrides["beads_cli_override"] = beads_cli_override

    if _resolve_dispatch_action(ctx.parsed) in {"goal_status", "goal_add", "interactive"}:
        overrides["runtime_mode"] = "queue"
    elif _resolve_dispatch_action(ctx.parsed) == "goal_once" and getattr(args, "dry_run", False):
        overrides["runtime_mode"] = "lean"

    ctx.runtime = ctx.runtime_factory(ctx.project_root, overrides=overrides or None)
    log_json("INFO", "aura_cli_online", details={"dry_run_mode": getattr(args, 'dry_run', False)})

    if not _check_project_writability(ctx.project_root):
        log_json("CRITICAL", "aura_cli_startup_aborted_not_writable")
        return 1
    return None


def _dispatch_rule(action: str, handler) -> DispatchRule:
    return DispatchRule(action, action_runtime_required(action), handler)


COMMAND_DISPATCH_REGISTRY = {
    "json_help": _dispatch_rule("json_help", _handle_json_help_dispatch),
    "help": _dispatch_rule("help", _handle_help_dispatch),
    "doctor": _dispatch_rule("doctor", _handle_doctor_dispatch),
    "bootstrap": _dispatch_rule("bootstrap", _handle_bootstrap_dispatch),
    "show_config": _dispatch_rule("show_config", _handle_show_config_dispatch),
    "contract_report": _dispatch_rule("contract_report", _handle_contract_report_dispatch),
    "mcp_tools": _dispatch_rule("mcp_tools", _handle_mcp_tools_dispatch),
    "mcp_call": _dispatch_rule("mcp_call", _handle_mcp_call_dispatch),
    "mcp_check": _dispatch_rule("mcp_check", _handle_mcp_check_dispatch),
    "mcp_setup": _dispatch_rule("mcp_setup", _handle_mcp_setup_dispatch),
    "diag": _dispatch_rule("diag", _handle_diag_dispatch),
    "logs": _dispatch_rule("logs", _handle_logs_dispatch),
    "watch": _dispatch_rule("watch", _handle_watch_dispatch),
    "studio": _dispatch_rule("studio", _handle_watch_dispatch),
    "queue_list": _dispatch_rule("queue_list", _handle_queue_list_dispatch),
    "queue_clear": _dispatch_rule("queue_clear", _handle_queue_clear_dispatch),
    "memory_search": _dispatch_rule("memory_search", _handle_memory_search_dispatch),
    "memory_reindex": _dispatch_rule("memory_reindex", _handle_memory_reindex_dispatch),
    "metrics_show": _dispatch_rule("metrics_show", _handle_metrics_show_dispatch),
    "skills_list": _dispatch_rule("skills_list", _handle_skills_list_dispatch),
    "workflow_run": _dispatch_rule("workflow_run", _handle_workflow_run_dispatch),
    "scaffold": _dispatch_rule("scaffold", _handle_scaffold_dispatch),
    "evolve": _dispatch_rule("evolve", _handle_evolve_dispatch),
    "goal_status": _dispatch_rule("goal_status", _handle_goal_status_dispatch),
    "goal_add": _dispatch_rule("goal_add", _handle_goal_add_dispatch),
    "goal_add_run": _dispatch_rule("goal_add_run", _handle_goal_add_run_dispatch),
    "goal_once": _dispatch_rule("goal_once", _handle_goal_once_dispatch),
    "goal_run": _dispatch_rule("goal_run", _handle_goal_run_dispatch),
    "interactive": _dispatch_rule("interactive", _handle_interactive_dispatch),
}


def dispatch_command(parsed, *, project_root: Path, runtime_factory=create_runtime):
    ctx = DispatchContext(parsed=parsed, project_root=project_root, runtime_factory=runtime_factory, args=parsed.namespace)

    warning_records = getattr(parsed, "warning_records", None) or []
    if warning_records:
        for warning in warning_records:
            print(f"Warning: {warning.message}", file=sys.stderr)
    else:
        for warning in parsed.warnings:
            print(f"Warning: {warning}", file=sys.stderr)

    action = _resolve_dispatch_action(parsed)
    rule = COMMAND_DISPATCH_REGISTRY.get(action)
    if rule is None:
        print(f"Error: No dispatch rule registered for action '{action}'", file=sys.stderr)
        return 1

    if rule.requires_runtime:
        prep_rc = _prepare_runtime_context(ctx)
        if prep_rc is not None:
            return prep_rc

    return rule.handler(ctx)


def main(project_root_override=None, argv=None):
    # Resolve project root
    if project_root_override:
        project_root = Path(project_root_override)
    else:
        # Since this file is now in aura_cli/cli_main.py, parent is aura_cli/, parent.parent is project root
        project_root = Path(__file__).resolve().parent.parent
        
        # If we are in a test environment, prefer the current working directory
        if os.getenv("AURA_SKIP_CHDIR") == "1":
            project_root = Path.cwd()
        else:
            # Change directory to the project root for consistency in real-world usage
            os.chdir(project_root)

    # Ensure project root is on path for module imports
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Setup readline history
    if readline:
        history_file = project_root / "memory" / ".aura_history"
        try:
            if history_file.exists():
                readline.read_history_file(str(history_file))
            readline.set_history_length(1000)
        except Exception:
            pass # Silent fail if history loading fails

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        parsed = parse_cli_args(raw_argv)
    except CLIParseError as exc:
        if "--json" in raw_argv:
            print(json.dumps(attach_cli_warnings(cli_parse_error_payload(exc))))
        else:
            print(f"Error: {exc}", file=sys.stderr)
            if exc.usage:
                print(exc.usage, file=sys.stderr)
        return exc.code

    try:
        return dispatch_command(parsed, project_root=project_root)
    finally:
        if readline:
            try:
                readline.write_history_file(str(project_root / "memory" / ".aura_history"))
            except Exception:
                pass

if __name__ == "__main__":
    raise SystemExit(main())
