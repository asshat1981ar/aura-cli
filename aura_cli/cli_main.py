import os
import sys
import json
import io
import urllib.request
import urllib.error
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

try:
    import readline
except ImportError:
    readline = None

from core.config_manager import DEFAULT_CONFIG, config
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
from core.vector_store import VectorStore
from core.runtime_paths import resolve_project_path
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
    unknown_command_help_topic_payload,
)
from aura_cli.options import action_runtime_required


def _mcp_headers():
    """Return auth headers for MCP requests, if MCP_API_TOKEN is set."""
    token = os.getenv("MCP_API_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _mcp_base_url():
    """Base URL for MCP server (default http://localhost:8001)."""
    return os.getenv("MCP_SERVER_URL", "http://localhost:8001")


def _mcp_request(method: str, path: str, data: dict | None = None):
    """Small HTTP helper for MCP server; returns (status, json/dict)."""
    url = f"{_mcp_base_url()}{path}"
    headers = {"Content-Type": "application/json", **_mcp_headers()}
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {"error": str(e)}
    except Exception as e:
        return 500, {"error": str(e)}


def cmd_mcp_tools():
    """List MCP tools via HTTP client."""
    status, data = _mcp_request("GET", "/tools")
    print(json.dumps({"status": status, "data": data}, indent=2))


def cmd_mcp_call(tool: str, args_json: str | None):
    """Call an MCP tool by name with JSON args."""
    try:
        args_obj = json.loads(args_json) if args_json else {}
    except json.JSONDecodeError as exc:
        print(f"Invalid args JSON: {exc}")
        return
    payload = {"tool_name": tool, "args": args_obj}
    status, data = _mcp_request("POST", "/call", payload)
    print(json.dumps({"status": status, "data": data}, indent=2))


def cmd_diag():
    """Fetch MCP health/metrics/limits/log tail and linter capabilities."""
    health = _mcp_request("GET", "/health")
    metrics = _mcp_request("GET", "/metrics")
    limits = _mcp_request("POST", "/call", {"tool_name": "limits", "args": {}})
    lcap = _mcp_request("POST", "/call", {"tool_name": "linter_capabilities", "args": {}})
    tail = _mcp_request("POST", "/call", {"tool_name": "tail_logs", "args": {"lines": 50}})
    print(json.dumps({
        "health": {"status": health[0], "data": health[1]},
        "metrics": {"status": metrics[0], "data": metrics[1]},
        "limits": {"status": limits[0], "data": limits[1]},
        "linter_capabilities": {"status": lcap[0], "data": lcap[1]},
        "tail_logs": {"status": tail[0], "data": tail[1]},
    }, indent=2))

def cli_interaction_loop(args, runtime):
    def _get_runtime_part(key):
        return runtime.get(key)

    project_root = runtime["project_root"]

    while True:
        try:
            command_line = input("\nCommand (add/run/exit/status/help) > ").strip()
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
                    _ensure_legacy_loop(runtime, project_root=project_root)
                
                _handle_run(
                    args, 
                    runtime["goal_queue"], 
                    runtime["goal_archive"], 
                    runtime["loop"], 
                    runtime["debugger"], 
                    runtime["planner"], 
                    project_root
                )
            elif cmd_name == "status":
                _handle_status(
                    runtime["goal_queue"], 
                    runtime["goal_archive"], 
                    runtime.get("loop"), 
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
                log_json("WARN", "invalid_cli_command", details={"command": cmd_name})
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


def _resolve_runtime_paths(project_root: Path) -> dict:
    """Resolve all required storage paths against the project root."""
    paths = {}
    for key in [
        "goal_queue_path", "goal_archive_path", "brain_db_path",
        "memory_store_path", "memory_persistence_path"
    ]:
        paths[key] = resolve_project_path(
            project_root,
            config.get(key, DEFAULT_CONFIG[key]),
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
        return brain_instance, _momento
    except Exception as _exc:
        log_json("WARN", "cache_adapter_init_failed", details={"error": str(_exc)})
        return Brain(db_path=str(brain_db_path)), None


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
    except Exception as _exc:
        log_json("WARN", "improvement_loops_setup_failed", details={"error": str(_exc)})

    # 2. CASPA-W
    try:
        from core.context_graph import ContextGraph
        from core.adaptive_pipeline import AdaptivePipeline
        from core.propagation_engine import PropagationEngine
        from core.autonomous_discovery import AutonomousDiscovery

        _context_graph = ContextGraph()
        _adaptive_pipeline = AdaptivePipeline(
            context_graph=_context_graph,
            skill_weight_adapter=_skill_adapt if "_skill_adapt" in locals() else None,
            memory_store=memory_store,
        )
        _propagation = PropagationEngine(goal_queue, _context_graph, memory_store)
        _discovery = AutonomousDiscovery(goal_queue, memory_store, project_root=str(project_root))

        orchestrator.attach_caspa(
            adaptive_pipeline=_adaptive_pipeline,
            propagation_engine=_propagation,
            context_graph=_context_graph,
        )
        orchestrator.attach_improvement_loops(_discovery)
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

    for k, v in runtime_overrides.items():
        config.set_runtime_override(k, v)

    paths = _resolve_runtime_paths(project_root)
    goal_queue = GoalQueue(str(paths["goal_queue_path"]))
    goal_archive = GoalArchive(str(paths["goal_archive_path"]))
    
    config_api_key = resolve_config_api_key(config.get("api_key", ""))
    provider_status = runtime_provider_status(config_api_key=config_api_key)

    if not provider_status["chat_ready"]:
        log_json("WARN", "aura_api_key_missing", details={"providers": runtime_provider_summary(provider_status)})
        print("⚠️  AURA: No chat provider configured.", file=sys.stderr)

    if runtime_mode == "queue":
        return {
            "goal_queue": goal_queue,
            "goal_archive": goal_archive,
            "orchestrator": SimpleNamespace(),
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

    orchestrator = LoopOrchestrator(
        agents=default_agents(brain_instance, model_adapter, context_manager=context_manager),
        memory_store=memory_store,
        policy=Policy.from_config(config.effective_config.copy()),
        project_root=project_root,
        strict_schema=config.get("strict_schema", False),
        debugger=debugger_instance,
        goal_queue=goal_queue,
    )

    _attach_advanced_loops(orchestrator, runtime_mode, brain_instance, memory_store, goal_queue, _momento, project_root)

    return {
        "goal_queue": goal_queue,
        "goal_archive": goal_archive,
        "orchestrator": orchestrator,
        "debugger": debugger_instance,
        "planner": planner_instance,
        # Legacy loop is initialized lazily so non-legacy commands do not
        # instantiate deprecated HybridClosedLoop unnecessarily.
        "loop": None,
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


def _ensure_legacy_loop(runtime: dict, *, project_root: Path) -> object:
    # Lazy import so HybridClosedLoop is only loaded for legacy commands.
    from core.hybrid_loop import HybridClosedLoop

    loop = runtime.get("loop")
    if loop is not None:
        return loop

    model_adapter = runtime.get("model_adapter")
    brain_instance = runtime.get("brain")
    git_tools_instance = runtime.get("git_tools")
    if git_tools_instance is None:
        git_tools_instance = GitTools(repo_path=str(project_root))
        runtime["git_tools"] = git_tools_instance

    loop = HybridClosedLoop(model_adapter, brain_instance, git_tools_instance)
    runtime["loop"] = loop
    return loop


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


def _handle_help_dispatch(ctx: DispatchContext) -> int:
    try:
        print(render_help(getattr(ctx.args, "help_topics", None)))
    except ValueError as exc:
        if getattr(ctx.args, "json", False):
            print(json.dumps(attach_cli_warnings(unknown_command_help_topic_payload(str(exc)), ctx.parsed)))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


def _handle_json_help_dispatch(_ctx: DispatchContext) -> int:
    print(render_help(format="json"))
    return 0


def _handle_doctor_dispatch(_ctx: DispatchContext) -> int:
    _handle_doctor(_ctx.project_root)
    return 0


def _handle_bootstrap_dispatch(_ctx: DispatchContext) -> int:
    config.interactive_bootstrap()
    return 0


def _handle_show_config_dispatch(_ctx: DispatchContext) -> int:
    """Print the resolved effective configuration as JSON."""
    print(json.dumps(config.show_config(), indent=2, default=str))
    return 0


def _handle_contract_report_dispatch(ctx: DispatchContext) -> int:
    from aura_cli.contract_report import (
        build_cli_contract_report,
        cli_contract_report_exit_code,
        cli_contract_report_failure_message,
        render_cli_contract_report,
    )

    report = build_cli_contract_report(
        include_dispatch=not getattr(ctx.args, "no_dispatch", False),
        dispatch_registry=COMMAND_DISPATCH_REGISTRY,
    )
    print(render_cli_contract_report(report, compact=getattr(ctx.args, "compact", False)), end="")

    exit_code = cli_contract_report_exit_code(report, check=getattr(ctx.args, "check", False))
    if exit_code:
        print(cli_contract_report_failure_message(report), file=sys.stderr)
    return exit_code


def _print_json_payload(payload: dict, *, parsed=None, **json_kwargs) -> None:
    print(json.dumps(attach_cli_warnings(payload, parsed), **json_kwargs))


def _run_json_printing_callable_with_warnings(ctx: DispatchContext, func, *args, **kwargs) -> None:
    warning_records = getattr(ctx.parsed, "warning_records", None) or []
    if not warning_records:
        func(*args, **kwargs)
        return

    buf = io.StringIO()
    with redirect_stdout(buf):
        func(*args, **kwargs)
    raw = buf.getvalue()
    if raw == "":
        return

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        print(raw, end="")
        return

    _print_json_payload(payload, parsed=ctx.parsed, indent=2)


def _handle_mcp_tools_dispatch(ctx: DispatchContext) -> int:
    _run_json_printing_callable_with_warnings(ctx, cmd_mcp_tools)
    return 0


def _handle_mcp_call_dispatch(ctx: DispatchContext) -> int:
    _run_json_printing_callable_with_warnings(ctx, cmd_mcp_call, ctx.args.mcp_call, ctx.args.mcp_args)
    return 0


def _handle_diag_dispatch(ctx: DispatchContext) -> int:
    _run_json_printing_callable_with_warnings(ctx, cmd_diag)
    return 0


def _handle_logs_dispatch(ctx: DispatchContext) -> int:
    from aura_cli.tui.log_streamer import LogStreamer

    streamer = LogStreamer(level_filter=getattr(ctx.args, "level", "info"))
    if getattr(ctx.args, "file", None):
        streamer.stream_file(Path(ctx.args.file), tail=getattr(ctx.args, "tail", None), follow=getattr(ctx.args, "follow", False))
    else:
        streamer.stream_stdin(tail=getattr(ctx.args, "tail", None))
    return 0


def _handle_watch_dispatch(ctx: DispatchContext) -> int:
    from aura_cli.tui.app import AuraStudio

    studio = AuraStudio(runtime=ctx.runtime or {})
    orchestrator = ctx.runtime.get("orchestrator")
    if orchestrator:
        orchestrator.attach_ui_callback(studio)
    
    studio.run(autonomous=getattr(ctx.args, "autonomous", False))
    return 0


def _handle_queue_list_dispatch(ctx: DispatchContext) -> int:
    goal_queue = ctx.runtime["goal_queue"]
    if not goal_queue.queue:
        print("Goal queue is empty.")
        return 0
    
    print(f"Goal Queue ({len(goal_queue.queue)} goals):")
    for i, goal in enumerate(goal_queue.queue, 1):
        print(f"  {i}. {goal}")
    return 0


def _handle_queue_clear_dispatch(ctx: DispatchContext) -> int:
    goal_queue = ctx.runtime["goal_queue"]
    count = len(goal_queue.queue)
    goal_queue.queue = []
    goal_queue._save()
    print(f"Cleared {count} goals from the queue.")
    return 0


def _handle_memory_search_dispatch(ctx: DispatchContext) -> int:
    from core.memory_types import RetrievalQuery
    
    vector_store = ctx.runtime["vector_store"]
    query = RetrievalQuery(
        query_text=ctx.args.query,
        k=ctx.args.limit
    )
    hits = vector_store.search(query)
    
    if not hits:
        print(f"No results found for '{ctx.args.query}'")
        return 0
        
    print(f"Memory Search Results for '{ctx.args.query}':\n")
    for i, hit in enumerate(hits, 1):
        print(f"[{i}] Score: {hit.score:.3f} | Source: {hit.source_ref}")
        print(f"Content: {hit.content[:200]}...")
        print("-" * 40)
    return 0


def _handle_metrics_show_dispatch(ctx: DispatchContext) -> int:
    brain = ctx.runtime["brain"]
    outcomes = [e for e in brain.recall_recent(limit=100) if "outcome:" in e]
    
    if not outcomes:
        print("No metrics recorded yet.")
        return 0
        
    successes = 0
    total_time = 0.0
    
    print("Recent Cycle Metrics (last 10 cycles):\n")
    print(f"{'Cycle ID':<10} | {'Status':<8} | {'Duration':<8} | {'Goal'}")
    print("-" * 60)
    
    for raw in outcomes[:10]:
        try:
            # Format: outcome:id -> json
            data = json.loads(raw.split("->", 1)[1])
            status = "SUCCESS" if data.get("success") else "FAILED"
            duration = data.get("completed_at", 0) - data.get("started_at", 0)
            
            if data.get("success"): successes += 1
            total_time += duration
            
            print(f"{data.get('cycle_id')[:8]:<10} | {status:<8} | {duration:>7.1f}s | {data.get('goal')[:30]}...")
        except Exception:
            continue
            
    print("-" * 60)
    avg_time = total_time / len(outcomes[:10]) if outcomes else 0
    win_rate = (successes / len(outcomes[:10]) * 100) if outcomes else 0
    print(f"Summary: {win_rate:.1f}% success rate | Avg duration: {avg_time:.1f}s")
    return 0


def _handle_workflow_run_dispatch(ctx: DispatchContext) -> int:
    args = ctx.args
    orchestrator = ctx.runtime["orchestrator"]
    result = orchestrator.run_loop(
        args.workflow_goal,
        max_cycles=args.workflow_max_cycles or args.max_cycles or config.get("policy_max_cycles", config.get("max_cycles", 5)),
        dry_run=args.dry_run,
    )
    _print_json_payload(
        {"goal": args.workflow_goal, "stop_reason": result.get("stop_reason"), "cycles": len(result.get("history", []))},
        parsed=ctx.parsed,
        indent=2,
    )
    return 0


def _handle_scaffold_dispatch(ctx: DispatchContext) -> int:
    args = ctx.args
    runtime = ctx.runtime
    from agents.scaffolder import ScaffolderAgent

    scaffolder = ScaffolderAgent(runtime.get("brain", None) or __import__("memory.brain", fromlist=["Brain"]).Brain(), runtime["model_adapter"])
    result = scaffolder.scaffold_project(args.scaffold, args.scaffold_desc)
    print(result)
    return 0


def _handle_evolve_dispatch(ctx: DispatchContext) -> int:
    args = ctx.args
    runtime = ctx.runtime
    from core.evolution_loop import EvolutionLoop
    from agents.mutator import MutatorAgent
    from core.vector_store import VectorStore

    _brain = runtime.get("brain") or __import__("memory.brain", fromlist=["Brain"]).Brain()
    _model = runtime["model_adapter"]
    _planner = runtime["planner"]
    _coder = default_agents(_brain, _model).get("act")
    _critic = default_agents(_brain, _model).get("critique")
    _git = GitTools(repo_path=str(ctx.project_root))
    _mutator = MutatorAgent(ctx.project_root)
    _vec = VectorStore(_model, _brain)
    evo = EvolutionLoop(_planner, _coder, _critic, _brain, _vec, _git, _mutator)
    goal = args.goal or args.workflow_goal or "evolve and improve the AURA system"
    result = evo.run(goal)
    _print_json_payload(result, parsed=ctx.parsed, indent=2, default=str)
    return 0


def _handle_goal_status_dispatch(ctx: DispatchContext) -> int:
    runtime = ctx.runtime
    if ctx.args.json:
        _run_json_printing_callable_with_warnings(
            ctx,
            _handle_status,
            runtime["goal_queue"],
            runtime["goal_archive"],
            runtime["orchestrator"],
            as_json=True,
            project_root=ctx.project_root,
            memory_persistence_path=runtime.get("memory_persistence_path"),
        )
    else:
        _handle_status(
            runtime["goal_queue"],
            runtime["goal_archive"],
            runtime["orchestrator"],
            as_json=False,
            project_root=ctx.project_root,
            memory_persistence_path=runtime.get("memory_persistence_path"),
        )
    return 0


def _maybe_add_goal(ctx: DispatchContext) -> None:
    if not getattr(ctx.args, "add_goal", None):
        return
    goal_queue = ctx.runtime["goal_queue"]
    goal_queue.add(ctx.args.add_goal)
    log_json("INFO", "goal_added_from_cli", goal=ctx.args.add_goal)
    if not getattr(ctx.args, "json", False):
        print(f"Added goal: {ctx.args.add_goal}")
        print(f"Queue length: {len(goal_queue.queue)}")


def _handle_goal_once_dispatch(ctx: DispatchContext) -> int:
    from core.explain import format_decision_log

    args = ctx.args
    orchestrator = ctx.runtime["orchestrator"]
    result = orchestrator.run_loop(
        args.goal,
        max_cycles=args.max_cycles or config.get("policy_max_cycles", config.get("max_cycles", 5)),
        dry_run=args.dry_run,
    )
    history = result.get("history", [])
    if args.explain:
        print(format_decision_log(history))
    
    if getattr(args, "json", False):
        _print_json_payload(
            {"goal": args.goal, "stop_reason": result.get("stop_reason"), "cycles": len(history), "dry_run": args.dry_run},
            parsed=ctx.parsed,
            indent=2,
        )
    else:
        print(f"\n--- Goal Result Summary ---")
        print(f"Goal: {args.goal}")
        print(f"Stop Reason: {result.get('stop_reason')}")
        print(f"Cycles Completed: {len(history)}")
        if args.dry_run:
            print("Mode: Dry-run (read-only)")
        print("---------------------------\n")
    return 0


def _handle_goal_run_dispatch(ctx: DispatchContext) -> int:
    args = ctx.args
    runtime = ctx.runtime
    loop = _ensure_legacy_loop(runtime, project_root=ctx.project_root)
    run_goals_loop(
        args,
        runtime["goal_queue"],
        loop,
        runtime["debugger"],
        runtime["planner"],
        runtime["goal_archive"],
        ctx.project_root,
        decompose=args.decompose,
    )
    return 0


def _handle_goal_add_dispatch(ctx: DispatchContext) -> int:
    _maybe_add_goal(ctx)
    return 0


def _handle_goal_add_run_dispatch(ctx: DispatchContext) -> int:
    _maybe_add_goal(ctx)
    return _handle_goal_run_dispatch(ctx)


def _handle_interactive_dispatch(ctx: DispatchContext) -> int:
    runtime = ctx.runtime
    cli_interaction_loop(ctx.args, runtime)
    return 0


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
    "diag": _dispatch_rule("diag", _handle_diag_dispatch),
    "logs": _dispatch_rule("logs", _handle_logs_dispatch),
    "watch": _dispatch_rule("watch", _handle_watch_dispatch),
    "studio": _dispatch_rule("studio", _handle_watch_dispatch),
    "queue_list": _dispatch_rule("queue_list", _handle_queue_list_dispatch),
    "queue_clear": _dispatch_rule("queue_clear", _handle_queue_clear_dispatch),
    "memory_search": _dispatch_rule("memory_search", _handle_memory_search_dispatch),
    "metrics_show": _dispatch_rule("metrics_show", _handle_metrics_show_dispatch),
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
