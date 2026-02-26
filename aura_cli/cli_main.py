import os
import sys
import argparse
import json
import urllib.request
import urllib.error
from pathlib import Path

try:
    import readline
except ImportError:
    readline = None

from core.config_manager import config
from core.goal_queue import GoalQueue
from core.hybrid_loop import HybridClosedLoop
from core.goal_archive import GoalArchive
from memory.brain import Brain
from core.model_adapter import ModelAdapter
from core.orchestrator import LoopOrchestrator
from core.policy import Policy
from memory.store import MemoryStore
from agents.registry import default_agents
from core.git_tools import GitTools
from core.exceptions import GitToolsError
from core.logging_utils import log_json
from agents.debugger import DebuggerAgent
from agents.planner import PlannerAgent
from agents.router import RouterAgent
from core.vector_store import VectorStore

from core.task_handler import _check_project_writability, run_goals_loop
from aura_cli.commands import _handle_add, _handle_run, _handle_status, _handle_exit, _handle_help, _handle_doctor, _handle_clear


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

def cli_interaction_loop(args, goal_queue, goal_archive, loop, debugger_instance, planner_instance, project_root):
    while True:
        try:
            command_line = input("\nCommand (add/run/exit/status/help) > ").strip()
            command_parts = command_line.split(maxsplit=1)
            if not command_parts:
                continue
            cmd_name = command_parts[0]
            
            if cmd_name == "add":
                _handle_add(goal_queue, command_line)
            elif cmd_name == "run":
                _handle_run(args, goal_queue, goal_archive, loop, debugger_instance, planner_instance, project_root)
            elif cmd_name == "status":
                _handle_status(goal_queue, goal_archive, loop)
            elif cmd_name == "doctor":
                _handle_doctor()
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
            import json
            goal = str(entry.get("phase_outputs", {}).get("plan", {}).get("goal", ""))
        if goal:
            self._escape.check_and_escape(goal, entry)


def create_runtime(project_root: Path, overrides: dict | None = None):
    """
    Library-friendly initializer that sets up shared AURA runtime objects
    without user input loops or forced chdir. Used by the HTTP server.
    """
    # Ensure project root is on path for module imports
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Apply optional config overrides
    if overrides:
        for k, v in overrides.items():
            config.set_runtime_override(k, v)

    goal_queue = GoalQueue()
    goal_archive = GoalArchive()

    model_adapter = ModelAdapter()
    brain_instance = Brain()
    # Enable prompt-response cache (1hr TTL) using Brain's SQLite connection
    model_adapter.enable_cache(brain_instance.db, ttl_seconds=3600)
    # Attach VectorStore for semantic memory recall
    vector_store = VectorStore(model_adapter, brain_instance)
    brain_instance.set_vector_store(vector_store)
    # Attach EMA-ranked router to model_adapter
    router = RouterAgent(brain_instance, model_adapter)
    model_adapter.set_router(router)
    debugger_instance = DebuggerAgent(brain_instance, model_adapter)
    planner_instance = PlannerAgent(brain_instance, model_adapter)
    memory_store = MemoryStore(Path(config.get("memory_store_path", "memory/store")))
    policy_config = config.effective_config.copy()
    policy = Policy.from_config(policy_config)
    orchestrator = LoopOrchestrator(
        agents=default_agents(brain_instance, model_adapter),
        memory_store=memory_store,
        policy=policy,
        project_root=project_root,
        strict_schema=config.get("strict_schema", False),
        debugger=debugger_instance,
    )
    git_tools_instance = GitTools(repo_path=str(project_root))

    # ── Self-improvement loops ────────────────────────────────────────────────
    try:
        from core.reflection_loop import DeepReflectionLoop
        from core.health_monitor import HealthMonitor
        from core.weakness_remediator import WeaknessRemediator
        from core.skill_weight_adapter import SkillWeightAdapter
        from core.convergence_escape import ConvergenceEscapeLoop
        from core.memory_compaction import MemoryCompactionLoop

        _reflection   = DeepReflectionLoop(memory_store, brain_instance)
        _health       = HealthMonitor(orchestrator.skills, goal_queue, memory_store, project_root)
        _remediator   = _WeaknessRemediatorLoop(WeaknessRemediator(), brain_instance, goal_queue)
        _skill_adapt  = SkillWeightAdapter(memory_root=str(Path(config.get("memory_store_path", "memory/store")).parent))
        _conv_escape  = _ConvergenceEscapeLoop(ConvergenceEscapeLoop(memory_store, goal_queue))
        _compaction   = MemoryCompactionLoop(memory_store)

        orchestrator.attach_improvement_loops(
            _reflection,
            _health,
            _remediator,
            _skill_adapt,
            _conv_escape,
            _compaction,
        )
    except Exception as _exc:
        log_json("WARN", "improvement_loops_setup_failed", details={"error": str(_exc)})
    # ─────────────────────────────────────────────────────────────────────────

    loop = HybridClosedLoop(model_adapter, brain_instance, git_tools_instance)

    return {
        "goal_queue": goal_queue,
        "goal_archive": goal_archive,
        "orchestrator": orchestrator,
        "debugger": debugger_instance,
        "planner": planner_instance,
        "loop": loop,
        "project_root": project_root,
        "model_adapter": model_adapter,
        "memory_store": memory_store,
        "brain": brain_instance,
        "vector_store": vector_store,
        "router": router,
    }


def main(project_root_override=None):
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

    parser = argparse.ArgumentParser(description="AURA CLI for autonomous development.")
    parser.add_argument("--dry-run", action="store_true", help="Run the AURA loop in dry-run mode.")
    parser.add_argument("--add-goal", type=str, help="Add a goal to the queue.")
    parser.add_argument("--run-goals", action="store_true", help="Run the AURA loop for goals currently in the queue.")
    parser.add_argument("--decompose", action="store_true", help="Decompose complex goals into hierarchical sub-tasks.")
    parser.add_argument("--bootstrap", action="store_true", help="Initialize a default aura.config.json file.")
    parser.add_argument("--model", type=str, help="Override the model to use.")
    parser.add_argument("--max-cycles", type=int, help="Maximum cycles per goal.")
    parser.add_argument("--explain", action="store_true", help="Print decision logs after each run.")
    parser.add_argument("--goal", type=str, help="Run a single goal without using the queue.")
    parser.add_argument("--status", action="store_true", help="Show status and exit.")
    parser.add_argument("--json", action="store_true", help="Use JSON output where supported.")
    parser.add_argument("--policy", type=str, help="Select convergence policy (e.g., sliding_window, time_bound).")
    # New CLI helpers
    parser.add_argument("--workflow-goal", type=str, help="Run a one-off workflow goal (shortcut for orchestrator.run_loop).")
    parser.add_argument("--workflow-max-cycles", type=int, help="Max cycles for workflow goal.")
    parser.add_argument("--mcp-tools", action="store_true", help="List MCP tools via HTTP client.")
    parser.add_argument("--mcp-call", type=str, help="Call MCP tool by name.")
    parser.add_argument("--mcp-args", type=str, help="JSON args for --mcp-call.")
    parser.add_argument("--diag", action="store_true", help="Fetch MCP health/metrics/limits/log tail.")
    parser.add_argument("--evolve", action="store_true", help="Run EvolutionLoop (self-improvement mode) instead of standard orchestrator.")
    parser.add_argument("--scaffold", type=str, metavar="PROJECT_NAME", help="Scaffold a new project (use with --scaffold-desc).")
    parser.add_argument("--scaffold-desc", type=str, default="", help="Description for --scaffold.")
    args = parser.parse_args()

    # Initialize Unified Config with CLI overrides
    if args.bootstrap:
        config.bootstrap()
        print("AURA bootstrapped! Edit aura.config.json and add your API key.")
        return

    # MCP / diagnostics shortcuts (no full runtime needed)
    if args.mcp_tools:
        cmd_mcp_tools()
        return
    if args.mcp_call:
        cmd_mcp_call(args.mcp_call, args.mcp_args)
        return
    if args.diag:
        cmd_diag()
        return

    if args.dry_run:
        config.set_runtime_override("dry_run", True)
    if args.decompose:
        config.set_runtime_override("decompose", True)
    if args.model:
        config.set_runtime_override("model_name", args.model)

    # Build runtime (CLI-specific overrides applied below)
    runtime = create_runtime(project_root, overrides=None)
    goal_queue = runtime["goal_queue"]
    goal_archive = runtime["goal_archive"]
    orchestrator = runtime["orchestrator"]
    debugger_instance = runtime["debugger"]
    planner_instance = runtime["planner"]
    loop = runtime["loop"]

    log_json("INFO", "aura_cli_online", details={"dry_run_mode": getattr(args, 'dry_run', False)})

    if not _check_project_writability(project_root):
        log_json("CRITICAL", "aura_cli_startup_aborted_not_writable")
        return

    # One-off workflow goal (bypasses queue)
    if args.workflow_goal:
        result = orchestrator.run_loop(
            args.workflow_goal,
            max_cycles=args.workflow_max_cycles or args.max_cycles or config.get("policy_max_cycles", config.get("max_cycles", 5)),
            dry_run=args.dry_run,
        )
        print(json.dumps({"goal": args.workflow_goal, "stop_reason": result.get("stop_reason"), "cycles": len(result.get("history", []))}, indent=2))
        return

    # Scaffold a new project via ScaffolderAgent
    if args.scaffold:
        from agents.scaffolder import ScaffolderAgent
        scaffolder = ScaffolderAgent(runtime.get("brain", None) or __import__("memory.brain", fromlist=["Brain"]).Brain(), runtime["model_adapter"])
        result = scaffolder.scaffold_project(args.scaffold, args.scaffold_desc)
        print(result)
        return

    # Self-improvement EvolutionLoop mode
    if args.evolve:
        from core.evolution_loop import EvolutionLoop
        from agents.mutator import MutatorAgent
        from core.vector_store import VectorStore
        _brain = runtime.get("brain") or __import__("memory.brain", fromlist=["Brain"]).Brain()
        _model = runtime["model_adapter"]
        _planner = runtime["planner"]
        _coder = default_agents(_brain, _model).get("act")
        _critic = default_agents(_brain, _model).get("critique")
        _git = GitTools(repo_path=str(project_root))
        _mutator = MutatorAgent(project_root)
        _vec = VectorStore(_model, _brain)
        evo = EvolutionLoop(_planner, _coder, _critic, _brain, _vec, _git, _mutator)
        goal = args.goal or args.workflow_goal or "evolve and improve the AURA system"
        result = evo.run(goal)
        print(json.dumps(result, indent=2, default=str))
        return

    if args.status:
        _handle_status(goal_queue, goal_archive, orchestrator, as_json=args.json)
        return

    if args.add_goal:
        goal_queue.add(args.add_goal)
        log_json("INFO", "goal_added_from_cli", goal=args.add_goal)

    try:
        if args.goal:
            from core.explain import format_decision_log
            result = orchestrator.run_loop(args.goal, max_cycles=args.max_cycles or config.get("policy_max_cycles", config.get("max_cycles", 5)), dry_run=args.dry_run)
            history = result.get("history", [])
            if args.explain:
                print(format_decision_log(history))
            return

        if args.run_goals:
            run_goals_loop(args, goal_queue, loop, debugger_instance, planner_instance, goal_archive, project_root, decompose=args.decompose)
            return

        cli_interaction_loop(args, goal_queue, goal_archive, loop, debugger_instance, planner_instance, project_root)
    finally:
        if readline:
            try:
                readline.write_history_file(str(project_root / "memory" / ".aura_history"))
            except Exception:
                pass

if __name__ == "__main__":
    main()
