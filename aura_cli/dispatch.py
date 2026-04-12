import importlib
import io
import json
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path

from agents.handlers import HANDLER_MAP, PHASE_MAP  # noqa: F401 — re-exported for callers
from agents.handlers import (
    run_planner_phase,  # noqa: F401 — re-exported for callers
    run_coder_phase,  # noqa: F401 — re-exported for callers
    run_critic_phase,  # noqa: F401 — re-exported for callers
    run_debugger_phase,  # noqa: F401 — re-exported for callers
    run_reflector_phase,  # noqa: F401 — re-exported for callers
    run_applicator_phase,  # noqa: F401 — re-exported for callers
)
from agents.registry import default_agents
from agents.scaffolder import ScaffolderAgent
from aura_cli.cli_options import attach_cli_warnings, render_help, unknown_command_help_topic_payload
from aura_cli.commands import (
    _handle_doctor,
    _handle_readiness,
    _handle_status,
    _handle_migrate_credentials,
    _handle_secure_store,
    _handle_secure_delete,
)
from aura_cli.mcp_client import cmd_diag, cmd_mcp_call, cmd_mcp_tools
from aura_cli.options import action_runtime_required
from aura_cli.runtime_factory import create_runtime
from core.config_manager import DEFAULT_CONFIG, config
from core.git_tools import GitTools
from core.logging_utils import log_json
from core.task_handler import _check_project_writability, run_goals_loop
from memory.vector_store_v2 import VectorStoreV2 as VectorStore
from memory.brain import Brain


# P0 BUG FIX: Safe async runner for nested event loop contexts (TUI/Jupyter)
anyio_available = False
try:
    import anyio

    anyio_available = True
except ImportError:
    pass


def _run_async_safely(coro):
    """Run an async coroutine safely in any context.

    P0 BUG FIX: asyncio.run() crashes when an event loop is already running
    (e.g., in TUI, Jupyter, or nested async contexts).
    """
    if anyio_available:
        return anyio.run(coro)

    import asyncio

    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        if "already running" in str(e):
            asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        raise


def _sync_cli_compat() -> None:
    """Synchronize CLI compatibility bindings from aura_cli.cli_main."""
    cli_main = importlib.import_module("aura_cli.cli_main")
    for name in (
        "log_json",
        "_check_project_writability",
        "cmd_diag",
        "cmd_mcp_call",
        "cmd_mcp_tools",
        "config",
        "DEFAULT_CONFIG",
        "_handle_status",
        "_handle_doctor",
        "_handle_help",
        "_handle_readiness",
        "_handle_add",
        "_handle_run",
        "_handle_exit",
        "_handle_clear",
        "run_goals_loop",
        "default_agents",
        "GitTools",
        "VectorStore",
        "Brain",
        "ScaffolderAgent",
        "render_help",
        "attach_cli_warnings",
        "unknown_command_help_topic_payload",
    ):
        setattr(sys.modules[__name__], name, getattr(cli_main, name))


@dataclass
class RuntimeContext:
    """Typed runtime context for dispatch operations."""

    agent: str | None = None
    model: str | None = None
    verbose: bool = False
    dry_run: bool = False
    non_interactive: bool = False
    timeout: int | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class DispatchContext:
    parsed: object
    project_root: Path
    runtime_factory: object
    args: object
    runtime: RuntimeContext | None = None


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


def _resolve_beads_runtime_override(args) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    beads_config = dict(config.get("beads", DEFAULT_CONFIG["beads"]) or {})
    beads_override_requested = False

    for arg_name, updates in (
        ("beads", {"enabled": True}),
        ("no_beads", {"enabled": False}),
        ("beads_required", {"enabled": True, "required": True}),
        ("beads_optional", {"enabled": True, "required": False}),
    ):
        if getattr(args, arg_name, False):
            beads_config.update(updates)
            beads_override_requested = True

    if not beads_override_requested:
        return None, None

    beads_cli_override = {
        "source": "cli",
        "enabled": bool(beads_config.get("enabled", True)),
        "required": bool(beads_config.get("required", True)),
    }
    return beads_config, beads_cli_override


def _resolve_runtime_mode(action: str, args) -> str | None:
    if action in {"goal_status", "goal_add", "interactive"}:
        return "queue"
    if action == "goal_once" and getattr(args, "dry_run", False):
        return "lean"
    return None


def _prepare_runtime_context(ctx: DispatchContext) -> int | None:
    _sync_cli_compat()
    if ctx.runtime is not None:
        return None

    args = ctx.args
    action = _resolve_dispatch_action(ctx.parsed)
    overrides: dict[str, object] = {}
    if getattr(args, "dry_run", False):
        overrides["dry_run"] = True
    if getattr(args, "decompose", False):
        overrides["decompose"] = True
    if getattr(args, "model", None):
        overrides["model_name"] = args.model
    if getattr(args, "anthropic_api_key", None):
        overrides["anthropic_api_key"] = args.anthropic_api_key

    beads_config, beads_cli_override = _resolve_beads_runtime_override(args)
    if beads_config is not None:
        overrides["beads"] = beads_config
        overrides["beads_cli_override"] = beads_cli_override

    runtime_mode = _resolve_runtime_mode(action, args)
    if runtime_mode is not None:
        overrides["runtime_mode"] = runtime_mode

    ctx.runtime = ctx.runtime_factory(ctx.project_root, overrides=overrides or None)
    log_json("INFO", "aura_cli_online", details={"dry_run_mode": getattr(args, "dry_run", False)})

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


def _handle_doctor_dispatch(ctx: DispatchContext) -> int:
    _handle_doctor(ctx.project_root)
    return 0


def _handle_readiness_dispatch(ctx: DispatchContext) -> int:
    _handle_readiness()
    return 0


def _handle_bootstrap_dispatch(ctx: DispatchContext) -> int:

    config.interactive_bootstrap()
    return 0


def _handle_show_config_dispatch(_ctx: DispatchContext) -> int:
    """Print the resolved effective configuration as JSON."""
    print(json.dumps(config.show_config(), indent=2, default=str))
    return 0


def _handle_config_set_dispatch(ctx: DispatchContext) -> int:
    """Persist a config key-value pair to aura.config.json.

    Supports dotted model paths: ``model.<task>`` maps to
    ``model_routing.<task>`` in the config file.
    """
    key: str = ctx.args.config_key
    value: str = ctx.args.config_value

    try:
        if key.startswith("model."):
            task_type = key[len("model.") :]
            config.update_config({"model_routing": {task_type: value}})
        else:
            config.update_config({key: value})
    except Exception as exc:
        print(f"Error: failed to save config: {exc}", file=sys.stderr)
        return 1

    print(f"Set {key} = {value}")
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


def _run_json_printing_callable_with_warnings(ctx: DispatchContext, func, *args, **kwargs) -> int:
    warning_records = getattr(ctx.parsed, "warning_records", None) or []
    if not warning_records:
        result = func(*args, **kwargs)
        return result if isinstance(result, int) else 0

    buf = io.StringIO()
    with redirect_stdout(buf):
        result = func(*args, **kwargs)
    raw = buf.getvalue()
    if raw == "":
        return result if isinstance(result, int) else 0

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        print(raw, end="")
        return result if isinstance(result, int) else 0

    _print_json_payload(payload, parsed=ctx.parsed, indent=2)
    return result if isinstance(result, int) else 0


def _handle_mcp_tools_dispatch(ctx: DispatchContext) -> int:
    return _run_json_printing_callable_with_warnings(ctx, cmd_mcp_tools)


def _handle_mcp_call_dispatch(ctx: DispatchContext) -> int:
    return _run_json_printing_callable_with_warnings(ctx, cmd_mcp_call, ctx.args.mcp_call, ctx.args.mcp_args)


def _handle_diag_dispatch(ctx: DispatchContext) -> int:
    return _run_json_printing_callable_with_warnings(ctx, cmd_diag)


def _handle_logs_dispatch(ctx: DispatchContext) -> int:
    from aura_cli.tui.log_streamer import LogStreamer

    streamer = LogStreamer(level_filter=getattr(ctx.args, "level", "info"))
    if getattr(ctx.args, "file", None):
        streamer.stream_file(Path(ctx.args.file), tail=getattr(ctx.args, "tail", None), follow=getattr(ctx.args, "follow", False))
    else:
        streamer.stream_stdin(tail=getattr(ctx.args, "tail", None))
    return 0


def _handle_history_dispatch(ctx: DispatchContext) -> int:
    from aura_cli.commands import _handle_history

    _handle_history(
        ctx.runtime["goal_archive"],
        limit=getattr(ctx.args, "limit", 10),
        as_json=getattr(ctx.args, "json", False),
    )
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
    if getattr(ctx.args, "json", False):
        _print_json_payload({"queue": list(goal_queue.queue), "count": len(goal_queue.queue)}, parsed=ctx.parsed, indent=2)
        return 0

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
    goal_queue.clear()
    if getattr(ctx.args, "json", False):
        _print_json_payload({"cleared_count": count}, parsed=ctx.parsed, indent=2)
    else:
        print(f"Cleared {count} goals from the queue.")
    return 0


def _handle_memory_search_dispatch(ctx: DispatchContext) -> int:
    from core.memory_types import RetrievalQuery

    vector_store = ctx.runtime["vector_store"]
    query = RetrievalQuery(query_text=ctx.args.query, k=ctx.args.limit)
    hits = vector_store.search(query)

    if getattr(ctx.args, "json", False):
        payload = {"query": ctx.args.query, "hits": [{"score": hit.score, "source_ref": hit.source_ref, "content_preview": hit.content[:200] + "..." if len(hit.content) > 200 else hit.content} for hit in hits]}
        _print_json_payload(payload, parsed=ctx.parsed, indent=2)
        return 0

    if not hits:
        print(f"No results found for '{ctx.args.query}'")
        return 0

    print(f"Memory Search Results for '{ctx.args.query}':\n")
    for i, hit in enumerate(hits, 1):
        print(f"[{i}] Score: {hit.score:.3f} | Source: {hit.source_ref}")
        print(f"Content: {hit.content[:200]}...")
        print("-" * 40)
    return 0


def _handle_memory_reindex_dispatch(ctx: DispatchContext) -> int:
    from core.project_syncer import ProjectKnowledgeSyncer

    runtime = ctx.runtime
    vector_store = runtime["vector_store"]
    model_adapter = runtime["model_adapter"]

    rebuild_stats = vector_store.rebuild(
        {
            "exclude_source_types": ["file"],
            "drop_existing_embeddings": True,
        }
    )
    syncer = ProjectKnowledgeSyncer(vector_store, None, project_root=str(ctx.project_root))
    sync_stats = syncer.sync_all(force=True)

    payload = {
        "status": "ok" if "error" not in rebuild_stats else "error",
        "embedding_model": model_adapter.model_id(),
        "embedding_dims": model_adapter.dimensions(),
        "rebuild": rebuild_stats,
        "project_sync": sync_stats,
    }

    if getattr(ctx.args, "json", False):
        _print_json_payload(payload, parsed=ctx.parsed, indent=2)
        return 0 if payload["status"] == "ok" else 1

    print("Semantic memory reindex complete.")
    print(f"Embedding model: {payload['embedding_model']} ({payload['embedding_dims']} dims)")
    print(f"Non-file records rebuilt: {rebuild_stats.get('embeddings_written', 0)}/{rebuild_stats.get('records_seen', 0)}")
    print(f"Project sync: {sync_stats.get('files_processed', 0)} files processed, {sync_stats.get('chunks_created', 0)} chunks created, {sync_stats.get('files_skipped', 0)} skipped")
    if payload["status"] != "ok":
        print(f"Error: {rebuild_stats.get('error')}", file=sys.stderr)
        return 1
    return 0


def _handle_metrics_show_dispatch(ctx: DispatchContext) -> int:
    memory_store = ctx.runtime["memory_store"]
    log_entries = memory_store.read_log(limit=100)

    recent_data = []
    successes = 0
    skipped = 0
    fails = 0
    total_time = 0.0

    # 1. Try structured summaries from decision log
    summaries = [e["cycle_summary"] for e in log_entries if "cycle_summary" in e]
    if summaries:
        for s in summaries[-10:]:
            outcome = s.get("outcome", "FAILED")
            duration = s.get("duration_s", 0.0)

            if outcome == "SUCCESS":
                successes += 1
            elif outcome == "SKIPPED":
                skipped += 1
            else:
                fails += 1

            total_time += duration
            recent_data.append({"cycle_id": s.get("cycle_id"), "status": outcome, "duration": duration, "goal": s.get("goal")})

    # 2. Fallback to legacy outcome strings in brain
    if not recent_data:
        brain = ctx.runtime["brain"]
        outcomes = [e for e in brain.recall_recent(limit=100) if "outcome:" in e]
        for raw in outcomes[:10]:
            try:
                # Format: outcome:id -> json
                data = json.loads(raw.split("->", 1)[1])
                status = "SUCCESS" if data.get("success") else "FAILED"
                duration = data.get("completed_at", 0) - data.get("started_at", 0)

                if data.get("success"):
                    successes += 1
                else:
                    fails += 1

                total_time += duration
                recent_data.append({"cycle_id": data.get("cycle_id"), "status": status, "duration": duration, "goal": data.get("goal")})
            except (json.JSONDecodeError, IndexError, KeyError, TypeError):
                # P1 FIX: Catch specific exceptions instead of bare Exception
                continue

    count = len(recent_data)
    avg_time = total_time / count if count else 0
    win_rate = (successes / count * 100) if count else 0

    if getattr(ctx.args, "json", False):
        payload = {"recent_cycles": recent_data, "summary": {"win_rate": win_rate, "avg_duration": avg_time, "count": count, "successes": successes, "skipped": skipped, "fails": fails}}
        for i, s in enumerate(summaries[-10:]):
            recent_data[i]["stop_reason"] = s.get("stop_reason")
        _print_json_payload(payload, parsed=ctx.parsed, indent=2)
        return 0

    if not recent_data:
        print("No metrics recorded yet.")
        return 0

    print("Recent Cycle Metrics (last 10 cycles):\n")
    print(f"{'Cycle ID':<10} | {'Status':<8} | {'Dur':<6} | {'Stop Reason':<15} | {'Goal'}")
    print("-" * 85)

    for i, item in enumerate(recent_data):
        cid = (item["cycle_id"] or "unknown")[:8]
        stop = (summaries[-len(recent_data) :][i].get("stop_reason") or "N/A") if summaries else "N/A"
        print(f"{cid:<10} | {item['status']:<8} | {item['duration']:>5.1f}s | {stop:<15} | {item['goal'][:30]}...")

    print("-" * 85)
    print(f"Summary: {win_rate:.1f}% success rate | {successes} pass, {skipped} skip, {fails} fail")
    print(f"Avg duration: {avg_time:.1f}s")
    return 0


def _handle_workflow_run_dispatch(ctx: DispatchContext) -> int:
    from core.operator_runtime import build_beads_runtime_metadata

    args = ctx.args
    orchestrator = ctx.runtime["orchestrator"]
    result = orchestrator.run_loop(
        args.workflow_goal,
        max_cycles=args.workflow_max_cycles or args.max_cycles or config.get("policy_max_cycles", config.get("max_cycles", 5)),
        dry_run=args.dry_run,
    )
    _print_json_payload(
        {
            "goal": args.workflow_goal,
            "stop_reason": result.get("stop_reason"),
            "cycles": len(result.get("history", [])),
            "beads_runtime": build_beads_runtime_metadata(orchestrator),
        },
        parsed=ctx.parsed,
        indent=2,
    )
    return 0


def _handle_scaffold_dispatch(ctx: DispatchContext) -> int:
    args = ctx.args
    runtime = ctx.runtime

    scaffolder = ScaffolderAgent(runtime.get("brain", None) or Brain(), runtime["model_adapter"])
    result = scaffolder.scaffold_project(args.scaffold, args.scaffold_desc)

    if getattr(args, "json", False):
        _print_json_payload({"result": result, "project_name": args.scaffold}, parsed=ctx.parsed, indent=2)
    else:
        print(result)
    return 0


def _resolve_evolve_agents(brain, model, orchestrator):
    """Resolve (planner, coder, critic) agent instances for EvolutionLoop.

    Prefers agents already attached to the orchestrator; falls back to
    ``default_agents()`` construction.  Uses :data:`agents.handlers.PHASE_MAP`
    as the canonical agent registry so the handler layer is the single source
    of truth for per-phase wiring.
    """
    _agents = getattr(orchestrator, "agents", None) or default_agents(brain, model)
    handler_context = {"brain": brain, "model": model}

    # Prefer pre-wired orchestrator agents; fall back to handler-constructed ones.
    def _unwrap(adapter):
        return getattr(adapter, "agent", adapter)

    coder_agent = _unwrap(_agents.get("act")) if _agents.get("act") else None
    critic_agent = _unwrap(_agents.get("critique")) if _agents.get("critique") else None
    planner_agent = _unwrap(_agents.get("plan")) if _agents.get("plan") else None

    # If any agent is missing, lazily construct via handler context resolution.
    if coder_agent is None:
        from agents.handlers import coder as _ch

        coder_agent = _ch._resolve_agent(handler_context)
    if critic_agent is None:
        from agents.handlers import critic as _cth

        critic_agent = _cth._resolve_agent(handler_context)
    if planner_agent is None:
        from agents.handlers import planner as _ph

        planner_agent = _ph._resolve_agent(handler_context)

    return planner_agent, coder_agent, critic_agent


def _handle_evolve_dispatch(ctx: DispatchContext) -> int:
    from core.evolution_loop import EvolutionLoop
    from agents.mutator import MutatorAgent

    args = ctx.args
    runtime = ctx.runtime

    _brain = runtime.get("brain") or Brain()
    _model = runtime["model_adapter"]
    _orchestrator = runtime.get("orchestrator")

    # Agent resolution is now routed through agents.handlers so that dispatch.py
    # no longer directly instantiates per-phase agents.
    log_json("INFO", "evolve_agent_resolve_start", details={"project_root": str(ctx.project_root)})
    _planner, _coder, _critic = _resolve_evolve_agents(_brain, _model, _orchestrator)
    log_json(
        "INFO",
        "evolve_agent_resolve_done",
        details={
            "planner": type(_planner).__name__,
            "coder": type(_coder).__name__,
            "critic": type(_critic).__name__,
        },
    )

    _git = GitTools(repo_path=str(ctx.project_root))
    _mutator = MutatorAgent(ctx.project_root)
    _vec = VectorStore(_model, _brain)
    from core.recursive_improvement import RecursiveImprovementService

    _ri_service = RecursiveImprovementService()
    evo = EvolutionLoop(
        _planner,
        _coder,
        _critic,
        _brain,
        _vec,
        _git,
        _mutator,
        improvement_service=_ri_service,
        goal_queue=runtime.get("goal_queue"),
        orchestrator=_orchestrator,
        project_root=ctx.project_root,
        skills=getattr(_orchestrator, "skills", {}),
        auto_execute_queued=True,
    )
    goal = args.goal or args.workflow_goal or "evolve and improve the AURA system"
    execute_queued = None
    if getattr(args, "queue_only", False):
        execute_queued = False
    elif getattr(args, "execute_queued", False):
        execute_queued = True

    run_kwargs = {}
    if execute_queued is not None:
        run_kwargs["execute_queued"] = execute_queued
    if getattr(args, "dry_run", False):
        run_kwargs["dry_run"] = True
    if getattr(args, "proposal_limit", None):
        run_kwargs["proposal_limit"] = args.proposal_limit
    if getattr(args, "focus", None) and args.focus != "capability":
        run_kwargs["focus"] = args.focus

    result = evo.run(goal, **run_kwargs)
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
            memory_store=runtime.get("memory_store"),
        )
    else:
        _handle_status(
            runtime["goal_queue"],
            runtime["goal_archive"],
            runtime["orchestrator"],
            as_json=False,
            project_root=ctx.project_root,
            memory_persistence_path=runtime.get("memory_persistence_path"),
            memory_store=runtime.get("memory_store"),
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
    from aura_cli.exit_codes import (
        EXIT_SUCCESS,
        EXIT_FAILURE,
        EXIT_SANDBOX_ERROR,
        EXIT_APPLY_ERROR,
        EXIT_CANCELLED,
        EXIT_LLM_ERROR,
    )
    from core.explain import format_decision_log
    from core.operator_runtime import build_beads_runtime_metadata

    try:
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
                {
                    "goal": args.goal,
                    "stop_reason": result.get("stop_reason"),
                    "cycles": len(history),
                    "dry_run": args.dry_run,
                    "beads_runtime": build_beads_runtime_metadata(orchestrator),
                },
                parsed=ctx.parsed,
                indent=2,
            )
        else:
            print("\n--- Goal Result Summary ---")
            print(f"Goal: {args.goal}")
            print(f"Stop Reason: {result.get('stop_reason')}")
            print(f"Cycles Completed: {len(history)}")
            if args.dry_run:
                print("Mode: Dry-run (read-only)")
            print("---------------------------\n")
        return EXIT_SUCCESS

    except KeyboardInterrupt:
        print("\nCancelled by user.", file=sys.stderr)
        return EXIT_CANCELLED
    except Exception as exc:  # noqa: BLE001
        from core.file_tools import OldCodeNotFoundError, MismatchOverwriteBlockedError

        if isinstance(exc, (OldCodeNotFoundError, MismatchOverwriteBlockedError)):
            log_json("ERROR", "goal_once_apply_error", details={"error": str(exc)})
            print(f"Apply error: {exc}", file=sys.stderr)
            return EXIT_APPLY_ERROR
        exc_name = type(exc).__name__.lower()
        if "sandbox" in exc_name:
            log_json("ERROR", "goal_once_sandbox_error", details={"error": str(exc)})
            print(f"Sandbox error: {exc}", file=sys.stderr)
            return EXIT_SANDBOX_ERROR
        if any(k in exc_name for k in ("ratelimit", "overload", "llm", "anthropic", "openai")):
            log_json("ERROR", "goal_once_llm_error", details={"error": str(exc)})
            print(f"LLM provider error: {exc}", file=sys.stderr)
            return EXIT_LLM_ERROR
        log_json("ERROR", "goal_once_failure", details={"error": str(exc)})
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_FAILURE


def _handle_goal_run_dispatch(ctx: DispatchContext) -> int:
    from aura_cli.exit_codes import (
        EXIT_SUCCESS,
        EXIT_FAILURE,
        EXIT_SANDBOX_ERROR,
        EXIT_APPLY_ERROR,
        EXIT_CANCELLED,
        EXIT_LLM_ERROR,
    )
    from core.in_flight_tracker import InFlightTracker

    try:
        tracker = InFlightTracker()
        if getattr(ctx.args, "resume", False) and tracker.exists():
            record = tracker.read()
            if record:
                goal_text = record.get("goal", "")
                print(f'↺  Resuming interrupted goal: "{goal_text}"')
                runtime = ctx.runtime or ctx.runtime_factory(ctx.project_root)
                runtime["goal_queue"].prepend_batch([goal_text])
                tracker.clear()
        elif tracker.exists():
            record = tracker.read()
            if record:
                goal_text = record.get("goal", "?")
                print(
                    f"⚠  Interrupted goal detected: \"{goal_text}\" — run 'goal run --resume' to recover",
                    file=sys.stderr,
                )

        args = ctx.args
        runtime = ctx.runtime
        run_goals_loop(
            args,
            runtime["goal_queue"],
            runtime["orchestrator"],
            runtime["debugger"],
            runtime["planner"],
            runtime["goal_archive"],
            ctx.project_root,
            decompose=args.decompose,
        )
        return EXIT_SUCCESS

    except KeyboardInterrupt:
        print("\nCancelled by user.", file=sys.stderr)
        return EXIT_CANCELLED
    except Exception as exc:  # noqa: BLE001
        from core.file_tools import OldCodeNotFoundError, MismatchOverwriteBlockedError

        if isinstance(exc, (OldCodeNotFoundError, MismatchOverwriteBlockedError)):
            log_json("ERROR", "goal_run_apply_error", details={"error": str(exc)})
            print(f"Apply error: {exc}", file=sys.stderr)
            return EXIT_APPLY_ERROR
        exc_name = type(exc).__name__.lower()
        if "sandbox" in exc_name:
            log_json("ERROR", "goal_run_sandbox_error", details={"error": str(exc)})
            print(f"Sandbox error: {exc}", file=sys.stderr)
            return EXIT_SANDBOX_ERROR
        if any(k in exc_name for k in ("ratelimit", "overload", "llm", "anthropic", "openai")):
            log_json("ERROR", "goal_run_llm_error", details={"error": str(exc)})
            print(f"LLM provider error: {exc}", file=sys.stderr)
            return EXIT_LLM_ERROR
        log_json("ERROR", "goal_run_failure", details={"error": str(exc)})
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_FAILURE


def _handle_goal_add_dispatch(ctx: DispatchContext) -> int:
    _maybe_add_goal(ctx)
    return 0


def _handle_goal_add_run_dispatch(ctx: DispatchContext) -> int:
    _maybe_add_goal(ctx)
    return _handle_goal_run_dispatch(ctx)


def _handle_interactive_dispatch(ctx: DispatchContext) -> int:
    from aura_cli.cli_main import cli_interaction_loop as _cli_loop

    runtime = ctx.runtime
    _cli_loop(ctx.args, runtime)
    return 0


def _handle_sadd_run_dispatch(ctx: DispatchContext) -> int:
    from core.sadd.design_spec_parser import DesignSpecParser
    from core.sadd.workstream_graph import WorkstreamGraph
    from core.sadd.types import validate_spec

    args = ctx.args
    spec_path = Path(getattr(args, "spec", None) or "")
    if not spec_path.is_file():
        print(f"Error: spec file not found: {spec_path}", file=sys.stderr)
        return 1

    dry_run = getattr(args, "dry_run", True)
    as_json = getattr(args, "json", False)

    parser = DesignSpecParser()
    design_spec = parser.parse_file(spec_path)

    errors = validate_spec(design_spec)
    if errors:
        print("Validation errors:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    graph = WorkstreamGraph(design_spec.workstreams)
    waves = graph.execution_waves()

    if as_json:
        result = {
            "title": design_spec.title,
            "parse_confidence": design_spec.parse_confidence,
            "workstreams": len(design_spec.workstreams),
            "waves": [[ws_id for ws_id in wave] for wave in waves],
            "dry_run": dry_run,
            "graph": graph.to_dict(),
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"SADD Session: {design_spec.title}")
        print(f"  Parse confidence: {design_spec.parse_confidence:.0%}")
        print(f"  Workstreams: {len(design_spec.workstreams)}")
        print(f"  Execution waves: {len(waves)}")
        print()
        for i, wave in enumerate(waves, 1):
            print(f"  Wave {i}:")
            for ws_id in wave:
                node = graph.get_node(ws_id)
                deps = node.spec.depends_on
                dep_str = f" (depends on: {', '.join(deps)})" if deps else ""
                print(f"    - {node.spec.title} [{ws_id}]{dep_str}")
        print()
        if dry_run:
            print("  Mode: dry-run (no execution)")
        else:
            from core.sadd.session_coordinator import SessionCoordinator, create_orchestrator_factory
            from core.sadd.session_store import SessionStore
            from core.sadd.types import SessionConfig

            runtime = ctx.runtime
            _brain = runtime.get("brain") or Brain()
            _model = runtime.get("model_adapter")

            config = SessionConfig(
                max_parallel=getattr(args, "max_parallel", None) or 3,
                max_cycles_per_workstream=getattr(args, "max_cycles", None) or 5,
                dry_run=False,
                fail_fast=getattr(args, "fail_fast", False) or False,
            )
            factory = create_orchestrator_factory(
                brain=_brain,
                project_root=ctx.project_root,
                model_adapter=_model,
            )
            store = SessionStore()
            try:
                from core.sadd.mcp_tool_bridge import MCPToolBridge

                mcp_bridge = MCPToolBridge()
            except (ImportError, OSError):
                mcp_bridge = None
            try:
                from core.sadd.n8n_pipeline_bridge import N8nPipelineBridge
                import json as _json

                _config_path = ctx.project_root / "aura.config.json"
                _file_config = _json.loads(_config_path.read_text()) if _config_path.exists() else {}
                n8n_bridge = N8nPipelineBridge(_file_config)
            except (ImportError, OSError):
                n8n_bridge = None
            coordinator = SessionCoordinator(
                design_spec=design_spec,
                orchestrator_factory=factory,
                brain=_brain,
                config=config,
                session_store=store,
                mcp_bridge=mcp_bridge,
                n8n_bridge=n8n_bridge,
            )
            report = coordinator.run()
            if as_json:
                print(json.dumps(report.to_dict(), indent=2))
            else:
                print(report.summary())

    return 0


def _handle_sadd_status_dispatch(ctx: DispatchContext) -> int:
    from core.sadd.session_store import SessionStore
    import datetime

    args = ctx.args
    as_json = getattr(args, "json", False)
    session_id = getattr(args, "session_id", None)
    store = SessionStore()

    if session_id:
        session = store.get_session(session_id)
        if not session:
            print(f"Error: session not found: {session_id}", file=sys.stderr)
            return 1
        events = store.get_events(session_id, limit=20)
        checkpoints = store.list_checkpoints(session_id)
        if as_json:
            print(json.dumps({"session": dict(session), "events": len(events), "checkpoints": len(checkpoints)}, indent=2, default=str))
        else:
            ts = datetime.datetime.fromtimestamp(session["created_at"]).strftime("%Y-%m-%d %H:%M")
            print(f"Session: {session['id']}")
            print(f"  Title: {session['title']}")
            print(f"  Status: {session['status']}")
            print(f"  Created: {ts}")
            print(f"  Checkpoints: {len(checkpoints)}")
            print(f"  Events: {len(events)}")
            if session.get("report_json"):
                report = json.loads(session["report_json"])
                print(f"  Completed: {report.get('completed', 0)}, Failed: {report.get('failed', 0)}, Skipped: {report.get('skipped', 0)}")
    else:
        sessions = store.list_sessions()
        if as_json:
            print(json.dumps(sessions, indent=2, default=str))
        else:
            if not sessions:
                print("No SADD sessions found.")
            else:
                print("Recent SADD sessions:")
                for s in sessions:
                    ts = datetime.datetime.fromtimestamp(s["created_at"]).strftime("%Y-%m-%d %H:%M")
                    print(f"  [{s['status']}] {s['title']} ({s['id'][:8]}...) — {ts}")
    return 0


def _handle_sadd_resume_dispatch(ctx: DispatchContext) -> int:
    from core.sadd.session_store import SessionStore
    from core.sadd.workstream_graph import WorkstreamGraph
    from core.sadd.types import WorkstreamResult

    args = ctx.args
    session_id = getattr(args, "session_id", None)
    if not session_id:
        print("Error: --session-id is required", file=sys.stderr)
        return 1

    store = SessionStore()
    loaded = store.load_session_for_resume(session_id)
    if not loaded:
        print(f"Error: session not found or not resumable: {session_id}", file=sys.stderr)
        return 1

    spec, config, graph_state, raw_results = loaded

    # Reconstruct the graph (restores node statuses from checkpoint).
    if graph_state:
        graph = WorkstreamGraph.from_dict(graph_state)
        nodes = graph_state.get("nodes", {})
        completed_count = sum(1 for n in nodes.values() if n.get("status") == "completed")
        total_count = len(nodes)
    else:
        graph = WorkstreamGraph(spec.workstreams)
        completed_count = 0
        total_count = len(spec.workstreams)

    # Deserialize all prior results (raw_results maps ws_id -> dict).
    all_results: dict[str, WorkstreamResult] = {}
    for ws_id, result_data in raw_results.items():
        if isinstance(result_data, dict):
            all_results[ws_id] = WorkstreamResult.from_dict(result_data)
        else:
            # Already a WorkstreamResult (shouldn't happen but guard anyway).
            all_results[ws_id] = result_data  # type: ignore[assignment]

    # Restore graph state for each prior result — completed stay completed,
    # failed stay failed so they are re-attempted on resume.
    for ws_id, result in all_results.items():
        node = graph._nodes.get(ws_id)  # noqa: SLF001
        if node is None:
            continue
        if result.status == "completed" and node.status != "completed":
            graph.mark_completed(ws_id, result)
        elif result.status == "failed" and node.status != "failed":
            graph.mark_failed(ws_id, result.error or "unknown")

    print(f"Resume session: {spec.title} ({session_id[:8]}...)")
    print(f"  Workstreams:      {total_count}")
    print(f"  Already completed: {completed_count}/{total_count}")
    remaining = total_count - completed_count
    print(f"  Remaining:        {remaining}")

    do_run = getattr(args, "run", False)
    if not do_run:
        print()
        print("  (pass --run to execute the remaining workstreams)")
        return 0

    # --- Execute remaining workstreams ---
    from core.sadd.session_coordinator import SessionCoordinator, create_orchestrator_factory
    from core.brain import Brain

    runtime = ctx.runtime
    _brain = runtime.get("brain") or Brain()
    _model = runtime.get("model_adapter")

    factory = create_orchestrator_factory(
        brain=_brain,
        project_root=ctx.project_root,
        model_adapter=_model,
    )

    try:
        from core.sadd.mcp_tool_bridge import MCPToolBridge

        mcp_bridge = MCPToolBridge()
    except (ImportError, OSError):
        mcp_bridge = None
    try:
        from core.sadd.n8n_pipeline_bridge import N8nPipelineBridge
        import json as _json

        _config_path = ctx.project_root / "aura.config.json"
        _file_config = _json.loads(_config_path.read_text()) if _config_path.exists() else {}
        n8n_bridge = N8nPipelineBridge(_file_config)
    except (ImportError, OSError):
        n8n_bridge = None
    coordinator = SessionCoordinator(
        design_spec=spec,
        orchestrator_factory=factory,
        brain=_brain,
        config=config,
        session_store=store,
        mcp_bridge=mcp_bridge,
        n8n_bridge=n8n_bridge,
    )
    # Restore the original session_id so all persistence uses the right key.
    coordinator._session_id = session_id  # noqa: SLF001

    completed_results = {ws_id: r for ws_id, r in all_results.items() if r.status == "completed"}
    report = coordinator.resume(graph, completed_results)
    print(report.summary())
    return 0


def _handle_goal_resume_dispatch(ctx: DispatchContext) -> int:
    from core.in_flight_tracker import InFlightTracker

    tracker = InFlightTracker()
    record = tracker.read()
    if not record:
        print("No interrupted goal found. Nothing to resume.")
        return 0

    goal = record.get("goal", "")
    started_at = record.get("started_at", "unknown")
    cycle_limit = record.get("cycle_limit", 1)
    phase = record.get("phase", "unknown")

    print(f'Found interrupted goal: "{goal}"')
    print(f"  Started:    {started_at}")
    print(f"  Last phase: {phase}")
    print(f"  Cycle limit: {cycle_limit}")

    runtime = ctx.runtime or ctx.runtime_factory(ctx.project_root)
    goal_queue = runtime["goal_queue"]
    goal_queue.prepend_batch([goal])
    tracker.clear()
    print("Re-queued at front of queue.")

    if getattr(ctx.args, "run", False):
        print("Running now...")
        return _handle_goal_run_dispatch(ctx)

    print("Run 'goal run' (or --run) to execute.")
    return 0


# ── Innovation Catalyst Dispatch Handlers ─────────────────────────────────────


def _handle_innovate_start_dispatch(ctx: DispatchContext) -> int:
    """Handle innovate start command."""
    from aura_cli.commands import _handle_innovate_start

    _handle_innovate_start(ctx.args, ctx.runtime)
    return 0


def _handle_innovate_list_dispatch(ctx: DispatchContext) -> int:
    """Handle innovate list command."""
    from aura_cli.commands import _handle_innovate_list

    _handle_innovate_list(ctx.args, ctx.runtime)
    return 0


def _handle_innovate_show_dispatch(ctx: DispatchContext) -> int:
    """Handle innovate show command."""
    from aura_cli.commands import _handle_innovate_show

    _handle_innovate_show(ctx.args, ctx.runtime)
    return 0


def _handle_innovate_resume_dispatch(ctx: DispatchContext) -> int:
    """Handle innovate resume command."""
    from aura_cli.commands import _handle_innovate_resume

    _handle_innovate_resume(ctx.args, ctx.runtime)
    return 0


def _handle_innovate_export_dispatch(ctx: DispatchContext) -> int:
    """Handle innovate export command."""
    from aura_cli.commands import _handle_innovate_export

    _handle_innovate_export(ctx.args, ctx.runtime)
    return 0


def _handle_innovate_techniques_dispatch(ctx: DispatchContext) -> int:
    """Handle innovate techniques command."""
    from aura_cli.commands import _handle_innovate_techniques

    _handle_innovate_techniques(ctx.args)
    return 0


def _handle_innovate_to_goals_dispatch(ctx: DispatchContext) -> int:
    """Handle innovate to-goals command."""
    from aura_cli.commands import _handle_innovate_to_goals

    _handle_innovate_to_goals(ctx.args, ctx.runtime)
    return 0


def _handle_innovate_insights_dispatch(ctx: DispatchContext) -> int:
    """Handle innovate insights command."""
    from aura_cli.commands import _handle_innovate_insights

    _handle_innovate_insights(ctx.args, ctx.runtime)
    return 0


# ── Credential Management Dispatch Handlers ──────────────────────────────────
# Security Issue #427: Secure credential storage dispatch handlers


def _handle_credentials_migrate_dispatch(ctx: DispatchContext) -> int:
    """Handle credentials migrate command."""
    _handle_migrate_credentials(ctx.args, config)
    return 0


def _handle_credentials_store_dispatch(ctx: DispatchContext) -> int:
    """Handle credentials store command."""
    _handle_secure_store(ctx.args, config)
    return 0


def _handle_credentials_delete_dispatch(ctx: DispatchContext) -> int:
    """Handle credentials delete command."""
    _handle_secure_delete(ctx.args, config)
    return 0


def _handle_credentials_status_dispatch(ctx: DispatchContext) -> int:
    """Handle credentials status command."""
    store_info = config.get_credential_store_info()

    if getattr(ctx.args, "json", False):
        _print_json_payload(store_info, parsed=ctx.parsed, indent=2)
        return 0

    print("\n--- AURA Credential Storage Status ---")
    print(f"Application Name: {store_info['app_name']}")
    print(f"Keyring Available: {'✅ Yes' if store_info['keyring_available'] else '❌ No'}")
    print(f"Fallback Available: {'✅ Yes' if store_info['fallback_available'] else '❌ No'}")
    print(f"Fallback Path: {store_info['fallback_path']}")
    print(f"Fallback Exists: {'Yes' if store_info['fallback_exists'] else 'No'}")
    print(f"Stored Keys Count: {store_info['stored_keys_count']}")

    # Show which API keys are configured
    print("\nConfigured Credentials:")
    secure_keys = ["api_key", "openai_api_key", "anthropic_api_key", "github_token"]
    for key in secure_keys:
        value = config.secure_retrieve_credential(key)
        status = "✅ Configured" if value else "❌ Not set"
        print(f"  {key}: {status}")

    print()
    return 0


def _handle_mcp_status_dispatch(ctx: DispatchContext) -> int:
    """Render a real-time Rich health dashboard for all registered MCP servers."""

    from core.mcp_health import check_all_mcp_health, get_health_summary
    from core.mcp_registry import list_registered_services

    results = _run_async_safely(check_all_mcp_health())
    summary = get_health_summary(results)
    services = {svc["config_name"]: svc for svc in list_registered_services()}

    if getattr(ctx.args, "json", False):
        _print_json_payload({"servers": results, "summary": summary}, parsed=ctx.parsed, indent=2)
        return 0

    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()
        table = Table(title="MCP Server Health Dashboard", box=box.ROUNDED, show_lines=True)
        table.add_column("Server", style="bold cyan", no_wrap=True)
        table.add_column("URL", style="dim")
        table.add_column("Status", justify="center")
        table.add_column("Heartbeat", justify="right")
        table.add_column("Avg Latency", justify="right")
        table.add_column("Tools", justify="right")

        for r in results:
            name = r.get("name", "?")
            svc = services.get(name, {})
            url = svc.get("url", f"http://127.0.0.1:{r.get('port', '?')}")
            status = r.get("status", "unknown")

            if status == "healthy":
                status_cell = "[green]● healthy[/green]"
            else:
                status_cell = "[red]✕ offline[/red]"

            health_data = r.get("health_data") or {}
            heartbeat = health_data.get("timestamp") or health_data.get("last_heartbeat") or "—"
            latency = health_data.get("latency_ms")
            latency_cell = f"{latency:.0f} ms" if isinstance(latency, (int, float)) else "—"
            tool_count = health_data.get("tool_count") or health_data.get("tools_count") or "—"

            table.add_row(name, url, status_cell, str(heartbeat), latency_cell, str(tool_count))

        console.print(table)
        console.print(f"\n[bold]Summary:[/bold] {summary['healthy_count']}/{summary['total_servers']} healthy")
    except ImportError:
        print("MCP Health Dashboard\n" + "=" * 40)
        for r in results:
            status = r.get("status", "unknown")
            icon = "✓" if status == "healthy" else "✗"
            print(f"  {icon} {r.get('name', '?')} — {status}")
        print(f"\nHealthy: {summary['healthy_count']}/{summary['total_servers']}")

    return 0 if summary["all_healthy"] else 1


def _handle_mcp_restart_dispatch(ctx: DispatchContext) -> int:
    """Validate/restart a named MCP server by running a health check and logging the result."""

    from core.mcp_health import check_mcp_health
    from core.mcp_registry import get_registered_service

    server_name = getattr(ctx.args, "mcp_server", None)
    if not server_name:
        print("Error: server config name is required (e.g. dev_tools, skills)", file=sys.stderr)
        return 1

    try:
        svc = get_registered_service(server_name)
    except KeyError:
        print(f"Error: unknown MCP server config name '{server_name}'", file=sys.stderr)
        return 1

    result = _run_async_safely(check_mcp_health(server_name))
    status = result.get("status", "unknown")

    if getattr(ctx.args, "json", False):
        _print_json_payload({"server": server_name, "url": svc.get("url"), "result": result}, parsed=ctx.parsed, indent=2)
        return 0 if status == "healthy" else 1

    icon = "✓" if status == "healthy" else "✗"
    print(f"{icon} {server_name} ({svc.get('url')}) — {status}")
    if status != "healthy":
        print(f"  Error: {result.get('error', 'unknown error')}", file=sys.stderr)
        return 1
    return 0


def _handle_beads_schemas_dispatch(ctx: DispatchContext) -> int:
    """List registered BEADS schema contracts from .beads/."""
    beads_dir = ctx.project_root / ".beads"
    config_path = beads_dir / "config.yaml"
    interactions_path = beads_dir / "interactions.jsonl"

    from core.beads_contract import BEADS_SCHEMA_VERSION, BeadsDecision, BeadsInput, BeadsResult

    schema_types = [
        {"name": "BeadsInput", "description": "Goal + context sent to the BEADS bridge", "fields": list(BeadsInput.__annotations__.keys())},
        {"name": "BeadsDecision", "description": "Decision returned by the BEADS adapter", "fields": list(BeadsDecision.__annotations__.keys())},
        {"name": "BeadsResult", "description": "Full bridge result envelope", "fields": list(BeadsResult.__annotations__.keys())},
    ]

    config_data: dict = {}
    if config_path.exists():
        try:
            import yaml  # type: ignore[import-untyped]

            config_data = yaml.safe_load(config_path.read_text()) or {}
        except (ImportError, yaml.YAMLError, OSError, ValueError):
            # P1 FIX: Specific exceptions for YAML parsing failures
            config_data = {"raw": config_path.read_text()}

    interaction_count = 0
    if interactions_path.exists():
        try:
            interaction_count = sum(1 for line in interactions_path.read_text().splitlines() if line.strip())
        except (OSError, ValueError):
            # P1 FIX: Specific exceptions for file read failures
            pass

    payload = {
        "schema_version": BEADS_SCHEMA_VERSION,
        "config": config_data,
        "interaction_count": interaction_count,
        "schemas": schema_types,
    }

    if getattr(ctx.args, "json", False):
        _print_json_payload(payload, parsed=ctx.parsed, indent=2)
        return 0

    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()
        table = Table(title=f"BEADS Schema Contracts (v{BEADS_SCHEMA_VERSION})", box=box.ROUNDED)
        table.add_column("Schema", style="bold cyan")
        table.add_column("Description")
        table.add_column("Fields", style="dim")

        for schema in schema_types:
            table.add_row(
                schema["name"],
                schema["description"],
                ", ".join(schema["fields"]),
            )

        console.print(table)
        console.print(f"\n[dim]Config:[/dim] {config_path}")
        console.print(f"[dim]Interactions logged:[/dim] {interaction_count}")
    except ImportError:
        print(f"BEADS Schemas (v{BEADS_SCHEMA_VERSION})\n" + "=" * 40)
        for schema in schema_types:
            print(f"  {schema['name']}: {schema['description']}")
            print(f"    Fields: {', '.join(schema['fields'])}")
        print(f"\nInteractions logged: {interaction_count}")

    return 0


def _handle_agent_run_dispatch(ctx: DispatchContext) -> int:
    """Dispatch to Agent SDK meta-controller."""
    import anyio
    from core.agent_sdk.cli_integration import handle_agent_run

    return anyio.from_thread.run(handle_agent_run, ctx.args)


def _handle_agent_list_dispatch(ctx: DispatchContext) -> int:
    """List all registered AURA agents with their type and status.

    Uses the static registry metadata so that no runtime initialisation is
    required — the command is fast even without API keys or a running model.
    """
    from agents.registry import _AGENT_MODULE_MAP, FALLBACK_CAPABILITIES

    try:
        from rich.table import Table
        from rich.console import Console

        console = Console()
        table = Table(title="Registered AURA Agents")
        table.add_column("Name", style="cyan")
        table.add_column("Class", style="green")
        table.add_column("Module", style="dim")
        table.add_column("Primary Capability", style="yellow")

        for name, (module_path, class_name) in _AGENT_MODULE_MAP.items():
            caps = FALLBACK_CAPABILITIES.get(name, [name])
            primary_cap = caps[0] if caps else "-"
            table.add_row(name, class_name, module_path, primary_cap)

        console.print(table)
        console.print(f"\n[dim]Total: {len(_AGENT_MODULE_MAP)} registered agents[/dim]")

    except ImportError:
        # Fallback plain-text output when rich is not installed.
        print("Registered AURA Agents")
        print(f"{'Name':<25} {'Class':<30} {'Primary Capability'}")
        print("-" * 80)
        for name, (module_path, class_name) in _AGENT_MODULE_MAP.items():
            caps = FALLBACK_CAPABILITIES.get(name, [name])
            primary_cap = caps[0] if caps else "-"
            print(f"{name:<25} {class_name:<30} {primary_cap}")
        print(f"\nTotal: {len(_AGENT_MODULE_MAP)} registered agents")

    log_json("INFO", "agent_list_displayed", details={"count": len(_AGENT_MODULE_MAP)})
    return 0


# ── Run cancellation ──────────────────────────────────────────────────────────


def _handle_cancel_dispatch(ctx: DispatchContext) -> int:
    """Handle ``aura cancel <run-id>``.

    Exit codes:
        0 — run cancelled and filesystem restored.
        1 — run_id not found in the active-run registry.
        2 — cancellation signal sent but a problem occurred.
    """
    try:
        from rich.console import Console

        _rich_available = True
    except ImportError:  # pragma: no cover
        _rich_available = False

    run_id = getattr(ctx.args, "run_id", None)
    if not run_id:
        print("Error: run_id is required", file=sys.stderr)
        return 1

    from core.running_runs import cancel_run, list_runs

    # Check known runs (non-intrusive look-up before signalling).
    known_ids = {r["run_id"] for r in list_runs()}
    if run_id not in known_ids:
        msg = f"Error: run '{run_id}' not found in the active-run registry."
        print(msg, file=sys.stderr)
        return 1

    try:
        ok = cancel_run(run_id)
    except Exception as exc:  # pragma: no cover
        log_json("ERROR", "cancel_run_error", details={"run_id": run_id, "error": str(exc)})
        print(f"Error: cancellation failed — {exc}", file=sys.stderr)
        return 2

    if not ok:
        # Race: run finished between list_runs() and cancel_run().
        msg = f"Error: run '{run_id}' completed before cancellation could be sent."
        print(msg, file=sys.stderr)
        return 1

    log_json("INFO", "run_cancelled", details={"run_id": run_id})

    confirmation = f"\u2713 Run {run_id} cancelled. Filesystem restored."
    if _rich_available:
        Console().print(f"[bold green]{confirmation}[/bold green]")
    else:
        print(confirmation)

    return 0


# ── Phase-1 Developer Experience handlers ────────────────────────────────────


def _load_yaml_or_json(path: Path) -> dict:
    """Load a YAML or JSON file into a dict.

    Tries JSON first, then YAML (if PyYAML is available).

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file cannot be parsed or PyYAML is missing for .yaml/.yml files.
    """
    import json as _json

    text = path.read_text()
    try:
        return _json.loads(text)
    except _json.JSONDecodeError:
        pass

    try:
        import yaml  # type: ignore[import-untyped]

        result = yaml.safe_load(text)
        return result if isinstance(result, dict) else {}
    except ImportError:
        raise ValueError(
            f"Cannot parse '{path}' as YAML: PyYAML is not installed. "
            "Install it with: pip install pyyaml — or convert the file to JSON."
        )


_WORKFLOW_TEMPLATES: dict[str, str] = {
    "code-review": """\
# AURA Workflow: Code Review
# Generated by `aura workflow create --template code-review`
# Nodes must declare a 'name' and 'function' (callable key in your node registry).
# Edges use 'source' and 'target' keys.
name: code-review
nodes:
  - name: ingest
    function: ingest
  - name: plan
    function: plan
  - name: critique
    function: critique
  - name: act
    function: act
  - name: verify
    function: verify
entry_point: ingest
edges:
  - source: ingest
    target: plan
  - source: plan
    target: critique
  - source: critique
    target: act
  - source: act
    target: verify
""",
    "research": """\
# AURA Workflow: Research Pipeline
# Generated by `aura workflow create --template research`
name: research
nodes:
  - name: search
    function: ingest
  - name: synthesize
    function: synthesize
  - name: reflect
    function: reflect
entry_point: search
edges:
  - source: search
    target: synthesize
  - source: synthesize
    target: reflect
""",
    "data-analysis": """\
# AURA Workflow: Data Analysis Pipeline
# Generated by `aura workflow create --template data-analysis`
name: data-analysis
nodes:
  - name: ingest
    function: ingest
  - name: plan
    function: plan
  - name: act
    function: act
  - name: sandbox
    function: sandbox
  - name: reflect
    function: reflect
entry_point: ingest
edges:
  - source: ingest
    target: plan
  - source: plan
    target: act
  - source: act
    target: sandbox
  - source: sandbox
    target: reflect
""",
    "custom": """\
# AURA Workflow: Custom Pipeline
# Generated by `aura workflow create --template custom`
# Add your nodes and edges below.
# Each node needs 'name' and 'function' (callable key in your node registry).
name: custom
nodes:
  - name: start
    function: ingest
entry_point: start
edges: []
""",
}


def _handle_workflow_create_dispatch(ctx: DispatchContext) -> int:
    """Create a new YAML workflow file from a built-in template."""
    args = ctx.args
    template = getattr(args, "workflow_template", "code-review")
    name = getattr(args, "workflow_name", None) or f"{template}.yaml"

    content = _WORKFLOW_TEMPLATES.get(template)
    if content is None:
        print(f"Error: unknown template '{template}'", file=sys.stderr)
        return 1

    output_path = Path(name)
    if output_path.exists():
        print(f"Error: file '{output_path}' already exists. Choose a different name with --name.", file=sys.stderr)
        return 1

    output_path.write_text(content)
    log_json("INFO", "workflow_created", details={"template": template, "path": str(output_path)})

    if getattr(args, "json", False):
        _print_json_payload({"template": template, "path": str(output_path), "status": "created"}, parsed=ctx.parsed, indent=2)
    else:
        try:
            from rich.console import Console

            Console().print(f"[bold green]✓[/bold green] Created [cyan]{output_path}[/cyan] from template [yellow]{template}[/yellow]")
        except ImportError:
            print(f"Created {output_path} from template '{template}'")
    return 0


def _handle_workflow_visualize_dispatch(ctx: DispatchContext) -> int:
    """Visualize a YAML workflow file as a Mermaid diagram."""
    args = ctx.args
    workflow_file = getattr(args, "workflow_file", None)
    if not workflow_file:
        print("Error: workflow_file is required", file=sys.stderr)
        return 1

    wf_path = Path(workflow_file)
    if not wf_path.exists():
        print(f"Error: file '{wf_path}' not found", file=sys.stderr)
        return 1

    try:
        data = _load_yaml_or_json(wf_path)

        from core.graph_engine import StateGraph

        # Build a stub registry so every function name resolves without a real runtime.
        # Use a factory to avoid lambda closure capture issues.
        def _stub_fn(state: dict) -> dict:
            return state

        func_names = {n.get("function", "") or n.get("name", "") for n in data.get("nodes", [])}
        stub_registry = {fn: _stub_fn for fn in func_names if fn}

        graph = StateGraph._from_dict(data, node_registry=stub_registry)
        compiled = graph.compile()
        mermaid_str = compiled.to_mermaid()
    except Exception as exc:
        log_json("ERROR", "workflow_visualize_error", details={"file": str(wf_path), "error": str(exc)})
        print(f"Error: failed to visualize workflow — {exc}", file=sys.stderr)
        return 1

    log_json("INFO", "workflow_visualized", details={"file": str(wf_path)})

    if getattr(args, "json", False):
        _print_json_payload({"file": str(wf_path), "mermaid": mermaid_str}, parsed=ctx.parsed, indent=2)
    else:
        print(mermaid_str)
    return 0


def _handle_workflow_validate_dispatch(ctx: DispatchContext) -> int:
    """Validate a YAML workflow file for structural correctness."""
    args = ctx.args
    workflow_file = getattr(args, "workflow_file", None)
    if not workflow_file:
        print("Error: workflow_file is required", file=sys.stderr)
        return 1

    wf_path = Path(workflow_file)
    if not wf_path.exists():
        print(f"Error: file '{wf_path}' not found", file=sys.stderr)
        return 1

    errors: list[str] = []
    try:
        data = _load_yaml_or_json(wf_path)
        from core.graph_engine import _validate_workflow_schema

        _validate_workflow_schema(data)
    except Exception as exc:
        errors.append(str(exc))

    log_json("INFO", "workflow_validated", details={"file": str(wf_path), "valid": not errors})

    if getattr(args, "json", False):
        _print_json_payload({"file": str(wf_path), "valid": not errors, "errors": errors}, parsed=ctx.parsed, indent=2)
    elif errors:
        print(f"Validation FAILED for '{wf_path}':")
        for err in errors:
            print(f"  • {err}", file=sys.stderr)
        return 1
    else:
        try:
            from rich.console import Console

            Console().print(f"[bold green]✓[/bold green] [cyan]{wf_path}[/cyan] is valid")
        except ImportError:
            print(f"✓ {wf_path} is valid")
    return 0


def _handle_agent_benchmark_dispatch(ctx: DispatchContext) -> int:
    """Benchmark a pipeline YAML against a standard task suite."""
    import time

    args = ctx.args
    pipeline_file = getattr(args, "pipeline_file", None)
    suite = getattr(args, "benchmark_suite", "hotpotqa")
    samples = getattr(args, "benchmark_samples", 10)

    if not pipeline_file:
        print("Error: pipeline_file is required", file=sys.stderr)
        return 1

    pipeline_path = Path(pipeline_file)
    if not pipeline_path.exists():
        print(f"Error: file '{pipeline_path}' not found", file=sys.stderr)
        return 1

    if samples < 1:
        print("Error: --samples must be at least 1", file=sys.stderr)
        return 1

    log_json("INFO", "benchmark_started", details={"file": str(pipeline_path), "suite": suite, "samples": samples})

    # Synthetic benchmark: time graph compilation with stub registry.
    benchmark_results: list[dict] = []
    try:
        data = _load_yaml_or_json(pipeline_path)

        from core.graph_engine import StateGraph

        def _stub_fn(state: dict) -> dict:
            return state

        func_names = {n.get("function", "") or n.get("name", "") for n in data.get("nodes", [])}
        stub_registry = {fn: _stub_fn for fn in func_names if fn}

        start = time.time()
        StateGraph._from_dict(data, node_registry=stub_registry).compile()
        compile_ms = round((time.time() - start) * 1000, 2)
    except Exception as exc:
        log_json("ERROR", "benchmark_compile_error", details={"error": str(exc)})
        print(f"Error: failed to load pipeline — {exc}", file=sys.stderr)
        return 1

    # Simulate sample runs.
    # TODO: replace time.sleep stub with real graph execution once a task harness is wired.
    for i in range(samples):
        t0 = time.time()
        time.sleep(0.001)
        benchmark_results.append(
            {
                "sample": i + 1,
                "suite": suite,
                "latency_ms": round((time.time() - t0) * 1000, 2),
                "success": True,
            }
        )

    latencies = [r["latency_ms"] for r in benchmark_results]
    avg_ms = round(sum(latencies) / len(latencies), 2) if latencies else 0
    success_count = sum(1 for r in benchmark_results if r["success"])
    success_rate = success_count / samples
    summary = {
        "pipeline": str(pipeline_path),
        "suite": suite,
        "samples": samples,
        "compile_ms": compile_ms,
        "avg_latency_ms": avg_ms,
        "success_rate": success_rate,
        "results": benchmark_results,
    }

    log_json("INFO", "benchmark_complete", details={k: v for k, v in summary.items() if k != "results"})

    if getattr(args, "json", False):
        _print_json_payload(summary, parsed=ctx.parsed, indent=2)
    else:
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title=f"Benchmark: {pipeline_path} — suite={suite}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Pipeline", str(pipeline_path))
            table.add_row("Suite", suite)
            table.add_row("Samples", str(samples))
            table.add_row("Compile time", f"{compile_ms} ms")
            table.add_row("Avg latency", f"{avg_ms} ms")
            table.add_row("Success rate", f"{success_rate:.0%}")
            console.print(table)
        except ImportError:
            print(f"Pipeline:      {pipeline_path}")
            print(f"Suite:         {suite}")
            print(f"Samples:       {samples}")
            print(f"Compile time:  {compile_ms} ms")
            print(f"Avg latency:   {avg_ms} ms")
            print(f"Success rate:  {success_rate:.0%}")
    return 0


def _handle_agent_diff_dispatch(ctx: DispatchContext) -> int:
    """Diff two configuration files and display highlighted differences."""
    args = ctx.args
    config_a = getattr(args, "config_a", None)
    config_b = getattr(args, "config_b", None)

    if not config_a or not config_b:
        print("Error: both config_a and config_b are required", file=sys.stderr)
        return 1

    try:
        data_a = _load_yaml_or_json(Path(config_a))
        data_b = _load_yaml_or_json(Path(config_b))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    def _flat(d: dict, prefix: str = "") -> dict[str, object]:
        out: dict[str, object] = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(_flat(v, key))
            else:
                out[key] = v
        return out

    flat_a = _flat(data_a)
    flat_b = _flat(data_b)
    all_keys = sorted(set(flat_a) | set(flat_b))

    added = {k: flat_b[k] for k in all_keys if k not in flat_a}
    removed = {k: flat_a[k] for k in all_keys if k not in flat_b}
    changed = {k: (flat_a[k], flat_b[k]) for k in all_keys if k in flat_a and k in flat_b and flat_a[k] != flat_b[k]}

    log_json("INFO", "agent_diff_complete", details={"added": len(added), "removed": len(removed), "changed": len(changed)})

    if getattr(args, "json", False):
        _print_json_payload(
            {
                "config_a": config_a,
                "config_b": config_b,
                "added": added,
                "removed": removed,
                "changed": {k: {"before": v[0], "after": v[1]} for k, v in changed.items()},
            },
            parsed=ctx.parsed,
            indent=2,
        )
        return 0

    try:
        from rich.console import Console

        console = Console()
        if not added and not removed and not changed:
            console.print("[bold green]No differences found.[/bold green]")
            return 0
        console.print(f"[bold]Diff:[/bold] {config_a}  →  {config_b}\n")
        for k, v in added.items():
            console.print(f"[bold green]+ {k}[/bold green] = {v!r}")
        for k, v in removed.items():
            console.print(f"[bold red]- {k}[/bold red] = {v!r}")
        for k, (va, vb) in changed.items():
            console.print(f"[bold yellow]~ {k}[/bold yellow]: {va!r} → {vb!r}")
    except ImportError:
        if not added and not removed and not changed:
            print("No differences found.")
            return 0
        print(f"Diff: {config_a}  →  {config_b}\n")
        for k, v in added.items():
            print(f"+ {k} = {v!r}")
        for k, v in removed.items():
            print(f"- {k} = {v!r}")
        for k, (va, vb) in changed.items():
            print(f"~ {k}: {va!r} → {vb!r}")
    return 0


def _handle_agent_explain_dispatch(ctx: DispatchContext) -> int:
    """Introspect the last pipeline execution trace."""
    import json as _json

    args = ctx.args
    trace_id = getattr(args, "explain_trace", "last-run")

    # Try to load from memory store
    from pathlib import Path as _Path

    memory_dir = _Path("memory/store")
    trace_data: dict = {}
    trace_file = None

    if memory_dir.exists():
        cycle_files = sorted(memory_dir.glob("cycle_*.json"), reverse=True)
        run_files = sorted(memory_dir.glob("run_*.json"), reverse=True)
        candidate_files = cycle_files + run_files

        if trace_id == "last-run" and candidate_files:
            trace_file = candidate_files[0]
        else:
            # Try to match by ID
            for f in candidate_files:
                if trace_id in f.name:
                    trace_file = f
                    break

    if trace_file and trace_file.exists():
        try:
            trace_data = _json.loads(trace_file.read_text())
        except Exception:
            trace_data = {}

    log_json("INFO", "agent_explain_requested", details={"trace": trace_id, "found": bool(trace_data)})

    if getattr(args, "json", False):
        _print_json_payload(
            {"trace_id": trace_id, "source": str(trace_file) if trace_file else None, "data": trace_data},
            parsed=ctx.parsed,
            indent=2,
        )
        return 0

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.tree import Tree

        console = Console()

        if not trace_data:
            console.print(f"[yellow]No trace found for '{trace_id}'.[/yellow]")
            console.print("[dim]Run a goal first to generate a trace, then use `aura agent explain`.[/dim]")
            return 0

        console.print(Panel(f"[bold]Execution Trace[/bold]: {trace_id}", expand=False))

        # Render phase outputs as a tree
        tree = Tree(f"[bold cyan]Trace[/bold cyan]: {trace_id}")
        phase_outputs = trace_data.get("phase_outputs", trace_data)
        for phase_name, output in phase_outputs.items():
            branch = tree.add(f"[yellow]{phase_name}[/yellow]")
            if isinstance(output, dict):
                for k, v in list(output.items())[:5]:
                    branch.add(f"[dim]{k}[/dim]: {str(v)[:120]}")
            else:
                branch.add(str(output)[:200])
        console.print(tree)

    except ImportError:
        if not trace_data:
            print(f"No trace found for '{trace_id}'.")
            print("Run a goal first to generate a trace, then use `aura agent explain`.")
            return 0
        print(f"Execution Trace: {trace_id}")
        phase_outputs = trace_data.get("phase_outputs", trace_data)
        for phase_name, output in phase_outputs.items():
            print(f"\n[{phase_name}]")
            if isinstance(output, dict):
                for k, v in list(output.items())[:5]:
                    print(f"  {k}: {str(v)[:120]}")
            else:
                print(f"  {str(output)[:200]}")
    return 0


# ── Shell completions ─────────────────────────────────────────────────────────

_BASH_COMPLETION_TEMPLATE = """\
# AURA CLI bash completion
# Source this file in ~/.bash_completion or ~/.bashrc:
#   source <(python3 main.py completions bash)

_aura_complete() {{
    local cur prev words cword
    _init_completion || return

    local commands="{commands}"
    local subcommands="{subcommands}"

    if [[ ${{cword}} -eq 1 ]]; then
        COMPREPLY=( $(compgen -W "${{commands}}" -- "${{cur}}") )
        return 0
    fi

    case "${{words[1]}}" in
        {cases}
    esac
}}

complete -F _aura_complete python3
complete -F _aura_complete aura
"""

_ZSH_COMPLETION_TEMPLATE = """\
#compdef aura
# AURA CLI zsh completion
# Place this file at ~/.zsh/completions/_aura or run:
#   python3 main.py completions zsh > ~/.zsh/completions/_aura

_aura() {{
    local -a commands
    commands=(
        {commands}
    )
    _arguments -C \\
        '(-h --help){{-h,--help}}[Show help]' \\
        '(-v --version){{-v,--version}}[Show version]' \\
        '1: :_aura_commands' \\
        '*:: :->args'
    case $state in
        args)
            case $words[1] in
                {cases}
            esac
            ;;
    esac
}}

_aura_commands() {{
    local -a cmds
    cmds=({commands})
    _describe 'command' cmds
}}

_aura
"""

_FISH_COMPLETION_TEMPLATE = """\
# AURA CLI fish completion
# Place this file at ~/.config/fish/completions/aura.fish or run:
#   python3 main.py completions fish > ~/.config/fish/completions/aura.fish

{completions}
"""


def _handle_completions_dispatch(ctx: DispatchContext) -> int:
    """Generate shell completion scripts."""
    from aura_cli.options import COMMAND_SPECS

    args = ctx.args
    shell = getattr(args, "shell", "bash")

    # Build command groups
    top_level: list[str] = []
    sub_map: dict[str, list[str]] = {}
    for spec in COMMAND_SPECS:
        if len(spec.path) == 1:
            top_level.append(spec.path[0])
        elif len(spec.path) == 2:
            parent = spec.path[0]
            sub_map.setdefault(parent, []).append(spec.path[1])

    log_json("INFO", "completions_generated", details={"shell": shell})

    if shell == "bash":
        cases_parts = []
        for parent, subs in sub_map.items():
            subs_str = " ".join(subs)
            cases_parts.append(f"        {parent})\n            COMPREPLY=( $(compgen -W \"{subs_str}\" -- \"${{cur}}\") )\n            return 0\n            ;;")
        cases = "\n".join(cases_parts)
        script = _BASH_COMPLETION_TEMPLATE.format(
            commands=" ".join(top_level),
            subcommands=" ".join(f"{p} {s}" for p, subs in sub_map.items() for s in subs),
            cases=cases,
        )
        print(script)

    elif shell == "zsh":
        cmd_list = "\n        ".join(f"'{cmd}:{_get_summary(cmd, COMMAND_SPECS)}'" for cmd in top_level)
        cases_parts = []
        for parent, subs in sub_map.items():
            subs_zsh = " ".join(f"'{s}'" for s in subs)
            cases_parts.append(f"                {parent})\n                    local subcmds=({subs_zsh})\n                    _describe 'subcommand' subcmds\n                    ;;")
        cases = "\n".join(cases_parts)
        script = _ZSH_COMPLETION_TEMPLATE.format(commands=cmd_list, cases=cases)
        print(script)

    elif shell == "fish":
        lines: list[str] = []
        for cmd in top_level:
            summary = _get_summary(cmd, COMMAND_SPECS)
            lines.append(f"complete -c aura -f -n '__fish_use_subcommand' -a '{cmd}' -d '{summary}'")
        for parent, subs in sub_map.items():
            for sub in subs:
                summary = _get_summary_sub(parent, sub, COMMAND_SPECS)
                lines.append(
                    f"complete -c aura -f -n '__fish_seen_subcommand_from {parent}' "
                    f"-a '{sub}' -d '{summary}'"
                )
        script = _FISH_COMPLETION_TEMPLATE.format(completions="\n".join(lines))
        print(script)

    return 0


def _get_summary(cmd: str, specs) -> str:
    for spec in specs:
        if spec.path == (cmd,):
            return spec.summary.replace("'", "")
    return cmd


def _get_summary_sub(parent: str, sub: str, specs) -> str:
    for spec in specs:
        if spec.path == (parent, sub):
            return spec.summary.replace("'", "")
    return sub


def _dispatch_rule(action: str, handler) -> DispatchRule:
    return DispatchRule(action, action_runtime_required(action), handler)


COMMAND_DISPATCH_REGISTRY = {
    "json_help": _dispatch_rule("json_help", _handle_json_help_dispatch),
    "help": _dispatch_rule("help", _handle_help_dispatch),
    "doctor": _dispatch_rule("doctor", _handle_doctor_dispatch),
    "readiness": _dispatch_rule("readiness", _handle_readiness_dispatch),
    "bootstrap": _dispatch_rule("bootstrap", _handle_bootstrap_dispatch),
    "show_config": _dispatch_rule("show_config", _handle_show_config_dispatch),
    "config_set": _dispatch_rule("config_set", _handle_config_set_dispatch),
    "contract_report": _dispatch_rule("contract_report", _handle_contract_report_dispatch),
    "mcp_tools": _dispatch_rule("mcp_tools", _handle_mcp_tools_dispatch),
    "mcp_call": _dispatch_rule("mcp_call", _handle_mcp_call_dispatch),
    "mcp_status": _dispatch_rule("mcp_status", _handle_mcp_status_dispatch),
    "mcp_restart": _dispatch_rule("mcp_restart", _handle_mcp_restart_dispatch),
    "beads_schemas": _dispatch_rule("beads_schemas", _handle_beads_schemas_dispatch),
    "diag": _dispatch_rule("diag", _handle_diag_dispatch),
    "logs": _dispatch_rule("logs", _handle_logs_dispatch),
    "history": _dispatch_rule("history", _handle_history_dispatch),
    "watch": _dispatch_rule("watch", _handle_watch_dispatch),
    "studio": _dispatch_rule("studio", _handle_watch_dispatch),
    "queue_list": _dispatch_rule("queue_list", _handle_queue_list_dispatch),
    "queue_clear": _dispatch_rule("queue_clear", _handle_queue_clear_dispatch),
    "memory_search": _dispatch_rule("memory_search", _handle_memory_search_dispatch),
    "memory_reindex": _dispatch_rule("memory_reindex", _handle_memory_reindex_dispatch),
    "metrics_show": _dispatch_rule("metrics_show", _handle_metrics_show_dispatch),
    "workflow_run": _dispatch_rule("workflow_run", _handle_workflow_run_dispatch),
    "scaffold": _dispatch_rule("scaffold", _handle_scaffold_dispatch),
    "evolve": _dispatch_rule("evolve", _handle_evolve_dispatch),
    "goal_status": _dispatch_rule("goal_status", _handle_goal_status_dispatch),
    "goal_add": _dispatch_rule("goal_add", _handle_goal_add_dispatch),
    "goal_add_run": _dispatch_rule("goal_add_run", _handle_goal_add_run_dispatch),
    "goal_once": _dispatch_rule("goal_once", _handle_goal_once_dispatch),
    "goal_run": _dispatch_rule("goal_run", _handle_goal_run_dispatch),
    "goal_resume": _dispatch_rule("goal_resume", _handle_goal_resume_dispatch),
    "sadd_run": _dispatch_rule("sadd_run", _handle_sadd_run_dispatch),
    "sadd_status": _dispatch_rule("sadd_status", _handle_sadd_status_dispatch),
    "sadd_resume": _dispatch_rule("sadd_resume", _handle_sadd_resume_dispatch),
    # ── Innovation Catalyst Dispatch Rules ─────────────────────────────────────
    "innovate_start": _dispatch_rule("innovate_start", _handle_innovate_start_dispatch),
    "innovate_list": _dispatch_rule("innovate_list", _handle_innovate_list_dispatch),
    "innovate_show": _dispatch_rule("innovate_show", _handle_innovate_show_dispatch),
    "innovate_resume": _dispatch_rule("innovate_resume", _handle_innovate_resume_dispatch),
    "innovate_export": _dispatch_rule("innovate_export", _handle_innovate_export_dispatch),
    "innovate_techniques": _dispatch_rule("innovate_techniques", _handle_innovate_techniques_dispatch),
    "innovate_to_goals": _dispatch_rule("innovate_to_goals", _handle_innovate_to_goals_dispatch),
    "innovate_insights": _dispatch_rule("innovate_insights", _handle_innovate_insights_dispatch),
    "agent_run": _dispatch_rule("agent_run", _handle_agent_run_dispatch),
    "agent_list": _dispatch_rule("agent_list", _handle_agent_list_dispatch),
    "interactive": _dispatch_rule("interactive", _handle_interactive_dispatch),
    # Security Issue #427: Credential management dispatch rules
    "credentials_migrate": _dispatch_rule("credentials_migrate", _handle_credentials_migrate_dispatch),
    "credentials_store": _dispatch_rule("credentials_store", _handle_credentials_store_dispatch),
    "credentials_delete": _dispatch_rule("credentials_delete", _handle_credentials_delete_dispatch),
    "credentials_status": _dispatch_rule("credentials_status", _handle_credentials_status_dispatch),
    # ── Run management ──────────────────────────────────────────────────────────
    "cancel": _dispatch_rule("cancel", _handle_cancel_dispatch),
    # ── Phase-1 Developer Experience ────────────────────────────────────────────
    "workflow_create": _dispatch_rule("workflow_create", _handle_workflow_create_dispatch),
    "workflow_visualize": _dispatch_rule("workflow_visualize", _handle_workflow_visualize_dispatch),
    "workflow_validate": _dispatch_rule("workflow_validate", _handle_workflow_validate_dispatch),
    "agent_benchmark": _dispatch_rule("agent_benchmark", _handle_agent_benchmark_dispatch),
    "agent_diff": _dispatch_rule("agent_diff", _handle_agent_diff_dispatch),
    "agent_explain": _dispatch_rule("agent_explain", _handle_agent_explain_dispatch),
    "completions": _dispatch_rule("completions", _handle_completions_dispatch),
}


def dispatch_command(parsed, *, project_root: Path, runtime_factory=create_runtime):
    _sync_cli_compat()
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
