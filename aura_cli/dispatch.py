import importlib
import io
import json
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

from agents.registry import default_agents
from agents.scaffolder import ScaffolderAgent
from aura_cli.cli_options import attach_cli_warnings, render_help, unknown_command_help_topic_payload
from aura_cli.commands import (
    _handle_add,
    _handle_clear,
    _handle_doctor,
    _handle_exit,
    _handle_help,
    _handle_readiness,
    _handle_run,
    _handle_status,
)
from aura_cli.mcp_client import cmd_diag, cmd_mcp_call, cmd_mcp_tools
from aura_cli.options import action_runtime_required
from aura_cli.runtime_factory import create_runtime
from core.config_manager import DEFAULT_CONFIG, config
from core.git_tools import GitTools
from core.logging_utils import log_json
from core.task_handler import _check_project_writability, run_goals_loop
from core.vector_store import VectorStore
from memory.brain import Brain


def _sync_cli_compat() -> None:
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
            except Exception:
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


def _handle_evolve_dispatch(ctx: DispatchContext) -> int:
    from core.evolution_loop import EvolutionLoop
    from agents.mutator import MutatorAgent

    args = ctx.args
    runtime = ctx.runtime

    _brain = runtime.get("brain") or Brain()
    _model = runtime["model_adapter"]
    _orchestrator = runtime.get("orchestrator")
    _agents = getattr(_orchestrator, "agents", None) or default_agents(_brain, _model)
    _coder_adapter = _agents.get("act")
    _critic_adapter = _agents.get("critique")
    _planner_adapter = _agents.get("plan")
    _coder = getattr(_coder_adapter, "agent", _coder_adapter)
    _critic = getattr(_critic_adapter, "agent", _critic_adapter)
    _planner = getattr(_planner_adapter, "agent", _planner_adapter)
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
    from core.explain import format_decision_log
    from core.operator_runtime import build_beads_runtime_metadata

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
    return 0


def _handle_goal_run_dispatch(ctx: DispatchContext) -> int:
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
                max_parallel=getattr(args, "max_parallel", 3),
                max_cycles_per_workstream=getattr(args, "max_cycles", 5),
                dry_run=False,
                fail_fast=getattr(args, "fail_fast", False),
            )
            factory = create_orchestrator_factory(
                brain=_brain, project_root=ctx.project_root, model_adapter=_model,
            )
            store = SessionStore()
            coordinator = SessionCoordinator(
                design_spec=design_spec,
                orchestrator_factory=factory,
                brain=_brain,
                config=config,
                session_store=store,
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

    spec, config, graph_state, results = loaded
    print(f"Resume session: {spec.title} ({session_id[:8]}...)")
    print(f"  Workstreams: {len(spec.workstreams)}")
    if graph_state:
        nodes = graph_state.get("nodes", {})
        completed = sum(1 for n in nodes.values() if n.get("status") == "completed")
        print(f"  Already completed: {completed}/{len(nodes)}")
    print("  Note: Full resume execution not yet implemented (R2 preview)")
    return 0


def _handle_goal_resume_dispatch(ctx: DispatchContext) -> int:
    from core.in_flight_tracker import InFlightTracker
    from core.goal_queue import GoalQueue

    tracker = InFlightTracker()
    record = tracker.read()
    if not record:
        print("No interrupted goal found. Nothing to resume.")
        return 0

    goal = record.get("goal", "")
    started_at = record.get("started_at", "unknown")
    cycle_limit = record.get("cycle_limit", 1)
    phase = record.get("phase", "unknown")

    print(f"Found interrupted goal: \"{goal}\"")
    print(f"  Started:    {started_at}")
    print(f"  Last phase: {phase}")
    print(f"  Cycle limit: {cycle_limit}")

    runtime = ctx.runtime or ctx.runtime_factory(ctx.project_root)
    goal_queue = runtime["goal_queue"]
    goal_queue.prepend_batch([goal])
    tracker.clear()
    print(f"Re-queued at front of queue.")

    if getattr(ctx.args, "run", False):
        print("Running now...")
        return _handle_goal_run_dispatch(ctx)

    print("Run 'goal run' (or --run) to execute.")
    return 0


def _dispatch_rule(action: str, handler) -> DispatchRule:
    return DispatchRule(action, action_runtime_required(action), handler)


COMMAND_DISPATCH_REGISTRY = {
    "json_help": _dispatch_rule("json_help", _handle_json_help_dispatch),
    "help": _dispatch_rule("help", _handle_help_dispatch),
    "doctor": _dispatch_rule("doctor", _handle_doctor_dispatch),
    "readiness": _dispatch_rule("readiness", _handle_readiness_dispatch),
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
    "interactive": _dispatch_rule("interactive", _handle_interactive_dispatch),
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


