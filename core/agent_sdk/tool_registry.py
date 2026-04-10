# core/agent_sdk/tool_registry.py
"""Custom MCP tools wrapping AURA infrastructure for Agent SDK.

Each tool bridges a Claude Agent SDK tool call to an existing AURA
subsystem (agents, skills, MCP servers, memory, workflows).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List


@dataclass
class AuraTool:
    """A custom tool definition for the Agent SDK MCP server."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[..., Any] = field(repr=False)


# ---------------------------------------------------------------------------
# Tool Handlers
# ---------------------------------------------------------------------------


def _handle_analyze_goal(
    args: Dict[str, Any],
    *,
    project_root: Path,
    brain: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """Classify goal and gather context."""
    from core.agent_sdk.context_builder import ContextBuilder

    builder = ContextBuilder(project_root=project_root, brain=brain)
    return builder.build(goal=args["goal"])


def _handle_dispatch_skills(
    args: Dict[str, Any],
    *,
    project_root: Path,
    brain: Any = None,
    model_adapter: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """Run adaptive skills matched to goal type.

    dispatch_skills(goal_type, skills, project_root) expects skills
    as a Dict[str, SkillBase] (name → instance), not a list.
    """
    goal_type = args.get("goal_type", "default")
    root = Path(args.get("project_root", str(project_root)))

    try:
        from core.skill_dispatcher import dispatch_skills, SKILL_MAP
        from agents.skills.registry import all_skills

        skill_names = SKILL_MAP.get(goal_type, SKILL_MAP.get("default", []))
        available = all_skills(brain=brain, model=model_adapter)
        # Filter to a dict of only the relevant skills
        skills = {n: available[n] for n in skill_names if n in available}

        if not skills:
            return {"skills_run": [], "results": {}, "note": "No matching skills loaded"}

        results = dispatch_skills(goal_type, skills, str(root))
        return {
            "skills_run": list(skills.keys()),
            "results": results,
        }
    except Exception as exc:
        return {
            "skills_run": [],
            "results": {},
            "error": str(exc),
        }


def _handle_create_plan(
    args: Dict[str, Any],
    *,
    brain: Any = None,
    model_adapter: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """Generate implementation plan via PlannerAgent."""
    try:
        from agents.planner import PlannerAgent

        agent = PlannerAgent(brain=brain, model=model_adapter)
        result = agent.plan(
            goal=args["goal"],
            memory_snapshot=args.get("memory_snapshot", ""),
            similar_past_problems=args.get("similar_past_problems", ""),
            known_weaknesses=args.get("known_weaknesses", ""),
        )
        if isinstance(result, list):
            return {"steps": result, "confidence": 0.7}
        return result
    except Exception as exc:
        return {"steps": [], "error": str(exc)}


def _handle_critique_plan(
    args: Dict[str, Any],
    *,
    brain: Any = None,
    model_adapter: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """Adversarial plan review via CriticAgent.

    CriticAgent.critique_plan() calls self.brain.recall_with_budget()
    internally, so brain must not be None.
    """
    if brain is None:
        return {"issues": [], "error": "CriticAgent requires a brain instance"}
    try:
        from agents.critic import CriticAgent

        agent = CriticAgent(brain=brain, model=model_adapter)
        result = agent.critique_plan(
            task=args["goal"],
            plan=args["plan"],
        )
        return result if isinstance(result, dict) else {"issues": [], "raw": str(result)}
    except Exception as exc:
        return {"issues": [], "error": str(exc)}


def _handle_generate_code(
    args: Dict[str, Any],
    *,
    brain: Any = None,
    model_adapter: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """Generate code changes via CoderAgent.

    CoderAgent.implement(task) takes a single positional string.
    Combine task + plan + context into one prompt string.
    """
    try:
        from agents.coder import CoderAgent

        agent = CoderAgent(brain=brain, model=model_adapter)
        # Build a single task string — implement() takes one positional arg
        task_parts = [args["task"]]
        if args.get("plan"):
            task_parts.append(f"\n\nPlan:\n{args['plan']}")
        if args.get("context"):
            task_parts.append(f"\n\nContext:\n{args['context']}")
        result = agent.implement("\n".join(task_parts))
        if isinstance(result, str):
            return {"code": result, "confidence": 0.7}
        return result
    except Exception as exc:
        return {"code": "", "error": str(exc)}


def _handle_run_sandbox(
    args: Dict[str, Any],
    *,
    project_root: Path,
    brain: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """Execute code in sandboxed subprocess.

    SandboxAgent.__init__(brain, timeout, python_exec) requires brain
    as a positional argument.
    """
    try:
        from agents.sandbox import SandboxAgent

        agent = SandboxAgent(brain=brain, timeout=args.get("timeout", 30))
        # SandboxAgent is NOT an Agent subclass — use run_code(), not run()
        # Returns a SandboxResult dataclass with passed, exit_code, stdout, stderr
        sandbox_result = agent.run_code(args["code"])
        return {
            "success": sandbox_result.passed,
            "exit_code": sandbox_result.exit_code,
            "stdout": sandbox_result.stdout,
            "stderr": sandbox_result.stderr,
        }
    except Exception as exc:
        return {"success": False, "stdout": "", "stderr": str(exc), "exit_code": 1}


def _handle_apply_changes(
    args: Dict[str, Any],
    *,
    project_root: Path,
    **_: Any,
) -> Dict[str, Any]:
    """Apply file changes atomically."""
    try:
        from core.file_tools import apply_change_with_explicit_overwrite_policy

        result = apply_change_with_explicit_overwrite_policy(
            project_root=project_root,
            file_path=args["file_path"],
            new_code=args["new_code"],
            old_code=args.get("old_code", ""),
            overwrite_file=args.get("overwrite_file", False),
        )
        return {"applied": True, "result": str(result)}
    except Exception as exc:
        return {"applied": False, "error": str(exc)}


def _handle_verify_changes(
    args: Dict[str, Any],
    *,
    project_root: Path = None,
    **_: Any,
) -> Dict[str, Any]:
    """Run tests and linters via VerifierAgent.

    VerifierAgent.__init__(timeout) takes only an optional timeout.
    No brain or model_adapter params.
    """
    try:
        from agents.verifier import VerifierAgent

        agent = VerifierAgent(timeout=120)
        # VerifierAgent.run() expects change_set with nested changes list,
        # not a flat changed_files key
        changed_files = args.get("changed_files", [])
        result = agent.run(
            {
                "project_root": str(project_root or "."),
                "change_set": {"changes": [{"file_path": f} for f in changed_files]},
            }
        )
        return result if isinstance(result, dict) else {"passed": False, "raw": str(result)}
    except Exception as exc:
        return {"passed": False, "error": str(exc)}


def _handle_reflect(
    args: Dict[str, Any],
    **_: Any,
) -> Dict[str, Any]:
    """Reflect on outcome via ReflectorAgent.

    ReflectorAgent extends Agent (ABC) with no __init__ params.
    Its run(input_data) accepts a dict with verification + skill_context.
    """
    try:
        from agents.reflector import ReflectorAgent

        agent = ReflectorAgent()
        result = agent.run(
            {
                "goal": args["goal"],
                "verification": args.get("verification", {}),
                "skill_context": args.get("skill_context", {}),
            }
        )
        return result if isinstance(result, dict) else {"summary": str(result)}
    except Exception as exc:
        return {"summary": "", "error": str(exc)}


def _handle_search_memory(
    args: Dict[str, Any],
    *,
    brain: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """Query semantic memory.

    Brain.recall_with_budget(max_tokens, tier) returns List[str] of
    memories truncated to fit within max_tokens. It does NOT accept
    a query string — it returns the most recent memories by budget.
    """
    if brain is None:
        return {"results": [], "note": "No brain configured"}
    try:
        max_tokens = args.get("max_tokens", 4000)
        results = brain.recall_with_budget(max_tokens=max_tokens)
        # Filter results by query keyword if provided
        query = args.get("query", "").lower()
        if query:
            results = [r for r in results if query in r.lower()] or results
        return {"results": results}
    except Exception as exc:
        return {"results": [], "error": str(exc)}


def _handle_store_memory(
    args: Dict[str, Any],
    *,
    brain: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """Store a memory in brain.

    Brain.remember(data) takes a single argument — either a string
    or a dict (dict is JSON-serialized internally).
    """
    if brain is None:
        return {"stored": False, "note": "No brain configured"}
    try:
        content = args["content"]
        tags = args.get("tags", [])
        # Pack tags into a dict if provided, else store raw string
        if tags:
            brain.remember({"content": content, "tags": tags})
        else:
            brain.remember(content)
        return {"stored": True}
    except Exception as exc:
        return {"stored": False, "error": str(exc)}


def _handle_manage_goals(
    args: Dict[str, Any],
    *,
    goal_queue: Any = None,
    goal_archive: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """Manage the goal queue (add/list/complete/archive)."""
    action = args.get("action", "list")
    try:
        if goal_queue is None:
            from core.goal_queue import GoalQueue

            goal_queue = GoalQueue()

        if action == "add":
            goal_queue.add(args["goal"])
            return {"action": "add", "success": True}
        elif action == "list":
            return {"action": "list", "goals": list(goal_queue._queue)}
        elif action == "complete":
            goal_queue.complete(args["goal"])
            return {"action": "complete", "success": True}
        elif action == "next":
            g = goal_queue.next()
            return {"action": "next", "goal": g}
        else:
            return {"error": f"Unknown action: {action}"}
    except Exception as exc:
        return {"error": str(exc)}


def _handle_discover_mcp_tools(
    args: Dict[str, Any],
    *,
    config: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """Discover tools across MCP servers."""
    try:
        # Use resilient client if available
        try:
            from core.agent_sdk.resilience import get_health_monitor, ResilientMCPClient

            monitor = get_health_monitor()
            client = ResilientMCPClient(health_monitor=monitor)
            return asyncio.run(client.invoke("discovery", "search_tools_semantically", {"query": args.get("query", "")}, timeout=5.0))
        except ImportError:
            # Fallback to direct request
            import requests

            discovery_url = "http://localhost:8025/call"
            resp = requests.post(
                discovery_url,
                json={"tool_name": "search_tools_semantically", "args": {"query": args.get("query", "")}},
                timeout=3,
            )
            if resp.ok:
                return resp.json()
            return {"tools": [], "error": f"Discovery server returned {resp.status_code}"}
    except Exception as exc:
        return {"tools": [], "error": str(exc)}


def _handle_invoke_mcp_tool(
    args: Dict[str, Any],
    *,
    config: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """Invoke a tool on any MCP server with resilience patterns."""
    import asyncio

    server = args["server"]
    tool_name = args["tool_name"]
    tool_args = args.get("tool_args", {})
    timeout = args.get("timeout", 30)

    try:
        # Use resilient client for production hardening
        from core.agent_sdk.resilience import get_health_monitor, ResilientMCPClient

        monitor = get_health_monitor()
        client = ResilientMCPClient(health_monitor=monitor)

        # Run async invocation in sync context
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(client.invoke(server, tool_name, tool_args, timeout=timeout))
    except Exception as exc:
        # Fallback to direct request on resilience failure
        try:
            import requests
            from core.agent_sdk.config import AgentSDKConfig

            cfg = config or AgentSDKConfig()
            port = cfg.mcp_ports.get(server)
            if not port:
                return {"error": f"Unknown MCP server: {server}"}

            resp = requests.post(
                f"http://localhost:{port}/call",
                json={"tool_name": tool_name, "args": tool_args},
                timeout=timeout,
            )
            return resp.json() if resp.ok else {"error": f"Server returned {resp.status_code}"}
        except Exception as fallback_exc:
            return {"error": str(exc), "fallback_error": str(fallback_exc)}


def _handle_run_workflow(
    args: Dict[str, Any],
    **_: Any,
) -> Dict[str, Any]:
    """Execute a named workflow definition."""
    try:
        from core.workflow_engine import WorkflowEngine

        engine = WorkflowEngine()
        result = engine.execute(
            workflow_name=args["workflow_name"],
            inputs=args.get("inputs", {}),
        )
        return result if isinstance(result, dict) else {"result": str(result)}
    except Exception as exc:
        return {"error": str(exc)}


def _handle_query_codebase(
    args: Dict[str, Any],
    *,
    semantic_querier: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """Dispatch query_type to SemanticQuerier methods."""
    if semantic_querier is None:
        return {"error": "Semantic index not available. Run 'agent scan' first."}

    query_type = args["query_type"]
    target = args.get("target", "")
    depth = args.get("depth", 2)

    dispatch = {
        "what_calls": lambda: semantic_querier.what_calls(target),
        "what_depends_on": lambda: semantic_querier.what_depends_on(target),
        "what_changes_break": lambda: semantic_querier.what_changes_break(target, depth=depth),
        "summarize": lambda: {"summary": semantic_querier.summarize(target)},
        "find_similar": lambda: {"results": semantic_querier.find_similar(target)},
        "architecture_overview": lambda: semantic_querier.architecture_overview(),
        "recent_changes": lambda: {"changes": semantic_querier.recent_changes(n_commits=depth)},
    }
    handler = dispatch.get(query_type)
    if not handler:
        return {"error": f"Unknown query_type: {query_type}"}
    try:
        return handler()
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------------

_TOOL_DEFS: List[Dict[str, Any]] = [
    {
        "name": "analyze_goal",
        "description": "Classify a development goal and gather project context including goal type, recommended skills, memory hints, and available MCP tool categories. Always call this first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "The development goal to analyze"},
            },
            "required": ["goal"],
        },
        "handler": _handle_analyze_goal,
    },
    {
        "name": "dispatch_skills",
        "description": "Run static analysis skills matched to a goal type. Returns findings from skills like linter, type checker, complexity scorer, security scanner, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "goal_type": {"type": "string", "enum": ["bug_fix", "feature", "refactor", "security", "docs", "default"]},
                "project_root": {"type": "string", "description": "Project root path (optional, uses config default)"},
            },
            "required": ["goal_type"],
        },
        "handler": _handle_dispatch_skills,
    },
    {
        "name": "create_plan",
        "description": "Generate a step-by-step implementation plan using the PlannerAgent (Senior Software Architect). Returns structured plan with steps, confidence, and complexity assessment.",
        "input_schema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "What to accomplish"},
                "memory_snapshot": {"type": "string", "description": "Current memory context snapshot"},
                "similar_past_problems": {"type": "string", "description": "Relevant past problems from memory"},
                "known_weaknesses": {"type": "string", "description": "Known system weaknesses to address"},
            },
            "required": ["goal"],
        },
        "handler": _handle_create_plan,
    },
    {
        "name": "critique_plan",
        "description": "Adversarial review of a plan by the CriticAgent (Principal Engineer). Checks correctness, security, completeness, feasibility. Returns issues with severity levels.",
        "input_schema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "The original goal"},
                "plan": {"type": "array", "items": {"type": "string"}, "description": "Plan steps to critique"},
            },
            "required": ["goal", "plan"],
        },
        "handler": _handle_critique_plan,
    },
    {
        "name": "generate_code",
        "description": "Generate code changes via the CoderAgent (Expert Python Developer). Uses CoT reasoning for problem analysis, approach selection, and testing strategy.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "What code to write or change"},
                "plan": {"type": "string", "description": "Implementation plan to follow"},
                "context": {"type": "string", "description": "Relevant code context"},
            },
            "required": ["task"],
        },
        "handler": _handle_generate_code,
    },
    {
        "name": "run_sandbox",
        "description": "Execute code in an isolated subprocess sandbox. Returns stdout, stderr, and exit code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Code to execute"},
                "language": {"type": "string", "default": "python"},
            },
            "required": ["code"],
        },
        "handler": _handle_run_sandbox,
    },
    {
        "name": "apply_changes",
        "description": "Apply file changes atomically with overwrite safety. Respects AURA's stale-snippet mismatch policy.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Relative path within project"},
                "new_code": {"type": "string", "description": "New file content or code block"},
                "old_code": {"type": "string", "description": "Expected existing content (empty for new file)"},
                "overwrite_file": {"type": "boolean", "default": False},
            },
            "required": ["file_path", "new_code"],
        },
        "handler": _handle_apply_changes,
    },
    {
        "name": "verify_changes",
        "description": "Run tests and linters against recent changes via VerifierAgent. Returns pass/fail status with details.",
        "input_schema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "What was being worked on"},
                "changed_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of changed file paths",
                },
            },
            "required": ["goal"],
        },
        "handler": _handle_verify_changes,
    },
    {
        "name": "reflect_on_outcome",
        "description": "Analyze the outcome of a development cycle. Updates skill weights, records learnings, detects patterns.",
        "input_schema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "verification": {"type": "object", "description": "Verification results from verify_changes"},
                "skill_context": {"type": "object", "description": "Skill execution results from dispatch_skills"},
            },
            "required": ["goal"],
        },
        "handler": _handle_reflect,
    },
    {
        "name": "search_memory",
        "description": "Retrieve recent memories from AURA's brain, optionally filtered by a keyword query. Returns memories truncated to fit within max_tokens.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Optional keyword to filter results by"},
                "max_tokens": {"type": "integer", "default": 4000, "description": "Token budget for returned memories"},
            },
        },
        "handler": _handle_search_memory,
    },
    {
        "name": "store_memory",
        "description": "Persist a learning, decision, or insight to AURA's semantic memory for future recall.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "What to remember"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for retrieval"},
            },
            "required": ["content"],
        },
        "handler": _handle_store_memory,
    },
    {
        "name": "manage_goals",
        "description": "Manage the persistent goal queue: add new goals, list pending, complete, or get next.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "list", "complete", "next"]},
                "goal": {"type": "string", "description": "Goal text (for add/complete actions)"},
            },
            "required": ["action"],
        },
        "handler": _handle_manage_goals,
    },
    {
        "name": "discover_mcp_tools",
        "description": "Search for available tools across all AURA MCP servers (dev_tools, skills, control, thinking, agentic_loop, copilot, sadd).",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What kind of tool you need"},
            },
            "required": ["query"],
        },
        "handler": _handle_discover_mcp_tools,
    },
    {
        "name": "invoke_mcp_tool",
        "description": "Call any tool on any AURA MCP server by name. Use discover_mcp_tools first to find available tools.",
        "input_schema": {
            "type": "object",
            "properties": {
                "server": {"type": "string", "enum": ["dev_tools", "skills", "control", "thinking", "agentic_loop", "copilot", "sadd"]},
                "tool_name": {"type": "string", "description": "Tool name on the server"},
                "tool_args": {"type": "object", "description": "Arguments for the tool"},
                "timeout": {"type": "integer", "default": 30},
            },
            "required": ["server", "tool_name"],
        },
        "handler": _handle_invoke_mcp_tool,
    },
    {
        "name": "run_workflow",
        "description": "Execute a named workflow definition from the workflow engine. Workflows are DAG-based multi-step pipelines.",
        "input_schema": {
            "type": "object",
            "properties": {
                "workflow_name": {"type": "string", "description": "Name of the workflow to run"},
                "inputs": {"type": "object", "description": "Input parameters for the workflow"},
            },
            "required": ["workflow_name"],
        },
        "handler": _handle_run_workflow,
    },
    {
        "name": "query_codebase",
        "description": "Query the semantic codebase index for deep code understanding. Use this to understand call chains, dependencies, impact of changes, and to find relevant code by description.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["what_calls", "what_depends_on", "what_changes_break", "summarize", "find_similar", "architecture_overview", "recent_changes"],
                },
                "target": {"type": "string", "description": "File path, symbol name, or description"},
                "depth": {"type": "integer", "default": 2, "description": "Recursion depth for transitive queries"},
            },
            "required": ["query_type"],
        },
        "handler": _handle_query_codebase,
    },
]


def create_aura_tools(
    project_root: Path,
    brain: Any = None,
    model_adapter: Any = None,
    goal_queue: Any = None,
    goal_archive: Any = None,
    config: Any = None,
    semantic_querier: Any = None,
) -> List[AuraTool]:
    """Create all custom AURA tools for Agent SDK registration."""
    deps = dict(
        project_root=project_root,
        brain=brain,
        model_adapter=model_adapter,
        goal_queue=goal_queue,
        goal_archive=goal_archive,
        config=config,
        semantic_querier=semantic_querier,
    )

    tools = []
    for defn in _TOOL_DEFS:
        raw_handler = defn["handler"]

        def make_bound(h: Callable, d: Dict) -> Callable:
            def bound(args: Dict[str, Any]) -> Any:
                return h(args, **d)

            return bound

        tools.append(
            AuraTool(
                name=defn["name"],
                description=defn["description"],
                input_schema=defn["input_schema"],
                handler=make_bound(raw_handler, deps),
            )
        )
    return tools
