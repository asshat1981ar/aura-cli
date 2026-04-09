import json
from pathlib import Path

from core.capability_manager import build_capability_status_report, capability_doctor_check
from core.goal_queue import GoalQueue
from core.goal_archive import GoalArchive
from core.logging_utils import log_json
from core.task_handler import run_goals_loop  # Import run_goals_loop from core.task_handler
from core.task_manager import TaskManager

# Module-level MetaConductor singleton for innovation sessions
_meta_conductor = None


def _get_meta_conductor(brain=None, use_llm: bool = True):
    """Get or create the shared MetaConductor instance."""
    global _meta_conductor
    if _meta_conductor is None:
        from agents.meta_conductor import MetaConductor
        _meta_conductor = MetaConductor(brain=brain, use_llm=use_llm)
    elif brain is not None and _meta_conductor.brain is None:
        # Update brain if conductor exists but brain was added
        _meta_conductor.brain = brain
    return _meta_conductor


def _handle_help():
    try:
        from aura_cli.cli_options import render_help

        print(render_help().rstrip())
    except Exception:
        # Fallback to a minimal static help if parser help rendering fails.
        print("\n--- AURA CLI Commands ---")
        print("add <goal_description> - Add a new goal to the queue.")
        print("run                    - Run the AURA loop for goals in the queue.")
        print("status                 - Show the current status of tasks and goals.")
        print("doctor                 - Run system health checks.")
        print("clear                  - Clear the screen.")
        print("exit                   - Exit the AURA CLI.")
        print("help                   - Show this help message.")
        print("-------------------------\n")


def _handle_doctor(project_root: Path | None = None):
    log_json("INFO", "aura_doctor_requested")
    from aura_cli.doctor import run_doctor_v2

    run_doctor_v2(
        project_root=project_root,
        rich_output=True,
        capability_check=capability_doctor_check,
    )


def _handle_readiness():
    """M7-004: Validate async runtime and MCP registry health."""
    import anyio
    from core.mcp_health import check_all_mcp_health, get_health_summary
    from core.mcp_agent_registry import agent_registry
    from core.config_manager import config

    print("\n--- AURA Readiness Check (V2) ---")
    print(f"Async Orchestrator: {'ENABLED' if config.get('enable_new_orchestrator') else 'DISABLED'}")
    print(f"Typed Registry:     {'ENABLED' if config.get('enable_mcp_registry') else 'DISABLED'}")

    results = anyio.run(check_all_mcp_health)
    summary = get_health_summary(results)

    print(f"\nMCP Servers: {summary['healthy_count']}/{summary['total_servers']} healthy")
    for res in results:
        status_icon = "✅" if res["status"] == "healthy" else "❌"
        print(f"  {status_icon} {res['name']:<15} (Port: {res.get('port', 'N/A')})")

    agents = agent_registry.list_agents()
    print(f"\nRegistered Agents: {len(agents)}")
    local_count = sum(1 for a in agents if a.source == "local")
    mcp_count = sum(1 for a in agents if a.source == "mcp")
    print(f"  - Local: {local_count}")
    print(f"  - MCP:   {mcp_count}")

    if summary["all_healthy"]:
        print("\n✅ System is READY for autonomous operation.\n")
    else:
        print("\n⚠️  System is DEGRADED. Check unhealthy MCP servers.\n")


def _handle_clear():
    import subprocess
    import os

    cmd = "clear" if os.name == "posix" else "cls"
    subprocess.run([cmd], shell=False)  # noqa: S603


def _handle_add(goal_queue: GoalQueue, command: str):
    goal = command[4:].strip()
    if not goal:
        log_json("WARN", "add_command_no_goal_provided")
        return

    # Basic Sanitization & Security
    if len(goal) > 500:
        log_json("ERROR", "goal_title_too_long", details={"length": len(goal)})
        print("Error: Goal description is too long (max 500 chars).")
        return

    # Check for suspicious shell-like characters if needed,
    # though GoalQueue usually just stores it as a string in JSON.
    # However, preventing characters that might be interpreted in some contexts is good.
    forbidden_chars = [";", "&&", "||", "`", "$("]
    if any(char in goal for char in forbidden_chars):
        log_json("SECURITY_WARN", "suspicious_goal_content", details={"goal": goal})
        print("Error: Goal description contains suspicious characters.")
        return

    goal_queue.add(goal)
    log_json("INFO", "goal_added", goal=goal)
    print(f"Added goal: {goal}")
    print(f"Queue length: {len(goal_queue.queue)}")


def _handle_run(args, goal_queue: GoalQueue, goal_archive: GoalArchive, orchestrator, debugger_instance, planner_instance, project_root: Path):
    if not goal_queue.has_goals():
        log_json("WARN", "run_command_no_goals_in_queue")
        return
    run_goals_loop(args, goal_queue, orchestrator, debugger_instance, planner_instance, goal_archive, project_root, decompose=getattr(args, "decompose", False))


def _handle_status(
    goal_queue: GoalQueue,
    goal_archive: GoalArchive,
    orchestrator,
    *,
    as_json: bool = False,
    project_root: Path | None = None,
    memory_persistence_path: Path | str | None = None,
    memory_store=None,
):
    log_json("INFO", "aura_status_requested")

    from core.operator_runtime import build_beads_runtime_metadata, build_operator_runtime_snapshot

    active_cycle = getattr(orchestrator, "active_cycle_summary", None)
    last_cycle = getattr(orchestrator, "last_cycle_summary", None)
    beads_runtime = build_beads_runtime_metadata(orchestrator)

    snapshot = build_operator_runtime_snapshot(
        goal_queue=goal_queue,
        goal_archive=goal_archive,
        active_cycle=active_cycle,
        last_cycle=last_cycle,
        beads_runtime=beads_runtime,
        memory_store=memory_store,
    )

    if as_json:
        # Include legacy capabilities field for now to avoid breaking tools
        capability_status = build_capability_status_report(
            Path(project_root or Path.cwd()),
            goal_queue=goal_queue,
            last_status=getattr(orchestrator, "last_capability_status", None),
        )
        snapshot["capabilities"] = capability_status
        print(json.dumps(snapshot))
        return

    print("\n--- AURA Status ---")
    print(f"Goals in queue: {snapshot['queue']['pending_count']}")
    for item in snapshot["queue"]["pending"]:
        print(f"  {item['position']}. {item['goal']}")

    print(f"Completed goals: {snapshot['queue']['completed_count']}")
    for item in snapshot["queue"]["completed"]:
        score_str = f" (Score: {item['score']:.2f})" if item["score"] is not None else ""
        print(f"  - '{item['goal']}'{score_str}")

    if beads_runtime:
        gate_mode = "required" if beads_runtime.get("required") else "optional"
        print("\n--- BEADS Gate ---")
        print(f"Enabled: {'yes' if beads_runtime.get('enabled') else 'no'}")
        print(f"Mode: {gate_mode}")
        print(f"Scope: {beads_runtime.get('scope', 'goal_run')}")

    if active_cycle:
        print("\n--- Active Cycle ---")
        print(f"ID: {active_cycle['cycle_id']}")
        print(f"Goal: {active_cycle['goal']}")
        print(f"Phase: {active_cycle['current_phase']} ({active_cycle['state']})")

    run_tool_audit = snapshot.get("run_tool_audit")
    if isinstance(run_tool_audit, dict):
        print("\n--- Run Tool Audit ---")
        print(f"Recent commands tracked: {run_tool_audit.get('count', 0)}")
        print(f"Last command: {run_tool_audit.get('last_command') or 'n/a'}")
        print(
            "Recent outcomes: "
            f"{run_tool_audit.get('success_count', 0)} ok, "
            f"{run_tool_audit.get('error_count', 0)} error, "
            f"{run_tool_audit.get('timeout_count', 0)} timed out, "
            f"{run_tool_audit.get('truncated_count', 0)} truncated"
        )
        if active_cycle.get("beads_status"):
            beads_label = active_cycle["beads_status"]
            if active_cycle.get("beads_decision_id"):
                beads_label += f" ({active_cycle['beads_decision_id']})"
            print(f"BEADS: {beads_label}")
            if active_cycle.get("beads_summary"):
                print(f"BEADS Summary: {active_cycle['beads_summary']}")
    elif last_cycle:
        print("\n--- Last Cycle ---")
        print(f"ID: {last_cycle['cycle_id']}")
        print(f"Goal: {last_cycle['goal']}")
        print(f"Outcome: {last_cycle['outcome']}")
        print(f"Stop Reason: {last_cycle['stop_reason']}")
        if last_cycle.get("beads_status"):
            beads_label = last_cycle["beads_status"]
            if last_cycle.get("beads_decision_id"):
                beads_label += f" ({last_cycle['beads_decision_id']})"
            print(f"BEADS: {beads_label}")
            if last_cycle.get("beads_summary"):
                print(f"BEADS Summary: {last_cycle['beads_summary']}")

    # Task Hierarchy
    task_manager = TaskManager(persistence_path=memory_persistence_path)
    if task_manager.root_tasks:
        print("\n--- Task Hierarchy ---")
        for task in task_manager.root_tasks:
            print(task.display())

    # Add Loop status
    print("\n--- AURA Loop Status ---")
    if hasattr(orchestrator, "current_score"):
        print(f"Current Loop Score: {orchestrator.current_score:.2f}")
        print(f"Regression Count: {orchestrator.regression_count}")
        print(f"Stable Convergence Count: {orchestrator.stable_convergence_count}")
        print(f"Current Goal: {orchestrator.current_goal if orchestrator.current_goal else 'None'}")
    elif active_cycle:
        print(f"Orchestrator active. Current Goal: {active_cycle['goal']}")
    else:
        print("Loop status: idle.")
    print("------------------------\n")


# ── Innovation Catalyst Command Handlers ─────────────────────────────────────


def _handle_innovate_start(args, runtime=None):
    """Start a new innovation session with the Innovation Catalyst.
    
    Args:
        args: Parsed CLI arguments
        runtime: Optional runtime dict with brain
    """
    from agents.brainstorming_bots import list_techniques
    from agents.schemas import InnovationPhase
    
    log_json("INFO", "innovate_start_requested")
    
    # Get brain from runtime if available
    brain = None
    if runtime and isinstance(runtime, dict):
        brain = runtime.get("brain")
    
    # Parse techniques
    techniques_str = getattr(args, "techniques", "")
    if techniques_str:
        techniques = [t.strip() for t in techniques_str.split(",")]
    else:
        techniques = list_techniques()  # Use all available
    
    # Validate techniques
    valid_techniques = list_techniques()
    invalid = [t for t in techniques if t not in valid_techniques]
    if invalid:
        print(f"Error: Invalid techniques: {invalid}")
        print(f"Valid techniques: {', '.join(valid_techniques)}")
        return
    
    # Parse constraints if provided
    constraints = {}
    constraints_json = getattr(args, "constraints", "")
    if constraints_json:
        try:
            constraints = json.loads(constraints_json)
        except json.JSONDecodeError:
            print("Error: Invalid JSON in --constraints")
            return
    
    # Get output format
    output_json = getattr(args, "json", False) or getattr(args, "output", "table") == "json"
    use_llm = getattr(args, "use_llm", True)
    
    # Log LLM mode
    if use_llm:
        log_json("INFO", "llm_mode_enabled")
    else:
        log_json("INFO", "llm_mode_disabled", details={"reason": "user_flag"})
    
    # Get problems - either from args, batch file, or error
    problems = []
    batch_file = getattr(args, "batch_file", None)
    
    if batch_file:
        # Read problems from file
        try:
            with open(batch_file, 'r') as f:
                problems = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        except FileNotFoundError:
            print(f"Error: Batch file not found: {batch_file}")
            return
        except Exception as e:
            print(f"Error reading batch file: {e}")
            return
    else:
        # Get problem from command line
        problem = " ".join(getattr(args, "problem_statement", []))
        if problem:
            problems = [problem]
    
    if not problems:
        print("Error: Problem statement required.")
        print('Usage: python3 main.py innovate start "How to improve X?"')
        print('   or: python3 main.py innovate start --batch problems.txt')
        return
    
    # Use shared conductor with brain
    conductor = _get_meta_conductor(brain=brain, use_llm=use_llm)
    
    # Process each problem
    sessions = []
    for problem in problems:
        session = conductor.start_session(
            problem_statement=problem,
            techniques=techniques,
            constraints=constraints
        )
        sessions.append(session)
    
    # Optionally execute phase for all sessions
    execute_phase = getattr(args, "execute_phase", None)
    phase_results = []
    if execute_phase:
        phase_map = {
            "immersion": InnovationPhase.IMMERSION,
            "divergence": InnovationPhase.DIVERGENCE,
            "convergence": InnovationPhase.CONVERGENCE,
            "incubation": InnovationPhase.INCUBATION,
            "transformation": InnovationPhase.TRANSFORMATION,
        }
        if execute_phase in phase_map:
            if not output_json:
                print(f"\n▶️  Executing {execute_phase} phase for {len(sessions)} session(s)...")
            for session in sessions:
                result = conductor.execute_phase(session.session_id, phase_map[execute_phase])
                phase_results.append(result)
    
    # Output results
    if output_json:
        output = {
            "sessions": [
                {
                    "session_id": s.session_id,
                    "problem": s.problem_statement,
                    "techniques": techniques,
                    "status": s.status,
                    "current_phase": s.current_phase.value,
                    "phases_completed": [p.value for p in s.phases_completed],
                }
                for s in sessions
            ],
            "count": len(sessions),
        }
        if phase_results:
            output["phase_results"] = phase_results
        print(json.dumps(output, indent=2))
    else:
        if len(sessions) == 1:
            s = sessions[0]
            print("\n🚀 Innovation Session Started")
            print(f"Problem: {s.problem_statement}")
            print(f"Techniques: {', '.join(techniques)}")
            print(f"Session ID: {s.session_id}")
            print(f"Status: {s.status}")
            print(f"Current phase: {s.current_phase.value}")
            if phase_results:
                print(f"\n✅ Phase executed: {execute_phase}")
                for r in phase_results:
                    if 'ideas_generated' in r:
                        print(f"Ideas generated: {r['ideas_generated']}")
            print("\nNext steps:")
            print(f"  Show:   python3 main.py innovate show {s.session_id}")
            print(f"  Resume: python3 main.py innovate resume {s.session_id} --phase divergence")
        else:
            print(f"\n🚀 Started {len(sessions)} Innovation Sessions")
            print(f"Techniques: {', '.join(techniques)}")
            print("\nSessions:")
            for s in sessions:
                print(f"  {s.session_id}: {s.problem_statement[:50]}...")
            if phase_results:
                total_ideas = sum(r.get('ideas_generated', 0) for r in phase_results)
                print(f"\n✅ Phase '{execute_phase}' executed for all sessions")
                print(f"Total ideas generated: {total_ideas}")
            print("\nShow all: python3 main.py innovate list")


def _handle_innovate_list(args, runtime=None):
    """List all innovation sessions."""
    log_json("INFO", "innovate_list_requested")
    
    # Get brain from runtime if available
    brain = None
    if runtime and isinstance(runtime, dict):
        brain = runtime.get("brain")
    
    conductor = _get_meta_conductor(brain=brain)
    sessions = conductor.list_sessions()
    
    # Get output format
    output_json = getattr(args, "json", False) or getattr(args, "output", "table") == "json"
    limit = getattr(args, "limit", 20)
    
    if output_json:
        output = {
            "sessions": [
                {
                    "session_id": s.session_id,
                    "problem": s.problem_statement,
                    "status": s.status,
                    "current_phase": s.current_phase.value,
                    "techniques": s.techniques,
                    "ideas_count": s.ideas_generated,
                    "selected_count": s.ideas_selected,
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                }
                for s in sessions[:limit]
            ],
            "total": len(sessions),
        }
        print(json.dumps(output, indent=2))
    else:
        print("\n📋 Innovation Sessions")
        print("-" * 60)
        
        if not sessions:
            print("\nNo active sessions.")
            print("\nStart a new session with:")
            print('  python3 main.py innovate start "How to improve X?"')
        else:
            # Group by status
            active = [s for s in sessions if s.status == "active"]
            completed = [s for s in sessions if s.status == "completed"]
            
            if active:
                print(f"\n🟢 Active Sessions ({len(active)}):")
                for s in active[:limit]:
                    ideas_count = s.ideas_generated
                    selected_count = s.ideas_selected
                    print(f"  {s.session_id} - {s.problem_statement[:40]}...")
                    print(f"    Phase: {s.current_phase.value} | Ideas: {ideas_count} | Selected: {selected_count}")
            
            if completed:
                print(f"\n✅ Completed Sessions ({len(completed)}):")
                for s in completed[:limit]:
                    ideas_count = s.ideas_generated
                    selected_count = s.ideas_selected
                    print(f"  {s.session_id} - {s.problem_statement[:40]}...")
                    print(f"    Ideas: {ideas_count} | Selected: {selected_count}")
            
            total_ideas = sum(s.ideas_generated for s in sessions)
            print(f"\nTotal: {len(sessions)} sessions, {total_ideas} ideas generated")


def _handle_innovate_show(args, runtime=None):
    """Show details of a specific innovation session."""
    log_json("INFO", "innovate_show_requested")
    
    session_id = getattr(args, "session_id", None) or getattr(args, "session_id", None)
    if not session_id:
        print("Error: session_id required")
        return
    
    # Get brain from runtime if available
    brain = None
    if runtime and isinstance(runtime, dict):
        brain = runtime.get("brain")
    
    conductor = _get_meta_conductor(brain=brain)
    session = conductor.get_session(session_id)
    
    # Get output format
    output_json = getattr(args, "json", False) or getattr(args, "output", "table") == "json"
    show_ideas = getattr(args, "show_ideas", False)
    
    if not session:
        if output_json:
            print(json.dumps({"error": f"Session {session_id} not found"}))
        else:
            print(f"\n❌ Session not found: {session_id}")
            print("\nUse 'python3 main.py innovate list' to see available sessions.")
        return
    
    # Get idea counts from output if available
    ideas_count = session.ideas_generated
    selected_count = session.ideas_selected
    all_ideas = session.output.all_ideas if session.output else []
    selected_ideas = session.output.selected_ideas if session.output else []
    
    if output_json:
        output = {
            "session_id": session.session_id,
            "problem": session.problem_statement,
            "status": session.status,
            "current_phase": session.current_phase.value,
            "phases_completed": [p.value for p in session.phases_completed],
            "techniques": session.techniques,
            "constraints": session.constraints,
            "ideas_count": ideas_count,
            "selected_count": selected_count,
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "updated_at": session.updated_at.isoformat() if session.updated_at else None,
        }
        if show_ideas and all_ideas:
            output["ideas"] = [
                {
                    "description": idea.description,
                    "technique": idea.technique,
                    "novelty": idea.novelty,
                    "feasibility": idea.feasibility,
                    "impact": idea.impact,
                }
                for idea in all_ideas
            ]
        print(json.dumps(output, indent=2))
    else:
        print(f"\n📊 Session: {session.session_id}")
        print("-" * 60)
        print(f"Problem: {session.problem_statement}")
        print(f"Status: {'🟢 Active' if session.status == 'active' else '✅ ' + session.status}")
        print(f"Current Phase: {session.current_phase.value}")
        print(f"Techniques: {', '.join(session.techniques)}")
        print("\nProgress:")
        print(f"  Ideas generated: {ideas_count}")
        print(f"  Ideas selected: {selected_count}")
        
        if session.output:
            print(f"  Diversity score: {session.output.diversity_score:.2f}")
            print(f"  Novelty score: {session.output.novelty_score:.2f}")
            print(f"  Feasibility score: {session.output.feasibility_score:.2f}")
        
        if show_ideas and all_ideas:
            print(f"\n💡 Ideas ({len(all_ideas)}):")
            for idea in all_ideas[:20]:  # Limit to 20 for display
                selected = "✓" if idea in selected_ideas else " "
                print(f"  [{selected}] ({idea.technique}) {idea.description[:60]}...")
            if len(all_ideas) > 20:
                print(f"  ... and {len(all_ideas) - 20} more")
        
        print("\nNext steps:")
        print(f"  Resume: python3 main.py innovate resume {session.session_id} --phase divergence")
        print(f"  Export: python3 main.py innovate export {session.session_id}")


def _handle_innovate_resume(args, runtime=None):
    """Resume an innovation session at a specific phase."""
    log_json("INFO", "innovate_resume_requested")
    
    session_id = getattr(args, "session_id", None) or getattr(args, "session_id", None)
    if not session_id:
        print("Error: session_id required")
        return
    
    phase = getattr(args, "phase", None)
    
    # Get brain from runtime if available
    brain = None
    if runtime and isinstance(runtime, dict):
        brain = runtime.get("brain")
    
    conductor = _get_meta_conductor(brain=brain)
    session = conductor.get_session(session_id)
    
    if not session:
        print(f"\n❌ Session not found: {session_id}")
        print("\nUse 'python3 main.py innovate list' to see available sessions.")
        return
    
    from agents.schemas import InnovationPhase
    phase_map = {
        "immersion": InnovationPhase.IMMERSION,
        "divergence": InnovationPhase.DIVERGENCE,
        "convergence": InnovationPhase.CONVERGENCE,
        "incubation": InnovationPhase.INCUBATION,
        "transformation": InnovationPhase.TRANSFORMATION,
    }
    
    # Determine which phase to execute
    target_phase = None
    if phase:
        target_phase = phase_map.get(phase)
        if not target_phase:
            print(f"Error: Invalid phase '{phase}'")
            print(f"Valid phases: {', '.join(phase_map.keys())}")
            return
    else:
        # Auto-determine next phase based on current state
        target_phase = session.current_phase
    
    # Get output format
    output_json = getattr(args, "json", False) or getattr(args, "output", "table") == "json"
    
    if not output_json:
        print(f"\n🔄 Resuming Session: {session_id}")
        print(f"Current phase: {session.current_phase.value}")
        print(f"Target phase: {target_phase.value}")
        print("\n▶️  Executing phase...")
    
    try:
        result = conductor.execute_phase(session_id, target_phase)
        
        # Update session state
        session = conductor.get_session(session_id)
        
        if output_json:
            output = {
                "session_id": session_id,
                "executed_phase": target_phase.value,
                "result": result,
                "current_phase": session.current_phase.value if session else None,
                "ideas_count": session.ideas_generated if session else 0,
            }
            print(json.dumps(output, indent=2))
        else:
            print(f"\n✅ Phase complete: {target_phase.value}")
            if 'ideas_count' in result:
                print(f"Ideas in this phase: {result['ideas_count']}")
            print(f"\nSession now at: {session.current_phase.value}")
            print(f"Total ideas: {session.ideas_generated}")
            print("\nNext steps:")
            print(f"  Show:   python3 main.py innovate show {session_id}")
            print(f"  Resume: python3 main.py innovate resume {session_id}")
            
    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"\n❌ Error executing phase: {e}")


def _handle_innovate_techniques(args):
    """List available brainstorming techniques."""
    log_json("INFO", "innovate_techniques_requested")
    
    from agents.brainstorming_bots import BRAINSTORMING_BOTS
    
    # Technique descriptions
    descriptions = {
        "scamper": "Substitute, Combine, Adapt, Modify, Put to other uses, Eliminate, Reverse",
        "six_hats": "Six Thinking Hats - Parallel thinking from different perspectives",
        "mind_map": "Visual brainstorming with hierarchical idea mapping",
        "reverse": "Reverse Brainstorming - Identify problems to find solutions",
        "worst_idea": "Worst Idea Technique - Invert bad ideas to find good ones",
        "lotus": "Lotus Blossom - Expand ideas in a grid pattern",
        "star": "Starbursting - Generate questions from different angles",
        "bia": "Bottleneck Identification & Analysis - Find constraints",
    }
    
    output_json = getattr(args, "json", False) or getattr(args, "output", "table") == "json"
    
    techniques = []
    for key, bot_class in BRAINSTORMING_BOTS.items():
        techniques.append({
            "id": key,
            "name": bot_class().technique_name,
            "description": descriptions.get(key, ""),
        })
    
    if output_json:
        print(json.dumps({"techniques": techniques}, indent=2))
    else:
        print("\n🧠 Available Brainstorming Techniques")
        print("-" * 60)
        for t in techniques:
            print(f"\n{t['id']}")
            print(f"  Name: {t['name']}")
            print(f"  Description: {t['description']}")
        print("\nUsage: python3 main.py innovate start 'Problem' --techniques scamper,six_hats")


def _generate_json_export(session, all_ideas, selected_ideas):
    """Generate JSON export content."""
    export_data = {
        "session_id": session.session_id,
        "problem_statement": session.problem_statement,
        "status": session.status,
        "current_phase": session.current_phase.value,
        "phases_completed": [p.value for p in session.phases_completed],
        "techniques_used": session.techniques,
        "constraints": session.constraints,
        "ideas_count": session.ideas_generated,
        "selected_count": session.ideas_selected,
        "ideas": [
            {
                "description": idea.description,
                "technique": idea.technique,
                "novelty": idea.novelty,
                "feasibility": idea.feasibility,
                "impact": idea.impact,
                "metadata": idea.metadata,
            }
            for idea in all_ideas
        ],
        "selected_ideas": [
            {
                "description": idea.description,
                "technique": idea.technique,
            }
            for idea in selected_ideas
        ],
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "updated_at": session.updated_at.isoformat() if session.updated_at else None,
    }
    if session.output:
        export_data["scores"] = {
            "diversity": session.output.diversity_score,
            "novelty": session.output.novelty_score,
            "feasibility": session.output.feasibility_score,
        }
    return json.dumps(export_data, indent=2)


def _generate_csv_export(session, all_ideas, selected_ideas):
    """Generate CSV export content."""
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        "Session ID", "Problem", "Status", "Current Phase",
        "Idea #", "Technique", "Description", "Novelty", "Feasibility", "Impact", "Selected"
    ])
    
    # Data rows
    for i, idea in enumerate(all_ideas, 1):
        is_selected = "Yes" if idea in selected_ideas else "No"
        writer.writerow([
            session.session_id,
            session.problem_statement,
            session.status,
            session.current_phase.value,
            i,
            idea.technique,
            idea.description,
            f"{idea.novelty:.2f}",
            f"{idea.feasibility:.2f}",
            f"{idea.impact:.2f}",
            is_selected
        ])
    
    return output.getvalue()


def _generate_html_export(session, all_ideas, selected_ideas):
    """Generate HTML export content."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Innovation Session Report - {session.session_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .meta {{ background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .meta-row {{ margin: 5px 0; }}
        .idea {{ border: 1px solid #ddd; padding: 15px; margin: 15px 0; border-radius: 5px; }}
        .idea.selected {{ border-left: 4px solid #4CAF50; background: #f8fff8; }}
        .idea-header {{ font-weight: bold; color: #333; margin-bottom: 10px; }}
        .scores {{ color: #666; font-size: 0.9em; margin-top: 10px; }}
        .technique {{ display: inline-block; background: #e3f2fd; padding: 2px 8px; border-radius: 3px; font-size: 0.85em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Innovation Session Report</h1>
        
        <div class="meta">
            <div class="meta-row"><strong>Session ID:</strong> {session.session_id}</div>
            <div class="meta-row"><strong>Problem:</strong> {session.problem_statement}</div>
            <div class="meta-row"><strong>Status:</strong> {session.status}</div>
            <div class="meta-row"><strong>Current Phase:</strong> {session.current_phase.value}</div>
            <div class="meta-row"><strong>Techniques:</strong> {', '.join(session.techniques)}</div>
            <div class="meta-row"><strong>Total Ideas:</strong> {session.ideas_generated} | <strong>Selected:</strong> {session.ideas_selected}</div>
        </div>
        
        <h2>💡 Ideas Generated ({len(all_ideas)})</h2>
"""
    
    for i, idea in enumerate(all_ideas, 1):
        selected_class = "selected" if idea in selected_ideas else ""
        selected_badge = " ✅ SELECTED" if idea in selected_ideas else ""
        html += f"""
        <div class="idea {selected_class}">
            <div class="idea-header">Idea {i}{selected_badge}</div>
            <div><span class="technique">{idea.technique}</span></div>
            <p>{idea.description}</p>
            <div class="scores">
                Novelty: {idea.novelty:.2f} | 
                Feasibility: {idea.feasibility:.2f} | 
                Impact: {idea.impact:.2f}
            </div>
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    return html


def _generate_markdown_export(session, all_ideas, selected_ideas):
    """Generate Markdown export content."""
    lines = [
        "# Innovation Session Report",
        "",
        f"**Session ID:** {session.session_id}",
        f"**Status:** {session.status}",
        f"**Current Phase:** {session.current_phase.value}",
        f"**Created:** {session.created_at.strftime('%Y-%m-%d %H:%M') if session.created_at else 'Unknown'}",
        "",
        "## Problem Statement",
        "",
        f"{session.problem_statement}",
        "",
        "## Techniques Used",
        "",
        f"{', '.join(session.techniques)}",
        "",
        "## Summary",
        "",
        f"- **Total Ideas:** {session.ideas_generated}",
        f"- **Selected Ideas:** {session.ideas_selected}",
        f"- **Phases Completed:** {', '.join(p.value for p in session.phases_completed) or 'None'}",
        "",
    ]
    
    if session.output:
        lines.extend([
            "## Scores",
            "",
            f"- **Diversity:** {session.output.diversity_score:.2f}",
            f"- **Novelty:** {session.output.novelty_score:.2f}",
            f"- **Feasibility:** {session.output.feasibility_score:.2f}",
            "",
        ])
    
    if all_ideas:
        lines.extend([
            "## Ideas Generated",
            "",
        ])
        
        for i, idea in enumerate(all_ideas, 1):
            selected = "✅ Selected" if idea in selected_ideas else ""
            lines.append(f"### Idea {i} ({idea.technique}) {selected}")
            lines.append("")
            lines.append(f"{idea.description}")
            lines.append("")
            lines.append(f"**Scores:** Novelty: {idea.novelty:.2f} | Feasibility: {idea.feasibility:.2f} | Impact: {idea.impact:.2f}")
            lines.append("")
    
    if selected_ideas:
        lines.append("## Selected Ideas Summary")
        lines.append("")
        for idea in selected_ideas:
            lines.append(f"- **{idea.technique}**: {idea.description[:100]}...")
        lines.append("")
    
    return "\n".join(lines)


def _handle_innovate_export(args, runtime=None):
    """Export innovation session results."""
    log_json("INFO", "innovate_export_requested")
    
    session_id = getattr(args, "session_id", None) or getattr(args, "session_id", None)
    if not session_id:
        print("Error: session_id required")
        return
    
    format_type = getattr(args, "format", "markdown")
    output_path = getattr(args, "output", None)
    
    # Get brain from runtime if available
    brain = None
    if runtime and isinstance(runtime, dict):
        brain = runtime.get("brain")
    
    conductor = _get_meta_conductor(brain=brain)
    session = conductor.get_session(session_id)
    
    if not session:
        print(f"\n❌ Session not found: {session_id}")
        return
    
    # Get ideas from output
    all_ideas = session.output.all_ideas if session.output else []
    selected_ideas = session.output.selected_ideas if session.output else []
    
    # Generate export content based on format
    if format_type == "json":
        content = _generate_json_export(session, all_ideas, selected_ideas)
    elif format_type == "csv":
        content = _generate_csv_export(session, all_ideas, selected_ideas)
    elif format_type == "html":
        content = _generate_html_export(session, all_ideas, selected_ideas)
    else:  # markdown
        content = _generate_markdown_export(session, all_ideas, selected_ideas)
    
    # Output or save
    if output_path:
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"\n✅ Exported to: {output_path}")
        print(f"Format: {format_type}")
        print(f"Session: {session_id}")
        print(f"Ideas: {len(all_ideas)}")
    else:
        print(content)
    
    return content


def _handle_innovate_to_goals(args, runtime=None):
    """Convert selected ideas from an innovation session to goals."""
    log_json("INFO", "innovate_to_goals_requested")
    
    session_id = getattr(args, "session_id", None)
    if not session_id:
        print("Error: --session-id required")
        return
    
    preview = getattr(args, "preview", False)
    max_goals = getattr(args, "max_goals", 5)
    
    # Get brain from runtime if available
    brain = None
    if runtime and isinstance(runtime, dict):
        brain = runtime.get("brain")
    
    conductor = _get_meta_conductor(brain=brain)
    session = conductor.get_session(session_id)
    
    if not session:
        print(f"\n❌ Session not found: {session_id}")
        return
    
    # Get selected ideas
    selected_ideas = []
    if session.output and session.output.selected_ideas:
        selected_ideas = session.output.selected_ideas
    
    if not selected_ideas:
        print(f"\n⚠️  No selected ideas in session {session_id}")
        print("Run convergence phase first to select ideas.")
        return
    
    # Limit to max_goals
    ideas_to_convert = selected_ideas[:max_goals]
    
    print(f"\n🎯 Converting {len(ideas_to_convert)} ideas to goals")
    print(f"Session: {session_id}")
    print(f"Problem: {session.problem_statement}")
    
    if preview:
        print("\n--- PREVIEW - Goals that would be created ---")
        for i, idea in enumerate(ideas_to_convert, 1):
            print(f"\n{i}. [{idea.technique}] {idea.description[:80]}...")
        print("\n--- End preview ---")
        return
    
    # Create goals from ideas
    created = 0
    goal_queue = None
    if runtime and isinstance(runtime, dict):
        goal_queue = runtime.get("goal_queue")
    
    for idea in ideas_to_convert:
        goal_text = f"[From innovation {session_id}] {idea.technique}: {idea.description[:100]}"
        
        if goal_queue:
            try:
                goal_queue.add(goal_text)
                created += 1
            except Exception as e:
                print(f"  ⚠️  Failed to add goal: {e}")
        else:
            # Fallback: just print what would be added
            print(f"  + {goal_text[:60]}...")
            created += 1
    
    print(f"\n✅ Created {created} goals from {len(ideas_to_convert)} ideas")
    
    if goal_queue:
        print("\nView goals: python3 main.py goal status")
        print("Run goals:  python3 main.py goal run")


def _handle_innovate_insights(args, runtime=None):
    """Show analytics and insights about innovation sessions."""
    log_json("INFO", "innovate_insights_requested")
    
    session_id = getattr(args, "session_id", None)
    output_json = getattr(args, "json", False) or getattr(args, "output", "table") == "json"
    
    # Get brain from runtime if available
    brain = None
    if runtime and isinstance(runtime, dict):
        brain = runtime.get("brain")
    
    conductor = _get_meta_conductor(brain=brain)
    
    if session_id:
        # Show insights for specific session
        session = conductor.get_session(session_id)
        if not session:
            print(f"\n❌ Session not found: {session_id}")
            return
        
        insights = _compute_session_insights(session)
        
        if output_json:
            print(json.dumps(insights, indent=2))
        else:
            _print_session_insights(insights)
    else:
        # Show global insights
        sessions = conductor.list_sessions()
        insights = _compute_global_insights(sessions)
        
        if output_json:
            print(json.dumps(insights, indent=2))
        else:
            _print_global_insights(insights)


def _compute_session_insights(session):
    """Compute detailed insights for a single session."""
    insights = {
        "session_id": session.session_id,
        "problem": session.problem_statement,
        "status": session.status,
        "techniques_used": session.techniques,
        "ideas_generated": session.ideas_generated,
        "ideas_selected": session.ideas_selected,
    }
    
    if session.output:
        all_ideas = session.output.all_ideas
        selected = session.output.selected_ideas
        
        # Technique breakdown
        technique_counts = {}
        for idea in all_ideas:
            technique_counts[idea.technique] = technique_counts.get(idea.technique, 0) + 1
        insights["technique_breakdown"] = technique_counts
        
        # Quality metrics
        if all_ideas:
            insights["quality_metrics"] = {
                "avg_novelty": sum(i.novelty for i in all_ideas) / len(all_ideas),
                "avg_feasibility": sum(i.feasibility for i in all_ideas) / len(all_ideas),
                "avg_impact": sum(i.impact for i in all_ideas) / len(all_ideas),
                "selected_avg_novelty": sum(i.novelty for i in selected) / len(selected) if selected else 0,
                "selected_avg_feasibility": sum(i.feasibility for i in selected) / len(selected) if selected else 0,
                "selected_avg_impact": sum(i.impact for i in selected) / len(selected) if selected else 0,
            }
        
        # Top ideas by score
        top_ideas = sorted(all_ideas, key=lambda i: i.novelty + i.feasibility + i.impact, reverse=True)[:5]
        insights["top_ideas"] = [
            {
                "technique": i.technique,
                "description": i.description[:100],
                "total_score": round(i.novelty + i.feasibility + i.impact, 2)
            }
            for i in top_ideas
        ]
    
    return insights


def _compute_global_insights(sessions):
    """Compute global insights across all sessions."""
    total_ideas = sum(s.ideas_generated for s in sessions)
    total_selected = sum(s.ideas_selected for s in sessions)
    
    # Technique frequency
    technique_counts = {}
    for s in sessions:
        for t in s.techniques:
            technique_counts[t] = technique_counts.get(t, 0) + 1
    
    # Average scores across sessions with output
    all_novelty = []
    all_feasibility = []
    all_diversity = []
    for s in sessions:
        if s.output:
            all_novelty.append(s.output.novelty_score)
            all_feasibility.append(s.output.feasibility_score)
            all_diversity.append(s.output.diversity_score)
    
    return {
        "total_sessions": len(sessions),
        "total_ideas": total_ideas,
        "total_selected": total_selected,
        "selection_rate": total_selected / total_ideas if total_ideas > 0 else 0,
        "technique_usage": technique_counts,
        "average_scores": {
            "novelty": sum(all_novelty) / len(all_novelty) if all_novelty else 0,
            "feasibility": sum(all_feasibility) / len(all_feasibility) if all_feasibility else 0,
            "diversity": sum(all_diversity) / len(all_diversity) if all_diversity else 0,
        },
        "sessions_by_status": {
            "active": len([s for s in sessions if s.status == "active"]),
            "completed": len([s for s in sessions if s.status == "completed"]),
        }
    }


def _print_session_insights(insights):
    """Print session insights in table format."""
    print(f"\n📊 Session Insights: {insights['session_id']}")
    print("-" * 60)
    print(f"Problem: {insights['problem']}")
    print(f"Status: {insights['status']}")
    print(f"Techniques: {', '.join(insights['techniques_used'])}")
    print("\n📈 Summary:")
    print(f"  Ideas generated: {insights['ideas_generated']}")
    print(f"  Ideas selected: {insights['ideas_selected']}")
    
    if 'technique_breakdown' in insights:
        print("\n🔧 Technique Breakdown:")
        for tech, count in sorted(insights['technique_breakdown'].items(), key=lambda x: -x[1]):
            print(f"  {tech}: {count} ideas")
    
    if 'quality_metrics' in insights:
        m = insights['quality_metrics']
        print("\n📊 Quality Metrics (All Ideas):")
        print(f"  Avg Novelty: {m['avg_novelty']:.2f}")
        print(f"  Avg Feasibility: {m['avg_feasibility']:.2f}")
        print(f"  Avg Impact: {m['avg_impact']:.2f}")
        
        if insights['ideas_selected'] > 0:
            print("\n⭐ Selected Ideas Quality:")
            print(f"  Avg Novelty: {m['selected_avg_novelty']:.2f}")
            print(f"  Avg Feasibility: {m['selected_avg_feasibility']:.2f}")
            print(f"  Avg Impact: {m['selected_avg_impact']:.2f}")
    
    if 'top_ideas' in insights:
        print("\n🏆 Top Ideas by Score:")
        for i, idea in enumerate(insights['top_ideas'], 1):
            print(f"  {i}. [{idea['technique']}] (Score: {idea['total_score']})")
            print(f"     {idea['description'][:60]}...")


def _print_global_insights(insights):
    """Print global insights in table format."""
    print("\n🌍 Global Innovation Insights")
    print("-" * 60)
    print("\n📊 Overview:")
    print(f"  Total sessions: {insights['total_sessions']}")
    print(f"  Total ideas: {insights['total_ideas']}")
    print(f"  Total selected: {insights['total_selected']}")
    print(f"  Selection rate: {insights['selection_rate']:.1%}")
    
    print("\n🔧 Technique Usage:")
    for tech, count in sorted(insights['technique_usage'].items(), key=lambda x: -x[1]):
        print(f"  {tech}: {count} sessions")
    
    print("\n📈 Average Scores:")
    scores = insights['average_scores']
    print(f"  Novelty: {scores['novelty']:.2f}")
    print(f"  Feasibility: {scores['feasibility']:.2f}")
    print(f"  Diversity: {scores['diversity']:.2f}")
    
    print("\n📋 Sessions by Status:")
    for status, count in insights['sessions_by_status'].items():
        print(f"  {status.capitalize()}: {count}")


def _handle_exit():
    log_json("INFO", "aura_cli_exit")


def _handle_migrate_credentials(args, config_manager=None):
    """
    Handle the credentials migration command.
    
    Security Issue #427: Migrate API keys from plaintext to secure storage.
    
    Args:
        args: CLI arguments
        config_manager: Optional ConfigManager instance
    """
    from core.config_manager import ConfigManager
    
    log_json("INFO", "credential_migration_command_invoked")
    
    # Get config manager
    if config_manager is None:
        config_manager = ConfigManager()
    
    # Check if credential store is available
    store_info = config_manager.get_credential_store_info()
    
    print("\n--- AURA Credentials Migration ---")
    print("Migrate API keys from plaintext config to secure storage.\n")
    
    # Show current storage status
    print("Storage Configuration:")
    print(f"  Keyring Available: {'✅ Yes' if store_info['keyring_available'] else '❌ No'}")
    print(f"  Fallback Available: {'✅ Yes' if store_info['fallback_available'] else '❌ No'}")
    print(f"  Fallback Path: {store_info['fallback_path']}")
    print()
    
    # Dry run first to show what would be migrated
    print("Scanning for credentials to migrate...")
    dry_run_results = config_manager.migrate_credentials(dry_run=True)
    
    if dry_run_results["migrated"]:
        print(f"\nFound {len(dry_run_results['migrated'])} credential(s) to migrate:")
        for key in dry_run_results["migrated"]:
            print(f"  - {key}")
    else:
        print("\nNo credentials found to migrate.")
    
    if dry_run_results["already_secure"]:
        print(f"\nAlready in secure storage: {', '.join(dry_run_results['already_secure'])}")
    
    # Confirm if not --yes flag
    execute_migration = getattr(args, "yes", False)
    
    if dry_run_results["migrated"] and not execute_migration:
        print()
        try:
            response = input("Proceed with migration? [y/N]: ").strip().lower()
            execute_migration = response in ("y", "yes")
        except EOFError:
            print("Migration cancelled (non-interactive mode). Use --yes to force.")
            return
    
    if not execute_migration or not dry_run_results["migrated"]:
        if not dry_run_results["migrated"]:
            print("\n✅ Nothing to migrate.")
        else:
            print("\n❌ Migration cancelled.")
        return
    
    # Execute migration
    print("\nMigrating credentials...")
    results = config_manager.migrate_credentials(dry_run=False)
    
    # Show results
    if results["migrated"]:
        print(f"\n✅ Successfully migrated {len(results['migrated'])} credential(s):")
        for key in results["migrated"]:
            print(f"  - {key}")
    
    if results["errors"]:
        print(f"\n❌ Errors ({len(results['errors'])}):")
        for key, error in results["errors"].items():
            print(f"  - {key}: {error}")
    
    print("\nMigration complete.")
    print("Your API keys are now stored securely.")
    print()
    print("Note: You can verify the migration with:")
    print("  python3 main.py config show --secure-status")


def _handle_secure_store(args, config_manager=None):
    """
    Handle the secure-store command for storing individual credentials.
    
    Args:
        args: CLI arguments
        config_manager: Optional ConfigManager instance
    """
    from core.config_manager import ConfigManager
    
    if config_manager is None:
        config_manager = ConfigManager()
    
    key = getattr(args, "key", None)
    value = getattr(args, "value", None)
    
    if not key:
        print("Error: --key is required")
        return
    
    # Interactive value input if not provided
    if not value:
        import getpass
        value = getpass.getpass(f"Enter value for {key}: ")
    
    if not value:
        print("Error: value is required")
        return
    
    success = config_manager.secure_store_credential(key, value)
    
    if success:
        print(f"✅ Credential '{key}' stored securely.")
        log_json("INFO", "credential_stored_via_cli", details={"key": key})
    else:
        print(f"❌ Failed to store credential '{key}'.")
        log_json("ERROR", "credential_store_via_cli_failed", details={"key": key})


def _handle_secure_delete(args, config_manager=None):
    """
    Handle the secure-delete command for removing credentials.
    
    Args:
        args: CLI arguments
        config_manager: Optional ConfigManager instance
    """
    from core.config_manager import ConfigManager
    
    if config_manager is None:
        config_manager = ConfigManager()
    
    key = getattr(args, "key", None)
    
    if not key:
        print("Error: --key is required")
        return
    
    # Confirm deletion
    if not getattr(args, "yes", False):
        try:
            response = input(f"Delete credential '{key}'? [y/N]: ").strip().lower()
            if response not in ("y", "yes"):
                print("Deletion cancelled.")
                return
        except EOFError:
            print("Deletion cancelled (non-interactive mode). Use --yes to force.")
            return
    
    success = config_manager.secure_delete_credential(key)
    
    if success:
        print(f"✅ Credential '{key}' deleted.")
        log_json("INFO", "credential_deleted_via_cli", details={"key": key})
    else:
        print(f"❌ Failed to delete credential '{key}' (may not exist).")
