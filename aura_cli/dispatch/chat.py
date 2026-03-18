import json
from pathlib import Path
from core.logging_utils import log_json
from core.file_tools import _aura_safe_loads
from aura_cli.commands import _handle_add

def interactive_chat(runtime, user_input: str, history: list) -> str:
    """
    Handles natural language input by sending it to the LLM.
    Returns an action string ("continue", "run_goals", "exit") to the main loop.
    """
    model = runtime.get("model_adapter")
    if not model:
        print("Model adapter not available for chat. Ensure the system is properly initialized.")
        return "continue"

    project_root = runtime.get("project_root", Path("."))
    vector_store = runtime.get("vector_store")

    history.append({"role": "user", "content": user_input})
    
    max_iterations = 5
    iterations = 0

    while iterations < max_iterations:
        iterations += 1
        # Keep a small history window to prevent context overflow
        recent_history = history[-8:]

        prompt = f"""
You are AURA, an autonomous software engineering AI CLI assistant.
You are currently interacting with the user via a terminal REPL.

You have access to tools to understand the codebase before answering.
If you need to look at code to answer accurately, use "read_file" or "search".

Your available actions are:
1. "read_file": Read the contents of a specific file. Provide "path".
2. "search": Semantically search the codebase. Provide "query".
3. "reply": Use this to talk to the user. Provide "text".
4. "add_goal": Use this if the user wants you to build, fix, refactor, or implement something. Provide "goal" description, and optional conversational "text".
5. "run": Use this if the user tells you to start working, execute the queue, or "go". Provide optional "text".

Respond ONLY with a valid JSON object in this exact format:
{{
  "action": "reply" | "add_goal" | "run" | "read_file" | "search",
  "text": "Your conversational reply (for reply, add_goal, or run)",
  "goal": "The technical goal (only if action is add_goal)",
  "path": "File path (only if action is read_file)",
  "query": "Search query (only if action is search)"
}}

Chat History:
{json.dumps(recent_history, indent=2)}
"""
        try:
            response = model.respond(prompt)
            
            # Cleanup potential markdown block formatting
            if response.strip().startswith("```json"):
                response = response.strip()[7:]
            if response.strip().startswith("```"):
                response = response.strip()[3:]
            if response.strip().endswith("```"):
                response = response.strip()[:-3]
                
            data = json.loads(response.strip())
            action = data.get("action", "reply")
            
            if action == "read_file":
                path = data.get("path")
                if not path:
                    history.append({"role": "system", "content": "Error: path not provided for read_file."})
                    continue
                
                target = (project_root / path).resolve()
                try:
                    if not str(target).startswith(str(project_root.resolve())):
                        content = f"Error: Cannot read outside project root ({path})"
                    elif target.exists() and target.is_file():
                        content = target.read_text(encoding="utf-8")
                        content = f"--- FILE: {path} ---\n{content}\n--- EOF ---"
                    else:
                        content = f"Error: File not found at {path}"
                except Exception as e:
                    content = f"Error reading file: {e}"
                
                print(f"[{action}: {path}]")
                history.append({"role": "assistant", "content": f"Action: read_file\nPath: {path}"})
                history.append({"role": "system", "content": content})
                continue
                
            elif action == "search":
                query = data.get("query")
                if not query:
                    history.append({"role": "system", "content": "Error: query not provided for search."})
                    continue
                
                print(f"[{action}: '{query}']")
                history.append({"role": "assistant", "content": f"Action: search\nQuery: {query}"})
                if vector_store:
                    try:
                        results = vector_store.search(query, limit=3)
                        content = f"--- SEARCH RESULTS ---\n" + "\n".join(results) if results else "No results found."
                    except Exception as e:
                        content = f"Error during search: {e}"
                else:
                    content = "Error: Vector store not available for semantic search."
                    
                history.append({"role": "system", "content": content})
                continue
                
            # Terminal actions
            text = data.get("text", "")
            if text:
                print(f"\nAURA: {text}")
                history.append({"role": "assistant", "content": text})
                
            if action == "add_goal":
                goal = data.get("goal")
                if goal:
                    print(f"[AURA Auto-Queuing Goal]: {goal}")
                    _handle_add(runtime["goal_queue"], f"add {goal}")
            elif action == "run":
                print("\n[AURA Auto-Running Goals]")
                return "run_goals"
                
            return "continue"
                
        except Exception as e:
            log_json("WARN", "chat_agent_failed", details={"error": str(e)})
            print(f"\nAURA Chat Error: Could not process request. ({str(e)})")
            return "continue"
            
    print("\nAURA Chat Error: Max iteration limit reached while using tools.")
    return "continue"
