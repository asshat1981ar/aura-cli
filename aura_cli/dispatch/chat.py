import json
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

    history.append({"role": "user", "content": user_input})

    # Keep a small history window to prevent context overflow
    recent_history = history[-6:]

    prompt = f"""
You are AURA, an autonomous software engineering AI CLI assistant.
You are currently interacting with the user via a terminal REPL.

The user's latest input: "{user_input}"

You must respond with a JSON object detailing your intended action.
Your available actions are:
1. "reply": Use this if the user asks a question, needs an explanation, or greets you.
2. "add_goal": Use this if the user wants you to build, fix, refactor, or implement something. Provide the goal description.
3. "run": Use this if the user tells you to start working, execute the queue, or "go".

Respond ONLY with a valid JSON object in this exact format:
{{
  "action": "reply" | "add_goal" | "run",
  "text": "Your conversational reply to the user",
  "goal": "The technical goal description (required if action is add_goal, else null)"
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
            
    except Exception as e:
        log_json("WARN", "chat_agent_failed", details={"error": str(e)})
        print(f"\nAURA Chat Error: Could not process request. ({str(e)})")
        
    return "continue"
