import re
from core.logging_utils import log_json

class CoderAgent:
    """
    The CoderAgent is responsible for autonomously generating and refining code
    based on a given task, previous memory, and feedback from a TesterAgent.
    It operates in an iterative loop, attempting to produce working Python code
    that adheres to specified requirements and conventions.
    """

    MAX_ITERATIONS = 3
    AURA_TARGET_DIRECTIVE = "# AURA_TARGET: "
    CODE_BLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)

    def __init__(self, brain, model, tester=None):
        self.brain = brain
        self.model = model
        self.tester = tester

    def implement(self, task):
        """
        Generates and refines Python code based on a given task, iteratively improving
        it with feedback from a TesterAgent if available.

        The method constructs a detailed prompt for the LLM, including the task,
        relevant memory, current code, tests, and sandbox feedback. It extracts
        the generated code and, if a TesterAgent is provided, uses it to generate
        and evaluate tests. The process continues for MAX_ITERATIONS or until
        the code is deemed successful by the TesterAgent.

        Args:
            task (str): The specific task or problem the CoderAgent needs to implement.

        Returns:
            str: The final generated Python code after refinement, or the initial
                 generated code if no TesterAgent is provided.
        """
        import json as _json
        code = ""
        tests = ""
        feedback = ""

        for i in range(self.MAX_ITERATIONS):
            prompt = f"""
You are an autonomous coding agent inside AURA.

Task:
{task}

Previous memory:
{"\n".join(self.brain.recall_with_budget(max_tokens=2000))}

{"Current code:\\n```python\\n" + code + "\\n```" if code else ""}
{"Tests:\\n```python\\n" + tests + "\\n```" if tests else ""}
{"Sandbox feedback:\\n" + feedback if feedback else ""}

IMPORTANT: Respond with a single JSON object on one line (no markdown), like:
{{"aura_target": "path/to/file.py", "code": "<full python code>"}}

Choose a path under agents/, core/, or memory/ that reflects the task.
The "code" value must be a valid Python string (escape newlines as \\n).
"""
            raw_response = self.model.respond(prompt)

            # Try structured JSON first (preferred)
            parsed_target = None
            parsed_code = None
            stripped = raw_response.strip()
            # Find the first '{' to tolerate any leading whitespace/text
            brace_idx = stripped.find("{")
            if brace_idx != -1:
                try:
                    obj = _json.loads(stripped[brace_idx:stripped.rfind("}") + 1])
                    if "aura_target" in obj and "code" in obj:
                        parsed_target = obj["aura_target"]
                        parsed_code = obj["code"]
                except (ValueError, KeyError):
                    pass

            # Legacy fallback: # AURA_TARGET: directive + code block
            if parsed_code is None:
                target_match = None
                for line in stripped.splitlines():
                    if line.startswith(self.AURA_TARGET_DIRECTIVE):
                        parsed_target = line[len(self.AURA_TARGET_DIRECTIVE):].strip()
                        break
                code_match = self.CODE_BLOCK_RE.search(stripped)
                parsed_code = code_match.group(1).strip() if code_match else stripped

            code = parsed_code or ""
            if parsed_target:
                # Embed target into code as a comment so ActAdapter can extract it
                if not code.startswith(self.AURA_TARGET_DIRECTIVE):
                    code = f"{self.AURA_TARGET_DIRECTIVE}{parsed_target}\n{code}"

            if self.tester:
                tests = self.tester.generate_tests(code, task)
                evaluation = self.tester.evaluate_code(code, tests)
                feedback = evaluation["summary"]

                if "likely pass" in feedback.lower():
                    log_json("INFO", "coder_iteration_pass", goal=task, details={"iteration": i+1, "status": "Tests likely pass. Code accepted."})
                    self.brain.remember(f"Code for '{task}': {code}")
                    self.brain.remember(f"Tests for '{task}': {tests}")
                    return code
                else:
                    log_json("WARN", "coder_iteration_feedback", goal=task, details={"iteration": i+1, "status": "Tests likely fail or need improvement. Applying feedback..."})
                    self.brain.remember(f"Attempt {i+1} for '{task}': {code} -> Feedback: {feedback}")
            else:
                self.brain.remember(f"Code for '{task}': {code}")
                return code

        log_json("ERROR", "coder_max_iterations_reached", goal=task, details={"max_iterations": self.MAX_ITERATIONS})
        self.brain.remember(f"Final code after max iterations for '{task}': {code}")
        return code
