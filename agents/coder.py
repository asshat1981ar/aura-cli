import re
from rich import print
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
        code = ""
        tests = ""
        feedback = ""

        for i in range(self.MAX_ITERATIONS):
            prompt = f"""
You are an autonomous coding agent inside AURA.

Task:
{task}

Previous memory:
{self.brain.recall_all()}

{"Current code:\\n```python\\n" + code + "\\n```" if code else ""}
{"Tests:\\n```python\\n" + tests + "\\n```" if tests else ""}
{"Sandbox feedback:\\n" + feedback if feedback else ""}

CRITICAL: The first line of your code block MUST be: {self.AURA_TARGET_DIRECTIVE}<path/to/file.py>
Choose a path under agents/, core/, or memory/ that reflects the task.

Then produce the complete, working Python code.
Wrap everything in a single ```python ... ``` block.
"""
            # Generate or refine code
            raw_response = self.model.respond(prompt)
            # Extract code from response (assuming it's in a markdown code block)
            new_code_match = self.CODE_BLOCK_RE.search(raw_response)
            if new_code_match:
                code = new_code_match.group(1).strip()
            else:
                code = raw_response.strip() # Fallback if no code block

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
                return code # No tester, so return the first generated code

        log_json("ERROR", "coder_max_iterations_reached", goal=task, details={"max_iterations": max_iterations})
        self.brain.remember(f"Final code after max iterations for '{task}': {code}")
        return code
