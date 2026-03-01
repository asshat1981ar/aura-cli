from agents.sandbox import SandboxAgent
from core.logging_utils import log_json

class TesterAgent:
    """
    The TesterAgent is responsible for generating unit tests for given code
    and evaluating code by executing those tests in an isolated sandbox environment.
    It provides feedback on test outcomes and code correctness.
    """
    def __init__(self, brain, model, sandbox: SandboxAgent):
        """
        Initializes the TesterAgent with access to the system's brain, model, and sandbox.

        Args:
            brain: An instance of the system's memory (Brain).
            model: An instance of the model adapter for LLM interactions.
            sandbox (SandboxAgent): An instance of the SandboxAgent for executing code and tests.
        """
        self.brain = brain
        self.model = model
        self.sandbox = sandbox

    def generate_tests(self, code: str, context: str = "") -> str:
        """
        Prompts the LLM to generate comprehensive unit tests for the provided Python code.

        Args:
            code (str): The Python code for which tests need to be generated.
            context (str, optional): Additional context or requirements for test generation. Defaults to "".

        Returns:
            str: The LLM-generated Python unit tests.
        """
        prompt = f"""
        You are an autonomous testing agent. Your task is to generate comprehensive unit tests for the provided Python code.

        Code to test:
        ```python
        {code}
        ```

        Additional context:
        {context}

        Previous memory:
        {chr(10).join(self.brain.recall_with_budget(max_tokens=1500))}

        Generate Python unit tests for the provided code. Ensure all functions and edge cases are covered. Use the 'unittest' or 'pytest' framework.
        """
        response = self.model.respond(prompt)
        self.brain.remember(f"Generated tests for code: {code[:100]}...")
        self.brain.remember(response)
        return response

    def evaluate_code(self, code: str, tests: str) -> dict:
        """
        Executes the provided tests against the given code in a sandbox environment
        and returns a structured report of the execution results.

        Args:
            code (str): The Python code to be evaluated.
            tests (str): The Python unit tests to execute against the code.

        Returns:
            dict: A dictionary containing a summary of the test execution and
                  detailed actual output from the sandbox.
        """
        log_json("INFO", "tester_evaluating_code", details={"code_snippet": code[:100], "tests_snippet": tests[:100]})
        sandbox_result = self.sandbox.run_tests(code, tests)

        summary_parts = []
        if sandbox_result.passed:
            summary_parts.append("Tests PASSED.")
        else:
            summary_parts.append("Tests FAILED.")
        
        if sandbox_result.timed_out:
            summary_parts.append(f"Execution TIMED OUT after {self.sandbox.timeout}s.")
        
        if sandbox_result.metadata:
            summary_parts.append(f"Pytest summary: {sandbox_result.metadata.get('passed', 0)} passed, {sandbox_result.metadata.get('failed', 0)} failed, {sandbox_result.metadata.get('errors', 0)} errors.")

        full_summary = " ".join(summary_parts)

        return {
            "summary": full_summary,
            "actual_output": {
                "stdout": sandbox_result.stdout,
                "stderr": sandbox_result.stderr,
                "exit_code": sandbox_result.exit_code,
                "timed_out": sandbox_result.timed_out,
                "passed": sandbox_result.passed,
                "metadata": sandbox_result.metadata
            }
        }

