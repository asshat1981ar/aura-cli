import inspect
import re
import json
from core.logging_utils import log_json
from core.file_tools import _aura_safe_loads
from pydantic import ValidationError

try:
    from agents.schemas import CoderOutput
    from agents.prompt_manager import render_prompt, get_cached_prompt_stats
    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False


class CoderAgent:
    """
    The CoderAgent generates code with Chain-of-Thought reasoning
    and structured outputs for better reliability and maintainability.
    
    Uses role-based system prompts (Expert Python Developer) and prompt caching.
    """

    capabilities = ["code_generation", "coding", "implement", "refactor"]

    MAX_ITERATIONS = 3
    AURA_TARGET_DIRECTIVE = "# AURA_TARGET: "
    CODE_BLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)

    def __init__(self, brain, model, tester=None):
        self.brain = brain
        self.model = model
        self.tester = tester
        self.use_structured = SCHEMAS_AVAILABLE

    def _respond(self, prompt: str) -> str:
        try:
            inspect.getattr_static(self.model, "respond_for_role")
        except AttributeError:
            return self.model.respond(prompt)
        responder = getattr(self.model, "respond_for_role", None)
        if callable(responder):
            return responder("code_generation", prompt)
        return self.model.respond(prompt)

    def implement(self, task):
        """Generates code with CoT reasoning and structured output."""
        code = ""
        tests = ""
        feedback = ""
        best_output = None

        for i in range(self.MAX_ITERATIONS):
            memory_text = "\n".join(self.brain.recall_with_budget(max_tokens=2000))
            code_section = f"Current code:\n```python\n{code}\n```" if code else ""
            tests_section = f"Tests:\n```python\n{tests}\n```" if tests else ""
            feedback_section = f"Sandbox feedback:\n{feedback}" if feedback else ""

            if self.use_structured:
                result = self._implement_structured(
                    task, memory_text, code_section, tests_section, feedback_section
                )
            else:
                result = self._implement_legacy(
                    task, memory_text, code_section, tests_section, feedback_section
                )

            if result.get("error"):
                log_json("ERROR", "coder_iteration_error", details={
                    "iteration": i + 1, "error": result["error"]
                })
                if i == 0:
                    return f"# Error: {result['error']}"
                break

            code = result.get("code", "")
            best_output = result

            if self.tester:
                tests = self.tester.generate_tests(code, task)
                evaluation = self.tester.evaluate_code(code, tests)
                feedback = evaluation.get("summary", "")

                if "likely pass" in feedback.lower():
                    log_json("INFO", "coder_iteration_pass", details={
                        "iteration": i + 1,
                        "confidence": result.get("confidence", 0),
                        "target": result.get("aura_target", "unknown")
                    })
                    self._remember_output(task, result, tests)
                    return self._format_final_code(result)
                else:
                    log_json("WARN", "coder_iteration_feedback", details={
                        "iteration": i + 1, "feedback": feedback[:100]
                    })
            else:
                self._remember_output(task, result, tests)
                return self._format_final_code(result)

        log_json("ERROR", "coder_max_iterations", details={"max_iterations": self.MAX_ITERATIONS})
        if best_output:
            return self._format_final_code(best_output)
        return "# Error: Max iterations reached without valid code"

    def _implement_structured(self, task, memory, code_section, tests_section, feedback_section):
        """Generate code using structured output with CoT and role-based prompt."""
        prompt = render_prompt(
            template_name="coder",
            role="coder",
            params={
                "task": task,
                "memory": memory,
                "code_section": code_section,
                "tests_section": tests_section,
                "feedback_section": feedback_section
            }
        )

        response = self._respond(prompt)

        try:
            parsed = _aura_safe_loads(response, "coder_structured_response")
            coder_output = CoderOutput(**parsed)

            log_json("INFO", "coder_cot_reasoning", details={
                "problem_analysis": coder_output.problem_analysis[:100],
                "approach": coder_output.approach_selection[:100],
                "confidence": coder_output.confidence,
                "target": coder_output.aura_target
            })

            return {
                "structured_output": coder_output.dict(),
                "aura_target": coder_output.aura_target,
                "code": coder_output.code,
                "explanation": coder_output.explanation,
                "dependencies": coder_output.dependencies,
                "edge_cases": coder_output.edge_cases_handled,
                "confidence": coder_output.confidence,
                "reasoning": {
                    "problem_analysis": coder_output.problem_analysis,
                    "approach_selection": coder_output.approach_selection,
                    "design_considerations": coder_output.design_considerations,
                    "testing_strategy": coder_output.testing_strategy
                },
                "error": None
            }

        except (json.JSONDecodeError, ValidationError) as e:
            log_json("WARN", "coder_structured_parse_failed", details={"error": str(e)})
            return self._implement_legacy(task, memory, code_section, tests_section, feedback_section)
        except Exception as e:
            log_json("ERROR", "coder_structured_error", details={"error": str(e)})
            return {"error": str(e)}

    def _implement_legacy(self, task, memory, code_section, tests_section, feedback_section):
        """Fallback legacy implementation."""
        prompt = f"""You are an autonomous coding agent.

Task:
{task}

Previous memory:
{memory}

{code_section}
{tests_section}
{feedback_section}

Respond with JSON: {{"aura_target": "path/to/file.py", "code": "<python code>"}}"""

        response = self._respond(prompt)

        try:
            stripped = response.strip()
            brace_idx = stripped.find("{")
            if brace_idx != -1:
                obj = json.loads(stripped[brace_idx:stripped.rfind("}") + 1])
                if "aura_target" in obj and "code" in obj:
                    return {
                        "aura_target": obj["aura_target"],
                        "code": obj["code"],
                        "explanation": "",
                        "dependencies": [],
                        "confidence": 0.5,
                        "structured_output": None,
                        "error": None
                    }
        except Exception as e:
            log_json("WARN", "coder_legacy_parse_failed", details={"error": str(e)})

        # Last resort: extract from markdown
        target = None
        for line in response.splitlines():
            if line.startswith(self.AURA_TARGET_DIRECTIVE):
                target = line[len(self.AURA_TARGET_DIRECTIVE):].strip()
                break

        code_match = self.CODE_BLOCK_RE.search(response)
        code = code_match.group(1).strip() if code_match else response

        if target and not code.startswith(self.AURA_TARGET_DIRECTIVE):
            code = f"{self.AURA_TARGET_DIRECTIVE}{target}\n{code}"

        return {
            "aura_target": target or "unknown.py",
            "code": code,
            "explanation": "Extracted from legacy format",
            "confidence": 0.3,
            "structured_output": None,
            "error": None
        }

    def _format_final_code(self, result):
        """Format the final code output."""
        code = result.get("code", "")
        target = result.get("aura_target", "")

        if target and not code.startswith(self.AURA_TARGET_DIRECTIVE):
            code = f"{self.AURA_TARGET_DIRECTIVE}{target}\n{code}"

        return code

    def _remember_output(self, task, result, tests):
        """Store successful output in memory."""
        self.brain.remember(f"Code for '{task[:50]}...': {result.get('code', '')[:100]}...")
        if tests:
            self.brain.remember(f"Tests: {tests[:100]}...")

    def get_structured_info(self):
        """Get info about structured output availability."""
        return {
            "structured_output_available": self.use_structured,
            "schema_version": "1.0.0" if SCHEMAS_AVAILABLE else None
        }

    def get_cache_stats(self) -> dict:
        """Get prompt cache statistics."""
        if SCHEMAS_AVAILABLE:
            return get_cached_prompt_stats()
        return {"error": "Prompt manager not available"}
