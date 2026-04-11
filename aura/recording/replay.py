"""Recording replay engine."""

import asyncio
import re
from typing import Any, Callable, Dict, Optional

from .models import Recording, RecordingStep, ReplayResult, StepStatus


class VariableInterpolator:
    """Interpolate variables in strings."""

    @staticmethod
    def interpolate(text: str, variables: Dict[str, str]) -> str:
        """Replace ${var} with variable value."""

        def replace_var(match):
            var_name = match.group(1)
            return variables.get(var_name, match.group(0))

        return re.sub(r"\$\{(\w+)\}", replace_var, text)

    @staticmethod
    def interpolate_dict(data: Dict[str, Any], variables: Dict[str, str]) -> Dict[str, Any]:
        """Interpolate variables in dictionary values."""
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = VariableInterpolator.interpolate(value, variables)
            else:
                result[key] = value
        return result

    @staticmethod
    def interpolate_list(data: list, variables: Dict[str, str]) -> list:
        """Interpolate variables in list items."""
        result = []
        for item in data:
            if isinstance(item, str):
                result.append(VariableInterpolator.interpolate(item, variables))
            else:
                result.append(item)
        return result


class ReplayEngine:
    """Execute recordings with retry and condition support."""

    def __init__(self, handlers: Optional[Dict[str, Callable]] = None):
        self.handlers = handlers or {}

    def register_handler(self, command: str, handler: Callable):
        """Register a handler for a command."""
        self.handlers[command] = handler

    async def replay(
        self,
        recording: Recording,
        variables: Optional[Dict[str, str]] = None,
        stop_on_error: bool = True,
    ) -> ReplayResult:
        """Replay a recording."""
        result = ReplayResult(
            recording_name=recording.name,
            success=True,
            variables=variables or {},
        )

        # Merge recording variables with provided variables
        all_variables = {**recording.variables, **(variables or {})}

        for step in recording.steps:
            step_result = await self._execute_step(step, all_variables)
            result.step_results.append(step_result)

            if not step_result["success"]:
                result.success = False
                if stop_on_error:
                    break

        result.completed_at = asyncio.get_event_loop().time()
        return result

    async def _execute_step(
        self,
        step: RecordingStep,
        variables: Dict[str, str],
    ) -> Dict[str, Any]:
        """Execute a single step with retry logic."""
        # Check condition
        if step.condition:
            interpolated_condition = VariableInterpolator.interpolate(step.condition, variables)
            if not self._evaluate_condition(interpolated_condition):
                return {
                    "command": step.command,
                    "success": True,
                    "skipped": True,
                    "reason": f"Condition not met: {step.condition}",
                }

        handler = self.handlers.get(step.command)
        if not handler:
            return {
                "command": step.command,
                "success": False,
                "error": f"No handler for command: {step.command}",
            }

        # Interpolate variables
        args = VariableInterpolator.interpolate_list(step.args, variables)
        kwargs = VariableInterpolator.interpolate_dict(step.kwargs, variables)

        # Execute with retry
        last_error = None
        for attempt in range(step.retry_count):
            step.status = StepStatus.RUNNING
            step.started_at = asyncio.get_event_loop().time()

            try:
                if asyncio.iscoroutinefunction(handler):
                    output = await asyncio.wait_for(
                        handler(*args, **kwargs),
                        timeout=step.timeout,
                    )
                else:
                    loop = asyncio.get_event_loop()
                    output = await asyncio.wait_for(
                        loop.run_in_executor(None, handler, *args, **kwargs),
                        timeout=step.timeout,
                    )

                step.status = StepStatus.SUCCESS
                step.completed_at = asyncio.get_event_loop().time()
                step.output = output

                return {
                    "command": step.command,
                    "success": True,
                    "output": output,
                    "attempts": attempt + 1,
                }

            except Exception as e:
                last_error = str(e)
                step.status = StepStatus.FAILED
                step.error = last_error

                if attempt < step.retry_count - 1:
                    await asyncio.sleep(step.retry_delay)

        return {
            "command": step.command,
            "success": False,
            "error": last_error,
            "attempts": step.retry_count,
        }

    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate a simple condition string."""
        # For security, only allow simple comparisons
        # This is a simplified implementation
        try:
            # Support simple conditions like "var == value" or "var != value"
            parts = condition.split()
            if len(parts) == 3 and parts[1] in ("==", "!="):
                left, op, right = parts
                # Remove quotes if present
                right = right.strip("\"'")
                if op == "==":
                    return left == right
                else:
                    return left != right
            # Support variable existence check
            return bool(condition)
        except Exception:
            return False
