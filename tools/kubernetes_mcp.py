from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SomeFunctionInput:
    param1: str
    param2: int
    param3: str


def some_function(inputs: SomeFunctionInput) -> str:
    """Processes the inputs and returns a string result."""
    validate_inputs(inputs)
    return f"Processed: {inputs.param1}, {inputs.param2}, {inputs.param3}"


def validate_inputs(inputs: SomeFunctionInput) -> None:
    """Validates the input parameters."""
    if not isinstance(inputs.param1, str):
        raise ValueError("param1 must be a string")
    if not isinstance(inputs.param2, int):
        raise ValueError("param2 must be an integer")
    if not isinstance(inputs.param3, str):
        raise ValueError("param3 must be a string")
