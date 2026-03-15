import pytest
from unittest.mock import patch, MagicMock
from core.workflow_models import WorkflowStep, RetryPolicy
from core.workflow_steps import wire_inputs, execute_step
from core.workflow_registry import WorkflowRegistry, WorkflowDefinition
from core.workflow_engine import WorkflowEngine


def test_wire_inputs():
    step = WorkflowStep(
        name="step2",
        inputs_from={"my_param": "step1.output_val", "other": "step1.other"},
        static_inputs={"static": 123}
    )
    step_outputs = {
        "step1": {"output_val": "hello", "other": "world"}
    }
    initial = {"root": "path"}
    
    result = wire_inputs(step, step_outputs, initial)
    assert result["my_param"] == "hello"
    assert result["other"] == "world"
    assert result["static"] == 123
    assert result["root"] == "path"


def test_execute_step_success():
    step = WorkflowStep(
        name="test_step",
        fn=lambda x: {"result": x["val"] * 2},
        static_inputs={"val": 21}
    )
    res = execute_step(step, {}, {})
    assert res.status == "ok"
    assert res.output["result"] == 42
    assert res.attempts == 1


def test_execute_step_retry_failure():
    # A function that always fails
    mock_fn = MagicMock(side_effect=Exception("Boom"))
    step = WorkflowStep(
        name="fail_step",
        fn=mock_fn,
        retry=RetryPolicy(max_attempts=2, backoff_base=0.01)
    )
    
    res = execute_step(step, {}, {})
    assert res.status == "failed"
    assert "Boom" in res.error
    assert res.attempts == 2
    assert mock_fn.call_count == 2


def test_workflow_registry():
    defn = WorkflowDefinition(name="my_flow", steps=[])
    WorkflowRegistry.register(defn)
    assert WorkflowRegistry.get("my_flow") == defn
    assert "my_flow" in WorkflowRegistry.list_all()


@patch("core.workflow_engine.execute_step")
def test_workflow_engine_run(mock_exec):
    # Mock successful step execution
    from core.workflow_models import StepResult
    mock_exec.return_value = StepResult(
        step_name="s1", status="ok", output={"o": 1}, attempts=1, elapsed_ms=10
    )
    
    engine = WorkflowEngine()
    # Register a test workflow
    step1 = WorkflowStep(name="s1", fn=lambda x: x)
    defn = WorkflowDefinition(name="test_flow", steps=[step1])
    WorkflowRegistry.register(defn)
    
    exec_id = engine.run_workflow("test_flow", {})
    
    # Access the execution object directly for this white-box test
    exc = engine._executions[exec_id]
    
    assert exc.status == "completed"
    assert len(exc.history) == 1
    assert exc.step_outputs["s1"] == {"o": 1}
    mock_exec.assert_called_once()
