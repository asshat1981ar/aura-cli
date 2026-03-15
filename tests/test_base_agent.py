import pytest

from agents.base import Agent


def test_agent_cannot_instantiate_abstract_base():
    with pytest.raises(TypeError) as excinfo:
        Agent()

    assert "abstract" in str(excinfo.value).lower()


class _ConcreteAgent(Agent):
    def run(self, input_data):
        return {"result": input_data}


def test_agent_concrete_run_executes():
    agent = _ConcreteAgent()
    payload = {"foo": "bar"}

    assert agent.run(payload) == {"result": payload}
