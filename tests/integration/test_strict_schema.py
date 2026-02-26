from core.orchestrator import LoopOrchestrator
from core.policy import Policy
from memory.store import MemoryStore


class BadAgent:
    def run(self, input_data):
        return {"bad": True}


def test_strict_schema_stops(tmp_path):
    agents = {
        "ingest": BadAgent(),
        "plan": BadAgent(),
        "critique": BadAgent(),
        "synthesize": BadAgent(),
        "act": BadAgent(),
        "verify": BadAgent(),
        "reflect": BadAgent(),
    }
    orchestrator = LoopOrchestrator(
        agents=agents,
        memory_store=MemoryStore(tmp_path),
        policy=Policy.from_config({}),
        project_root=tmp_path,
        strict_schema=True,
    )

    result = orchestrator.run_loop("bad", max_cycles=1, dry_run=True)
    assert result["history"][0]["stop_reason"] == "INVALID_OUTPUT"
