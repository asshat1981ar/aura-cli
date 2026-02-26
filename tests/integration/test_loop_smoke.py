from pathlib import Path

from core.orchestrator import LoopOrchestrator
from core.policy import Policy
from memory.store import MemoryStore
from tests.fakes.fake_agents import make_fake_agents


def test_loop_smoke(tmp_path: Path):
    store = MemoryStore(tmp_path)
    policy = Policy(max_cycles=1)
    orchestrator = LoopOrchestrator(make_fake_agents(), store, policy=policy, project_root=tmp_path)

    result = orchestrator.run_loop("demo", max_cycles=1, dry_run=True)

    assert result["stop_reason"] == "PASS"
    assert result["history"], "Expected at least one cycle"
    phases = result["history"][0]["phase_outputs"]
    for phase in ["context", "plan", "critique", "task_bundle", "change_set", "verification", "reflection"]:
        assert phase in phases
