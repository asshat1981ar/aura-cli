from pathlib import Path
from types import SimpleNamespace

from core.orchestrator import LoopOrchestrator


class StubStore:
    def __init__(self, summaries=None, error: Exception | None = None):
        self._summaries = list(summaries or [])
        self._error = error
        self.queries: list[tuple[str, int]] = []

    def query(self, tier: str, limit: int = 100):
        self.queries.append((tier, limit))
        if self._error is not None:
            raise self._error
        return list(self._summaries)


def _make_orchestrator(tmp_path: Path, store) -> LoopOrchestrator:
    orchestrator = LoopOrchestrator(
        agents={},
        project_root=tmp_path,
    )
    orchestrator.memory_controller = SimpleNamespace(persistent_store=store)
    return orchestrator


def test_retrieve_hints_returns_top_ranked_summaries(tmp_path: Path):
    store = StubStore(
        [
            {"id": "old-fail", "goal": "fix auth bug", "status": "failure"},
            {
                "id": "mid-success",
                "goal": "refactor auth flow",
                "status": "success",
            },
            {"id": "mid-unrelated", "goal": "docs cleanup", "status": "success"},
            {
                "id": "new-success",
                "goal": "token auth refresh",
                "status": "success",
            },
        ]
    )
    orchestrator = _make_orchestrator(tmp_path, store)

    hints = orchestrator._retrieve_hints("auth", limit=3)

    assert [hint["id"] for hint in hints] == [
        "mid-success",
        "old-fail",
        "new-success",
    ]
    assert store.queries == [("cycle_summaries", 200)]


def test_retrieve_hints_returns_empty_when_store_query_fails(tmp_path: Path):
    orchestrator = _make_orchestrator(
        tmp_path,
        StubStore(error=RuntimeError("boom")),
    )

    assert orchestrator._retrieve_hints("auth") == []


def test_retrieve_hints_handles_zero_limit_and_low_data(tmp_path: Path):
    store = StubStore(
        [
            {"id": "one", "goal": "auth workflow", "status": "success"},
            {"id": "two", "goal": "auth docs", "status": "failure"},
        ]
    )
    orchestrator = _make_orchestrator(tmp_path, store)

    assert orchestrator._retrieve_hints("auth", limit=0) == []
    hints = orchestrator._retrieve_hints("auth", limit=5)

    assert [hint["id"] for hint in hints] == ["one", "two"]
