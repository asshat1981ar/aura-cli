from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import scripts.benchmark_retrieval as benchmark_retrieval


def test_benchmark_retrieval_honors_env_var_brain_db_path(tmp_path, monkeypatch):
    fake_brain = MagicMock()
    fake_vector_store = MagicMock()
    fake_vector_store.stats.return_value = {"record_count": 0}
    original_runtime_overrides = dict(benchmark_retrieval.config.runtime_overrides)

    monkeypatch.chdir(tmp_path)

    with patch.dict(
        os.environ,
        {
            "AURA_BRAIN_DB_PATH": "state/benchmark_brain.db",
            "OPENAI_API_KEY": "test-key",
        },
        clear=False,
    ):
        benchmark_retrieval.config.refresh()
        try:
            with patch.object(benchmark_retrieval, "Brain", return_value=fake_brain) as mock_brain_cls, \
                 patch.object(benchmark_retrieval, "ModelAdapter"), \
                 patch.object(benchmark_retrieval, "VectorStore", return_value=fake_vector_store), \
                 patch.object(benchmark_retrieval.console, "print"):
                with pytest.raises(SystemExit) as exc:
                    benchmark_retrieval.main()
        finally:
            benchmark_retrieval.config.runtime_overrides = original_runtime_overrides
            benchmark_retrieval.config.refresh()

    assert exc.value.code == 0
    mock_brain_cls.assert_called_once_with(
        db_path=str(Path(tmp_path) / "state" / "benchmark_brain.db")
    )
