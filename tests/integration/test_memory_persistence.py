from pathlib import Path

from memory.store import MemoryStore


def test_memory_log_persistence(tmp_path: Path):
    store = MemoryStore(tmp_path)
    entry = {"cycle_id": "c1", "phase_outputs": {}, "stop_reason": None}
    store.append_log(entry)

    reread = store.read_log()
    assert reread and reread[-1]["cycle_id"] == "c1"
