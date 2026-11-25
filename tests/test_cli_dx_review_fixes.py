"""Regression tests for the review findings on the Phase-1 CLI DX commands."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from aura_cli import dispatch as d


def _ctx(**kw):
    kw.setdefault("json", True)
    return d.DispatchContext(parsed=None, project_root=Path("."), runtime_factory=None, args=SimpleNamespace(**kw))


def _write(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


GOOD = {"name": "wf", "entry_point": "a", "nodes": [{"name": "a", "function": "fa"}, {"name": "b", "function": "fb"}],
        "edges": [{"source": "a", "target": "b"}, {"source": "b", "target": "__end__"}]}


def test_validate_json_mode_fails_on_invalid(tmp_path, capsys):
    p = _write(tmp_path, "bad.json", {})
    assert d._handle_workflow_validate_dispatch(_ctx(workflow_file=str(p))) == 1
    assert json.loads(capsys.readouterr().out)["valid"] is False


def test_validate_rejects_unknown_entry_point(tmp_path, capsys):
    bad = dict(GOOD, entry_point="missing")
    p = _write(tmp_path, "ref.json", bad)
    assert d._handle_workflow_validate_dispatch(_ctx(workflow_file=str(p))) == 1


def test_visualize_handles_conditional_edges(tmp_path, capsys):
    wf = dict(GOOD, edges=[{"source": "b", "target": "__end__"}],
              conditional_edges=[{"source": "a", "condition": "route", "mapping": {"go": "b"}}])
    p = _write(tmp_path, "cond.json", wf)
    assert d._handle_workflow_visualize_dispatch(_ctx(workflow_file=str(p))) == 0
    assert "mermaid" in json.loads(capsys.readouterr().out)


def test_validate_accepts_good_workflow(tmp_path, capsys):
    p = _write(tmp_path, "good.json", GOOD)
    assert d._handle_workflow_validate_dispatch(_ctx(workflow_file=str(p))) == 0


def test_stub_registry_includes_conditions():
    reg = d._stub_workflow_registry({"nodes": [{"name": "a", "function": "fa"}],
                                     "conditional_edges": [{"from": "a", "condition": "route", "mapping": {}}]})
    assert {"fa", "route"} <= set(reg)


def test_benchmark_reports_no_simulated_samples(tmp_path, capsys):
    p = _write(tmp_path, "wf.json", GOOD)
    rc = d._handle_agent_benchmark_dispatch(_ctx(pipeline_file=str(p), benchmark_suite="hotpotqa", benchmark_samples=5))
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["samples_executed"] == 0 and out["results"] == []
    assert out["success_rate"] is None and out["status"] == "compile_only"


def test_diff_keeps_empty_mappings(tmp_path, capsys):
    a = _write(tmp_path, "a.json", {"routing": {}})
    b = _write(tmp_path, "b.json", {})
    d._handle_agent_diff_dispatch(_ctx(config_a=str(a), config_b=str(b)))
    out = json.loads(capsys.readouterr().out)
    assert "routing" in json.dumps(out)


def test_explain_reads_decision_log(tmp_path, monkeypatch, capsys):
    store = tmp_path / "memory" / "store"
    store.mkdir(parents=True)
    (store / "decision_log.jsonl").write_text(
        json.dumps({"cycle_id": "c1", "phase_outputs": {"plan": {"x": 1}}}) + "\n"
        + json.dumps({"cycle_id": "c2", "phase_outputs": {"plan": {"x": 2}}}) + "\n")
    monkeypatch.chdir(tmp_path)
    d._handle_agent_explain_dispatch(_ctx(explain_trace="last-run"))
    assert json.loads(capsys.readouterr().out)["data"]["cycle_id"] == "c2"
    d._handle_agent_explain_dispatch(_ctx(explain_trace="c1"))
    assert json.loads(capsys.readouterr().out)["data"]["cycle_id"] == "c1"


def test_bash_completion_handles_python_script_word():
    assert "base=2" in d._BASH_COMPLETION_TEMPLATE
    assert '${{words[base]}}' in d._BASH_COMPLETION_TEMPLATE
