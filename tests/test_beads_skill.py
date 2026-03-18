from __future__ import annotations

import subprocess
from unittest.mock import patch

from agents.skills.beads_skill import BeadsSkill


def test_beads_skill_wraps_ready_list_in_dict():
    skill = BeadsSkill()
    proc = subprocess.CompletedProcess(
        args=["bd", "--json", "ready"],
        returncode=0,
        stdout='[{"id":"bd-1","title":"Fix tests"}]',
        stderr="",
    )

    with patch("agents.skills.beads_skill.subprocess.run", return_value=proc):
        result = skill.run({"cmd": "ready"})

    assert result.as_dict()["ready"] == [{"id": "bd-1", "title": "Fix tests"}]


def test_beads_skill_preserves_error_metadata_for_wrapped_list_payload():
    skill = BeadsSkill()
    proc = subprocess.CompletedProcess(
        args=["bd", "--json", "ready"],
        returncode=2,
        stdout='[{"id":"bd-1","title":"Fix tests"}]',
        stderr="database locked",
    )

    with patch("agents.skills.beads_skill.subprocess.run", return_value=proc):
        result = skill.run({"cmd": "ready"})

    data = result.as_dict()
    assert data["ready"] == [{"id": "bd-1", "title": "Fix tests"}]
    assert data["returncode"] == 2
    assert data["stderr"] == "database locked"


def test_beads_skill_wraps_scalar_json_in_dict():
    skill = BeadsSkill()
    proc = subprocess.CompletedProcess(
        args=["bd", "--json", "prime"],
        returncode=0,
        stdout='"ok"',
        stderr="",
    )

    with patch("agents.skills.beads_skill.subprocess.run", return_value=proc):
        result = skill.run({"cmd": "prime"})

    assert result.as_dict()["value"] == "ok"


def test_beads_skill_prefers_resolved_cli_path():
    skill = BeadsSkill()
    proc = subprocess.CompletedProcess(
        args=["/tmp/project/node_modules/.bin/bd", "--no-daemon", "ready", "--json"],
        returncode=0,
        stdout="[]",
        stderr="",
    )

    with patch("agents.skills.beads_skill.resolve_beads_cli", return_value="/tmp/project/node_modules/.bin/bd"), patch(
        "agents.skills.beads_skill.uses_repo_local_beads_cli",
        return_value=True,
    ), patch("agents.skills.beads_skill.subprocess.run", return_value=proc) as mock_run:
        skill.run({"cmd": "ready"})

    assert mock_run.call_args.args[0] == ["/tmp/project/node_modules/.bin/bd", "--no-daemon", "ready", "--json"]


def test_beads_skill_maps_missing_cli_to_capability_unavailable():
    skill = BeadsSkill()

    with patch("agents.skills.beads_skill.resolve_beads_cli", return_value="bd"), patch(
        "agents.skills.beads_skill.subprocess.run",
        side_effect=FileNotFoundError("bd not found"),
    ):
        result = skill.run({"cmd": "ready"})

    assert result.as_dict()["error"] == "capability_unavailable"
