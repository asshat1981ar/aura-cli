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

    assert result == {"ready": [{"id": "bd-1", "title": "Fix tests"}]}


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

    assert result["ready"] == [{"id": "bd-1", "title": "Fix tests"}]
    assert result["returncode"] == 2
    assert result["stderr"] == "database locked"


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

    assert result == {"value": "ok"}
