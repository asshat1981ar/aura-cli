from pathlib import Path

from agents.applicator import ApplicatorAgent


class StubBrain:
    def __init__(self):
        self.messages = []

    def remember(self, text):
        self.messages.append(text)



def test_apply_success_with_explicit_target_path(tmp_path):
    brain = StubBrain()
    agent = ApplicatorAgent(brain, backup_dir=tmp_path / "backups")
    target = tmp_path / "explicit.py"
    llm_output = "```python\nx = 1\n```"

    result = agent.apply(llm_output, target_path=str(target))

    assert result.success is True
    assert result.target_path == str(target)
    assert result.backup_path is None
    assert target.read_text(encoding="utf-8") == "x = 1"
    assert brain.messages


def test_apply_detects_aura_target_directive(tmp_path):
    brain = StubBrain()
    agent = ApplicatorAgent(brain, backup_dir=tmp_path / "backups")
    target = tmp_path / "directive.py"
    llm_output = f"```python\n# AURA_TARGET: {target}\nvalue = 2\n```"

    result = agent.apply(llm_output)

    assert result.success is True
    assert result.target_path == str(target)
    assert result.backup_path is None
    assert target.read_text(encoding="utf-8").splitlines() == [
        f"# AURA_TARGET: {target}",
        "value = 2",
    ]


def test_apply_fails_without_code_block(tmp_path):
    brain = StubBrain()
    agent = ApplicatorAgent(brain, backup_dir=tmp_path / "backups")

    result = agent.apply("no fenced code here")

    assert result.success is False
    assert result.code is None
    assert "No ```python``` code block found" in result.error


def test_apply_prevents_overwrite_when_disallowed(tmp_path):
    brain = StubBrain()
    agent = ApplicatorAgent(brain, backup_dir=tmp_path / "backups")
    target = tmp_path / "existing.py"
    original = "original\n"
    target.write_text(original, encoding="utf-8")
    llm_output = "```python\nnew_content = True\n```"

    result = agent.apply(llm_output, target_path=str(target), allow_overwrite=False)

    assert result.success is False
    assert "allow_overwrite=False" in result.error
    assert result.backup_path is None
    assert target.read_text(encoding="utf-8") == original


def test_apply_restores_backup_on_write_error(tmp_path, monkeypatch):
    brain = StubBrain()
    backup_dir = tmp_path / "backups"
    agent = ApplicatorAgent(brain, backup_dir=backup_dir)
    target = tmp_path / "fail.py"
    original = "original content\n"
    target.write_text(original, encoding="utf-8")
    llm_output = "```python\nupdated = True\n```"

    original_write_text = Path.write_text

    def failing_write(self, data, encoding=None, errors=None):
        if self == target:
            raise OSError("disk full")
        return original_write_text(self, data, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "write_text", failing_write)

    result = agent.apply(llm_output, target_path=str(target))

    assert result.success is False
    assert result.backup_path is not None
    assert Path(result.backup_path).exists()
    assert target.read_text(encoding="utf-8") == original
    assert "Filesystem write failed" in result.error
    assert not brain.messages
