import pytest
from unittest.mock import patch, MagicMock
from core.plugin_loader import discover_plugins, register_skill, get_registered_skills, _PLUGIN_REGISTRY

@pytest.fixture(autouse=True)
def reset_registry():
    _PLUGIN_REGISTRY.clear()
    yield
    _PLUGIN_REGISTRY.clear()

def test_register_skill():
    class DummySkill: pass
    register_skill("dummy", DummySkill)
    assert get_registered_skills()["dummy"] == DummySkill

@patch("importlib.metadata.entry_points")
def test_discover_plugins(mock_entry_points):
    mock_ep1 = MagicMock()
    mock_ep1.name = "test_skill"
    mock_ep1.load.return_value = "TestSkillClass"
    mock_entry_points.return_value = [mock_ep1]
    
    loaded = discover_plugins()
    assert loaded == 1
    assert get_registered_skills()["test_skill"] == "TestSkillClass"

@patch("importlib.metadata.entry_points")
def test_discover_plugins_error_handling(mock_entry_points):
    mock_ep1 = MagicMock()
    mock_ep1.name = "broken_skill"
    mock_ep1.load.side_effect = ImportError("Module not found")
    mock_entry_points.return_value = [mock_ep1]
    
    loaded = discover_plugins()
    assert loaded == 0
    assert "broken_skill" not in get_registered_skills()
