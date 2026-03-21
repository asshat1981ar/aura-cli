import importlib.metadata
from typing import Any, Dict
from core.logging_utils import log_json

_PLUGIN_REGISTRY: Dict[str, Any] = {}

def register_skill(name: str, skill_class: Any) -> None:
    _PLUGIN_REGISTRY[name] = skill_class

def get_registered_skills() -> Dict[str, Any]:
    return _PLUGIN_REGISTRY

def discover_plugins(group: str = 'aura.skills') -> int:
    """Dynamically loads third-party AURA skills and agents via PyPI."""
    loaded_count = 0
    try:
        # Python 3.10+
        eps = importlib.metadata.entry_points(group=group)
    except TypeError:
        # Fallback for Python < 3.10
        eps = importlib.metadata.entry_points().get(group, [])

    for entry_point in eps:
        try:
            skill_class = entry_point.load()
            register_skill(entry_point.name, skill_class)
            log_json("info", "plugin_loaded", details={"name": entry_point.name, "group": group})
            loaded_count += 1
        except Exception as e:
            log_json("error", "plugin_load_failed", details={"name": entry_point.name, "error": str(e)})
    return loaded_count
