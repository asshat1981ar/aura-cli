import importlib
import sys


def test_cli_imports_without_optional_dependencies(monkeypatch):
    missing = ["requests", "numpy", "git", "git.exc"]
    for name in missing:
        monkeypatch.setitem(sys.modules, name, None)

    for module_name in [
        "core.model_adapter",
        "core.vector_store",
        "core.memory_types",
        "core.git_tools",
        "aura_cli.cli_main",
    ]:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    cli_main = importlib.import_module("aura_cli.cli_main")
    assert cli_main is not None

    model_adapter = importlib.import_module("core.model_adapter")
    assert isinstance(model_adapter.requests, model_adapter._MissingPackage)
    assert isinstance(model_adapter.np, model_adapter._MissingPackage)
    model_adapter.ModelAdapter()

    vector_store = importlib.import_module("core.vector_store")
    assert isinstance(vector_store.np, vector_store._MissingPackage)

    memory_types = importlib.import_module("core.memory_types")
    assert isinstance(memory_types.np, memory_types._MissingPackage)

    git_tools = importlib.import_module("core.git_tools")
    assert git_tools.Repo is None
