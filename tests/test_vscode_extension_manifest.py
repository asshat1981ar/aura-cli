import json
import pathlib


def test_package_json_valid():
    pkg = json.loads((pathlib.Path("vscode-extension/package.json")).read_text())
    assert pkg["name"] == "aura-cli-vscode"
    assert "aura.runGoal" in [c["command"] for c in pkg["contributes"]["commands"]]
    assert "aura.status" in [c["command"] for c in pkg["contributes"]["commands"]]
    assert pkg["engines"]["vscode"]


def test_extension_ts_exists():
    assert pathlib.Path("vscode-extension/src/extension.ts").exists()


def test_readme_mentions_transport():
    readme = pathlib.Path("vscode-extension/README.md").read_text()
    assert "transport" in readme.lower()
    assert "JSON-RPC" in readme
