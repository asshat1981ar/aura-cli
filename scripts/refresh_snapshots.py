import json
import subprocess
from pathlib import Path

SNAPSHOT_DIR = Path("tests/snapshots")

def run_aura(*args):
    result = subprocess.run(
        ["python3", "main.py"] + list(args),
        capture_output=True,
        text=True
    )
    return result

def update_json_snapshot(name, *args):
    print(f"Updating {name}...")
    res = run_aura(*args)
    # Some commands might fail (exit 2) for error snapshots
    try:
        data = json.loads(res.stdout)
    except json.JSONDecodeError:
        try:
            data = json.loads(res.stderr)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON for {name}")
            print(f"STDOUT: {res.stdout}")
            print(f"STDERR: {res.stderr}")
            return

    (SNAPSHOT_DIR / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

def update_text_snapshot(name, *args):
    print(f"Updating {name}...")
    res = run_aura(*args)
    content = res.stderr if res.stderr else res.stdout
    (SNAPSHOT_DIR / name).write_text(content)

# Contract reports
update_json_snapshot("cli_contract_report.json", "contract-report")

# Error snapshots
update_json_snapshot("cli_error_unrecognized_subcommand_argument.json", "goal", "status", "--run-goals", "--json")
update_text_snapshot("cli_error_unrecognized_subcommand_argument.txt", "goal", "status", "--run-goals")

update_json_snapshot("cli_error_conflicting_legacy_actions.json", "--diag", "--status", "--json")
update_text_snapshot("cli_error_conflicting_legacy_actions.txt", "--diag", "--status")

update_json_snapshot("cli_error_mixed_subcommand_legacy.json", "--run-goals", "goal", "status", "--json")
update_text_snapshot("cli_error_mixed_subcommand_legacy.txt", "--run-goals", "goal", "status")

update_json_snapshot("cli_error_unknown_command.json", "goa", "--json")
update_text_snapshot("cli_error_unknown_command.txt", "goa")

print("Done!")
