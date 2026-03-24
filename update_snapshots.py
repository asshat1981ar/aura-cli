import json
from tests.cli_entrypoint_test_utils import run_main_subprocess
from tests.cli_snapshot_utils import snapshot_dir_for

SNAPSHOT_DIR = snapshot_dir_for("tests/test_cli_error_snapshots.py")

commands = {
    "cli_error_unknown_command": ["goa"],
    "cli_error_unknown_help_topic": ["help", "nope"],
    "cli_error_mixed_subcommand_legacy": ["--run-goals", "goal", "status"],
    "cli_error_conflicting_legacy_actions": ["--diag", "--status"],
    "cli_error_unrecognized_subcommand_argument": ["goal", "status", "--run-goals"],
    "cli_error_missing_goal_subcommand": ["goal"],
    "cli_error_missing_mcp_subcommand": ["mcp"],
    "cli_error_missing_workflow_subcommand": ["workflow"],
}

for name, args in commands.items():
    # txt
    proc = run_main_subprocess(*args)
    with open(SNAPSHOT_DIR / f"{name}.txt", "w") as f:
        f.write(proc.stderr)
    
    # json
    proc = run_main_subprocess(*args, "--json")
    with open(SNAPSHOT_DIR / f"{name}.json", "w") as f:
        f.write(json.dumps(json.loads(proc.stdout), indent=2, sort_keys=True) + "\n")

print("Snapshots updated.")
