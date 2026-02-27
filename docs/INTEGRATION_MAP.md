# AURA Integration Map

This document provides an overview of the AURA CLI's architecture, key components, and their interactions.

## Module Overview

*   **`core/`**: Contains the fundamental logic of the AURA system.
    *   `closed_loop.py`: (Likely) Implements a closed-loop control mechanism.
    *   `evolution_loop.py`: (Likely) Handles evolutionary algorithms or iterative refinement.
    *   `file_tools.py`: Provides utilities for file system operations, including code replacement.
    *   `git_tools.py`: Encapsulates all Git-related operations (commit, rollback, stash).
    *   `goal_archive.py`: Manages the archiving of completed goals.
    *   `goal_queue.py`: Manages the queue of active goals using a JSON file for persistence.
    *   `hybrid_loop.py`: The core orchestrator that combines different loops and agents.
    *   `model_adapter.py`: Interface for interacting with the AI model.
    *   `vector_store.py`: (Likely) Handles vector embeddings for semantic search or similar.

*   **`memory/`**: Manages the system's persistent memory.
    *   `brain.py`: Implements a "Brain" that stores general memories, identified weaknesses, and (potentially) vector store data using an SQLite database.
    *   `brain.db`: SQLite database file for the `Brain` instance.
    *   `goal_queue.json`: JSON file used by `core/goal_queue.py` for persisting the goal queue.
    *   `goal_queue.db`: (Discrepancy noted: Not explicitly used by `core/goal_queue.py`'s current implementation, which uses JSON. May be a remnant or used by an unexamined component.)

*   **`agents/`**: Contains specialized agent modules that perform specific tasks within the AURA loop.
    *   `applicator.py`: (Likely) Applies changes or solutions.
    *   `coder.py`: (Likely) Generates or modifies code.
    *   `critic.py`: (Likely) Evaluates solutions or code.
    *   `debugger.py`: (Likely) Assists in debugging.
    *   `mutator.py`: (Likely) Introduces variations or mutations.
    *   `planner.py`: (Likely) Creates plans.
    *   `router.py`: (Likely) Directs control flow or tasks.
    *   `sandbox.py`: (Likely) Executes code in an isolated environment.
    *   `scaffolder.py`: (Likely) Generates boilerplate or initial structures.
    *   `tester.py`: (Likely) Runs tests.

*   **`tests/`**: Unit and integration tests for various modules.
    *   `core/test_goal_queue.py`: Tests the functionality of the `GoalQueue`.

## Entrypoints and Loop Invocation

The primary entrypoint for the AURA CLI is `main.py`.

The `main()` function in `main.py` performs the following:
1.  Initializes core components: `GoalQueue`, `GoalArchive`, `ModelAdapter`, `Brain`, and `GitTools`.
2.  Instantiates the `HybridClosedLoop`.
3.  Enters an infinite `while True` loop, waiting for user commands:
    *   `add <goal>`: Adds a new goal to the `GoalQueue`.
    *   `run`: Initiates the core iterative development loop. It dequeues goals from `GoalQueue` and processes them one by one.
    *   `exit`: Terminates the CLI.
    *   `status`: Displays the current state of the goal queue and completed goals.

The core iterative process is managed by `HybridClosedLoop.run(current_goal)`, which orchestrates the interaction between various agents and components.

## Model I/O and `replace_code` Application

*   **Model I/O**: The `core/model_adapter.py` module is responsible for handling communication with the AI model. It provides an interface for sending prompts to the model and receiving its responses. The `HybridClosedLoop.run()` method receives model responses, which are expected to be in JSON format.
*   **`replace_code`**: The actual code modification logic resides in `core/file_tools.py` within the `replace_code` function. This function takes `file_path`, `old_code`, `new_code`, and an optional `dry_run` flag.
    *   In `main.py`, after `HybridClosedLoop.run(current_goal)` returns its JSON result, the `IMPLEMENT` section of the JSON is parsed to extract `file_path`, `old_code`, and `new_code`.
    *   Currently, `main.py` *simulates* the code replacement by printing a message: `SIMULATING: replace('{file_path}', old_code, new_code)`. The actual call to `file_tools.replace_code` is not directly made in `main.py` based on the current code, implying that the agents within the `HybridClosedLoop` or a later stage are responsible for invoking `file_tools.replace_code` if modifications are to be persisted.

### Explicit Overwrite Safety Policy (Autonomous Apply Paths)

Autonomous apply paths in the repo (queue loop, orchestrator, hybrid loop, mutator, and atomic change application) use a stricter policy than raw `replace_code(...)` to avoid accidental full-file overwrites when an LLM supplies a stale `old_code` snippet.

*   **Blocked by default**: `overwrite_file=True` does **not** permit mismatch-overwrite fallback unless the change is expressed as an explicit full-file replacement.
*   **Allowed explicit full-file form**:
    *   `overwrite_file=True`
    *   `old_code=""` (empty string)
*   **Policy-block event**:
    *   `old_code_mismatch_overwrite_blocked`
    *   policy tag: `explicit_overwrite_file_required`

This behavior is centralized in `core/file_tools.py` through:

*   `allow_mismatch_overwrite_for_change(...)`
*   `apply_change_with_explicit_overwrite_policy(...)`
*   `MismatchOverwriteBlockedError`

## Persistence

AURA utilizes two primary mechanisms for persistence: SQLite databases and JSON files.

*   **SQLite Databases**:
    *   **`memory/brain.db`**: Used by the `memory.brain.Brain` class.
        *   **Tables**:
            *   `memory`: Stores general textual content.
            *   `weaknesses`: Records identified system weaknesses with descriptions and timestamps.
            *   `vector_store_data`: Stores content and their embeddings, likely for semantic search or retrieval.
*   **JSON Files**:
    *   **`memory/goal_queue.json`**: Used by `core/goal_queue.py` to store the list of active goals. The `GoalQueue` class serializes its `deque` object to this file.
    *   **`memory/goal_queue.db`**: (Discrepancy) A `.db` file with this name exists in the `memory/` directory, but the current `core/goal_queue.py` implementation uses `memory/goal_queue.json` for its persistence. This `.db` file may be a leftover from a previous implementation or used by an unexamined component.

## Git Operations

All Git operations are centralized in `core/git_tools.py`, which leverages the `gitpython` library (imported as `git`). The `GitTools` class provides the following functionalities:

*   **Initialization**: `GitTools(repo_path)` initializes a `git.Repo` object, detecting the repository root.
*   **`commit_all(message)`**: Stages all changes (including untracked files) and creates a new commit with the provided message.
*   **`rollback_last_commit()`**: Performs a hard reset to the previous commit (`HEAD~1`), effectively undoing the last commit.
*   **`stash(message)`**: Stashes current modifications, both staged and unstaged.
*   **`stash_pop()`**: Applies the most recently stashed changes and removes the stash entry.
*   **Error Handling**: Custom exception classes (`GitToolsError`, `GitRepoError`, etc.) are defined for robust error management during Git operations.

## Test Suite Overview

The AURA project utilizes `unittest` for its testing framework.

*   **`core/test_goal_queue.py`**: This is the primary discovered test file, focusing on the `GoalQueue` functionality.
    *   **`setUp` / `tearDown`**: Manages a temporary test database file (`test_goal_queue.db`) for isolated testing.
    *   **Assertions**:
        *   `test_init_db_and_load_empty`: Verifies that an empty queue is correctly initialized.
        *   `test_add_goal`: Checks if goals are added correctly to the queue and persisted.
        *   `test_next_goal`: Asserts that the `next()` method retrieves the correct goal and re-indexes remaining goals.
        *   `test_next_with_empty_queue`: Confirms appropriate behavior when `next()` is called on an empty queue.
        *   `test_persistence`: Verifies that goals are correctly loaded and saved across `GoalQueue` instances.
    *   **Discrepancy**: This test file tests a version of `GoalQueue` that uses a SQLite database (`db_path`), while the current `core/goal_queue.py` implementation uses a JSON file (`queue_path`). This test is either outdated or pertains to an alternative `GoalQueue` implementation.
