import difflib
import json
import re
import time
from pathlib import Path

from core.file_tools import (
    MISMATCH_OVERWRITE_BLOCK_EVENT,
    MismatchOverwriteBlockedError,
    OldCodeNotFoundError,
    apply_change_with_explicit_overwrite_policy,
    mismatch_overwrite_block_log_details,
)
from core.logging_utils import log_json
from core.task_manager import TaskManager, Task


_REPO_SCAN_SKIP_PARTS = {".git", "__pycache__", "node_modules", ".pytest_cache", "venv", ".venv"}
_TOKEN_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "after",
    "before",
    "path",
    "file",
    "invalid",
    "implement",
    "retry",
    "grounded",
    "validate",
    "fix",
    "this",
    "that",
    "goal",
    "loop",
    "code",
    "logic",
    "handler",
    "system",
    "current",
}
_SYMBOL_INDEX_CACHE_CANDIDATES = (
    "memory/symbol_index.json",
    "memory/symbol_indexer.json",
    "memory/symbol_map.json",
)
_TEXT_LIKE_CANDIDATE_SUFFIXES = {
    "",
    ".py",
    ".sh",
    ".md",
    ".txt",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".cfg",
}


def _check_project_writability(project_root: Path) -> bool:
    """
    Checks if the project directory is writable by attempting to create a temporary file.
    """
    try:
        test_file = project_root / ".aura_write_test"
        test_file.touch()
        test_file.unlink()
        return True
    except Exception as e:
        log_json("ERROR", "project_not_writable", details={"error": str(e), "path": str(project_root)})
        return False


def _goal_cycle_limit(args) -> int:
    """Return the per-goal cycle limit, honoring CLI overrides when present."""
    raw_limit = getattr(args, "max_cycles", None)
    if raw_limit is None:
        return 10
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        return 10
    return max(1, limit)


def _validate_change_target_path(project_root: Path, file_path: str) -> tuple[Path | None, str | None]:
    """Validate an IMPLEMENT change target path against repo boundaries and file existence."""
    if not file_path:
        return None, "missing_file_path"
    try:
        repo_root = project_root.resolve()
        target = (repo_root / file_path).resolve()
        target.relative_to(repo_root)
    except Exception:
        return None, "outside_project_root"
    if not target.is_file():
        return None, "file_not_found"
    return target, None


def _allow_new_test_file_target(
    project_root: Path,
    file_path: str,
    goal_text: str,
    old_code: object,
    overwrite_file: object,
) -> Path | None:
    """Allow safe creation of clearly test-focused new files under ``tests/``."""
    lower_goal = (goal_text or "").lower()
    if not any(token in lower_goal for token in ("test", "regression", "snapshot", "coverage")):
        return None

    if old_code not in ("", None) and not overwrite_file:
        return None

    try:
        repo_root = project_root.resolve()
        target = (repo_root / file_path).resolve()
        target.relative_to(repo_root)
    except Exception:
        return None

    relative = target.relative_to(repo_root)
    if not relative.parts or relative.parts[0] != "tests":
        return None
    if target.suffix.lower() != ".py":
        return None
    if not target.name.startswith("test_"):
        return None
    if target.exists():
        return target
    if not target.parent.is_dir():
        return None
    return target


def _tokenize_for_path_matching(text: str) -> list[str]:
    tokens = []
    for token in re.findall(r"[a-zA-Z0-9]+", (text or "").lower()):
        if len(token) < 2 or token in _TOKEN_STOPWORDS:
            continue
        if token == "py":
            continue
        tokens.append(token)
    return tokens


def _normalize_cached_candidate_path(project_root: Path, file_path: str) -> str | None:
    """Return a repo-relative existing path from cached data, or ``None`` if invalid/stale."""
    if not isinstance(file_path, str) or not file_path.strip():
        return None
    try:
        repo_root = project_root.resolve()
        candidate = (repo_root / file_path).resolve()
        candidate.relative_to(repo_root)
    except Exception:
        return None
    if not candidate.is_file():
        return None
    try:
        return str(candidate.relative_to(repo_root))
    except Exception:
        return None


def _candidate_files_from_symbol_index_cache(project_root: Path, query_tokens: list[str], limit: int = 6) -> list[str]:
    if not query_tokens:
        return []
    token_set = set(query_tokens)
    for rel_cache_path in _SYMBOL_INDEX_CACHE_CANDIDATES:
        cache_path = project_root / rel_cache_path
        if not cache_path.is_file():
            continue
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        candidates: list[str] = []
        seen: set[str] = set()

        name_index = payload.get("name_index") if isinstance(payload, dict) else None
        if isinstance(name_index, dict):
            for name, locations in name_index.items():
                name_tokens = set(_tokenize_for_path_matching(str(name)))
                if not (name_tokens & token_set):
                    continue
                if not isinstance(locations, list):
                    continue
                for loc in locations:
                    if not isinstance(loc, dict):
                        continue
                    normalized = _normalize_cached_candidate_path(project_root, loc.get("file"))
                    if normalized and normalized not in seen:
                        seen.add(normalized)
                        candidates.append(normalized)
                        if len(candidates) >= limit:
                            return candidates

        if candidates:
            return candidates
    return []


def _iter_repo_candidate_paths(project_root: Path):
    for path in project_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(project_root)
        rel_parts = set(rel.parts)
        if rel_parts & _REPO_SCAN_SKIP_PARTS:
            continue
        if path.suffix.lower() not in _TEXT_LIKE_CANDIDATE_SUFFIXES:
            continue
        yield rel


def _candidate_files_by_exact_name(project_root: Path, invalid_file_path: str, limit: int = 6) -> list[str]:
    target_name = Path(invalid_file_path or "").name
    if not target_name:
        return []

    matches: list[str] = []
    for rel in _iter_repo_candidate_paths(project_root):
        if rel.name != target_name:
            continue
        matches.append(str(rel))

    matches.sort(key=lambda item: (len(Path(item).parts), item))
    return matches[:limit]


def _candidate_files_from_repo_scan(project_root: Path, invalid_file_path: str, query_tokens: list[str], limit: int = 6) -> list[str]:
    token_set = set(query_tokens)
    target = Path(invalid_file_path or "")
    target_name = target.name
    target_stem = target.stem
    target_suffix = target.suffix.lower()
    scored: list[tuple[int, str]] = []
    all_paths: list[str] = []

    for rel in _iter_repo_candidate_paths(project_root):
        rel_str = str(rel)
        all_paths.append(rel_str)
        haystack_tokens = set(_tokenize_for_path_matching(rel_str))

        score = 0
        if token_set:
            score += len(token_set & haystack_tokens)
        if target_name and rel.name == target_name:
            score += 12
        if target_stem and rel.stem == target_stem:
            score += 6
        if target_suffix and rel.suffix.lower() == target_suffix:
            score += 2
        if target.parent.parts and rel.parts[: len(target.parent.parts)] == target.parent.parts:
            score += 1

        if score > 0:
            scored.append((score, rel_str))

    scored.sort(key=lambda item: (-item[0], len(Path(item[1]).parts), item[1]))
    ordered = [path for _, path in scored]

    if len(ordered) < limit and target_name:
        fuzzy = difflib.get_close_matches(target_name, all_paths, n=limit, cutoff=0.2)
        for path in fuzzy:
            if path not in ordered:
                ordered.append(path)
                if len(ordered) >= limit:
                    break

    return ordered[:limit]


def _candidate_existing_files(project_root: Path, invalid_file_path: str, goal_text: str, limit: int = 6) -> list[str]:
    path_tokens = _tokenize_for_path_matching(invalid_file_path)
    goal_tokens = _tokenize_for_path_matching(goal_text)

    # Mild domain bias for CLI/task-handler goals.
    lower_goal = (goal_text or "").lower()
    bias_tokens: list[str] = []
    if any(t in lower_goal for t in ("cli", "command", "arg", "help", "dispatch")):
        bias_tokens.extend(["aura", "cli"])
    if any(t in lower_goal for t in ("task", "loop", "goal", "queue", "core")):
        bias_tokens.extend(["core", "task", "goal", "loop"])
    if "test" in lower_goal:
        bias_tokens.extend(["tests", "test"])

    query_tokens: list[str] = []
    for token in path_tokens + goal_tokens + bias_tokens:
        if token not in query_tokens:
            query_tokens.append(token)

    seen: set[str] = set()
    results: list[str] = []

    for path in _candidate_files_by_exact_name(project_root, invalid_file_path, limit=limit):
        if path not in seen:
            seen.add(path)
            results.append(path)
            if len(results) >= limit:
                return results

    for path in _candidate_files_from_symbol_index_cache(project_root, query_tokens, limit=limit):
        if path not in seen:
            seen.add(path)
            results.append(path)
            if len(results) >= limit:
                return results

    for path in _candidate_files_from_repo_scan(project_root, invalid_file_path, query_tokens, limit=limit):
        if path not in seen:
            seen.add(path)
            results.append(path)
            if len(results) >= limit:
                return results

    return results


def _compose_loop_goal(task_title: str, grounding_hint: str | None) -> str:
    if not grounding_hint:
        return task_title
    return f"{task_title}\n\nGROUNDING_HINT:\n{grounding_hint}"


def _invalid_path_grounding_hint(file_path: str, reason: str, candidate_files: list[str]) -> str:
    base = (
        "Previous IMPLEMENT proposed an invalid file target "
        f"('{file_path}', reason: {reason}). "
        "Use an existing repository file path for the next IMPLEMENT and keep edits targeted."
    )
    if candidate_files:
        primary = candidate_files[0]
        return (
            f"{base}\nClosest existing match: {primary}\n"
            "Do not invent a new top-level directory when an exact filename match already exists.\n"
            "Candidate existing files (choose one if relevant):\n- " + "\n- ".join(candidate_files)
        )
    return base


def _mismatch_overwrite_blocked_grounding_hint(file_path: str) -> str:
    return (
        "Previous IMPLEMENT targeted an existing file but the provided old_code did not match the current file "
        f"('{file_path}'). The queue safety policy blocked automatic full-file overwrite fallback. "
        "For the next IMPLEMENT, either provide an exact current old_code snippet from the file, or if a full-file "
        "replacement is intentional, set overwrite_file to true, set old_code to an empty string, and provide the "
        "complete replacement content."
    )


def run_goals_loop(args, goal_queue, loop, debugger_instance, planner_instance, goal_archive, project_root, decompose=False):
    """
    Processes all goals in the queue using the hierarchical TaskManager.
    """
    task_manager = TaskManager()
    cycle_limit = _goal_cycle_limit(args)

    while goal_queue.has_goals():
        goal = goal_queue.next()
        log_json("INFO", "processing_goal", goal=goal)

        if decompose and planner_instance:
            root_task = task_manager.decompose_goal(goal, planner_instance)
            tasks_to_process = root_task.subtasks
        else:
            root_task = Task(id=f"goal_{int(time.time())}", title=goal)
            task_manager.add_task(root_task)
            tasks_to_process = [root_task]

        for task in tasks_to_process:
            if task != root_task:
                log_json("INFO", "executing_subtask", details={"task_id": task.id, "title": task.title})

            task.status = "in_progress"
            task_manager.save()

            converged = False
            loop.previous_score = 0
            loop.regression_count = 0
            loop.stable_convergence_count = 0

            cycle_count = 0
            changes_applied_successfully = True
            grounding_hint = None
            invalid_path_retry_used = False
            mismatch_overwrite_retry_used = False

            while not converged and changes_applied_successfully:
                cycle_count += 1
                log_json("INFO", "cycle_start", goal=task.title, details={"cycle_count": cycle_count})
                retry_after_invalid_target = False
                retry_after_mismatch_overwrite_blocked = False

                loop_goal = _compose_loop_goal(task.title, grounding_hint)
                loop_result_json_str = loop.run(loop_goal, dry_run=getattr(args, "dry_run", False))

                try:
                    result_json = json.loads(loop_result_json_str)
                except json.JSONDecodeError:
                    log_json("ERROR", "invalid_json_from_loop", goal=task.title)
                    break

                implement_data = result_json.get("IMPLEMENT")
                if implement_data:
                    changes_to_apply = []
                    if isinstance(implement_data, dict):
                        if all(k in implement_data for k in ["file_path", "old_code", "new_code"]):
                            changes_to_apply.append(implement_data)
                        elif "changes" in implement_data and isinstance(implement_data["changes"], list):
                            changes_to_apply.extend(implement_data["changes"])

                    for change in changes_to_apply:
                        file_path = change.get("file_path")
                        old_code = change.get("old_code")
                        new_code = change.get("new_code")
                        overwrite_file = change.get("overwrite_file", False)

                        if all([file_path is not None, old_code is not None, new_code is not None]):
                            full_target_path, invalid_reason = _validate_change_target_path(project_root, str(file_path))
                            if invalid_reason == "file_not_found":
                                allowed_new_test_target = _allow_new_test_file_target(
                                    project_root,
                                    str(file_path),
                                    task.title,
                                    old_code,
                                    overwrite_file,
                                )
                                if allowed_new_test_target is not None:
                                    full_target_path = allowed_new_test_target
                                    invalid_reason = None
                                    log_json(
                                        "INFO",
                                        "allowed_new_test_target",
                                        goal=task.title,
                                        details={"file": str(file_path)},
                                    )
                            if invalid_reason:
                                candidate_files = _candidate_existing_files(project_root, str(file_path), task.title, limit=6)
                                grounding_hint = _invalid_path_grounding_hint(str(file_path), invalid_reason, candidate_files)
                                log_json(
                                    "WARN",
                                    "invalid_implement_target_path",
                                    goal=task.title,
                                    details={
                                        "file": file_path,
                                        "reason": invalid_reason,
                                        "dry_run": bool(getattr(args, "dry_run", False)),
                                        "candidate_files": candidate_files,
                                        "retry_with_grounding_hint": (not invalid_path_retry_used and cycle_count < cycle_limit),
                                    },
                                )
                                if not invalid_path_retry_used and cycle_count < cycle_limit:
                                    invalid_path_retry_used = True
                                    retry_after_invalid_target = True
                                    log_json(
                                        "INFO",
                                        "grounding_retry_scheduled",
                                        goal=task.title,
                                        details={"reason": "invalid_implement_target_path", "next_cycle": cycle_count + 1},
                                    )
                                else:
                                    changes_applied_successfully = False
                                break

                            grounding_hint = None
                            if not getattr(args, "dry_run", False):
                                try:
                                    log_json("INFO", "applying_code_change", goal=task.title, details={"file": file_path})
                                    apply_change_with_explicit_overwrite_policy(
                                        project_root,
                                        str(file_path),
                                        old_code,
                                        new_code,
                                        overwrite_file=overwrite_file,
                                    )
                                except MismatchOverwriteBlockedError as e:
                                    log_json(
                                        "ERROR",
                                        MISMATCH_OVERWRITE_BLOCK_EVENT,
                                        goal=task.title,
                                        details=mismatch_overwrite_block_log_details(e, str(file_path)),
                                    )
                                    if not mismatch_overwrite_retry_used and cycle_count < cycle_limit:
                                        mismatch_overwrite_retry_used = True
                                        retry_after_mismatch_overwrite_blocked = True
                                        grounding_hint = _mismatch_overwrite_blocked_grounding_hint(str(file_path))
                                        log_json(
                                            "INFO",
                                            "grounding_retry_scheduled",
                                            goal=task.title,
                                                details={
                                                    "reason": MISMATCH_OVERWRITE_BLOCK_EVENT,
                                                    "next_cycle": cycle_count + 1,
                                                },
                                            )
                                    else:
                                        changes_applied_successfully = False
                                    break
                                except OldCodeNotFoundError as e:
                                    log_json(
                                        "ERROR",
                                        "old_code_not_found",
                                        goal=task.title,
                                        details={"error": str(e), "file": file_path},
                                    )
                                    changes_applied_successfully = False
                                    break
                                except Exception as e:
                                    log_json(
                                        "ERROR",
                                        "apply_change_failed",
                                        goal=task.title,
                                        details={"error": str(e), "file": file_path},
                                    )
                                    changes_applied_successfully = False
                                    break
                            else:
                                log_json("INFO", "replace_code_skipped", goal=task.title, details={"reason": "dry_run", "file": file_path})

                if retry_after_invalid_target or retry_after_mismatch_overwrite_blocked:
                    continue

                if "FINAL_STATUS" in result_json:
                    converged = True
                    task.status = "completed"
                    task.result = result_json["FINAL_STATUS"]
                    task_manager.save()
                    log_json(
                        "INFO",
                        "goal_completed" if task == root_task else "subtask_completed",
                        goal=task.title,
                        details={"status": result_json["FINAL_STATUS"]},
                    )

                if cycle_count >= cycle_limit:
                    log_json("WARN", "cycle_limit_reached", goal=task.title, details={"cycle_limit": cycle_limit})
                    break

            if not converged and not changes_applied_successfully:
                task.status = "failed"
                task_manager.save()
                log_json("WARN", "goal_terminated_without_convergence" if task == root_task else "subtask_failed", goal=task.title)
                break

        if decompose and planner_instance:
            if all(st.status == "completed" for st in root_task.subtasks):
                root_task.status = "completed"
                log_json("INFO", "goal_completed", goal=goal)
            else:
                root_task.status = "failed"
                log_json("WARN", "goal_failed", goal=goal)
            task_manager.save()

        final_score = loop.current_score if hasattr(loop, "current_score") else 0.0
        goal_archive.record(goal, final_score)
