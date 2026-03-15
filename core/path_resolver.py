import difflib
import json
import re
from pathlib import Path

# Constants moved from task_handler.py
_REPO_SCAN_SKIP_PARTS = {".git", "__pycache__", "node_modules", ".pytest_cache", "venv", ".venv"}
_TOKEN_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "after", "before", "path", "file",
    "invalid", "implement", "retry", "grounded", "validate", "fix", "this", "that",
    "goal", "loop", "code", "logic", "handler", "system", "current",
}
_SYMBOL_INDEX_CACHE_CANDIDATES = (
    "memory/symbol_index.json",
    "memory/symbol_indexer.json",
    "memory/symbol_map.json",
)
_TEXT_LIKE_CANDIDATE_SUFFIXES = {
    "", ".py", ".sh", ".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".ini", ".cfg",
}


def check_project_writability(project_root: Path) -> bool:
    """
    Checks if the project directory is writable by attempting to create a temporary file.
    """
    try:
        test_file = project_root / ".aura_write_test"
        test_file.touch()
        test_file.unlink()
        return True
    except Exception:
        return False


def validate_change_target_path(project_root: Path, file_path: str) -> tuple[Path | None, str | None]:
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


def allow_new_test_file_target(
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


def find_candidate_existing_files(project_root: Path, invalid_file_path: str, goal_text: str, limit: int = 6) -> list[str]:
    """
    Suggests existing files when a file path is invalid, using fuzzy matching,
    token overlap with the goal, and symbol index lookups.
    """
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
