"""Skill: match runtime errors against known patterns and suggest fixes."""
from __future__ import annotations
import re
from typing import Any, Dict, List

from agents.skills.base import SkillBase
from core.logging_utils import log_json

_PATTERNS: List[Dict] = [
    {
        "name": "import_error", "regex": re.compile(r"ImportError|ModuleNotFoundError", re.IGNORECASE),
        "suggested_fix": "Run `pip install <module_name>` or check PYTHONPATH.",
        "fix_steps": ["Identify the missing module from the error message", "Run: pip install <module>", "If in a venv, ensure it is activated"],
        "confidence_base": 0.90,
    },
    {
        "name": "attribute_error", "regex": re.compile(r"AttributeError", re.IGNORECASE),
        "suggested_fix": "Check spelling of the attribute. Use `dir(obj)` or inspect the class definition.",
        "fix_steps": ["Check the attribute name for typos", "Verify the object type with type(obj)", "Check if the attribute was renamed in a recent version"],
        "confidence_base": 0.80,
    },
    {
        "name": "type_error", "regex": re.compile(r"TypeError", re.IGNORECASE),
        "suggested_fix": "Check argument types. The function received an unexpected type.",
        "fix_steps": ["Read the full traceback to identify which function raised it", "Check the function signature", "Ensure you are passing the correct types"],
        "confidence_base": 0.75,
    },
    {
        "name": "key_error", "regex": re.compile(r"KeyError", re.IGNORECASE),
        "suggested_fix": "Use `.get(key, default)` instead of `dict[key]`, or check `key in dict` before access.",
        "fix_steps": ["Replace dict[key] with dict.get(key, fallback)", "Add a guard: `if key in d: ...`"],
        "confidence_base": 0.85,
    },
    {
        "name": "index_error", "regex": re.compile(r"IndexError|list index out of range", re.IGNORECASE),
        "suggested_fix": "Check list/array bounds before indexing. Use `len(lst)` to guard.",
        "fix_steps": ["Add a bounds check: `if i < len(lst):`", "Consider using `.get()` on dicts or iterating safely"],
        "confidence_base": 0.85,
    },
    {
        "name": "indentation_error", "regex": re.compile(r"IndentationError|unexpected indent", re.IGNORECASE),
        "suggested_fix": "Fix mixed tabs/spaces. Use 4 spaces per indent level consistently.",
        "fix_steps": ["Run: `python3 -m py_compile <file>` to locate the issue", "Set your editor to expand tabs to spaces", "Use `autopep8 --in-place <file>`"],
        "confidence_base": 0.95,
    },
    {
        "name": "name_error", "regex": re.compile(r"NameError.*not defined", re.IGNORECASE),
        "suggested_fix": "The variable or function is not defined in this scope. Check imports and spelling.",
        "fix_steps": ["Check for typos in the name", "Ensure the variable is defined before use", "Add the missing import"],
        "confidence_base": 0.85,
    },
    {
        "name": "value_error", "regex": re.compile(r"ValueError", re.IGNORECASE),
        "suggested_fix": "The value passed is of correct type but inappropriate. Add validation before the call.",
        "fix_steps": ["Add input validation", "Check what values are expected from the function's docstring"],
        "confidence_base": 0.70,
    },
    {
        "name": "file_not_found", "regex": re.compile(r"FileNotFoundError|No such file or directory", re.IGNORECASE),
        "suggested_fix": "The file path does not exist. Check the path and use Path.exists() to guard.",
        "fix_steps": ["Verify the file path is correct", "Use `Path(path).exists()` before opening", "Check working directory with `os.getcwd()`"],
        "confidence_base": 0.90,
    },
    {
        "name": "permission_error", "regex": re.compile(r"PermissionError", re.IGNORECASE),
        "suggested_fix": "Insufficient permissions. Check file/directory ownership and chmod.",
        "fix_steps": ["Run `ls -la` on the target path", "Use `chmod` or run with appropriate privileges"],
        "confidence_base": 0.80,
    },
    {
        "name": "recursion_error", "regex": re.compile(r"RecursionError|maximum recursion depth", re.IGNORECASE),
        "suggested_fix": "Add a base case or increase sys.setrecursionlimit(). Consider converting to iteration.",
        "fix_steps": ["Add or fix the base case", "Convert recursive function to iterative with a stack", "Temporarily: sys.setrecursionlimit(5000) to debug"],
        "confidence_base": 0.90,
    },
    {
        "name": "timeout_error", "regex": re.compile(r"TimeoutError|timed out", re.IGNORECASE),
        "suggested_fix": "Operation exceeded time limit. Increase timeout or optimize the slow operation.",
        "fix_steps": ["Increase the timeout parameter", "Profile the operation to find the bottleneck", "Add caching for repeated expensive calls"],
        "confidence_base": 0.75,
    },
]


def _similar_past(current_error: str, history: List[Dict]) -> List[Dict]:
    results = []
    for entry in history:
        past = entry.get("error", "")
        # Simple word overlap similarity
        cur_words = set(current_error.lower().split())
        past_words = set(past.lower().split())
        intersection = cur_words & past_words
        union = cur_words | past_words
        sim = len(intersection) / max(len(union), 1)
        if sim > 0.2:
            results.append({**entry, "similarity": round(sim, 2)})
    return sorted(results, key=lambda x: -x["similarity"])[:5]


class ErrorPatternMatcherSkill(SkillBase):
    name = "error_pattern_matcher"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        current_error: str = input_data.get("current_error", "")
        history: List[Dict] = input_data.get("error_history", [])

        matched = None
        for pattern in _PATTERNS:
            if pattern["regex"].search(current_error):
                matched = pattern
                break

        similar = _similar_past(current_error, history)

        if matched:
            fix = matched["suggested_fix"]
            confidence = matched["confidence_base"]
            if similar and similar[0].get("success"):
                fix = similar[0].get("fix", fix)
                confidence = min(0.98, confidence + 0.05)
            log_json("INFO", "error_pattern_matcher_matched", details={"pattern": matched["name"]})
            return {"matched_pattern": matched["name"], "suggested_fix": fix, "similar_past_errors": similar, "confidence": confidence, "fix_steps": matched["fix_steps"]}

        log_json("INFO", "error_pattern_matcher_no_match", details={"error_snippet": current_error[:80]})
        return {"matched_pattern": None, "suggested_fix": "No known pattern matched. Review the full traceback and search for the exact error message.", "similar_past_errors": similar, "confidence": 0.2, "fix_steps": ["Read the full traceback carefully", "Search the error message online", "Check recent code changes"]}
