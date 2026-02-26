"""Skill: extract and validate FastAPI/Flask endpoint contracts."""
from __future__ import annotations
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.skills.base import SkillBase
from core.logging_utils import log_json

_HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}
_ROUTE_DECORATORS = {"route", "get", "post", "put", "patch", "delete", "api_route"}


def _extract_string(node: ast.expr) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _parse_endpoints(source: str, file_path: str) -> List[Dict]:
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return []
    endpoints = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            path = None
            method = "unknown"
            if isinstance(dec, ast.Call):
                func = dec.func
                dec_name = ""
                if isinstance(func, ast.Attribute):
                    dec_name = func.attr
                elif isinstance(func, ast.Name):
                    dec_name = func.id
                if dec_name in _ROUTE_DECORATORS:
                    if dec.args:
                        path = _extract_string(dec.args[0])
                    if dec_name in _HTTP_METHODS:
                        method = dec_name.upper()
                    else:
                        for kw in dec.keywords:
                            if kw.arg == "methods" and isinstance(kw.value, (ast.List, ast.Tuple)):
                                for elt in kw.value.elts:
                                    s = _extract_string(elt)
                                    if s:
                                        method = s.upper()
                                        break
                    if path:
                        # Extract response_model kwarg
                        response_model = None
                        for kw in dec.keywords:
                            if kw.arg == "response_model":
                                try:
                                    response_model = ast.unparse(kw.value)
                                except Exception:
                                    pass
                        endpoints.append({"path": path, "method": method, "function": node.name, "line": node.lineno, "file": file_path, "response_model": response_model, "params": [a.arg for a in node.args.args if a.arg != "self"]})
    return endpoints


def _check_breaking_changes(old_spec: Dict, new_spec: Dict) -> List[Dict]:
    breaking = []
    old_eps = {(e["path"], e["method"]): e for e in old_spec.get("endpoints", [])}
    new_eps = {(e["path"], e["method"]): e for e in new_spec.get("endpoints", [])}
    for key, old_ep in old_eps.items():
        if key not in new_eps:
            breaking.append({"type": "removed_endpoint", "path": key[0], "method": key[1], "severity": "critical"})
        else:
            new_ep = new_eps[key]
            old_params = set(old_ep.get("params", []))
            new_params = set(new_ep.get("params", []))
            removed_params = old_params - new_params
            if removed_params:
                breaking.append({"type": "removed_params", "path": key[0], "method": key[1], "params": list(removed_params), "severity": "high"})
    return breaking


class APIContractValidatorSkill(SkillBase):
    name = "api_contract_validator"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        code: Optional[str] = input_data.get("code")
        file_path: str = input_data.get("file_path", "<string>")
        project_root_str: Optional[str] = input_data.get("project_root")
        old_spec: Optional[Dict] = input_data.get("old_spec")
        new_spec: Optional[Dict] = input_data.get("new_spec")

        endpoints: List[Dict] = []

        if code:
            endpoints = _parse_endpoints(code, file_path)
        elif project_root_str:
            root = Path(project_root_str)
            for f in root.rglob("*.py"):
                if ".git" in f.parts or "node_modules" in f.parts or "__pycache__" in f.parts:
                    continue
                try:
                    src = f.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                rel = str(f.relative_to(root))
                endpoints.extend(_parse_endpoints(src, rel))

        breaking: List[Dict] = []
        if old_spec and new_spec:
            breaking = _check_breaking_changes(old_spec, new_spec)
        elif old_spec and endpoints:
            breaking = _check_breaking_changes(old_spec, {"endpoints": endpoints})

        warnings: List[str] = []
        for ep in endpoints:
            if not ep.get("response_model"):
                warnings.append(f"No response_model on {ep['method']} {ep['path']}")

        compat_score = round(max(0.0, 1.0 - len(breaking) / max(len(endpoints), 1)), 2)

        log_json("INFO", "api_contract_validator_complete", details={"endpoints": len(endpoints), "breaking": len(breaking)})
        return {"endpoints": endpoints, "breaking_changes": breaking, "warnings": warnings[:50], "compatibility_score": compat_score, "endpoint_count": len(endpoints)}
