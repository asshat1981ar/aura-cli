"""Skill: auto-generate docstring templates and README sections from Python source."""
from __future__ import annotations
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.skills.base import SkillBase
from core.logging_utils import log_json


def _annotation_str(node: Optional[ast.expr]) -> str:
    if node is None:
        return "Any"
    try:
        return ast.unparse(node)
    except Exception:
        return "Any"


def _make_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = node.args
    params = []
    for arg in args.args:
        if arg.arg == "self":
            continue
        ann = _annotation_str(arg.annotation)
        params.append(f"    {arg.arg} ({ann}): TODO")
    ret = _annotation_str(node.returns) if node.returns else None
    lines = [f'"""TODO: describe {node.name}.\n']
    if params:
        lines.append("Args:")
        lines.extend(params)
        lines.append("")
    if ret and ret not in ("None", "Any"):
        lines.append("Returns:")
        lines.append(f"    {ret}: TODO")
        lines.append("")
    lines.append('"""')
    return "\n".join(lines)


def _analyze_source(source: str, file_path: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError as exc:
        return {"error": str(exc), "generated_docstrings": [], "undocumented_count": 0}

    missing = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not (isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)):
                missing.append({"name": node.name, "type": type(node).__name__.replace("Def", ""), "line": node.lineno, "template": _make_docstring(node) if hasattr(node, "args") else f'"""TODO: describe {node.name}."""'})

    module_docstring = ""
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        module_docstring = str(tree.body[0].value.value)

    public_api = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not n.name.startswith("_")]
    readme_section = f"## `{file_path}`\n\n"
    if module_docstring:
        readme_section += module_docstring.strip() + "\n\n"
    if public_api:
        readme_section += "**Public API:** " + ", ".join(f"`{n}`" for n in public_api[:10]) + "\n"

    return {"generated_docstrings": missing, "readme_section": readme_section, "undocumented_count": len(missing), "module_docstring": module_docstring}


class DocGeneratorSkill(SkillBase):
    name = "doc_generator"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        code: Optional[str] = input_data.get("code")
        file_path: str = input_data.get("file_path", "<string>")
        project_root_str: Optional[str] = input_data.get("project_root")

        if code:
            result = _analyze_source(code, file_path)
            log_json("INFO", "doc_generator_complete", details={"undocumented": result.get("undocumented_count", 0)})
            return result

        if project_root_str:
            root = Path(project_root_str)
            all_missing: List[Dict] = []
            all_readme: List[str] = []
            total_undocumented = 0
            for f in root.rglob("*.py"):
                if ".git" in f.parts or "__pycache__" in f.parts or "node_modules" in f.parts:
                    continue
                try:
                    src = f.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                rel = str(f.relative_to(root))
                r = _analyze_source(src, rel)
                all_missing.extend(r.get("generated_docstrings", []))
                if r.get("readme_section"):
                    all_readme.append(r["readme_section"])
                total_undocumented += r.get("undocumented_count", 0)
            log_json("INFO", "doc_generator_complete", details={"undocumented": total_undocumented})
            return {"generated_docstrings": all_missing[:100], "readme_section": "\n".join(all_readme[:10]), "undocumented_count": total_undocumented}

        return {"error": "Provide 'code' or 'project_root'"}
