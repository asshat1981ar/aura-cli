"""Skill: validate JSON schemas and discover Pydantic model definitions."""
from __future__ import annotations
import ast
from typing import Any, Dict, List, Optional

from agents.skills.base import SkillBase
from core.logging_utils import log_json


def _validate_manually(schema: Dict, instance: Any, path: str = "") -> List[str]:
    errors = []
    if not isinstance(schema, dict):
        return errors
    schema_type = schema.get("type")
    type_map = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict, "null": type(None)}
    if schema_type and schema_type in type_map:
        expected = type_map[schema_type]
        if not isinstance(instance, expected):
            errors.append(f"{path or 'root'}: expected {schema_type}, got {type(instance).__name__}")
    if isinstance(instance, dict):
        for req in schema.get("required", []):
            if req not in instance:
                errors.append(f"{path or 'root'}: missing required field '{req}'")
        for key, subschema in schema.get("properties", {}).items():
            if key in instance:
                errors.extend(_validate_manually(subschema, instance[key], f"{path}.{key}" if path else key))
    if isinstance(instance, list):
        item_schema = schema.get("items", {})
        for i, item in enumerate(instance):
            errors.extend(_validate_manually(item_schema, item, f"{path}[{i}]"))
    return errors


def _find_pydantic_models(source: str) -> List[Dict]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    models = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = [getattr(b, "id", getattr(b, "attr", "")) for b in node.bases]
            if any(b in ("BaseModel", "Model") for b in bases):
                fields = []
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        try:
                            ann = ast.unparse(item.annotation)
                        except Exception:
                            ann = "Any"
                        required = item.value is None
                        fields.append({"name": item.target.id, "type": ann, "required": required})
                models.append({"name": node.name, "line": node.lineno, "fields": fields})
    return models


class SchemaValidatorSkill(SkillBase):
    name = "schema_validator"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        schema: Optional[Dict] = input_data.get("schema")
        instance: Optional[Any] = input_data.get("instance")
        code: Optional[str] = input_data.get("code")

        errors: List[str] = []
        valid = True

        if schema is not None and instance is not None:
            # Try jsonschema first
            try:
                import jsonschema
                v = jsonschema.Draft7Validator(schema)
                errors = [e.message for e in v.iter_errors(instance)]
            except ImportError:
                errors = _validate_manually(schema, instance)
            valid = len(errors) == 0

        pydantic_models: List[Dict] = []
        if code:
            pydantic_models = _find_pydantic_models(code)

        warnings = []
        if schema and not schema.get("$schema"):
            warnings.append("No $schema declaration – consider adding draft version")

        log_json("INFO", "schema_validator_complete", details={"valid": valid, "errors": len(errors), "models": len(pydantic_models)})
        return {"valid": valid, "errors": errors, "warnings": warnings, "pydantic_models": pydantic_models}
