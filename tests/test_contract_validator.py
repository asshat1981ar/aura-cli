"""Unit tests for agents/contract_validator.py."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock


def _write_openapi(directory: str, filename: str, spec: dict) -> str:
    """Write an OpenAPI spec as JSON to a temp file and return its path."""
    path = os.path.join(directory, filename)
    with open(path, "w") as f:
        json.dump(spec, f)
    return path


_MINIMAL_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Test API", "version": "1.0.0"},
    "paths": {
        "/users": {"get": {"responses": {"200": {"description": "ok"}}}},
    },
}

_EXTENDED_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Test API", "version": "2.0.0"},
    "paths": {
        "/users": {"get": {"responses": {"200": {"description": "ok"}}}},
        "/orders": {"get": {"responses": {"200": {"description": "ok"}}}},
    },
}

_REDUCED_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Test API", "version": "2.0.0"},
    "paths": {},
}


class TestContractValidatorInit(unittest.TestCase):
    """Tests for __init__ and _load_spec."""

    def test_raises_file_not_found_for_missing_spec(self):
        from agents.contract_validator import ContractValidator
        with self.assertRaises(FileNotFoundError):
            ContractValidator("/nonexistent/old.yaml", "/nonexistent/new.yaml")

    def test_loads_both_specs(self):
        from agents.contract_validator import ContractValidator
        with tempfile.TemporaryDirectory() as tmpdir:
            old_path = _write_openapi(tmpdir, "old.json", _MINIMAL_SPEC)
            new_path = _write_openapi(tmpdir, "new.json", _EXTENDED_SPEC)

            with patch("agents.contract_validator.read_from_filename") as mock_read:
                mock_read.side_effect = [
                    (_MINIMAL_SPEC, None),
                    (_EXTENDED_SPEC, None),
                ]
                v = ContractValidator(old_path, new_path)

        self.assertIsNotNone(v.old_spec)
        self.assertIsNotNone(v.new_spec)


class TestCompareSpecs(unittest.TestCase):
    """Tests for compare_specs()."""

    def _make_validator(self, old_spec, new_spec):
        from agents.contract_validator import ContractValidator
        with tempfile.TemporaryDirectory() as tmpdir:
            old_path = _write_openapi(tmpdir, "old.json", old_spec)
            new_path = _write_openapi(tmpdir, "new.json", new_spec)
            with patch("agents.contract_validator.read_from_filename") as mock_read:
                mock_read.side_effect = [(old_spec, None), (new_spec, None)]
                return ContractValidator(old_path, new_path)

    def test_added_endpoint_detected(self):
        v = self._make_validator(_MINIMAL_SPEC, _EXTENDED_SPEC)
        result = v.compare_specs()
        self.assertIn("/orders", result["added_endpoints"])
        self.assertEqual(result["removed_endpoints"], [])

    def test_removed_endpoint_detected(self):
        v = self._make_validator(_EXTENDED_SPEC, _MINIMAL_SPEC)
        result = v.compare_specs()
        self.assertIn("/orders", result["removed_endpoints"])
        self.assertEqual(result["added_endpoints"], [])

    def test_identical_specs_no_changes(self):
        v = self._make_validator(_MINIMAL_SPEC, _MINIMAL_SPEC)
        result = v.compare_specs()
        self.assertEqual(result["added_endpoints"], [])
        self.assertEqual(result["removed_endpoints"], [])
        self.assertEqual(result["modified_endpoints"], [])

    def test_compare_returns_dict_with_required_keys(self):
        v = self._make_validator(_MINIMAL_SPEC, _EXTENDED_SPEC)
        result = v.compare_specs()
        for key in ("added_endpoints", "removed_endpoints", "modified_endpoints"):
            self.assertIn(key, result)


class TestValidateBackwardCompatibility(unittest.TestCase):
    """Tests for validate_backward_compatibility()."""

    def _make_validator(self, old_spec, new_spec):
        from agents.contract_validator import ContractValidator
        with tempfile.TemporaryDirectory() as tmpdir:
            old_path = _write_openapi(tmpdir, "old.json", old_spec)
            new_path = _write_openapi(tmpdir, "new.json", new_spec)
            with patch("agents.contract_validator.read_from_filename") as mock_read:
                mock_read.side_effect = [(old_spec, None), (new_spec, None)]
                return ContractValidator(old_path, new_path)

    def test_adding_endpoint_is_backward_compatible(self):
        v = self._make_validator(_MINIMAL_SPEC, _EXTENDED_SPEC)
        self.assertTrue(v.validate_backward_compatibility())

    def test_removing_endpoint_breaks_compatibility(self):
        v = self._make_validator(_EXTENDED_SPEC, _MINIMAL_SPEC)
        self.assertFalse(v.validate_backward_compatibility())

    def test_identical_specs_are_compatible(self):
        v = self._make_validator(_MINIMAL_SPEC, _MINIMAL_SPEC)
        self.assertTrue(v.validate_backward_compatibility())


class TestValidatePayload(unittest.TestCase):
    """Tests for validate_payload()."""

    def _make_validator_with_same_spec(self, spec):
        from agents.contract_validator import ContractValidator
        with tempfile.TemporaryDirectory() as tmpdir:
            old_path = _write_openapi(tmpdir, "old.json", spec)
            new_path = _write_openapi(tmpdir, "new.json", spec)
            with patch("agents.contract_validator.read_from_filename") as mock_read:
                mock_read.side_effect = [(spec, None), (spec, None)]
                return ContractValidator(old_path, new_path)

    def test_valid_payload_returns_true(self):
        v = self._make_validator_with_same_spec(_MINIMAL_SPEC)
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        self.assertTrue(v.validate_payload({"name": "Alice"}, schema))

    def test_invalid_payload_returns_false(self):
        v = self._make_validator_with_same_spec(_MINIMAL_SPEC)
        schema = {"type": "object", "properties": {"age": {"type": "integer"}}, "required": ["age"]}
        self.assertFalse(v.validate_payload({}, schema))


class TestValidateParameterCompatibility(unittest.TestCase):
    """Tests for validate_parameter_compatibility()."""

    def _make_validator_with_same_spec(self):
        from agents.contract_validator import ContractValidator
        with tempfile.TemporaryDirectory() as tmpdir:
            old_path = _write_openapi(tmpdir, "old.json", _MINIMAL_SPEC)
            new_path = _write_openapi(tmpdir, "new.json", _MINIMAL_SPEC)
            with patch("agents.contract_validator.read_from_filename") as mock_read:
                mock_read.side_effect = [(_MINIMAL_SPEC, None), (_MINIMAL_SPEC, None)]
                return ContractValidator(old_path, new_path)

    def test_compatible_params_returns_true(self):
        v = self._make_validator_with_same_spec()
        old = [{"name": "id", "type": "string", "required": True}]
        new = [{"name": "id", "type": "string", "required": True}]
        self.assertTrue(v.validate_parameter_compatibility(old, new))

    def test_required_param_removed_returns_false(self):
        v = self._make_validator_with_same_spec()
        old = [{"name": "id", "type": "string", "required": True}]
        new = []
        self.assertFalse(v.validate_parameter_compatibility(old, new))

    def test_type_change_returns_false(self):
        v = self._make_validator_with_same_spec()
        old = [{"name": "count", "type": "integer", "required": False}]
        new = [{"name": "count", "type": "string", "required": False}]
        self.assertFalse(v.validate_parameter_compatibility(old, new))

    def test_param_made_optional_when_was_required_returns_false(self):
        v = self._make_validator_with_same_spec()
        old = [{"name": "token", "type": "string", "required": True}]
        new = [{"name": "token", "type": "string", "required": False}]
        self.assertFalse(v.validate_parameter_compatibility(old, new))

    def test_new_optional_param_added_is_compatible(self):
        v = self._make_validator_with_same_spec()
        old = [{"name": "id", "type": "string", "required": True}]
        new = [
            {"name": "id", "type": "string", "required": True},
            {"name": "filter", "type": "string", "required": False},
        ]
        self.assertTrue(v.validate_parameter_compatibility(old, new))


if __name__ == "__main__":
    unittest.main()
