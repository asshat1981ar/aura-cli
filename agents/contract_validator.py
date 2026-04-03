import json
import os
from typing import Dict, Any, List
from openapi_spec_validator import validate_spec
from openapi_spec_validator.readers import read_from_filename
from jsonschema import validate, ValidationError

class ContractValidator:
    def __init__(self, old_spec_path: str, new_spec_path: str):
        self.old_spec = self._load_spec(old_spec_path)
        self.new_spec = self._load_spec(new_spec_path)

    def _load_spec(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Spec file not found: {path}")
        spec_dict, _ = read_from_filename(path)
        return spec_dict

    def compare_specs(self) -> Dict[str, Any]:
        added_endpoints = []
        removed_endpoints = []
        modified_endpoints = []

        old_paths = set(self.old_spec.get('paths', {}).keys())
        new_paths = set(self.new_spec.get('paths', {}).keys())

        for path in new_paths - old_paths:
            added_endpoints.append(path)
        for path in old_paths - new_paths:
            removed_endpoints.append(path)
        for path in old_paths & new_paths:
            if self.old_spec['paths'][path] != self.new_spec['paths'][path]:
                modified_endpoints.append(path)

        return {
            'added_endpoints': added_endpoints,
            'removed_endpoints': removed_endpoints,
            'modified_endpoints': modified_endpoints
        }

    def validate_backward_compatibility(self) -> bool:
        comparison = self.compare_specs()
        return len(comparison['removed_endpoints']) == 0 and len(comparison['modified_endpoints']) == 0

    def validate_payload(self, payload: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        try:
            validate(instance=payload, schema=schema)
            return True
        except ValidationError as e:
            print(f"Payload validation error: {e.message}")
            return False

    def validate_schema_integrity(self, schema: Dict[str, Any]) -> bool:
        try:
            validate_spec(schema)
            return True
        except Exception as e:
            print(f"Schema validation error: {str(e)}")
            return False

    def validate_parameter_compatibility(self, old_params: List[Dict], new_params: List[Dict]) -> bool:
        old_param_map = {param['name']: param for param in old_params}
        new_param_map = {param['name']: param for param in new_params}

        for name, old_param in old_param_map.items():
            if name in new_param_map:
                new_param = new_param_map[name]
                if old_param.get('required', False) and not new_param.get('required', False):
                    print(f"Parameter {name} changed from required to optional")
                    return False
                if old_param.get('type') and new_param.get('type') and old_param.get('type') != new_param.get('type'):
                    print(f"Parameter {name} type changed from {old_param.get('type')} to {new_param.get('type')}")
                    return False
            elif old_param.get('required', False):
                print(f"Required parameter {name} removed")
                return False

        return True

def main():
    try:
        validator = ContractValidator('openapi_old.yaml', 'openapi_new.yaml')
    except FileNotFoundError as e:
        print(str(e))
        return
    
    # Validate schema integrity
    if not validator.validate_schema_integrity(validator.old_spec):
        print("Old spec validation failed")
        return
    if not validator.validate_schema_integrity(validator.new_spec):
        print("New spec validation failed")
        return

    comparison = validator.compare_specs()
    compatible = validator.validate_backward_compatibility()
    print(json.dumps({
        'comparison': comparison,
        'backward_compatible': compatible
    }, indent=2))

if __name__ == '__main__':
    main()
