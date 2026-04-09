from pydantic import BaseModel, ValidationError
import logging

def validate_api_input(data: dict) -> str:
    """Validates the input dict against the expected API schema."""
    try:
        input_model = ApiInputModel(**data)
    except ValidationError as e:
        logging.error(f'Validation error: {e}')
        raise ValueError('Invalid input data')
    return 'Validation passed'

class ApiInputModel(BaseModel):
    id: int
    name: str
    active: bool
    
# Add appropriate unit tests
import pytest

def test_validate_api_input_valid():
    data = {'id': 123, 'name': 'test', 'active': True}
    assert validate_api_input(data) == 'Validation passed'

def test_validate_api_input_invalid():
    data = {'id': 'not-an-int', 'name': 'test', 'active': True}
    with pytest.raises(ValueError, match='Invalid input data'):
        validate_api_input(data)