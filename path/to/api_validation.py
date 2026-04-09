from pydantic import BaseModel, ValidationError
import pytest
from typing import List

class ApiResponse(BaseModel):
    status: str
    data: List[dict]
    message: str


def validate_api_response(response: dict) -> ApiResponse:
    """Validates the API response against the ApiResponse model."""
    return ApiResponse(**response)


def test_validate_api_response():
    """Tests for validate_api_response function."""
    valid_response = {'status': 'success', 'data': [], 'message': 'Operation successful.'}
    try:
        response = validate_api_response(valid_response)
        assert response.status == 'success'
        assert response.message == 'Operation successful.'
    except ValidationError as e:
        pytest.fail(f'Validation failed: {e}')

    invalid_response = {'status': 'error', 'message': 'Missing data.'}
    with pytest.raises(ValidationError):
        validate_api_response(invalid_response)