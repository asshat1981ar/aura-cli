from pydantic import BaseModel
import pytest


class ApiResponse(BaseModel):
    success: bool
    data: dict
    error: str = None


@pytest.mark.parametrize(
    'input_data, expected_output', [
        ({'input': 'valid'}, {'success': True, 'data': {}}),
        ({'input': 'invalid'}, {'success': False, 'error': 'Invalid input'}),
    ]
)
def test_api_contract_validation(input_data: dict, expected_output: dict) -> None:
    """
    Tests the API response format against expected output.
    Handles valid and invalid inputs to validate the API contract.
    """
    pytest.skip("call_api not implemented — skeleton test")
