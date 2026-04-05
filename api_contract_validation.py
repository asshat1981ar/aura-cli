from pydantic import BaseModel, ValidationError
import requests

# Define the expected User model from the API response
class User(BaseModel):
    id: int
    name: str
    email: str

# Function to fetch user data from the API
def fetch_user(user_id: int) -> User:
    '''Fetch a user by ID and return a validated User model.'''
    url = f'https://api.example.com/users/{user_id}'
    response = requests.get(url)
    response.raise_for_status()  # Handle HTTP errors
    try:
        user = User(**response.json())  # Validate and parse response
    except ValidationError as e:
        raise ValueError('Invalid user data received from API') from e
    return user

# Unit test using pytest
import pytest
from unittest.mock import patch

@patch('requests.get')
def test_fetch_user(mock_get):
    # Mock valid response data
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {'id': 1, 'name': 'John Doe', 'email': 'john@example.com'}
    user = fetch_user(1)
    assert user.id == 1
    assert user.name == 'John Doe'
    assert user.email == 'john@example.com'

    # Mock invalid response data
    mock_get.return_value.json.return_value = {'id': 'not-an-int', 'name': 'John Doe'}
    with pytest.raises(ValueError, match='Invalid user data received from API'):
        fetch_user(1)
