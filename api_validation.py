from typing import List

from pydantic import BaseModel, ValidationError
import requests

class User(BaseModel):
    id: int
    name: str
    email: str

class ApiClient:
    API_URL = 'https://api.example.com/users'

    @classmethod
    def fetch_users(cls) -> List[User]:
        response = requests.get(cls.API_URL)
        response.raise_for_status()  # Raises HTTPError if the status is 4xx or 5xx
        users_data = response.json()
        return [User(**user) for user in users_data]

if __name__ == '__main__':
    try:
        users = ApiClient.fetch_users()
        print(users)
    except requests.RequestException as e:
        print(f'Error fetching users: {e}')
    except ValidationError as e:
        print(f'Data validation error: {e}')
