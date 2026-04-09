from pathlib import Path
from pydantic import BaseModel, ValidationError
from typing import List

class InputModel(BaseModel):
    name: str
    age: int
    email: str

    @classmethod
    def validate_email(cls, email: str) -> str:
        if not isinstance(email, str) or '@' not in email:
            raise ValueError('Invalid email format')
        return email


def process_input(data: dict) -> None:
    try:
        model = InputModel(**data)
        model.validate_email(model.email)
        # Process the validated data
        print(f'Processing data for: {model.name}')
    except ValidationError as e:
        print(f'Validation error: {e.errors()}')
    except ValueError as ve:
        print(f'Value error: {ve}')