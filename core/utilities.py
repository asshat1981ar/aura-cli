from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, ValidationError
from contextlib import contextmanager


class FileHandler(BaseModel):
    path: Path

    def read_file(self) -> str:
        if not self.path.exists():
            raise FileNotFoundError(f'File {self.path} does not exist.')
        with self.path.open('r') as file:
            return file.read()


@contextmanager
def temporary_file(file_path: Path, content: str):
    try:
        with file_path.open('w') as file:
            file.write(content)
        yield file_path
    finally:
        if file_path.exists():
            file_path.unlink()