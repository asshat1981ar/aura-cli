from dataclasses import dataclass
from pathlib import Path

class FileHandler:
    """Handles file operations safely."""

    @dataclass
    class FileData:
        path: Path
        content: str

    def __init__(self, file_path: str) -> None:
        """Initialize FileHandler with a valid file path."""
        self.file_data = self.FileData(Path(file_path), '')
        self.validate_file()  # Ensure the file is valid at initialization

    def validate_file(self) -> None:
        """Validate the file path and check if it exists."""
        if not self.file_data.path.is_file():
            raise FileNotFoundError(f"File '{self.file_data.path}' does not exist.")

    def read_file(self) -> str:
        """Read the content from the file and return it."""
        try:
            with self.file_data.path.open('r') as file:
                self.file_data.content = file.read()
        except IOError as e:
            raise IOError(f"Error reading file: {self.file_data.path}") from e
        return self.file_data.content

    def write_file(self, content: str) -> None:
        """Write the given content to the file."""
        try:
            with self.file_data.path.open('w') as file:
                file.write(content)
        except IOError as e:
            raise IOError(f"Error writing to file: {self.file_data.path}") from e
