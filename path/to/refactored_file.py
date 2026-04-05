from dataclasses import dataclass
from pathlib import Path

@dataclass
class FileHandler:
    file_path: Path

    def read_file(self) -> str:
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        with self.file_path.open('r') as file:
            return file.read()

    def write_file(self, content: str) -> None:
        with self.file_path.open('w') as file:
            file.write(content)

    @classmethod
    def from_string(cls, path_str: str) -> 'FileHandler':
        return cls(Path(path_str))

# Usage Example:
# handler = FileHandler.from_string('example.txt')
# content = handler.read_file()
# handler.write_file('New content')
