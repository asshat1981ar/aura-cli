from pathlib import Path
from dataclasses import dataclass
from typing import List

@dataclass
class FileManager:
    base_path: Path

    def get_full_path(self, file_name: str) -> Path:
        """Returns the full path for a given file name."""
        return self.base_path / file_name

    def list_files(self) -> List[Path]:
        """Lists all files in the base path."""
        return [f for f in self.base_path.iterdir() if f.is_file()]

    def read_file(self, file_name: str) -> str:
        """Reads the content of a file and returns it as a string."""
        file_path = self.get_full_path(file_name)
        try:
            with file_path.open('r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            raise ValueError(f'File not found: {file_path}')
        except Exception as e:
            raise RuntimeError(f'Error reading file: {file_path}') from e

# Example usage
# fm = FileManager(Path('/path/to/directory'))
# print(fm.list_files())
# print(fm.read_file('example.txt'))
