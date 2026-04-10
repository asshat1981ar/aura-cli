from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class FileHandler:
    filepath: Path

    def read_lines(self) -> List[str]:
        try:
            with self.filepath.open("r") as file:
                return [line.strip() for line in file.readlines()]
        except IOError as e:
            raise FileNotFoundError(f"Cannot open {self.filepath}: {e}")

    def write_lines(self, lines: List[str]) -> None:
        try:
            with self.filepath.open("w") as file:
                file.writelines(f"{line}\n" for line in lines)
        except IOError as e:
            raise PermissionError(f"Cannot write to {self.filepath}: {e}")

    @staticmethod
    def validate_path(filepath: Path) -> None:
        if not filepath.exists():
            raise ValueError(f"The provided path does not exist: {filepath}")
