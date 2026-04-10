from pathlib import Path
from dataclasses import dataclass
from typing import List


@dataclass
class DataProcessor:
    data_path: Path

    def load_data(self) -> List[str]:
        """Load data from the specified data path, ensuring it exists and is readable."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path {self.data_path} does not exist.")
        if not self.data_path.is_file():
            raise ValueError(f"Path {self.data_path} is not a file.")
        # Load data logic here...
        return []

    def process_data(self, data: List[str]) -> List[str]:
        """Process the loaded data to extract meaningful insights."""
        # Processing logic (e.g., filtering, mapping) here...
        return data


# Example usage:
# processor = DataProcessor(data_path=Path('path/to/data.txt'))
# raw_data = processor.load_data()
# processed_data = processor.process_data(raw_data)
