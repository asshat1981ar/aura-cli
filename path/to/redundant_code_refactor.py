from pathlib import Path
from typing import List

def process_files(file_paths: List[Path]) -> None:
    """
    Process a list of file paths, performing some operation on each.
    """
    for file_path in file_paths:
        if file_path.exists() and file_path.is_file():
            try:
                # Perform the intended operation
                pass  # Placeholder for actual processing
            except Exception as e:
                print(f'Error processing {file_path}: {e}')
        else:
            print(f'{file_path} is not a valid file')