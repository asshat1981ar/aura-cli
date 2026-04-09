import pytest

from pathlib import Path
from typing import List, Optional

class Goal:
    def __init__(self, title: str, status: str) -> None:
        self.title = title
        self.status = status

    def __repr__(self) -> str:
        return f"Goal(title={self.title}, status={self.status})"

def get_goals(filepath: Path) -> List[Goal]:
    """
    Load goals from a given file.

    :param filepath: Path to the goals file
    :return: List of Goal objects
    """
    if not filepath.is_file():
        raise FileNotFoundError(f'File {filepath} does not exist')

    goals = []
    with filepath.open() as file:
        for line in file:
            title, status = line.strip().split(',')
            goals.append(Goal(title.strip(), status.strip()))
    return goals

# Example of usage: Uncomment to use
# if __name__ == '__main__':
#     goals = get_goals(Path('path/to/goals.txt'))
#     print(goals)

@pytest.fixture
def sample_goals(tmp_path):
    goals_file = tmp_path / 'goals.txt'
    goals_file.write_text('Goal1, InProgress\nGoal2, Completed\n')
    return goals_file

def test_get_goals(sample_goals):
    goals = get_goals(sample_goals)
    assert len(goals) == 2
    assert goals[0].title == 'Goal1'
    assert goals[0].status == 'InProgress'
    assert goals[1].title == 'Goal2'
    assert goals[1].status == 'Completed'


def test_get_goals_file_not_found():
    with pytest.raises(FileNotFoundError):
        get_goals(Path('non_existing_file.txt'))
