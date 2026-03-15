import sys
from pathlib import Path
import pytest

# Add project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.eval_ascm import run_eval

def test_run_eval_script(capsys):
    run_eval()
    captured = capsys.readouterr()
    print(captured.out)
    assert "Final Score (MRR)" in captured.out
