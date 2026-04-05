import subprocess
from pathlib import Path

def run_linter_tools() -> None:
    """Run flake8 and mypy checks on the codebase."""
    # Define paths for flake8 and mypy
    flake8_path = Path('venv/bin/flake8')  # Adjust for your environment
    mypy_path = Path('venv/bin/mypy')  # Adjust for your environment

    # Run flake8
    flake8_result = subprocess.run([str(flake8_path), '--max-line-length=88', '.'],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print('flake8 output:', flake8_result.stdout.decode())
    if flake8_result.returncode != 0:
        print('flake8 failed with return code:', flake8_result.returncode)
        print('Errors:\n', flake8_result.stderr.decode())

    # Run mypy
    mypy_result = subprocess.run([str(mypy_path), '--ignore-missing-imports', '.'],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print('mypy output:', mypy_result.stdout.decode())
    if mypy_result.returncode != 0:
        print('mypy failed with return code:', mypy_result.returncode)
        print('Errors:\n', mypy_result.stderr.decode())

# Call the function to run linters
run_linter_tools()
