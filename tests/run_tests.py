import pathlib
import subprocess


def run_tests_with_coverage() -> None:
    """Run all tests and generate coverage report."""
    test_dir = pathlib.Path('tests')
    if not test_dir.exists():  # Validate test directory exists
        raise FileNotFoundError(f'Test directory {test_dir} does not exist.')
    
    try:
        result = subprocess.run(['pytest', '--cov=.', '--cov-fail-under=70', str(test_dir)],
                                check=True, text=True, capture_output=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Tests failed. Please investigate the following error:")
        print(e.stderr)
        raise
    except Exception as e:
        print(f'An error occurred: {str(e)}')
        raise

if __name__ == '__main__':
    run_tests_with_coverage()