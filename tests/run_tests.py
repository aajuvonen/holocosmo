"""
run_tests.py

Runs all unit and integration tests in the `tests/` directory using pytest.
Place this script inside the `tests/` folder. Requires pytest to be installed.
"""

import subprocess
import sys
import os

def main():
    test_dir = os.path.dirname(__file__)
    print(f"🔍 Running tests in {test_dir} using pytest...\n")
    
    result = subprocess.run(["pytest", test_dir], text=True)

    if result.returncode == 0:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed.")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()