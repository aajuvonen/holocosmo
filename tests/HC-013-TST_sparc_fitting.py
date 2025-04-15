# HC-013-TST: Sparc fitting
# Created on 2025-04-15T19:38:24.505819
import os
import subprocess
import sys
from pathlib import Path

def test_sparc_fitting_list_files():
    """
    Tests whether sparc_fitting.py can find and list .dat files.
    This is a dry run (non-interactive check only).
    """
    script = Path(__file__).parents[1] / "src/analysis/HC-013-ANA_sparc_fitting.py"
    env = os.environ.copy()

    proc = subprocess.run(
        [sys.executable, str(script)],
        input="999\n",  # simulate invalid input to exit cleanly
        capture_output=True,
        text=True
    )
    
    assert "No .dat files found" not in proc.stdout, "No .dat files found in data/sparc/"
    assert "Found the following .dat files" in proc.stdout, "File listing failed"
    assert proc.returncode == 0 or proc.returncode == 1
