# HC-002-TST: Entanglement curvature 1D
# Created on 2025-04-14T19:35:45.004753

import subprocess
import sys
from pathlib import Path

def test_entanglement_curvature_tensor_demo_runs():
    """
    Basic sanity check: ensures script runs without error and generates plots.
    """
    script_path = Path(__file__).parents[1] / "src/modeling/HC-002-MOD_entanglement_curvature_1d.py"
    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"