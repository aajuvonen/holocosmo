# HC-003-TST: Entanglement curvature 3D
# Created on 2025-04-14T19:42:32.612677
import subprocess
import sys
from pathlib import Path

def test_entanglement_curvature_tensor_3d_demo_runs():
    """
    Smoke test: verifies that the 3D curvature demo runs and completes.
    """
    script_path = Path(__file__).parents[1] / "src/modeling/HC-003-MOD_entanglement_curvature_3d.py"
    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"
    