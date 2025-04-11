import subprocess
import sys
from pathlib import Path

def test_entanglement_curvature_tensor_3d_demo_runs():
    """
    Smoke test: verifies that the 3D curvature demo runs and completes.
    """
    script_path = Path(__file__).parents[1] / "src/modeling/entanglement_curvature_tensor_3d_demo.py"
    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"

