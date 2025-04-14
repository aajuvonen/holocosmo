import subprocess
import sys
from pathlib import Path

def test_entanglement_peps_3d_demo_runs():
    """
    Smoke test: Verifies that the 3D PEPS demo script runs without errors.
    This test does not validate outputs or tensor values — just execution.
    """
    script_path = Path(__file__).parents[1] / "src/modeling/entanglement_peps_3d_demo.py"
    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"

