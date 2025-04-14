# HC-004-TST: Tensorial entanglement curvature
# Created on 2025-04-14T20:16:07.642826
import subprocess
import sys
from pathlib import Path

def test_entanglement_peps_3d_demo_runs():
    """
    Smoke test: Verifies that the 3D PEPS demo script runs without errors.
    This test does not validate outputs or tensor values — just execution.
    """
    script_path = Path(__file__).parents[1] / "src/modeling/HC-004-MOD_tensorial_entanglement_curvature.py"
    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"
