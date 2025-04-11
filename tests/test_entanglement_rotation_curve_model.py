import subprocess
import sys
from pathlib import Path

def test_entanglement_rotation_curve_model_runs():
    """
    Smoke test for entanglement_rotation_curve_model.py.
    Ensures the script runs end-to-end and prints RMSE.
    """
    script_path = Path(__file__).parents[1] / "src/modeling/entanglement_rotation_curve_model.py"
    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
    
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"
    assert "RMSE of fit:" in result.stdout, "Expected RMSE output not found."

