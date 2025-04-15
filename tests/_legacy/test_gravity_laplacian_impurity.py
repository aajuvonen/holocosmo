import subprocess
import os
import sys
from pathlib import Path

def test_gravity_laplacian_impurity_runs(tmp_path):
    project_root = Path(__file__).parents[1]
    script_path = project_root / "src/modeling/gravity_laplacian_impurity.py"
    output_dir = tmp_path / "impurity_test"
    os.makedirs(output_dir, exist_ok=True)

    result = subprocess.run([
        sys.executable, str(script_path),
        "--lattice", "8", "8", "8",
        "--steps", "1",
        "--h-impurity", "2.5",
        "--output-dir", str(output_dir)
    ], capture_output=True, text=True)

    assert result.returncode == 0, f"Script failed:\n{result.stderr}"

    output_files = list(output_dir.glob("*.csv"))
    assert output_files, "No CSV file was generated."

    with open(output_files[0], "r") as f:
        header = f.readline().strip()
    assert header == "x,y,z,entropy,laplacian", "CSV header does not match expected format."
