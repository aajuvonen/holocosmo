import subprocess
import os
import glob

def test_simulation_runs_and_outputs_csv(tmp_path):
    output_dir = tmp_path / "test_outputs"
    os.makedirs(output_dir, exist_ok=True)

    result = subprocess.run([
        "python", "src/modeling/gravity_laplacian_simulation.py",
        "--lattice", "4", "4", "4",
        "--steps", "1",
        "--output-dir", str(output_dir)
    ], capture_output=True, text=True)

    assert result.returncode == 0, f"Script failed:\n{result.stderr}"
    
    # Find output file
    csv_files = list(output_dir.glob("*.csv"))
    assert csv_files, "No CSV file was created."

    # Check basic structure
    with open(csv_files[0], "r") as f:
        header = f.readline().strip()
    assert header == "x,y,z,entropy,laplacian", "CSV header malformed or missing."


