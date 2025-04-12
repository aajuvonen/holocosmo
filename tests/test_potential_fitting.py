import os
import subprocess
import sys
from pathlib import Path
import tempfile
import shutil
import pandas as pd

def test_potential_fitting_runs_on_generated_sample():
    """
    Integration test for potential_fitting.py using a generated sample dataset.
    Ensures the script runs end-to-end and produces output files.
    """
    root = Path(__file__).parents[1]
    script = root / "src/analysis/potential_fitting.py"
    assert script.exists(), f"Script not found: {script}"

    # Create temporary directory for data
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sample_file = tmp_path / "sample_radial_profile.csv"

        # Create test input data
        df = pd.DataFrame({
            "radial_bin_center": [1.0, 2.0, 3.0, 4.0, 5.0],
            "mean_laplacian": [0.05, 0.02, 0.01, 0.005, 0.0],
            "std_laplacian": [0.01, 0.01, 0.01, 0.002, 0.002],
            "count": [10, 15, 12, 8, 6]
        })
        df.to_csv(sample_file, index=False)

        # Run the potential fitting script
        result = subprocess.run(
            [
                sys.executable, str(script),
                "--input-file", str(sample_file),
                "--model", "yukawa",
                "--output-file", "test_potential_fit.png",
                "--output-dir", str(tmp_path)  # <-- tell it to save here
            ],
            capture_output=True,
            text=True,
            cwd=tmp_path
        )


        # Validate execution
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        assert "Fitted parameters" in result.stdout

        # Look for output files
        figs = list(tmp_path.glob("*potential_fit.png"))
        csvs = list(tmp_path.glob("*potential_fit_params.csv"))
        assert figs, "Potential fit figure was not generated"
        assert csvs, "Potential fit parameters CSV was not generated"

        print("Test passed. Outputs:", [f.name for f in figs + csvs])
