# HC-015-TST: Potential fitting
# Created on 2025-04-15T20:12:40.665329
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import shutil
import pandas as pd

def test_potential_fitting_runs_on_generated_sample():
    """
    Integration test for HC-015-ANA_potential_fitting.py using a generated sample dataset.
    Ensures the script runs end-to-end and produces output files in the appropriate directories.
    """
    # Setup temporary project structure
    root = Path(__file__).parents[1]
    # Copy the script to be tested
    script = root / "src" / "analysis" / "HC-015-ANA_potential_fitting.py"
    assert script.exists(), f"Script not found: {script}"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Create subdirectories mimicking project structure
        processed_dir = tmp_path / "data" / "processed"
        figures_dir = tmp_path / "data" / "figures"
        tests_dir = tmp_path / "tests"
        src_dir = tmp_path / "src" / "analysis"
        processed_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        tests_dir.mkdir(parents=True, exist_ok=True)
        src_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the script into the temporary src/analysis directory.
        tmp_script = src_dir / script.name
        shutil.copy(script, tmp_script)
        
        # Create a sample radial profile CSV in the tests directory.
        sample_file = tests_dir / "mock_radial_profile.csv"
        df = pd.DataFrame({
            "radial_bin_center": [1.0, 2.0, 3.0, 4.0, 5.0],
            "mean_laplacian": [0.05, 0.02, 0.01, 0.005, 0.0],
            "std_laplacian": [0.01, 0.01, 0.01, 0.002, 0.002],
            "count": [10, 15, 12, 8, 6]
        })
        df.to_csv(sample_file, index=False)
        
        # Set output base name for the potential fitting
        output_basename = "potential_fitting"
        
        # Set environment variable to use the non-interactive 'Agg' backend
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        
        # Run the potential fitting script from tmpdir, passing in our paths.
        result = subprocess.run(
        [
            sys.executable, str(tmp_script),
            "--input-file", str(sample_file),
            "--model", "yukawa",
            "--output-file", output_basename,
            "--output-dir", str(processed_dir),
            "--figure-dir", str(figures_dir)  # ← this was missing
        ],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env=env
        )
        
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        assert "Fitted parameters" in result.stdout, "No fitting output found in script output."
        
        # Locate the output CSV file in processed_dir.
        csv_files = list(processed_dir.glob("HC-015-ANA_*_potential_fitting.csv"))
        assert csv_files, "Potential fitting CSV was not generated."
        assert csv_files[-1].stat().st_size > 0, "Potential fitting CSV is empty."
        
        # Locate the output figure (SVG) — search the entire temp project tree
        fig_files = list(tmp_path.rglob("HC-015-ANA_*_potential_fitting.svg"))
        assert fig_files, "Potential fitting figure was not generated."
        assert fig_files[-1].stat().st_size > 0, "Potential fitting figure is empty."
        
        print("HC-015-ANA_potential_fitting.py integration test passed.")
        print("Generated files:", [f.name for f in csv_files + fig_files])
        
        # Cleanup: The TemporaryDirectory context will remove all files.

if __name__ == "__main__":
    test_potential_fitting_runs_on_generated_sample()
