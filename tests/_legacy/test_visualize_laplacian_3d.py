import os
import subprocess
import sys
from pathlib import Path
import tempfile
import pandas as pd
import numpy as np

def test_visualize_laplacian_3d_runs_minimal():
    """
    Integration test for visualize_laplacian_3d.py.
    It generates synthetic cluster analysis data and checks that the script
    runs and produces a PNG image in a temp figures directory.
    """
    root = Path(__file__).parents[1]
    script = root / "src/analysis/visualize_laplacian_3d.py"
    assert script.exists(), f"Script not found: {script}"

    # Create temporary data and output folders
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        input_csv = tmp_path / "test_cluster_analysis.csv"
        output_dir = tmp_path / "figures"
        output_dir.mkdir()

        # Generate synthetic data
        N = 5000
        df = pd.DataFrame({
            "x": np.random.randint(0, 48, size=N),
            "y": np.random.randint(0, 48, size=N),
            "z": np.random.randint(0, 48, size=N),
            "laplacian": np.random.normal(0, 0.05, size=N),
            "cluster_label": np.random.randint(-1, 3, size=N)
        })
        df.to_csv(input_csv, index=False)

        # Run the visualization script
        proc = subprocess.run(
            [
                sys.executable, str(script),
                "--input", str(input_csv),
                "--threshold", "0.03",
                "--sample", "1000",
                "--output-dir", str(output_dir)
            ],
            capture_output=True,
            text=True,
            cwd=tmp_path
        )

        # Check script ran successfully
        assert proc.returncode == 0, f"Script failed:\n{proc.stderr}"
        assert "Figure saved to" in proc.stdout

        # Confirm figure created
        pngs = list(output_dir.glob("*.png"))
        assert pngs, "No PNG figure output was generated."

