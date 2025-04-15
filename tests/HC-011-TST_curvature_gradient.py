# HC-011-TST: Curvature gradient
# Created on 2025-04-15T19:14:28.877452
import subprocess
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

def test_entanglement_curvature_gradient_runs(tmp_path):
    """
    Runs the entanglement_curvature_gradient.py script on mock data to verify it executes without error.
    """
    # Create a mock PEPS output CSV with minimal 4×4×4 lattice (64 rows)
    size = 4
    total_points = size**3
    df = pd.DataFrame({
        "x": [i % size for i in range(total_points)],
        "y": [(i // size) % size for i in range(total_points)],
        "z": [i // (size * size) for i in range(total_points)],
        "entropy": np.random.rand(total_points),
        "laplacian": np.random.randn(total_points)
    })

    input_csv = tmp_path / "mock_peps.csv"
    df.to_csv(input_csv, index=False)

    script_path = Path(__file__).parents[1] / "src/analysis/HC-011-ANA_curvature_gradient.py"
    result = subprocess.run([
        sys.executable, str(script_path),
        "--input", str(input_csv),
        "--lattice-size", str(size),
        "--slice-z", "2"
    ], capture_output=True, text=True)

    assert result.returncode == 0, f"Script failed:\n{result.stderr}"
