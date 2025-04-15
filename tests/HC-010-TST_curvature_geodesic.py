# HC-010-TST: Curvature geodesic
# Created on 2025-04-15T19:07:16.081302
import subprocess
import sys
import numpy as np
import pandas as pd
from pathlib import Path

def test_entanglement_curvature_geodesic_runs(tmp_path):
    # Create small 6×6×6 curvature dataset
    size = 6
    total = size ** 3
    df = pd.DataFrame({
        "x": [i % size for i in range(total)],
        "y": [(i // size) % size for i in range(total)],
        "z": [i // (size * size) for i in range(total)],
        "laplacian": np.abs(np.random.randn(total))
    })
    input_csv = tmp_path / "mock_cluster.csv"
    df.to_csv(input_csv, index=False)

    script_path = Path(__file__).parents[1] / "src/analysis/HC-010-ANA_curvature_geodesic.py"
    result = subprocess.run([
        sys.executable, str(script_path),
        "--input", str(input_csv),
        "--start", "1", "1", "1",
        "--end", "4", "4", "4"
    ], capture_output=True, text=True)

    assert result.returncode == 0, f"Script failed:\n{result.stderr}"

