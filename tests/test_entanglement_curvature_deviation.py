import subprocess
import sys
import numpy as np
import pandas as pd
from pathlib import Path

def test_entanglement_curvature_deviation_runs(tmp_path):
    # Small synthetic dataset (6×6×6)
    size = 6
    N = size ** 3
    df = pd.DataFrame({
        "x": [i % size for i in range(N)],
        "y": [(i // size) % size for i in range(N)],
        "z": [i // (size * size) for i in range(N)],
        "laplacian": np.abs(np.random.randn(N))
    })

    input_csv = tmp_path / "mock_cluster.csv"
    df.to_csv(input_csv, index=False)

    script_path = Path(__file__).parents[1] / "src/analysis/entanglement_curvature_deviation.py"
    result = subprocess.run([
        sys.executable, str(script_path),
        "--input", str(input_csv),
        "--start", "1", "1", "1",
        "--end", "4", "4", "4"
    ], capture_output=True, text=True)

    assert result.returncode == 0, f"Script failed:\n{result.stderr}"

