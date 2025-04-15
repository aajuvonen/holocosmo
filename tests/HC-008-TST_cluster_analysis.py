# HC-008-TST: Cluster analysis
# Created on 2025-04-15T18:39:35.885562
import subprocess
import os
import sys
from pathlib import Path
import pandas as pd

def test_entanglement_cluster_analysis_runs(tmp_path):
    # Create a minimal test dataset
    data = pd.DataFrame({
        "x": [0, 1, 2, 3, 4, 5],
        "y": [0, 1, 2, 3, 4, 5],
        "z": [0, 1, 2, 3, 4, 5],
        "laplacian": [0.1, -0.2, 0.3, -0.4, 0.2, 0.0]
    })
    input_csv = tmp_path / "mock_peps.csv"
    data.to_csv(input_csv, index=False)

    output_dir = tmp_path / "results"
    os.makedirs(output_dir, exist_ok=True)

    script_path = Path(__file__).parents[1] / "src/analysis/HC-008-ANA_cluster_analysis.py"
    result = subprocess.run([
        sys.executable, str(script_path),
        "--input", str(input_csv),
        "--output-dir", str(output_dir),
        "--pairs", "100",  # small number for speed
        "--bins", "10"
    ], capture_output=True, text=True)

    assert result.returncode == 0, f"Script failed:\n{result.stderr}"

    output_files = list(output_dir.glob("*_cluster_analysis.csv"))
    correlation_files = list(output_dir.glob("*_spatial_correlation.csv"))

    assert output_files, "Cluster analysis file not generated."
    assert correlation_files, "Correlation file not generated."

    with open(output_files[0], "r") as f:
        header = f.readline().strip()
    assert "cluster_label" in header, "Missing 'cluster_label' in cluster CSV header."

