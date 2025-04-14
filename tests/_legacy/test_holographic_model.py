import subprocess
import sys
from pathlib import Path
import glob

def test_holographic_model_runs():
    """
    Runs the holographic_model.py script and checks:
    - Successful execution
    - Output CSV and figures with timestamped names exist
    """
    root = Path(__file__).parents[1]
    script = root / "src/modeling/holographic_model.py"
    processed_dir = root / "data/processed"
    figures_dir = root / "data/figures"

    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, cwd=root
    )

    assert result.returncode == 0, f"Script failed:\n{result.stderr}"

    # Check for timestamped CSV
    csv_files = sorted(processed_dir.glob("*holographic_output.csv"))
    assert csv_files, "No output CSV generated."

    # Check for all 5 timestamped figures
    expected_prefixes = [
        "holographic_hubble_vs_lcdm",
        "holographic_weff_vs_time",
        "holographic_energy_components",
        "holographic_matter_to_vacuum_ratio",
        "holographic_weff_vs_z"
    ]
    for prefix in expected_prefixes:
        found = list(figures_dir.glob(f"*{prefix}.png"))
        assert found, f"Missing figure for: {prefix}"
