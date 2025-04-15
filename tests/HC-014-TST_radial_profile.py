# HC-014-TST: Radial profile
# Created on 2025-04-15T19:58:03.818034
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

def test_radial_profile_output():
    """
    Integration test for HC-014-ANA_radial_profile.py:
    - Generates a small synthetic curvature CSV.
    - Runs the radial_profile script with known paths.
    - Confirms that the output file is generated and non-empty.
    """
    # Setup paths
    base_dir = Path(__file__).parents[1]
    script = base_dir / "src/analysis/HC-014-ANA_radial_profile.py"
    output_dir = base_dir / "data/processed"
    input_path = base_dir / "tests/mock_lattice.csv"
    output_basename = "HC-014-ANA_radial_profile.csv"  # Base name (the script adds timestamp and prefix)

    # Create mock lattice data (5x5x5 cube with a simple Laplacian field)
    records = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                entropy = 0.0
                laplacian = -((x - 2)**2 + (y - 2)**2 + (z - 2)**2)
                records.append([x, y, z, entropy, laplacian])
    df = pd.DataFrame(records, columns=["x", "y", "z", "entropy", "laplacian"])
    df.to_csv(input_path, index=False)

    # Run the script with specified arguments
    env = os.environ.copy()
    proc = subprocess.run(
        [sys.executable, str(script),
         "--input-file", str(input_path),
         "--output-file", output_basename,
         "--output-dir", str(output_dir),
         "--bins", "5"],
        capture_output=True,
        text=True
    )

    assert proc.returncode == 0, f"Script failed:\n{proc.stderr}"

    # Locate the output file using the expected filename pattern
    output_files = list(output_dir.glob("HC-014-ANA_*_radial_profile.csv"))
    assert output_files, "No radial profile output file generated"
    assert output_files[-1].stat().st_size > 0, "Output file is empty"

    print("HC-014-ANA_radial_profile.py integration test passed.")

    # Cleanup temporary input and output files
    input_path.unlink()
    for f in output_files:
        f.unlink()

if __name__ == "__main__":
    test_radial_profile_output()
