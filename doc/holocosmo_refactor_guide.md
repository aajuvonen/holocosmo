# HoloCosmo Script Refactoring Guide

This checklist outlines the steps to standardize modeling scripts in the `src/modeling/` folder.
Use it for each script refactor to maintain clarity, reproducibility, and scientific grounding.

1. Rename + Relocate
- Rename the script to a clear, descriptive name that reflects its function
- Move it to `src/modeling/` if not already there

2. Add Docstring Header
Include a top-level docstring at the top of the script with the following:
- Script name and purpose
- Inputs (CLI flags, file paths, parameters)
- Outputs (e.g. CSV, plots)
- Scientific context (mention related papers in `/doc/papers/`)
- Date and authorship (HoloCosmo Project)

3. Refactor Parameters & Paths
- Use argparse for all parameters with sensible defaults
- Allow --output-dir to control file saving location (default: data/processed/)
- Format output filenames with timestamp + key params
- Use sys.executable and Path() if needed to resolve script paths in tests

4. Add Inline Comments
- Explain key model parts (e.g., gate construction, evolution, entropy)
- Link theory to code where possible, e.g.:
 # Based on Eq. (3) in "Entanglement Curvature"

5. Create Unit Test
In `tests/`, add a test like:
 def test_script_runs(tmp_path):
 ...
 result = subprocess.run([
 sys.executable, str(script_path),
 "--lattice", "6", "6", "6",
 "--steps", "1",
 "--output-dir", str(tmp_path / "out")
 ])
 ...
- Test script runs with minimal params
- Assert a CSV is created with correct headers
- Use tmp_path to isolate test output

6. Update EXAMPLE.md
In `src/modeling/EXAMPLE.md`:
- Add one-liner description of what the script does
- Add one example command block
 ## script_name.py
 Brief description.
 python script_name.py --steps 2 --plot

7. Run with run_tests.py (Optional)
After everything's in place:
 python tests/run_tests.py

Example Scripts Already Converted:
- gravity_laplacian_simulation.py
- gravity_laplacian_impurity.py