# HC-015-ANA: Potential fitting
# Created on 2025-04-15T20:12:40.665329
"""
Author: The HoloCosmo Project
Date: April 2025

Description:
-------------
This script fits an effective potential model to the radial profile of the entanglement curvature field.
It reads a CSV file (generated by HC-014-ANA) containing:
    - radial_bin_center: the center of each radial bin,
    - mean_laplacian: the mean Laplacian (curvature proxy) in that bin,
    - std_laplacian: the standard deviation in that bin,
    - count: the number of points in the bin.

Available models:
    1. Yukawa (default):   Φ(r) = A * exp(-r/λ) / r
    2. Gaussian:           Φ(r) = A * exp(-r²/(2σ²))

Outputs:
---------
- A parameter summary CSV saved to:
      ../../data/processed/HC-015-ANA_{timestamp}_potential_fitting.csv
- A figure showing the fit saved to:
      ../../data/figures/HC-015-ANA_{timestamp}_potential_fitting.svg

Inputs (via CLI):
------------------
--input-file FILE.csv
    Input from HC-014-ANA. If not provided, scans for matching files and prompts the user.
--model {yukawa, gaussian}
    Model to be used for fitting (default: yukawa).
--output-file BASENAME
    Base name for output files (no extension).
--output-dir DIR
    Directory for output CSV file (default: ../../data/processed/).
--figure-dir DIR
    Directory for output figure (default: ../../data/figures/).
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime
import os
import glob

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fit an effective potential to radial curvature data.")
    parser.add_argument("--input-file", type=str, default=None,
                        help="Path to the radial profile CSV file. If not provided, the script will prompt.")
    parser.add_argument("--model", type=str, default="yukawa", choices=["yukawa", "gaussian"],
                        help="Choose the fit model: 'yukawa' (default) or 'gaussian'.")
    parser.add_argument("--output-file", type=str, default="potential_fitting",
                        help="Base name for output files (without extension).")
    parser.add_argument("--output-dir", type=str, default="../../data/processed/",
                        help="Output directory for the CSV file.")
    parser.add_argument("--figure-dir", type=str, default="../../data/figures/",
                        help="Directory for output SVG figure.")
    return parser.parse_args()

def prompt_for_input_file():
    search_pattern = "../../data/processed/HC-014-*_radial_profile.csv"
    matching_files = sorted(glob.glob(search_pattern))
    if not matching_files:
        print("No matching radial profile files found. Exiting.")
        exit(0)
    print("Found the following files:")
    for idx, fname in enumerate(matching_files):
        print(f"[{idx}] {fname}")
    try:
        choice = int(input("Enter the index of the file to use: "))
        return matching_files[choice]
    except Exception:
        print("Invalid selection. Exiting.")
        exit(0)

def yukawa_model(r, A, lam):
    r_safe = np.where(r == 0, 1e-6, r)
    return A * np.exp(-r_safe / lam) / r_safe

def gaussian_model(r, A, sigma):
    return A * np.exp(-r**2 / (2 * sigma**2))

def fit_potential(r, y, model_type="yukawa"):
    valid = np.isfinite(r) & np.isfinite(y)
    r_valid, y_valid = r[valid], y[valid]
    if len(r_valid) < 2:
        raise ValueError("Not enough valid data points to fit.")
    if model_type == "yukawa":
        p0 = [y_valid[0] if y_valid[0] != 0 else 1e-3, r_valid.max() / 2]
        popt, pcov = curve_fit(yukawa_model, r_valid, y_valid, p0=p0, maxfev=10000)
        return popt, pcov, yukawa_model
    elif model_type == "gaussian":
        p0 = [y_valid[0] if y_valid[0] != 0 else 1e-3, r_valid.max() / 2]
        popt, pcov = curve_fit(gaussian_model, r_valid, y_valid, p0=p0, maxfev=10000)
        return popt, pcov, gaussian_model
    else:
        raise ValueError("Unsupported model.")

def main():
    args = parse_arguments()

    if args.input_file is None:
        print("No --input-file provided. Scanning for radial profile files...")
        args.input_file = prompt_for_input_file()

    print(f"Loading data from {args.input_file}")
    df = pd.read_csv(args.input_file)
    r = df["radial_bin_center"].values
    y = df["mean_laplacian"].values

    print(f"Fitting model: {args.model}")
    popt, pcov, model_func = fit_potential(r, y, model_type=args.model)
    perr = np.sqrt(np.diag(pcov))

    r_fit = np.linspace(r.min(), r.max(), 200)
    y_fit = model_func(r_fit, *popt)
    residuals = y - model_func(r, *popt)
    rmse = np.sqrt(np.mean(residuals**2))

    print(f"Fitted parameters: {popt}")
    print(f"Uncertainties: {perr}")
    print(f"RMSE: {rmse:.5f}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    output_csv_name = f"HC-015-ANA_{timestamp}_potential_fitting.csv"
    output_csv_path = os.path.join(args.output_dir, output_csv_name)
    os.makedirs(args.output_dir, exist_ok=True)

    param_names = ["A", "lambda"] if args.model == "yukawa" else ["A", "sigma"]
    results = pd.DataFrame({
        "model": [args.model] * len(popt),
        "parameter": param_names,
        "value": popt,
        "error": perr,
        "rmse": [rmse] * len(popt)
    })
    results.to_csv(output_csv_path, index=False)
    print(f"Saved fit parameters to {output_csv_path}")

    fig = plt.figure(figsize=(8, 6))
    plt.errorbar(r, y, yerr=df.get("std_laplacian", None), fmt='o', capsize=3, label="Data")
    plt.plot(r_fit, y_fit, 'r-', label=f"{args.model} fit")
    plt.xlabel("Radial Distance")
    plt.ylabel("Mean Laplacian")
    plt.title("Effective Potential Fit")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(args.figure_dir, exist_ok=True)
    fig_filename = f"HC-015-ANA_{timestamp}_potential_fitting.svg"
    fig_path = os.path.join(args.figure_dir, fig_filename)
    plt.savefig(fig_path, format="svg")
    print(f"Saved figure to {fig_path}")

if __name__ == "__main__":
    main()
