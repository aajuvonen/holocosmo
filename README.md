# The HoloCosmo Project

## Overview

The **HoloCosmo Project** is an independent scientific exploration aiming to understand cosmological phenomena through two foundational assumptions:

- Physical reality is a holographic projection of information encoded on a discrete, lower-dimensional surface that evolves over cosmic time.
- Gravitation emerges naturally from quantum entanglement structures.

This project adopts a scientifically rigorous yet non-dogmatic stance. It neither explicitly endorses nor seeks to refute existing cosmological models. Instead, it explores whether these assumptions can yield internally consistent, computationally viable, and observationally testable interpretations of cosmological data.

## Repository Structure

The repository is structured to separate raw data, scripts, analysis, and theoretical work. This supports transparency, modular development, and reproducibility.

```
holocosmo/ ├── data/
           │ ├── raw/             # Untouched input datasets (e.g. observational catalogs, .dat files)
           │ ├── interim/         # Intermediate files produced during computations
           │ └── processed/       # Final data products suitable for analysis or plotting
           ├── notebooks/         # Jupyter Notebooks for exploration, modeling, and results
           │ ├── 01_intro.ipynb
           │ ├── 02_holography.ipynb
           │ ├── 03_entangled_gravity.ipynb
           │ └── 04_analysis.ipynb
           ├── src/               # Source code and models (parameterized & modular)
           │ ├── data_processing/ # Scripts to clean, convert, and prepare datasets
           │ ├── modeling/        # Simulation and computation modules
           │ └── visualization/   # Plotting and presentation tools
           ├── reports/           # Generated figures and summaries for publication or presentation
           │ └── figures/
           ├── doc/               # Theoretical documents and PDFs
           ├── cdk/               # AWS CDK project for batch/cloud compute
           ├── tests/             # Integration tests and unit tests for scripts
           ├── requirements.txt   # Python package dependencies
           ├── LICENSE            # MIT License — chosen to maximize openness and collaboration.
           └── README.md          # Project overview
```

## How to Engage with This Project

- **Explore the Notebooks**: Go to `/notebooks` for the story — step-by-step walkthroughs of the ideas, models, and analysis.
- **Run the Code**: Modular Python scripts in `/src` implement the key computational logic. Scripts accept command-line parameters and produce timestamped outputs to avoid overwriting.
- **Check the Data**: All outputs are saved to `/data`, organized into `raw`, `interim`, and `processed` stages to ensure data provenance.
- **Read the Theory**: Dive into derivations and conceptual underpinnings in the `/doc` directory.
- **Use the Infrastructure** *(Experimental)*: The `/cdk` folder contains an AWS CDK TypeScript project intended for scaling computations in the cloud (documentation coming soon).

## Scientific Posture

The HoloCosmo Project emphasizes:

- **Neutrality**: It is not intended as a replacement to ΛCDM or any other framework — but as a tool to examine what falls out from a different starting point.
- **Transparency**: All simulations, results, and inconsistencies are documented honestly.
- **Humility and Curiosity**: The goal is understanding and coherence, not confirmation or refutation.

## Objectives

- **Test internal coherence** of holography- and entanglement-based cosmological principles.
- **Generate simulation outputs** that are computationally viable and can be compared to empirical data.
- **Foster interdisciplinary dialogue** between cosmology, quantum information, and gravitational physics.

## Status

The project is evolving. Some components are mature, others are in flux. Current priorities:

- Refactoring scripts for reproducibility and resource efficiency
- Cloud processing for heavy simulations
- Wrapping models and data pipelines into documented workflows

## License

MIT License — chosen to maximize openness and collaboration.