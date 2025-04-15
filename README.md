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
           │ ├── figures/         # Output figures, usually .svg or .png
           │ ├── raw/             # Untouched input datasets (e.g. observational catalogs, .dat files)
           │ ├── interim/         # Intermediate files produced during computations
           │ └── processed/       # Final data products suitable for analysis or plotting
           ├── notebooks/         # Jupyter Notebooks for exploration, modeling, and results (TODO)
           │ ├── 01_intro.ipynb
           │ ├── 02_holography.ipynb
           │ ├── 03_entangled_gravity.ipynb
           │ └── 04_analysis.ipynb
           ├── src/               # Source code and models (parameterized & modular)
           │ ├── modeling/        # Simulation and computation modules
           │ └── analysis/        # Modules for analyzing modeling results
           ├── doc/               # Documents and PDFs
           │ ├── papers/          # Theoretical basis for models
           │ ├── requirements.txt # Python package dependencies
           │ ├── hc_entry.py      # A script for initializing HC Identifiers
           │ ├── hc_registry.yaml # The catalog of HC Identifiers
           │ ├── TODO.md          # A memo for things that need taking care of
           │ └── CONTRIBUTING.md  # Hop in, the water is lovely
           ├── cdk/               # AWS CDK project for batch/cloud compute
           ├── tests/             # Integration tests and unit tests for scripts
           ├── LICENSE            # MIT License — chosen to maximize openness and collaboration.
           └── README.md          # Project overview
           
```

## HC Identifier System

The repository is in the process of adopting a unified identifier scheme for all modeling, analysis, outputs, notebooks, and documentation using the format:

HC-NNN-CAT

Where:
- HC is the project prefix.
- NNN is a unique, running number (e.g., 001, 002, ...).
- CAT is an optional category tag such as:
  - MOD: Modeling script
  - ANA: Analysis script
  - NBK: Jupyter Notebook
  - OUT: Output file
  - FIG: Figure file
  - DOC: Documentation or publication draft
  - TST: Test script
  - DAT: Dataset

This system allows for consistent cross-referencing between related components. For example, HC-006 may include a modeling script (HC-006-MOD.py), analysis notebook (HC-006-NBK.ipynb), output file (HC-006-OUT.csv), and an associated paper (HC-006-DOC.tex).

### Adding New Entries

To initialize a new HC entry, use:

  `python hc_entry.py "Description of new experiment" --categories MOD NBK OUT DOC`

Optional flags:
- --dry-run: Preview changes without writing files.
- --folder <dir>: Place all generated files in a specified directory.
- --update-registry-only: Skip file creation and only register the new ID.

This will:
- Assign the next available HC identifier
- Generate blank files in their appropriate folders
- Update the hc_registry.yaml tracking file

For details, refer to `doc/CONTRIBUTING.md`.

### Migrating Existing Scripts

Legacy files are not renamed directly. Instead, for each migrated effort:
- A new HC ID is assigned using the same script
- The related code, notebooks, outputs, or documents are manually copied into the newly generated blank files
- Once the migration is validated, original non-HC files may be deprecated and removed

This incremental migration approach ensures minimal disruption while gradually introducing traceability and modularity across the project.

## How to Engage with This Project

- **Explore the Notebooks**: Go to `/notebooks` for the story — step-by-step walkthroughs of the ideas, models, and analysis. (In progress)
- **Run the Code**: Modular Python scripts in `/src` implement the key computational logic. Scripts accept command-line parameters and produce timestamped outputs to avoid overwriting.
- **Check the Data**: All outputs are saved to `/data`, organized into `raw`, `interim`, and `processed` stages to ensure data provenance.
- **Read the Theory**: Dive into derivations and conceptual underpinnings in the `/doc` directory.
- **Use the Infrastructure** *(Experimental)*: The `/cdk` folder contains an AWS CDK TypeScript project intended for scaling computations in the cloud (Documentation in progress).

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