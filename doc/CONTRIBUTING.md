# Contributing to the HoloCosmo Project

Don't worry, we don't adhere to dogmas. But we do have a system - and it works.

This document outlines the principles and practices that guide contributions to the HoloCosmo Project – whether you're running simulations, writing code, proposing theory, or simply following along.

This is not a typical research project. There are no positions, titles, or credentials within HoloCosmo. There are only contributors and threads to follow.

## Our Posture

The HoloCosmo Project follows four modes:  
**Speculate. Formulate. Calculate. Observe.**  
In that order. In that rhythm.

We explore ideas that may not yet have conventional justification – but we treat them with rigorous method, precise language, and replicable structure. We follow them where they lead, and document the path clearly, even (especially) when they fall apart.

We do not claim or impose a theory.  
We do not assert that any of our models are "right."  
We do insist that they are clearly stated, traceably tested, and falsifiably defined.
Moreover, we do realize that there are times when the models are correct, but we can be mistaken nonetheless. Nature is like that.

Our guiding principle is **epistemic fidelity**.  
That means: staying faithful to what the work actually shows. Not what we want it to show, or what others expect it to mean. If we succeed, it is the result of the model – and ideas don't respect hierarchy, so whether it's a seasoned modeler or not... Well, you catch the drift.

If we go astray, the trail is documented too. That's actually hugely important: we stress null results and dead ends. We've been there before. We'll be there again. Let's not let that stop us from enjoying the path, whatever it is and wherever it takes.

## Authorship and Attribution

HoloCosmo is not anonymous, but it is non–possessive.

Contributors are free to associate themselves with their work – or not. All commits, notebooks, papers, and discussion threads are traceable. But the project does not issue bylines, rank contributions, or assign ownership. We do not maintain internal hierarchies.

If we publish something at some point, attributions will go where they belong: in the grounding work, past research, foundations, and obviously the authors – including the project. We're not here to build a cathedral but scaffolding.

## Participation Guidelines

- Precision. Use clear definitions and be explicit about assumptions.
- Generosity. Respect others' ideas, and extend them thoughtfully.
- Non-territoriality. We’re not defending disciplines or borders.
- Rigor. Conjectures are welcomed – if they can be formalized, tested, or refuted.
- Friction. There is no rush here. We value depth over momentum. But if you can help us speed up our routines, whether by caching or whatever, we'd probably buy you flowers, maybe a bottle of white.

## When You Contribute

Any contribution – a code commit, a simulation run, a comment – participates in a broader collective inquiry. Please:

1. Document your work clearly and self-sufficiently.
2. Link back to the theoretical thread or question it engages.
3. Flag assumptions and approximations honestly.
4. If you're refining or challenging previous work, do so constructively.
5. We are FAFO-compliant (fool around, find out). We hope you're too. We have enough curly-headed superiors at work and at home already.

## If You Disagree

We welcome dissent and divergence. If your view forks from the main path, document it. Show your reasoning. We don’t resolve disagreement through authority – only through clarity.

Don't disagree just because of a dogma, though. Not everything popular is the best thing out there. The survivor bias cuts both ways, and many nice things have been wiped off of history books.

## Why This Exists

Not to publish first. Not to defend a theory. Not to own an idea.

This project exists because we are curious whether the structure of quantum entanglement gives rise to something geometric, gravitational, or otherwise coherent. We're following that question with rigor and epistemic honesty – and no promise of resolution.

If we do publish, well, we'll figure it out. And so will you. The record is out there in the public anyway.

You are welcome to contribute or to wander about.


# HoloCosmo Entry Initialization & Migration Guide

This document outlines the procedure for initializing new HoloCosmo entries using the `doc/hc_entry.py `script and provides guidance on gradually migrating existing files into the new HC-NNN-CAT system.

---

## A. Using hc_entry.py

The hc_entry.py script automates the creation of new entries by:
- Assigning the next available identifier in the format HC-###.
- Updating a central YAML registry (hc_registry.yaml) with details about the new entry.
- Creating blank template files for specific categories (e.g., Modeling, Notebook, Output, Documentation) in the appropriate project directories.

### 1. Basic Usage

To run the script with a description and a set of category codes, use:

  `python hc_entry.py "Descriptive title for your new entry" --categories MOD NBK OUT DOC`

This command will:
- Create a new HC entry (e.g., HC-006) if the last entry was HC-005.
- Generate files such as:
  - src/modeling/HC-006-MOD_descriptive_title.py
  - notebooks/HC-006-NBK_descriptive_title.ipynb
  - data/processed/HC-006-OUT_descriptive_title.csv
  - doc/papers/tex/HC-006-DOC_descriptive_title.tex
- Update the hc_registry.yaml registry file with details of the new entry.

### 2. Optional Flags

#### Dry Run Mode

To preview changes without writing to disk, add the --dry-run flag:

  `python hc_entry.py "Test run of halo simulation" --categories MOD NBK OUT DOC --dry-run`

The script will then print what it intends to do (file creation paths and registry updates) without actually modifying any files.

#### Update Registry Only

If you only want to update the registry (without generating new files), use the flag --update-registry-only:

  `python hc_entry.py "Registry update entry" --categories MOD NBK OUT DOC --update-registry-only`

#### Specifying a Base Folder

If you prefer to nest all generated files in a specific folder (for example, a project-specific folder), use the --folder flag:

  `python hc_entry.py "New experimental run" --categories MOD NBK OUT DOC --folder projects/HC-XXX`

This will create the files inside the specified folder rather than in the default directory mapping.

---

## B. Migration from Existing Data

Rather than renaming old files directly, the migration strategy is to create new HC-tagged files and then gradually populate them with content from existing files. Once you’re satisfied that the new files fully replace the old ones, you can remove the originals.

### 1. Create New HC-Tagged Files

For each new HC entry, run the hc_entry.py script as explained above. For example:

  `python hc_entry.py "Entangled gravity simulation" --categories MOD NBK DOC`

This will generate:
- A new modeling script (e.g., src/modeling/HC-XYZ-MOD_entangled_gravity_simulation.py)
- A new notebook (e.g., notebooks/HC-XYZ-NBK_entangled_gravity_simulation.ipynb)
- A documentation file (e.g., doc/papers/HC-XYZ-DOC_entangled_gravity_simulation.md)

### 2. Migrate Content Manually

Once the new blank files have been created:
- **Step 1:** Open the new HC-tagged file.
- **Step 2:** Copy and paste the content from the corresponding existing file.
- **Step 3:** Adapt any paths or references as needed so that the new file works correctly in its new location.
- **Step 4:** Test the newly created file to ensure proper functionality.

### 3. Gradual Decommissioning

- **Keep the Old Files:** Do not remove the old files immediately. Continue to work and test with both sets until you confirm the migration is successful.
- **Record in Registry:** Update the hc_registry.yaml registry by noting which new HC entries replace which original files.
- **Final Cleanup:** Once you’re fully confident with the migration, remove or archive the original files lacking the HC identifiers.

### 4. Maintaining the Registry

All changes and new file creations are logged in the registry file (hc_registry.yaml), which serves as a central reference index linking the HC identifiers to the new files. This way, any new analysis, paper, or script can easily refer to its corresponding HC entry using a command such as:

  `find . -name 'HC-XYZ*'`

This process helps ensure traceability and ease of maintenance as your project evolves.