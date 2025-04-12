# HoloCosmo Entry Initialization & Migration Guide

This document outlines the procedure for initializing new HoloCosmo entries using the init_hc_entry.py script and provides guidance on gradually migrating existing files into the new HC-NNN-CAT system.

---

## A. Using init_hc_entry.py

The init_hc_entry.py script automates the creation of new entries by:
- Assigning the next available identifier in the format HC-###.
- Updating a central YAML registry (hc_registry.yaml) with details about the new entry.
- Creating blank template files for specific categories (e.g., Modeling, Notebook, Output, Documentation) in the appropriate project directories.

### 1. Basic Usage

To run the script with a description and a set of category codes, use:

  python init_hc_entry.py "Descriptive title for your new entry" --categories MOD NBK OUT DOC

This command will:
- Create a new HC entry (e.g., HC-006) if the last entry was HC-005.
- Generate files such as:
  - src/modeling/HC-006-MOD_descriptive_title.py
  - notebooks/HC-006-NBK_descriptive_title.ipynb
  - data/processed/HC-006-OUT_descriptive_title.csv
  - doc/papers/HC-006-DOC_descriptive_title.md
- Update the hc_registry.yaml registry file with details of the new entry.

### 2. Optional Flags

#### Dry Run Mode

To preview changes without writing to disk, add the --dry-run flag:

  python init_hc_entry.py "Test run of halo simulation" --categories MOD NBK OUT DOC --dry-run

The script will then print what it intends to do (file creation paths and registry updates) without actually modifying any files.

#### Update Registry Only

If you only want to update the registry (without generating new files), use the flag --update-registry-only:

  python init_hc_entry.py "Registry update entry" --categories MOD NBK OUT DOC --update-registry-only

#### Specifying a Base Folder

If you prefer to nest all generated files in a specific folder (for example, a project-specific folder), use the --folder flag:

  python init_hc_entry.py "New experimental run" --categories MOD NBK OUT DOC --folder projects/HC-XXX

This will create the files inside the specified folder rather than in the default directory mapping.

---

## B. Migration from Existing Data

Rather than renaming old files directly, the migration strategy is to create new HC-tagged files and then gradually populate them with content from existing files. Once you’re satisfied that the new files fully replace the old ones, you can remove the originals.

### 1. Create New HC-Tagged Files

For each new HC entry, run the init_hc_entry.py script as explained above. For example:

  python init_hc_entry.py "Entangled gravity simulation" --categories MOD NBK DOC

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

  find . -name 'HC-XYZ*'

This process helps ensure traceability and ease of maintenance as your project evolves.

---

## Final Notes

- **Incremental Approach:** Migrate new work using the init_hc_entry.py script, and gradually transition legacy files.
- **Version Control:** Use Git or another version control system to commit changes at every step.
- **Customization:** Feel free to modify the script and this guide as your workflow evolves.

Happy coding, and may the HC entries bring order to your Sisyphean labors!
