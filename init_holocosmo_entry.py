#!/usr/bin/env python3
"""
init_hc_entry.py

A command-line tool to initialize a new HoloCosmo (HC-NNN) entry.
It assigns a new identifier, updates a YAML registry file, and scaffolds template files 
for selected categories in the current repository structure.

Usage:
  python init_hc_entry.py "Descriptive title of experiment" --categories MOD NBK OUT DOC [--folder path/to/base] [--dry-run] [--update-registry-only]
"""

import os
import sys
import argparse
import datetime
import re
import json
import yaml

def slugify(title):
    """Generate a simple slug from the title."""
    return re.sub(r'[^a-z0-9_]+', '', title.lower().replace(' ', '_'))

def load_registry(registry_path):
    """Load the YAML registry if it exists; otherwise, return an empty list."""
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            try:
                registry = yaml.safe_load(f)
                return registry if registry is not None else []
            except Exception as e:
                print(f"Error loading registry: {e}")
                return []
    else:
        return []

def save_registry(registry, registry_path, dry_run):
    """Save the registry data to the registry file (unless in dry-run mode)."""
    if dry_run:
        print(f"[Dry-run] Would save registry to {registry_path} with {len(registry)} entries.")
    else:
        with open(registry_path, 'w') as f:
            yaml.dump(registry, f)
        print(f"Registry updated: {registry_path}")

def get_next_id(registry):
    """Return the next HC identifier in the format HC-###."""
    max_id = 0
    for entry in registry:
        id_str = entry.get('id', '')
        match = re.match(r'HC-(\d{3})', id_str)
        if match:
            num = int(match.group(1))
            if num > max_id:
                max_id = num
    next_id = max_id + 1
    if next_id > 999:
        raise Exception("ID limit reached (999)")
    return f"HC-{next_id:03d}"

def generate_file_path(base_dir, id_str, category, title_slug, mapping):
    """
    Generate a file path based on the category mapping.
    If a base folder is provided, all files are created inside that folder.
    Otherwise, the default folder mapping is used.
    """
    if category in mapping:
        folder, ext = mapping[category]
        target_folder = base_dir if base_dir else folder
    else:
        target_folder = base_dir if base_dir else '.'
        ext = '.txt'
    filename = f"{id_str}-{category}_{title_slug}{ext}"
    return os.path.join(target_folder, filename)

def create_file(path, content, dry_run):
    """Create a file at the given path with the provided content."""
    if dry_run:
        print(f"[Dry-run] Would create file: {path}")
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        print(f"Created file: {path}")

def main():
    parser = argparse.ArgumentParser(description="Initialize a new HC entry.")
    parser.add_argument("title", help="Title/description for the new HC entry.")
    parser.add_argument("--categories", nargs='*', default=[], help="List of category codes (e.g., MOD, NBK, OUT, DOC) for file scaffolding.")
    parser.add_argument("--folder", help="Optional base folder to nest all generated files.")
    parser.add_argument("--dry-run", action="store_true", help="Simulate actions without writing to disk.")
    parser.add_argument("--update-registry-only", action="store_true", help="Only update the registry file, skip file creation.")

    args = parser.parse_args()

    # Define category mapping: each key maps to (default folder, default file extension)
    category_mapping = {
        'MOD': ('src/modeling', '.py'),
        'ANA': ('src/analysis', '.py'),
        'NBK': ('notebooks', '.ipynb'),
        'OUT': ('data/processed', '.csv'),
        'DAT': ('data/raw', '.csv'),
        'DOC': ('doc/papers', '.md'),
        'VIS': ('data/figures', '.png'),
        'TST': ('tests', '.py')
    }
    
    registry_path = "hc_registry.yaml"
    
    # Load the existing registry
    registry = load_registry(registry_path)

    # Determine the next HC id
    next_id = get_next_id(registry)
    title_slug = slugify(args.title)
    
    # Create a new registry entry
    entry = {
        'id': next_id,
        'title': args.title,
        'created': datetime.datetime.now().isoformat(),
        'categories': [cat.upper() for cat in args.categories],
        'files': []
    }
    
    # Generate file scaffolding unless update-registry-only is set
    if not args.update_registry_only:
        for cat in args.categories:
            cat = cat.upper()
            file_path = generate_file_path(args.folder, next_id, cat, title_slug, category_mapping)
            # Build template content based on category
            content = f"# {next_id}-{cat}: {args.title}\n# Created on {entry['created']}\n"
            if cat == "TST":
                # A simple unit test template
                content += "\nimport unittest\n\nclass TestSomething(unittest.TestCase):\n    def test_example(self):\n        self.assertTrue(True)\n\nif __name__ == '__main__':\n    unittest.main()\n"
            elif cat == "NBK":
                # Create a minimal Jupyter Notebook (in JSON)
                nb = {
                    "cells": [
                        {
                            "cell_type": "markdown",
                            "metadata": {},
                            "source": [f"# {args.title}\n", f"ID: {next_id}-{cat}\n", f"Created: {entry['created']}\n"]
                        },
                        {
                            "cell_type": "code",
                            "execution_count": None,
                            "metadata": {},
                            "outputs": [],
                            "source": [f"# Write your analysis here\n"]
                        }
                    ],
                    "metadata": {
                        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                        "language_info": {"name": "python", "version": "3.x"}
                    },
                    "nbformat": 4,
                    "nbformat_minor": 2
                }
                content = json.dumps(nb, indent=2)
            
            create_file(file_path, content, args.dry_run)
            entry['files'].append(file_path)
    
    # Append the new entry to the registry and save it
    registry.append(entry)
    if args.dry_run:
        print(f"[Dry-run] Would update registry {registry_path} with entry:\n{entry}")
    else:
        save_registry(registry, registry_path, args.dry_run)

if __name__ == '__main__':
    main()

