#!/usr/bin/env python3
"""
Scan an SDF directory and list every original <name>.sdf that has
no corresponding <name>_fixed.sdf.
"""

from pathlib import Path

SDF_DIR = Path("/home/calvin/code/chemprop_phd_customised/habnet/data/processed/sdf_data")

missing = []

for sdf in SDF_DIR.glob("*.sdf"):
    if sdf.stem.endswith("_fixed"):          # skip the fixed copies
        continue

    fixed_path = sdf.with_name(f"{sdf.stem}_fixed.sdf")
    if not fixed_path.exists():
        missing.append(sdf.name)

if missing:
    print("SDF files without a matching _fixed copy:")
    for fname in sorted(missing):
        print(" •", fname)
else:
    print("All originals have matching _fixed.sdf files.")
