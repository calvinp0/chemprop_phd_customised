import os
import glob
import json
from pathlib import Path
from rdkit import Chem
from arc.species import ARCSpecies
from periodictable import elements

VALID_SYMBOLS = {el.symbol for el in elements if el.symbol}

SDF_DIR = Path("/home/calvin/code/chemprop_phd_customised/habnet/data/processed/sdf_new")
CONVERGED_ROOT = Path("/home/calvin/Dropbox/PersonalFolders/Calvin")
TS_PATHS = ["calcs/TS_guesses/TS0/tsg0/crest_0/crest_best.xyz", "calcs/TS_guesses/TS0/tsg0/crest_best.xyz", "calcs/TS_guesses/TS0/tsg1/crest_best.xyz", "calcs/TS_guesses/TS0/tsg1/crest_0/crest_best.xyz"]
MISSING=[]
def find_converged_folders():
    return [p for p in CONVERGED_ROOT.glob("*Converged") if p.is_dir()]

def find_all_ts_geometries(reaction_name):
    found = []
    for folder in find_converged_folders():
        possible_dirs = [folder / reaction_name, folder / "NonRMG" / reaction_name]
        for rxn_dir in possible_dirs:
            if rxn_dir.exists():
                for ts_path in TS_PATHS:
                    xyz_path = rxn_dir / ts_path
                    if xyz_path.exists():
                        found.append((parse_xyz_file(xyz_path), "crest_best.xyz"))
                heuristics_files = list((rxn_dir / "calcs/TS_guesses/TS0").rglob("Heuristics*"))
                autotst_files = list((rxn_dir / "calcs/TS_guesses/TS0/").rglob("AutoTST F*"))
                for h in heuristics_files:
                    found.append((parse_heuristics_file(h), "Heuristics"))
                for a in autotst_files:
                    found.append((parse_heuristics_file(a), "AutoTST"))
                conf_dirs = [rxn_dir / "calcs/TSs/TS0/conf_opt_1", rxn_dir / "calcs/TSs/TS0/conformer1"]
                for conf_dir in conf_dirs:
                    log_path = conf_dir / "input.log"
                    if log_path.exists():
                        found.append((parse_gaussian_log(log_path), "gaussian_log"))
    return found

def parse_gaussian_log(log_path):
    atoms, coords = [], []
    inside_block = False
    after_charge = False
    lines = []

    with open(log_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        if "Symbolic Z-matrix" in line:
            inside_block = True
            i += 1
            continue
        if inside_block:
            stripped = line.strip()
            if not stripped:
                # Check next line for 'Add' → end condition
                if (i + 1) < len(lines) and lines[i + 1].strip().lower().startswith("add"):
                    break
                i += 1
                continue
            if stripped.lower().startswith("charge ="):
                after_charge = True
                i += 1
                continue
            if after_charge:
                parts = stripped.split()
                if len(parts) >= 4:
                    try:
                        symbol = parts[0]
                        if symbol not in VALID_SYMBOLS:
                            break  # Skip this line, likely an extra header or metadata
                        atoms.append(symbol)
                        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except ValueError:
                        break  # Stop if non-coordinate line (e.g., '=') is found
        i += 1

    if atoms and coords:
        return {"symbols": atoms, "coords": coords}
    else:
        raise ValueError(f"Could not parse geometry from Gaussian log: {log_path}")


def parse_xyz_file(xyz_path):
    atoms, coords = [], []
    with open(xyz_path) as f:
        lines = f.readlines()[2:]
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 4:
            atoms.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
    return {"symbols": atoms, "coords": coords}

def parse_heuristics_file(h_path):
    atoms, coords = [], []
    with open(h_path) as f:
        lines = f.readlines()[2:]
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 4:
            atoms.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
    return {"symbols": atoms, "coords": coords}

def reinit_sdf_mol(rxn, arc_spc, input_type, mol_properties=None, geom_source=None):
    from rdkit.Chem import AllChem
    import json
    rmg_mol = arc_spc.mol.copy(deep=True)
    rdkit_mol_mapped = rmg_mol.to_rdkit_mol(remove_h=False, save_order=True, return_mapping=True)
    rdkit_mol = rdkit_mol_mapped[0]
    xyz = arc_spc.get_xyz()
    try:
        AllChem.EmbedMolecule(rdkit_mol)
    except Exception as e:
        raise RuntimeError(f"Could not embed the molecule: {e}")
    if rdkit_mol.GetNumConformers() > 0:
        conf = rdkit_mol.GetConformer(id=0)
        for i, pos in enumerate(xyz['coords']):
            conf.SetAtomPosition(i, pos)
    else:
        raise RuntimeError("RDKit molecule has no conformers after embedding.")
    rdkit_mol.SetProp('reaction', str(rxn))
    rdkit_mol.SetProp('mol_properties', json.dumps(mol_properties) if mol_properties else '{}')
    rdkit_mol.SetProp('type', input_type)
    if geom_source:
        rdkit_mol.SetProp('geom_source', geom_source)
    return rdkit_mol

def process_sdf(sdf_path):
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mols = [mol for mol in suppl if mol is not None]
    reaction_name = None
    mol_props = None
    for mol in mols:
        if mol.HasProp("type") and mol.GetProp("type") == "ts":
            reaction_name = mol.GetProp("reaction")
            mol_props = json.loads(mol.GetProp("mol_properties"))
            break
    if not reaction_name or not mol_props:
        print(f"[WARN] Could not find TS entry in {sdf_path.name}")
        MISSING.append(sdf_path.name)
        return

    ts_geometries = find_all_ts_geometries(reaction_name)
    if not ts_geometries:
        print(f"[MISSING] TS geometries not found for {reaction_name}")
        MISSING.append(reaction_name)
        return

    for idx, (ts_geometry, source) in enumerate(ts_geometries, start=1):
        arc_spc = ARCSpecies(label=f"{reaction_name}_unop_{idx}", xyz=ts_geometry)
        new_mol = reinit_sdf_mol(
            reaction_name, arc_spc, f"unop_{idx}", mol_props, geom_source=source
        )
        mols.append(new_mol)

    out_path = sdf_path.with_name(sdf_path.stem + "_updated.sdf")
    writer = Chem.SDWriter(str(out_path))
    for mol in mols:
        writer.write(mol)
    writer.close()


def main():
    from tqdm import tqdm
    sdf_files = list(SDF_DIR.glob("*.sdf"))
    for sdf_file in tqdm(sdf_files, desc="Processing SDF files"):
        process_sdf(sdf_file)
        
    if MISSING:
        print(f"[MISSING TS GEOMETRIES] {len(MISSING)} reactions did not have TS geometries:")
        for missing in MISSING:
            print(f" - {missing}")
if __name__ == "__main__":
    main()
