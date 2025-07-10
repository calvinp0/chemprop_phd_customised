import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import AllChem
import pandas as pd
import ast

def get_distance(coords, i, j):
    return np.linalg.norm(coords[i] - coords[j])

def get_angle(coords, i, j, k):
    # Angle at j (i-j-k)
    v1 = coords[i] - coords[j]
    v2 = coords[k] - coords[j]
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos_theta, -1, 1))

def get_dihedral(coords, i, j, k, l):
    # Dihedral angle between i-j-k-l
    p0, p1, p2, p3 = coords[i], coords[j], coords[k], coords[l]
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.arctan2(y, x)

def get_mol_coords(mol):
    """Returns (num_atoms, 3) array of atom positions from the first conformer."""
    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(num_atoms)])
    return coords

def compute_atomfeats(mol, label, reference_labels=('d_hydrogen', 'a_hydrogen', 'acceptor')):
    num_atoms = mol.GetNumAtoms()
    coords = np.array([list(mol.GetConformer().GetAtomPosition(i)) for i in range(num_atoms)])
    
    # Build role flags lookup
    atom_roles = {i: {"is_donor": 0, "is_acceptor": 0, "is_donor_H": 0, "is_acceptor_neighbor": 0, "is_donor_neighbor": 0} for i in range(num_atoms)}

    for key, value in label.items():
        idx = int(key)
        role = value.get("label")
        if role == "donator":
            atom_roles[idx]["is_donor"] = 1
        elif role == "acceptor":
            atom_roles[idx]["is_acceptor"] = 1
        elif role in ["d_hydrogen", "a_hydrogen"]:
            atom_roles[idx]["is_donor_H"] = 1

    # Infer neighbors for donor and acceptor
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if atom_roles[idx]["is_donor"]:
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() != "H":
                    atom_roles[neighbor.GetIdx()]["is_donor_neighbor"] = 1
        if atom_roles[idx]["is_acceptor"]:
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() != "H":
                    atom_roles[neighbor.GetIdx()]["is_acceptor_neighbor"] = 1

    # Find reference atom
    ref_indices = [int(k) for k, v in label.items() if v["label"] in reference_labels]
    if not ref_indices:
        raise ValueError("No valid reference atom (d_hydrogen or acceptor) found in label")
    reference_idx = ref_indices[0]

    feat_records = []
    for target_idx in range(num_atoms):
        record = {'atom_idx': target_idx, 'path': None, 'features': None}
        role_flags = atom_roles[target_idx]

        if target_idx == reference_idx:
            record['path'] = [target_idx]
            record['features'] = [np.nan, np.nan, np.nan, 0, 0, 0]
            record.update(role_flags)
            feat_records.append(record)
            continue

        path = list(rdmolops.GetShortestPath(mol, reference_idx, target_idx))
        plen = len(path)
        flags = [0, 0, 0]
        radius = angle = dihedral = np.nan
        if plen >= 2:
            flags[0] = 1
            radius = get_distance(coords, reference_idx, target_idx)
        if plen >= 3:
            flags[1] = 1
            angle = get_angle(coords, reference_idx, path[1], target_idx)
        if plen >= 4:
            flags[2] = 1
            dihedral = get_dihedral(coords, reference_idx, path[1], path[2], target_idx)
        record['path'] = path
        record['features'] = [radius, angle, dihedral, *flags]
        record.update(role_flags)
        feat_records.append(record)

    return feat_records


def all_z_zero(mol):
    conf = mol.GetConformer()
    z = [conf.GetAtomPosition(i).z for i in range(mol.GetNumAtoms())]
    return all(abs(val) < 1e-8 for val in z)


def get_sdf_file_list(directory):
    """Returns a list of SDF files in the given directory."""
    import os
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.sdf')]


def create_atom_feat_RAD_table(sdf_file_list, reference_labels=None):
    """
    Create a table of atom features (radius, angle, dihedral) from SDF files.
    
    Args:
        sdf_file_list (list): List of paths to SDF files.
        
    Returns:
        pd.DataFrame: DataFrame containing atom features.
    """

    all_feat_rows = []

    for sdf_file in sdf_file_list:
        suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
        for idx,mol in enumerate(suppl):
            if mol is None:
                continue
            rxn_id = mol.GetProp('reaction') if mol.HasProp('reaction') else 'unknown'
            mol_type = mol.GetProp('type') if mol.HasProp('type') else 'unknown'  # 'r1h' or 'r2h'
            if mol_type not in ['r1h', 'r2h', 'r2']:
                continue

            if all_z_zero(mol):
                rxn_id = mol.GetProp('reaction') if mol.HasProp('reaction') else 'unknown'
                mol_type = mol.GetProp('type') if mol.HasProp('type') else 'unknown'
                print(f"Flat Z in: {sdf_file}, molecule #{idx}, rxn_id={rxn_id}, type={mol_type}")

            label = ast.literal_eval(mol.GetProp('mol_properties'))
            feat_records = compute_atomfeats(mol, label, reference_labels=reference_labels)
            # Extract features and create rows
            for rec in feat_records:
                focus_atom = rec['path'][-1]
                atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

                row = {
                    'rxn_id': rxn_id,
                    'mol_type': mol_type,
                    'focus_atom_idx': focus_atom,
                    'path': str(rec['path']),
                    'radius': rec['features'][0],
                    'angle': rec['features'][1],
                    'dihedral': rec['features'][2],
                    'r_exist': rec['features'][3],
                    'a_exist': rec['features'][4],
                    'd_exist': rec['features'][5],
                    'focus_atom_symbol': atom_symbols[focus_atom],
                }
                row.update({
                    "is_donor": rec.get("is_donor", 0),
                    "is_acceptor": rec.get("is_acceptor", 0),
                    "is_donor_H": rec.get("is_donor_H", 0),
                    "is_acceptor_neighbor": rec.get("is_acceptor_neighbor", 0),
                })
                all_feat_rows.append(row)

    # --- Combine and save as CSV ---
    df_all_feats = pd.DataFrame(all_feat_rows)
    df_all_feats.to_csv("all_sdf_features.csv", index=False)


if __name__ == "__main__":
    # Example usage
    sdf_directory = "/home/calvin/code/chemprop_phd_customised/habnet/data/processed/sdf_r2"  # Replace with your SDF directory
    sdf_files = get_sdf_file_list(sdf_directory)
    create_atom_feat_RAD_table(sdf_files, reference_labels=('d_hydrogen', 'a_hydrogen', 'acceptor'))
    print("Atom features table created and saved to 'all_sdf_features.csv'.")