import os
import pandas as pd
import ast
from rdkit import Chem
from arc.species import ARCSpecies
import json

# Your main logic as functions
def get_symbols_from_r1h_column(row):
    r1h = row['r1h']
    if pd.isna(r1h):
        return []
    r1h_mol_coords_list = ast.literal_eval(r1h)
    symbols = []
    for coord in r1h_mol_coords_list:
        if isinstance(coord, str):
            # String like "Cl 0.0 0.0 0.0" → split by space
            symbol = coord.strip().split()[0]
        elif isinstance(coord, list):
            # List like ['Cl', 0.0, 0.0, 0.0]
            symbol = coord[0]
        else:
            raise ValueError(f"Unexpected format: {coord}")
        symbols.append(symbol)
    return symbols

def remove_a_hydrogen_and_adjust_keys(data):
    acceptor_key = None
    a_hydrogen_key = None
    for key, value in data.items():
        if value.get("label") == "acceptor":
            acceptor_key = int(key)
        elif value.get("label") == "a_hydrogen":
            a_hydrogen_key = int(key)
    if a_hydrogen_key is None:
        return data
    new_data = {k: v for k, v in data.items() if v.get("label") != "a_hydrogen"}
    if acceptor_key is not None and a_hydrogen_key < acceptor_key:
        acceptor_value = new_data.pop(str(acceptor_key))
        new_data[str(acceptor_key - 1)] = acceptor_value
    return dict(sorted(new_data.items(), key=lambda x: int(x[0])))

def reinit_sdf_mol(rxn, arc_spc, input_type, mol_properties=None):
    from rdkit.Chem import AllChem
    rmg_mol = arc_spc.mol.copy(deep=True)
    rdkit_mol_mapped = rmg_mol.to_rdkit_mol(remove_h=False, save_order=True, return_mapping=True)
    rdkit_mol = rdkit_mol_mapped[0]
    xyz = arc_spc.get_xyz()
    AllChem.EmbedMolecule(rdkit_mol)
    if rdkit_mol.GetNumConformers() > 0:
        conf = rdkit_mol.GetConformer(id=0)
        for i, pos in enumerate(xyz['coords']):
            conf.SetAtomPosition(i, pos)
    rdkit_mol.SetProp('reaction', str(rxn))
    rdkit_mol.SetProp('mol_properties', json.dumps(mol_properties) if mol_properties else '{}')
    rdkit_mol.SetProp('type', input_type)
    return rdkit_mol

def write_spc_to_sdf(mol_dict, file_path, rxn_value):
    from rdkit import Chem
    rdkit_mols = []
    for key, value in mol_dict.items():
        rdkit_mol = reinit_sdf_mol(
            rxn=value['mol_rxn'],
            arc_spc=value['arc'],
            input_type=key,
            mol_properties=value['props'] if isinstance(value['props'], dict) else ast.literal_eval(value['props'])
        )
        rdkit_mols.append(rdkit_mol)
    w = Chem.SDWriter(os.path.join(file_path, f'{rxn_value}.sdf'))
    for mol in rdkit_mols:
        w.write(mol)
    w.close()

# Main loop script
def process_sdf_directory(input_sdf_dir, output_sdf_dir, coords_csv_path):
    os.makedirs(output_sdf_dir, exist_ok=True)
    coords = pd.read_csv(coords_csv_path)

    sdf_files = [f for f in os.listdir(input_sdf_dir) if f.endswith('_fixed.sdf')]

    for sdf_file in sdf_files:
        sdf_path = os.path.join(input_sdf_dir, sdf_file)
        mols = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)

        try:
            mol_r1h = [m for m in mols if m.GetProp('type') == 'r1h'][0]
            mol_r2h = [m for m in mols if m.GetProp('type') == 'r2h'][0]
            mol_ts = [m for m in mols if m.GetProp('type') == 'ts'][0]
        except IndexError:
            print(f"Skipping {sdf_file}: Missing r1h, r2h, or ts.")
            continue

        rxn_value = mol_r1h.GetProp('reaction')
        filtered = coords[coords['rxn'] == rxn_value]

        # Match the correct row in coords
        reference_symbols = [atom.GetSymbol() for atom in mol_r1h.GetAtoms()]
        matching_indices = []
        for index, row in filtered.iterrows():
            symbols = get_symbols_from_r1h_column(row)
            if symbols == reference_symbols:
                matching_indices.append(index)

        if not matching_indices:
            print(f"No matching row found for {rxn_value} in {sdf_file}. Skipping.")
            continue

        matching_row = filtered.loc[matching_indices[0]]

        # Prepare ARC species
        import ast
        r1h_arc = ARCSpecies(label='r1h', xyz="\n".join(ast.literal_eval(matching_row['r1h'])))
        r2_arc = ARCSpecies(label='r2', xyz="\n".join(ast.literal_eval(matching_row['r2'])))
        r2h_arc = ARCSpecies(label='r2h', xyz="\n".join(ast.literal_eval(matching_row['r2h'])))

        conf_ts = mol_ts.GetConformer()
        output_lines = []
        for atom in mol_ts.GetAtoms():
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()
            pos = conf_ts.GetAtomPosition(idx)
            output_lines.append(f"{symbol} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
        ts_arc = ARCSpecies(label='ts', xyz="\n".join(output_lines), is_ts=True)

        mol_dict = {
            'r1h': {'mol': mol_r1h, 'props': mol_r1h.GetProp('mol_properties'), 'mol_rxn': mol_r1h.GetProp('reaction'), 'arc': r1h_arc},
            'r2': {'mol': mol_r2h, 'props': remove_a_hydrogen_and_adjust_keys(ast.literal_eval(mol_r2h.GetProp('mol_properties'))), 'mol_rxn': mol_r2h.GetProp('reaction'), 'arc': r2_arc},
            'r2h': {'mol': mol_r2h, 'props': mol_r2h.GetProp('mol_properties'), 'mol_rxn': mol_r2h.GetProp('reaction'), 'arc': r2h_arc},
            'ts': {'mol': mol_ts, 'props': mol_ts.GetProp('mol_properties'), 'mol_rxn': mol_ts.GetProp('reaction'), 'arc': ts_arc}
        }

        write_spc_to_sdf(mol_dict, output_sdf_dir, rxn_value)
        print(f"Processed {sdf_file} → {rxn_value}.sdf")

# Example usage
process_sdf_directory(
    input_sdf_dir='/home/calvin/code/chemprop_phd_customised/habnet/data/processed/sdf_data',
    output_sdf_dir='/home/calvin/code/chemprop_phd_customised/habnet/data/processed/sdf_new',
    coords_csv_path='/home/calvin/code/chemprop_phd_customised/habnet/data/preprocessing/ordered_reactions_data/ordered_habstraction_coords.csv'
)
