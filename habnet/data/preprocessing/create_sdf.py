import json
import ast

from rdkit.Chem import AllChem, SDWriter
import pandas as pd

from arc.species import ARCSpecies
from electronegativity import electronegativity

def read_connections(file_path: str) -> pd.DataFrame:
    """
    Read the connections file and return a DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the connected reactions
    """

    connection_df = pd.read_csv(file_path, index_col=0)
    connection_df['r1h'] = connection_df['r1h'].apply(ast.literal_eval)
    connection_df['r2h'] = connection_df['r2h'].apply(ast.literal_eval)
    connection_df['index_dict'] = connection_df['index_dict'].apply(ast.literal_eval)
    connection_df['ts'] = connection_df['ts'].apply(ast.literal_eval)
    connection_df['ts_results'] = connection_df['ts_results'].apply(ast.literal_eval)

    return connection_df

def rdkit_conf_from_arcspc(rxn, arc_spc, mol_type: str, label_map: dict):
    """
    Generate an RDKit molecule object from an ARCSpecies object and assign
    atom-level properties based on the label_map.

    Args:
        idx (str): The index of the reaction.
        arc_spc: An ARCSpecies-like object containing the RMG molecule (`arc_spc.mol`).
        mol_type (str): Either 'reactant' or 'product', which determines how 
            special labels are assigned.
        label_map (dict): A mapping of special labels (e.g. '*1', '*2', '*3') to 
            atom indices (integers). The interpretation of each label depends on 
            `mol_type`.

    Raises:
        RuntimeError: If the molecule fails to embed or has no conformers.

    Returns:
        rdkit.Chem.rdchem.Mol: An RDKit molecule with embedded 3D coordinates and 
        assigned atom-level labels.
    """

    assert mol_type in ['reactant', 'product', 'ts'], "Type must be either 'reactant', 'product'"

    rmg_mol = arc_spc.mol.copy(deep=True)
    rmg_atoms = rmg_mol.atoms.copy()
    #print(rmg_atoms)
    rdkit_mol_mapped = rmg_mol.to_rdkit_mol(remove_h=False, save_order=True, return_mapping=True)
    rdkit_mol = rdkit_mol_mapped[0]

    # Validate XYZ
    xyz = arc_spc.get_xyz()
    # Embed molecule so we have a conformer to manipulate
    try:
        AllChem.EmbedMolecule(rdkit_mol)
    except Exception as e:
        raise RuntimeError(f"Could not embed the molecule: {e}")

    # If embedding was successful, set 3D positions from xyz['coords']
    if rdkit_mol.GetNumConformers() > 0:
        conf = rdkit_mol.GetConformer(id=0)
        for i, pos in enumerate(xyz['coords']):
            conf.SetAtomPosition(i, pos)
    else:
        raise RuntimeError("RDKit molecule has no conformers after embedding.")

    atom_props = {}
    electro_map = {}

    # Assign labels based on whether it's a reactant or product.
    # Each entry in label_map is something like: '*1' -> 3, '*2' -> 5, ...
    if mol_type == 'reactant':
        for label, atom_idx in label_map.items():
            if label == 'r1h_a':
                # if *1, then that atom_index is donator
                atom_props[atom_idx] = 'donator'
            elif label == 'r1h_h':
                # if *2, then that atom_index is d_hydrogen
                atom_props[atom_idx] = 'd_hydrogen'

                reorder = electronegativity.reorder_atoms_by_en(arc_spc, atom_idx)
                reorder_xyz = electronegativity.reorder_xyz(xyz, reorder)
                new_spc = ARCSpecies(label='r1h', xyz=reorder_xyz)
                new_map = reorder_xyz['map']
                zmat = electronegativity.xyz_to_zmat(new_spc.get_xyz(), mapping = new_map)

                electro_map = electronegativity.electro_mapping(zmat)
                

            # Feel free to add more label logic here if needed
    elif mol_type == 'product':
        for label, atom_idx in label_map.items():
            if label == 'r2h_a':
                # if *3, then that atom_index is acceptor
                atom_props[atom_idx] = 'acceptor'
            elif label == 'r2h_h':
                # if *2, then that atom_index is a_hydrogen
                atom_props[atom_idx] = 'a_hydrogen'

                reorder = electronegativity.reorder_atoms_by_en(arc_spc, atom_idx)
                reorder_xyz = electronegativity.reorder_xyz(xyz, reorder)
                new_spc = ARCSpecies(label='r1h', xyz=reorder_xyz)
                new_map = reorder_xyz['map']
                zmat = electronegativity.xyz_to_zmat(new_spc.get_xyz(), mapping = new_map)
                electro_map = electronegativity.electro_mapping(zmat)

            # Feel free to add more label logic here if needed
    elif mol_type == 'ts':
        for label, atom_idx in label_map.items():
            if label == '*0':
                atom_props[atom_idx] = '*0'
            elif label == '*1':
                atom_props[atom_idx] = '*1'
            elif label == '*2':
                atom_props[atom_idx] = '*2'
            elif label == '*3':
                atom_props[atom_idx] = '*3'
            elif label == '*4':
                atom_props[atom_idx] = '*4'


    mol_properties = {}
    # Set the RDKit atom property 'label' for each atom
    for atom in rdkit_mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in atom_props:
            atom.SetProp('label', atom_props[idx])
            atom.SetProp('atom_type', rmg_mol.atoms[idx].atomtype.label)
            mol_properties[idx] = {'label': atom_props[idx], 'atom_type': rmg_mol.atoms[idx].atomtype.label}
        else:
            # If no special label applies, you can either clear it or leave it empty
            atom.ClearProp('label')
    
    if mol_properties:
        rdkit_mol.SetProp('reaction', str(rxn))
        if mol_type == 'reactant':
            rdkit_mol.SetProp('type', 'r1h')
        elif mol_type == 'product':
            rdkit_mol.SetProp('type', 'r2h')
        elif mol_type == 'ts':
            rdkit_mol.SetProp('type', 'ts')
        rdkit_mol.SetProp('mol_properties', json.dumps(mol_properties))

    # Optionally, attach the electro_map as a property of the molecule.
    if electro_map:
        rdkit_mol.SetProp('electro_map', json.dumps(electro_map))

    return rdkit_mol
                
            
def species_props_to_sdf_file(idx, arc_species_list, label_map, sdf_file_path, ts_map):
    """
    Write a list of ARCSpecies objects to an SDF file.

    Args:
        idx (str): The index of the reaction.
        arc_species_list (list): A list of ARCSpecies objects.
        label_map (dict): A mapping of special labels to atom indices.
        sdf_file_path (str): The path to write the SDF file to.
        ts_map (dict): A mapping of special labels to atom indices for the TS.
    """

    # Create an RDKit molecule object for each ARCSpecies
    rdkit_mols = []
    for arc_spc in arc_species_list:
        if arc_spc.label == 'r1h':
            rdkit_mol = rdkit_conf_from_arcspc(idx, arc_spc, mol_type='reactant', label_map=label_map)
        elif arc_spc.label == 'r2h':
            rdkit_mol = rdkit_conf_from_arcspc(idx, arc_spc, mol_type='product', label_map=label_map)
        elif arc_spc.label == 'ts':
            rdkit_mol = rdkit_conf_from_arcspc(idx, arc_spc, mol_type='ts', label_map=ts_map)
        rdkit_mols.append(rdkit_mol)

    # Write the RDKit molecules to an SDF file
    w = SDWriter(sdf_file_path)
    for mol in rdkit_mols:
        w.write(mol)
    w.close()
    
    return sdf_file_path

def create_sdf(connections_file_path: str, save_sdf_path: str):
    connection_df = read_connections(connections_file_path)
    for idx, row in connection_df.iterrows():
        print(f"Processing {idx}")
        r1h_spc = ARCSpecies(label='r1h', xyz=row['r1h'])
        r2h_spc = ARCSpecies(label='r2h',  xyz=row['r2h'])
        ts_spc = ARCSpecies(label='ts', xyz=row['ts'], is_ts=True)
        species_props_to_sdf_file(idx, [r1h_spc, r2h_spc, ts_spc], label_map=row['index_dict'],ts_map=row['ts_results'], sdf_file_path = save_sdf_path + f"/{idx}.sdf")

if __name__ == '__main__':
    create_sdf('/home/calvin/code/chemprop_phd_customised/habnet/data/preprocessing/connect_reactions_data/connected_reactions.csv', './data/processed/sdf_files/')
