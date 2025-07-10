import os
import ast

from rdkit import Chem
import pandas as pd

from arc.species.vectors import calculate_dihedral_angle, calculate_angle, calculate_distance
from arc.species import ARCSpecies

def read_sdf(path, mol_type='ts', removeHs=False):
    """
    Read an sdf file and return a list of rdkit molecules

    Args:
        path (str): Path to the sdf file
        mol_type (str): Type of molecule to return. Must be one of ['all', 'ts', 'r1h', 'r2h']
        removeHs (bool): Whether to remove hydrogens from the molecule

    Returns:
        list: List of rdkit molecules
    """
    assert mol_type in ['all', 'ts', 'r1h', 'r2h'], f"mol_type must be one of ['all', 'ts', 'r1h', 'r2h']"
    suppl = Chem.SDMolSupplier(path, removeHs=removeHs)
    
    if mol_type == 'all':
        mols = [mol for mol in suppl]
        return mols
    else:
        mols = [mol for mol in suppl if mol.GetProp('type') == mol_type]
        return mols
    
def get_xyz(mol):
    """
    Get the xyz coordinates of a molecule

    Args:
        mol (rdkit.Chem.rdchem.Mol): The molecule

    Returns:
        str: The xyz coordinates of the molecule
    """
    xyz = Chem.MolToXYZBlock(mol)
    # Remove the first 2 lines
    xyz = '\n'.join(xyz.split('\n')[2:])
    return xyz

def get_reaction_id(mol):
    """
    Get the reaction ID of a molecule

    Args:
        mol (rdkit.Chem.rdchem.Mol): The molecule

    Returns:
        str: The reaction ID
    """
    return mol.GetProp('reaction')

def get_atoms_stars(mol):

    #>  <mol_properties>  (3) 
    #{"0": {"label": "*1", "atom_type": "N3s"}, "1": {"label": "*0", "atom_type": "N3s"}, "5": {"label": "*4", "atom_type": "Cs"}, "6": {"label": "*3", "atom_type": "N3s"}, "12": {"label": "*2", "atom_type": "H0"}}

    mol_properties = mol.GetProp('mol_properties')
    # Make the label the key and the atom index the value
    mol_properties = ast.literal_eval(mol_properties)
    mol_properties = dict(sorted({v['label']: k for k, v in mol_properties.items()}.items()))
    return mol_properties

def get_radius(arc_spc, atoms: list, sub_value: int):

    coords = arc_spc.get_xyz()

    if None not in atoms:
        return calculate_distance(coords, atoms)
    else:
        return sub_value

def get_dihedral(arc_spc, atoms: list, sub_value: int):
    """
    Find the alpha dihedral angle in a molecule

    Args:
        arc_spc (ARCSpecies): The ARC species
        atoms (list): List of atoms to find the dihedral angle for

    Returns:
        float: The alpha dihedral angle
    """
    coords = arc_spc.get_xyz()

    if None not in atoms:
        return calculate_dihedral_angle(coords, atoms)
    else:
        return sub_value

def get_angle(arc_spc, atoms: list, sub_value: int):
    """
    Find the angle in a molecule

    Args:
        arc_spc (ARCSpecies): The ARC species
        atoms (list): List of atoms to find the angle for

    Returns:
        float: The angle
    """
    coords = arc_spc.get_xyz()

    if None not in atoms:
        return calculate_angle(coords, atoms)
    else:
        return sub_value


def extract_internal_coords(mol):

    
    arc_spc = ARCSpecies(label='temp', xyz=get_xyz(mol))
    
    atom_stars = get_atoms_stars(mol)

    alpha_keys = ['*1','*2', '*3']
    r1_keys = ['*1','*2']
    r2_keys = ['*2', '*3']
    #beta_keys = ['*1','*2', '*3']
    #gamma_keys = ['*0','*1', '*2']
    psi_2_keys = ['*1','*2', '*3', '*4'] # Ditto
    psi_1_keys = ['*0','*1', '*2', '*3'] # Pivot of ps1 is r1 [*1, *2]

    r1_atoms = [int(atom_stars[key]) if key in atom_stars else None for key in r1_keys]
    r2_atoms = [int(atom_stars[key]) if key in atom_stars else None for key in r2_keys]
    alpha_atoms = [int(atom_stars[key]) if key in atom_stars else None for key in alpha_keys]
    #beta_atoms  = [int(atom_stars[key]) if key in atom_stars else None for key in beta_keys]
    #gamma_atoms = [int(atom_stars[key]) if key in atom_stars else None for key in gamma_keys]
    psi_1_atoms = [int(atom_stars[key]) if key in atom_stars else None for key in psi_1_keys]
    psi_2_atoms = [int(atom_stars[key]) if key in atom_stars else None for key in psi_2_keys]

    r1_radius = get_radius(arc_spc, r1_atoms, -10)
    r2_radius = get_radius(arc_spc, r2_atoms, -10)
    alpha_angle = get_angle(arc_spc, alpha_atoms, -10)
    # beta_angle = get_angle(arc_spc, beta_atoms, -10)
    # gamma_angle = get_angle(arc_spc, gamma_atoms, -10)
    psi_1_dihedral = get_dihedral(arc_spc, psi_1_atoms, -10)
    psi_2_dihedral = get_dihedral(arc_spc, psi_2_atoms, -10)

    # convert to atom-index lists (or None)
    def idxs(keys):
        return [atom_stars.get(k) for k in keys]

    r1_atoms    = idxs(r1_keys)
    r2_atoms    = idxs(r2_keys)
    alpha_atoms = idxs(alpha_keys)
    psi1_atoms  = idxs(psi_1_keys)
    psi2_atoms  = idxs(psi_2_keys)

    # basic internal coords
    r1_radius     = get_radius(arc_spc, r1_atoms, -10)
    r2_radius     = get_radius(arc_spc, r2_atoms, -10)
    alpha_angle   = get_angle(arc_spc, alpha_atoms, -10)
    psi1_dihedral = get_dihedral(arc_spc, psi1_atoms, -10)
    psi2_dihedral = get_dihedral(arc_spc, psi2_atoms, -10)

    # NEW: distance from *4 to neighbor of *0 (excluding *1)
    idx0 = atom_stars.get('*0')
    idx1 = atom_stars.get('*1')
    idx3 = atom_stars.get('*3')
    idx4 = atom_stars.get('*4')

    # neighbor of *0
    if idx0 is not None:
        nbrs0 = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(int(idx0)).GetNeighbors()]
        nb0_exc1 = next((n for n in nbrs0 if n != idx1), None)
    else:
        nb0_exc1 = None

    # neighbor of *4
    if idx4 is not None:
        nbrs4 = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(int(idx4)).GetNeighbors()]
        nb4_exc3 = next((n for n in nbrs4 if n != idx3), None)
    else:
        nb4_exc3 = None

    dist_4_to_nb0 = get_radius(arc_spc, [idx4, nb0_exc1], -10)
    dist_0_to_nb4 = get_radius(arc_spc, [idx0, nb4_exc3], -10)




    return r1_radius, r2_radius, alpha_angle, psi_1_dihedral, psi_2_dihedral


def create_target(path_to_sdfs, target_data_path, target_errors_path):
    
    target_data = pd.DataFrame(columns=['rxn', 'r1_radius', 'r2_radius', 'alpha_angle', 'psi_1_dihedral', 'psi_2_dihedral'])
    target_errors = pd.DataFrame(columns=['rxn', 'error'])
    for file in os.listdir(path_to_sdfs):
        # Get file name without extension
        try:
            print(f"Processing {file}")
            mol = read_sdf(os.path.join(path_to_sdfs, file), mol_type='ts', removeHs=False)[0]
            r1_radius, r2_radius, alpha_angle, psi_1_dihedral, psi_2_dihedral = extract_internal_coords(mol)

            # Concatenate the data - can't append as it is deprecated
            target_data = pd.concat(
        [
            target_data, 
            pd.DataFrame(
                {
                    'rxn': get_reaction_id(mol),
                    'r1_radius': r1_radius,
                    'r2_radius': r2_radius,
                    'alpha_angle': alpha_angle,
                    'psi_1_dihedral': psi_1_dihedral,
                    'psi_2_dihedral': psi_2_dihedral
                },
                index=[0]
            )
        ],
        ignore_index=True
    )
        except Exception as e:
            print(f"**{get_reaction_id(mol)} Error: {e}**")
            target_errors = pd.concat(
                [
                    target_errors,
                    pd.DataFrame(
                        {
                            'rxn': get_reaction_id(mol),
                            'error': str(e)
                        },
                        index=[0]
                    )
                ],
                ignore_index=True
            )

    target_data.to_csv(target_data_path, index=False)
    print(f"Reactions successfully processed: {len(target_data)}")
    print(f"Errors: {len(target_errors)}")
    target_errors.to_csv(target_errors_path, index=False)

if __name__ == '__main__':
    create_target('/home/calvin/code/chemprop_phd_customised/habnet/data/processed/sdf_data', '/home/calvin/code/chemprop_phd_customised/habnet/data/processed/target_data.csv', '/home/calvin/code/chemprop_phd_customised/habnet/data/processed/target_errors.csv')
