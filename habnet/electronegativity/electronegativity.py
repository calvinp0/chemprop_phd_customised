from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union, Any
import os
from arc.common import ARC_PATH, key_by_val, read_yaml_file
import heapq
from arc.species.species import ARCSpecies
from arc.species.converter import compare_zmats, relocate_zmat_dummy_atoms_to_the_end, zmat_from_xyz
from arc.species.zmat import consolidate_zmat, get_connectivity, _add_nth_atom_to_zmat

import math
import numpy as np
import operator
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from rmgpy.molecule.molecule import Molecule

from arc.common import get_logger, is_angle_linear, key_by_val, determine_top_group_indices
from arc.exceptions import ZMatError
from arc.species.vectors import calculate_param, get_vector_length

if TYPE_CHECKING:
    from rmgpy.molecule.molecule import Atom


logger = get_logger()

DEFAULT_CONSOLIDATION_R_TOL = 1e-4
DEFAULT_CONSOLIDATION_A_TOL = 1e-3
DEFAULT_CONSOLIDATION_D_TOL = 1e-3
DEFAULT_COMPARISON_R_TOL = 0.01  # Angstrom
DEFAULT_COMPARISON_A_TOL = 2.0  # degrees
DEFAULT_COMPARISON_D_TOL = 2.0  # degrees
TOL_180 = 0.9  # degrees
KEY_FROM_LEN = {2: 'R', 3: 'A', 4: 'D'}

def get_neighbors_by_electronegativity(spc: 'ARCSpecies',
                                       atom_index: int,
                                       exclude_index: int,
                                       two_neighbors: bool = True) -> Tuple[int, List[int]]:
    """
    Retrieve the top two neighbors of a given atom, sorted by their effective electronegativity.
    The effective electronegativity for a neighbor is calculated as:
        EN (of neighbor) * bond order
    Ties are broken by comparing the sum of the effective electronegativities of the neighbor's bonded atoms.
    """
    # Get all neighbors except the excluded atom.
    neighbors = [neighbor for neighbor in spc.mol.atoms[atom_index].edges.keys()
                 if spc.mol.atoms.index(neighbor) != exclude_index]

    if not neighbors:
        raise ValueError(f"Atom at index {atom_index} has no valid neighbors.")

    electronegativities = read_yaml_file(os.path.join(ARC_PATH, 'data', 'electronegativity.yml'))

    def get_neighbor_total_electronegativity(neighbor: 'Atom') -> float:
        return sum(
            electronegativities[n.symbol] * neighbor.edges[n].order
            for n in neighbor.edges.keys()
        )

    # Calculate primary and secondary values for each neighbor.
    effective_electronegativities = [
        (
            electronegativities[n.symbol] * spc.mol.atoms[atom_index].edges[n].order,  # primary effective EN
            get_neighbor_total_electronegativity(n),                                   # secondary sum of neighbor's EN
            spc.mol.atoms.index(n)                                                       # tertiary: atom index
        )
        for n in neighbors
    ]
    # Sort in descending order: higher primary value, then higher secondary.
    effective_electronegativities.sort(reverse=True, key=lambda x: (x[0], x[1]))
    sorted_neighbors = [item[2] for item in effective_electronegativities]

    most_electronegative = sorted_neighbors[0]
    remaining_neighbors = sorted_neighbors[1:] if two_neighbors else []
    return most_electronegative, remaining_neighbors

def reorder_atoms_by_en(spc: 'ARCSpecies', start_index: int) -> List[int]:
    """
    Reorder all atoms in the molecule based on a best-first search that prioritizes atoms with higher effective electronegativity.
    For each atom, the effective electronegativity is calculated as:
        atomic EN * bond order
    Ties are broken by comparing the sum of the effective electronegativities of the atom's bonded neighbors.
    """
    electronegativities = read_yaml_file(os.path.join(ARC_PATH, 'data', 'electronegativity.yml'))
    
    def effective_en(atom: 'Atom', bond_order: float) -> float:
        return electronegativities[atom.symbol] * bond_order
    
    def neighbor_total_en(atom: 'Atom') -> float:
        return sum(electronegativities[n.symbol] * atom.edges[n].order for n in atom.edges.keys())
    
    visited = set()
    ordering = []
    
    # The priority queue elements are tuples of:
    # (negative primary effective EN, negative secondary total EN, atom_index)
    # Negative values are used to simulate a max-heap with Python's min-heap implementation.
    queue = []
    
    visited.add(start_index)
    ordering.append(start_index)
    
    # Initialize the queue with all neighbors of the starting atom.
    current_atom = spc.mol.atoms[start_index]
    for neighbor in current_atom.edges.keys():
        neighbor_index = spc.mol.atoms.index(neighbor)
        if neighbor_index not in visited:
            bond_order = current_atom.edges[neighbor].order
            primary = effective_en(neighbor, bond_order)
            secondary = neighbor_total_en(neighbor)
            heapq.heappush(queue, (-primary, -secondary, neighbor_index))
    
    while queue:
        neg_primary, neg_secondary, atom_index = heapq.heappop(queue)
        if atom_index in visited:
            continue
        visited.add(atom_index)
        ordering.append(atom_index)
        current_atom = spc.mol.atoms[atom_index]
        for neighbor in current_atom.edges.keys():
            neighbor_index = spc.mol.atoms.index(neighbor)
            if neighbor_index not in visited:
                bond_order = current_atom.edges[neighbor].order
                primary = effective_en(neighbor, bond_order)
                secondary = neighbor_total_en(neighbor)
                heapq.heappush(queue, (-primary, -secondary, neighbor_index))
                
    return ordering

def reorder_xyz(xyz, ordering):
    """
    Reorder the xyz dictionary based on the provided ordering.
    
    Args:
        xyz (dict): A dictionary with keys 'symbols', 'isotopes', and 'coords'.
        ordering (list): A list of indices specifying the new order.
    
    Returns:
        dict: A new xyz dictionary with reordered values.
    """
    new_symbols = tuple(xyz['symbols'][i] for i in ordering)
    new_isotopes = tuple(xyz['isotopes'][i] for i in ordering)
    new_coords = tuple(xyz['coords'][i] for i in ordering)
    new_mapping = {new_idx: original_idx for new_idx, original_idx in enumerate(ordering)}


    return {'symbols': new_symbols, 'isotopes': new_isotopes, 'coords': new_coords, 'map': new_mapping}

def xyz_to_zmat(xyz: Dict[str, tuple],
                mol: Optional[Molecule] = None,
                constraints: Optional[Dict[str, List[Tuple[int, ...]]]] = None,
                consolidate: bool = True,
                consolidation_tols: Dict[str, float] = None,
                fragments: Optional[List[List[int]]] = None,
                mapping: Optional[Dict[int, int]] = None) -> Dict[str, tuple]:

    """
    Generate a z-matrix from cartesian coordinates.
    The zmat is a dictionary with the following keys:
      - 'symbols': a tuple of atomic symbols.
      - 'coords': a tuple of tuples representing internal coordinates.
      - 'vars': a dictionary of all variables defined for the coordinates.
      - 'map': a dictionary connecting atom indices in the zmat (keys) to atom indices in the mol/coords (values).
    This function assumes ``xyz`` has no dummy atoms and does not attempt to resolve constraint locks.
    
    Args:
        xyz (dict): The xyz coordinates.
        mol (Molecule, optional): The corresponding RMG Molecule. If given, bonding information is used
                                  to generate a more meaningful zmat.
        constraints (dict, optional): Constraints for distances ('R'), angles ('A'), and dihedrals ('D').
        consolidate (bool, optional): Whether to consolidate the zmat after generation.
        consolidation_tols (dict, optional): Absolute tolerances for consolidating almost equal internal coordinates.
        fragments (List[List[int]], optional): Fragments represented as lists of atom indices.
        
    Raises:
        ZMatError: If the zmat could not be generated.
        
    Returns:
        Dict[str, tuple]: The generated z-matrix.
    """
    fragments = fragments or [list(range(len(xyz['symbols'])))]
    constraints = constraints or dict()
    
    if mol is None and any('group' in key for key in constraints.keys()):
        raise ZMatError(f'Cannot generate a constrained zmat without mol. Got mol=None and constraints=\n{constraints}')
    
    for constraint_list in constraints.values():
        for constraint_tuple in constraint_list:
            for index in constraint_tuple:
                if mol is not None and index >= len(mol.atoms):
                    raise ZMatError(f'Constraint containing atom index {index} is invalid for a molecule with only '
                                    f'{len(mol.atoms)} atoms:\n{constraints}')
                if index >= len(xyz['symbols']):
                    raise ZMatError(f'Constraint containing atom index {index} is invalid for coordinates with only '
                                    f'{len(xyz["symbols"])} atoms:\n{constraints}')
    
    xyz = xyz.copy()
    zmat = {'symbols': list(), 'coords': list(), 'vars': dict(), 'map': dict()}
    
    # Instead of computing an order with get_atom_order, we simply use the order from xyz['symbols']
    atom_order = list(range(len(xyz['symbols'])))
    
    connectivity = get_connectivity(mol=mol) if mol is not None else None
    skipped_atoms = list()  # Atoms for which constraints are applied
    for atom_index in atom_order:
        zmat, xyz, skipped = _add_nth_atom_to_zmat(
            zmat=zmat,
            xyz=xyz,
            connectivity=connectivity,
            n=len(zmat['symbols']),
            atom_index=atom_index,
            constraints=constraints,
            fragments=fragments,
        )
        skipped_atoms.extend(skipped)

    while len(skipped_atoms):
        num_of_skipped_atoms = len(skipped_atoms)
        indices_to_pop = list()
        for i, atom_index in enumerate(skipped_atoms):
            zmat, xyz, skipped = _add_nth_atom_to_zmat(
                zmat=zmat,
                xyz=xyz,
                connectivity=connectivity,
                n=len(zmat['symbols']),
                atom_index=atom_index,
                constraints=constraints,
                fragments=fragments,
            )
            if not skipped:
                indices_to_pop.append(i)
        for i in reversed(indices_to_pop):
            skipped_atoms.pop(i)
        if num_of_skipped_atoms == len(skipped_atoms):
            raise ZMatError(f"Could not generate the zmat, skipped atoms could not be assigned, "
                            f"there's probably a constraint lock. Partial zmat:\n{zmat}\n\nSkipped atoms:\n{skipped_atoms}.")

    if consolidate and not constraints:
        try:
            zmat = consolidate_zmat(zmat, mol, consolidation_tols)
        except (KeyError, ZMatError) as e:
            logger.error(f'Could not consolidate zmat, got: {e.__class__.__name__}: {str(e)}')
            logger.error('Generating zmat without consolidation.')

    zmat['symbols'] = tuple(zmat['symbols'])
    zmat['coords'] = tuple(zmat['coords'])
    zmat['map'] = mapping
    return zmat


def electro_mapping(zmat):

    electro_map = {}

    for orig_idx, new_idx in zmat['map'].items():
        r_label, a_label, d_label = zmat['coords'][orig_idx]
        r_value = zmat['vars'][r_label] if r_label is not None else None
        a_value = zmat['vars'][a_label] if a_label is not None else None
        d_value = zmat['vars'][d_label] if d_label is not None else None

        electro_map[str(new_idx)] = {'R': r_value, 'A': a_value, 'D': d_value}
    
    return electro_map
