###
#
# Note: Assuming that when implememnted into ARC, Heuristics will use `get_reaction_family_product` and take the first result
#
###
from rdkit import Chem
from arc.species.species import ARCSpecies
from arc.reaction.reaction import ARCReaction
from arc.common import generate_resonance_structures
from arc.family.family import ReactionFamily, get_reaction_family_products
from arc.exceptions import InputError, RotorError, SpeciesError, TSError
import pandas as pd
import yaml
from tqdm import tqdm
import string
import os
import logging
import re
from typing import List, Optional, Tuple

PARSED_DATA_PATH = "data/parsed_reactions.csv"

script_dir = os.path.dirname(__file__)
logs_dir = os.path.join(script_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
log_file = os.path.join(logs_dir, "final_order_reactions_rxnfamily.log")
logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    filemode="w",  # Overwrite the log file on each run
    format="%(asctime)s - %(levelname)s - %(message)s"  # Include timestamps and log levels
)

logger = logging.getLogger(__name__)
thermo_data_all = []
valid_chars = "-_[]=.,%s%s" % (string.ascii_letters, string.digits)

def check_label(label: str,
                is_ts: bool = False,
                underscore_rule: bool = True,
                verbose: bool = False,
                ) -> Tuple[str, Optional[str]]:
    """
    Check whether a species (or reaction) label is legal, modify it if needed.

    Args:
        label (str): A label.
        is_ts (bool, optional): Whether the species label belongs to a transition state.
        verbose (bool, optional): Whether to log errors.

    Raises:
        TypeError: If the label is not a string type.
        SpeciesError: If label is illegal and cannot be automatically fixed.

    Returns: Tuple[str, Optional[str]]
        - A legal label.
        - The original label if the label was modified, else ``None``.
    """
    if not isinstance(label, str):
        raise TypeError(f'A species label must be a string type, got {label} which is a {type(label)}.')
    if label[:2] == 'TS' and all(char.isdigit() for char in label[2:]) and not is_ts:
        raise SpeciesError(f'A non-TS species cannot be named "TS" with a subsequent index, got {label}')

    if underscore_rule:
        transformed_label = re.sub(r'\[([^]]+)\]', r'_\1_', label)
        modified = transformed_label != label
        label = transformed_label

    char_replacement = {'#': 't',
                        '=': 'd',
                        '(': '[',
                        ')': ']',
                        ' ': '_',
                        '%': 'c',
                        '$': 'd',
                        '*': 's',
                        '@': 'a',
                        '+': 'p',
                        }
    modified = False
    original_label = label
    for char in original_label:
        if char not in valid_chars:
            if verbose:
                logger.error(f'Label {label} contains an invalid character: "{char}"')
            if char in char_replacement.keys():
                label = label.replace(char, char_replacement[char])
            else:
                label = label.replace(char, '_')
            modified = True
    if modified:
        if verbose:
            logger.warning(f'Replaced species label.\n'
                           f'Original label was: "{original_label}".\n'
                           f'New label is: "{label}"')
    else:
        original_label = None
    return label, original_label


def parse_reactions_if_needed(input_csv, converged_rxn_dirs: list, output_csv=PARSED_DATA_PATH, force=False):
    """
    Only parse data if parsed file doesn't exist or if force=True.
    """
    if not force and os.path.exists(output_csv):
        print(f"Found existing parsed data at '{output_csv}', skipping parse...")
        return pd.read_csv(output_csv)
    else:
        print("Parsing data...")
        df_in = pd.read_csv(input_csv)
        parsed_df = parse_reactions(df_in, converged_rxn_dirs=converged_rxn_dirs)  # your actual function
        # Set the index as a column called 'rxn' and then reset the index
        parsed_df['rxn'] = parsed_df.index
        parsed_df.reset_index(drop=True, inplace=True)
        parsed_df.to_csv(output_csv, index=False)
        return parsed_df

def read_csv(path):
    """
    :param path: str
    :return: pd.DataFrame
    """
    dataframe = pd.read_csv(path)
    if 'rxn' not in list(dataframe.columns) and 'reaction' in list(dataframe.columns):
        dataframe['reaction'] = dataframe['reaction'].str.replace('reaction_', 'rxn_')
    elif 'rxn' in list(dataframe.columns) and 'reaction' in list(dataframe.columns):
        dataframe.drop('reaction', axis=1, inplace=True)

    return dataframe


def yml_parse(file):
    with open(file, 'r') as file:
        yml = yaml.load(file, Loader=yaml.FullLoader)
        # grab the geometry
        xyz = yml['xyz']
        # remove the first two lines
        xyz = xyz.split("\n")[2:]
        # # Check if list in list
        # if any(isinstance(i, list) for i in xyz):
        #     xyz = [item for sublist in xyz for item in sublist]

    return xyz


def xyz_parse(file: str):
    with open(file, 'r') as f:
        xyz = f.read()
        # Read from line 3 onwards
        xyz = xyz.split("\n")[2:]
        xyz = list(filter(None, xyz))

    return xyz


def determine_mol_type(folder_name: str) -> str:
    """
    :param folder_name: str
    :return: str
    """
    if '_r1_' in folder_name:
        mol_type = 'r0'
    elif '_r2_' in folder_name:
        mol_type = 'r1'
    elif '_p1_' in folder_name:
        mol_type = 'p0'
    elif '_p2_' in folder_name:
        mol_type = 'p1'
    else:
        mol_type = None

    return mol_type


def match_label(reaction_input_species: dict, species: str) -> str:
    """
    :param reaction_input: dict
    :param species: dict
    :return: str
    """
    matching_label = next((entry['label'] for entry in reaction_input_species if entry['smiles'] == species), None)

    return matching_label


def monoatomic_atoms(molecule: str):
    """
    Check if the molecule is monoatomic
    """

    mono_atomic = {
        '[H]': [['H       0.00000000    0.00000000    0.00000000']],
        '[C]': [['C       0.00000000    0.00000000    0.00000000']],
        '[N]': [['N       0.00000000    0.00000000    0.00000000']],
        '[O]': [['O       0.00000000    0.00000000    0.00000000']],
        '[Cl]': [['Cl       0.00000000    0.00000000    0.00000000']],
        '[F]': [['F       0.00000000    0.00000000    0.00000000']],
        '[S]': [['S       0.00000000    0.00000000    0.00000000']],
    }

    try:
        return mono_atomic[molecule]
    except KeyError:
        return molecule


from functools import lru_cache


def resonance_match(smiles_1, smiles_2, return_smiles: Optional[str] = None):
    arc_spc_1 = ARCSpecies(label="smiles_1", smiles=smiles_1)
    arc_spc_2 = ARCSpecies(label="smiles_2", smiles=smiles_2)

    res_arc_1 = generate_resonance_structures(arc_spc_1.mol)
    res_arc_2 = generate_resonance_structures(arc_spc_2.mol)

    res_arc_1_smiles = [mol.to_smiles() for mol in res_arc_1]
    res_arc_2_smiles = [mol.to_smiles() for mol in res_arc_2]

    if sorted(res_arc_1_smiles) == sorted(res_arc_2_smiles):
        if return_smiles == "first":
            return smiles_1
        elif return_smiles == "last":
            return smiles_2
        elif return_smiles == "all":
            return res_arc_1_smiles
        elif return_smiles is None:
            return True
        else:
            raise ValueError("Incorrect 'return_smiles' option - 'first', 'last', 'all' or None")
    else:
        # Attempt InChI
        import rdkit
        if rdkit.Chem.MolToInchi(rdkit.Chem.MolFromSmiles(smiles_1)) == rdkit.Chem.MolToInchi(
                rdkit.Chem.MolFromSmiles(smiles_2)):
            logger.info(f"Using InChi check as last resort - {smiles_1} and {smiles_2}")
            if return_smiles == "first":
                return smiles_1
            elif return_smiles == "last":
                return smiles_2
            elif return_smiles == "all":
                return res_arc_1_smiles
            elif return_smiles is None:
                return True
            else:
                raise ValueError("Incorrect 'return_smiles' option - 'first', 'last', 'all' or None")
        else:
            return False


def validate_roles(role_tuple, tuple_name="Tuple"):
    """
    Validates that the role_tuple contains:
    - Exactly one role matching the X_H pattern.
    - Exactly one role matching the Y_rad pattern.

    The validation is performed in two attempts:
    1. First attempt uses the original patterns:
       - X_H role: Starts with 'X' and ends with 'H'.
       - Y_rad role: Starts with 'Y' and ends with 'rad'.
    2. If the first attempt fails, a second attempt is made using alternative patterns:
       - X_H role: Ends with '_H'.
       - Y_rad role: Starts with 'Y' and ends with 'rad' (same as before).

    Parameters:
    - role_tuple (tuple): The tuple containing role strings to validate.
    - tuple_name (str): A descriptive name for the tuple (e.g., "Reactants", "Products").

    Raises:
    - ValueError: If the roles do not match the required patterns.
    """
    # First attempt patterns
    pattern_X_H_first = re.compile(r'^X.*H$')  # Starts with 'X', ends with 'H'
    pattern_Y_rad = re.compile(r'^Y.*rad$')  # Starts with 'Y', ends with 'rad'

    # Initialize counts
    count_X_H = 0
    count_Y_rad = 0

    # List to collect unexpected roles
    unexpected_roles = []

    # First attempt
    for role in role_tuple:
        if pattern_X_H_first.match(role):
            count_X_H += 1
        elif pattern_Y_rad.match(role):
            count_Y_rad += 1
        else:
            unexpected_roles.append(role)

    # Check counts
    if count_X_H == 1 and count_Y_rad == 1:
        # Successful validation with first attempt
        if unexpected_roles:
            logger.info(f"Warning: Unexpected role(s) {set(unexpected_roles)} found in {tuple_name} {role_tuple}. "
                        f"However, the tuple contains the required valid patterns.")
        else:
            logger.info(f"All roles in {tuple_name} are valid.")
        return  # Validation successful

    # If first attempt fails, try second patterns
    # Reset counts and unexpected_roles
    count_X_H = 0
    count_Y_rad = 0
    unexpected_roles = []

    # Second attempt patterns
    pattern_X_H_second = re.compile(r'.*_H$')  # Ends with '_H'
    # Note: pattern_Y_rad remains the same

    for role in role_tuple:
        if pattern_X_H_second.match(role):
            count_X_H += 1
        elif pattern_Y_rad.match(role):
            count_Y_rad += 1
        else:
            unexpected_roles.append(role)

    # Check counts after second attempt
    if count_X_H == 1 and count_Y_rad == 1:
        # Successful validation with second attempt
        logger.info(f"Roles in {tuple_name} are valid after applying alternative patterns.")
        if unexpected_roles:
            logger.info(f"Warning: Unexpected role(s) {set(unexpected_roles)} found in {tuple_name} {role_tuple}. "
                        f"However, the tuple contains the required valid patterns.")
    else:
        # Collect errors
        errors = []
        if count_X_H != 1:
            errors.append(f"Expected exactly one role matching X_H pattern in {tuple_name}, but found {count_X_H}.")
        if count_Y_rad != 1:
            errors.append(f"Expected exactly one role matching Y_rad pattern in {tuple_name}, but found {count_Y_rad}.")
        error_message = " ".join(errors) + f" Roles provided: {set(role_tuple)}."
        raise ValueError(error_message)


def categorize_roles(role_tuple, tuple_name="Tuple"):
    """
    Categorizes roles into 'X_H' and 'Y_rad' based on predefined patterns.
    
    The categorization is performed in four steps:
    1. First attempt uses the original patterns:
       - X_H role: Starts with 'X' and ends with 'H'.
       - Y_rad role: Starts with 'Y' and ends with 'rad'.
    2. Second attempt uses alternative patterns:
       - X_H role: Ends with '_H'.
       - Y_rad role: Starts with 'Y' and ends with 'rad' (same as before).
    3. Third step assumes the other role if only one is filled after two attempts.
       - Special handling for roles like 'N_atom_quartet' or 'N_atom_doublet', which are assumed as Y_rad.
    4. If roles still cannot be categorized, raises a ValueError identifying missing categories.
    
    Parameters:
    - role_tuple (tuple): The tuple containing role strings to categorize.
    - tuple_name (str): A descriptive name for the tuple (e.g., "Reactants", "Products").
    
    Returns:
    - dict: A dictionary mapping role categories to their corresponding role strings.
    
    Raises:
    - ValueError: If multiple roles match the same category or no valid roles are found.
    """
    import re

    # Define regex patterns for the first attempt
    pattern_X_H_first = re.compile(r'^X.*H$')  # Starts with 'X' and ends with 'H'
    pattern_Y_rad = re.compile(r'^Y.*rad$')  # Starts with 'Y' and ends with 'rad'

    # Define regex patterns for the second attempt
    pattern_X_H_second = re.compile(r'.*_H$')  # Ends with '_H'
    # Y_rad pattern remains the same

    # Define additional assumptions for third step
    assumed_Y_rad_patterns = re.compile(r'N_atom_(quartet|doublet)$')  # Matches specific cases

    # Initialize categorized roles
    categorized_roles = {'X_H': None, 'Y_rad': None}
    unexpected_roles = []

    # First Attempt
    for role in role_tuple:
        if pattern_X_H_first.match(role):
            if categorized_roles['X_H'] is not None:
                raise ValueError(
                    f"Multiple roles matching 'X_H' pattern found in {tuple_name}: {role_tuple}"
                )
            categorized_roles['X_H'] = role
        elif pattern_Y_rad.match(role):
            if categorized_roles['Y_rad'] is not None:
                raise ValueError(
                    f"Multiple roles matching 'Y_rad' pattern found in {tuple_name}: {role_tuple}"
                )
            categorized_roles['Y_rad'] = role
        else:
            unexpected_roles.append(role)

    # Check if both categories are filled
    if all(value is not None for value in categorized_roles.values()):
        if unexpected_roles:
            logger.info(
                f"Unexpected role(s) {set(unexpected_roles)} found in {tuple_name} {role_tuple}. "
                f"However, the required patterns are satisfied."
            )
        else:
            logger.info(f"All roles in {tuple_name} are categorized correctly using original patterns.")
        return categorized_roles

    # Second Attempt
    categorized_roles = {'X_H': None, 'Y_rad': None}
    unexpected_roles = []

    for role in role_tuple:
        if pattern_X_H_second.match(role):
            if categorized_roles['X_H'] is not None:
                raise ValueError(
                    f"Multiple roles matching alternative 'X_H' pattern found in {tuple_name}: {role_tuple}"
                )
            categorized_roles['X_H'] = role
        elif pattern_Y_rad.match(role):
            if categorized_roles['Y_rad'] is not None:
                raise ValueError(
                    f"Multiple roles matching 'Y_rad' pattern found in {tuple_name}: {role_tuple}"
                )
            categorized_roles['Y_rad'] = role
        else:
            unexpected_roles.append(role)

    # Check if both categories are filled after second attempt
    if all(value is not None for value in categorized_roles.values()):
        logger.info(f"Roles in {tuple_name} are categorized correctly using alternative patterns.")
        if unexpected_roles:
            logger.info(
                f"Unexpected role(s) {set(unexpected_roles)} found in {tuple_name} {role_tuple}. "
                f"However, the required patterns are satisfied."
            )
        return categorized_roles

    # Third Step: Handle missing roles and assumptions
    if categorized_roles['X_H'] is not None and categorized_roles['Y_rad'] is None:
        # Assume unexpected roles as Y_rad if they match specific patterns
        for role in unexpected_roles:
            if assumed_Y_rad_patterns.match(role):
                categorized_roles['Y_rad'] = role
                logger.info(f"Assuming role '{role}' as 'Y_rad' in {tuple_name}.")
                break
        else:
            categorized_roles['Y_rad'] = 'Assumed_Y_rad'
            logger.info(f"Only 'X_H' is categorized in {tuple_name}, assuming the other role is 'Y_rad'.")
    elif categorized_roles['Y_rad'] is not None and categorized_roles['X_H'] is None:
        categorized_roles['X_H'] = 'Assumed_X_H'
        logger.info(f"Only 'Y_rad' is categorized in {tuple_name}, assuming the other role is 'X_H'.")
    else:
        # If still not categorized properly, identify missing categories
        missing_categories = [
            key for key, value in categorized_roles.items() if value is None
        ]
        error_message = (
            f"Missing role(s) for categories {missing_categories} in {tuple_name} {role_tuple}. "
            f"Expected one role starting with 'X' and ending with 'H', "
            f"and another starting with 'Y' and ending with 'rad'."
        )
        raise ValueError(error_message)

    return categorized_roles



def parse_opt(species_folder, folder, smiles):
    opt_coords = []
    for files in os.listdir(os.path.join(species_folder, folder)):
        if files.endswith(".yml"):
            xyz = yml_parse(os.path.join(species_folder, folder, files))
            opt_coords.append(xyz)
    if not opt_coords:
        geometry_folder = os.path.join(species_folder, folder, "geometry")
        if os.path.isdir(geometry_folder):
            for file in os.listdir(geometry_folder):
                if file.endswith(".xyz"):
                    xyz = xyz_parse(os.path.join(geometry_folder, file))
                    opt_coords.append(xyz)
    if not opt_coords:
        opt_coords = monoatomic_atoms(smiles)

    # Find frequency output path
    freq_path = None
    noBac = os.path.join(species_folder, folder)
    for file in os.listdir(noBac):
        if file.endswith("_no_BAC.py"):
            freq_path = os.path.join(noBac, file)
            break

    energy_path = None
    if freq_path and os.path.isfile(freq_path):
        with open(freq_path, 'r') as f:
            block = f.read()
            energy_m = re.search(r"geometry\s*=\s*Log\(\s*['\"]([^'\"]+)['\"]", block)
            if energy_m:
                raw_energy_path = energy_m.group(1)
                # Only manipulate the path if '/calcs/' is present
                if '/calcs/' in raw_energy_path:
                    base_folder = species_folder.split('/output/Species')[0]
                    suffix = raw_energy_path.split('/calcs/', 1)[1]
                    energy_path = os.path.join(base_folder, 'calcs', suffix)
                else:
                    energy_path = raw_energy_path # fallback to the raw path if structure is unexpected
    # else: energy_path remains None

    return opt_coords, energy_path

def extract_thermo_data_for_species(base_path: str, rxn: str, species_info: dict):
    """
    Extracts E0 and other thermochemical properties for each species in a reaction.
    
    Args:
        base_path (str): Path to the root of the reaction folder.
        rxn (str): Reaction name/id.
        species_info (dict): A dict like {'r0': smiles, 'r1': smiles, ..., 'ts': smiles}

    Returns:
        List[dict]: One dict per species with thermo info.
    """
    thermo_data = []

    for key, smiles in species_info.items():
        if smiles is None:
            continue  # skip missing

        if key == 'ts':
            arkane_path = os.path.join(base_path, 'output', 'rxns', 'TS0', 'arkane', 'output.py')
        else:
            species_dir = check_label(smiles)[0]
            arkane_path = os.path.join(base_path, 'output', 'Species', species_dir, 'arkane', 'output.py')

        if os.path.exists(arkane_path):
            with open(arkane_path, 'r') as f:
                block = f.read()
            parsed = parse_conformer(block)
        else:
            parsed = {k: None for k in ['E0_value', 'E0_units', 'spin_multiplicity',
                                        'optical_isomers', 'frequency_value', 'frequency_units', 'frequency_list']}

        parsed.update({'rxn': rxn, 'species_type': key, 'smiles': smiles})
        thermo_data.append(parsed)

    return thermo_data



def parse_conformer(block: str) -> dict:
    out = {}
    # label
    label_m = re.search(r"label\s*=\s*['\"]([^'\"]+)['\"]", block)
    out["label"] = label_m.group(1) if label_m else None

    # E0 (allowing scientific notation)
    e0_m = re.search(
        r"E0\s*=\s*\(\s*([-\d\.eE\+]+)\s*,\s*['\"]([^'\"]+)['\"]",
        block
    )
    if e0_m:
        out["E0_value"], out["E0_units"] = float(e0_m.group(1)), e0_m.group(2)
    else:
        out["E0_value"], out["E0_units"] = None, None

    # spin multiplicity & optical isomers
    spin_m = re.search(r"spin_multiplicity\s*=\s*(\d+)", block)
    iso_m  = re.search(r"optical_isomers\s*=\s*(\d+)", block)
    out["spin_multiplicity"] = int(spin_m.group(1)) if spin_m else None
    out["optical_isomers"]   = int(iso_m.group(1))  if iso_m  else None

    # modes
    modes_m = re.search(r"modes\s*=\s*\[(.*?)]\s*,", block, re.S)
    out["modes"] = modes_m.group(1).strip().replace("\n"," ") if modes_m else None

    # frequency: first try a list (e.g. frequency = ([…],'cm^-1'))
    freq_list_m = re.search(
        r"frequency\s*=\s*\(\s*\[\s*([^\]]+?)\s*\]\s*,\s*['\"]([^'\"]+)['\"]",
        block, re.S
    )
    if freq_list_m:
        out["frequency_list"] = [float(x) for x in freq_list_m.group(1).split(",")]
        out["frequency_units"] = freq_list_m.group(2)
    else:
        # then try a single value: frequency = ( -933.988 , 'cm^-1')
        freq_val_m = re.search(
            r"frequency\s*=\s*\(\s*([-\d\.eE\+]+)\s*,\s*['\"]([^'\"]+)['\"]",
            block
        )
        if freq_val_m:
            out["frequency_value"] = float(freq_val_m.group(1))
            out["frequency_units"] = freq_val_m.group(2)
        else:
            out["frequency_value"] = None
            out["frequency_units"] = None
            out["frequency_list"]  = None

    return out


def extract_coords(rows_data):
    coords = '\n'.join(rows_data)

    return coords


def parse_reactions(dataframe: pd.DataFrame, converged_rxn_dirs: List[str]) -> pd.DataFrame:
    """
    :param dataframe: pd.DataFrame
    :param converged_rxn_dirs: List[str]
    :return: pd.DataFrame
    """
    logger.info("Parsing reactions...")
    dict_reactions = {}
    freq_out_map = {}
    for path in converged_rxn_dirs:
        for folder in os.listdir(path):
            # reset freq_out_map for this reaction

            dict_molecules_smiles = {
                "reactant0_smiles": [],
                "reactant1_smiles": [],
                "product0_smiles": [],
                "product1_smiles": [],
            }
            if folder == "NonRMG":
                continue

            # 1. Get the folder path of the converged reaction
            folder_path = os.path.join(path, folder) if os.path.isdir(os.path.join(path, folder)) else None

            # 2. Get the reactants and products using folder name and the data file
            reaction_name = folder
            logger.info(f"Processing reaction: {reaction_name}")
            df_reactants = dataframe[dataframe['original_reaction_id'] == folder]

            # 3. Load the input file
            if os.path.exists(os.path.join(folder_path, "input.yml")):
                with open(os.path.join(folder_path, "input.yml"), "r") as f:
                    reaction_input = yaml.safe_load(f)
            else:
                input_files = [f for f in os.listdir(folder_path) if f.endswith("_info.yml")]
                if not input_files:
                    print("No input file found")
                    continue
                input_file_path = os.path.join(folder_path, input_files[0])
                with open(input_file_path, 'r') as input_file:
                    reaction_input = yaml.load(input_file, Loader=yaml.FullLoader)
                

            # 4. Get Reactants and Products from the input file
            data_reaction = reaction_input['reactions'][0]
            reactants, products = data_reaction['label'].split(" <=> ")
            r0_label, r1_label = reactants.split(" + ")
            p0_label, p1_label = products.split(" + ")

            # 5. Get the smiles of the reactants and products
            for species in reaction_input['species']:
                if species["label"] == r0_label:
                    dict_molecules_smiles["reactant0_smiles"].append(species["smiles"])
                    dict_molecules_smiles["reactant0_smiles"].append(
                        resonance_match(species["smiles"], df_reactants["reactant_0"].values[0],
                                        return_smiles="first") if resonance_match(species["smiles"],
                                                                                  df_reactants["reactant_0"].values[
                                                                                      0]) else
                        df_reactants["reactant_1"].values[0])
                elif species["label"] == r1_label:
                    dict_molecules_smiles["reactant1_smiles"].append(species["smiles"])
                    dict_molecules_smiles["reactant1_smiles"].append(
                        resonance_match(species["smiles"], df_reactants["reactant_1"].values[0],
                                        return_smiles="first") if resonance_match(species["smiles"],
                                                                                  df_reactants["reactant_1"].values[
                                                                                      0]) else
                        df_reactants["reactant_0"].values[0])
                elif species["label"] == p0_label:
                    dict_molecules_smiles["product0_smiles"].append(species["smiles"])
                    dict_molecules_smiles["product0_smiles"].append(
                        resonance_match(species["smiles"], df_reactants["product_0"].values[0],
                                        return_smiles="first") if resonance_match(species["smiles"],
                                                                                  df_reactants["product_0"].values[
                                                                                      0]) else
                        df_reactants["product_1"].values[0])
                elif species["label"] == p1_label:
                    dict_molecules_smiles["product1_smiles"].append(species["smiles"])
                    dict_molecules_smiles["product1_smiles"].append(
                        resonance_match(species["smiles"], df_reactants["product_1"].values[0],
                                        return_smiles="first") if resonance_match(species["smiles"],
                                                                                  df_reactants["product_1"].values[
                                                                                      0]) else
                        df_reactants["product_0"].values[0])

            ## 5.1 Check that smiles lists in each key are the same, if not, raise an error
            for key, value in dict_molecules_smiles.items():
                if len(set(value)) > 1:
                    raise ValueError(f"Error in {key} key: {value}")
            ## 5.2 Set the smiles of the reactants and products
            dict_molecules_smiles = {key: value[0] for key, value in dict_molecules_smiles.items()}

            output_folder = os.path.join(folder_path, "output")
            species_path = os.path.join(output_folder, "Species")

            r0, r1, p0, p1, ts = [], [], [], [], []
            p0_smiles, p1_smiles, r0_smiles, r1_smiles = None, None, None, None

            try:
                species_folder_map: dict[str, str] = {}
                for spc_folder in os.listdir(species_path):
                    mol_type = determine_mol_type(spc_folder)

                    spc_folder_path = os.path.join(species_path, check_label(spc_folder, underscore_rule=False)[0]) if df_reactants[
                        "source"].values[0] != "KFIR_ZEUS" else os.path.join(species_path, spc_folder)
                    if os.path.isdir(os.path.join(species_path, spc_folder_path)):
                        if mol_type is not None:
                            if mol_type == "r0":
                                if r0 == []:
                                    coords, freq = parse_opt(species_path, spc_folder, dict_molecules_smiles["reactant0_smiles"])
                                    r0 = coords
                                    freq_out_map.setdefault(reaction_name, {})['r0'] = freq
                                    r0_smiles = dict_molecules_smiles["reactant0_smiles"]
                                    species_folder_map['r0'] = spc_folder
                                else:
                                    raise ValueError("r0 already exists - code is attempting to overwrite")
                            elif mol_type == "r1":
                                if r1 == []:
                                    coords, freq = parse_opt(species_path, spc_folder, dict_molecules_smiles["reactant1_smiles"])
                                    r1 = coords
                                    freq_out_map.setdefault(reaction_name, {})['r1'] = freq
                                    r1_smiles = dict_molecules_smiles["reactant1_smiles"]
                                    species_folder_map['r1'] = spc_folder
                                else:
                                    raise ValueError("r1 already exists - code is attempting to overwrite")
                            elif mol_type == "p0":
                                if p0 == []:
                                    coords, freq = parse_opt(species_path, spc_folder, dict_molecules_smiles["product0_smiles"])
                                    p0 = coords
                                    freq_out_map.setdefault(reaction_name, {})['p0'] = freq
                                    p0_smiles = dict_molecules_smiles["product0_smiles"]
                                    species_folder_map['p0'] = spc_folder
                                else:
                                    raise ValueError("p0 already exists - code is attempting to overwrite")
                            elif mol_type == "p1":
                                if p1 == []:
                                    coords, freq = parse_opt(species_path, spc_folder, dict_molecules_smiles["product1_smiles"])
                                    p1 = coords
                                    freq_out_map.setdefault(reaction_name, {})['p1'] = freq
                                    p1_smiles = dict_molecules_smiles["product1_smiles"]
                                    species_folder_map['p1'] = spc_folder
                                else:
                                    raise ValueError("p1 already exists - code is attempting to overwrite")
                        elif df_reactants["source"].values[0] == "KFIR_ZEUS":
                            if check_label(dict_molecules_smiles["reactant0_smiles"])[0] == spc_folder_path.split("/")[-1]:
                                if r0 == []:
                                    coords, freq = parse_opt(species_path, spc_folder, dict_molecules_smiles["reactant0_smiles"])
                                    r0 = coords
                                    freq_out_map.setdefault(reaction_name, {})['r0'] = freq
                                    r0_smiles = dict_molecules_smiles["reactant0_smiles"]
                                    species_folder_map['r0'] = spc_folder
                                else:
                                    raise ValueError("r0 already exists - code is attempting to overwrite")
                            elif check_label(dict_molecules_smiles["reactant1_smiles"])[0] == spc_folder_path.split("/")[-1]:
                                if r1 == []:
                                    coords, freq = parse_opt(species_path, spc_folder, dict_molecules_smiles["reactant1_smiles"])
                                    r1 = coords
                                    freq_out_map.setdefault(reaction_name, {})['r1'] = freq
                                    r1_smiles = dict_molecules_smiles["reactant1_smiles"]
                                    species_folder_map['r1'] = spc_folder
                                else:
                                    raise ValueError("r1 already exists - code is attempting to overwrite")
                            elif check_label(dict_molecules_smiles["product0_smiles"])[0] == spc_folder_path.split("/")[-1]:
                                if p0 == []:
                                    coords, freq = parse_opt(species_path, spc_folder, dict_molecules_smiles["product0_smiles"])
                                    p0 = coords
                                    freq_out_map.setdefault(reaction_name, {})['p0'] = freq
                                    p0_smiles = dict_molecules_smiles["product0_smiles"]
                                    species_folder_map['p0'] = spc_folder
                                else:
                                    raise ValueError("p0 already exists - code is attempting to overwrite")
                            elif check_label(dict_molecules_smiles["product1_smiles"])[0] == spc_folder_path.split("/")[-1]:
                                if p1 == []:
                                    coords, freq = parse_opt(species_path, spc_folder, dict_molecules_smiles["product1_smiles"])
                                    p1 = coords
                                    freq_out_map.setdefault(reaction_name, {})['p1'] = freq
                                    p1_smiles = dict_molecules_smiles["product1_smiles"]
                                    species_folder_map['p1'] = spc_folder
                                else:
                                    raise ValueError("p1 already exists - code is attempting to overwrite")
                        ### Do not use underscore transformation

                        elif check_label(dict_molecules_smiles["reactant0_smiles"])[0] == os.path.join(species_path, spc_folder).split("/")[-1].split(reaction_name+'_',1)[1]:
                            if r0 == []:
                                coords, freq = parse_opt(species_path, spc_folder, dict_molecules_smiles["reactant0_smiles"])
                                r0 = coords
                                freq_out_map.setdefault(reaction_name, {})['r0'] = freq
                                r0_smiles = dict_molecules_smiles["reactant0_smiles"]
                                species_folder_map['r0'] = spc_folder
                            else:
                                raise ValueError("r0 already exists - code is attempting to overwrite")
                        elif check_label(dict_molecules_smiles["reactant1_smiles"])[0] == os.path.join(species_path, spc_folder).split("/")[-1].split(reaction_name+'_',1)[1]:
                            if r1 == []:
                                coords, freq = parse_opt(species_path, spc_folder, dict_molecules_smiles["reactant1_smiles"])
                                r1 = coords
                                freq_out_map.setdefault(reaction_name, {})['r1'] = freq
                                r1_smiles = dict_molecules_smiles["reactant1_smiles"]
                                species_folder_map['r1'] = spc_folder
                            else:
                                raise ValueError("r1 already exists - code is attempting to overwrite")
                        elif check_label(dict_molecules_smiles["product0_smiles"])[0] == os.path.join(species_path, spc_folder).split("/")[-1].split(reaction_name+'_',1)[1]:
                            if p0 == []:
                                coords, freq = parse_opt(species_path, spc_folder, dict_molecules_smiles["product0_smiles"])
                                p0 = coords
                                freq_out_map.setdefault(reaction_name, {})['p0'] = freq
                                p0_smiles = dict_molecules_smiles["product0_smiles"]
                                species_folder_map['p0'] = spc_folder
                            else:
                                raise ValueError("p0 already exists - code is attempting to overwrite")
                        elif check_label(dict_molecules_smiles["product1_smiles"])[0] == os.path.join(species_path, spc_folder).split("/")[-1].split(reaction_name+'_',1)[1]:
                            if p1 == []:
                                coords, freq = parse_opt(species_path, spc_folder, dict_molecules_smiles["product1_smiles"])
                                p1 = coords
                                freq_out_map.setdefault(reaction_name, {})['p1'] = freq
                                p1_smiles = dict_molecules_smiles["product1_smiles"]
                                species_folder_map['p1'] = spc_folder
                            else:
                                raise ValueError("p1 already exists - code is attempting to overwrite")           
                        else:
                            ## Logic for Non RMG/ATLAS runs
                            if check_label(dict_molecules_smiles["reactant0_smiles"], underscore_rule=False)[0] == spc_folder_path.split("_")[-1]:
                                if r0 == []:
                                    coords, freq = parse_opt(species_path, spc_folder, dict_molecules_smiles["reactant0_smiles"])
                                    r0 = coords
                                    freq_out_map.setdefault(reaction_name, {})['r0'] = freq
                                    r0_smiles = dict_molecules_smiles["reactant0_smiles"]
                                    species_folder_map['r0'] = spc_folder
                                else:
                                    raise ValueError("r0 already exists - code is attempting to overwrite")
                            elif check_label(dict_molecules_smiles["reactant1_smiles"], underscore_rule=False)[0] == spc_folder_path.split("_")[-1]:
                                if r1 == []:
                                    coords, freq = parse_opt(species_path, spc_folder, dict_molecules_smiles["reactant1_smiles"])
                                    r1 = coords
                                    freq_out_map.setdefault(reaction_name, {})['r1'] = freq
                                    r1_smiles = dict_molecules_smiles["reactant1_smiles"]
                                    species_folder_map['r1'] = spc_folder
                                else:
                                    raise ValueError("r1 already exists - code is attempting to overwrite")
                            elif check_label(dict_molecules_smiles["product0_smiles"], underscore_rule=False)[0] == spc_folder_path.split("_")[-1]:
                                if p0 == []:
                                    coords, freq = parse_opt(species_path, spc_folder, dict_molecules_smiles["product0_smiles"])
                                    p0 = coords
                                    freq_out_map.setdefault(reaction_name, {})['p0'] = freq
                                    p0_smiles = dict_molecules_smiles["product0_smiles"]
                                    species_folder_map['p0'] = spc_folder
                                else:
                                    raise ValueError("p0 already exists - code is attempting to overwrite")
                            elif check_label(dict_molecules_smiles["product1_smiles"], underscore_rule=False)[0] == spc_folder_path.split("_")[-1]:
                                if p1 == []:
                                    coords, freq = parse_opt(species_path, spc_folder, dict_molecules_smiles["product1_smiles"])
                                    p1 = coords
                                    freq_out_map.setdefault(reaction_name, {})['p1'] = freq
                                    p1_smiles = dict_molecules_smiles["product1_smiles"]
                                    species_folder_map['p1'] = spc_folder
                                else:
                                    raise ValueError("p1 already exists - code is attempting to overwrite")
            except Exception as e:
                print(f"Error in {reaction_name}: {e}")
                continue
                
                # Species could be monotomic
            for items in [r0, r1, p0, p1]:
                if r0 == []:
                    r0 = monoatomic_atoms(dict_molecules_smiles["reactant0_smiles"])
                    r0_smiles = dict_molecules_smiles["reactant0_smiles"]
                elif r1 == []:
                    r1 = monoatomic_atoms(dict_molecules_smiles["reactant1_smiles"])
                    r1_smiles = dict_molecules_smiles["reactant1_smiles"]
                elif p0 == []:
                    p0 = monoatomic_atoms(dict_molecules_smiles["product0_smiles"])
                    p0_smiles = dict_molecules_smiles["product0_smiles"]
                elif p1 == []:
                    p1 = monoatomic_atoms(dict_molecules_smiles["product1_smiles"])
                    p1_smiles = dict_molecules_smiles["product1_smiles"]

            ts_folder = os.path.join(output_folder, "rxns", "TS0", "geometry")
            if os.path.exists(ts_folder):
                for file in os.listdir(ts_folder):
                    if file.endswith(".xyz"):
                        xyz = xyz_parse(os.path.join(ts_folder, file))
                        ts = xyz
            else:
                print(f"TS folder not found for {reaction_name}")
                # Skip the reaction
                continue

            ### DEBUG
            if r0 == []:
                print("r0 empty")
            if r1 == []:
                print("r1 empty")
            if p0 == []:
                print("p0 empty")
            if p1 == []:
                print("p1 empty")
            if r0_smiles is None:
                print("r0_smiles empty")
            if r1_smiles is None:
                print("r1_smiles empty")
            if p0_smiles is None:
                print("p0_smiles empty")
            if p1_smiles is None:
                print("p1_smiles empty")

            dict_reactions[reaction_name] = {"r0": r0,
                                             "r0_smiles": r0_smiles,
                                             "r1": r1,
                                             "r1_smiles": r1_smiles,
                                             "p0": p0,
                                             "p0_smiles": p0_smiles,
                                             "p1": p1,
                                             "p1_smiles": p1_smiles,
                                             "ts": ts}

            try:
                # Build Arkane paths
                species_paths = {}

                for role in ('r0', 'r1', 'p0', 'p1'):
                    smiles = locals()[f"{role}_smiles"]
                    folder = species_folder_map.get(role, "")
                    if folder:
                        ark_path = os.path.join(
                            species_path,      # the base Species/… folder
                            folder,            # exactly the subfolder you saw
                            'arkane',
                            'output.py'
                        )
                    else:
                        ark_path = None
                    species_paths[role] = (smiles, ark_path)
                species_paths['ts'] = (
                    None,
                    os.path.join(output_folder, 'rxns', 'TS0', 'arkane', 'output.py')
                )

                # Attempt to get TS SMILES
                if ts:
                    try:
                        if isinstance(ts, list):
                            ts_str = "\n".join(ts)
                            ts_spc = ARCSpecies(label="TS", xyz=ts_str, is_ts=True)
                        else:
                            ts_str = ts
                            ts_spc = ARCSpecies(label="TS", xyz=ts, is_ts=True)

                        species_paths['ts'] = (ts_spc.mol.to_smiles(), species_paths['ts'][1])
                    except Exception as e:
                        raise ValueError(f"Could not generate SMILES for TS in {reaction_name}: {e}")
                        logger.warning(f"Could not generate SMILES for TS in {reaction_name}: {e}")

                # Collect thermo data
                for key, (smiles, ark_path) in species_paths.items():
                    if smiles:
                        thermo_data_all.append(get_species_thermo(smiles, ark_path, key, reaction_name))
            except Exception as e:
                raise ValueError(f"Error in {reaction_name}: {e}")
                logger.warning(f"Thermo data extraction failed for {reaction_name}: {e}")


    # Save thermo data to CSV
    if thermo_data_all:
        df_thermo = pd.DataFrame(thermo_data_all)
        df_thermo.to_csv("/home/calvin/code/chemprop_phd_customised/habnet/data/processed/target_data/thermo_species_data.csv", index=False)
        print(f"Saved thermo data with {len(df_thermo)} entries to 'data/thermo_species_data.csv'")
    else:
        print("No thermo data collected.")

    # save collected input.log and freq.out paths
    if freq_out_map:
        # include rxn and species_type columns explicitly
        freq_out_map_df = pd.DataFrame.from_dict(freq_out_map, orient='index').reset_index()
        freq_out_map_df.rename(columns={'index': 'rxn'}, inplace=True)
        freq_out_map_df = freq_out_map_df[['rxn', 'r0', 'r1', 'p0', 'p1']]
        freq_out_map_df.to_csv("/home/calvin/code/chemprop_phd_customised/habnet/data/processed/target_data/freq_out_map.csv", index=False)
        print(f"Saved input/freq log paths to 'data/processed/target_data/freq_out_map.csv'")

    return pd.DataFrame(dict_reactions).T


def order_reactions(data, read_csv=False):
    ordered_HAbstract_coords = pd.DataFrame(columns=['rxn', 'r2h', 'r1', 'r2', 'r1h', 'ts'])
    ordered_HAbstract_smiles = pd.DataFrame(columns=['rxn', 'r2h', 'r1', 'r2', 'r1h', 'ts'])
    rxn_issues = pd.DataFrame(columns=['rxn', 'error'])
    error_df = pd.DataFrame(columns=['index', 'reaction_smiles', 'error'])

    for index, rows in tqdm(data.iterrows(), total=data.shape[0], desc="Processing Reactions"):
        ordered_HAbstract_coords, ordered_HAbstract_smiles, rxn_issues = check_reaction_order(index, rows,
                                                                                              ordered_HAbstract_coords,
                                                                                              ordered_HAbstract_smiles,
                                                                                              rxn_issues,
                                                                                              read_csv=read_csv)

    return ordered_HAbstract_coords, ordered_HAbstract_smiles, rxn_issues


def get_species_mapping(label_map, species_1, species_2, reactants=True):
    """
    Map labels to their corresponding ARCSpecies based on the label_map.

    H_Abstraction reaction template:  1R-2H + 3R* <=> 1R* + 2H-3R

    X_H relates to R1H and R2H, and Y_rad relates to R1 and R2.

    Args:
        label_map (dict): A dictionary mapping labels ('*1', '*2', '*3') to indices.
        species_1 (ARCSpecies or Molecule): The first ARCSpecies object, or a Molecule object.
        species_2 (ARCSpecies or Molecule): The second ARCSpecies object, or a Molecule object.
        reactants (bool): Whether the species are reactants (True) or products (False).

    Returns:
        dict: A dictionary with keys 'X_H' and 'Y_rad', mapping to their respective ARCSpecies.

    Raises:
        ValueError: If the atom at the specified label is not hydrogen ('H').
    """
    species_mapping = {}


    if type(species_1) == ARCSpecies and type(species_2) == ARCSpecies:
        spc_1 = species_1.mol.copy(deep=True)
        spc_2 = species_2.mol.copy(deep=True)
    else:
        spc_1 = species_1.copy(deep=True)
        spc_2 = species_2.copy(deep=True)

    spc_1_atoms = spc_1.atoms.copy()
    spc_2_atoms = spc_2.atoms.copy()

    if reactants:
        if list(label_map.keys())[0] == '*3':
            species_mapping = {'Y_rad': species_1, 'X_H': species_2}
            order = ['r1', 'r0']
        else:
            species_mapping = {'X_H': species_1, 'Y_rad': species_2}
            order = ['r0', 'r1']

    else:
        if '*1' in label_map and '*1' == list(label_map.keys())[0]:
            species_mapping = {'Y_rad': species_1, 'X_H': species_2}
            order = ['p1', 'p0']
        else:
            species_mapping = {'X_H': species_1, 'Y_rad': species_2}
            order = ['p0', 'p1']

    return species_mapping, order


def check_reaction_order(index, rows, df_coords, df_smiles, df_issues, read_csv=False):
    try:
        r1_un, r2_un, p1, p2, r1, r2, r1h, r2h, r1_coords, r2_coords, r1h_coords, r2h_coords, rxn_0a = None, None, None, None, None, None, None, None, None, None, None, None, None
        if read_csv:
            import ast
            reactant_0 = extract_coords(ast.literal_eval(rows['r0'])[0])
            reactant_1 = extract_coords(ast.literal_eval(rows['r1'])[0])
            product_0 = extract_coords(ast.literal_eval(rows['p0'])[0])
            product_1 = extract_coords(ast.literal_eval(rows['p1'])[0])
            ts = extract_coords(ast.literal_eval(rows['ts']))
        else:
            reactant_0 = extract_coords(rows['r0'][0])
            reactant_1 = extract_coords(rows['r1'][0])
            product_0 = extract_coords(rows['p0'][0])
            product_1 = extract_coords(rows['p1'][0])
            ts = extract_coords(rows['ts'])

        RxnFamily = ReactionFamily(label="H_Abstraction")
        orig_r0_smiles = rows['r0_smiles']
        orig_r1_smiles = rows['r1_smiles']
        orig_p0_smiles = rows['p0_smiles']
        orig_p1_smiles = rows['p1_smiles']


        r1_un = ARCSpecies(label="R1", xyz=reactant_0, smiles=rows["r0_smiles"])
        r2_un = ARCSpecies(label="R2", xyz=reactant_1, smiles=rows["r1_smiles"])
        p1 = ARCSpecies(label="P1", xyz=product_0, smiles=rows["p0_smiles"])
        p2 = ARCSpecies(label="P2", xyz=product_1, smiles=rows["p1_smiles"])

        reaction = ARCReaction(r_species=[r1_un, r2_un],
                               p_species=[p1, p2])

        # 1. Generate products based on reaction based on reactants and products
        gen_rxn_prod = get_reaction_family_products(reaction)
        # gen_rxn_prod_top = gen_rxn_prod[0] # TODO: Big assumption
        for gen_rxn_prod_top in gen_rxn_prod:
            # 2. Generate all possible products based on reactants
            products = RxnFamily.generate_products(reactants=[r1_un, r2_un])

            # 2.1 Get the label mapping
            r_label_map = gen_rxn_prod_top['r_label_map']
            p_label_map = gen_rxn_prod_top['p_label_map']

            # 2.3 Get the generated RMG Molecule products
            gen_p_1, gen_p_2 = gen_rxn_prod_top['products']

            # 3. Get the mapping, X_H and Y_rad
            r_mapping, order_r = get_species_mapping(r_label_map, r1_un, r2_un)
            p_mapping, order_p = get_species_mapping(p_label_map, gen_p_1, gen_p_2, reactants=False)

            # Record where each output came from
            orig_reactant_order = order_r
            orig_product_order = order_p


            # 4. Set Reactants based on mapping
            r1h = r_mapping["X_H"].mol.to_smiles()
            r1h_coords = r_mapping["X_H"].get_xyz(return_format='str').split("\n")

            r2 = r_mapping["Y_rad"].mol.to_smiles()
            r2_coords = r_mapping["Y_rad"].get_xyz(return_format='str').split("\n")

            # 5. Set Products based on mapping
            user_products = [p1, p2]
            replace_molecule_arcspecies(p_mapping, user_products)

            r2h = p_mapping["X_H"].mol.smiles
            r2h_coords = p_mapping["X_H"].get_xyz(return_format='str').split("\n")

            r1 = p_mapping["Y_rad"].mol.to_smiles()
            r1_coords = p_mapping["Y_rad"].get_xyz(return_format='str').split("\n")

            ts_smiles = ARCSpecies(label='TS', xyz=ts, is_ts=True).mol.to_smiles()

            label_map = gen_rxn_prod_top["r_label_map"]

            # 4–5. Your existing coordinate and smiles extraction here…
            # …then when you build tempdf_smiles:
            tempdf_smiles = pd.DataFrame([[
                rows['rxn'],
                r2h, r1, r2, r1h, ts_smiles, label_map,
                orig_product_order[1], orig_product_order[0], orig_reactant_order[1], orig_reactant_order[0]
            ]], columns=[
                'rxn',
                'r2h', 'r1', 'r2', 'r1h', 'ts', 'label_map',
                'orig_r2h', 'orig_r1', 'orig_r2', 'orig_r1h'
            ])
            df_smiles = pd.concat([df_smiles, tempdf_smiles], ignore_index=True)
    
            tempdf_coords = pd.DataFrame([[
                rows['rxn'],
                r2h_coords, r1_coords, r2_coords, r1h_coords,
                rows['ts'], label_map,
                orig_reactant_order[1], orig_product_order[0], orig_product_order[1], orig_reactant_order[0]
            ]], columns=[
                'rxn',
                'r2h','r1','r2','r1h','ts','label_map',
                'orig_r2h','orig_r1','orig_r2','orig_r1h'
            ])
            df_coords = pd.concat([df_coords, tempdf_coords], ignore_index=True)


    except Exception as e:
        print(f"Error in reaction {rows['rxn']}: {e}")
        tempdf = pd.DataFrame([[rows['rxn'], e, rows['r0'], rows['r1'], rows['p0'], rows['p1']]],
                              columns=['rxn', 'error', 'r0', 'r1', 'p0', 'p1'])
        df_issues = pd.concat([df_issues, tempdf], ignore_index=True)

    return df_coords, df_smiles, df_issues


def link_label_smiles(dict_smiles, mapped_labels):
    updated_dict = {}
    for new_key, value in dict_smiles.items():
        original_key = mapped_labels.get(new_key)
        if original_key:
            updated_dict[original_key] = value
        else:
            updated_dict[new_key] = value

    return updated_dict


def replace_molecule_arcspecies(p_mapping, products):
    """
    Replace RMG Molecule objects in the p_mapping dictionary with matching ARCSpecies objects from the products list.

    Args:
        p_mapping (dict): A dictionary where keys are identifiers and values are RMG Molecule objects.
        products (list): A list of ARCSpecies objects to match against.

    Returns:
        None. The `p_mapping` dictionary is updated in place.
    """

    arc_smiles_to_species = {arc_species.mol.to_smiles(): arc_species for arc_species in products}

    for key, rmg_mol in p_mapping.items():
        rmg_smiles = rmg_mol.to_smiles()
        species = arc_smiles_to_species.get(rmg_smiles)
        if species:
            p_mapping[key] = species
            print(f"Replaced '{key}': '{rmg_smiles}' with ARCSpecies object '{species.label}'")
        else:
            resonance_mols = generate_resonance_structures(rmg_mol)
            resonance_smiles = [resonance_mol.to_smiles() for resonance_mol in resonance_mols]

            matching_species = [arc_smiles_to_species.get(smiles) for smiles in resonance_smiles if
                                smiles in arc_smiles_to_species]
            if len(matching_species) == 1:
                # Unique match
                species = matching_species[0]
                p_mapping[key] = species
                print(f"Replaced '{key}': '{rmg_smiles}' with ARCSpecies object '{species.label}' (via resonance)")
            elif len(matching_species) > 1:
                raise ValueError(
                    f"Multiple ARCSpecies found for RMG Molecule '{rmg_mol.label} ({rmg_smiles})' in resonance structures. Matches are : {matching_species}")
            elif len(matching_species) == 0:
                match_found = False
                for existing_species in arc_smiles_to_species.values():
                    try:
                        existing_resonance = generate_resonance_structures(existing_species.mol)
                        existing_resonance_smiles = [res_spc.to_smiles() for res_spc in existing_resonance]
                        if rmg_smiles in existing_resonance_smiles:
                            # Match found; use the existing_species
                            p_mapping[key] = existing_species
                            print(
                                f"Replaced '{key}': '{rmg_smiles}' with ARCSpecies object '{existing_species.label}' (via opposite resonance)")
                            match_found = True
                            break  # Exit loop after finding the first match
                    except Exception as e:
                        print(f"Error generating resonance for existing ARCSpecies '{existing_species.label}': {e}")
                        continue
                if not match_found:
                    # Try RDKIT
                    for existing_species in arc_smiles_to_species.values():
                        try:
                            resonance_mols_to_rdkit = generate_resonance_structures(existing_species.mol)
                            resonance_smiles_to_rdkit = [resonance_mol.to_smiles() for resonance_mol in
                                                         resonance_mols_to_rdkit]
                            # Convert to RDKit Mol and then back to Smiles
                            replace_smiles_rdkt = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in
                                                   resonance_smiles_to_rdkit]
                            if rmg_smiles in replace_smiles_rdkt:
                                # Match found; use the existing_species
                                p_mapping[key] = existing_species
                                print(
                                    f"Replaced '{key}': '{rmg_smiles}' with ARCSpecies object '{existing_species.label}' (via RDKit)")
                                match_found = True
                                break
                        except Exception as e:
                            print(f"Error generating RDKit for existing ARCSpecies '{existing_species.label}': {e}")
                            continue
                    if not match_found:
                        print(
                            f"Warning: No ARCSpecies found for RMG Molecule '{rmg_mol} ({rmg_smiles})' in resonance structures, even after RDKit attempts.")
            else:
                raise ValueError(
                    f"No ARCSpecies found for RMG Molecule '{rmg_mol} ({rmg_smiles})' in resonance structures, even with backup list")


def replace_smiles_arc(dict_ordered_molecules, arc_spc_list, dataframe_species):
    smiles_to_species = {species.mol.smiles: species for species in arc_spc_list}
    for key, smiles in dict_ordered_molecules.items():
        species = smiles_to_species.get(smiles)
        if species:
            dict_ordered_molecules[key] = species
            print(f"Replaced '{key}': '{smiles}' with ARCSpecies object '{species.label}'")
        else:
            # Try match with the backup list
            species = dataframe_species.get(smiles)
            if species:
                dict_ordered_molecules[key] = species
                print(f"Replaced '{key}': '{smiles}' with ARCSpecies object '{species.label}'")
            else:
                ## RESONANCE
                temp_species = ARCSpecies(label="tempspecies", smiles=smiles)
                temp_resonance = generate_resonance_structures(temp_species.mol)
                temp_species_list = [spc.to_smiles() for spc in temp_resonance]
                species_results = [smiles_to_species.get(smiles) for smiles in temp_species_list if
                                   smiles_to_species.get(smiles) is not None]
                if len(species_results) > 1:
                    print(f"Warning: Got more than one result for resonance SMILES in '{key}")
                elif len(species_results) == 0:
                    match_found = False
                    for existing_species in smiles_to_species.values():
                        try:
                            existing_resonance = generate_resonance_structures(existing_species.mol)
                            existing_resonance_smiles = [res_spc.to_smiles() for res_spc in existing_resonance]

                            if temp_species.mol.smiles in existing_resonance_smiles:
                                # Match found; use the existing_species
                                dict_ordered_molecules[key] = existing_species
                                print(
                                    f"Replaced '{key}': '{smiles}' with ARCSpecies object '{existing_species.label}' (via opposite resonance)")
                                match_found = True
                                break  # Exit loop after finding the first match
                        except Exception as e:
                            print(f"Error generating resonance for existing ARCSpecies '{existing_species.label}': {e}")
                            continue
                    if not match_found:
                        print(
                            f"Warning: No ARCSpecies found for SMILES '{smiles}' in key '{key}', even after resonance attempts.")
                elif len(species_results) == 1:
                    species = species_results[0]
                    dict_ordered_molecules[key] = species
                else:
                    print(f"Warning: No ARCSpecies found for SMILES '{smiles}' in key '{key}', even with backup list")


def dodgy_solution(upd_order_reactants, p1, p2):
    upd_order_products = {}

    # We have p1 and p2

    # We first check if len(p1) is not equal to len(p2) or that they are one less or one more than each other

    if len(p1.mol.atoms) == len(p2.mol.atoms):
        print("Error - p1 and p2 have the same number of atoms")
        return None
    elif len(p1.mol.atoms) == len(p2.mol.atoms) + 1:
        print("Error - p1 has one more atom than p2")
        return None
    elif len(p1.mol.atoms) == len(p2.mol.atoms) - 1:
        print("Error - p1 has one less atom than p2")
        return None

    elif len(p1.mol.atoms) == len(upd_order_reactants['Y_rad'].mol.atoms) + 1:
        if len(p2.mol.atoms) == len(upd_order_reactants['X_H'].mol.atoms) - 1:
            upd_order_products['X_H'] = p1
            upd_order_products['Y_rad'] = p2
        else:
            print("Error - p1 has one more atom than Y_rad but p2 does not have one less atom than X_H")
            return None
    elif len(p1.mol.atoms) == len(upd_order_reactants['X_H'].mol.atoms) - 1:
        if len(p2.mol.atoms) == len(upd_order_reactants['Y_rad'].mol.atoms) + 1:
            upd_order_products['X_H'] = p2
            upd_order_products['Y_rad'] = p1
        else:
            print("Error - p1 has less than 1 atom than X_H but p2 does not have one more atom than Y_rad")
            return None

    connections = {
        'Reactants': upd_order_reactants,
        'Products': upd_order_products
    }

    return connections


def validation_parse_data(data: pd.DataFrame, list_length_exceeds: int = 30):

    if data.isnull().any().any():
        print("There are empty cells in the DataFrame.")
        empty_cells = data.isnull()
        print("Empty cells found at:")
        print(empty_cells[empty_cells == True].stack())
    else:
        print("No empty cells found in the DataFrame.")

    columns_to_check = ['r0', 'r1', 'p0', 'p1', 'ts']
    for column in columns_to_check:
        not_list = data[column].apply(lambda cell: not isinstance(cell, list))
        if not_list.any():
            print(f"There are non-lists in the {column} column.")
            print("Non-lists found at:")
            print(not_list[not_list])
        else:
            print(f"No non-lists found in the {column} column.")

    list_length_violations = data.apply(
        lambda row: any(isinstance(cell, list) and len(cell) > list_length_exceeds for cell in row),
        axis=1
    )

    if list_length_violations.any():
        print("There are lists that exceed the length of 30.")
        print("List length exceeds 30 at:")
        print(list_length_violations[list_length_violations])
    else:
        print(f"No lists exceed the length of {list_length_exceeds}.")

def get_species_thermo(smiles: str, arkane_path: str, species_type: str, rxn_name: str) -> dict:
    # All keys that parse_conformer may produce
    thermo_keys = [
        'label',
        'E0_value', 'E0_units',
        'spin_multiplicity', 'optical_isomers',
        'modes',
        'frequency_list', 'frequency_units', 'frequency_value'
    ]

    # If we have a real path and the file exists, parse it
    if arkane_path and os.path.exists(arkane_path):
        with open(arkane_path, 'r') as f:
            block = f.read()
        parsed = parse_conformer(block)
        # Ensure every key is present
        for key in thermo_keys:
            parsed.setdefault(key, None)
        path_exists = True
    else:
        # File missing or path is None: blank out every field
        parsed = {key: None for key in thermo_keys}
        path_exists = False

    # Now append your metadata
    parsed.update({
        'rxn':          rxn_name,
        'species_type': species_type,
        'smiles':      smiles,
        'arkane_path': arkane_path,
        'path_exists': path_exists
    })
    return parsed


def parse_and_order_reactions(parse_input_csv: str, parse_output_csv: str, parse_converged_rxn_dirs: list, parse_force: bool, ordered_coords_save_path, ordered_smiles_save_path, error_save_path, parsed_ordered: bool = False):
    parsed_data = parse_reactions_if_needed(
        input_csv=parse_input_csv,
        output_csv=parse_output_csv,
        converged_rxn_dirs=parse_converged_rxn_dirs,
        force=parse_force
    )

    validation_parse_data(parsed_data)

    if parsed_ordered:
        print("Data has been parsed and ordered.")
    else:
        ordered_HAbstract_coords, ordered_HAbstract_smiles, error_df = order_reactions(parsed_data, read_csv=True)
        ordered_HAbstract_coords.to_csv(ordered_coords_save_path, index=False)
        ordered_HAbstract_smiles.to_csv(ordered_smiles_save_path, index=False)
        error_df.to_csv(error_save_path)

        print("Finished")

if __name__ == '__main__':
    logger.info("Reading data file...")
    parse_and_order_reactions(
        parse_input_csv="/home/calvin/code/chemprop_phd_customised/habnet/data/extraction/processed/combined_data.csv",
        parse_output_csv="/home/calvin/code/chemprop_phd_customised/habnet/data/extraction/processed/parsed_data.csv",
        parse_converged_rxn_dirs=[
            '/home/calvin/Dropbox/PersonalFolders/Calvin/ATLAS_Converged',
            '/home/calvin/Dropbox/PersonalFolders/Calvin/HAb_Converged',
            '/home/calvin/Dropbox/PersonalFolders/Calvin/ATLAS_Converged/NonRMG',
            '/home/calvin/Dropbox/PersonalFolders/Calvin/ZEUS_Converged'],
        parse_force=False,
        ordered_coords_save_path="/home/calvin/code/chemprop_phd_customised/habnet/data/preprocessing/ordered_reactions_data/ordered_habstraction_coords.csv",
        ordered_smiles_save_path="/home/calvin/code/chemprop_phd_customised/habnet/data/preprocessing/ordered_reactions_data/ordered_habstraction_smiles.csv",
        error_save_path="/home/calvin/code/chemprop_phd_customised/habnet/data/preprocessing/ordered_reactions_data/order_habstraction_error.csv"
    )
