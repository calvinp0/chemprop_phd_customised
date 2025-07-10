"""
The aim of this script is to utilise the ordered reactions data from 01_order_reactions_data.py, R1H,  R2H and the TS,
then use the TS Splitting algorithm to create TS-R1H and TS-R2H and attempt to connect them to the R1H and R2H data.

The output of the script will be a dataframe with correctly connected data.

To be discussed:

If we are splitting the TS into a TS-R1H and TS-R2H, and they have been properly centralised, does this mean they are not OPTIMISED anymore?

"""

import os
import sys

import pandas as pd
import json
from ast import literal_eval
from typing import Optional

bitsplitter_path = "~/code/BiTSplitter"
arc_path = "~/code/ARC"
rmg_path = "~/code/RMG-Py"
rmg_database_path = "~/code/RMG-database"
sys.path.append(os.path.expanduser(bitsplitter_path))
sys.path.append(os.path.expanduser(arc_path))
sys.path.append(os.path.expanduser(rmg_path))
sys.path.append(os.path.expanduser(rmg_database_path))

import h_abs_atoms, split, perception
from arc.species import ARCSpecies
from arc.species.converter import check_xyz_dict


def read_csv(file_path, dedupe: Optional[list] = None):
    
    if dedupe:
        return pd.read_csv(file_path).drop_duplicates(subset=dedupe).reset_index(drop=True)
    return pd.read_csv(file_path)

def merge_dataframes(df1, df2, on: Optional[str]=None, left_on: Optional[str]=None, right_on: Optional[str]=None, how: str='inner'):
    
    if on:
        return pd.merge(df1, df2, on=on, how=how)
    elif left_on and right_on:
        return pd.merge(df1, df2, left_on=left_on, right_on=right_on, how=how)
    else:
        raise ValueError(f"Please specify either on or left_on and right_on parameters - you provided on={on}, left_on={left_on}, right_on={right_on}")
    

def check_xyz_validity(g_xyz, index_h, index_a, distances: list = [ 1.1,1.0, 0.8,]):
    """
    Try to pull atoms closer using specified distances. If none of them work, return g_xyz.

    Args:
        g_xyz (dict): The geometry in XYZ format.
        index_h (int): Index of the hydrogen atom to adjust.
        index_a (int): Index of the atom to pull closer.
        distances (list): List of distances to try pulling the atoms closer.

    Returns:
        dict: The modified geometry or the original geometry if no modification succeeded.
    """

    for i in distances:
        try:
            new_xyz = h_abs_atoms.pull_atoms_closer(g_xyz, index_h, index_a, i)
            temp_mol = ARCSpecies(label='temp', xyz=new_xyz)
        except Exception as e:
            print(f"Error: {e}")
            continue
        else:
            if i != 0.8:
                print(f"Successfully pulled atoms closer using distance {i}")
            return new_xyz
        
    print("Could not pull atoms closer. Returning original geometry.")
    return g_xyz



def connect_reactions_data(idx, row, connections, errors):

    g1_xyz = None
    g2_xyz = None
    index_dict = None

    g1_xyz_pulled = None
    g2_xyz_pulled = None

    try:
        ts_list = literal_eval(row['ts'])
        ts_str = "\n".join(ts_list)
        ts_xyz = check_xyz_dict(ts_str)
        ts_df = h_abs_atoms.convert_xyz_to_df(ts_xyz)
        ts_results = h_abs_atoms.get_h_abs_atoms(ts_df)

        # Split the data
        g1_xyz, g2_xyz, index_dict = split.get_group_xyzs_and_key_indices_from_ts(
            xyz=ts_xyz,
            a=ts_results['A'],
            b=ts_results['B'],
            h=ts_results['H'],
        )

        # Validate the g1 / g2 groups
        g1_xyz_pulled = check_xyz_validity(g1_xyz, index_dict['g1_h'], index_dict['g1_a'])
        g2_xyz_pulled = check_xyz_validity(g2_xyz, index_dict['g2_h'], index_dict['g2_a'])

        # We’ll track which group aligns to which column
        # Example: assignments['g1'] = 'r1h' means g1_xyz_pulled aligns with r1h
        assignments = {'g1': None, 'g2': None}

        claimed_alignments = set()
        column_list_names = ['r1h', 'r2h']

        # We loop through each "pulled" group (g1, g2) and each target column (r1h, r2h)
        for j in [g1_xyz_pulled, g2_xyz_pulled]:
            for i, molecules in enumerate([row['r1h'], row['r2h']]):
                if column_list_names[i] in claimed_alignments:
                    # Already matched this column, skip it
                    continue

                # Compare j with the molecules from the row
                if perception.check_isomorphism(j, check_xyz_dict("\n".join(literal_eval(molecules)))):
                    rxn_key = row['rxn']
                    if rxn_key not in connections:
                        connections[rxn_key] = {column_list_names[i]: j}
                    else:
                        connections[rxn_key][column_list_names[i]] = j

                    claimed_alignments.add(column_list_names[i])

                    # Determine which group (g1 or g2) is j
                    if j is g1_xyz_pulled:
                        assignments['g1'] = column_list_names[i]
                    else:
                        assignments['g2'] = column_list_names[i]

                    break

        # If we haven't matched both columns, record an error
        if len(claimed_alignments) != 2:
            print(f"Could not connect {row['rxn']}")
            errors[row['rxn']] = "Could not connect the data"
        else:
            print(f"Successfully connected {row['rxn']}")

        # ------------------------------------------------
        # RENAME index_dict KEYS BASED ON THE ASSIGNMENT
        # ------------------------------------------------
        # For example, if g1_xyz_pulled matched 'r1h', then rename
        #    g1_a → r1h_a, g1_h → r1h_h.
        # If g2_xyz_pulled matched 'r2h', then rename
        #    g2_a → r2h_a, g2_h → r2h_h, etc.

        # We'll do a small helper function to rename the dictionary keys
        def rename_in_index_dict(group_prefix, assigned_prefix, idx_dict):
            """
            For keys that start with group_prefix (e.g. 'g1_'),
            rename them to assigned_prefix (e.g. 'r1h_').
            """
            old_a = f"{group_prefix}_a"
            old_h = f"{group_prefix}_h"
            new_a = f"{assigned_prefix}_a"
            new_h = f"{assigned_prefix}_h"

            if old_a in idx_dict:
                idx_dict[new_a] = idx_dict.pop(old_a)
            if old_h in idx_dict:
                idx_dict[new_h] = idx_dict.pop(old_h)

        # Rename for g1
        if assignments['g1'] is not None:
            rename_in_index_dict('g1', assignments['g1'], index_dict)

        # Rename for g2
        if assignments['g2'] is not None:
            rename_in_index_dict('g2', assignments['g2'], index_dict)

        # Now index_dict is updated with the new key names
        connections[row['rxn']]['index_dict'] = index_dict
        
        connections[row['rxn']]['ts'] = ts_xyz
        # Rename 'H' to '*2', 'A' to '*1' and 'B' to '*3' in ts_results keys
        ts_results = {k.replace('H', '*2').replace('A', '*1').replace('B', '*3').replace('C', '*0').replace('D', '*4'): v for k, v in ts_results.items()}
        connections[row['rxn']]['ts_results'] = ts_results


    except Exception as e:
        print(f"Error: {e}")
        errors[row['rxn']] = str(e)
    


    return connections, errors, g1_xyz_pulled, g2_xyz_pulled, index_dict




def connect_reactions(ordered_coords_path, reaction_data_path, out_path_connections, out_path_errors):    # Load in the ordered reactions data
    ordered_reactions = read_csv(ordered_coords_path, dedupe=['rxn', 'r1h', 'r2h', 'r1', 'r2','ts'])
    reaction_data = read_csv(reaction_data_path)
    coord_data = merge_dataframes(ordered_reactions, reaction_data, left_on='rxn', right_on='reaction', how='inner')
    coord_data = coord_data[coord_data['converged'] == True]
    connections = dict()
    errors = dict()
    g1_xyz = None
    g2_xyz = None
    index_dict = None

    coord_data['g1_xyz'] = None
    coord_data['g2_xyz'] = None
    coord_data['index_dict'] = None

    coord_data['g1_xyz'] = coord_data['g1_xyz'].astype(object)
    coord_data['g2_xyz'] = coord_data['g2_xyz'].astype(object)
    coord_data['index_dict'] = coord_data['index_dict'].astype(object)

    for idx, row in coord_data.iterrows():
        connections, errors, g1_xyz, g2_xyz, index_dict = connect_reactions_data(idx, row, connections, errors)
        coord_data.at[idx, 'g1_xyz'] = json.dumps(g1_xyz)
        coord_data.at[idx, 'g2_xyz'] = json.dumps(g2_xyz)
        coord_data.at[idx, 'index_dict'] = json.dumps(index_dict)

    # Remove connections that do not have both R1H and R2H data
    for key in list(connections.keys()):
        if 'r1h' not in connections[key] or 'r2h' not in connections[key]:
            del connections[key]
    # Save the connected data to a dataframe
    connected_data = pd.DataFrame(connections).T
    connected_data.to_csv(out_path_connections)

    # Report how many connections were made compared to the total number of reactions
    print(f"Connections made: {len(connected_data)}")
    print(f"Total number of reactions: {len(coord_data)}")

    # Save the errors to a file
    errors_df = pd.DataFrame(errors.items(), columns=['rxn', 'error'])
    errors_df.to_csv(out_path_errors)

    return connected_data, errors, g1_xyz, g2_xyz, index_dict

if __name__ == "__main__":
    connect_reactions(
        "/home/calvin/code/chemprop_phd_customised/habnet/data/preprocessing/ordered_reactions_data/ordered_habstraction_coords.csv",
        "/home/calvin/code/arc_analysis/reaction_data.csv",
        "data/connected_reactions.csv",
        "data/error_connections.csv"
    )


