"""
A script for combining the processed data files into one file
Relies on:
    - /data/processed/converged_zeus_data.csv
    - /data/processed/converged_zeus_data_rmg.csv
    - /data/processed/converged_atlas_data_rmg.csv
    - /data/processed/converged_atlas_data.csv
"""
import os
import pandas as pd

PATH_TO_DATAFILES = "../data/processed"


def check_data_files(path_to_datafiles):
    """
    Check if the necessary files are present
    """

    data_files = [
        "converged_zeus_data.csv",
        "converged_zeus_data_rmg.csv",
        "converged_atlas_data_rmg.csv",
        "converged_atlas_data.csv",
        "converged_zeus_kfir_data.csv"
    ]

    for file in data_files:
        if not os.path.exists(os.path.join(path_to_datafiles, file)):
            print(f"Error: {file} not found")
            return False
        
    return True


def process_csv(file_path, source_name, starting_reaxction_id):
    """
    Processes a CSV file to standardize columns, add source info, and assign unique IDs.

    Parameters:
    - file_path (str): The path to the CSV file.
    - source_name (str): 'ZEUS' or 'ATLAS'.
    - starting_reaction_id (int): The starting point for unique IDs.

    Returns:
    - df (DataFrame): The processed DataFrame.
    - next_reaction_id (int): The next available unique reaction ID.
    """
    df = pd.read_csv(file_path)
    
    required_columns = ['reactant_0', 'reactant_1', 'product_0', 'product_1', 'reaction']
    column_mapping = {col: col for col in df.columns if col in required_columns}
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Error: {col} not found in {file_path}")
    
    df = df.rename(columns=column_mapping)
    
    df = df[required_columns]
    
    df['original_reaction_id'] = df['reaction']
    df.drop(columns=['reaction'], inplace=True)
    # Assign unique reaction IDs
    num_reactions = len(df)
    df['reaction_id'] = range(starting_reaxction_id, starting_reaxction_id + num_reactions)
    next_reaction_id = starting_reaxction_id + num_reactions
    
    # Add source info
    df['source'] = source_name
    
    return df, next_reaction_id

def CombineProcessedData(path_to_datafiles):

    starting_reaction_id = 0
    combined_data = pd.DataFrame()

    if not check_data_files(path_to_datafiles):
        raise FileNotFoundError("Error: Missing data files")

    files_to_process = [
        {'file_path': os.path.join(path_to_datafiles, "converged_zeus_data.csv"), 'source_name': 'ZEUS'},
        {'file_path': os.path.join(path_to_datafiles, "converged_zeus_data_rmg.csv"), 'source_name': 'ZEUS'},
        {'file_path': os.path.join(path_to_datafiles, "converged_atlas_data_rmg.csv"), 'source_name': 'ATLAS'},
        {'file_path': os.path.join(path_to_datafiles, "converged_atlas_data.csv"), 'source_name': 'ATLAS'},
        {'file_path': os.path.join(path_to_datafiles, "converged_zeus_kfir_data.csv"), 'source_name': 'KFIR_ZEUS'}
    ]

    for file_info in files_to_process:
        file_path = file_info['file_path']
        source_name = file_info['source_name']
        
        df_processed, starting_reaction_id = process_csv(file_path, source_name, starting_reaction_id)
        
        combined_data = pd.concat([combined_data, df_processed], ignore_index=True)

    combined_data.to_csv(os.path.join(path_to_datafiles, "combined_data.csv"), index=False)

    print("Data files combined successfully")

if __name__ == "__main__":
    CombineProcessedData()
