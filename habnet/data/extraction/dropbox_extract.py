"""
Determine which reactions from /data/raw/... are converged and already in the dropbox folder
"""
import os
import pandas as pd


def read_connect_zeus_kfir_data(path_dropbox, save_path, data):

    df = pd.read_csv(data)

    converged_zeus_data_folder = os.path.join(path_dropbox, "Kfir_HAb/inputs")

    output_data_frame = df.copy()
    output_data_frame['reaction'] = output_data_frame['reaction'].str.replace('reaction_', 'rxn_')

    list_of_folders = pd.DataFrame(os.listdir(converged_zeus_data_folder), columns=['reaction'])
    # Remove the following folders from the list
    # "chat_lot.sh", "submit.sh", "test_out.sh", ".chat_lot.sh.swp", ".submit.sh.swp", ".test_out.sh.swp"
    list_of_folders = list_of_folders[~list_of_folders['reaction'].str.contains("chat_lot.sh|submits.sh|test_out.sh|.chat_lot.sh.swp|.submits.sh.swp|.test_out.sh.swp")]
    # Change list_of_folders['reaction'] to int64 type
    # list_of_folders['reaction_folder'] = list_of_folders['reaction_folder'].astype('int64')

    # Merge the two dataframes via an inner join, on the 'reaction' column
    # output_data_frame = pd.merge(output_data_frame, list_of_folders, left_on="folder_name", right_on="reaction_folder", how='inner')
    output_data_frame = pd.merge(output_data_frame, list_of_folders, on="reaction", how='inner')

    output_data_frame.to_csv(os.path.join(save_path, "converged_zeus_kfir_data.csv"), index=False)



def read_connect_zeus_data(path_dropbox, save_path, data):

    df = pd.read_csv(data)

    converged_zeus_data_folder = os.path.join(path_dropbox, "HAb_Converged")

    output_data_frame = df.copy()
    output_data_frame['reaction'] = output_data_frame['reaction'].str.replace('reaction_', 'rxn_')

    list_of_folders = pd.DataFrame(os.listdir(converged_zeus_data_folder), columns=['reaction'])
    
    # Merge the two dataframes via an inner join, on the 'reaction' column
    output_data_frame = pd.merge(output_data_frame, list_of_folders, on='reaction', how='inner')

    output_data_frame.to_csv(os.path.join(save_path, "converged_zeus_data.csv"), index=False)

def read_connect_zeus_data_rmg(path_dropbox, save_path, data):

    df = pd.read_csv(data)

    converged_zeus_data_rmg_folder = os.path.join(path_dropbox, "ZEUS_Converged")

    output_data_frame = df.copy()
    output_data_frame['reaction'] = output_data_frame['reaction'].str.replace('reaction_', 'rxn_')

    list_of_folders = pd.DataFrame(os.listdir(converged_zeus_data_rmg_folder), columns=['reaction'])

    output_data_frame = pd.merge(output_data_frame, list_of_folders, on='reaction', how='inner')

    output_data_frame.to_csv(os.path.join(save_path, "converged_zeus_data_rmg.csv"), index=False)


def read_connect_atlas_data_rmg(path_dropbox, save_path, data):
    
    df = pd.read_csv(data)
    
    converged_atlas_data_rmg_folder = os.path.join(path_dropbox, "ATLAS_Converged")

    output_data_frame = df.copy()

    list_of_folders = pd.DataFrame(os.listdir(converged_atlas_data_rmg_folder), columns=['reaction'])
    
    output_data_frame = pd.merge(output_data_frame, list_of_folders, on='reaction', how='inner')
    
    output_data_frame.to_csv(os.path.join(save_path, "converged_atlas_data_rmg.csv"), index=False)
    
def read_connect_atlas_data(path_dropbox, save_path, data):
    
    df = pd.read_csv(data)
    
    converged_atlas_data_folder = os.path.join(path_dropbox, "ATLAS_Converged", "NonRMG")
    
    output_data_frame = df.copy()

    list_of_folders = pd.DataFrame(os.listdir(converged_atlas_data_folder), columns=['reaction'])
    
    output_data_frame = pd.merge(output_data_frame, list_of_folders, on='reaction', how='inner')
    
    output_data_frame.to_csv(os.path.join(save_path, "converged_atlas_data.csv"), index=False)


def RawDataDropbox(path_dropbox, raw_data_path, save_path):# Run the functions
    read_connect_zeus_kfir_data(path_dropbox=path_dropbox, data=os.path.join(raw_data_path, "data_kfir_reactions.csv"), save_path=save_path)
    read_connect_zeus_data(path_dropbox=path_dropbox, data=os.path.join(raw_data_path,"zeus_data.csv"), save_path=save_path)
    read_connect_zeus_data_rmg(path_dropbox=path_dropbox, data=os.path.join(raw_data_path, "zeus_data_rmg.csv"), save_path=save_path)
    read_connect_atlas_data_rmg(path_dropbox=path_dropbox, data = os.path.join(raw_data_path,  "atlas_data_rmg.csv"), save_path=save_path)
    read_connect_atlas_data(path_dropbox=path_dropbox, data = os.path.join(raw_data_path, "atlas_data.csv"), save_path=save_path)

if __name__ == "__main__":
    RawDataDropbox()