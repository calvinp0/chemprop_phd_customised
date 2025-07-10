import pandas as pd
import yaml
import os
import re

main_folder = "/home/calvin/Dropbox/PersonalFolders/Calvin/Kfir_HAb_Converged"

data_reactions = pd.DataFrame(columns=['reactant_0', 'reactant_1', 'product_0', 'product_1', 'reaction', 'folder_name'])

def standardize_label(label):
    
    label = re.sub(r'(?<=\S)\+(?=\S)', 'p', label)

    return label.strip().replace('(', '[').replace(')', ']')

for folder in os.listdir(main_folder):
    folder_path = os.path.join(main_folder, folder)
    
    print("Processing folder:", folder)
    using_info_file = None
    input_yml_path = os.path.join(folder_path, "input.yml")
    
    # If input.yml does not exist, look for a _info.yml file
    if os.path.exists(input_yml_path):
        input_file_path = input_yml_path
    else:
        input_files = [f for f in os.listdir(folder_path) if f.endswith("_info.yml")]
        if not input_files:
            print("No input file found in folder:", folder)
            continue
        using_info_file = input_files[0]
        info_file_path = os.path.join(folder_path, using_info_file)
        
        # Load the _info.yml file
        with open(info_file_path, 'r') as input_file:
            input_data = yaml.load(input_file, Loader=yaml.FullLoader)
        
        # Standardize the reaction label if needed
        if 'reactions' in input_data and input_data['reactions']:
            reaction = input_data['reactions'][0]['label']
            # Standardize species-related parts (without affecting delimiters like " + ")
            input_data['reactions'][0]['label'] = standardize_label(reaction)
        
        # Write a new input.yml with the standardized reaction label
        with open(input_yml_path, 'w') as out_file:
            yaml.dump(input_data, out_file)
        input_file_path = input_yml_path

    # Load input data from input.yml
    with open(input_file_path, 'r') as input_file:
        input_data = yaml.load(input_file, Loader=yaml.FullLoader)
    
    # Determine the project name.
    # If 'project' isn't available, derive it from the _info.yml filename (if applicable) or use the folder name.
    project_name = input_data['project'] if 'project' in input_data else (using_info_file[:-9] if using_info_file else folder)
    
    # Avoid a double 'kfir_' prefix: if project_name already starts with it, use it as-is.
    if project_name.startswith("kfir_"):
        new_folder_name = project_name
    else:
        new_folder_name = "kfir_" + project_name
    
    new_folder_path = os.path.join(main_folder, new_folder_name)
    
    # Rename the folder if the new folder name does not already exist
    if not os.path.exists(new_folder_path):
        os.rename(folder_path, new_folder_path)
        folder_name = new_folder_name
    else:
        folder_name = folder
    
    # Process the reaction details
    reaction = input_data['reactions'][0]['label']
    reactants, products = reaction.split('<=>')
    reactant_0, reactant_1 = reactants.split(" + ")
    product_0, product_1 = products.split(" + ")
    
    dict_reactions = {"reactant_0": [], "reactant_1": [], "product_0": [], "product_1": []}
    species = input_data['species']
    
    # Attempt matching species labels directly
    for specie in species:
        if specie['label'] == reactant_0.strip():
            dict_reactions['reactant_0'].append(specie['smiles'])
        elif specie['label'] == reactant_1.strip():
            dict_reactions['reactant_1'].append(specie['smiles'])
        elif specie['label'] == product_0.strip():
            dict_reactions['product_0'].append(specie['smiles'])
        elif specie['label'] == product_1.strip():
            dict_reactions['product_1'].append(specie['smiles'])
    
    # If any key is still empty, try matching with standardized labels
    if (not dict_reactions['reactant_0'] or not dict_reactions['reactant_1'] or 
        not dict_reactions['product_0'] or not dict_reactions['product_1']):
        for specie in species:
            if specie['label'] == standardize_label(reactant_0.strip()) and not dict_reactions['reactant_0']:
                dict_reactions['reactant_0'].append(specie['smiles'])
            elif specie['label'] == standardize_label(reactant_1.strip()) and not dict_reactions['reactant_1']:
                dict_reactions['reactant_1'].append(specie['smiles'])
            elif specie['label'] == standardize_label(product_0.strip()) and not dict_reactions['product_0']:
                dict_reactions['product_0'].append(specie['smiles'])
            elif specie['label'] == standardize_label(product_1.strip()) and not dict_reactions['product_1']:
                dict_reactions['product_1'].append(specie['smiles'])
    
    df = pd.DataFrame(dict_reactions)
    df['reaction'] = new_folder_name  # corresponds to the standardized project name
    df['folder_name'] = folder_name
    data_reactions = pd.concat([data_reactions, df], ignore_index=True)

data_reactions.to_csv("/home/calvin/code/chemprop_phd_customised/habnet/data/extraction/raw/data_kfir_reactions.csv", index=False)
print("Data extracted, folders renamed, and input files updated successfully")
