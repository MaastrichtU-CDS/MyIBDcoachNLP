import pickle
import os

def load_tokens_and_dictionaries(load_path):
    # Define the patterns for tokens and dictionaries
    tokens_files = {
        "tokens_mes.pkl": "mes",
        "tokens_mes_lem.pkl": "mes_lem",
        "tokens_mes_stem.pkl": "mes_stem",
        "tokens_sen.pkl": "sen",
        "tokens_sen_lem.pkl": "sen_lem",
        "tokens_sen_stem.pkl": "sen_stem"
    }
    
    dict_files = {
        "dict_mes.pkl": "mes",
        "dict_mes_lem.pkl": "mes_lem",
        "dict_mes_stem.pkl": "mes_stem",
        "dict_sen.pkl": "sen",
        "dict_sen_lem.pkl": "sen_lem",
        "dict_sen_stem.pkl": "sen_stem"
    }

    # Initialize a dictionary to store tokens and their corresponding dictionaries together
    loaded_data = {}

    # Load all token files and store in the dictionary
    for filename, key in tokens_files.items():
        file_path = os.path.join(load_path, filename)
        with open(file_path, "rb") as f:
            loaded_data[key] = {
                "tokens": pickle.load(f),
                "dictionary": None  # Placeholder, to be filled later
            }

    # Load all dictionary files and pair with their corresponding tokens
    for filename, key in dict_files.items():
        file_path = os.path.join(load_path, filename)
        
        with open(file_path, "rb") as f:
            # Assign the dictionary to the corresponding key
            loaded_data[key]["dictionary"] = pickle.load(f)

    print("All tokens and dictionaries loaded successfully.")
    
    # Return the combined data
    return loaded_data
