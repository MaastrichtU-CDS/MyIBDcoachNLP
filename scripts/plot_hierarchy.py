#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from bertopic import BERTopic
import pickle
import random

def get_hierarchical_topics(topic_model):
    get_hierarchical_topics = topic_model.get_hierarchy()
    hierarchy_graph = 
    
    return None

def main():
    # Define model configurations - FIXED TYPO IN QWEN3 PATH
    model_paths = {
        'mpnet': 'new_analysis/results/models/mpnet_reduced',
        'robbert': 'new_analysis/results/models/robbert_reduced', 
        'qwen3': 'new_analysis/results/models/qwen3_reduced'  # Fixed: removed extra 'l' and leading dot
    }
    
    # Load data once
    data = pd.read_excel("./data/sentence_data_for_analysis.xlsx", index_col=0)
    sentences = data["sentence"].to_list()

    for model_name in model_paths.keys():
        print(f"\n=== Processing {model_name} model ===")
        
        # Create output dir
        output_dir = f"./new_analysis/results/graphs"
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Load model with appropriate embedding model
            model_path = model_paths[model_name]
            
            if model_name == "mpnet":
                embedding_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            elif model_name == "robbert":
                embedding_model = "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers" 
            else:  # qwen3
                embedding_model = "Qwen/qwen3-embedding-8b"
            
            print(f"Loading model from {model_path}")
            topic_model = BERTopic.load(model_path, embedding_model=embedding_model)
            print("Model (Reduced Outliers) loaded successfully")

            get_hierarchical_topics = get_hierarchical_topics(topic_model)


        except Exception as e:
            print(f"Error processing model {model_name}: {str(e)}")
            continue