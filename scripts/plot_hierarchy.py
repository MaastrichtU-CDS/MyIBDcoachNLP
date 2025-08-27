#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from bertopic import BERTopic
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)


def get_hierarchical_topics(topic_model, docs):
    hierarchical_topics = topic_model.hierarchical_topics(docs)
    fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
        # Adjust figure size
    fig.update_layout(
        width=1200,   # wider
        height=3000,  # much taller so nodes don't overlap
    )
    return hierarchical_topics, fig

def main():
    # Define model configurations - FIXED TYPO IN QWEN3 PATH
    model_paths = {
        'mpnet': 'new_analysis/results/models/mpnet_reduced',
        'robbert': 'new_analysis/results/models/robbert_reduced' 
        #'qwen3': 'new_analysis/results/models/qwen3_reduced'  # Fixed: removed extra 'l' and leading dot
    }
    
    # Load data once
    data = pd.read_excel("./data/sentence_data_for_analysis.xlsx", index_col=0)
    sentences = data["sentence"].to_list()

    for model_name in model_paths.keys():
        print(f"\n=== Processing {model_name} model ===")
        
        # Create output dir
        output_dir = f"./new_analysis/results"
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
            topic_model = BERTopic.load(model_path, embedding_model=None)
            print(f"Model (Reduced Outliers) {model_name} loaded successfully")

            hierarchical_topics, fig = get_hierarchical_topics(topic_model, sentences)
            
            fig.write_html(os.path.join(output_dir, f"hierarchy_{model_name}.html"))
            hierarchical_topics.to_csv(os.path.join(model_paths[model_name], f"topic_hierarchy_{model_name}.csv"))
            print(f"Results of {model_name} saved.")

        except Exception as e:
            print(f"Error processing model {model_name}: {str(e)}")
            continue


if __name__ == "__main__":
    main()