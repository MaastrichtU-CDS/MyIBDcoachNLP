#!/usr/bin/env python3
import os
import pandas as pd
from bertopic import BERTopic
import ast
import joblib   # for saving/loading reduced embeddings
import matplotlib.pyplot as plt
import datamapplot
from umap import UMAP
import numpy as np

def read_topic_groups(filepath):
    topic_groups = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                try:
                    # Parse the line into a Python object (list of ints)
                    group = ast.literal_eval(line)
                    topic_groups.append(group)
                except Exception as e:
                    print(f"Could not parse line: {line} ({e})")
    return topic_groups

def automatic_merge_topics(topics_to_merge, topic_model, docs, output_dir):

    if topics_to_merge:
        topic_model.merge_topics(docs, topics_to_merge)

    # get topic and doc info after merging
    topic_info = topic_model.get_topic_info()
    document_info = topic_model.get_document_info(docs)

    #print("Saving merged model...")
    topic_info_path = os.path.join(output_dir, "topic_info_final.csv")
    doc_info_path = os.path.join(output_dir, "doc_info_final.csv")
    # Save topic and document info
    topic_info.to_csv(topic_info_path, index=False)
    #topic_info.to_excel(topic_info_path, index=False)
    document_info.to_csv(doc_info_path, index=False)

    return topic_model

def main():
    # Define model configurations
    model_paths = {
        #'mpnet': 'new_analysis/results/models/mpnet_reduced_top_diversity',
        'robbert': 'new_analysis/results/models/robbert_reduced',
        #'qwen3': 'new_analysis/results/models/qwen3_reduced_top_diversity'
    }
    

    print("\n=== Automatically Merging Topics ===")

    # Load data once
    data = pd.read_excel("./data/sentence_data_for_analysis.xlsx", index_col=0)
    sentences = data["sentence"].to_list()


    for model_name, model_path in model_paths.items():
        # import the list of topics to merge
        topics_to_merge_file = f"topics_to_merge_{model_name}.txt"
        topics_to_merge_robbert = read_topic_groups(topics_to_merge_file)

        # Create output dir
        output_dir = f"./new_analysis/results/models/{model_name}_final"
        os.makedirs(output_dir, exist_ok=True)

        if model_name == "mpnet":
            embedding_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        elif model_name == "robbert":
            embedding_model = "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers" 
        else:  # qwen3
            embedding_model = "Qwen/qwen3-embedding-8b"
        
        
        try:
            # load model
            print(f"Loading model from {model_path}")
            topic_model = BERTopic.load(model_path, embedding_model=None)
            print("Model loaded successfully")

            # Merge topics
            topic_model = automatic_merge_topics(topics_to_merge=topics_to_merge_robbert, topic_model=topic_model, docs=sentences, output_dir=output_dir)

            # save the topic model after merging
            print("Saving merged model...")
            topic_model.save(
                output_dir, 
                serialization="pytorch", 
                save_ctfidf=True, 
                save_embedding_model=embedding_model
            )

            print(f"Model {model_name} saved after merging.")

        except Exception as e:
            print(f"❌ Error processing {model_name}: {e}")

if __name__ == "__main__":

    main()
