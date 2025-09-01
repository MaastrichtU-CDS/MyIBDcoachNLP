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


def visualize_documents(topic_model, docs, reduced_embeddings, output_dir, fig_name, model_name="robbert", labels=None):
    # Get the 20 most frequent topics (excluding -1)
    freq = topic_model.get_topic_freq()
    top_topics = freq.loc[freq.Topic != -1, "Topic"].head(15).tolist()
    selected_topics = top_topics
    fig = topic_model.visualize_document_datamap(
        docs,
        topics=selected_topics,
        topic_prefix=True,
        reduced_embeddings=reduced_embeddings,
        title="Document Map with Top 15 Topics"
    )

    fig.savefig(os.path.join(output_dir, fig_name), bbox_inches="tight", dpi=300)
    return None

def reduce_embeddings(embeddings_path, model_name):

    # Load embeddings for this model
    
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    embeddings = np.load(embeddings_path)

    # Path to save/load reduced embeddings
    reduced_embed_path = f"./data/umap_embeddings_{model_name}.pkl"

    if os.path.exists(reduced_embed_path):
        print(f"Loading reduced embeddings from {reduced_embed_path}")
        reduced_embeddings = joblib.load(reduced_embed_path)
    else:
        print("Computing embeddings and UMAP reduction...")
        reduce_embeddings(reduced_embed_path, embeddings)

    reducer = UMAP(
        n_neighbors=10,
        n_components=2,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )
    reduced_embeddings = reducer.fit_transform(embeddings)

    # Save for later reuse
    joblib.dump(reduced_embeddings, reduced_embed_path)
    print(f"Saved reduced embeddings to {reduced_embed_path}")

    return None

def main():
    # Define model configurations
    model_paths = {
        #'mpnet': 'new_analysis/results/models/mpnet_reduced_top_diversity',
        'robbert': 'new_analysis/results/models/robbert_final',
        #'qwen3': 'new_analysis/results/models/qwen3_reduced_top_diversity'
    }
    

    print("\n=== Visualizing Documents ===")

    # Load data once
    data = pd.read_excel("./data/sentence_data_for_analysis.xlsx", index_col=0)
    sentences = data["sentence"].to_list()


    for model_name, model_path in model_paths.items():
        # Create output dir
        output_dir = f"./new_analysis/results/graphs/"
        os.makedirs(output_dir, exist_ok=True)

        # get embeddings path
        embeddings_path = f"./data/embeddings_{model_name}.npy"

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
            print(f"Model loaded successfully from {model_path}")

            # get reduced embeddings
            reduced_embeddings = reduce_embeddings(embeddings_path, model_name)

            # visualize documents for the model before merging
            visualize_documents(
                topic_model,
                sentences,
                reduced_embeddings,
                output_dir,
                fig_name=f"doc_map_{model_name}_merged.png"
            )
            print(f"Document map for {model_name} saved.")

        except Exception as e:
            print(f"❌ Error processing {model_name}: {e}")

if __name__ == "__main__":

    main()
