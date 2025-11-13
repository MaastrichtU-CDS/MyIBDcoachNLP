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

import os
import datamapplot
import numpy as np
import pandas as pd
import datamapplot

def visualize_docs(topic_model, docs, reduced_embeddings):
    fig =  topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings)

    return fig


def visualize_datamap(
    topic_model,
    docs,
    reduced_embeddings,
    output_dir,
    fig_name="docs_datamap_selected_topics_labels.png",
    selected_topics = None,
    model_name="robbert",
    labels=None
):
    # Get the 30 most frequent topics (excluding -1)
    #freq = topic_model.get_topic_freq()
    #top_topics = freq.loc[freq.Topic != -1, "Topic"].head(30).tolist()

    # Use BERTopic's built-in datamap
    fig = topic_model.visualize_document_datamap(
        docs,
        topics=selected_topics,
        topic_prefix=True,
        custom_labels=labels,
        reduced_embeddings=reduced_embeddings,
        title="Document Map with Labels"
    )


    # Save figure
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

    return reduced_embeddings

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

        try:
            # load model
            print(f"Loading model from {model_path}")
            topic_model = BERTopic.load(model_path, embedding_model=None)
            print(f"Model loaded successfully from {model_path}")

            # get reduced embeddings
            reduced_embeddings = reduce_embeddings(embeddings_path, model_name)

            # import labels for the selected topics if any
            with open("./labels_selected_topics.txt", "r", encoding="utf-8") as f:
                lines = f.readlines()

            # strip newline characters (\n)
            labels = [line.strip() for line in lines]

            # Map selected topic IDs to labels
            custom_labels = dict(zip(
                [0, 1, 2, 3, 4, 5, 7, 12, 16, 21, 26, 29, 33, 48, 54, 79, 119],
                labels
            ))

            print(custom_labels)

            # visualize documents for the model
            visualize_datamap(
                topic_model,
                sentences,
                reduced_embeddings,
                output_dir,
                selected_topics=[0, 1, 2, 3, 4, 5, 7, 12, 16, 21, 26, 29, 33, 48, 54, 79, 119],
                labels = custom_labels
            )

            print(f"Document map for {model_name} saved.")

            #fig = visualize_docs(
                #topic_model,
                #sentences,
                #reduced_embeddings
            #)
            #fig.write_html(os.path.join(output_dir, f"visualized_docs_{model_name}.html"))

        except Exception as e:
            print(f"❌ Error processing {model_name}: {e}")

if __name__ == "__main__":

    main()
