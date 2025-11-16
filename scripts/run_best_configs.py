#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import pickle

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
from umap import UMAP
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from run_grid_chunk import get_topic_diversity
from run_grid_chunk import get_top_words
from run_grid_chunk import get_coverage
from run_grid_chunk import get_nr_topics


# ========================================================
# Argparse
# ========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BERTopic with best parameters selected from grid search."
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name to run (e.g. mpnet, robbert, qwen3)"
    )

    parser.add_argument(
        "--best-configs-file",
        type=str,
        required=True,
        help="CSV file containing filtered best configs (e.g. top_model_configs.csv)"
    )

    parser.add_argument(
        "--input-data",
        type=str,
        default="./data/cleaned_patient_deduplicated_sentences_for_embedding.xlsx",
        help="Path to Excel file containing sentences"
    )

    parser.add_argument(
        "--text-column",
        type=str,
        default="sentence",
        help="Name of column containing text data"
    )

    parser.add_argument(
        "--tokenized-texts",
        type=str,
        default="/home/jzhang/mijnidbcoachnlp/data/tokens/tokenized_sentences.pkl",
        help="Pickle file with tokenized texts for coherence evaluation"
    )

    parser.add_argument(
        "--stopwords-file",
        type=str,
        default="./data/stopwords-nl-extended.txt",
        help="Stopword file (one per line)"
    )

    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="./data",
        help="Directory containing embeddings_{model_name}.npy"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the BERTopic model and outputs"
    )

    return parser.parse_args()

# ========================================================
# MAIN
# ========================================================
def main():
    args = parse_args()

    model_name = args.model_name

    # ----------------------------------------------------
    # Load best parameter configs
    # ----------------------------------------------------
    best_df = pd.read_csv(args.best_configs_file)

    if model_name not in best_df["model_name"].unique():
        raise ValueError(
            f"Model '{model_name}' not found in best configs file."
        )

    best_row = (
        best_df[best_df["model_name"] == model_name]
        .sort_values("diversity_score")
        .iloc[0]
    )

    # Extract parameters
    min_cluster_size = int(best_row["min_cluster_size"])
    n_components = int(best_row["n_components"])
    n_neighbors = int(best_row["n_neighbors"])

    print("\n=== Running with selected best configuration ===")
    print(f"Model:            {model_name}")
    print(f"min_cluster_size: {min_cluster_size}")
    print(f"n_components:     {n_components}")
    print(f"n_neighbors:      {n_neighbors}")

    # ----------------------------------------------------
    # Load data
    # ----------------------------------------------------
    df = pd.read_excel(args.input_data)
    sentences = df[args.text_column].astype(str).tolist()

    with open(args.tokenized_texts, "rb") as f:
        tokenized_texts = pickle.load(f)
    dictionary = Dictionary(tokenized_texts)

    embeddings_path = os.path.join(
        args.embeddings_dir,
        f"embeddings_{model_name}.npy"
    )
    embeddings = np.load(embeddings_path)

    with open(args.stopwords_file, "r") as f:
        stopwords = [line.strip() for line in f]

    # ----------------------------------------------------
    # Build BERTopic with selected params
    # ----------------------------------------------------
    vectorizer = CountVectorizer(
        stop_words=stopwords,
        min_df=2,
        ngram_range=(1, 1),
        token_pattern=r'\b[a-zA-Z]{3,}\b'
    )

    model = BERTopic(
        vectorizer_model=vectorizer,
        calculate_probabilities=False,
        verbose=True
    )

    model.hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False
    )

    model.umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )

    # ----------------------------------------------------
    # Fit model
    # ----------------------------------------------------
    print("\nFitting BERTopic model...")
    topics, _ = model.fit_transform(sentences, embeddings)

    # ----------------------------------------------------
    # Evaluate metrics (print only)
    # ----------------------------------------------------
    print("\n=== Evaluating metrics ===")
    top_words = get_top_words(model)
    coverage = get_coverage(model)
    nr_topics = get_nr_topics(model)
    diversity = get_topic_diversity(top_words)

    # Compute C_V coherence
    cm = CoherenceModel(
        topics=top_words,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence="c_v"
    )
    c_v = cm.get_coherence()

    print(f"Document coverage:  {coverage:.4f}")
    print(f"Number of topics:   {nr_topics}")
    print(f"Diversity score:    {diversity:.4f}")
    print(f"C_V coherence:      {c_v:.4f}")

    # ----------------------------------------------------
    # Save model + outputs
    # ----------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    print("\nSaving BERTopic model...")
    model.save(args.output_dir, serialization="pytorch", save_ctfidf=True)

    print("\nSaving topic info and documents...")
    model.get_topic_info().to_csv(
        os.path.join(args.output_dir, "topic_info.csv"), index=False
    )
    model.get_document_info(docs=sentences).to_csv(
        os.path.join(args.output_dir, "document_info.csv"), index=False
    )

    print("\n=== Done ===")
    print(f"Outputs saved under: {args.output_dir}")


if __name__ == "__main__":
    main()
