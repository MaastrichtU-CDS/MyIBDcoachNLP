#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import pickle
import random

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

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)


# ========================================================
# Argparse
# ========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Run BERTopic with best config and outlier reduction.")

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name (e.g. mpnet, robbert, qwen3)."
    )

    parser.add_argument(
        "--input-data",
        type=str,
        default="data/cleaned_patient_deduplicated_sentences_for_embedding.xlsx",
        help="Excel file containing sentences."
    )

    parser.add_argument(
        "--text-column",
        type=str,
        default="sentence",
        help="Column name containing text data."
    )

    parser.add_argument(
        "--input-embeddings",
        type=str,
        required=True,
        help="NumPy .npy file containing text embeddings."
    )

    parser.add_argument(
        "--input-tokens",
        type=str,
        default="data/tokens/tokenized_sentences.pkl",
        help="Pickle file containing tokenized texts."
    )

    parser.add_argument(
        "--input-stopwords",
        type=str,
        default="data/stopwords-nl-extended.txt",
        help="Text file containing Dutch stopwords, one per line."
    )

    parser.add_argument(
        "--best-configs-file",
        type=str,
        required=True,
        help="CSV file containing best parameter configurations per model."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/",
        help="Base output directory for checkpoints."
    )
    return parser.parse_args()


# ========================================================
# Outlier reduction helper
# ========================================================
import re

def remove_placeholders(text: str) -> str:
    # Remove all [UPPERCASE] or [UPPERCASE_UPPERCASE] style placeholders
    return re.sub(r'\[[A-Z]+(?:_[A-Z]+)?\]', ' ', text)

def reduce_outlier_for_model(topic_model, docs, embeddings, stopwords):
    """Reduce outliers for a BERTopic model using embeddings strategy."""
    topics = topic_model.topics_

    print(f"Original outliers: {sum(1 for t in topics if t == -1)}")

    new_topics = topic_model.reduce_outliers(
        documents=docs,
        topics=topics,
        strategy="embeddings",
        embeddings=embeddings
    )

    print(f"Remaining outliers after reduction: {sum(1 for t in new_topics if t == -1)}")

    # Rebuild vectorizer (same spec as initial one, but with given stopwords)
    vectorizer_model = CountVectorizer(
        stop_words=stopwords,
        min_df=2,
        ngram_range=(1, 1),
        token_pattern=r'\b[a-zA-Z]{3,}\b'
    )

    # Update topics in the model
    topic_model.update_topics(
        docs,
        topics=new_topics,
        vectorizer_model=vectorizer_model
    )

    return topic_model


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
        raise ValueError(f"Model '{model_name}' not found in best configs file.")

    best_row = (
        best_df[best_df["model_name"] == model_name]
        .sort_values("diversity_score")
        .iloc[0]
    )

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
    print("\nLoading data...")
    df = pd.read_excel(args.input_data)
    sentences = df[args.text_column].astype(str).tolist()

    with open(args.input_tokens, "rb") as f:
        tokenized_texts = pickle.load(f)
    dictionary = Dictionary(tokenized_texts)

    embeddings = np.load(args.input_embeddings)

    with open(args.input_stopwords, "r") as f:
        stopwords = [line.strip() for line in f if line.strip()]

    # ----------------------------------------------------
    # Build BERTopic with selected params
    # ----------------------------------------------------
    print("\nBuilding BERTopic model...")
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

    sentences_clean = [remove_placeholders(s) for s in sentences]

    topics, _ = model.fit_transform(sentences_clean, embeddings)
    '''
    vocab = model.vectorizer_model.vocabulary_
    tokens_with_persoon = [t for t in vocab.keys() if "persoon" in t]
    print(tokens_with_persoon)

    # Extract UMAP reduced embeddings
    umap_embeddings = model.umap_model.embedding_

    # Save them for later use
    np.save(
        f"embeddings/{model_name}_umap_embeddings.npy",
        umap_embeddings
    )
    print(f"Saved UMAP embeddings to embeddings/{model_name}_umap_embeddings.npy")
    '''
    # ----------------------------------------------------
    # Evaluate metrics (before outlier reassignment)
    # ----------------------------------------------------
    print("\n=== Evaluating metrics (before outlier reassignment) ===")
    top_words = get_top_words(model)
    coverage = get_coverage(model)
    nr_topics = get_nr_topics(model)
    diversity = get_topic_diversity(top_words)

    cm = CoherenceModel(
        topics=top_words,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence="c_v"
    )
    c_v = cm.get_coherence()

    print(f"Document assignment%:  {coverage:.4f}")
    print(f"Number of topics:   {nr_topics}")
    print(f"Diversity score:    {diversity:.4f}")
    print(f"C_V coherence:      {c_v:.4f}")

    # ----------------------------------------------------
    # Apply outlier reassignment
    # ----------------------------------------------------
    print("\nApplying automatic outlier reassignment...")
    model_reduced = reduce_outlier_for_model(
        topic_model=model,
        docs=sentences_clean,
        embeddings=embeddings,
        stopwords=stopwords
    )

    # ----------------------------------------------------
    # Evaluate metrics (after outlier reassignment)
    # ----------------------------------------------------
    print("\n=== Evaluating metrics (after outlier reassignment) ===")
    top_words_new = get_top_words(model_reduced)
    coverage_new = get_coverage(model_reduced)
    nr_topics_new = get_nr_topics(model_reduced)
    diversity_new = get_topic_diversity(top_words_new)

    cm_new = CoherenceModel(
        topics=top_words_new,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence="c_v"
    )
    c_v_new = cm_new.get_coherence()

    print(f"Document assignment %:  {coverage_new:.4f}")
    print(f"Number of topics:   {nr_topics_new}")
    print(f"Diversity score:    {diversity_new:.4f}")
    print(f"C_V coherence:      {c_v_new:.4f}")

    # ----------------------------------------------------
    # Save model + outputs
    # ----------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    base_model_name = f"{model_name}"

    model_path = os.path.join(args.output_dir, f"{base_model_name}_model")
    reduced_model_path = os.path.join(args.output_dir, f"{base_model_name}_model_reduced")

    print("\nSaving BERTopic models...")
    model.save(model_path, serialization="pytorch", save_ctfidf=True)
    model_reduced.save(reduced_model_path, serialization="pytorch", save_ctfidf=True)

    print("Saving topic info and document info...")
    topic_info = model.get_topic_info()
    reduced_topic_info = model_reduced.get_topic_info()
    doc_info = model.get_document_info(docs=sentences_clean)
    doc_info["Document_Original"] = sentences
    reduced_doc_info = model_reduced.get_document_info(docs=sentences_clean)
    reduced_doc_info["Document_Original"] = sentences
    topic_info.to_csv(
        os.path.join(args.output_dir, f"{base_model_name}_topic_info.csv"),
        index=False
    )
    reduced_topic_info.to_csv(
        os.path.join(args.output_dir, f"{base_model_name}_topic_info_reduced.csv"),
        index=False
    )
    doc_info.to_csv(
        os.path.join(args.output_dir, f"{base_model_name}_document_info.csv"),
        index=False
    )
    reduced_doc_info.to_csv(
        os.path.join(args.output_dir, f"{base_model_name}_document_info_reduced.csv"),
        index=False
    )

    print("\n=== Done ===")
    print(f"Outputs saved under: {args.output_dir}")


if __name__ == "__main__":
    main()
