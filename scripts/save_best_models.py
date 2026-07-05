#!/usr/bin/env python3
import os
import re
import argparse
import random

import numpy as np
import pandas as pd

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
from umap import UMAP


# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)


# ========================================================
# Argparse
# ========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run BERTopic models from top_ranked_models.csv, "
            "save base models, reduce outliers, and save reduced models."
        )
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "Optional model name to run. If omitted, all models in the specs CSV "
            "are run. Example: robbert, qwen3."
        ),
    )

    parser.add_argument(
        "--input-data",
        type=str,
        default="data/cleaned_patient_deduplicated_sentences_for_embedding.xlsx",
        help="Excel file containing sentences.",
    )

    parser.add_argument(
        "--text-column",
        type=str,
        default="sentence",
        help="Column name containing text data.",
    )

    parser.add_argument(
        "--input-embeddings",
        type=str,
        default="embeddings/{model_name}/{model_name}_sentence_embeddings.npy",
        help=(
            "NumPy .npy embeddings fiile path. For example: embeddings/robbert/robbert_sentence_embeddings.npy"
        ),
    )

    parser.add_argument(
        "--input-stopwords",
        type=str,
        default="data/stopwords-nl-extended.txt",
        help="Text file containing Dutch stopwords, one per line.",
    )

    parser.add_argument(
        "--model-specs-file",
        type=str,
        default="results/model_comparison/top_ranked_models.csv",
        help="CSV file containing the selected model specs per model.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/{model_name}",
        help="Base output directory for saved models and topic/document info.",
    )

    return parser.parse_args()


# ========================================================
# Helpers
# ========================================================
def remove_placeholders(text: str) -> str:
    """Remove [UPPERCASE] or [UPPERCASE_UPPERCASE] placeholders."""
    return re.sub(r"\[[A-Z]+(?:_[A-Z]+)?\]", " ", text)


def make_vectorizer(stopwords):
    return CountVectorizer(
        stop_words=stopwords,
        min_df=2,
        ngram_range=(1, 1),
        token_pattern=r"\b[a-zA-Z]{3,}\b",
    )


def build_model(min_cluster_size, n_components, n_neighbors, stopwords):
    model = BERTopic(
        vectorizer_model=make_vectorizer(stopwords),
        calculate_probabilities=False,
        verbose=True,
    )

    model.hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False,
    )

    model.umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    return model


def reduce_outliers_for_model(topic_model, docs, embeddings, stopwords):
    """Reduce outliers for a BERTopic model using the embeddings strategy."""
    topics = topic_model.topics_
    print(f"Original outliers: {sum(1 for t in topics if t == -1)}")
    print(f"Total topics before reduction: {len(topic_model.get_topic_info())}")
    new_topics = topic_model.reduce_outliers(
        documents=docs,
        topics=topics,
        strategy="embeddings",
        embeddings=embeddings,
    )

    print(f"Remaining outliers after reduction: {sum(1 for t in new_topics if t == -1)}")
    topic_model.update_topics(
        docs,
        topics=new_topics,
        vectorizer_model=make_vectorizer(stopwords),
    )
    print(f"Total topics after reduction: {len(topic_model.get_topic_info())}")

    return topic_model


def resolve_embeddings_path(input_embeddings, model_name):
    if "{model_name}" in input_embeddings:
        return input_embeddings.format(model_name=model_name)
    return input_embeddings


def save_outputs(
    model,
    output_dir,
    model_name,
    suffix="",
    save_doc_info=False,
):
    model_path = os.path.join(output_dir, f"{model_name}_model{suffix}")
    model.save(model_path, serialization="pytorch", save_ctfidf=True)


def run_one_model(spec_row, args, sentences_clean, stopwords):
    model_name = spec_row["model_name"]
    min_cluster_size = int(spec_row["min_cluster_size"])
    n_components = int(spec_row["n_components"])
    n_neighbors = int(spec_row["n_neighbors"])

    embeddings_path = resolve_embeddings_path(args.input_embeddings, model_name)
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(
            f"Embeddings file not found for model '{model_name}': {embeddings_path}"
        )

    print("\n" + "=" * 72)
    print(f"Running model:      {model_name}")
    print(f"Embeddings:         {embeddings_path}")
    print(f"min_cluster_size:   {min_cluster_size}")
    print(f"n_components:       {n_components}")
    print(f"n_neighbors:        {n_neighbors}")
    print("=" * 72)

    embeddings = np.load(embeddings_path)

    model = build_model(
        min_cluster_size=min_cluster_size,
        n_components=n_components,
        n_neighbors=n_neighbors,
        stopwords=stopwords,
    )

    print("\nFitting BERTopic model...")
    model.fit_transform(sentences_clean, embeddings)

    # Save the base model BEFORE outlier reduction because update_topics mutates the model.
    print("\nSaving base model outputs before outlier reduction...")
    save_outputs(
        model=model,
        output_dir=args.output_dir,
        model_name=model_name,
        suffix="_base",
        save_doc_info=False,
    )

    print("\nApplying automatic outlier reassignment...")
    model_reduced = reduce_outliers_for_model(
        topic_model=model,
        docs=sentences_clean,
        embeddings=embeddings,
        stopwords=stopwords,
    )

    print("\nSaving reduced model outputs...")
    save_outputs(
        model=model_reduced,

        output_dir=args.output_dir,
        model_name=model_name,
        suffix="_reduced",
        save_doc_info=True,
    )


# ========================================================
# MAIN
# ========================================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\nLoading model specs...")
    specs = pd.read_csv(args.model_specs_file)

    required_columns = {
        "model_name",
        "min_cluster_size",
        "n_components",
        "n_neighbors",
    }
    missing_columns = required_columns - set(specs.columns)
    if missing_columns:
        raise ValueError(
            f"Model specs file is missing required columns: {sorted(missing_columns)}"
        )

    if args.model_name is not None:
        specs = specs[specs["model_name"] == args.model_name]
        if specs.empty:
            raise ValueError(
                f"Model '{args.model_name}' not found in {args.model_specs_file}."
            )

    # In case the specs file contains duplicates, keep the first row per model.
    specs = specs.drop_duplicates(subset=["model_name"], keep="first")

    print("\nLoading data...")
    df = pd.read_excel(args.input_data)
    sentences_original = df[args.text_column].astype(str).tolist()
    sentences_clean = [remove_placeholders(s) for s in sentences_original]

    print("Loading stopwords...")
    with open(args.input_stopwords, "r", encoding="utf-8") as f:
        stopwords = [line.strip() for line in f if line.strip()]

    for _, spec_row in specs.iterrows():
        run_one_model(
            spec_row=spec_row,
            args=args,
            sentences_clean=sentences_clean,
            stopwords=stopwords,
        )

    print("\n=== Done ===")
    print(f"Outputs saved under: {args.output_dir}")


if __name__ == "__main__":
    main()
