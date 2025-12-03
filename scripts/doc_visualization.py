#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
from bertopic import BERTopic
import datamapplot as dmp
from sklearn.decomposition import PCA


# -----------------------------------------------------------
# Argument parser
# -----------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Plot UMAP datamap for BERTopic model.")
    
    parser.add_argument(
        "--model-name",
        required=True,
        choices=["robbert", "mpnet", "qwen3"],
        help="Name of the model to load."
    )

    parser.add_argument(
        "--results-dir",
        default="results/",
        help="Base directory containing BERTopic models."
    )

    parser.add_argument(
        "--embeddings-dir",
        default="embeddings/",
        help="Directory containing UMAP embeddings."
    )

    parser.add_argument(
        "--max-topic-rank",
        type=int,
        default=30,
        help="How many top-ranked topics should be labeled."
    )

    return parser.parse_args()


# -----------------------------------------------------------
# Clean "Name" → top 4 words
# -----------------------------------------------------------
def extract_topwords(name_str):
    parts = name_str.split("_")
    if parts[0].isdigit():
        parts = parts[1:]
    return " ".join(parts)


# -----------------------------------------------------------
# Load everything
# -----------------------------------------------------------
def load_model_and_data(model_name, results_dir, embeddings_dir):
    model_path = os.path.join(results_dir, model_name, f"{model_name}_model_reduced")
    
    umap_path = os.path.join(embeddings_dir, f"{model_name}_umap_embeddings.npy")
    doc_info_path = os.path.join(results_dir, model_name, f"{model_name}_document_info_reduced.csv")
    topic_info_path = os.path.join(results_dir, model_name, f"{model_name}_topic_info_reduced.csv")

    print(f"Loading model: {model_path}")
    model = BERTopic.load(model_path)

    print(f"Loading UMAP embeddings: {umap_path}")
    umap_embeddings = np.load(umap_path)

    print(f"Reducing UMAP embeddings to 2D...")
    pca = PCA(n_components=2, random_state=42)
    umap_embeddings = pca.fit_transform(umap_embeddings)
    print(f"UMAP embeddings shape after PCA: {umap_embeddings.shape}")

    print(f"Loading document info: {doc_info_path}")
    doc_info = pd.read_csv(doc_info_path)

    print(f"Loading topic info: {topic_info_path}")
    topic_info = pd.read_csv(topic_info_path)

    return model, umap_embeddings, doc_info, topic_info


# -----------------------------------------------------------
# Main workflow
# -----------------------------------------------------------
def main():
    args = parse_args()
    model_name = args.model_name

    model, umap_embeddings, doc_info, topic_info = load_model_and_data(
        model_name, args.results_dir, args.embeddings_dir
    )

    # Rank topics by Count
    topic_info = topic_info.sort_values(by="Count", ascending=False).reset_index(drop=True)
    topic_info["Rank"] = topic_info.index + 1

    # Extract labels using Name column
    topic_info["CleanLabel"] = topic_info["Name"].apply(extract_topwords)

    topic_labels = topic_info[["Topic", "CleanLabel", "Rank"]]
    doc_labels = doc_info.merge(topic_labels, on="Topic", how="left")

    # Only label the top N topics — everything else = Unlabelled
    max_rank = args.max_topic_rank
    selected_ranks = set(range(1, max_rank + 1))

    def build_label(row):
        if row["Rank"] in selected_ranks:
            return f"{row['Rank']}: {row['CleanLabel']}"
        return "Unlabelled"

    doc_labels["plot_labels"] = doc_labels.apply(build_label, axis=1)
    plot_labels = doc_labels["plot_labels"].tolist()

    print(f"\nGenerating UMAP datamap for {model_name}...")

    # -----------------------------------------------------------
    # IMPORTANT: no point_color here (not supported in your version)
    # -----------------------------------------------------------
    fig = dmp.create_plot(
        umap_embeddings,
        plot_labels,
        title=f"Top {max_rank} Topics Identified by {model_name.upper()}",
        sub_title="2D UMAP datamap of Dutch IBD patient messages",
        label_over_points=True,
        dynamic_label_size=True,
        max_font_size=40,
        min_font_size=6,
    )

    # Save
    plot_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    save_path = os.path.join(plot_dir, f"{model_name}_{max_rank}topics_map.png")
    print(f"Saving plot → {save_path}")
    fig[0].savefig(save_path, dpi=600, bbox_inches="tight")

    print("\n✔ Done!")


if __name__ == "__main__":
    main()
