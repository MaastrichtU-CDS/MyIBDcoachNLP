#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
from umap import UMAP
import datamapplot as dmp
import numpy as np


def extract_first_n_words(representations, top_n=4):
    representations = str(representations)
    return (
        " ".join(representations.strip("[").strip("]").split(",")[:top_n])
        .replace("'", "")
        .replace('"', "")
        .strip()
        .replace("  ", ", ")
    )


def label_topic(row, selected_topics):
    if row["Topic"] in selected_topics and not pd.isna(row["Top_Words"]):
        return f"{int(row['Rank'])}: {row['Top_Words']}"
    return "Unlabelled"

def restore_embedding_duplicates(deduplicated_sentences, full_sentences, reduced_embeddings):
    lookup = dict(zip(deduplicated_sentences, reduced_embeddings))
    full_reduced_embeddings = np.vstack([lookup[s] for s in full_sentences])
    return full_reduced_embeddings

def reduce_embeddings(
    deduplicated_sentences,
    full_sentences,
    input_embeddings_path,
    output_embeddings_path,
):
    if not os.path.exists(input_embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {input_embeddings_path}")

    embeddings = np.load(input_embeddings_path)

    if os.path.exists(output_embeddings_path):
        print(f"Loading reduced embeddings from {output_embeddings_path}")
        full_reduced_embeddings = np.load(output_embeddings_path)
        return full_reduced_embeddings

    print(
        "Reducing embeddings using UMAP with n_neighbors=10, "
        "n_components=2, min_dist=0.0, metric='cosine'"
    )

    reducer = UMAP(
        n_neighbors=10,
        n_components=2,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    reduced_embeddings = reducer.fit_transform(embeddings)
    print(f"Reduced embeddings shape: {reduced_embeddings.shape}")
    print(f"Number of deduplicated sentences: {len(deduplicated_sentences)}")
    assert len(deduplicated_sentences) == reduced_embeddings.shape[0], (
        "Mismatch between deduplicated sentences and embedding array sizes."
    )

    full_reduced_embeddings = restore_embedding_duplicates(
        deduplicated_sentences, full_sentences, reduced_embeddings
    )

    assert full_reduced_embeddings.shape[0] == len(full_sentences), (
        "Mismatch between full sentences and restored embedding array sizes."
    )
    print(f"Number of full sentences: {len(full_sentences)}")
    print(f"Full reduced embeddings shape: {full_reduced_embeddings.shape}")

    np.save(output_embeddings_path, full_reduced_embeddings)
    print(f"Reduced UMAP 2D embeddings saved to {output_embeddings_path}")

    return full_reduced_embeddings


def visualize_docs(
    model_name,
    reduced_embeddings,
    sorted_topic_info,
    doc_info,
    output_dir,
    n_topics=30,
    use_english_label=True,
):
    sorted_topic_info = sorted_topic_info.copy()

    if use_english_label:
        sorted_topic_info["Top_Words"] = sorted_topic_info["Translation"].apply(
            lambda x: extract_first_n_words(x, top_n=4)
        )
    else:
        sorted_topic_info["Top_Words"] = sorted_topic_info["Representation"].apply(
            lambda x: extract_first_n_words(x, top_n=4)
        )

    selected_topics = sorted_topic_info.head(n_topics)["Topic"].tolist()

    labeled_doc_info = doc_info[["sentence", "Topic"]].merge(
        sorted_topic_info[["Topic", "Rank", "Top_Words"]],
        on="Topic",
        how="left",
    )

    labeled_doc_info["plot_labels"] = labeled_doc_info.apply(
        lambda row: label_topic(row, selected_topics),
        axis=1,
    )

    plot_labels = labeled_doc_info["plot_labels"].tolist()

    fig = dmp.create_plot(
        reduced_embeddings,
        plot_labels,
        marker_type="o",
        title=f"Top {n_topics} Topics Identified by {model_name.upper()} Model",
        sub_title="A 2-dimensional data map of sentences from Dutch IBD patient messages",
        label_over_points=True,
        dynamic_label_size=True,
        max_font_size=60,
        min_font_size=6,
    )

    if use_english_label:
        plot_path = os.path.join(
            output_dir, f"top_{n_topics}_topics_datamap_english.png"
        )
    else:
        plot_path = os.path.join(output_dir, f"top_{n_topics}_topics_datamap.png")

    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Document visualization saved to {plot_path}")


def main():
    model_name = "robbert"
    n_topics_to_visualize = 30

    topics_to_exclude = [
        32, 78, 150, 11, 54, 40, 24, 30, 57, 29,
        80, 63, 108, 88, 76, 124, 89, 143, 149,
    ]

    print("\n=== Visualizing Documents ===")

    deduplicated_sentences_df = pd.read_excel(
        "data/cleaned_patient_deduplicated_sentences_for_embedding.xlsx"
    )

    document_info = pd.read_csv(
        f"results/{model_name}/{model_name}_final_document_info.csv"
    )

    sorted_topic_info = pd.read_csv(
        f"results/{model_name}/{model_name}_final_topic_info_labeled.csv"
    )

    topics_excluded_df = sorted_topic_info[
        sorted_topic_info["Topic"].isin(topics_to_exclude)
    ]

    topics_excluded_df.to_csv(
        f"results/{model_name}/{model_name}_topics_excluded_from_visualization.csv",
        index=False,
    )

    sorted_topic_info = sorted_topic_info[
        ~sorted_topic_info["Topic"].isin(topics_to_exclude)
    ].copy()

    sorted_topic_info = sorted_topic_info.sort_values("Rank").reset_index(drop=True)
    sorted_topic_info["Rank"] = range(1, len(sorted_topic_info) + 1)

    selected_topics = sorted_topic_info.head(n_topics_to_visualize)["Topic"].tolist()

    print(f"Excluded {len(topics_to_exclude)} topics.")
    print(f"Visualizing top {len(selected_topics)} remaining topics:")

    deduplicated_sentences = deduplicated_sentences_df["sentence"].to_list()
    full_sentences = document_info["sentence"].to_list()

    output_dir = f"./results/{model_name}/"
    os.makedirs(output_dir, exist_ok=True)

    embeddings_dir = f"./embeddings/{model_name}/"
    input_embeddings_path = os.path.join(
        embeddings_dir, f"{model_name}_sentence_embeddings.npy"
    )
    output_embeddings_path = os.path.join(
        embeddings_dir, f"{model_name}_umap_reduced_embeddings.npy"
    )

    reduced_embeddings = reduce_embeddings(
        deduplicated_sentences,
        full_sentences,
        input_embeddings_path,
        output_embeddings_path,
    )

    print(f"Visualizing documents for {model_name}")

    visualize_docs(
        model_name=model_name,
        reduced_embeddings=reduced_embeddings,
        sorted_topic_info=sorted_topic_info,
        doc_info=document_info,
        output_dir=output_dir,
        n_topics=n_topics_to_visualize,
        use_english_label=True,
    )


if __name__ == "__main__":
    main()