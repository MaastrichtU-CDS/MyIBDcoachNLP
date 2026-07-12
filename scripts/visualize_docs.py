#!/usr/bin/env python3

import os

import datamapplot as dmp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from umap import UMAP
from ast import literal_eval

def extract_first_n_words(value, top_n=4):
    """Return the first `top_n` unique topic words, preserving order."""
    if isinstance(value, str):
        words = literal_eval(value)
    else:
        words = list(value)

    seen = set()
    unique_words = []

    for word in words:
        word = str(word).strip()
        if word not in seen:
            seen.add(word)
            unique_words.append(word)
        if len(unique_words) == top_n:
            break

    return ", ".join(unique_words)

def label_topic(
    row,
    selected_topics,
    label_column="Plot_Label",
):
    if pd.isna(row["Label"]):
        return "Unlabelled"

    prefixes = {
        "Clinical": "C",
        "Non-Clinical": "N",
    }

    prefix = prefixes.get(str(row["Label"]).strip())
    if prefix is None:
        return "Unlabelled"

    if (
        row["Topic"] in selected_topics
        and pd.notna(row[label_column])
        and pd.notna(row["Rank"])
    ):
        return f"{prefix}{int(row['Rank'])}: {row[label_column]}"

    return "Unlabelled"


def restore_embedding_duplicates(deduplicated_sentences, full_sentences, reduced_embeddings):
    """Assign duplicate sentences the embedding of their deduplicated version."""
    lookup = dict(zip(deduplicated_sentences, reduced_embeddings))
    missing = [sentence for sentence in full_sentences if sentence not in lookup]
    if missing:
        raise KeyError(
            f"{len(missing)} full sentences were not found in the deduplicated "
            f"sentence lookup. Examples: {missing[:5]}"
        )
    return np.vstack([lookup[sentence] for sentence in full_sentences])


def reduce_embeddings(
    deduplicated_sentences, full_sentences, input_embeddings_path, output_embeddings_path
):
    """Reduce embeddings to two dimensions with UMAP, or load cached output."""
    if not os.path.exists(input_embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {input_embeddings_path}")

    if os.path.exists(output_embeddings_path):
        print(f"Loading reduced embeddings from {output_embeddings_path}")
        reduced = np.load(output_embeddings_path)
        if len(reduced) != len(full_sentences):
            raise ValueError(
                "Saved embeddings do not match document_info rows: "
                f"{len(reduced)} embeddings, {len(full_sentences)} sentences"
            )
        return reduced

    embeddings = np.load(input_embeddings_path)
    print("Reducing embeddings with UMAP")
    reduced = UMAP(
        n_neighbors=10,
        n_components=2,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    ).fit_transform(embeddings)

    if len(deduplicated_sentences) != len(reduced):
        raise ValueError(
            "Deduplicated sentence and embedding counts differ: "
            f"{len(deduplicated_sentences)} vs {len(reduced)}"
        )

    full_reduced = restore_embedding_duplicates(
        deduplicated_sentences, full_sentences, reduced
    )
    os.makedirs(os.path.dirname(output_embeddings_path), exist_ok=True)
    np.save(output_embeddings_path, full_reduced)
    print(f"Saved reduced embeddings to {output_embeddings_path}")
    return full_reduced


def prepare_separate_topic_ranks(topic_info, n_topics_per_label=10):
    """Rank Clinical and Non-Clinical topics separately by count."""
    topic_info = topic_info.copy()
    topic_info["Label"] = topic_info["Label"].astype("string").str.strip()
    ranked = topic_info[topic_info["Label"].isin(["Clinical", "Non-Clinical"])].copy()

    if ranked.empty:
        raise ValueError("No Clinical or Non-Clinical topics were found.")

    ranked["Count"] = pd.to_numeric(ranked["Count"], errors="coerce")
    if ranked["Count"].isna().any():
        bad_rows = ranked.loc[ranked["Count"].isna(), ["Topic", "Count"]]
        raise ValueError(f"Invalid topic Count values:\n{bad_rows}")

    ranked = ranked.sort_values(
        ["Label", "Count", "Topic"], ascending=[True, False, True]
    ).reset_index(drop=True)
    ranked["Rank"] = ranked.groupby("Label").cumcount() + 1
    selected = ranked[ranked["Rank"] <= n_topics_per_label].reset_index(drop=True)
    return ranked, selected


def visualize_docs(
    model_name,
    reduced_embeddings,
    ranked_topic_info,
    selected_topic_info,
    doc_info,
    output_dir,
    n_topics_per_label=10,
    label_source="interpretation",
):
    """
    Create a DataMapPlot of the top Clinical and Non-Clinical topics.

    label_source:
        "interpretation"     -> use Interpretation_Label
        "english_top_words"  -> use Translation
        "dutch_top_words"    -> use Representation
    """
    if len(reduced_embeddings) != len(doc_info):
        raise ValueError(
            "Embedding and document counts differ: "
            f"{len(reduced_embeddings)} vs {len(doc_info)}"
        )

    ranked = ranked_topic_info.copy()

    if label_source == "interpretation":
        if "Interpretation_Label" not in ranked.columns:
            raise KeyError("Missing column: Interpretation_Label")

        ranked["Plot_Label"] = ranked["Interpretation_Label"].fillna("Unlabelled")

    elif label_source == "english_top_words":
        ranked["Plot_Label"] = ranked["Translation"].apply(
            extract_first_n_words
        )

    elif label_source == "dutch_top_words":
        ranked["Plot_Label"] = ranked["Representation"].apply(
            extract_first_n_words
        )

    else:
        raise ValueError(
            "label_source must be 'interpretation', "
            "'english_top_words', or 'dutch_top_words'."
        )

    selected_topics = set(selected_topic_info["Topic"])

    labeled_docs = doc_info[["sentence", "Topic"]].merge(
        ranked[["Topic", "Rank", "Plot_Label", "Label"]],
        on="Topic",
        how="left",
        validate="many_to_one",
    )

    print(
        "Document rows without a ranked Clinical or Non-Clinical label: "
        f"{labeled_docs['Label'].isna().sum():,}"
    )

    plot_labels = labeled_docs.apply(
        label_topic,
        axis=1,
        selected_topics=selected_topics,
        label_column="Plot_Label",
    ).tolist()

    counts = selected_topic_info["Label"].value_counts()
    print(
        f"Selected {counts.get('Clinical', 0)} Clinical topics and "
        f"{counts.get('Non-Clinical', 0)} Non-Clinical topics."
    )

    fig, ax = dmp.create_plot(
        reduced_embeddings,
        plot_labels,
        marker_type="o",
        title=(
            f"Top {n_topics_per_label} Clinical and Non-Clinical Topics Identified by {model_name.upper()}"
        ),
        sub_title=(
            "A 2-dimensional data map of sentences "
            "from Dutch IBD patient messages"
        ),
        label_over_points=True,
        dynamic_label_size=True,
        max_font_size=60,
        min_font_size=6,
    )
    plot_path = os.path.join(output_dir, f"{model_name}_top_{n_topics_per_label}_{label_source}_topics_plot.png")
    print(f"Saving plot to {plot_path}")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    model_name = "robbert"
    n_topics_per_label = 15
    results_dir = f"results/{model_name}"
    embeddings_dir = f"embeddings/{model_name}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)

    with open("topics_excluded_from_visualization.txt") as f:
        topics_to_exclude = [int(line) for line in f if line.strip().isdigit()]

    print("\n=== Visualizing Documents ===")
    deduplicated_df = pd.read_excel(
        "data/cleaned_patient_deduplicated_sentences_for_embedding.xlsx"
    )
    document_info = pd.read_csv(
        f"{results_dir}/{model_name}_final_document_info.csv"
    )
    topic_info = pd.read_csv(f"{results_dir}/{model_name}_final_topic_info_labeled.csv")

    required = {
        "document_info": ({"sentence", "Topic"}, document_info),
        "topic_info": (
            {"Topic", "Count", "Representation", "Translation", "Label", "Interpretation_Label"},
            topic_info,
        ),
    }
    for name, (columns, frame) in required.items():
        missing = columns - set(frame.columns)
        if missing:
            raise KeyError(f"Missing {name} columns: {sorted(missing)}")

    for frame in (document_info, topic_info):
        frame["Topic"] = pd.to_numeric(frame["Topic"], errors="coerce").astype("Int64")

    excluded_mask = topic_info["Topic"].isin(topics_to_exclude)
    excluded_topics = topic_info[excluded_mask].copy()
    excluded_topics.to_csv(
        f"{results_dir}/{model_name}_topics_excluded_from_visualization.csv",
        index=False,
    )

    ranked_topics, selected_topics = prepare_separate_topic_ranks(
        topic_info[~excluded_mask], n_topics_per_label
    )


    print(f"Excluded {len(excluded_topics)} topics.")
    print("\nSelected topics with label-specific ranks:")
    print(
        selected_topics[["Topic", "Label", "Rank", "Count", "Translation", "Interpretation_Label"]]
        .to_string(index=False)
    )

    reduced_embeddings = reduce_embeddings(
        deduplicated_df["sentence"].tolist(),
        document_info["sentence"].tolist(),
        f"{embeddings_dir}/{model_name}_sentence_embeddings.npy",
        f"{embeddings_dir}/{model_name}_umap_reduced_embeddings.npy",
    )

    print(f"\nVisualizing documents for {model_name}")
    visualize_docs(
        model_name=model_name,
        reduced_embeddings=reduced_embeddings,
        ranked_topic_info=ranked_topics,
        selected_topic_info=selected_topics,
        doc_info=document_info,
        output_dir=results_dir,
        n_topics_per_label=n_topics_per_label,
        label_source="dutch_top_words", # choose from "interpretation", "english_top_words", or "dutch_top_words"
    )


if __name__ == "__main__":
    main()
