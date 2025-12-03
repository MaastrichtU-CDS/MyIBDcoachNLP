#!/usr/bin/env python3
import os
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
from topic_diversity import TopicDiversity  # local file


# ============================================================
# Argument parser
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Run BERTopic grid search chunk.")

    parser.add_argument("--model-name", type=str, required=True,
                        help="Model name (e.g. mpnet, robbert, qwen3).")

    parser.add_argument("--chunk-id", type=int, required=True,
                        help="Chunk ID (integer).")

    parser.add_argument("--chunk-dir", type=str, default="chunks/new_chunks",
                    help="Directory containing chunk files")

    parser.add_argument("--input-data", type=str,
                        help="Excel file containing sentences.",
                        default="data/cleaned_patient_deduplicated_sentences_for_embedding.xlsx")

    parser.add_argument("--text-column", type=str, default="sentence",
                        help="Column name containing text data.")

    parser.add_argument("--input-embeddings", type=str, required=True,
                        help="NumPy .npy file containing text embeddings.")

    parser.add_argument("--input-tokens", type=str,
                        help="Pickle file containing tokenized texts.",
                        default="data/tokens/tokenized_sentences.pkl")

    parser.add_argument("--input-stopwords", type=str,
                        help="Text file containing Dutch stopwords, one per line.",
                        default="data/stopwords-nl-extended.txt")

    parser.add_argument("--output-dir", type=str,
                        help="Base output directory for checkpoints.",
                        default="results/checkpoints/")

    parser.add_argument("--run-id", type=str, required=True)

    return parser.parse_args()

# ============================================================
# Metric helpers
# ============================================================
def get_top_words(topic_model, top_n=10):
    topics = topic_model.get_topics()
    top_words = []
    for topic_num, word_score_list in topics.items():
        if topic_num == -1:
            continue
        words = [word for word, _ in word_score_list[:top_n] if word.strip()]
        if words:
            top_words.append(words)
    return top_words


def get_coverage(topic_model):
    topics = topic_model.topics_
    valid_topic_count = sum(1 for topic in topics if topic != -1)
    return valid_topic_count / len(topics)


def get_nr_topics(topic_model):
    topic_info = topic_model.get_topic_info()
    return topic_info[topic_info.Topic != -1].shape[0]


def get_topic_diversity(top_words, topk=10):
    return TopicDiversity(topk=topk).score({"topics": top_words})


def get_c_v(top_words, tokenized_texts, dictionary):
    cm = CoherenceModel(
        topics=top_words,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence="c_v"
    )
    return cm.get_coherence()


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    model_name = args.model_name
    chunk_id = args.chunk_id

    # --------------------------------------------------------
    # Prepare output directories
    # --------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    run_dir = os.path.join(args.output_dir, args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    checkpoint_file = os.path.join(
        run_dir,
        f"{model_name}_results_chunk_{chunk_id}.csv"
    )

    # --------------------------------------------------------
    # Load chunk parameters
    # --------------------------------------------------------
    chunk_file = os.path.join(args.chunk_dir, f"chunk_{args.chunk_id}.json")

    if not os.path.exists(chunk_file):
        raise FileNotFoundError(f"Chunk file not found: {chunk_file}")

    with open(chunk_file, "r") as f:
        chunk_combinations = json.load(f)

    if len(chunk_combinations) == 0:
        print(f"Warning: Chunk {chunk_id} is empty. Nothing to do.")
        return

    # --------------------------------------------------------
    # Load base data
    # --------------------------------------------------------
    df = pd.read_excel(args.input_data)
    if args.text_column not in df.columns:
        raise ValueError(f"Column '{args.text_column}' not found in input Excel file.")

    sentences = df[args.text_column].astype(str).tolist()

    with open(args.input_tokens, "rb") as f:
        tokenized_texts = pickle.load(f)

    dictionary = Dictionary(tokenized_texts)

    embeddings = np.load(args.input_embeddings)
    if len(embeddings) != len(sentences):
        raise ValueError("Embeddings and sentences do not have the same length.")

    # Load stopwords
    with open(args.input_stopwords, "r") as file:
        stopwords = [line.strip() for line in file.readlines()]

    # BERTopic default settings
    bertopic_settings = {
        "vectorizer_model": CountVectorizer(
            stop_words=stopwords,
            min_df=2,
            ngram_range=(1, 1),
            token_pattern=r'\b[a-zA-Z]{3,}\b'
        ),
        "calculate_probabilities": False,
        "verbose": False
    }

    # --------------------------------------------------------
    # Checkpointing
    # --------------------------------------------------------
    if os.path.exists(checkpoint_file):
        existing = pd.read_csv(checkpoint_file)
        processed = set(
            tuple(row)
            for row in existing[["min_cluster_size", "n_components", "n_neighbors"]].values
        )
    else:
        existing = pd.DataFrame()
        processed = set()

    # --------------------------------------------------------
    # Grid search execution
    # --------------------------------------------------------
    for mcs, ncomp, nnei in chunk_combinations:

        params_tuple = (mcs, ncomp, nnei)

        if params_tuple in processed:
            print(f"Skipping completed: {params_tuple}")
            continue

        print(f"[{model_name}] Running: mcs={mcs}, nc={ncomp}, nn={nnei}")

        # Build custom BERTopic
        topic_model = BERTopic(**bertopic_settings)

        topic_model.hdbscan_model = HDBSCAN(
            min_cluster_size=mcs,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=False
        )

        topic_model.umap_model = UMAP(
            n_neighbors=nnei,
            n_components=ncomp,
            min_dist=0.0,
            metric="cosine",
            random_state=42
        )

        topics, _ = topic_model.fit_transform(sentences, embeddings)

        top_words = get_top_words(topic_model, top_n=10)
        coverage = get_coverage(topic_model)
        nr_topics = get_nr_topics(topic_model)
        diversity_score = get_topic_diversity(top_words)
        c_v_score = get_c_v(top_words, tokenized_texts, dictionary)

        result = {
            "min_cluster_size": mcs,
            "n_components": ncomp,
            "n_neighbors": nnei,
            "diversity_score": diversity_score,
            "c_v_score": c_v_score,
            "doc_assignment_pct": coverage,
            "number_of_topics": nr_topics
        }

        # Append to checkpoint
        new_row = pd.DataFrame([result])
        existing = pd.concat([existing, new_row], ignore_index=True)
        existing.to_csv(checkpoint_file, index=False)

    print(f"[{model_name}] Finished chunk {chunk_id}. Results saved to {checkpoint_file}")


if __name__ == "__main__":
    main()
