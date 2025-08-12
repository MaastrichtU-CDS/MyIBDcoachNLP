#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import pandas as pd
from itertools import product
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
from umap import UMAP
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from topic_diversity import TopicDiversity
import pickle

# ======================
# === Input arguments ===
# ======================
if len(sys.argv) != 3:
    print("Usage: python run_grid_chunk.py <model_name> <chunk_id>")
    sys.exit(1)

model_name = sys.argv[1]  # e.g., "qwen3", "mpnet", "robbert"
chunk_id = int(sys.argv[2])

chunk_path = f"./chunks/chunk_{chunk_id}.json"
checkpoint_path = f"./checkpoints/{model_name}_results_rerun_chunk_{chunk_id}.csv"

# ======================
# === Load parameters ===
# ======================
with open(chunk_path, "r") as f:
    chunk_combinations = json.load(f)

# Load base data (same for all models)
data = pd.read_excel("./data/sentence_data_for_analysis.xlsx", index_col=0)
sentences = data["sentence"].to_list()
with open("/home/jzhang/mijnidbcoachnlp/data/tokens/tokenized_sentences.pkl", "rb") as f:
    tokenized_texts = pickle.load(f)
dictionary = Dictionary(tokenized_texts)

# Load embeddings for this model
embeddings_path = f"./data/embeddings_{model_name}.npy"
if not os.path.exists(embeddings_path):
    raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
embeddings = np.load(embeddings_path)

# Load stopwords
with open('./data/stopwords-nl-extended.txt', 'r') as file: # an extended version of the stopwords was used
    dutch_stopwords = [line.strip() for line in file.readlines()]

bertopic_settings = {
    "vectorizer_model": CountVectorizer(
        stop_words=dutch_stopwords,
        min_df=2,
        ngram_range=(1, 1),
        token_pattern=r'\b[a-zA-Z]{3,}\b'
    ),
    "calculate_probabilities": False,
    "verbose": False
}

# ======================
# === Metric helpers ===
# ======================
def get_top_words(topic_model, top_n):
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

def get_c_v(top_words):
    cm = CoherenceModel(
        topics=top_words,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    return cm.get_coherence()

# ======================
# === Checkpointing ===
# ======================
os.makedirs("checkpoints", exist_ok=True)
os.makedirs(f"./new_analysis/results/slurm_outputs/{model_name}", exist_ok=True)

if os.path.exists(checkpoint_path):
    existing_df = pd.read_csv(checkpoint_path)
    done_combinations = set(
        tuple(row) for row in existing_df[["min_cluster_size", "n_components", "n_neighbors"]].values
    )
else:
    existing_df = pd.DataFrame()
    done_combinations = set()

# ======================
# === Run grid search ===
# ======================
results = []

for comb in chunk_combinations:
    min_cluster_size, n_components, n_neighbors = comb
    if tuple(comb) in done_combinations:
        print(f"Skipping completed: {comb}")
        continue

    print(f"[{model_name}] Running: mcs={min_cluster_size}, nc={n_components}, nn={n_neighbors}")

    model = BERTopic(**bertopic_settings)
    model.hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=False
    )
    model.umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )

    topics, _ = model.fit_transform(sentences, embeddings)

    top_words = get_top_words(model, top_n=10)
    coverage = get_coverage(model)
    nr_topics = get_nr_topics(model)
    diversity_score = get_topic_diversity(top_words)
    c_v_score = get_c_v(top_words)

    result = {
        "min_cluster_size": min_cluster_size,
        "n_components": n_components,
        "n_neighbors": n_neighbors,
        "diversity_score": diversity_score,
        "c_v_score": c_v_score,
        "document_coverage": coverage,
        "number_of_topics": nr_topics
    }

    results.append(result)

    # Save model and topic info
    folder = f"./new_analysis/results/slurm_outputs/{model_name}/mcs{min_cluster_size}_nc{n_components}_nn{n_neighbors}"
    os.makedirs(folder, exist_ok=True)

    if (nr_topics > 100) and (nr_topics < 300) and (coverage > 0.6):
        model.save(folder, serialization="pytorch", save_ctfidf=True)

    # Append to checkpoint
    new_df = pd.DataFrame([result])
    existing_df = pd.concat([existing_df, new_df], ignore_index=True)
    existing_df.to_csv(checkpoint_path, index=False)

print(f"[{model_name}] Finished chunk {chunk_id}. Results saved to {checkpoint_path}")
