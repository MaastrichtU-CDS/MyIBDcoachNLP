#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
from umap import UMAP
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from topic_diversity import TopicDiversity
import pickle

# ======================
# === Configuration ===
# ======================
CONFIGS = {
    "mpnet": {"min_cluster_size": 25, "n_components": 7, "n_neighbors": 10},
    "robbert": {"min_cluster_size": 30, "n_components": 10, "n_neighbors": 10},
    "qwen3": {"min_cluster_size": 40, "n_components": 15, "n_neighbors": 10}
}

# ======================
# === Input arguments ===
# ======================
if len(sys.argv) != 2:
    print("Usage: python run_specific_configs.py <model_name>")
    print("Available models: mpnet, robbert, qwen3")
    sys.exit(1)

model_name = sys.argv[1]

if model_name not in CONFIGS:
    print(f"Error: Model '{model_name}' not supported.")
    print("Available models: mpnet, robbert, qwen3")
    sys.exit(1)

config = CONFIGS[model_name]
print(f"Running {model_name} with config: {config}")

# ======================
# === Load data ===
# ======================
print("Loading data...")

# Load base data
data = pd.read_excel("./data/sentence_data_for_analysis.xlsx", index_col=0)
sentences = data["sentence"].to_list()

# Load tokenized texts for coherence calculation
with open("/home/jzhang/mijnidbcoachnlp/data/tokens/tokenized_sentences.pkl", "rb") as f:
    tokenized_texts = pickle.load(f)
dictionary = Dictionary(tokenized_texts)

# Load embeddings for this model
embeddings_path = f"./data/embeddings_{model_name}.npy"
if not os.path.exists(embeddings_path):
    raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
embeddings = np.load(embeddings_path)

# Load stopwords
with open('./data/stopwords-nl-extended.txt', 'r') as file:
    dutch_stopwords = [line.strip() for line in file.readlines()]

print(f"Loaded {len(sentences)} sentences and embeddings shape: {embeddings.shape}")

# ======================
# === BERTopic settings ===
# ======================
bertopic_settings = {
    "vectorizer_model": CountVectorizer(
        stop_words=dutch_stopwords,
        min_df=2,
        ngram_range=(1, 1),
        token_pattern=r'\b[a-zA-Z]{3,}\b'
    ),
    "calculate_probabilities": False,
    "verbose": True
}

# ======================
# === Metric functions ===
# ======================
def get_top_words(topic_model, top_n=10):
    """Extract top words for each topic"""
    topics = topic_model.get_topics()
    top_words = []
    for topic_num, word_score_list in topics.items():
        if topic_num == -1:  # Skip outlier topic
            continue
        words = [word for word, _ in word_score_list[:top_n] if word.strip()]
        if words:
            top_words.append(words)
    return top_words

def get_coverage(topic_model):
    """Calculate document coverage (non-outlier documents)"""
    topics = topic_model.topics_
    valid_topic_count = sum(1 for topic in topics if topic != -1)
    return valid_topic_count / len(topics)

def get_nr_topics(topic_model):
    """Get number of topics (excluding outliers)"""
    topic_info = topic_model.get_topic_info()
    return topic_info[topic_info.Topic != -1].shape[0]

def get_topic_diversity(top_words, topk=10):
    """Calculate topic diversity score"""
    return TopicDiversity(topk=topk).score({"topics": top_words})

def get_c_v_coherence(top_words):
    """Calculate C_V coherence score"""
    cm = CoherenceModel(
        topics=top_words,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    return cm.get_coherence()

# ======================
# === Run BERTopic ===
# ======================
print(f"Setting up BERTopic model with parameters:")
print(f"  - min_cluster_size: {config['min_cluster_size']}")
print(f"  - n_components: {config['n_components']}")
print(f"  - n_neighbors: {config['n_neighbors']}")

# Initialize BERTopic model
model = BERTopic(**bertopic_settings)

# Configure HDBSCAN clustering
model.hdbscan_model = HDBSCAN(
    min_cluster_size=config['min_cluster_size'],
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=False
)

# Configure UMAP dimensionality reduction
model.umap_model = UMAP(
    n_neighbors=config['n_neighbors'],
    n_components=config['n_components'],
    min_dist=0.0,
    metric='cosine',
    random_state=42
)

print("Fitting BERTopic model...")
topics, probabilities = model.fit_transform(sentences, embeddings)

print(f"Model fitted. Found {len(set(topics))} topics (including outliers)")

# ======================
# === Calculate metrics ===
# ======================
print("Calculating metrics...")

top_words = get_top_words(model, top_n=10)
coverage = get_coverage(model)
nr_topics = get_nr_topics(model)
diversity_score = get_topic_diversity(top_words)
c_v_score = get_c_v_coherence(top_words)

# Compile results
results = {
    "model_name": model_name,
    "min_cluster_size": config['min_cluster_size'],
    "n_components": config['n_components'],
    "n_neighbors": config['n_neighbors'],
    "diversity_score": diversity_score,
    "c_v_score": c_v_score,
    "document_coverage": coverage,
    "number_of_topics": nr_topics
}

print("\n=== Results ===")
for key, value in results.items():
    print(f"{key}: {value}")

# ======================
# === Save results ===
# ======================
output_dir = f"./new_analysis/results/{model_name}_base"
os.makedirs(output_dir, exist_ok=True)

# Save metrics
metrics_file = os.path.join(output_dir, "metrics.json")
with open(metrics_file, "w") as f:
    json.dump(results, f, indent=2)

# Save metrics as CSV for easy reading
metrics_df = pd.DataFrame([results])
csv_file = os.path.join(output_dir, "metrics.csv")
metrics_df.to_csv(csv_file, index=False)

# Save the BERTopic model
print("Saving BERTopic model...")
model.save(output_dir, serialization="pytorch", save_ctfidf=True)

# Save topic information
print("Saving topic information...")
topic_info = model.get_topic_info()
topic_info.to_csv(os.path.join(output_dir, "topic_info.csv"), index=False)

# Save document information (topics assigned to each document)
document_info = model.get_document_info(docs=sentences)
document_info.to_csv(os.path.join(output_dir, "document_info.csv"), index=False)

# Save summary
summary = {
    "model_name": model_name,
    "config": config,
    "results": results,
    "total_documents": len(sentences),
    "total_topics": nr_topics,
    "outlier_documents": sum(1 for t in topics if t == -1),
    "files_saved": [
        "metrics.json", "metrics.csv", "topic_info.csv", 
        "document_info.csv", "model files"
    ]
}

with open(os.path.join(output_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n=== Completed ===")
print(f"All results saved to: {output_dir}")
print(f"Files saved:")
print(f"  - metrics.json & metrics.csv: Evaluation metrics")
print(f"  - topic_info.csv: Topic information from BERTopic")
print(f"  - document_info.csv: Document-topic assignments")
print(f"  - summary.json: Complete summary")
print(f"  - BERTopic model files: For reloading the model")