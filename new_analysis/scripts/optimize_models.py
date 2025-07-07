from collections import defaultdict

def run_grid_search_for_model(name, embedding_model, embeddings, docs, param_combinations, tokenized_texts, dictionary):
    from umap import UMAP
    from hdbscan import HDBSCAN
    from bertopic import BERTopic

    metrics_vs_combination = defaultdict(dict)
    topic_similarities_combination = defaultdict(dict)

    for combination in param_combinations:
        min_cluster_size, n_components, n_neighbors = combination

        topic_model = BERTopic(**bertopic_settings)
        topic_model.hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=False
        )
        topic_model.umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )

        print(f"Tuning model: {name} at param combination: min_cluster_size = {min_cluster_size}, n_components = {n_components}, n_neighbors = {n_neighbors}")

        topics, probs = topic_model.fit_transform(docs, embeddings)

        coverage = get_coverage(topic_model)
        nr_topics = get_nr_topics(topic_model)
        print(f"Document: {coverage}")
        print(f"Number of topics: {nr_topics}")

        try:
            top_words = get_top_words(topic_model, top_n=10)
            diversity_score = get_topic_diversity(top_words)
            print(f"Diversity score: {diversity_score}")

            c_v_score = get_c_v(topic_model, tokenized_texts, dictionary, top_words)
            print(f"C_V coherence: {c_v_score}")
        except Exception:
            diversity_score = None
            c_v_score = None
            print("Exception Error")

        weighted_sim, unweighted_sim, topic_sim_dict = get_intra_topic_similarity(
            topic_model, docs, embeddings
        )
        print(f"Average embedding cosine similarity weighted: {weighted_sim}, unweighted: {unweighted_sim}")
        print(" ")

        key = (name, combination)
        metrics_vs_combination[key]["min_cluster_size"] = min_cluster_size
        metrics_vs_combination[key]["n_components"] = n_components
        metrics_vs_combination[key]["n_neighbors"] = n_neighbors
        metrics_vs_combination[key]["diversity_score"] = diversity_score
        metrics_vs_combination[key]["c_v_score"] = c_v_score
        metrics_vs_combination[key]["weighted_avg_sim"] = weighted_sim
        metrics_vs_combination[key]["unweighted_avg_sim"] = unweighted_sim
        metrics_vs_combination[key]["document_coverage"] = coverage
        metrics_vs_combination[key]["number_of_topics"] = nr_topics
        topic_similarities_combination[key]["per_topic_similarities"] = topic_sim_dict

    return metrics_vs_combination, topic_similarities_combination

from concurrent.futures import ProcessPoolExecutor, as_completed

def parallel_coarse_grid_search(docs, models_and_embeddings, param_combinations, tokenized_texts, dictionary):
    all_metrics = {}
    all_similarities = {}

    with ProcessPoolExecutor(max_workers=len(models_and_embeddings)) as executor:
        futures = [
            executor.submit(run_grid_search_for_model, name, model, embeddings, docs, param_combinations, tokenized_texts, dictionary)
            for name, model, embeddings in models_and_embeddings
        ]

        for future in as_completed(futures):
            model_metrics, model_similarities = future.result()
            all_metrics.update(model_metrics)
            all_similarities.update(model_similarities)

    return all_metrics, all_similarities

coarse_metrics_results, coarse_similarities_results = parallel_coarse_grid_search(
    docs=sentences,
    models_and_embeddings=models_and_embeddings,
    param_combinations=coarse_param_combinations,
    tokenized_texts=tokenized_texts,
    dictionary=dictionary
)

# first install numpy=1.26.4 before installing pandas and other packages
# %pip install numpy=1.26.4
# %pip install pandas openpyxl

import pandas as pd
# read the sentence data 
df = pd.read_excel("/workspace/persistent/mijnidbcoachnlp/data/analysis_data/sentence_data_for_analysis.xlsx", index_col=0)
sentences = df["sentence"].to_list()

### Importing the list of Dutch stopwords (note that there are customized dutch words in there)

with open('/workspace/persistent/mijnidbcoachnlp/data/analysis_data/stopwords_extended.txt', 'r') as file:
    lines = [line.strip() for line in file.readlines()]

dutch_stopwords = lines

# %pip install gensim

import pickle
from gensim.corpora import Dictionary

# Load the tokenized texts
with open("/workspace/persistent/mijnidbcoachnlp/data/tokens/tokenized_sentences.pkl", "rb") as f:
    tokenized_texts = pickle.load(f)

dictionary = Dictionary(tokenized_texts)

import numpy as np
from sentence_transformers import SentenceTransformer

# Define model names and embeddings
models_and_embeddings = [
    ("stv1", SentenceTransformer("distiluse-base-multilingual-cased-v1"), np.load("/workspace/persistent/mijnidbcoachnlp/data/embeddings_full_docs/embeddings_st_v1_sentence_placeholder.npy")),
    ("stv2", SentenceTransformer("distiluse-base-multilingual-cased-v2"), np.load("/workspace/persistent/mijnidbcoachnlp/data/embeddings_full_docs/embeddings_st_v2_sentence_placeholder.npy")),
    ("mini", SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2"), np.load("/workspace/persistent/mijnidbcoachnlp/data/embeddings_full_docs/embeddings_st_miniL12v2_sentence_placeholder.npy")),
    ("mpnet", SentenceTransformer("paraphrase-multilingual-mpnet-base-v2"), np.load("/workspace/persistent/mijnidbcoachnlp/data/embeddings_full_docs/embeddings_st_mpnet_v2_sentence_placeholder.npy")),
    ("robbert", SentenceTransformer("NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers"), np.load("/workspace/persistent/mijnidbcoachnlp/data/embeddings_full_docs/embeddings_st_robbert2022_sentence_placeholder.npy")),
    ("e5", SentenceTransformer("intfloat/multilingual-e5-large-instruct", trust_remote_code=True), np.load("/workspace/persistent/mijnidbcoachnlp/data/embeddings_full_docs/embeddings_e5.npy"))
]

# disable parallelism to avoid some warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %pip install umap-learn hdbscan 

from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# Shared settings with multiple representation models
bertopic_settings = {
    "vectorizer_model": CountVectorizer(stop_words=dutch_stopwords, min_df=2, ngram_range=(1, 1), token_pattern=r'\b[a-zA-Z]{3,}\b'),
    "calculate_probabilities": False,
    "verbose": False
    #"representation_model": {
        #"Default": "default",  # This uses the default c-TF-IDF representation
        #"KeyBERTInspired": KeyBERTInspired()
    #}
}

# function to return top n words 
# function to get top words
from typing import List

def get_top_words(topic_model, top_n: int) -> List[List[str]]:
    """Extract top words for each topic from BERTopic (excluding outliers and empty words)."""
    topics = topic_model.get_topics()  # topics is a dict: {topic_num: [(word, score), ...]}
    top_words = []
    for topic_num, word_score_list in topics.items():
        if topic_num == -1:
            continue  # Skip outlier topic (-1)
        words = [word for word, _ in word_score_list[:top_n] if word.strip()]  # Skip empty words
        if words:  # Only append if the words list is not empty
            top_words.append(words)
    return top_words


# %pip install octis

# functions for diversity, number of topics and coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from collections import defaultdict
from typing import List

# function to evaluate topic diversity
def get_topic_diversity(top_words, topk=10):
    metric = TopicDiversity(topk=topk)
    diversity_score = metric.score({"topics": top_words}) 
    return diversity_score

# function to calculate coverage
def get_coverage(topic_model):
    """Calculate the proportion of documents assigned to valid topics (topic != -1)."""
    topics = topic_model.topics_
    valid_topic_count = sum(1 for topic in topics if topic != -1)
    return valid_topic_count / len(topics)

def get_nr_topics(topic_model):
    topic_info = topic_model.get_topic_info()
    # Exclude the outlier class (-1) if you want only real topics
    nr_topics = topic_info[topic_info.Topic != -1].shape[0]
    return nr_topics

from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

def get_c_v(topic_model, tokenized_texts, dictionary, top_words):
    
    # Prepare CoherenceModel
    coherence_model = CoherenceModel(
        topics=top_words,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    
    # Compute coherence
    coherence_score = coherence_model.get_coherence()
    
    return coherence_score


# functions to calculate embedding coherence
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def cluster_coherence(embeddings: np.ndarray) -> float:
    """
    Compute the average pairwise cosine similarity among embeddings.
    If only one item in cluster, returns NaN.
    """
    n = embeddings.shape[0]
    if n < 2:
        return np.nan
    sim_mat = cosine_similarity(embeddings)
    iu = np.triu_indices(n, k=1)
    return sim_mat[iu].mean()

def get_intra_topic_similarity(
    topic_model, 
    sentences: List[str], 
    embeddings: np.ndarray
) -> Tuple[float, float, Dict[int, float]]:
    document_info = topic_model.get_document_info(sentences)
    clusters = document_info['Topic'].to_list()

    df = pd.DataFrame({
        'sentence': sentences,
        'embedding': list(embeddings),
        'cluster': clusters
    })

    per_topic_similarities = {}
    total_weighted_sim = 0.0
    total_docs = 0
    similarities = []

    for cluster_label, group in df.groupby('cluster'):
        if cluster_label == -1:
            continue  # Skip outlier topic

        embs = np.vstack(group['embedding'].values)
        mean_sim = cluster_coherence(embs)
        cluster_size = len(group)

        if not np.isnan(mean_sim):
            per_topic_similarities[cluster_label] = mean_sim
            similarities.append(mean_sim)

            total_weighted_sim += mean_sim * cluster_size
            total_docs += cluster_size

    # Compute weighted and unweighted averages
    weighted_mean_sim = total_weighted_sim / total_docs if total_docs > 0 else np.nan
    unweighted_mean_sim = np.mean(similarities) if similarities else np.nan

    return weighted_mean_sim, unweighted_mean_sim, per_topic_similarities


