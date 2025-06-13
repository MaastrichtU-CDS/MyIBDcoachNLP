# Imports
import nltk
from nltk.tokenize import word_tokenize
import string
from nltk.stem.snowball import SnowballStemmer
from gensim.corpora.dictionary import Dictionary
from typing import List, Dict
import pandas as pd
import spacy
import os
import pickle

# Download tokenizer data
nltk.download("punkt_tab")

# Load NLP tools
stemmer = SnowballStemmer("dutch")
nlp = spacy.load("nl_core_news_sm")

# ----------------------------------------
# Helper functions
# ----------------------------------------

def clean_texts(texts: List[str]) -> List[str]:
    return [text.translate(str.maketrans("", "", string.punctuation)).lower() for text in texts]

def tokenize_texts(texts: List[str]) -> List[List[str]]:
    return [word_tokenize(text, language="dutch") for text in texts]

def stem_texts(tokenized: List[List[str]]) -> List[List[str]]:
    return [[stemmer.stem(word) for word in doc] for doc in tokenized]

def lemmatize_texts(tokenized: List[List[str]]) -> List[List[str]]:
    lemmatized = []
    for tokens in tokenized:
        doc = nlp(" ".join(tokens))
        lemmatized.append([token.lemma_ for token in doc])
    return lemmatized

def build_dictionaries(data: Dict[str, List[List[str]]]) -> Dict[str, Dictionary]:
    return {name: Dictionary(tokens) for name, tokens in data.items()}

def save_data(data: Dict[str, object], path: str, file_ext: str = ".pkl"):
    os.makedirs(path, exist_ok=True)
    for name, obj in data.items():
        file_path = os.path.join(path, f"{name}{file_ext}")
        if file_ext == ".pkl":
            with open(file_path, "wb") as f:
                pickle.dump(obj, f)
        elif file_ext == ".dict":
            obj.save(file_path)

# ----------------------------------------
# Main script
# ----------------------------------------

if __name__ == "__main__":
    save_path = "/workspace/persistent/mijnidbcoachnlp/data/tokens"

    # Load and clean text
    messages_df = pd.read_excel(
        "/workspace/persistent/mijnidbcoachnlp/data/analysis_data/translated_clean_message_data.xlsx",
        index_col=0
    )
    messages = clean_texts(messages_df["clean_message"].tolist())

    sentences_df = pd.read_excel(
        "/workspace/persistent/mijnidbcoachnlp/data/analysis_data/sentence_data_for_analysis.xlsx",
        index_col=0
    )
    sentences = clean_texts(sentences_df["sentence"].tolist())

    # Process each type
    results = {}
    for name, texts in {"messages": messages, "sentences": sentences}.items():
        tokenized = tokenize_texts(texts)
        stemmed = stem_texts(tokenized)
        lemmatized = lemmatize_texts(tokenized)

        # Save token lists
        results[f"tokenized_{name}"] = tokenized
        results[f"stemmed_{name}"] = stemmed
        results[f"lemmatized_{name}"] = lemmatized

    # Save tokens as .pkl
    save_data(results, save_path, file_ext=".pkl")

    # Build and save dictionaries
    dicts = build_dictionaries({
        f"{key}_dict": value for key, value in results.items()
    })
    save_data(dicts, save_path, file_ext=".dict")

    print("All tokens and dictionaries saved successfully.")
