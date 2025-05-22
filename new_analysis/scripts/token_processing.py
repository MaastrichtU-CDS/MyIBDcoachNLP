# download nltk and spacy library
import nltk
from nltk.tokenize import word_tokenize
import string
import pickle
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm
import pickle
from gensim.corpora.dictionary import Dictionary
import simplemma
from typing import List
import pandas as pd
import spacy

# Download NLTK's tokenizer (only needed once)
nltk.download("punkt_tab")

# load the stemmer
stemmer = SnowballStemmer("dutch")

# load the lemmatizer
nlp = spacy.load("nl_core_news_lg")
nlp.tokenizer = lambda text: Doc(nlp.vocab, words=text.split(" "))

# Example
def stem_word(stemmer, words):
    stems = [stemmer.stem(word) for word in words]
    return stems


def clean_texts(texts: list):
    texts = [text.translate(str.maketrans("", "", string.punctuation)) for text in texts]
    texts = [text.lower() for text in texts]
    return texts

def stem_and_lemma(texts: List[List[str]]):
    stemmed_texts = stemmed_texts = [stem_word(stemmer, doc) for doc in texts]
    lemmatized_texts = []
    for tokens in texts:
        doc = nlp(" ".join(tokens))  # Reconstruct sentence
        lemmas = [token.lemma_ for token in doc]
        lemmatized_texts.append(lemmas)
    
    return stemmed_texts, lemmatized_texts

if __name__ == "__main__":  # Only runs when script.py is executed directly

    # to calculate c_v, we need to import the original messages as the reference corpus
    messages_df = pd.read_excel("/workspace/persistent/mijnidbcoachnlp/data/analysis_data/translated_clean_message_data.xlsx", index_col=0)
    messages = messages_df["clean_message"].to_list()
    messages = clean_texts(messages)

    # read the sentence data 
    df = pd.read_excel("/workspace/persistent/mijnidbcoachnlp/data/analysis_data/sentence_data_for_analysis.xlsx", index_col=0)
    sentences = df["sentence"].to_list()
    sentences = clean_texts(sentences)

    #tokenize_messages
    mes_tokens = [word_tokenize(message, language="dutch") for message in messages]
    stem_mes_tokens, lem_mes_tokens = stem_and_lemma(texts=mes_tokens)

    # tokenize sentences
    sen_tokens = [word_tokenize(sentence, language="dutch") for sentence in sentences]
    stem_sen_tokens, lem_sen_tokens = stem_and_lemma(texts=sen_tokens)

    # convert to dictionaries
    dict_mes = Dictionary(mes_tokens)
    dict_mes_stem = Dictionary(stem_mes_tokens)
    dict_mes_lem = Dictionary(lem_mes_tokens)

    # convert sentence tokens to dictionaries
    dict_sen = Dictionary(sen_tokens)
    dict_sen_stem = Dictionary(stem_sen_tokens)
    dict_sen_lem = Dictionary(lem_sen_tokens)

    import os
    import pickle

    # Define the save path
    save_path = "/workspace/persistent/mijnidbcoachnlp/data/tokens"
    os.makedirs(save_path, exist_ok=True)  # Ensure the folder exists

    # Dictionary of objects to save
    objects_to_save = {
        "stem_mes_tokens.pkl": stem_mes_tokens,
        "lem_mes_tokens.pkl": lem_mes_tokens,
        "stem_sen_tokens.pkl": stem_sen_tokens,
        "lem_sen_tokens.pkl": lem_sen_tokens,
        "tokenized_messages.pkl": mes_tokens,
        "tokenized_sentences.pkl": sen_tokens,
        "dict_mes.pkl": dict_mes,
        "dict_mes_stem.pkl": dict_mes_stem,
        "dict_mes_lem.pkl": dict_mes_lem,
        "dict_sen.pkl": dict_sen,
        "dict_sen_stem.pkl": dict_sen_stem,
        "dict_sen_lem.pkl": dict_sen_lem,
    }

    # Save each object to file
    for filename, obj in objects_to_save.items():
        with open(os.path.join(save_path, filename), "wb") as f:
            pickle.dump(obj, f)

    print("All tokens and dictionaries saved successfully.")

