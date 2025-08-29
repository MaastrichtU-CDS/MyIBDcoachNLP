#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import pandas as pd
from bertopic import BERTopic
import pickle
import random
from sklearn.feature_extraction.text import CountVectorizer



# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def reduce_outlier_for_model(topic_model, docs, embeddings, stopwords):
    """Reduce outliers for a BERTopic model"""
    # Get current topics (not from model.topics_ but from transform)
    topics = topic_model.topics_
    
    print(f"Original outliers: {sum(1 for t in topics if t == -1)}")
    
    # Reduce outliers using embeddings strategy
    new_topics = topic_model.reduce_outliers(
        documents=docs, 
        topics=topics, 
        strategy="embeddings",
        embeddings=embeddings  # Use precomputed embeddings if available
    )

    vectorizer_model = CountVectorizer(
        stop_words=stopwords,
        min_df=2,
        ngram_range=(1, 1),
        token_pattern=r'\b[a-zA-Z]{3,}\b'
    )

    print(f"Remaining outliers after reduction: {sum(1 for t in new_topics if t == -1)}")
    
    # Update the model with new topics
    topic_model.update_topics(docs, topics=new_topics, vectorizer_model = vectorizer_model)
    
    return topic_model

def main():
    # Define model configurations - FIXED TYPO IN QWEN3 PATH
    model_paths = {
        'mpnet': 'new_analysis/results/models/mpnet_base_top_coherence',
        'robbert': 'new_analysis/results/models/robbert_base_top_coherence', 
        'qwen3': 'new_analysis/results/models/qwen3_base_top_coherence'  # Fixed: removed extra 'l' and leading dot
    }
    
    # Load data once
    data = pd.read_excel("./data/sentence_data_for_analysis.xlsx", index_col=0)
    sentences = data["sentence"].to_list()
    # Load stopwords
    with open('./data/stopwords-nl-extended.txt', 'r') as file:
        dutch_stopwords = [line.strip() for line in file.readlines()]


    for model_name in model_paths.keys():
        print(f"\n=== Processing {model_name} model ===")
        
        # Create output dir
        output_dir = f"./new_analysis/results/models/{model_name}_reduced_top_coherence"
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Load model with appropriate embedding model
            model_path = model_paths[model_name]
            
            if model_name == "mpnet":
                embedding_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            elif model_name == "robbert":
                embedding_model = "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers" 
            else:  # qwen3
                embedding_model = "Qwen/qwen3-embedding-8b"
            
            print(f"Loading model from {model_path}")
            topic_model = BERTopic.load(model_path, embedding_model=None)
            print("Model loaded successfully")

            
            # Load precomputed embeddings if available
            embeddings_path = f"./data/embeddings_{model_name}.npy"
            embeddings = None
            if os.path.exists(embeddings_path):
                print(f"Loading precomputed embeddings from {embeddings_path}")
                embeddings = np.load(embeddings_path)
            else:
                print("No precomputed embeddings found, please check embedding path")

            # Reduce outliers
            print("Reducing outliers...")
            topic_model_reduced = reduce_outlier_for_model(
                topic_model=topic_model, 
                docs=sentences,
                embeddings=embeddings,
                stopwords=dutch_stopwords
            )
            
            # Get updated information
            new_topic_info = topic_model_reduced.get_topic_info()
            new_document_info = topic_model_reduced.get_document_info(sentences)  # Pass docs to get_document_info
            
            # Save model after reducing outliers
            print("Saving reduced model...")
            topic_info_path = os.path.join(output_dir, "topic_info_reduced_outliers.csv")
            doc_info_path = os.path.join(output_dir, "doc_info_reduced_outliers.csv")
            
            # Save the model
            topic_model_reduced.save(
                output_dir, 
                serialization="pytorch", 
                save_ctfidf=True, 
                save_embedding_model=embedding_model
            )
            
            # Save topic and document info
            new_topic_info.to_csv(topic_info_path, index=False)
            new_document_info.to_csv(doc_info_path, index=False)

            print(f"Model {model_name} after outlier reduction saved successfully")
            print(f"Files saved in: {output_dir}")
            
        except Exception as e:
            print(f"Error processing model {model_name}: {str(e)}")
            continue

if __name__ == "__main__":
    main()