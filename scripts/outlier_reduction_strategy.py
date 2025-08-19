#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import pandas as pd
from bertopic import BERTopic
import pickle
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def load_model_and_analyze(model_name, model_path):
    """Load a BERTopic model and perform outlier analysis"""
    print(f"\n=== Analyzing model: {model_name} ===")
    
    # Load the BERTopic model
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    
    model = BERTopic.load(model_path)
    
    # Load base data
    data = pd.read_excel("./data/sentence_data_for_analysis.xlsx", index_col=0)
    sentences = data["sentence"].to_list()
    
    # Load embeddings for this model
    embeddings_path = f"./data/embeddings_{model_name}.npy"
    if not os.path.exists(embeddings_path):
        print(f"Embeddings file not found: {embeddings_path}")
        return None
    
    embeddings = np.load(embeddings_path)
    
    # Get topic assignments for all documents
    topics, _ = model.transform(sentences, embeddings=embeddings)
    
    # Find outlier documents (topic = -1)
    outlier_indices = [i for i, topic in enumerate(topics) if topic == -1]
    print(f"Total outlier documents: {len(outlier_indices)}")
    
    if len(outlier_indices) == 0:
        print("No outlier documents found!")
        return None
    
    # Sample 10 outlier documents (or all if less than 10)
    sample_size = min(10, len(outlier_indices))
    sampled_outlier_indices = random.sample(outlier_indices, sample_size)
    
    print(f"Sampled {sample_size} outlier documents for analysis")
    
    # Get the sampled sentences and their embeddings
    sampled_sentences = [sentences[i] for i in sampled_outlier_indices]
    
    # Dictionary to store results for each strategy
    strategy_results = {}
    
    # Test different outlier reduction strategies
    strategies = ["distributions", "c-tf-idf", "embeddings"]
    
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        
        try:
            # Create a copy of the model for this strategy
            # Load fresh copy with embedding model specified
            if model_name == "mpnet":
                embedding_model = "sentence-transformers/all-mpnet-base-v2"
            elif model_name == "robbert":
                embedding_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" 
            else:
                embedding_model = "sentence-transformers/all-MiniLM-L6-v2"  # default
                
            model_copy = BERTopic.load(model_path, embedding_model=embedding_model)
            
            # Perform outlier reduction
            new_topics = model_copy.reduce_outliers(
                documents=sentences,
                topics=topics,
                embeddings=embeddings,
                strategy=strategy
            )
            
            # Get new topic assignments for our sampled outlier documents
            new_assignments = [new_topics[i] for i in sampled_outlier_indices]
            
            
            # Get topic representations for the new assignments
            topic_representations = {}
            topic_info = pd.read_csv("new_analysis/results/mpnet_base/topic_info.csv", index_col=0)
            for topic_id in set(new_assignments):
                if topic_id != -1:  # Skip outliers
                    try:
                        # Look for the topic in the Topic column (safer than using index)
                        if 'Topic' in topic_info.columns:
                            topic_row = topic_info[topic_info['Topic'] == topic_id]
                        else:
                            # Fallback to index-based lookup
                            topic_row = topic_info[topic_info.index == topic_id]
                        
                        if not topic_row.empty and 'Representation' in topic_info.columns:
                            topic_words = topic_row['Representation'].iloc[0]
                            topic_representations[topic_id] = str(topic_words)
                        else:
                            topic_representations[topic_id] = "No representation found"
                    except Exception as e:
                        topic_representations[topic_id] = f"Error: {str(e)}"
                else:
                    topic_representations[topic_id] = "Outlier"
            
            # Store results
            strategy_results[strategy] = {
                'new_topics': new_assignments,
                'topic_representations': topic_representations,
                'sentences': sampled_sentences,
                'original_indices': sampled_outlier_indices
            }
            
            # Print some statistics
            reassigned_count = sum(1 for topic in new_assignments if topic != -1)
            print(f"  Documents reassigned: {reassigned_count}/{sample_size}")
            
        except Exception as e:
            print(f"  Error with strategy {strategy}: {str(e)}")
            strategy_results[strategy] = None
    
    return strategy_results, sampled_outlier_indices, sampled_sentences

def create_comparison_dataframe(all_results):
    """Create a comprehensive comparison dataframe"""
    comparison_data = []
    
    for model_name, results in all_results.items():
        if results is None:
            continue
            
        strategy_results, indices, sentences = results
        
        for i, (idx, sentence) in enumerate(zip(indices, sentences)):
            row = {
                'model': model_name,
                'document_index': idx,
                'sentence': sentence[:100] + "..." if len(sentence) > 100 else sentence,  # Truncate for readability
                'original_topic': -1  # All were outliers originally
            }
            
            # Add results for each strategy
            for strategy in ["distributions", "c-tf-idf", "embeddings"]:
                if strategy in strategy_results and strategy_results[strategy] is not None:
                    topic_id = strategy_results[strategy]['new_topics'][i]
                    row[f'topic_after_{strategy}'] = topic_id
                    
                    # Get topic representation using topic_id as key, not index i
                    if topic_id in strategy_results[strategy]['topic_representations']:
                        row[f'topic_words_after_{strategy}'] = strategy_results[strategy]['topic_representations'][topic_id]
                    else:
                        row[f'topic_words_after_{strategy}'] = "No representation available"
                else:
                    row[f'topic_after_{strategy}'] = 'Error'
                    row[f'topic_words_after_{strategy}'] = 'Error'
            
            comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def main():
    # Define model configurations
    model_configs = {
        'mpnet': 'new_analysis/results/mpnet_base',
        #'robbert': './models/bertopic_robbert.pkl', 
        #'qwen3': './models/bertopic_qwen3.pkl'
    }
    
    # Store results for all models
    all_results = {}
    
    # Process each model
    for model_name, model_path in model_configs.items():
        try:
            result = load_model_and_analyze(model_name, model_path)
            all_results[model_name] = result
        except Exception as e:
            print(f"Error processing model {model_name}: {str(e)}")
            all_results[model_name] = None
    
    # Create comparison dataframe
    print("\n=== Creating comparison dataframe ===")
    comparison_df = create_comparison_dataframe(all_results)
    
    if not comparison_df.empty:
        # Display summary statistics
        print(f"\nTotal documents analyzed: {len(comparison_df)}")
        print(f"Models processed: {comparison_df['model'].nunique()}")
        
        # Save results
        output_path = "./new_analysis/results/outlier_reduction/outlier_reduction_comparison.xlsx"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        comparison_df.to_excel(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        # Display sample of results
        print("\nSample of comparison results:")
        # Show fewer columns for better readability in console
        display_cols = ['model', 'document_index', 'sentence', 'topic_after_distributions', 'topic_words_distributions']
        if all(col in comparison_df.columns for col in display_cols):
            print(comparison_df[display_cols].head(5).to_string(max_colwidth=40))
        else:
            print(comparison_df.head(5).to_string(max_colwidth=30))
        
        # Strategy effectiveness summary
        print("\n=== Strategy Effectiveness Summary ===")
        for strategy in ["distributions", "c-tf-idf", "embeddings"]:
            col_name = f'topic_after_{strategy}'
            if col_name in comparison_df.columns:
                reassigned = comparison_df[col_name] != -1
                success_rate = reassigned.sum() / len(comparison_df) * 100
                print(f"{strategy:12}: {success_rate:.1f}% documents reassigned")
        
        return comparison_df
    else:
        print("No results to compare - check model paths and data files")
        return None

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("./results", exist_ok=True)
    
    # Run the analysis
    results_df = main()
    
    print("\n=== Analysis Complete ===")