#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)


def main():
    # Define model configurations
    model_paths = {
        'mpnet': 'new_analysis/results/models/mpnet_reduced_top_coherence',
        'robbert': 'new_analysis/results/models/robbert_reduced_top_coherence',
        'qwen3': 'new_analysis/results/models/qwen3_reduced_top_coherence'
    }
    
    # Ensure output dir exists
    output_dir = "./new_analysis/results/tables"
    os.makedirs(output_dir, exist_ok=True)

    combined = []

    for model_name, model_path in model_paths.items():
        print(f"\n=== Processing {model_name} model ===")

        try:
            topic_info_path = os.path.join(model_path, "topic_info_reduced_outliers.csv")
            topic_info = pd.read_csv(topic_info_path)
            topic_info_sorted = topic_info.sort_values(by="Count", ascending=False)


            # Drop "Representative_Docs" if it exists
            if "Representative_Docs" in topic_info_sorted.columns:
                topic_info_sorted = topic_info_sorted.drop(columns=["Representative_Docs"])
            if "Topic" in topic_info_sorted.columns:
                topic_info_sorted = topic_info_sorted.drop(columns=["Topic"])

            # Take top 10
            top_10_topics = topic_info_sorted.head(10).copy().reset_index(drop=True)

            # --- Extract first 5 words from Representation ---
            if "Representation" in top_10_topics.columns:
                # Representation might be stored as a string like "['word1', 'word2', ...]"
                top_10_topics["Top 5 words"] = (
                    top_10_topics["Representation"]
                    .astype(str)  # ensure string
                    .apply(lambda x: [w.strip(" []'") for w in x.split(",")][:5])  # split & clean
                    .apply(lambda words: ", ".join(words))
                )
                # Drop original Representation
                top_10_topics = top_10_topics.drop(columns=["Representation"])

            # Add model name suffix to columns
            top_10_topics = top_10_topics.add_suffix(f"_{model_name}")

            combined.append(top_10_topics)

        except Exception as e:
            print(f"Error processing model {model_name}: {str(e)}")
            continue

    # Combine side by side (parallel)
    if combined:
        combined_df = pd.concat(combined, axis=1)
        combined_path = os.path.join(output_dir, "top10_topics_coherence.csv")
        combined_path_xl = os.path.join(output_dir, "top10_topics_coherence.xlsx")
        combined_df.to_csv(combined_path, index=False)
        combined_df.to_excel(combined_path_xl, index=False)
        print(f"\n✅ Parallel top 10 topics saved to {combined_path}")


if __name__ == "__main__":
    main()
