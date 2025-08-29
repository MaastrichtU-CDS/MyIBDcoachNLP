#!/usr/bin/env python3
import os
import pandas as pd

def main():
    # Define model configurations
    model_paths = {
        'mpnet': 'new_analysis/results/models/mpnet_reduced_top_coherence',
        'robbert': 'new_analysis/results/models/robbert_reduced_top_coherence',
        'qwen3': 'new_analysis/results/models/qwen3_reduced_top_coherence'
    }

    print("\n=== Summing topic counts per model ===")
    totals = {}

    for model_name, model_path in model_paths.items():
        topic_info_path = os.path.join(model_path, "topic_info_reduced_outliers.csv")
        
        try:
            df = pd.read_csv(topic_info_path)

            if "Count" not in df.columns:
                print(f"⚠️ Skipping {model_name}: no 'Count' column found.")
                continue

            total_count = df["Count"].sum()
            totals[model_name] = total_count
            print(f"{model_name}: total topic counts = {total_count}")

        except Exception as e:
            print(f"❌ Error processing {model_name}: {e}")

    # Summary
    print("\n=== Summary of total counts ===")
    for model_name, total in totals.items():
        print(f"{model_name}: {total}")

if __name__ == "__main__":
    main()
