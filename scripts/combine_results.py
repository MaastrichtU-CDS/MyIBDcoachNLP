import os
import re
import glob
import pandas as pd

def extract_model_name(filename):
    """
    Extracts model_name from pattern: {model_name}_results_chunk_{chunk_id}.csv
    Works even if model_name contains underscores.
    """
    base = os.path.basename(filename)
    match = re.match(r"(.+)_results_chunk_\d+\.csv", base)
    if match:
        return match.group(1)
    return None

def main():
    checkpoints_dir = "./checkpoints"
    pattern = os.path.join(checkpoints_dir, "*_results_chunk_*.csv")
    
    csv_files = sorted(glob.glob(pattern))
    
    if not csv_files:
        print("No CSV files found matching pattern:", pattern)
        return

    print(f"Found {len(csv_files)} CSV files. Concatenating...")
    
    dfs = []
    for f in csv_files:
        model_name = extract_model_name(f)
        if not model_name:
            print(f"Could not extract model name from: {f}")
            continue

        try:
            df = pd.read_csv(f)
            df["model_name"] = model_name  # Add new column
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not dfs:
        print("No dataframes to concatenate.")
        return

    combined_df = pd.concat(dfs, ignore_index=True)
    
    output_file = "new_analysis/results/df_model_comparison.csv"
    combined_df.to_csv(output_file, index=False)
    
    print(f"Saved combined DataFrame to {output_file}")

if __name__ == "__main__":
    main()