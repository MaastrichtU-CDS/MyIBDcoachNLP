#!/usr/bin/env python3
import os
import re
import glob
import argparse
import pandas as pd
from sklearn import base


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


def extract_run_id(filepath):
    """
    Extracts run_id from path: results/checkpoints/<run_id>/file.csv
    """
    parts = filepath.split(os.sep)
    # Expect: .../results/checkpoints/run_XXX/filename.csv
    for i, p in enumerate(parts):
        if p == "checkpoints" and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown_run"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine BERTopic grid-search checkpoint CSV files across runs."
    )

    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="results/checkpoints",
        help="Directory containing run_* folders with checkpoint CSV files."
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default="results/model_comparison/model_comparison.csv",
        help="Path to the output combined CSV file."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    checkpoints_dir = args.checkpoints_dir
    output_file = args.output_file

    # NEW: recursive search
    pattern = os.path.join(checkpoints_dir, "**", "*_results_chunk_*.csv")
    csv_files = sorted(glob.glob(pattern, recursive=True))

    if not csv_files:
        print(f"No CSV files found matching pattern: {pattern}")
        return

    print(f"Found {len(csv_files)} checkpoint CSVs across runs.")

    dfs = []
    for f in csv_files:
        model_name = extract_model_name(f)
        run_id = extract_run_id(f)

        if not model_name:
            print(f"Could not extract model name from '{f}' — skipping.")
            continue

        try:
            df = pd.read_csv(f)
            # Normalize column names
            if "doc_assignment_pct" not in df.columns and "document_coverage" in df.columns:
                df = df.rename(columns={"document_coverage": "doc_assignment_pct"})
            df["model_name"] = model_name
            df["run_id"] = run_id
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not dfs:
        print("No valid dataframes to concatenate.")
        return

    combined_df = pd.concat(dfs, ignore_index=True)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined_df.to_csv(output_file, index=False)

    print(f"Saved combined DataFrame to: {output_file}")


if __name__ == "__main__":
    main()
