#!/usr/bin/env python3
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Filter and rank model results.")
    parser.add_argument("--in", dest="inp", default="new_analysis/results/df_model_comparison.csv",
                        help="Input CSV file (default: new_analysis/results/df_model_comparison.csv)")
    parser.add_argument("--out", dest="out", default="new_analysis/results/df_model_comparison_top.csv",
                        help="Output CSV file (default: new_analysis/results/df_model_comparison_top.csv)")
    args = parser.parse_args()

    # Read CSV
    df = pd.read_csv(args.inp)

    # 1) Filter by IQR for number_of_topics
    df = df[df["number_of_topics"].between(100, 300, inclusive="both")]

    # 2) For each model, keep only top 5% by document_coverage
    coverage_thresh = df.groupby("model_name")["document_coverage"].transform(lambda x: x.quantile(0.95))
    df = df[df["document_coverage"] >= coverage_thresh]

    # 3) Rank by diversity_score within each model and take top 3 rows
    df = (
        df.sort_values(["model_name", "diversity_score"], ascending=[True, False])
          .groupby("model_name", group_keys=False)
          .head(3)
    )

    # Save result
    df.to_csv(args.out, index=False)
    print(f"Saved filtered top-3 rows per model to {args.out}")

if __name__ == "__main__":
    main()
