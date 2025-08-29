#!/usr/bin/env python3
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Filter and rank model results.")
    parser.add_argument("--in", dest="inp", default="new_analysis/results/tables/df_model_comparison.csv",
                        help="Input CSV file (default: new_analysis/results/tables/df_model_comparison.csv)")
    args = parser.parse_args()

    # Read CSV
    df = pd.read_csv(args.inp)

    # 1) Filter by IQR for number_of_topics
    df = df[df["number_of_topics"].between(100, 300, inclusive="both")]

    # 2) For each model, keep only top 5% by document_coverage
    coverage_thresh = df.groupby("model_name")["document_coverage"].transform(lambda x: x.quantile(0.95))
    df = df[df["document_coverage"] >= coverage_thresh]

    df_td = df.copy()
    df_cv = df.copy()
    # 3) Rank by diversity_score within each model and take top 3 rows
    df_td = (
        df_td.sort_values(["diversity_score"], ascending= [False])
          .groupby("model_name", group_keys=False)
          .head(1)
    )

    # 3) Rank by cv coherence within each model and take top 3 rows
    df_cv = (
        df_cv.sort_values(["c_v_score"], ascending= [False])
          .groupby("model_name", group_keys=False)
          .head(1)
    )

    # Save result
    df_td.to_csv("new_analysis/results/tables/df_model_comparison_top_diversity.csv", index=False)
    df_cv.to_csv("new_analysis/results/tables/df_model_comparison_top_coherence.csv", index=False)
    print(f"Saved filtered top-3 rows per model")

if __name__ == "__main__":
    main()
