import pandas as pd

def rank_models(df, save_ranked=False, metric="diversity"):
    """
    Rank BERTopic model configurations:
    - One best configuration per model
    - Globally best configuration using chosen metric
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with model comparison results.
    save_ranked : bool
        Save ranked per-model result to CSV.
    metric : str
        Ranking by metric: "diversity", "cv", or "npmi".

    Returns
    -------
    ranked_df : pandas.DataFrame
        Best configuration per model (sorted by selected metric).
    """

    # -----------------------------
    # Validate metric
    # -----------------------------
    metric = metric.lower()
    valid_metrics = ["diversity", "cv", "npmi"]
    if metric not in valid_metrics:
        raise ValueError(f"metric must be one of {valid_metrics}, got '{metric}'")
    
    # -----------------------------
    # Compute the score to use
    # -----------------------------
    df = df.copy()

    metric_map = {
        "diversity": "diversity_score",
        "cv": "c_v_score",
        "npmi": "npmi",
    }
    metric_col = metric_map[metric]

    # --------------------------------------------------
    # 1. Top 20% doc_assignment_pct per model
    # --------------------------------------------------
    doc_assignment_thresholds = (
        df.groupby("model_name")["doc_assignment_pct"]
          .quantile(0.80)
          .rename("doc_top20pct")
    )
    df1 = df.merge(doc_assignment_thresholds, on="model_name")
    df1 = df1[df1["doc_assignment_pct"] >= df1["doc_top20pct"]]

    # --------------------------------------------------
    # 2. Restrict to IQR(number_of_topics)
    # --------------------------------------------------
    q1 = df["number_of_topics"].quantile(0.25)
    q3 = df["number_of_topics"].quantile(0.75)
    df2 = df1[df1["number_of_topics"].between(q1, q3)]

    # --------------------------------------------------
    # 3. One best config per model
    # --------------------------------------------------
    ranked_df = (
        df2.sort_values(["model_name", metric_col], ascending=[True, False])
           .groupby("model_name")
           .first()
           .reset_index()
           .sort_values(metric_col, ascending=False)
    )

    # --------------------------------------------------
    # Optional save
    # --------------------------------------------------
    if save_ranked:
        # drop "doc_top20pct" "run_id" and "chunk_id" column before saving
        ranked_df.drop(columns=["doc_top20pct", "run_id", "chunk_id"], inplace=True)
        ranked_df.to_csv(
            "/home/jzhang/mijnidbcoachnlp/results/model_comparison/top_ranked_models.csv",
            index=False
        )

        print("Saved per-model rankings to: top_ranked_models.csv")
    return ranked_df

def main():
    # Load the combined model comparison DataFrame
    df = pd.read_csv("/home/jzhang/mijnidbcoachnlp/results/model_comparison/model_comparison.csv")
    metric = "diversity"  # Change this to "cv" or "npmi" as needed
    # Rank models and get the best configuration per model
    ranked_df = rank_models(df, save_ranked=True, metric=metric)
    # Print the top-ranked models
    print(f"Top-ranked models based on selected metric {metric}:")
    print(ranked_df)


if __name__ == "__main__":
    main()