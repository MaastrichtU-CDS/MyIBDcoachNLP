#!/usr/bin/env python3
"""
Clean, corrected version of model_trend_plotter.py
- Fixes all indentation errors and unmatched parentheses
- Removes leftover MIN/MAX topics logic
- Uses new in-range definition:
      * number_of_topics within model group's IQR
- Replaces all references to document_coverage
- Adds DISPLAY_NAMES dict for prettier labels
"""

import argparse
from ast import For
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Standardize display names
DISPLAY_NAMES = {
    "c_v_score": "C_V Coherence Score",
    "diversity_score": "Topic Diversity Score",
    "doc_assignment_pct": "Document Assignment %",
    "npmi": "NPMI",
    "min_cluster_size": "HDBSCAN Min_Cluster_Size",
    "n_components": "UMAP N_Components",
    "n_neighbors": "UMAP N_Neighbors",
}

# Parameters and metrics
PARAMETERS = ["min_cluster_size", "n_components", "n_neighbors"]
METRICS = ["c_v_score", "diversity_score", "doc_assignment_pct", "npmi"]

# Model colors
MODEL_COLORS = {
    "mpnet": "crimson",
    "robbert": "darkcyan",
    "qwen3": "orchid",
}

# =============================================
# Compute in-range flag
# =============================================

def compute_inrange_flags(df):
    """Compute IQR filter + top 10% doc_assignment_pct per model."""

    # Global IQR for number_of_topics
    q1 = df["number_of_topics"].quantile(0.25)
    q3 = df["number_of_topics"].quantile(0.75)

    df["in_range"] = (
        df["number_of_topics"].between(q1, q3)
    )
    return df

# =============================================
# Column validator
# =============================================

def check_columns(df):
    required = set(["model_name", "number_of_topics"] + PARAMETERS + METRICS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

# =============================================
# Per-model plot
# =============================================

def plot_per_model(df, model_name, outdir):
    """
    For each parameter (min_cluster_size, n_components, n_neighbors):
        Create a row of 4 subplots:
            1. C_v Coherence (raw)
            2. Topic Diversity (raw)
            3. Document Assignment % (raw)
            4. Normalized Pointwise Mutual Information (NPMI) Score (raw)
    """

    df_model = df[df["model_name"] == model_name].copy()
    df_model = compute_inrange_flags(df_model)
    check_columns(df_model)
    main_color = MODEL_COLORS.get(model_name, "steelblue")

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(24, 18))

    for r, parameter in enumerate(PARAMETERS):
        group_params = [p for p in PARAMETERS if p != parameter]
        

        for c, metric in enumerate(METRICS):
            ax = axes[r][c]
            


            # Scatter points
            ax.scatter(
                df_model.loc[:, parameter],
                df_model.loc[:, metric],
                color=main_color, alpha=0.4)

            # Grey connecting lines
            grouped = df_model.groupby(group_params)
            for _, group in grouped:
                group_sorted = group.sort_values(parameter)
                ax.plot(
                    group_sorted[parameter],
                    group_sorted[metric],
                    color=main_color, alpha=0.2, linewidth=1,
                )

            # Mean (all)
            mean_all = df_model.groupby(parameter)[metric].mean().sort_index()
            ax.plot(
                mean_all.index, mean_all.values,
                color=main_color, linestyle="--", marker="o",
                label=f"Mean {DISPLAY_NAMES[metric]}" if (r == 0 and c == 0) else None,
            )

            if parameter == "n_components":
                ax.set_xticks([5, 7, 10, 12, 15])

            # Labels
            ax.set_title(f"{DISPLAY_NAMES[metric]} vs {DISPLAY_NAMES[parameter]}")
            ax.set_xlabel(DISPLAY_NAMES[parameter])
            ax.set_ylabel(DISPLAY_NAMES[metric])
            ax.grid(True)


    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")


    fig.suptitle(f"Trends for Model: {model_name}", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])


    outpath = os.path.join(outdir, f"trends_{model_name}.png")
    plt.savefig(outpath, dpi=150)
    plt.show()
    plt.close()


# =============================================
# Average trends comparison between models (1×3 per parameter)
# =============================================

def plot_average_trends_all(df, outdir):
    metrics = [
        "c_v_score",
        "diversity_score",
        "doc_assignment_pct",
        "number_of_topics",
        "npmi",
    ]

    metric_display = {
        "c_v_score": "C_v Coherence",
        "diversity_score": "Topic Diversity",
        "doc_assignment_pct": "Document Assignment %",
        "number_of_topics": "Number of Topics",
        "npmi": "Normalized Pointwise Mutual Information (NPMI) Score",
    }

    metric_ylims = {}
    for metric in metrics:
        ymin = df[metric].min()
        ymax = df[metric].max()
        metric_ylims[metric] = (ymin, ymax)

    for parameter in PARAMETERS:
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(30, 6))
        axes = axes.flatten()

        handles_for_legend = []
        labels_for_legend = []

        for model_name in df["model_name"].unique():
            df_model = df[df["model_name"] == model_name].copy()
            color = MODEL_COLORS.get(model_name, "black")

            grouped = (
                df_model.groupby(parameter)[metrics]
                .mean()
                .sort_index()
            )

            for i, metric in enumerate(metrics):
                ax = axes[i]

                line = ax.plot(
                    grouped.index,
                    grouped[metric],
                    marker="o",
                    linestyle="-",
                    linewidth=2,
                    alpha=0.9,
                    color=color,
                    label=model_name,
                )

                if model_name not in labels_for_legend:
                    handles_for_legend.append(line[0])
                    labels_for_legend.append(model_name)

        for i, metric in enumerate(metrics):
            ax = axes[i]
            ymin, ymax = metric_ylims[metric]

            ax.set_title(
                f"{metric_display[metric]} vs {DISPLAY_NAMES[parameter]}",
                fontsize=14,
            )
            ax.set_xlabel(DISPLAY_NAMES[parameter])
            ax.set_ylabel(metric_display[metric])
            ax.set_ylim(ymin, ymax)
            ax.grid(True)
            ax.set_xticks(sorted(df[parameter].unique()))

        fig.legend(
            handles_for_legend,
            labels_for_legend,
            loc="upper right",
            ncol=len(labels_for_legend),
            bbox_to_anchor=(0.5, 1.12),
            fontsize=12,
        )

        fig.suptitle(
            f"Mean Trends Across Models — Parameter: {DISPLAY_NAMES[parameter]}",
            fontsize=18,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        outpath = os.path.join(outdir, f"average_trends_{parameter}.png")
        plt.savefig(outpath, dpi=150)
        plt.show()
        plt.close()

# =============================================
# Main
# =============================================

def main():
    parser = argparse.ArgumentParser(description="Model trend plotter.")
    parser.add_argument("--input", required=True, help="CSV input file")
    parser.add_argument("--outdir", required=True, help="Directory for plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input)
    print(df.head())

    for model_name in df["model_name"].unique():
        plot_per_model(df, model_name, args.outdir)

    plot_average_trends_all(df, args.outdir)

if __name__ == "__main__":
    main()
