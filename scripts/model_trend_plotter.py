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
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# -------------------------------
# Display names for nicer labels
# -------------------------------
DISPLAY_NAMES = {
    "c_v_score": "C_V Coherence Score",
    "diversity_score": "Topic Diversity Score",
    "doc_assignment_pct": "Document Assignment %",
    "min_cluster_size": "HDBSCAN Min_Cluster_Size",
    "n_components": "UMAP N_Components",
    "n_neighbors": "UMAP N_Neighbors",
}

# Parameters and metrics
PARAMETERS = ["min_cluster_size", "n_components", "n_neighbors"]
METRICS = ["c_v_score", "diversity_score", "doc_assignment_pct"]

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
    df_model = df[df["model_name"] == model_name].copy()
    df_model = compute_inrange_flags(df_model)
    check_columns(df_model)
    main_color = MODEL_COLORS.get(model_name, "steelblue")

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(24, 18))

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
                        
            # Compute mean number_of_topics per parameter value
            mean_topics = df_model.groupby(parameter)["number_of_topics"].mean().sort_index()

            ax2 = ax.twinx()
            mean_topics = df_model.groupby(parameter)["number_of_topics"].mean().sort_index()
            ax2.plot(
            mean_topics.index, mean_topics.values,
            color="black", linestyle="--", linewidth=2, alpha=0.7,
            label="Mean # Topics" if (r == 0 and c == 0) else None,
            )
            ax2.set_ylabel("Mean # Topics", color="black")
            ax2.tick_params(axis='y', labelcolor='black')


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
# Average trends plot (1×3 per parameter)
# =============================================

def plot_average_trends(df, outdir):
    """
    For each parameter:
        - Row of 3 subplots (one per metric)
        - Solid model-color line = mean normalized metric value
        - Dashed model-color line on twin y-axis = mean #topics
    """

    for parameter in PARAMETERS:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
        axes = axes.flatten()

        for model_name in df["model_name"].unique():
            df_model = df[df["model_name"] == model_name].copy()

            # Normalize metrics for fair comparison
            scaler = MinMaxScaler()
            df_model[METRICS] = scaler.fit_transform(df_model[METRICS])

            color = MODEL_COLORS.get(model_name, "black")

            # Precompute topic trend (not normalized!)
            mean_topics = (
                df_model.groupby(parameter)["number_of_topics"]
                .mean()
                .sort_index()
            )

            for i, metric in enumerate(METRICS):
                ax = axes[i]

                # -------------------------------
                # Metric trend (solid line)
                # -------------------------------
                mean_metric = (
                    df_model.groupby(parameter)[metric]
                    .mean()
                    .sort_index()
                )

                ax.plot(
                    mean_metric.index,
                    mean_metric.values,
                    marker="o",
                    linestyle="-",
                    linewidth=2,
                    alpha=0.9,
                    color=color,
                    label=f"{model_name} metric" if i == 0 else None,
                )

                # -------------------------------
                # Topic count trend (twin axis)
                # -------------------------------
                ax2 = ax.twinx()
                ax2.plot(
                    mean_topics.index,
                    mean_topics.values,
                    linestyle="--",
                    linewidth=2,
                    alpha=0.8,
                    color=color,
                    label=f"{model_name} #topics" if i == 0 else None,
                )

                # Right axis label only once per subplot
                if i == 2:  # right-most panel
                    ax2.set_ylabel("Mean # Topics", color="grey")
                    ax2.tick_params(axis='y', labelcolor='grey')

        # ---------------------------------------
        # Formatting per subplot
        # ---------------------------------------
        for ax, metric in zip(axes, METRICS):
            ax.set_title(
                f"Normalized {DISPLAY_NAMES[metric]} vs {DISPLAY_NAMES[parameter]}"
            )
            ax.set_xlabel(DISPLAY_NAMES[parameter])
            ax.set_ylabel(f"Normalized {DISPLAY_NAMES[metric]}")
            ax.set_ylim(0, 1)
            ax.grid(True)
            ax.set_xticks(sorted(df[parameter].unique()))

        # ---------------------------------------
        # Combined legends (metric + topic lines)
        # ---------------------------------------
        # Build a combined legend using both axes
        handles_main = []
        labels_main = []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            handles_main.extend(h)
            labels_main.extend(l)

            # also fetch legend handles from twin axes
            h2, l2 = ax.twinx().get_legend_handles_labels()
            handles_main.extend(h2)
            labels_main.extend(l2)

        # Remove duplicates
        combined = dict(zip(labels_main, handles_main))
        fig.legend(
            combined.values(),
            combined.keys(),
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 1.12),
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

    plot_average_trends(df, args.outdir)

if __name__ == "__main__":
    main()
