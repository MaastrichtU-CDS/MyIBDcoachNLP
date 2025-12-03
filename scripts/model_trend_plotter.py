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
# Average trends plot (1×3 per parameter)
# =============================================

def plot_average_trends_all(df, outdir):
    """
    For each parameter (min_cluster_size, n_components, n_neighbors):
        Create a row of 4 subplots:
            1. C_v Coherence (raw)
            2. Topic Diversity (raw)
            3. Document Assignment % (raw)
            4. Number of Topics (raw)

        Each line = one model, summarized by group means for that parameter.
        Y-limits are consistent across ALL models and parameters (global),
        with padding added to improve visibility.
    """

    # -----------------------------------------
    # Metric definitions
    # -----------------------------------------
    metrics = ["c_v_score", "diversity_score", "doc_assignment_pct"]
    metric_display = {
        "c_v_score": "C_v Coherence",
        "diversity_score": "Topic Diversity",
        "doc_assignment_pct": "Document Assignment %"
    }
    topic_display = "Number of Topics"

    # -----------------------------------------
    # Compute global y-limits for metrics
    # -----------------------------------------
    metric_ylims = {}
    for metric in metrics:
        ymin = df[metric].min()
        ymax = df[metric].max()
        metric_ylims[metric] = (ymin, ymax)

    # Global y-limits for number_of_topics
    tmin = df["number_of_topics"].min()
    tmax = df["number_of_topics"].max()
    topic_ylim = (tmin, tmax)

    # -----------------------------------------
    # Main plotting loop
    # -----------------------------------------
    for parameter in PARAMETERS:
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(30, 6))
        axes = axes.flatten()

        handles_for_legend = []
        labels_for_legend = []

        for model_name in df["model_name"].unique():
            df_model = df[df["model_name"] == model_name].copy()
            color = MODEL_COLORS.get(model_name, "black")

            # Compute group means over the parameter sweep
            grouped = (
                df_model.groupby(parameter)[metrics + ["number_of_topics"]]
                .mean()
                .sort_index()
            )

            # Add model to legend ONCE
            if model_name not in labels_for_legend:
                h = axes[0].plot(
                    grouped.index,
                    grouped[metrics[0]],
                    color=MODEL_COLORS.get(model_name, "black")
                )[0]
                handles_for_legend.append(h)
                labels_for_legend.append(model_name)

            # -------------------------
            # Plot raw metrics
            # -------------------------
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
                    label=model_name
                )

                # Capture legend only once
                if not handles_for_legend:
                    handles_for_legend.append(line[0])
                    labels_for_legend.append(model_name)

            # -------------------------
            # Plot number of topics
            # -------------------------
            ax_topics = axes[3]
            ax_topics.plot(
                grouped.index,
                grouped["number_of_topics"],
                marker="s",
                linestyle="-",
                linewidth=2,
                alpha=0.9,
                color=color,
                label=model_name
            )

        # -----------------------------------------
        # Formatting metric subplots
        # -----------------------------------------
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

        # -----------------------------------------
        # Formatting topics subplot
        # -----------------------------------------
        ax_topics = axes[3]
        ax_topics.set_title(
            f"{topic_display} vs {DISPLAY_NAMES[parameter]}",
            fontsize=14,
        )
        ax_topics.set_xlabel(DISPLAY_NAMES[parameter])
        ax_topics.set_ylabel(topic_display)
        ax_topics.set_xticks(sorted(df[parameter].unique()))
        ax_topics.set_ylim(topic_ylim)
        ax_topics.grid(True)

        # -----------------------------------------
        # Combined legend
        # -----------------------------------------
        fig.legend(
            handles_for_legend,
            labels_for_legend,
            loc="upper right",
            ncol=len(labels_for_legend),
            bbox_to_anchor=(0.5, 1.12),
            fontsize=12
        )

        # Title
        fig.suptitle(
            f"Mean Trends Across Models — Parameter: {DISPLAY_NAMES[parameter]}",
            fontsize=18
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save figure
        outpath = os.path.join(outdir, f"average_trends_{parameter}.png")
        plt.savefig(outpath, dpi=150)
        plt.show()
        plt.close()


def plot_average_trends(df, outdir):
    """
    For each parameter (min_cluster_size, n_components, n_neighbors):
        Create a row of 2 subplots:
            1. C_v Coherence (raw)
            2. Topic Diversity (raw)

        Each line = one model, summarized by group means for that parameter.
        Y-limits are consistent across ALL models and parameters (global),
        with padding added to improve visibility.
    """

    # -----------------------------------------
    # Metric definitions
    # -----------------------------------------
    metrics = ["c_v_score", "diversity_score"]
    metric_display = {
        "c_v_score": "Cv Coherence",
        "diversity_score": "Topic Diversity"
    }

    # -----------------------------------------
    # Compute global y-limits for metrics
    # -----------------------------------------
    metric_ylims = {}
    for metric in metrics:
        ymin = df[metric].min()
        ymax = df[metric].max()
        metric_ylims[metric] = (ymin, ymax)

    # -----------------------------------------
    # Create one figure with 3 rows x 2 columns
    # -----------------------------------------
    n_params = len(PARAMETERS)
    fig, axes = plt.subplots(nrows=n_params, ncols=2, figsize=(15, 4.5 * n_params))

    handles_for_legend = []
    labels_for_legend = []

    # -----------------------------------------
    # Main plotting loop
    # -----------------------------------------
    for p_idx, parameter in enumerate(PARAMETERS):
        row_axes = axes[p_idx]

        for model_name in df["model_name"].unique():
            df_model = df[df["model_name"] == model_name].copy()
            color = MODEL_COLORS.get(model_name, "black")

            # Compute group means over the parameter sweep
            grouped = (
                df_model.groupby(parameter)[metrics + ["number_of_topics"]]
                .mean()
                .sort_index()
            )

            # Add model to legend ONCE
            if model_name not in labels_for_legend:
                h = row_axes[0].plot(
                    grouped.index,
                    grouped[metrics[0]],
                    color=MODEL_COLORS.get(model_name, "black")
                )[0]
                handles_for_legend.append(h)
                labels_for_legend.append(model_name)

            # -------------------------
            # Plot raw metrics
            # -------------------------
            for i, metric in enumerate(metrics):
                ax = row_axes[i]
                line = ax.plot(
                    grouped.index,
                    grouped[metric],
                    marker="o",
                    linestyle="-",
                    linewidth=2,
                    alpha=0.9,
                    color=color,
                    label=model_name
                )

                # Capture legend only once
                if not handles_for_legend:
                    handles_for_legend.append(line[0])
                    labels_for_legend.append(model_name)

        # -----------------------------------------
        # Formatting metric subplots for this row
        # -----------------------------------------
        for i, metric in enumerate(metrics):
            ax = row_axes[i]
            ymin, ymax = metric_ylims[metric]

            ax.set_title(
                f"{metric_display[metric]} vs {DISPLAY_NAMES[parameter]}",
                fontsize=16,
            )
            ax.set_xlabel(DISPLAY_NAMES[parameter])
            ax.set_ylabel(metric_display[metric])
            ax.set_ylim(ymin, ymax)
            ax.grid(True)
            ax.set_xticks(sorted(df[parameter].unique()))

    # -----------------------------------------
    # Combined legend
    # -----------------------------------------
    fig.suptitle(
        "Mean Trends of Cv Coherence and Topic Diversity per Embedding Model",
        fontsize=18,
          y=0.99   # move the title slightly up to make space for the legend
    )

    fig.text(
        0.5, 0.955,     # x, y position in figure coordinates
        "Over varying Min_Cluster_Size, N_Components, and N_Neighbors",
        ha='center',
        fontsize=12
    )

    fig.legend(
        handles_for_legend,
        labels_for_legend,
        loc="upper center",
        ncol=len(labels_for_legend),
        bbox_to_anchor=(0.5, 0.95),   # <-- legend directly under title
        fontsize=12
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure
    outpath = os.path.join(outdir, "average_trends_all_parameters_cv_td.png")
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
