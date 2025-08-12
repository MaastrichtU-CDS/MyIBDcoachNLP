import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_trends_permodel(df, model_name, metrics, parameter_to_plot, main_color, output_dir="./results/graphs"):
    df_model = df[df["model_name"] == model_name]
    
    # Acceptable number of topics range
    min_topics, max_topics = 100, 300

    # create sub-plots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharex=True)

    for ax, metric in zip(axes, metrics):
        in_range = (df_model["number_of_topics"] >= min_topics) & (df_model["number_of_topics"] <= max_topics)
        out_range = ~in_range

        # Scatter plots
        ax.scatter(df_model[parameter_to_plot][out_range], df_model[metric][out_range],
                color="lightgrey", alpha=0.3, label="Out of range")

        ax.scatter(df_model[parameter_to_plot][in_range], df_model[metric][in_range],
                color=main_color, alpha=0.5, label="In range")

        # Overall mean trend (all points)
        mean_all = df_model.groupby(parameter_to_plot)[metric].mean()
        ax.plot(mean_all.index, mean_all.values, color="black", linestyle="--", marker="o", label="Mean (All)")

        # Mean trend
        mean_in = df_model[in_range].groupby(parameter_to_plot)[metric].mean()
        ax.plot(mean_in.index, mean_in.values, color=main_color, marker="o", label="Mean (In range)")

        # Connect points for same (number_of_topics, cluster_selection_epsilon)
        grouped = df_model.groupby(["number_of_topics", "cluster_selection_epsilon"])

        for (topics, epsilon), group in grouped:
            group_sorted = group.sort_values(parameter_to_plot)
            ax.plot(
                group_sorted[parameter_to_plot],
                group_sorted[metric],
                color="blue" if (min_topics <= topics <= max_topics) else "grey",
                alpha=0.3,
                linewidth=1
            )

        title = None
        if metric == "c_v_score":
            title = "Cv Coherence"
        elif metric == "diversity_score":
            title = "Topic Diversity"
        elif metric == "document_coverage":
            title = "Topic Assignment Proportion"

        # Formatting
        ax.set_title(title.title())
        ax.set_ylabel(title.title())
        ax.set_xlabel(parameter_to_plot)
        ax.grid(True)

    # Shared legend (only once)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    # Overall title
    fig.suptitle(f"Metric Trends vs {parameter_to_plot} with In/Out Topic Ranges — Model {model_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/trends_{parameter_to_plot}_{model_name}.png")
    plt.close()  # Close figure to free memory


def plot_inrange_mean_trends_all_models(df, models, metrics, parameter_to_plot, model_colors, output_dir="./results/graphs"):
    min_topics, max_topics = 100, 300

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5), sharex=False)
    axes = axes.flatten()

    for model_name in models:
        df_model = df[df["model_name"] == model_name].copy()

        # All data
        df_all = df_model

        # In-range data
        in_range = (df_model["number_of_topics"] >= min_topics) & (df_model["number_of_topics"] <= max_topics)
        df_inrange = df_model[in_range]

        for i, metric in enumerate(metrics):
            # Plot full-data trends (solid line)
            mean_all = df_all.groupby(parameter_to_plot)[metric].mean()
            axes[i].plot(
                mean_all.index,
                mean_all.values,
                marker="o",
                linestyle="--",
                label=f"{model_name} (all)",
                color=model_colors[model_name],
                alpha=0.4
            )

            # Plot in-range trends (dotted line)
            mean_in = df_inrange.groupby(parameter_to_plot)[metric].mean()
            axes[i].plot(
                mean_in.index,
                mean_in.values,
                marker="o",
                label=f"{model_name} (in-range)",
                color=model_colors[model_name],
                alpha=1
            )

    # Formatting
    for ax, metric in zip(axes, metrics):
        if metric == "c_v_score":
            title = "Cv Coherence"
        elif metric == "diversity_score":
            title = "Topic Diversity"
        elif metric == "document_coverage":
            title = "Topic Assignment Proportion"
        
        ax.set_title(title.title())
        ax.set_ylabel("Value")
        ax.set_xlabel(parameter_to_plot)
        ax.set_ylim(0.3, 0.8)
        ax.grid(True)
        ax.legend()
        if parameter_to_plot == "n_components":
            unique_vals = sorted(df[parameter_to_plot].unique())
            ax.set_xticks(unique_vals)

    fig.suptitle(f"Mean Trends by Model — Parameter: {parameter_to_plot}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/trends_{parameter_to_plot}_all_models.png")
    plt.close()  # Close figure to free memory


# =================== main =================== #
def main(): 
    # Define paths (make them configurable)
    data_path = "./new_analysis/results/df_model_comparison.csv"  # Made relative
    output_dir = "./new_analysis/results"
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    # read data
    df = pd.read_csv(data_path, index_col=0)
    
    model_colors = {
        "mpnet": "crimson",
        "robbert": "darkcyan",
        "qwen3": "orchid"
    }    
    models = list(model_colors.keys())

    parameters = ["min_cluster_size", "n_components", "n_neighbors"]

    # Metrics to plot
    metrics = [
        "document_coverage",
        "diversity_score",
        "c_v_score"
    ]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create graphs directory
    graphs_dir = f"{output_dir}/graphs"
    os.makedirs(graphs_dir, exist_ok=True)

    for parameter in parameters:
        plot_inrange_mean_trends_all_models(df, models, metrics, parameter_to_plot=parameter, 
                                          model_colors=model_colors, output_dir=graphs_dir)
        #for model in models:
            #visualize_trends_permodel(df=df, model_name=model, metrics=metrics, 
                                    #parameter_to_plot=parameter, main_color=model_colors[model],
                                    #output_dir=graphs_dir)


# =========================== execution ======================= #
if __name__ == "__main__":
    main()