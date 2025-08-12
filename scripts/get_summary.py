import pandas as pd
import os

def get_summary(
    df: pd.DataFrame,
    models: list,
    metrics=("c_v_score", "diversity_score", "document_coverage"),
    min_topics: int = 100,
    max_topics: int = 300,
) -> pd.DataFrame:
    """
    Summary stats for each metric across all models.
    
    Returns a DataFrame with:
    - Columns: model, c_v_score, topic_diversity, document_coverage
    - For each model, 3 sub-rows:
      1. Mean [SD] (in-range)
      2. Mean [SD] (all-data) 
      3. Max value
      4. Model combination at max value
    """
    summary_data = []
    
    # Map metric names to desired column names
    metric_mapping = {
        "c_v_score": "c_v_score",
        "diversity_score": "topic_diversity", 
        "document_coverage": "document_coverage"
    }
    
    for model in models:
        df_model = df[df["model_name"] == model]
        
        # Check if the model has the required metrics
        available_metrics = [m for m in metrics if m in df_model.columns]
        if not available_metrics:
            raise ValueError(f"None of the requested metrics are present in df_model for model {model}.")
        
        # Create mask for in-range data
        in_mask = df_model["number_of_topics"].between(min_topics, max_topics, inclusive="both")
        df_inrange = df_model.loc[in_mask]
        
        # Row 1: Mean [SD] (in-range)
        row1_data = {"model": f"{model} - Mean [SD] (in-range)"}
        for metric in available_metrics:
            column_name = metric_mapping[metric]
            if len(df_inrange) > 0:
                mean_val = df_inrange[metric].mean()
                std_val = df_inrange[metric].std()
                row1_data[column_name] = f"{mean_val:.4f} [{std_val:.4f}]"
            else:
                row1_data[column_name] = "NaN [NaN]"
        
        # Row 2: Mean [SD] (all-data)
        row2_data = {"model": f"{model} - Mean [SD] (all-data)"}
        for metric in available_metrics:
            column_name = metric_mapping[metric]
            mean_val = df_model[metric].mean()
            std_val = df_model[metric].std()
            row2_data[column_name] = f"{mean_val:.4f} [{std_val:.4f}]"
        
        # Row 3: Max value
        row3_data = {"model": f"{model} - Max value"}
        max_indices = {}  # Store indices of max values for each metric
        
        for metric in available_metrics:
            column_name = metric_mapping[metric]
            max_val = df_model[metric].max()
            max_idx = df_model[metric].idxmax()
            max_indices[metric] = max_idx
            row3_data[column_name] = f"{max_val:.4f}"
        
        # Row 4: Model combination at max value
        row4_data = {"model": f"{model} - Config at max"}
        
        for metric in available_metrics:
            column_name = metric_mapping[metric]
            max_idx = max_indices[metric]
            
            # Get the combination value at the row where this metric reaches its max
            if "combination" in df_model.columns:
                combination_at_max = df_model.loc[max_idx, "combination"]
                row4_data[column_name] = str(combination_at_max)
            else:
                row4_data[column_name] = "combination column not found"
        
        # Add all rows for this model
        summary_data.extend([row1_data, row2_data, row3_data, row4_data])
    
    # Create DataFrame with desired format
    final_summary = pd.DataFrame(summary_data)
    
    # Ensure column order
    desired_columns = ["model", "c_v_score", "topic_diversity", "document_coverage"]
    final_summary = final_summary.reindex(columns=desired_columns)
    
    return final_summary


# Example usage and formatting:
def print_formatted_summary(summary_df):
    """Print the summary with proper formatting"""
    
    # Set pandas display options for better readability
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    
    print(summary_df.to_string(index=False))
    
    # Reset options
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width') 
    pd.reset_option('display.max_colwidth')

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

    # Metrics to plot
    metrics = [
        "document_coverage",
        "diversity_score",
        "c_v_score"
    ]

    # Get the summary for all models
    model_summary = get_summary(df, models, metrics)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the summary to a CSV file
    model_summary.to_csv(f"{output_dir}/model_summaries_all.csv")

# =========================== execution ======================= #
if __name__ == "__main__":
    main()