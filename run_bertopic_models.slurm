#!/bin/bash
#SBATCH --job-name=best_configs
#SBATCH --output=logs/bertopic_%A_%a.out
#SBATCH --error=logs/bertopic_%A_%a.err
#SBATCH --array=0-2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=rome  # adjust based on your cluster

# Define the models array
models=("mpnet" "robbert" "qwen3")

# Get the model for this array task
model=${models[$SLURM_ARRAY_TASK_ID]}

echo "Starting job for model: $model"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"

# Activate conda environment if needed
source surfvenv/bin/activate

# Run the Python script
python scripts/run_best_configs.py $model

echo "Completed job for model: $model"