#!/bin/bash
#SBATCH --job-name=visualize_selected_topics
#SBATCH --time=00:10:00            # walltime
#SBATCH --cpus-per-task=1
#SBATCH -N 1
#SBATCH --ntasks=1      # one CPU core
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out      # log file


set -euo pipefail
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
source surfvenv/bin/activate

srun python ./scripts/visualize_docs.py
