#!/bin/bash
#SBATCH --job-name=visualize_merged
#SBATCH --time=00:30:00            # walltime
#SBATCH --cpus-per-task=4
#SBATCH -N 1
#SBATCH --ntasks=1      # one CPU core
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out      # log file
#SBATCH --error=logs/%x_%j.err


set -euo pipefail
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
source surfvenv/bin/activate

srun python ./scripts/visualize_docs.py
