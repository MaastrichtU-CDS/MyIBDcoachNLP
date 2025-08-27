#!/bin/bash
#SBATCH --job-name=plot_hierarchy
#SBATCH --time=00:20:00            # walltime
#SBATCH --cpus-per-task=1 
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

srun python ./scripts/plot_hierarchy.py
