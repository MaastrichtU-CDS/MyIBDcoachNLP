#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=10-11%2
#SBATCH --output=/home/jzhang/mijnidbcoachnlp/logs/%x_%A_%a.out
#SBATCH --error=/home/jzhang/mijnidbcoachnlp/logs/%x_%A_%a.err

MODEL_NAME="qwen3"   # set once here

source /home/jzhang/mijnidbcoachnlp/surfvenv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} \
  python new_analysis/scripts/run_grid_chunk.py "$MODEL_NAME" "$SLURM_ARRAY_TASK_ID"
