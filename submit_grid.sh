#!/bin/bash

# Count chunk files automatically
NUM_CHUNKS=$(ls chunks/chunk_*.json | wc -l)

echo "Submitting ${NUM_CHUNKS} array jobs..."

sbatch --array=0-$((NUM_CHUNKS-1))%4 submit_chunk.slurm
