from itertools import product
import json
import os
from glob import glob


# -----------------------------
# 1. Define ranges for the hyperparameter grid search
# -----------------------------
min_cluster_sizes = [10, 15, 20, 25, 30, 35, 40, 45, 50]
n_components = [5, 7, 10, 12, 15]
n_neighbors = [10, 20, 30, 40, 50]

all_combinations = list(product(min_cluster_sizes, n_components, n_neighbors))

# -----------------------------
# 2. Save the chunks
# -----------------------------
chunk_size = 10
os.makedirs("chunks", exist_ok=True)

for i in range(0, len(all_combinations), chunk_size):
    chunk = all_combinations[i:i + chunk_size]
    with open(f"chunks/chunk_{i // chunk_size}.json", "w") as f:
        json.dump(chunk, f)

n_chunks = (len(all_combinations) + chunk_size - 1) // chunk_size
print(f"Created {n_chunks} chunks.")