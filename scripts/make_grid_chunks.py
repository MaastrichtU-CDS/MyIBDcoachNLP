from itertools import product
import json
import os

min_cluster_sizes = [10, 15, 20, 25, 30, 35, 40]
n_components = [5, 7, 10, 15]
n_neighbors = [10, 20, 30, 40]

all_combinations = list(product(min_cluster_sizes, n_components, n_neighbors))
chunk_size = 10  # Or 20, depending on your compute budget

os.makedirs("chunks", exist_ok=True)

for i in range(0, len(all_combinations), chunk_size):
    chunk = all_combinations[i:i + chunk_size]
    with open(f"chunks/chunk_{i // chunk_size}.json", "w") as f:
        json.dump(chunk, f)

print(f"Created {len(all_combinations) // chunk_size + 1} chunks.")
