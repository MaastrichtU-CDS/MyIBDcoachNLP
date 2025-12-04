from itertools import product
import json
import os
from glob import glob

# -----------------------------
# 1. Load previously tested combos
# -----------------------------
old_combinations = set()

if os.path.exists("chunks_test/tested_chunks"):
    for path in glob("chunks_test/tested_chunks/*.json"):
        with open(path, "r") as f:
            chunk = json.load(f)
            for combo in chunk:
                old_combinations.add(tuple(combo))

print(f"Loaded {len(old_combinations)} previously tested combinations.")

# -----------------------------
# 2. Define NEW ranges to test
# -----------------------------
min_cluster_sizes = [10, 15, 20, 25, 30, 35, 40, 45, 50]  # expanded example
n_components = [5, 7, 10, 12, 15]
n_neighbors = [10, 20, 30, 40, 50]

all_combinations = list(product(min_cluster_sizes, n_components, n_neighbors))

# -----------------------------
# 3. Filter new combinations
# -----------------------------
new_combinations = [c for c in all_combinations if c not in old_combinations]

print(f"Found {len(new_combinations)} new combinations to test.")

# -----------------------------
# 4. Save new chunks
# -----------------------------
chunk_size = 10
os.makedirs("chunks_test", exist_ok=True)

for i in range(0, len(new_combinations), chunk_size):
    chunk = new_combinations[i:i + chunk_size]
    with open(f"chunks_test/chunk_{i // chunk_size}.json", "w") as f:
        json.dump(chunk, f)

print(f"Created {len(new_combinations) // chunk_size + 1} new chunks.")

# -----------------------------
# 5. Update master log (tested.json)
# -----------------------------
tested_log_path = "chunks_test/tested.json"
tested = set(old_combinations)

# Add newly generated ones as well (they are scheduled to run)
for c in new_combinations:
    tested.add(c)

with open(tested_log_path, "w") as f:
    json.dump([list(x) for x in tested], f)

print("Updated chunks/tested.json")