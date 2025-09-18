from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import os
from time import perf_counter

DATA_PATH = "quotes_data.pkl"
EMBEDDINGS_PATH = "quotes_embeddings.npy"

BEST_K = 3

# --- Pre-run check ---
if not os.path.isfile(EMBEDDINGS_PATH) or not os.path.isfile(DATA_PATH):
    print(f"Error: Data ({DATA_PATH}) or embeddings ({EMBEDDINGS_PATH}) file not found.")
    print(f"Please run make_index.py successfully before running this script.")
    exit()


print("Loading model...")
model = SentenceTransformer('avsolatorio/GIST-large-Embedding-v0', device='mps')


print(f"Loading embeddings from {EMBEDDINGS_PATH}...")
embeddings = np.load(EMBEDDINGS_PATH)

print("Building FAISS index from embeddings...")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
print("Index built successfully.")


print(f"Loading quotes data from {DATA_PATH}...")
with open(DATA_PATH, "rb") as f:
    quotes_data = pickle.load(f)


# Open the results file in 'a' (append) mode
results_file = open("recent_found_indices.txt", "a")

# Loop so you can do multiple queries
try:
    while True:
        sample_quote = input("Paraphrase your quote: ")

        if sample_quote.strip() == "":
            break

        start_time = perf_counter()

        # The key change is here: normalize_embeddings=True
        embedding = model.encode(
            [sample_quote],
            normalize_embeddings=True
        )

        D, I = index.search(embedding, BEST_K)

        end_time = perf_counter()

        for i in range(BEST_K):
            idx = int(I[0][i])
            print(f"{idx}", file=results_file)
            sentence, attributes = quotes_data[idx]
            print(f"\n{idx}: '{sentence}'\n\t{attributes}")
        
        print(f"Time taken: {end_time - start_time:.3f} seconds")
        print("\n")
finally:
    results_file.close()
    print("Exiting.")
