from sentence_transformers import SentenceTransformer
from csv import reader
import faiss
import pickle
import os
import numpy as np
from time import perf_counter

DATA_PATH = "quotes_data.pkl"
EMBEDDINGS_PATH = "quotes_embeddings.npy" # We will save the raw embeddings here

# --- Clean up old files ---
if os.path.isfile(EMBEDDINGS_PATH):
    print(f"Removing old embeddings file: {EMBEDDINGS_PATH}")
    os.remove(EMBEDDINGS_PATH)
if os.path.isfile(DATA_PATH):
    print(f"Removing old data file: {DATA_PATH}")
    os.remove(DATA_PATH)
# Also remove old index files if they exist
if os.path.isfile("quotes.index"):
    print("Removing old index file: quotes.index")
    os.remove("quotes.index")


start_time = perf_counter()

print("Loading model...")
# Load the model from HuggingFace
# Tell the model to use the Mac's 'mps' GPU for acceleration
model = SentenceTransformer('avsolatorio/GIST-large-Embedding-v0', device='mps')

after_loading_model_time = perf_counter()

# Read the CSV file of quotes
print("Reading CSV...")
quotes_data = []
with open("quotes.csv", "r", encoding='utf-8') as file:
    quote_reader = reader(file)
    next(quote_reader, None) # Skip header
    for row in quote_reader:
        if len(row) < 2:
            continue
        quotes_data.append((row[0], row[1]))

sentences = [item[0] for item in quotes_data]
after_csv_time = perf_counter()

print(f"Read {len(sentences)} quotes. Embedding...")
embeddings = model.encode(sentences, show_progress_bar=True, batch_size=64) # Smaller batch size for stability
after_embedding_time = perf_counter()

print("Normalizing embeddings...")
# Normalize the embeddings to unit vectors for cosine similarity search
# This is a necessary step for the IndexFlatIP index
faiss.normalize_L2(embeddings)

print(f"Storing embeddings to {EMBEDDINGS_PATH}...")
# Save the numpy array directly to a file. This is very reliable.
np.save(EMBEDDINGS_PATH, embeddings)
after_npy_time = perf_counter()

print(f"Storing text data to {DATA_PATH}...")
# Save the list of (quote, author) tuples to a pickle file
with open(DATA_PATH, "wb") as f:
    pickle.dump(quotes_data, f)
after_pickle_time = perf_counter()

print("\n--- All files created successfully! ---")
print(f"Wrote {len(sentences)} embeddings and quotes to disk.")

print(f"\nTotal time: {after_pickle_time - start_time:.2f} seconds")
print(f"\tLoading model: {after_loading_model_time - start_time:.2f} seconds")
print(f"\tReading CSV: {after_csv_time - after_loading_model_time:.2f} seconds")
print(f"\tLLM Embedding: {after_embedding_time - after_csv_time:.2f} seconds")
print(f"\tStoring Embeddings (.npy): {after_npy_time - after_embedding_time:.2f} seconds")
print(f"\tStoring Quote Data (.pkl): {after_pickle_time - after_npy_time:.2f} seconds")
